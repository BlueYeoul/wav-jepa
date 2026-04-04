import math
from logging import getLogger
from multiprocessing import Value
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

logger = getLogger()

class DynamicMaskCollator1D:
    """
    Meta V-JEPA 스타일의 1D Mask Collator.
    이미 패딩된 waveform을 입력받아 Feature Extractor 출력 길이에 맞는 마스크 생성.
    """
    def __init__(
        self,
        cfgs_mask,
        compute_output_length,
        dynamic_config=None,
    ):
        super().__init__()
        self.compute_output_length = compute_output_length
        self.dynamic_config = dynamic_config or {}
        self.current_epoch = 0

        # Dynamic Masking 설정 추출
        self.dynamic_mask_cfg = self.dynamic_config.get("dynamic_mask", {})
        
        self.mask_generators = []
        for m in cfgs_mask:
            mask_generator = _MaskGenerator1D(
                pred_mask_scale=self._get_dynamic_mask_scale(), # 초기 스케일
                npred=m.get("num_blocks", 1),
                max_context_ratio=m.get("max_temporal_keep", 1.0),
                max_keep=m.get("max_keep", None),
                full_complement=m.get("full_complement", False),
                pred_full_complement=m.get("pred_full_complement", False),
                inv_block=m.get("inv_block", False),
            )
            self.mask_generators.append(mask_generator)

    def _get_dynamic_mask_scale(self):
        """에포크에 따른 동적 마스크 비율 계산 (Meta 코드에는 없던 스케줄링 추가)"""
        if not self.dynamic_mask_cfg.get("enabled", False):
            return (0.15, 0.5)
        
        min_r = self.dynamic_mask_cfg.get("min_mask_ratio", 0.15)
        max_r = self.dynamic_mask_cfg.get("max_mask_ratio", 0.5)
        warmup = self.dynamic_mask_cfg.get("warmup_epochs", 20)
        
        progress = min(self.current_epoch / warmup, 1.0)
        curr_min = min_r + (max_r - min_r) * progress
        return (curr_min, curr_min + 0.3)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        new_scale = self._get_dynamic_mask_scale()
        for mg in self.mask_generators:
            mg.set_epoch(epoch, new_scale)

    def step(self):
        for mg in self.mask_generators:
            mg.step()

    def __call__(self, batch):
        # 1. 가변 길이 audio를 직접 패딩하여 배치 구성
        #    (default_collate는 같은 크기 tensor만 stack 가능하므로 audio는 별도 처리)
        if isinstance(batch[0], dict):
            audio_list = [item["audio"] for item in batch]
            lengths    = torch.tensor([item["seq_len"] for item in batch], dtype=torch.long)
            max_len    = int(lengths.max().item())

            audio_padded = torch.stack([
                F.pad(a, (0, max_len - a.shape[0]), value=0.0)
                for a in audio_list
            ])  # (B, T_max)

            # non-audio 필드는 default_collate로 처리 (float/int 스칼라 등)
            other_keys = [k for k in batch[0] if k != "audio"]
            other = default_collate([{k: item[k] for k in other_keys} for item in batch])

            collated_batch = {"audio": audio_padded, **other}
            waveforms = audio_padded

        elif isinstance(batch[0], (list, tuple)):
            collated_batch = default_collate(batch)
            waveforms, lengths = collated_batch[0], collated_batch[1]
        else:
            collated_batch = default_collate(batch)
            waveforms = collated_batch.get("audio")
            lengths   = collated_batch.get("seq_len")

        # 3. Feature Extractor 이후의 동적 길이 계산
        feat_lengths = self.compute_output_length(lengths)
        max_feat_len = int(feat_lengths.max().item())
        batch_size = waveforms.shape[0]

        collated_masks_enc, collated_masks_pred = [], []
        
        # 4. 각 마스크 설정(Generator)별로 마스크 생성
        for mg in self.mask_generators:
            masks_enc, masks_pred = mg(batch_size, max_feat_len)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        return collated_batch, collated_masks_enc, collated_masks_pred


class _MaskGenerator1D:
    def __init__(
        self,
        pred_mask_scale=(0.15, 0.5),
        npred=1,
        max_context_ratio=1.0,
        max_keep=None,
        inv_block=False,
        full_complement=False,
        pred_full_complement=False,
    ):
        super().__init__()
        self.pred_mask_scale = pred_mask_scale
        self.npred = npred
        self.max_context_ratio = max_context_ratio
        self.max_keep = max_keep
        self.inv_block = inv_block
        self.full_complement = full_complement
        self.pred_full_complement = pred_full_complement
        
        self._itr_counter = Value("i", -1)
        self.current_epoch = 0

    def set_epoch(self, epoch, new_scale):
        self.current_epoch = epoch
        self.pred_mask_scale = new_scale

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, num_tokens):
        """1차원 블록 크기 샘플링 (Meta의 _sample_block_size 수정)"""
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = self.pred_mask_scale
        mask_scale = min_s + _rand * (max_s - min_s)
        return max(1, int(num_tokens * mask_scale))

    def _sample_block_mask(self, num_tokens, b_size, max_ctx_len):
        """무작위 위치에 1D 블록 마스킹 (Meta의 _sample_block_mask 수정)"""
        start = torch.randint(0, num_tokens - b_size + 1, (1,)).item()
        
        mask = torch.ones(num_tokens, dtype=torch.int32)
        mask[start : start + b_size] = 0

        # Context 제한 (앞부분 X%만 인코더가 볼 수 있게 함)
        if max_ctx_len < num_tokens:
            mask[max_ctx_len:] = 0
        return mask

    def __call__(self, batch_size, num_tokens):
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        # 1. 이번 배치의 공통 블록 크기 결정 (Meta 방식: 시드 기반 공유)
        p_size = self._sample_block_size(g, num_tokens)
        max_ctx_len = max(1, int(num_tokens * self.max_context_ratio))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = num_tokens

        for _ in range(batch_size):
            empty_context = True
            while empty_context:
                mask_e = torch.ones(num_tokens, dtype=torch.int32)
                # Meta의 npred (Multi-block) 적용
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(num_tokens, p_size, max_ctx_len)

                mask_p = torch.argwhere(mask_e == 0).squeeze(-1) # 예측 대상
                mask_e = torch.nonzero(mask_e).squeeze(-1)      # 인코더 입력

                empty_context = len(mask_e) == 0
                if not empty_context:
                    # 차원 방어
                    if mask_p.ndim == 0: mask_p = mask_p.unsqueeze(0)
                    if mask_e.ndim == 0: mask_e = mask_e.unsqueeze(0)

                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        # Meta의 핵심 로직: 배치 내 최소 길이에 맞춘 Truncation (Rectangular Tensor 생성용)
        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]

        # Complement 로직 (인코더 마스크의 여집합을 예측 대상으로 삼을지 여부)
        if self.full_complement:
            collated_masks_pred = [
                torch.tensor(sorted(list(set(range(num_tokens)) - set(cm.tolist()))), dtype=cm.dtype)
                for cm in collated_masks_enc
            ]
        elif self.pred_full_complement:
            collated_masks_enc = [
                torch.tensor(sorted(list(set(range(num_tokens)) - set(cm.tolist()))), dtype=cm.dtype)
                for cm in collated_masks_pred
            ]

        collated_masks_enc = default_collate(collated_masks_enc)
        collated_masks_pred = default_collate(collated_masks_pred)

        if self.inv_block:
            return collated_masks_pred, collated_masks_enc
        return collated_masks_enc, collated_masks_pred

# Alias for backward compatibility
DynamicMaskCollator = DynamicMaskCollator1D


# ==========================================
# 동작 검증을 위한 테스트 코드
# ==========================================
if __name__ == "__main__":
    def dummy_compute_len(lengths): return lengths // 320 # Feature 압축 시뮬레이션

    cfgs = [{"num_blocks": 5, "max_keep": 50}]
    collator = DynamicMaskCollator1D(cfgs, dummy_compute_len)
    
    # 이미 패딩된 더미 데이터 (Batch=2, Audio_len=1600)
    # 실제 길이는 각각 1600, 800
    seq = 50
    batch = [
        (torch.randn(16000*seq), 16000*seq),
        (torch.randn(16000*seq), 8000*seq)
    ]
    
    collated_batch, masks_enc, masks_pred = collator(batch)
    
    print(f"Max Feature Length: {16000 // 320}")
    print(f"Encoder Mask Shape: {masks_enc[0].shape}") # [Batch, min_keep_enc]
    print(f"Predictor Mask Shape: {masks_pred[0].shape}") # [Batch, min_keep_pred]
    print(f"Sample Encoder Indices: {masks_enc[0][0]}")