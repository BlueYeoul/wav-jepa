import torch
import torch.nn as nn
import math

class FeatureEncoder(nn.Module):
    """
    Wav2Vec 2.0 정석적인 Feature Extractor.
    Total stride = 320 (5 * 2 * 2 * 2 * 2 * 2 * 2)
    """

    def __init__(self, in_chans=1, embed_dim=768, feature_dim=512):
        super().__init__()
        
        # (channels, kernel, stride)
        self.conv_cfg = [
            (feature_dim, 10, 5),
            (feature_dim, 3, 2),
            (feature_dim, 3, 2),
            (feature_dim, 3, 2),
            (feature_dim, 3, 2),
            (feature_dim, 2, 2),
            (feature_dim, 2, 2),
        ]

        self.conv_layers = nn.ModuleList()

        # Layer 0: Conv + GroupNorm + GELU
        # Wav2Vec2는 첫 번째 레이어에서 채널 통계량을 맞추기 위해 GroupNorm을 권장함
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv1d(in_chans, self.conv_cfg[0][0], kernel_size=self.conv_cfg[0][1], stride=self.conv_cfg[0][2], bias=False),
                nn.GroupNorm(self.conv_cfg[0][0], self.conv_cfg[0][0]),
                nn.GELU()
            )
        )

        # Layers 1-6: Conv + GELU (Standard setup)
        for i in range(1, len(self.conv_cfg)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(self.conv_cfg[i-1][0], self.conv_cfg[i][0], kernel_size=self.conv_cfg[i][1], stride=self.conv_cfg[i][2], bias=False),
                    nn.GELU()
                )
            )

        # 최종 임베딩 차원으로 투영
        self.proj = nn.Linear(feature_dim, embed_dim)

    def get_output_seq_len_fn(self) -> callable:

        def _conv_out_length(input_length, kernel_size, stride):
            # Conv1d의 output length 공식 (padding=0, dilation=1 기준)
            return torch.div(input_length - kernel_size, stride, rounding_mode='floor') + 1

        def _compute_output_length(input_length):
            for _, kernel, stride in self.conv_cfg:
                input_length = _conv_out_length(input_length, kernel, stride)
            return input_length
        return _compute_output_length

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        입력 길이에 따른 출력 길이를 계산하는 공식
        L_out = floor((L_in - kernel_size) / stride + 1)
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # Conv1d의 output length 공식 (padding=0, dilation=1 기준)
            return torch.div(input_length - kernel_size, stride, rounding_mode='floor') + 1

        for _, kernel, stride in self.conv_cfg:
            input_lengths = _conv_out_length(input_lengths, kernel, stride)
        return input_lengths
    

    def forward(self, x, input_lengths=None):
        """
        Args:
            x: (B, C, T) - 보통 C=1 (Raw waveform)
            input_lengths: (B,) - 각 샘플의 실제 길이 (Padding 제외)
        Returns:
            x: (B, N, embed_dim)
            output_lengths: (B,) - 줄어든 시퀀스 길이
        """
        # Feature Extraction
        for layer in self.conv_layers:
            x = layer(x)

        # (B, feature_dim, N) -> (B, N, feature_dim)
        x = x.transpose(1, 2)
        
        # Linear Projection
        x = self.proj(x)

        # Length 계산
        output_lengths = None
        if input_lengths is not None:
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

        return x, output_lengths

# --- 사용 예시 ---
if __name__ == "__main__":
    # 모델 초기화
    model = FeatureEncoder(in_chans=1, embed_dim=768)
    
    # 가상의 입력 (Batch=4, Channel=1, Time=16000) -> 1초 분량(16kHz)
    batch_size = 4
    raw_audio = torch.randn(batch_size, 1, 16000)
    input_lengths = torch.LongTensor([16000, 15000, 12000, 8000]) # 가변 길이 가정

    # Forward
    features, out_lengths = model(raw_audio, input_lengths)

    print(f"Input Shape: {raw_audio.shape}")   # [4, 1, 16000]
    print(f"Output Shape: {features.shape}")   # [4, 49, 768] (16000 // 320 = 50인데, kernel edge 효과로 49가 됨)
    print(f"Output Lengths: {out_lengths}")    # [49, 46, 37, 24]