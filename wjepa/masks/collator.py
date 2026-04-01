from multiprocessing import Value

import torch


class MaskCollator:
    """
    Collator for W-JEPA 1-D pre-training.

    Generates paired (masks_enc, masks_pred) for each clip-length group in the batch.

    Args:
        cfgs_mask     : list of mask-config dicts, one per mask block type
        dataset_fpcs  : list of sequence lengths used in the dataset
        seq_len       : total number of tokens per sequence
        patch_size    : patch size (number of time steps per token)
    """

    def __init__(self, cfgs_mask, dataset_fpcs, seq_len=1024, patch_size=16):
        self.mask_generators = {}
        for fpc in dataset_fpcs:
            n_tokens = (fpc * seq_len) // (fpc * patch_size)  # = seq_len // patch_size
            self.mask_generators[fpc] = [
                _MaskGenerator1D(
                    num_tokens=n_tokens,
                    pred_mask_scale=m.get("scale", (0.15, 0.5)),
                    n_blocks=m.get("num_blocks", 1),
                    max_context_ratio=m.get("max_context_ratio", 1.0),
                    max_keep=m.get("max_keep", None),
                    full_complement=m.get("full_complement", False),
                    pred_full_complement=m.get("pred_full_complement", False),
                    inv_block=m.get("inv_block", False),
                )
                for m in cfgs_mask
            ]

    def step(self):
        for fpc in self.mask_generators:
            for mg in self.mask_generators[fpc]:
                mg.step()

    def __call__(self, batch):
        """
        Returns list of (collated_batch, collated_masks_enc, collated_masks_pred),
        one entry per fpc group in the batch.
        """
        filtered = {fpc: [] for fpc in self.mask_generators}
        for sample in batch:
            # infer fpc from the sequence length of the last element
            try:
                fpc = len(sample[-1][-1]) if isinstance(sample[-1], (list, tuple)) else 1
            except (TypeError, IndexError):
                fpc = 1
            if fpc in filtered:
                filtered[fpc].append(sample)

        collations = []
        for fpc, fpc_batch in filtered.items():
            if not fpc_batch:
                continue
            B = len(fpc_batch)
            collated_batch   = torch.utils.data.default_collate(fpc_batch)
            masks_enc_list   = []
            masks_pred_list  = []
            for mg in self.mask_generators[fpc]:
                m_enc, m_pred = mg(B)
                masks_enc_list.append(m_enc)
                masks_pred_list.append(m_pred)
            collations.append((collated_batch, masks_enc_list, masks_pred_list))

        return collations


class _MaskGenerator1D:
    """
    Generates one type of (encoder_mask, predictor_mask) pair for 1-D sequences.

    Predictor mask : one or more contiguous token segments.
    Encoder mask   : complement (visible tokens).
    """

    def __init__(
        self,
        num_tokens=1024,
        pred_mask_scale=(0.15, 0.5),
        n_blocks=1,
        max_context_ratio=1.0,
        max_keep=None,
        full_complement=False,
        pred_full_complement=False,
        inv_block=False,
    ):
        self.num_tokens       = num_tokens
        self.pred_mask_scale  = pred_mask_scale
        self.n_blocks         = n_blocks
        self.max_ctx_tokens   = max(1, int(num_tokens * max_context_ratio))
        self.max_keep         = max_keep
        self.full_complement  = full_complement
        self.pred_full_complement = pred_full_complement
        self.inv_block        = inv_block
        self._itr_counter     = Value("i", -1)

    def step(self):
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    def _sample_block(self, generator):
        """Sample a contiguous masked segment; return a (num_tokens,) int32 mask (0=masked)."""
        min_s, max_s = self.pred_mask_scale
        r = torch.rand(1, generator=generator).item()
        n_mask = max(1, int(self.num_tokens * (min_s + r * (max_s - min_s))))
        n_mask = min(n_mask, self.num_tokens)

        start = torch.randint(0, self.num_tokens - n_mask + 1, (1,), generator=generator).item()

        mask = torch.ones(self.num_tokens, dtype=torch.int32)
        mask[start:start + n_mask] = 0

        # restrict context to first max_ctx_tokens
        if self.max_ctx_tokens < self.num_tokens:
            mask[self.max_ctx_tokens:] = 0

        return mask

    def __call__(self, batch_size):
        """Returns (masks_enc, masks_pred) each of shape (B, K)."""
        seed = self.step()
        g    = torch.Generator()
        g.manual_seed(seed)

        masks_pred, masks_enc = [], []
        min_keep_enc  = self.num_tokens
        min_keep_pred = self.num_tokens

        for _ in range(batch_size):
            empty_ctx = True
            while empty_ctx:
                mask_e = torch.ones(self.num_tokens, dtype=torch.int32)
                for _ in range(self.n_blocks):
                    mask_e = mask_e * self._sample_block(g)

                mask_p = torch.argwhere(mask_e == 0).squeeze()   # masked   → predict
                mask_e = torch.nonzero(mask_e).squeeze()          # unmasked → context

                empty_ctx = len(mask_e) == 0
                if not empty_ctx:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc  = min(min_keep_enc,  len(mask_e))
                    masks_pred.append(mask_p)
                    masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        masks_enc  = [m[:min_keep_enc]  for m in masks_enc]
        masks_pred = [m[:min_keep_pred] for m in masks_pred]

        n_total = self.num_tokens

        if self.full_complement:
            masks_pred = [
                torch.tensor(sorted(set(range(n_total)) - set(m.tolist())), dtype=m.dtype)
                for m in masks_enc
            ]
        elif self.pred_full_complement:
            masks_enc = [
                torch.tensor(sorted(set(range(n_total)) - set(m.tolist())), dtype=m.dtype)
                for m in masks_pred
            ]

        masks_enc  = torch.utils.data.default_collate(masks_enc)
        masks_pred = torch.utils.data.default_collate(masks_pred)

        if self.inv_block:
            return masks_pred, masks_enc
        return masks_enc, masks_pred
