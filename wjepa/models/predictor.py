"""
AudioTransformerPredictor for W-JEPA 1-D pre-training.

Given masked context tokens from the encoder, predicts target tokens.
Positional encoding is handled entirely by RoPE inside each Block –
no learned positional embedding is stored here.

When return_all_tokens=True the predictor also outputs predictions for
the context tokens (used for the optional context loss λ).
"""

import math
from functools import partial

import torch
import torch.nn as nn

from .modules import Block
from .utils import apply_masks, repeat_interleave_batch, trunc_normal_


class AudioTransformerPredictor(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        predictor_embed_dim=384,
        out_embed_dim=None,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        is_causal=False,
        use_activation_checkpointing=False,
        return_all_tokens=False,
        n_registers=0,
        has_cls_first=False,
        n_output_distillation=4,
        **kwargs,  # absorb unused kwargs (seq_len, patch_size from old configs)
    ):
        super().__init__()
        self.return_all_tokens = return_all_tokens
        self.has_cls_first     = has_cls_first
        self.use_activation_checkpointing = use_activation_checkpointing

        # Hierarchical output layer indices
        _layer_map = {
            4:  [0, 1, 2, 3],
            8:  [1, 3, 5, 7],
            12: [2, 5, 8, 11],
            20: [4, 9, 14, 19],
            24: [4, 11, 17, 23],
            40: [9, 19, 29, 39],
        }
        assert depth in _layer_map, f"Unsupported predictor depth {depth}"
        self.hierarchical_layers = _layer_map[depth][-n_output_distillation:]

        # Input projection
        act_layer_mlp = nn.SiLU if use_silu else nn.GELU
        n_levels = len(self.hierarchical_layers)
        if n_levels == 1:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        else:
            self.predictor_embed = nn.Sequential(
                nn.Linear(embed_dim * n_levels, embed_dim, bias=True),
                act_layer_mlp(),
                nn.Linear(embed_dim, predictor_embed_dim, bias=True),
            )

        # Learnable mask tokens (one per mask config)
        self.mask_tokens    = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for _ in range(num_mask_tokens)
            ])

        # No predictor_pos_embed: RoPE is applied inside each Block

        # Transformer blocks (always RoPE)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=nn.SiLU if use_silu else nn.GELU,
                wide_silu=wide_silu,
                norm_layer=norm_layer,
                use_sdpa=True,
                is_causal=is_causal,
                use_rope=True,      # always RoPE for audio
                n_registers=n_registers,
                has_cls_first=has_cls_first,
            )
            for i in range(depth)
        ])

        # Output projection
        if out_embed_dim is None:
            out_embed_dim = embed_dim
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(
            predictor_embed_dim, n_levels * out_embed_dim, bias=True
        )
        if return_all_tokens:
            self.predictor_proj_context = nn.Linear(
                predictor_embed_dim, n_levels * out_embed_dim, bias=True
            )

        self.init_std = init_std
        if use_mask_tokens and not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    # ---- init helpers -------------------------------------------------------

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        for i, layer in enumerate(self.predictor_blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (i + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (i + 1)))

    # ---- forward ------------------------------------------------------------

    def forward(self, x, masks_x, masks_y, mask_index=0, **kwargs):
        """
        Args:
            x       : context tokens from encoder  (B, N_ctx, D*L)
            masks_x : list of (B, K_ctx) – context token indices in original sequence
            masks_y : list of (B, K_tgt) – target token indices in original sequence

        Returns:
            z_pred    : (B, K_tgt, D_out*L)  predicted target tokens
            z_context : (B, K_ctx, D_out*L)  predicted context tokens, or None
        """
        if not isinstance(masks_x, list): masks_x = [masks_x]
        if not isinstance(masks_y, list): masks_y = [masks_y]

        B = len(x) // len(masks_x)

        # Project to predictor space
        x = self.predictor_embed(x)    # (B, N_ctx, predictor_embed_dim)
        _, N_ctxt, D = x.shape

        # Build target mask tokens dynamically (size = max token index + 1)
        mask_index = mask_index % max(self.num_mask_tokens, 1)
        masks_y_cat = torch.cat(masks_y, dim=0)                      # (B*n_masks, K_tgt)
        max_idx = int(masks_y_cat.max().item()) + 1                   # dynamic upper bound

        if self.mask_tokens is not None:
            pred_tokens = self.mask_tokens[mask_index].expand(
                B * len(masks_x), max_idx, -1
            ).clone()
        else:
            pred_tokens = torch.zeros(
                B * len(masks_x), max_idx, D, device=x.device, dtype=x.dtype
            )
        pred_tokens = apply_masks(pred_tokens, masks_y)               # (B*n, K_tgt, D)
        # No positional embedding added here — RoPE uses position indices from masks

        # Repeat context for each mask config and concatenate with target tokens
        x = x.repeat(len(masks_x), 1, 1)                             # (B*n, N_ctx, D)
        x = torch.cat([x, pred_tokens], dim=1)                        # (B*n, N_ctx+K_tgt, D)

        # Merge position index lists and sort so attention sees tokens in order
        masks_x_cat = torch.cat(masks_x, dim=0)                      # (B*n, K_ctx)
        masks       = torch.cat([masks_x_cat, masks_y_cat], dim=1)   # (B*n, K_ctx+K_tgt)

        argsort = torch.argsort(masks, dim=1)
        masks   = torch.stack([masks[i, row] for i, row in enumerate(argsort)])
        x       = torch.stack([x[i, row, :] for i, row in enumerate(argsort)])

        # Predictor transformer (RoPE positions = sorted token indices)
        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing:
                x, _ = torch.utils.checkpoint.checkpoint(
                    blk, x, masks, use_reentrant=False
                )
            else:
                x, _ = blk(x, mask=masks)
        x = self.predictor_norm(x)

        # Restore original ordering
        reverse = torch.argsort(argsort, dim=1)
        x = torch.stack([x[i, row, :] for i, row in enumerate(reverse)])

        z_pred = self.predictor_proj(x[:, N_ctxt:, :])
        if self.return_all_tokens:
            z_context = self.predictor_proj_context(x[:, :N_ctxt, :])
            return z_pred, z_context
        return z_pred, None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def audio_predictor(**kwargs):
    return AudioTransformerPredictor(
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
