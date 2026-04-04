"""
AudioTransformer encoder for W-JEPA pre-training on 1-D audio sequences.

Architecture:
  - Wav2Vec2-style CNN feature extractor  →  token sequence
  - Transformer blocks with 1-D RoPE (no learned positional embedding)
  - Hierarchical output: concatenation of intermediate layer norms

Input  : (B, 1, T)                         raw waveform, variable T
Output (training=True)  : (B, N, D * L)    hierarchical concat (L levels)
Output (training=False) : (B, N, D)        last-layer norm only
"""

import math
from functools import partial

import torch
import torch.nn as nn

from .modules import Block
from .feature_extractor import AudioFeatureExtractor
from .utils import apply_masks, trunc_normal_


class AudioTransformer(nn.Module):
    def __init__(
        self,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        is_causal=False,
        init_type="default",
        n_registers=0,
        has_cls_first=False,
        n_output_distillation=4,
        **kwargs,  # absorb unused kwargs (e.g. seq_len, patch_size from old configs)
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads    = num_heads
        self.init_type    = init_type

        # CNN feature extractor (Wav2Vec2 style, stride=320)
        self.patch_embed = AudioFeatureExtractor(in_chans=in_chans, embed_dim=embed_dim)

        # No learned positional embedding – RoPE is applied inside each Block
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
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
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                use_rope=True,          # always RoPE for audio
                n_registers=n_registers,
                has_cls_first=has_cls_first,
            )
            for i in range(depth)
        ])

        # Hierarchical output layers
        _layer_map = {
            12: [2,  5,  8, 11],
            24: [5, 11, 17, 23],
            40: [9, 19, 29, 39],
            48: [11, 23, 37, 47],
        }
        assert depth in _layer_map, f"Unsupported depth {depth}"
        self.hierarchical_layers = _layer_map[depth]
        if n_output_distillation == 4:
            self.out_layers = self.hierarchical_layers
        elif n_output_distillation == 1:
            self.out_layers = [self.hierarchical_layers[-1]]
        else:
            self.out_layers = self.hierarchical_layers[-n_output_distillation:]

        self.norms_block = nn.ModuleList([
            norm_layer(embed_dim) for _ in range(len(self.out_layers))
        ])

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    # ---- init helpers -------------------------------------------------------

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            return
        if self.init_type == "default":
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif self.init_type in ("xavier_uniform", "xavier_normal"):
            fn = (nn.init.xavier_uniform_ if self.init_type == "xavier_uniform"
                  else nn.init.xavier_normal_)
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                fn(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        for i, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (i + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (i + 1)))

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}   # no pos_embed to exclude

    # ---- forward ------------------------------------------------------------

    def forward(self, x, masks=None, training=False):
        """
        Args:
            x       : (B, 1, T)  raw waveform (variable T, right-zero-padded)
            masks   : (B, K) index tensor of context tokens, or list of such tensors
                      None for target encoder (full sequence)
            training: True  → return hierarchical concat (B, N, D*L)
                      False → return last norm           (B, N, D)

        RoPE notes:
            - masks=None  → RoPEAttention uses arange(N) as positions
            - masks=(B,K) → RoPEAttention uses the K actual token indices as positions
              This ensures padding tokens are never "seen" as context and that
              masked context tokens retain their true temporal positions.
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        x = self.patch_embed(x)      # (B, N, D) – N depends on actual audio length

        if masks is not None:
            x = apply_masks(x, masks)
            masks_cat = torch.cat(masks, dim=0)   # (B*n_masks, K) – positions for RoPE
        else:
            masks_cat = None                       # full sequence; RoPE uses arange

        hier = []
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x, _ = torch.utils.checkpoint.checkpoint(
                    blk, x, masks_cat, use_reentrant=False
                )
            else:
                x, _ = blk(x, mask=masks_cat)

            if i in self.out_layers:
                idx = self.out_layers.index(i)
                hier.append(self.norms_block[idx](x))

        if training:
            return torch.cat(hier, dim=2)   # (B, N, D * L)
        else:
            return self.norms_block[-1](x)  # (B, N, D)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def audio_transformer_base(**kwargs):
    return AudioTransformer(embed_dim=768, depth=12, num_heads=12,
                            mlp_ratio=4, qkv_bias=True,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def audio_transformer_large(**kwargs):
    return AudioTransformer(embed_dim=1024, depth=24, num_heads=16,
                            mlp_ratio=4, qkv_bias=True,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def audio_transformer_giant(**kwargs):
    return AudioTransformer(embed_dim=1408, depth=40, num_heads=22,
                            mlp_ratio=48 / 11, qkv_bias=True,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


EMBED_DIMS = {
    "audio_transformer_base":  768,
    "audio_transformer_large": 1024,
    "audio_transformer_giant": 1408,
}
