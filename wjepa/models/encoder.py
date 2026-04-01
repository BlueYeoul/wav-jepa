import math
from functools import partial

import torch
import torch.nn as nn

from .modules import Block
from .patch_embed import PatchEmbed
from .utils import apply_masks, trunc_normal_


class AudioTransformer(nn.Module):
    """
    Encoder backbone for W-JEPA pre-training on 1-D sequences.

    Input  : (B, C, T)                             – C channels, T time steps
    Output (training=True) : (B, N, D * L)         – hierarchical concat (L levels)
    Output (training=False): (B, N, D)              – last-layer norm only
    """

    def __init__(
        self,
        seq_len=16000,
        patch_size=16,
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
        use_rope=False,
        init_type="default",
        n_registers=0,
        has_cls_first=False,
        n_output_distillation=4,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads    = num_heads
        self.init_type    = init_type

        # Wav2Vec2 style feature extractor as patch embedding
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
        self.patch_size = self.patch_embed.patch_size  # Fixed at 320

        self.use_rope     = use_rope
        self.use_activation_checkpointing = use_activation_checkpointing
        self.num_patches = seq_len // self.patch_size

        # learned positional embedding (used when use_rope=False)
        if not use_rope:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

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
                use_rope=use_rope,
                n_registers=n_registers,
                has_cls_first=has_cls_first,
            )
            for i in range(depth)
        ])

        # ---- hierarchical output layers ------------------------------------
        _layer_map = {
            12: [2,  5,  8,  11],
            24: [5, 11, 17,  23],
            40: [9, 19, 29,  39],
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
        return {"pos_embed"} if not self.use_rope else {}

    # ---- forward ------------------------------------------------------------

    def forward(self, x, masks=None, training=False):
        """
        x      : (B, C, T)
        masks  : list of (B, K) index tensors – context masking (None for target encoder)
        training: True → return hierarchical concat (B, N, D*L)
                  False → return last norm only    (B, N, D)
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        x = self.patch_embed(x)             # (B, N, D)

        if not self.use_rope:
            x = x + self.pos_embed

        if masks is not None:
            x = apply_masks(x, masks)
            masks_cat = torch.cat(masks, dim=0)
        else:
            masks_cat = None

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


def audio_transformer_large_rope(**kwargs):
    return AudioTransformer(embed_dim=1024, depth=24, num_heads=16,
                            mlp_ratio=4, qkv_bias=True, use_rope=True,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def audio_transformer_giant(**kwargs):
    return AudioTransformer(embed_dim=1408, depth=40, num_heads=22,
                            mlp_ratio=48/11, qkv_bias=True,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


EMBED_DIMS = {
    "audio_transformer_base":       768,
    "audio_transformer_large":      1024,
    "audio_transformer_large_rope": 1024,
    "audio_transformer_giant":      1408,
}
