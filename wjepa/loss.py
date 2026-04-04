"""
W-JEPA pre-training loss primitives.

  forward_target    : EMA target encoder (no_grad + per-level LayerNorm)
  forward_context   : context encoder → predictor
  loss_fn           : L1/L2 loss, optional CLS prepend, optional distance weighting
  LambdaWarmupHold  : linear warmup schedule for context loss coefficient λ
"""

import torch
import torch.nn.functional as F

from .masks.distance import compute_mask_distance
from .models.utils import apply_masks


# ---------------------------------------------------------------------------
# Internal normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_level(tensor, embed_dim, n_levels=4):
    """LayerNorm each embed_dim-sized chunk along the last axis, then concat."""
    chunks = [
        F.layer_norm(tensor[:, :, i * embed_dim:(i + 1) * embed_dim], (embed_dim,))
        for i in range(n_levels)
    ]
    return torch.cat(chunks, dim=2)


def _normalize_nested(nested, embed_dim):
    return [
        [[_normalize_level(z, embed_dim) for z in inner] for inner in outer]
        for outer in nested
    ]


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------

def forward_target(clips, target_encoder, embed_dim, levels_predictor):
    """
    EMA target encoder, no gradient.
    Splits the concatenated multi-level output and applies LayerNorm per level.

    Returns:
        h: list[Tensor(B, N, embed_dim * levels_predictor)], one per clip group
    """
    with torch.no_grad():
        h = target_encoder(clips, gram_mode=False, training_mode=True)
        new_h = []
        for hi in h:
            if levels_predictor > 1:
                new_h.append(_normalize_level(hi, embed_dim))
            else:
                new_h.append(F.layer_norm(hi, (hi.size(-1),)))
    return new_h


def forward_context(clips, masks_enc, masks_pred, encoder, predictor,
                    embed_dim, normalize_predictor, predict_all):
    """
    Context encoder → predictor forward.

    Returns:
        z_pred    : nested list [fpc][mask] – predicted target tokens
        z_context : nested list [fpc][mask] – predicted context tokens (or None)
    """
    z = encoder(clips, masks_enc, gram_mode=False, training_mode=True)
    z_pred, z_context = predictor(z, masks_enc, masks_pred)

    if normalize_predictor:
        z_pred = _normalize_nested(z_pred, embed_dim)
        if predict_all:
            z_context = _normalize_nested(z_context, embed_dim)

    return z_pred, z_context


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def loss_fn(z, h, masks_to_apply, loss_exp, d_weights=None, has_cls_first=False):
    """
    L = mean( |z - h|^loss_exp ) / loss_exp,  averaged over all (fpc, mask) pairs.

    Args:
        z              : nested list [fpc][mask] of predictor outputs  (B, K, D)
        h              : list [fpc] of target tensors                  (B, N, D)
        masks_to_apply : nested mask indices [fpc][mask] → (B, K)
        loss_exp       : 1 = smooth-L1 style, 2 = L2
        d_weights      : optional distance weights [fpc][mask] (B, K)  – context loss
        has_cls_first  : prepend CLS token (h[:, 0]) before masked tokens
    """
    if has_cls_first:
        h_cls = [hi[:, 0].unsqueeze(1) for hi in h]
        h = [apply_masks(hi[:, 1:], mi, concat=False) for hi, mi in zip(h, masks_to_apply)]
        loss, n = 0, 0
        for zi, hi, hi_cls in zip(z, h, h_cls):
            for zij, hij in zip(zi, hi):
                h_term = torch.cat([hi_cls, hij], dim=1)
                loss += torch.mean(torch.abs(zij - h_term) ** loss_exp) / loss_exp
                n += 1
        return loss / n

    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_to_apply)]

    if d_weights is not None:
        loss, n = 0, 0
        for zi, hi, d_i in zip(z, h, d_weights):
            for zij, hij, d_ij in zip(zi, hi, d_i):
                loss += torch.mean(
                    torch.abs(zij - hij) ** loss_exp * (1.0 / d_ij.unsqueeze(2))
                ) / loss_exp
                n += 1
        return loss / n

    loss, n = 0, 0
    for zi, hi in zip(z, h):
        for zij, hij in zip(zi, hi):
            loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
            n += 1
    return loss / n


# ---------------------------------------------------------------------------
# Progressive λ schedule for context loss
# ---------------------------------------------------------------------------

class LambdaWarmupHold:
    """
    0  →  lambda_value  linearly over [start_iter, end_iter], then constant.
    """

    def __init__(self, lambda_value: float, start_iter: int = 15_000, end_iter: int = 30_000):
        assert end_iter > start_iter
        self.lambda_value = float(lambda_value)
        self.start = int(start_iter)
        self.end = int(end_iter)
        self.span = self.end - self.start

    def value(self, global_iter: int) -> float:
        if global_iter < self.start:
            return 0.0
        if global_iter >= self.end:
            return self.lambda_value
        return self.lambda_value * (global_iter - self.start) / self.span
