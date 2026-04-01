import math

import torch


# ---------------------------------------------------------------------------
# Truncated normal initializer
# ---------------------------------------------------------------------------

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def apply_masks(x, masks, concat=True):
    """
    :param x: [B, N, D]
    :param masks: list of [B, K] index tensors – patches to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    return torch.cat(
        [torch.cat([x[i * B:(i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)],
        dim=0,
    )
