import torch


def compute_mask_distance(masks_pred, masks_enc, seq_len, offset=False):
    """
    Compute per-context-token distance to its nearest predicted token (1-D).

    Used to weight the context loss: context tokens far from the predicted region
    get up-weighted (penalised more).

    Args:
        masks_pred : nested list [fpc][mask_idx] of (B, N_pred) int tensors
        masks_enc  : nested list [fpc][mask_idx] of (B, N_enc)  int tensors
        seq_len    : total number of tokens in the sequence (used for normalisation)
        offset     : if True, scale distances by 1 / (seq_len / 64) to normalise

    Returns:
        distances  : nested list [fpc][mask_idx] of (B, N_enc) float tensors
    """
    distances = []
    for masks_pred_i, masks_enc_i in zip(masks_pred, masks_enc):
        row_distances = []
        for mp, me in zip(masks_pred_i, masks_enc_i):
            # mp: (B, N_pred)  me: (B, N_enc)  – both are token indices
            enc_pos  = me.float().unsqueeze(-1)          # (B, N_enc,  1)
            pred_pos = mp.float().unsqueeze(-2)          # (B, 1,      N_pred)

            diffs = torch.abs(enc_pos - pred_pos)        # (B, N_enc, N_pred)
            dmin  = diffs.min(dim=-1).values             # (B, N_enc)

            if offset:
                dmin = dmin * (1.0 / (seq_len / 64))

            dmin = dmin ** 0.5   # soften the decay
            row_distances.append(dmin)
        distances.append(row_distances)

    return distances
