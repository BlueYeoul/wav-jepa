import torch.nn as nn


class MultiSeqWrapper(nn.Module):
    """
    Wraps an encoder to handle multiple clip-length groups.

    forward(x, masks, training_mode) where:
        x     = list of tensors, one per clip-length group  (B, C, T)
        masks = list of mask-lists, one per group (None for target encoder)
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone  = backbone
        self.embed_dim = backbone.embed_dim

    def forward(self, x, masks=None, gram_mode=False, training_mode=False):
        # target encoder: no masks
        if masks is None:
            return [self.backbone(x_i, training=training_mode) for x_i in x]

        # context encoder: one forward per (clip, mask) pair
        outs = [[] for _ in x]
        for i, (x_i, m_i) in enumerate(zip(x, masks)):
            for m in m_i:
                outs[i].append(self.backbone(x_i, masks=m, training=training_mode))
        return outs


class PredictorMultiSeqWrapper(nn.Module):
    """
    Wraps a predictor to handle multiple clip-length groups.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks_x, masks_y, mod=None):
        outs_pred    = [[] for _ in x]
        outs_context = [[] for _ in x]
        for i, (x_i, mx_i, my_i) in enumerate(zip(x, masks_x, masks_y)):
            for j, (xij, mx, my) in enumerate(zip(x_i, mx_i, my_i)):
                z_pred, z_ctx = self.backbone(xij, mx, my, mask_index=j)
                outs_pred[i].append(z_pred)
                outs_context[i].append(z_ctx)
        return outs_pred, outs_context
