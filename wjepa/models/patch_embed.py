import torch.nn as nn


class PatchEmbed1D(nn.Module):
    """
    1-D sequence → patch tokens.

    Input : (B, C, T)  – C channels, T time steps
    Output: (B, N, D)  – N = T // patch_size tokens
    """

    def __init__(self, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).transpose(1, 2)   # (B, D, N) → (B, N, D)
