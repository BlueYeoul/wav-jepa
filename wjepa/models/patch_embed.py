import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Wav2Vec2 style convolutional feature extractor as Patch Embedding.
    Total stride = 5 * 2 * 2 * 2 * 2 * 2 * 2 = 320.
    Output length is T // 320.
    """

    def __init__(self, in_chans=1, embed_dim=768, feature_dim=512):
        super().__init__()
        self.patch_size = 320  # Effective patch size

        # Standard Wav2Vec2 feature encoder configuration
        self.conv_layers = nn.ModuleList([
            # Layer 0: Conv (in_chans -> 512, 10, 5, p=3)
            nn.Sequential(
                nn.Conv1d(in_chans, feature_dim, kernel_size=10, stride=5, padding=3, bias=False),
                nn.GroupNorm(feature_dim, feature_dim),
                nn.GELU()
            ),
            # Layers 1-4: Conv (512 -> 512, 3, 2, p=1)
            *[nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GELU()
            ) for _ in range(4)],
            # Layers 5-6: Conv (512 -> 512, 2, 2, p=0)
            *[nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim, kernel_size=2, stride=2, padding=0, bias=False),
                nn.GELU()
            ) for _ in range(2)],
        ])
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        # Input x: (B, C, T)
        for layer in self.conv_layers:
            x = layer(x)
        # x: (B, feature_dim, N)
        x = x.transpose(1, 2)  # (B, N, feature_dim)
        x = self.proj(x)       # (B, N, embed_dim)
        return x
