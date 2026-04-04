"""
Wav2Vec2-style convolutional feature extractor for W-JEPA.

Converts raw waveform (B, 1, T) → token sequence (B, N, embed_dim).
Total stride = 320  (5 × 2^6), so 20 ms/token at 16 kHz.

No patch_size concept: the CNN architecture implicitly defines the
temporal compression; use compute_audio_output_length() for exact counts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Standalone length utility (no model instantiation needed)
# ---------------------------------------------------------------------------

# Conv config: (out_channels, kernel_size, stride) – mirrors AudioFeatureExtractor
_CONV_CFG = [
    (512, 10, 5),
    (512,  3, 2),
    (512,  3, 2),
    (512,  3, 2),
    (512,  3, 2),
    (512,  2, 2),
    (512,  2, 2),
]


def compute_audio_output_length(lengths: torch.Tensor) -> torch.Tensor:
    """
    Map waveform sample counts → feature-extractor output token counts.

    Args:
        lengths: (B,) LongTensor of sample counts (before padding)
    Returns:
        (B,) LongTensor of token counts
    """
    lengths = lengths.clone().long()
    for _, kernel, stride in _CONV_CFG:
        lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
    return lengths


def compute_max_output_length(max_samples: int) -> int:
    """Scalar version for model initialisation."""
    length = max_samples
    for _, kernel, stride in _CONV_CFG:
        length = (length - kernel) // stride + 1
    return length


# ---------------------------------------------------------------------------
# Feature extractor module
# ---------------------------------------------------------------------------
class SnakeBeta(nn.Module):
    def __init__(self, in_features, min_alpha=1e-2, max_inv=10.0):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(1, in_features, 1))
        self.min_alpha = min_alpha
        self.max_inv = max_inv

    def forward(self, x):
        alpha = F.softplus(self.raw) + self.min_alpha
        inv = (1.0 / alpha).clamp_max(self.max_inv)
        return x + inv * (torch.sin(alpha * x) ** 2)
    
class AudioFeatureExtractor(nn.Module):
    """
    Wav2Vec2 convolutional feature extractor.

    Input : (B, C, T)   – raw waveform, C=1 for mono
    Output: (B, N, embed_dim)  – token sequence, no lengths returned
    """

    def __init__(self, in_chans: int = 1, embed_dim: int = 768, feature_dim: int = 512):
        super().__init__()
        self.conv_layers = nn.ModuleList()

        # Layer 0: Conv + GroupNorm + GELU (Wav2Vec2 convention)
        c_in = in_chans
        c_out, k, s = _CONV_CFG[0]
        self.conv_layers.append(nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, bias=False),
            nn.GroupNorm(c_out, c_out),
            SnakeBeta(c_out),
        ))

        # Layers 1-6: Conv + GELU
        for i in range(1, len(_CONV_CFG)):
            c_in = _CONV_CFG[i - 1][0]
            c_out, k, s = _CONV_CFG[i]
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, bias=False),
                SnakeBeta(c_out),
            ))

        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, N, embed_dim)
        """
        for layer in self.conv_layers:
            x = layer(x)
        x = x.transpose(1, 2)   # (B, feature_dim, N) → (B, N, feature_dim)
        return self.proj(x)      # (B, N, embed_dim)
