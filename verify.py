import argparse
import sys
from pathlib import Path

import torch
import torchaudio
import matplotlib.pyplot as plt

# ── 1. Model Loading Logic (Extracted & Cleaned) ──────────────────────────────

def load_fe(ckpt_path: str, device: str = "cpu"):
    """Loads the target feature extractor from the encoder's patch_embed."""
    from wjepa.models.feature_extractor import AudioFeatureExtractor
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc_sd = ckpt.get("encoder", ckpt)

    fe_sd = {
        k.replace("backbone.patch_embed.", ""): v
        for k, v in enc_sd.items() if "patch_embed" in k
    }
    
    if not fe_sd:
        raise RuntimeError("No patch_embed keys found in checkpoint['encoder']")

    fe = AudioFeatureExtractor()
    
    # Handle SnakeBeta and torch.compile legacy prefixes
    mapped = {}
    for mk in fe.state_dict():
        if mk in fe_sd:
            mapped[mk] = fe_sd[mk]
        else:
            stripped = mk.replace("._orig_mod.", ".")
            if stripped in fe_sd:
                mapped[mk] = fe_sd[stripped]
            else:
                added = mk.replace(".raw", "._orig_mod.raw")
                if added in fe_sd:
                    mapped[mk] = fe_sd[added]

    fe.load_state_dict(mapped, strict=False)
    return fe.eval().to(device)


def load_ema(ckpt_path: str, model_name: str = "audio_transformer_base", device: str = "cpu"):
    """Loads the full EMA target encoder."""
    import wjepa.models.encoder as enc_module
    from wjepa.models.wrappers import MultiSeqWrapper

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    te_sd = ckpt.get("target_encoder")
    
    if te_sd is None:
        print("Warning: target_encoder key not found in checkpoint.", file=sys.stderr)
        return None

    backbone = enc_module.__dict__[model_name]()
    model = MultiSeqWrapper(backbone)

    # Handle torch.compile legacy prefixes
    mapped = {}
    for mk in model.state_dict():
        if mk in te_sd:
            mapped[mk] = te_sd[mk]
        else:
            stripped = mk.replace("._orig_mod.", ".")
            if stripped in te_sd:
                mapped[mk] = te_sd[stripped]

    model.load_state_dict(mapped, strict=False)
    return model.eval().to(device)


# ── 2. Audio Processing Logic (Extracted) ─────────────────────────────────────

def load_audio(audio_path: str, sr: int = 16000, max_duration_sec: int = 3, device: str = "cpu"):
    """Loads a .wav file, resamples if necessary, and extracts a chunk."""
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    wav, orig_sr = torchaudio.load(audio_path)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    
    wav = wav[:1]  # Force mono
    
    # Clip to max_duration to prevent massive plots, or keep whole file depending on needs.
    # The original script chunked it to 3 seconds.
    chunk_size = sr * max_duration_sec
    if wav.shape[-1] > chunk_size:
        wav = wav[:, :chunk_size]
        
    # Model expects shape: (Batch=1, Channels=1, Time)
    return wav.unsqueeze(0).to(device)


# ── 3. Matplotlib Visualization ───────────────────────────────────────────────

def plot_features(fe_out: torch.Tensor, ema_out: torch.Tensor, title_prefix=""):
    """
    Plots the output tensors using matplotlib.
    Expects tensors of shape (N_tokens, D_dimensions).
    """
    # Detach and move to CPU for plotting
    fe_np = fe_out.detach().cpu().numpy()
    
    # Create subplots based on whether EMA exists
    n_plots = 2 if ema_out is not None else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 5 * n_plots), sharex=True)
    if n_plots == 1: axes = [axes]  # Normalize to list

    # 1. Plot Feature Extractor (FE)
    # Transpose (.T) so Time (N) is on the X-axis and Dimensions (D) is on Y-axis
    im_fe = axes[0].imshow(fe_np.T, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f"{title_prefix} - Feature Extractor (FE) Output")
    axes[0].set_ylabel("Dimension (D)")
    fig.colorbar(im_fe, ax=axes[0], label="Activation Value")

    # 2. Plot EMA Target Encoder (if available)
    if ema_out is not None:
        ema_np = ema_out.detach().cpu().numpy()
        im_ema = axes[1].imshow(ema_np.T, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title(f"{title_prefix} - EMA Target Encoder Output")
        axes[1].set_ylabel("Dimension (D)")
        axes[1].set_xlabel("Time Token (N)")
        fig.colorbar(im_ema, ax=axes[1], label="Activation Value")
    else:
        axes[0].set_xlabel("Time Token (N)")

    plt.tight_layout()
    plt.savefig(f"{title_prefix}_features.png")


# ── Main Execution ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="W-JEPA Feature Visualization Tool")
    parser.add_argument("checkpoint", help="Path to checkpoint (.pt / .pth.tar)")
    parser.add_argument("--audio", required=True, help="Path to audio file (.wav/.flac)")
    parser.add_argument("--model", default="audio_transformer_base", help="EMA model name")
    parser.add_argument("--device", default="cpu", help="Compute device (e.g., cpu, cuda)")
    args = parser.parse_args()

    print(f"[*] Loading audio from {args.audio}...")
    x = load_audio(args.audio, device=args.device)

    print("[*] Loading Feature Extractor...")
    fe = load_fe(args.checkpoint, device=args.device)

    print("[*] Loading EMA Target Encoder...")
    ema = load_ema(args.checkpoint, model_name=args.model, device=args.device)

    print("[*] Running forward pass...")
    with torch.no_grad():
        # FE output shape: (1, N, D) -> Squeeze batch dim to (N, D)
        fe_out = fe(x)[0]
        
        ema_out = None
        if ema is not None:
            # MultiSeqWrapper: masks=None → [backbone(x_i) for x_i in clips]
            # backbone returns (B, N, D), so squeeze batch and clip dims
            ema_out = ema([x])[0][0]

    print(f"[*] Plotting features... FE shape: {fe_out.shape}" + 
          (f", EMA shape: {ema_out.shape}" if ema_out is not None else ""))
          
    plot_features(fe_out, ema_out, title_prefix=Path(args.audio).name)

if __name__ == "__main__":
    main()