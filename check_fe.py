"""
check_fe.py – Feature Extractor + EMA model collapse diagnostic (TUI)

Usage:
    python check_fe.py <checkpoint.pt>
    python check_fe.py <checkpoint.pt> --audio path/to/file.wav
    python check_fe.py <checkpoint.pt> --model audio_transformer_large

Sections printed:
  1. FE Output Statistics table
  2. FE per-dim / per-token heatmaps
  3. EMA (target encoder) Output Statistics table
  4. EMA per-dim / per-token heatmaps
  5. Time-aligned view: waveform / FE norm+std / EMA norm+std
  6. Verdict
"""

import argparse
import sys
from pathlib import Path

import torch

# ── rich TUI ──────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None


# ── helpers ───────────────────────────────────────────────────────────────────

def _sparkline(values, width: int = 30) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = hi - lo + 1e-9
    chars = []
    step = max(1, len(values) // width)
    for i in range(0, len(values), step):
        idx = int((values[i] - lo) / span * (len(blocks) - 1))
        chars.append(blocks[idx])
        if len(chars) >= width:
            break
    return "".join(chars)


def _hist_row(tensor_1d, bins: int = 20) -> str:
    counts = torch.histc(tensor_1d.float(), bins=bins)
    counts = counts / counts.max().clamp(min=1)
    blocks = " ▁▂▃▄▅▆▇█"
    return "".join(blocks[int(c * (len(blocks) - 1))] for c in counts.tolist())


def _collapse_score(out: torch.Tensor) -> float:
    """Mean per-dim std across tokens. Near 0 → collapsed."""
    return out.std(dim=0).mean().item()


# ── model loading ─────────────────────────────────────────────────────────────

def _map_snake_keys(fe_sd: dict, model_sd: dict) -> dict:
    """
    Map SnakeBeta keys between checkpoint and model state dicts.
    Also handles legacy _orig_mod. prefix from old torch.compile checkpoints.
    """
    mapped = {}
    for model_key in model_sd:
        if model_key in fe_sd:
            mapped[model_key] = fe_sd[model_key]
        else:
            stripped = model_key.replace("._orig_mod.", ".")
            if stripped in fe_sd:
                mapped[model_key] = fe_sd[stripped]
            else:
                added = model_key.replace(".raw", "._orig_mod.raw")
                if added in fe_sd:
                    mapped[model_key] = fe_sd[added]
    return mapped


def load_fe(ckpt_path: str, device: str = "cpu"):
    from wjepa.models.feature_extractor import AudioFeatureExtractor

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch", "?")
    enc_sd = ckpt.get("encoder", ckpt)

    fe_sd = {
        k.replace("backbone.patch_embed.", ""): v
        for k, v in enc_sd.items()
        if "patch_embed" in k
    }
    if not fe_sd:
        raise RuntimeError("No patch_embed keys found in checkpoint['encoder']")

    fe = AudioFeatureExtractor()
    mapped = _map_snake_keys(fe_sd, fe.state_dict())
    msg = fe.load_state_dict(mapped, strict=False)
    fe.eval().to(device)

    # Extract SnakeBeta raw params: { layer_idx → raw tensor (1, C, 1) }
    snake_raws = {}
    for k, v in fe_sd.items():
        k_clean = k.replace("._orig_mod.", ".")
        # keys like "conv_layers.0.2.raw" or "conv_layers.1.1.raw"
        parts = k_clean.split(".")
        if parts[-1] == "raw" and parts[0] == "conv_layers":
            layer_idx = int(parts[1])
            snake_raws[layer_idx] = v

    return fe, epoch, msg, snake_raws


def load_ema(ckpt_path: str, model_name: str = "audio_transformer_base",
             device: str = "cpu"):
    """
    Load the target_encoder (EMA model) as a full AudioTransformer.
    Returns (model, msg) or (None, reason_str) if not found.
    """
    from functools import partial
    import wjepa.models.encoder as enc_module
    from wjepa.models.wrappers import MultiSeqWrapper

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    te_sd = ckpt.get("target_encoder")
    if te_sd is None:
        return None, "target_encoder key not in checkpoint"

    backbone = enc_module.__dict__[model_name]()
    model = MultiSeqWrapper(backbone)

    # Handle _orig_mod. in SnakeBeta keys
    model_sd = model.state_dict()
    mapped = {}
    for mk in model_sd:
        if mk in te_sd:
            mapped[mk] = te_sd[mk]
        else:
            stripped = mk.replace("._orig_mod.", ".")
            if stripped in te_sd:
                mapped[mk] = te_sd[stripped]

    msg = model.load_state_dict(mapped, strict=False)
    model.eval().to(device)
    return model, msg


# ── run models ────────────────────────────────────────────────────────────────

def _build_inputs(audio_path, n_samples, sr, device):
    inputs = []
    if audio_path:
        try:
            import torchaudio
            wav, orig_sr = torchaudio.load(audio_path)
            if orig_sr != sr:
                wav = torchaudio.functional.resample(wav, orig_sr, sr)
            wav = wav[:1]  # mono
            chunk = sr * 3
            for i in range(min(n_samples, wav.shape[-1] // chunk)):
                seg = wav[:, i * chunk:(i + 1) * chunk]
                inputs.append((f"wav chunk {i}", seg.unsqueeze(0).to(device)))
        except Exception as e:
            if HAS_RICH:
                console.print(f"[yellow]Audio load failed ({e}), using synthetic.[/]")
            else:
                print(f"Audio load failed ({e}), using synthetic.")

    if not inputs:
        for name, sig in [
            ("white noise",     torch.randn(1, 1, sr * 3)),
            ("sine 440 Hz",     torch.sin(2 * 3.14159 * 440
                                * torch.arange(sr * 3).float() / sr).reshape(1, 1, -1)),
            ("sine 1 kHz",      torch.sin(2 * 3.14159 * 1000
                                * torch.arange(sr * 3).float() / sr).reshape(1, 1, -1)),
            ("zeros (silence)", torch.zeros(1, 1, sr * 3)),
        ][:n_samples]:
            inputs.append((name, sig.to(device)))
    return inputs


def run_models(fe, ema, audio_path=None, n_samples=4, sr=16000, device="cpu"):
    """
    Returns list of (label, fe_out (N,D), ema_out (N,D) or None, wav_1d (T,))
    """
    inputs = _build_inputs(audio_path, n_samples, sr, device)
    results = []
    with torch.no_grad():
        for label, x in inputs:
            fe_out = fe(x)[0]          # (N, D)
            if ema is not None:
                # MultiSeqWrapper: masks=None → [backbone(x_i) for x_i in clips]
                # backbone returns (B, N, D), so squeeze batch dim
                ema_out = ema([x])[0][0]   # (N, D)
            else:
                ema_out = None
            results.append((label, fe_out, ema_out, x[0, 0]))
    return results


# ── TUI blocks ────────────────────────────────────────────────────────────────

def _stats_table(c, title: str, rows, color: str = "cyan"):
    """
    rows: list of (label, out_tensor (N,D))
    Returns True if any sample is collapsed.
    """
    table = Table(title=title, box=box.ROUNDED, show_lines=True)
    table.add_column("Input", style=color, width=18)
    table.add_column("Shape (N×D)", justify="center")
    table.add_column("mean",  justify="right")
    table.add_column("std",   justify="right")
    table.add_column("min",   justify="right")
    table.add_column("max",   justify="right")
    table.add_column("token-std\n(collapse?)", justify="center")
    table.add_column("dim distribution", width=22)

    any_collapse = False
    for label, out in rows:
        mu   = out.mean().item()
        sd   = out.std().item()
        lo   = out.min().item()
        hi   = out.max().item()
        tsco = _collapse_score(out)

        if tsco < 1e-4:
            any_collapse = True
            flag = Text("YES ⚠", style="bold red")
        elif tsco < 1e-2:
            flag = Text(f"{tsco:.4f} ~", style="yellow")
        else:
            flag = Text(f"{tsco:.4f} ✓", style="green")

        table.add_row(
            label,
            f"{out.shape[0]}×{out.shape[1]}",
            f"{mu:+.4f}", f"{sd:.4f}", f"{lo:+.4f}", f"{hi:+.4f}",
            flag,
            _hist_row(out.flatten(), bins=22),
        )
    c.print(table)
    return any_collapse


def _heatmaps(c, label: str, out: torch.Tensor):
    """Per-dim and per-token heatmaps for one sample."""
    N, D = out.shape

    # per-dim token-std
    sample_dims = min(D, 64)
    stride = max(1, D // sample_dims)
    dim_stds = out.std(dim=0).tolist()

    c.print(f"\n[bold]Per-dim token-std[/]  ({label},  showing {sample_dims}/{D} dims)")
    c.print("[dim]  Each cell = std across N tokens.  Red≈0 → collapsed dim.[/]")
    row_len = 32
    for start in range(0, sample_dims, row_len):
        chunk = dim_stds[start * stride:(start + row_len) * stride:stride]
        max_v = max(chunk) if chunk else 1.0
        line = Text()
        line.append(f"  dim {start*stride:4d}-{min((start+row_len)*stride-1, D-1):4d}: ")
        for v in chunk:
            r = v / (max_v + 1e-9)
            if r < 0.05:
                line.append("▓", style="bold red")
            elif r < 0.2:
                line.append("▒", style="yellow")
            elif r < 0.5:
                line.append("░", style="dim white")
            else:
                line.append("█", style="bright_white")
        c.print(line)

    # per-token dim-std
    sample_n = min(N, 60)
    stride_t = max(1, N // sample_n)
    token_stds = out.std(dim=1).tolist()
    chunk_t = token_stds[::stride_t][:sample_n]
    max_v = max(chunk_t) if chunk_t else 1.0

    c.print(f"\n[bold]Per-token dim-std[/]  ({label},  showing {sample_n}/{N} tokens)")
    c.print("[dim]  Each cell = std across D dims.  Red≈0 → constant token.[/]")
    row = Text("  ")
    for v in chunk_t:
        r = v / (max_v + 1e-9)
        if r < 0.05:
            row.append("▓", style="bold red")
        elif r < 0.2:
            row.append("▒", style="yellow")
        elif r < 0.5:
            row.append("░", style="dim white")
        else:
            row.append("█", style="bright_white")
    c.print(row)
    c.print(f"  sparkline: [dim]{_sparkline(chunk_t, width=sample_n)}[/]")
    c.print(f"  range: min={min(token_stds):.4f}  max={max(token_stds):.4f}"
            f"  mean={sum(token_stds)/len(token_stds):.4f}")


# ── time-aligned view ─────────────────────────────────────────────────────────

def _aligned_view(c, label: str, wav_1d: torch.Tensor,
                  fe_out: torch.Tensor, ema_out, sr: int = 16000,
                  wave_height: int = 6, bar_height: int = 3):
    """
    Stacked time-aligned display:

      wav  ─┤  [waveform envelope]
      ──────┤
    fe norm ┤  [FE per-token L2 norm bar]
     fe std ┤  [FE per-token dim-std sparkline]
      ──────┤
   ema norm ┤  [EMA per-token L2 norm bar]   (skipped if ema_out is None)
    ema std ┤  [EMA per-token dim-std]
         sec┤  [time axis]
    """
    T = len(wav_1d)
    N = fe_out.shape[0]
    total_sec = T / sr
    stride_s = T / N

    margin = 12
    W = max(40, (c.width or 120) - margin)

    # ── waveform envelope per column ──────────────────────────────────────────
    chunk = max(1, T // W)
    col_max = []
    col_min = []
    for i in range(W):
        seg = wav_1d[i * chunk:(i + 1) * chunk]
        if len(seg):
            col_max.append(seg.max().item())
            col_min.append(seg.min().item())
        else:
            col_max.append(0.0)
            col_min.append(0.0)

    peak = max(max(abs(v) for v in col_max), max(abs(v) for v in col_min), 1e-6)
    half_h = wave_height // 2

    # ── token metrics → columns ───────────────────────────────────────────────
    def _tok_to_cols(out_nd):
        n_acc  = [0.0] * W
        s_acc  = [0.0] * W
        cnt    = [0]   * W
        norms  = out_nd.norm(dim=1).tolist()
        stds   = out_nd.std(dim=1).tolist()
        for ti in range(out_nd.shape[0]):
            col = min(W - 1, int(ti * stride_s * W / T))
            n_acc[col] += norms[ti]
            s_acc[col] += stds[ti]
            cnt[col]   += 1
        for col in range(W):
            if cnt[col]:
                n_acc[col] /= cnt[col]
                s_acc[col] /= cnt[col]
        mn = max(n_acc) or 1.0
        ms = max(s_acc) or 1.0
        return [v / mn for v in n_acc], [v / ms for v in s_acc]

    fe_norm_n, fe_std_n = _tok_to_cols(fe_out)
    if ema_out is not None:
        ema_norm_n, ema_std_n = _tok_to_cols(ema_out)

    # ── header ────────────────────────────────────────────────────────────────
    c.print(f"  [bold cyan]{label}[/]  "
            f"[dim]{T} samples · {total_sec:.2f}s · "
            f"{N} tokens · stride≈{stride_s:.0f} samp[/]")

    def _pr(line):
        c.print(line, no_wrap=True)

    def _sep(left_label=""):
        t = Text()
        t.append(f"{left_label:>9}┤ " + "─" * W, style="dim")
        _pr(t)

    # ── waveform ──────────────────────────────────────────────────────────────
    center = half_h
    for row in range(wave_height + 1):
        line = Text()
        lbl = "  wav ─" if row == center else "       "
        line.append(f"{lbl}┤ ", style="dim")
        for ci in range(W):
            hi =  col_max[ci] / peak
            lo = -col_min[ci] / peak
            in_pos = (row < center) and (center - row) <= max(1, int(hi * half_h + 0.5))
            in_neg = (row > center) and (row - center) <= max(1, int(lo * half_h + 0.5))
            if in_pos:
                frac = (center - row) / (half_h + 1e-9)
                line.append("█", style="bright_cyan" if frac > 0.6 else "cyan")
            elif in_neg:
                frac = (row - center) / (half_h + 1e-9)
                line.append("█", style="cyan" if frac > 0.6 else "blue")
            elif row == center:
                line.append("▬" if max(hi, lo) > 0.05 else "·", style="dim cyan")
            else:
                line.append(" ")
        _pr(line)

    # ── FE section ────────────────────────────────────────────────────────────
    _sep("  ──────")

    _blk = " ▁▂▃▄▅▆▇█"

    def _norm_bar(norm_n, label_str, color_hi, color_lo):
        for row in range(bar_height):
            line = Text()
            line.append(f"{label_str:>9}┤ " if row == 0 else "         ┤ ", style="dim")
            for ci in range(W):
                filled = int(norm_n[ci] * bar_height)
                if (bar_height - 1 - row) < filled:
                    v = norm_n[ci]
                    line.append("█", style=color_hi if v > 0.75
                                else (color_lo if v > 0.4 else f"dim {color_lo}"))
                else:
                    line.append(" ")
            _pr(line)

    def _std_row(std_n, label_str):
        line = Text()
        line.append(f"{label_str:>9}┤ ", style="dim")
        for ci in range(W):
            v = std_n[ci]
            ch = _blk[int(v * (len(_blk) - 1))]
            line.append(ch, style="bold red" if v < 0.1
                        else ("yellow" if v < 0.35 else "green"))
        _pr(line)

    _norm_bar(fe_norm_n, " fe norm", "bright_green", "green")
    _std_row(fe_std_n,  "  fe std")

    # ── EMA section ───────────────────────────────────────────────────────────
    if ema_out is not None:
        _sep("  ──────")
        _norm_bar(ema_norm_n, "ema norm", "bright_yellow", "yellow")
        _std_row(ema_std_n,   " ema std")

    # ── time axis ─────────────────────────────────────────────────────────────
    line = Text()
    line.append("      sec┤ ", style="dim")
    n_lbl = min(8, W // 8)
    ticks = [" "] * W
    for i in range(n_lbl + 1):
        pos = int(i * (W - 1) / n_lbl)
        ts = f"{pos * total_sec / W:.1f}"
        for j, ch in enumerate(ts):
            if pos + j < W:
                ticks[pos + j] = ch
    line.append("".join(ticks), style="dim")
    _pr(line)


# ── SnakeBeta alpha view ─────────────────────────────────────────────────────

def _snakebeta_panel(c, snake_raws: dict):
    """
    Visualize learned SnakeBeta α per conv layer.

    alpha = softplus(raw) + 0.01
    inv   = (1/alpha).clamp(max=10)

    Layout per layer:
      Layer N │ α dist sparkline │ mean/min/max │ inv_mean │ status
      + per-channel bar showing α distribution (top-20 / bottom-20)
    """
    import torch.nn.functional as F

    if not snake_raws:
        c.print("[dim italic]No SnakeBeta raw params found in checkpoint.[/]")
        return

    c.rule("[bold]⓪ SnakeBeta α  (conv feature extractor activations)[/]")
    c.print("[dim]  alpha = softplus(raw) + 0.01   "
            "inv = (1/alpha).clamp(max=10.0)[/]")
    c.print("[dim]  Large α → sin²(αx) oscillates fast → spiky/noisy output[/]")
    c.print("[dim]  Small α → sin²(αx) ≈ 0 → output ≈ x  (near-linear, healthy)[/]")
    c.print("[dim]  inv ≈ 10 (saturated) → all channels clamped → collapse risk[/]\n")

    MIN_ALPHA = 0.01
    MAX_INV   = 10.0
    W = max(40, (c.width or 120) - 30)  # sparkline width

    # summary table
    table = Table(box=box.ROUNDED, show_lines=True,
                  title="Per-layer SnakeBeta α summary")
    table.add_column("Layer", justify="center", style="dim", width=6)
    table.add_column("C", justify="right", width=5)
    table.add_column("α mean", justify="right")
    table.add_column("α min",  justify="right")
    table.add_column("α max",  justify="right")
    table.add_column("inv mean", justify="right")
    table.add_column("sat%\n(inv=10)", justify="center")
    table.add_column("status", justify="center")

    any_bad = False
    layer_data = {}  # for per-channel bars below

    for idx in sorted(snake_raws.keys()):
        raw = snake_raws[idx].float()        # (1, C, 1)
        alpha = F.softplus(raw) + MIN_ALPHA  # (1, C, 1)
        inv   = (1.0 / alpha).clamp_max(MAX_INV)

        alpha_ch = alpha[0, :, 0]  # (C,)
        inv_ch   = inv[0, :, 0]

        a_mean = alpha_ch.mean().item()
        a_min  = alpha_ch.min().item()
        a_max  = alpha_ch.max().item()
        i_mean = inv_ch.mean().item()
        sat_pct = (inv_ch >= MAX_INV - 1e-3).float().mean().item() * 100

        layer_data[idx] = (alpha_ch, inv_ch, sat_pct)

        # status
        if sat_pct > 50:
            any_bad = True
            status = Text("⚠ SATURATED", style="bold red")
        elif sat_pct > 20:
            status = Text("~ warn", style="yellow")
        elif a_mean > 5.0:
            status = Text("~ large α", style="yellow")
        else:
            status = Text("✓", style="green")

        sat_style = "bold red" if sat_pct > 50 else ("yellow" if sat_pct > 20 else "green")

        table.add_row(
            str(idx),
            str(alpha_ch.shape[0]),
            f"{a_mean:.4f}",
            f"{a_min:.4f}",
            f"{a_max:.4f}",
            f"{i_mean:.4f}",
            Text(f"{sat_pct:.1f}%", style=sat_style),
            status,
        )

    c.print(table)

    # per-channel α distribution bar per layer
    c.print("\n[bold]Per-channel α distribution[/]  "
            "[dim](sorted low→high, each char = one channel)[/]")
    c.print("[dim]  green=small α (linear-ish)  yellow=medium  red=large (oscillating)[/]\n")

    _blk = " ▁▂▃▄▅▆▇█"
    for idx in sorted(layer_data.keys()):
        alpha_ch, inv_ch, sat_pct = layer_data[idx]
        sorted_a = alpha_ch.sort().values.tolist()
        C = len(sorted_a)

        # downsample to terminal width
        step = max(1, C // W)
        sampled = sorted_a[::step][:W]
        a_max_here = max(sampled) if sampled else 1.0

        line = Text()
        line.append(f"  L{idx} │ ", style="dim")
        for v in sampled:
            ratio = v / (a_max_here + 1e-9)
            ch = _blk[int(ratio * (len(_blk) - 1))]
            if v < 0.5:
                line.append(ch, style="green")
            elif v < 2.0:
                line.append(ch, style="yellow")
            else:
                line.append(ch, style="bold red")
        line.append(f" │ α=[{min(sorted_a):.2f},{max(sorted_a):.2f}]"
                    f"  sat={sat_pct:.0f}%", style="dim")
        c.print(line, no_wrap=True)

    if any_bad:
        c.print(Panel(
            "[bold red]SnakeBeta α is SATURATED in ≥1 layer[/]\n"
            "inv=(1/α) is clamped to 10 for >50% of channels.\n"
            "This means those channels output  x + 10·sin²(αx)\n"
            "with very fast oscillation → after GroupNorm/downstream layers\n"
            "the signal averages out → constant token vectors.\n\n"
            "Quick fixes:\n"
            "  • Lower max_inv:  SnakeBeta(..., max_inv=3.0)\n"
            "  • Add α regularization / weight_decay on raw params\n"
            "  • Switch to GELU for first few layers",
            title="[red]⚠ SnakeBeta collapse risk[/]", border_style="red",
        ))


# ── top-level render ──────────────────────────────────────────────────────────

def render_tui(results, epoch, ckpt_path, fe_msg, ema_msg, snake_raws):
    if HAS_RICH:
        _render_rich(results, epoch, ckpt_path, fe_msg, ema_msg, snake_raws)
    else:
        _render_plain(results)


def _render_rich(results, epoch, ckpt_path, fe_msg, ema_msg, snake_raws):
    c = console

    c.rule(f"[bold cyan]W-JEPA Diagnostic[/]  epoch={epoch}")
    c.print(f"[dim]checkpoint :[/] {ckpt_path}")
    c.print(f"[dim]fe  weights:[/] {fe_msg}")
    c.print(f"[dim]ema weights:[/] {ema_msg}\n")

    # ── 0. SnakeBeta α ────────────────────────────────────────────────────────
    _snakebeta_panel(c, snake_raws)

    # ── 1. FE stats ───────────────────────────────────────────────────────────
    c.rule("[bold]① Feature Extractor (CNN patch_embed)[/]")
    fe_rows = [(lbl, fe_out) for lbl, fe_out, _, _ in results]
    fe_collapse = _stats_table(c, "FE Output  (N tokens × D dims)", fe_rows, color="cyan")

    label0, fe_out0, _, _ = results[0]
    _heatmaps(c, label0, fe_out0)

    # ── 2. EMA stats ──────────────────────────────────────────────────────────
    has_ema = results[0][2] is not None
    ema_collapse = False
    if has_ema:
        c.rule("[bold]② EMA Target Encoder (full AudioTransformer)[/]")
        ema_rows = [(lbl, ema_out) for lbl, _, ema_out, _ in results]
        ema_collapse = _stats_table(c, "EMA Output  (N tokens × D dims)", ema_rows, color="yellow")
        _heatmaps(c, label0, results[0][2])
    else:
        c.print("\n[dim italic]EMA model not loaded — skipping section ②[/]\n")

    # ── 3. Time-aligned view ──────────────────────────────────────────────────
    c.rule("[bold]③ Time-aligned: Waveform / FE / EMA[/]")
    for label, fe_out, ema_out, wav_1d in results:
        _aligned_view(c, label, wav_1d, fe_out, ema_out)
        c.print()

    # ── 4. Verdict ────────────────────────────────────────────────────────────
    c.print()
    any_collapse = fe_collapse or ema_collapse
    if any_collapse:
        who = []
        if fe_collapse:  who.append("FE")
        if ema_collapse: who.append("EMA")
        c.print(Panel(
            f"[bold red]COLLAPSE DETECTED in: {', '.join(who)}[/]\n"
            "Token outputs are near-constant across time.\n"
            "Possible causes:\n"
            "  • SnakeBeta α saturated → sin²(αx) → constant output\n"
            "  • GroupNorm dead channels\n"
            "  • LR too high → weights exploded then reset\n"
            "  • Gradient vanished through conv stack",
            title="[red]⚠ Verdict[/]", border_style="red",
        ))
    else:
        c.print(Panel(
            "[bold green]Both FE and EMA look healthy.[/]\n"
            "Token-wise std is non-trivial → outputs vary across time.",
            title="[green]✓ Verdict[/]", border_style="green",
        ))


def _render_plain(results):
    for label, fe_out, ema_out, _wav in results:
        print(f"\n=== {label} ===")
        for name, out in [("FE", fe_out), ("EMA", ema_out)]:
            if out is None:
                continue
            tsco = _collapse_score(out)
            print(f"  [{name}] shape={tuple(out.shape)}"
                  f"  mean={out.mean():.4f}  std={out.std():.4f}"
                  f"  token-std={tsco:.6f}"
                  + ("  *** COLLAPSED ***" if tsco < 1e-4 else ""))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="W-JEPA FE + EMA diagnostic")
    parser.add_argument("checkpoint", help="Path to checkpoint (.pt / .pth.tar)")
    parser.add_argument("--audio",  default=None,   help="Audio file (.wav/.flac)")
    parser.add_argument("--model",  default="audio_transformer_base",
                        help="EMA model name (default: audio_transformer_base)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n",      type=int, default=4, help="Number of test inputs")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    if HAS_RICH:
        console.print("[dim]Loading FE weights…[/]")
    fe, epoch, fe_msg, snake_raws = load_fe(args.checkpoint, device=args.device)

    if HAS_RICH:
        console.print("[dim]Loading EMA (target encoder) weights…[/]")
    ema, ema_msg = load_ema(args.checkpoint, model_name=args.model, device=args.device)
    if ema is None:
        ema_msg_str = f"[yellow]{ema_msg}[/]" if HAS_RICH else ema_msg
        if HAS_RICH:
            console.print(f"  [yellow]EMA not loaded: {ema_msg}[/]")

    if HAS_RICH:
        console.print("[dim]Running models…[/]")
    results = run_models(fe, ema, audio_path=args.audio,
                         n_samples=args.n, device=args.device)

    render_tui(results, epoch, args.checkpoint, fe_msg, ema_msg, snake_raws)


if __name__ == "__main__":
    main()
