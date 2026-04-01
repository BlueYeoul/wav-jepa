"""
V-JEPA 2.1 training utilities:
  init_video_model  – build encoder + predictor
  init_opt          – build optimizer / scheduler / scaler
  load_checkpoint   – restore model + optimizer state
"""

import logging
import sys

import torch
from wjepa.models import encoder as audio_enc_module
from wjepa.models import predictor as audio_pred_module
from wjepa.models.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model init
# ---------------------------------------------------------------------------

def init_audio_model(
    device,
    seq_len=16000,
    patch_size=16,
    in_chans=1,
    model_name="audio_transformer_large",
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=True,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=True,
    is_causal=False,
    pred_is_causal=False,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    init_type="default",
    n_registers=0,
    n_registers_predictor=0,
    has_cls_first=False,
    n_output_distillation=4,
    **kwargs,
):
    # Note: AudioTransformer now uses Wav2Vec2-style PatchEmbed which fixes patch_size to 320.
    # The patch_size passed here is kept for compatibility but overridden in the backbone.
    enc = audio_enc_module.__dict__[model_name](
        seq_len=seq_len,
        patch_size=patch_size,
        in_chans=in_chans,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        is_causal=is_causal,
        use_rope=use_rope,
        init_type=init_type,
        n_registers=n_registers,
        has_cls_first=has_cls_first,
        n_output_distillation=n_output_distillation,
    )
    encoder = MultiSeqWrapper(enc)

    # Use the actual patch_size from the encoder for the predictor
    actual_patch_size = encoder.backbone.patch_size

    pred = audio_pred_module.audio_predictor(
        seq_len=seq_len,
        patch_size=actual_patch_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        is_causal=pred_is_causal,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=return_all_tokens,
        n_registers=n_registers_predictor,
        has_cls_first=has_cls_first,
        n_output_distillation=n_output_distillation,
    )
    predictor = PredictorMultiSeqWrapper(pred)

    encoder.to(device)
    predictor.to(device)

    def _n_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    logger.info(f"Encoder params  : {_n_params(encoder):,}")
    logger.info(f"Predictor params: {_n_params(predictor):,}")

    return encoder, predictor


# ---------------------------------------------------------------------------
# Optimizer / scheduler init
# ---------------------------------------------------------------------------

def init_opt(
    is_anneal,
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    use_radamw=False,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    from wjepa.schedulers import CosineWDSchedule, LinearDecaySchedule, WarmupCosineSchedule

    param_groups = [
        {
            "params": (
                p for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p for n, p in encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (
                p for n, p in predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    if use_radamw:
        from wjepa.adamw import AdamW as RAdamW
        logger.info("Using Rescaled-AdamW")
        optimizer = RAdamW(param_groups, betas=betas, eps=eps)
    else:
        logger.info("Using AdamW")
        optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    T_max = int(ipe_scale * num_epochs * iterations_per_epoch)

    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=T_max,
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=T_max,
        )

    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=T_max,
    )

    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(r_path, encoder, predictor, target_encoder, opt, scaler, is_anneal=False):
    logger.info(f"Loading checkpoint: {r_path}")
    checkpoint = torch.load(r_path, map_location="cpu")

    epoch = 0 if is_anneal else checkpoint["epoch"]

    def _load(model, key):
        sd = checkpoint[key]
        model_sd = model.state_dict()
        for k, v in model_sd.items():
            if k not in sd:
                logger.info(f'  missing key "{k}"')
            elif sd[k].shape != v.shape:
                logger.info(f'  shape mismatch "{k}": {sd[k].shape} vs {v.shape}')
                sd[k] = v
        msg = model.load_state_dict(sd, strict=False)
        logger.info(f"  loaded {key} from epoch {epoch}: {msg}")

    _load(encoder, "encoder")
    _load(predictor, "predictor")
    if target_encoder is not None:
        _load(target_encoder, "target_encoder")

    try:
        opt.load_state_dict(checkpoint["opt"])
    except ValueError:
        logger.warning("Optimizer groups mismatch; reinitializing optimizer.")

    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"Checkpoint loaded (epoch {epoch})")
    del checkpoint

    return encoder, predictor, target_encoder, opt, scaler, epoch


def save_checkpoint(path, encoder, predictor, target_encoder, optimizer, scaler,
                    epoch, loss, batch_size, world_size, lr, rank=0):
    if rank != 0:
        return
    save_dict = {
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "opt": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "batch_size": batch_size,
        "world_size": world_size,
        "lr": lr,
    }
    try:
        torch.save(save_dict, path)
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")
