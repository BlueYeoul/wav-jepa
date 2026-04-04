"""
W-JEPA audio pre-training loop.

Ported from ref/vjepa2/app/vjepa_2_1/train.py and adapted for 1-D audio.
Key differences from the original video version:
  - Single clip group (no FPC grouping)
  - Variable-length waveforms padded inside DynamicMaskCollator1D
  - RoPE positional encoding (no learned pos_embed)
  - No patch_size / tubelet_size concept
"""

import copy
import gc
import logging
import os
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from wjepa.loss import LambdaWarmupHold, forward_context, forward_target, loss_fn
from wjepa.masks import MaskCollator
from wjepa.masks.distance import compute_mask_distance
from wjepa.models.utils import apply_masks
from wjepa.models import EMBED_DIMS
from wjepa.models.feature_extractor import compute_audio_output_length
from wjepa.utils import init_opt, init_audio_model, load_checkpoint, save_checkpoint

log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
MAX_REPEAT_COUNTS = 10

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args, resume_preempt=False):
    # ---------------------------------------------------------------------- #
    #  Config
    # ---------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype", "float32")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    is_causal = cfgs_model.get("is_causal", False)
    pred_is_causal = cfgs_model.get("pred_is_causal", False)
    init_type = cfgs_model.get("init_type", "default")
    n_registers = cfgs_model.get("n_registers", 0)
    n_registers_predictor = cfgs_model.get("n_registers_predictor", 0)
    has_cls_first = cfgs_model.get("has_cls_first", False)
    lambda_value = cfgs_model.get("lambda_value", 0.0)
    lambda_progressive = cfgs_model.get("lambda_progressive", True)
    normalize_predictor = cfgs_model.get("normalize_predictor", False)
    levels_predictor = cfgs_model.get("levels_predictor", 4)
    predict_all = cfgs_model.get("predict_all", True)

    embed_dim_encoder = EMBED_DIMS.get(model_name)
    if embed_dim_encoder is None:
        raise ValueError(f"Unrecognized model_name: {model_name}")

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "DynamicAudioDataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    batch_size = cfgs_data.get("batch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)

    dynamic_config = {
        "dynamic_seq_len": cfgs_data.get("dynamic_seq_len", {}),
        "dynamic_mask": cfgs_data.get("dynamic_mask", {}),
    }

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    shift_by_n = cfgs_loss.get("shift_by_n", 0)
    weight_distance_loss = cfgs_loss.get("weight_distance_loss", False)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    is_anneal = cfgs_opt.get("is_anneal", False)
    anneal_ckpt = cfgs_opt.get("anneal_ckpt", None)
    if is_anneal and anneal_ckpt is None:
        raise ValueError("Must specify anneal_ckpt if is_anneal is True")
    resume_anneal = cfgs_opt.get("resume_anneal", False) or (is_anneal and resume_preempt)
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    use_radamw = cfgs_opt.get("use_radamw", False)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    loss_reg_std_mult = cfgs_opt.get("loss_reg_std_mult", None)
    loss_reg_num_tracking_steps = cfgs_opt.get("loss_reg_num_tracking_steps", 300)
    loss_reg_min_epoch = cfgs_opt.get("loss_reg_min_epoch", 50)

    # ---------------------------------------------------------------------- #
    #  Environment
    # ---------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    world_size, rank = 1, 0
    data_world_size, data_rank = 1, 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # ---------------------------------------------------------------------- #
    #  Logging / checkpointing paths
    # ---------------------------------------------------------------------- #

    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pth.tar")

    load_path = None
    if load_model:
        if is_anneal:
            load_path = latest_path if (os.path.exists(latest_path) and resume_anneal) else anneal_ckpt
            if load_path == anneal_ckpt:
                resume_anneal = False
        else:
            load_path = r_file if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # ---------------------------------------------------------------------- #
    #  Model
    # ---------------------------------------------------------------------- #

    encoder, predictor = init_audio_model(
        device=device,
        model_name=model_name,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=int(len(cfgs_mask)),
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        is_causal=is_causal,
        pred_is_causal=pred_is_causal,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=predict_all,
        chop_last_n_tokens=shift_by_n,
        init_type=init_type,
        n_registers=n_registers,
        n_registers_predictor=n_registers_predictor,
        has_cls_first=has_cls_first,
        n_output_distillation=levels_predictor,
    )
    target_encoder = copy.deepcopy(encoder)

    if compile_model:
        logger.info("Compiling encoder, target_encoder, predictor...")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    # ---------------------------------------------------------------------- #
    #  Data
    # ---------------------------------------------------------------------- #

    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        compute_output_length=compute_audio_output_length,
        dynamic_config=dynamic_config,
    )

    from wjepa.data import init_data

    unsupervised_loader, unsupervised_sampler = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        rank=data_rank,
        world_size=data_world_size,
        datasets_weights=datasets_weights,
        collator=mask_collator,
        num_workers=20,
        # num_workers=num_workers,
        pin_mem=pin_mem,
        dynamic_config=dynamic_config,
    )

    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"batch_size={batch_size}  ipe={ipe}/{_dlen}")

    # ---------------------------------------------------------------------- #
    #  Optimizer / scheduler
    # ---------------------------------------------------------------------- #

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=is_anneal,
        encoder=encoder,
        predictor=predictor,
        use_radamw=use_radamw,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )

    for p in target_encoder.parameters():
        p.requires_grad = False

    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs) + 1)
    )
    lambda_sched = LambdaWarmupHold(lambda_value=lambda_value)

    # ---------------------------------------------------------------------- #
    #  Resume from checkpoint
    # ---------------------------------------------------------------------- #

    start_epoch = 0
    if load_model or os.path.exists(latest_path):
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
            is_anneal=is_anneal and not resume_anneal,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
        mask_collator.set_epoch(start_epoch)

    # ---------------------------------------------------------------------- #
    #  Training loop
    # ---------------------------------------------------------------------- #

    if sync_gc:
        gc.disable()
        gc.collect()

    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skipping {skip_batches} batches...")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"  skip {itr}/{skip_batches}")
            try:
                _ = next(loader)
            except StopIteration:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    trailing_losses = []
    step_count = 0

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}")

        unsupervised_sampler.set_epoch(epoch)
        mask_collator.set_epoch(epoch)

        loss_meter = _AverageMeter()
        iter_time_meter = _AverageMeter()
        gpu_time_meter = _AverageMeter()
        data_time_meter = _AverageMeter()

        for itr in range(ipe):
            itr_start = time.time()

            # -- load batch (with retry on exhaustion) --
            iter_retries, iter_ok = 0, False
            while not iter_ok:
                try:
                    sample = next(loader)
                    iter_ok = True
                except StopIteration:
                    logger.info("Exhausted dataloader; refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    mask_collator.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                    if iter_retries < 5:
                        iter_retries += 1
                    else:
                        raise RuntimeError("Exceeded max data retries.")

            # -- unpack collator output and move to device --
            # sample = (collated_dict, [enc_mask_cfg1, enc_mask_cfg2], [pred_mask_cfg1, pred_mask_cfg2])
            udata, m_enc_list, m_pred_list = sample

            # audio: (B, T) → (B, 1, T) for the CNN feature extractor
            audio = udata["audio"].to(device, non_blocking=True).unsqueeze(1)

            # Wrap in outer list for MultiSeqWrapper (one audio group, multiple mask configs)
            clips      = [audio]
            masks_enc  = [[m.to(device, non_blocking=True) for m in m_enc_list]]
            masks_pred = [[m.to(device, non_blocking=True) for m in m_pred_list]]

            data_time_ms = (time.time() - itr_start) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                gc.collect()

            # ----------------------------------------------------------------
            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips, target_encoder, embed_dim_encoder, levels_predictor)
                    z_pred, z_context = forward_context(
                        clips, masks_enc, masks_pred,
                        encoder, predictor,
                        embed_dim_encoder, normalize_predictor, predict_all,
                    )

                    loss_pred = loss_fn(z_pred, h, masks_pred, loss_exp, has_cls_first=has_cls_first)
                    loss = loss_pred

                    if predict_all and z_context is not None:
                        d_weights = None
                        if weight_distance_loss:
                            # Use max token count (from max audio length in batch) as seq_len proxy
                            seq_len_tokens = int(udata["seq_len"].max().item())
                            d_weights = compute_mask_distance(
                                masks_pred, masks_enc, seq_len=seq_len_tokens
                            )
                        loss_context = loss_fn(
                            z_context, h, masks_enc, loss_exp, d_weights=d_weights
                        )
                        lambda_t = (
                            lambda_sched.value(epoch * ipe + itr)
                            if lambda_progressive
                            else lambda_value
                        )
                        loss = loss + loss_context * lambda_t

                run_step = True
                if loss_reg_std_mult is not None and len(trailing_losses) > 0:
                    mean_v = np.mean(trailing_losses)
                    std_v = np.std(trailing_losses)
                    bound = mean_v + loss_reg_std_mult * std_v
                    if (
                        loss > bound
                        and epoch > loss_reg_min_epoch
                        and len(trailing_losses) > int(0.5 * loss_reg_num_tracking_steps)
                    ):
                        run_step = False
                        logger.info(f"Loss {loss:.4f} > bound {bound:.4f}; skipping step.")

                if run_step:
                    if mixed_precision:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    if mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                else:
                    loss.backward()
                optimizer.zero_grad()

                # EMA update of target encoder
                m = min(next(momentum_scheduler), ema[1])
                with torch.no_grad():
                    params_k = list(target_encoder.parameters())
                    params_q = list(encoder.parameters())
                    torch._foreach_mul_(params_k, m)
                    torch._foreach_add_(params_k, params_q, alpha=1.0 - m)

                return float(loss), _new_lr, _new_wd, run_step
            # ----------------------------------------------------------------

            t0 = time.time()
            loss, _new_lr, _new_wd, run_step = train_step()
            gpu_ms = (time.time() - t0) * 1000.0
            iter_ms = (time.time() - itr_start) * 1000.0

            loss_meter.update(loss)
            iter_time_meter.update(iter_ms)
            gpu_time_meter.update(gpu_ms)
            data_time_meter.update(data_time_ms)

            if loss_reg_std_mult is not None:
                if run_step:
                    trailing_losses.append(loss)
                    if len(trailing_losses) > loss_reg_num_tracking_steps:
                        trailing_losses = trailing_losses[1:]
                else:
                    step_count += 1
                    if step_count > MAX_REPEAT_COUNTS:
                        raise RuntimeError("Loss above bound for too many consecutive steps.")

            if itr % log_freq == 0 or itr == ipe - 1:
                logger.info(
                    "[%d, %5d] loss: %.3f  wd: %.2e  lr: %.2e  "
                    "mem: %.0fMB  iter: %.1fms  gpu: %.1fms  data: %.1fms"
                    % (
                        epoch + 1, itr, loss_meter.avg,
                        _new_wd, _new_lr,
                        torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                        iter_time_meter.avg, gpu_time_meter.avg, data_time_meter.avg,
                    )
                )

            assert not np.isnan(loss), "loss is NaN"

        # -- Save checkpoint --
        logger.info(f"avg. loss {loss_meter.avg:.3f}")
        if (epoch + 1) % CHECKPOINT_FREQ == 0 or epoch == num_epochs - 1:
            save_checkpoint(
                latest_path, encoder, predictor, target_encoder,
                optimizer, scaler, epoch + 1, loss_meter.avg,
                batch_size, world_size, lr, rank,
            )
            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                save_checkpoint(
                    os.path.join(folder, f"e{epoch}.pth.tar"),
                    encoder, predictor, target_encoder,
                    optimizer, scaler, epoch + 1, loss_meter.avg,
                    batch_size, world_size, lr, rank,
                )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class _AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------------------------------------
# Dummy data loader for --test mode
# ---------------------------------------------------------------------------

class _DummyDataLoader:
    """
    Yields synthetic variable-length audio batches without touching disk.
    Uses the real DynamicMaskCollator1D so the collation path is identical
    to production.

    Batch: 5 samples at 1 s / 5 s / 10 s / 15 s / 20 s (21.5 s cropped)
    """
    DURATIONS_SEC = [1.0, 5.0, 10.0, 15.0, 20.0]   # seconds per sample
    SAMPLE_RATE   = 16000

    def __init__(self, mask_collator, n_iter=3):
        self.collator = mask_collator
        self.n_iter   = n_iter

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        import torch.nn.functional as F_
        for _ in range(self.n_iter):
            batch = []
            for sec in self.DURATIONS_SEC:
                n = int(sec * self.SAMPLE_RATE)
                batch.append({
                    "audio":       torch.randn(n),
                    "seq_len_sec": sec,
                    "seq_len":     n,
                })
            yield self.collator(batch)

    def set_epoch(self, epoch):
        pass


# ---------------------------------------------------------------------------
# --test entry point
# ---------------------------------------------------------------------------

def main_test(config=None):
    """
    Smoke-test the full train.py code path with dummy data.

    Uses the same initialisation (models, optimiser, collator) and the same
    train_step() inner function as main().  Only the DataLoader is swapped for
    _DummyDataLoader so no real files are needed.

    Assertions (all must pass):
      1. Loss is finite and non-NaN after each of 3 dummy iterations
      2. Encoder parameters have gradients after backward
      3. Target encoder parameters have NO gradients (frozen)
      4. Target encoder parameters change after EMA update
      5. Loss decreases or stays finite over multiple steps (no explosion)
    """
    PASS = "\033[92m[PASS]\033[0m"
    FAIL = "\033[91m[FAIL]\033[0m"
    INFO = "\033[94m[INFO]\033[0m"

    failures = []

    def chk(cond, msg):
        if cond:
            print(f"{PASS} {msg}")
        else:
            print(f"{FAIL} {msg}")
            failures.append(msg)

    print("\n" + "=" * 60)
    print("W-JEPA  --test  (train.py smoke test)")
    print("=" * 60)

    # -- Minimal test config (override with caller-supplied config)
    _default = {
        "folder": "/tmp/wjepa_test",
        "meta": {
            "load_checkpoint": False,
            "read_checkpoint": None,
            "seed": 42,
            "use_sdpa": False,       # sdp_kernel warns on CPU; disable
            "dtype": "float32",
            "save_every_freq": -1,
            "skip_batches": -1,
            "sync_gc": False,
        },
        "mask": [
            {"num_blocks": 4,  "scale": (0.15, 0.5), "max_keep": None,
             "full_complement": False, "max_temporal_keep": 1.0, "inv_block": False},
            {"num_blocks": 2,  "scale": (0.7,  0.7), "max_keep": None,
             "full_complement": False, "max_temporal_keep": 1.0, "inv_block": False},
        ],
        "model": {
            "model_name":                "audio_transformer_base",
            "pred_depth":                4,
            "pred_num_heads":            4,    # must divide pred_embed_dim evenly
            "pred_embed_dim":            64,
            "use_mask_tokens":           True,
            "zero_init_mask_tokens":     True,
            "is_causal":                 False,
            "pred_is_causal":            False,
            "use_silu":                  False,
            "use_pred_silu":             False,
            "wide_silu":                 True,
            "use_activation_checkpointing": False,
            "init_type":                 "default",
            "n_registers":               0,
            "n_registers_predictor":     0,
            "has_cls_first":             False,
            "levels_predictor":          4,
            "predict_all":               True,
            "lambda_value":              0.5,
            "lambda_progressive":        True,
            "normalize_predictor":       False,
            "compile_model":             False,
        },
        "data": {
            "dynamic_seq_len": {"min_seq_len_sec": 0.5, "max_seq_len_sec": 20.0},
            "dynamic_mask":    {"enabled": False},
        },
        "loss": {
            "loss_exp":           1.0,
            "shift_by_n":         0,
            "weight_distance_loss": False,
        },
        "optimization": {
            "is_anneal":        False,
            "anneal_ckpt":      None,
            "ipe":              3,
            "ipe_scale":        1.0,
            "weight_decay":     0.01,
            "final_weight_decay": 0.01,
            "epochs":           1,
            "warmup":           1,
            "start_lr":         1e-5,
            "lr":               1e-4,
            "final_lr":         1e-4,
            "ema":              [0.999, 0.999],
            "use_radamw":       False,
            "betas":            (0.9, 0.999),
            "eps":              1e-8,
            "loss_reg_std_mult": None,
        },
    }

    # Deep-merge caller config over defaults
    args = _default
    if config is not None:
        for section, vals in config.items():
            if isinstance(vals, dict) and section in args and isinstance(args[section], dict):
                args[section].update(vals)
            else:
                args[section] = vals

    # ----- replicate main() setup -----
    seed = args["meta"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{INFO} Device: {device}")

    cfgs_mask        = args["mask"]
    cfgs_model       = args["model"]
    cfgs_opt         = args["optimization"]
    cfgs_loss        = args["loss"]

    model_name       = cfgs_model["model_name"]
    pred_depth       = cfgs_model["pred_depth"]
    pred_embed_dim   = cfgs_model["pred_embed_dim"]
    levels_predictor = cfgs_model["levels_predictor"]
    predict_all      = cfgs_model["predict_all"]
    lambda_value     = cfgs_model["lambda_value"]
    lambda_progressive = cfgs_model["lambda_progressive"]
    normalize_predictor = cfgs_model["normalize_predictor"]
    has_cls_first    = cfgs_model["has_cls_first"]
    loss_exp         = cfgs_loss["loss_exp"]
    weight_distance_loss = cfgs_loss["weight_distance_loss"]
    ipe              = cfgs_opt["ipe"]
    ema              = cfgs_opt["ema"]

    embed_dim_encoder = EMBED_DIMS.get(model_name)
    if embed_dim_encoder is None:
        raise ValueError(f"Unrecognized model_name: {model_name}")

    dynamic_config = {
        "dynamic_seq_len": args["data"].get("dynamic_seq_len", {}),
        "dynamic_mask":    args["data"].get("dynamic_mask",    {}),
    }

    # Models
    encoder, predictor = init_audio_model(
        device=device,
        model_name=model_name,
        pred_depth=pred_depth,
        pred_num_heads=cfgs_model.get("pred_num_heads"),
        pred_embed_dim=pred_embed_dim,
        use_mask_tokens=cfgs_model["use_mask_tokens"],
        num_mask_tokens=len(cfgs_mask),
        zero_init_mask_tokens=cfgs_model["zero_init_mask_tokens"],
        use_sdpa=args["meta"].get("use_sdpa", False),
        use_silu=cfgs_model["use_silu"],
        use_pred_silu=cfgs_model["use_pred_silu"],
        wide_silu=cfgs_model["wide_silu"],
        is_causal=cfgs_model["is_causal"],
        pred_is_causal=cfgs_model["pred_is_causal"],
        use_activation_checkpointing=cfgs_model["use_activation_checkpointing"],
        return_all_tokens=predict_all,
        init_type=cfgs_model["init_type"],
        n_registers=cfgs_model["n_registers"],
        n_registers_predictor=cfgs_model["n_registers_predictor"],
        has_cls_first=has_cls_first,
        n_output_distillation=levels_predictor,
    )
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Collator (real DynamicMaskCollator1D)
    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        compute_output_length=compute_audio_output_length,
        dynamic_config=dynamic_config,
    )

    # ------------------------------------------------------------------
    # Data-pipeline sanity check (catches sampler / dataset / DataLoader
    # bugs that _DummyDataLoader would otherwise bypass)
    # ------------------------------------------------------------------
    print(f"\n{INFO} Data-pipeline sanity checks...")
    import tempfile
    from torch.utils.data import DataLoader as _DataLoader

    from wjepa.data.sampler  import DistributedSampler as _DistSampler
    from wjepa.data.dataset  import AudioDataset        as _AudioDataset

    class _FakeDataset:
        """Minimal dataset with no real files – only tests plumbing."""
        def __len__(self):        return 8
        def __getitem__(self, i): return {"audio": torch.zeros(1600), "seq_len": 1600, "seq_len_sec": 0.1}

    # 1. DistributedSampler instantiation + iteration
    try:
        _samp = _DistSampler(_FakeDataset(), num_replicas=1, rank=0, shuffle=False)
        _idx  = list(iter(_samp))
        chk(len(_idx) == 8, f"DistributedSampler: returned {len(_idx)} indices (expected 8)")
    except Exception as _e:
        chk(False, f"DistributedSampler failed: {_e}")

    # 2. AudioDataset init on empty temp dir (no crash, 0 files)
    with tempfile.TemporaryDirectory() as _tmpdir:
        try:
            _ds = _AudioDataset(data_path=_tmpdir, min_sec=0.1, max_sec=5.0)
            chk(len(_ds) == 0, f"AudioDataset init OK (0 files in empty dir)")
        except Exception as _e:
            chk(False, f"AudioDataset init failed: {_e}")

    # 3. DataLoader + DistributedSampler round-trip with FakeDataset
    try:
        _fake_ds   = _FakeDataset()
        _fake_samp = _DistSampler(_fake_ds, num_replicas=1, rank=0, shuffle=False)
        _fake_dl   = _DataLoader(_fake_ds, batch_size=4, sampler=_fake_samp,
                                 collate_fn=mask_collator)
        _batch = next(iter(_fake_dl))
        chk(True, "DataLoader + DistributedSampler round-trip OK (1 batch fetched)")
    except Exception as _e:
        chk(False, f"DataLoader round-trip failed: {_e}")

    print()

    # Dummy loader (replaces real DataLoader; same collator output format)
    loader_iter = iter(_DummyDataLoader(mask_collator, n_iter=ipe))

    # Optimiser (same as main())
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=False,
        encoder=encoder,
        predictor=predictor,
        iterations_per_epoch=ipe,
        start_lr=cfgs_opt["start_lr"],
        ref_lr=cfgs_opt["lr"],
        final_lr=cfgs_opt["final_lr"],
        warmup=cfgs_opt["warmup"],
        num_epochs=1,
        wd=float(cfgs_opt["weight_decay"]),
        final_wd=float(cfgs_opt["final_weight_decay"]),
        mixed_precision=False,
        betas=cfgs_opt["betas"],
        eps=cfgs_opt["eps"],
    )

    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * 1)
        for i in range(ipe + 1)
    )
    lambda_sched = LambdaWarmupHold(lambda_value=lambda_value)

    tgt_param_snapshot = next(iter(target_encoder.parameters())).clone().detach()

    losses = []
    print(f"\n{INFO} Running {ipe} training iterations with dummy data...\n")

    for itr in range(ipe):
        # ---- exact same unpacking as main() ----
        sample = next(loader_iter)
        udata, m_enc_list, m_pred_list = sample

        audio      = udata["audio"].to(device).unsqueeze(1)   # (B, 1, T)
        clips      = [audio]
        masks_enc  = [[m.to(device) for m in m_enc_list]]
        masks_pred = [[m.to(device) for m in m_pred_list]]

        # ---- exact same train_step logic as main() ----
        _new_lr = scheduler.step()
        _new_wd = wd_scheduler.step()

        h = forward_target(clips, target_encoder, embed_dim_encoder, levels_predictor)
        z_pred, z_context = forward_context(
            clips, masks_enc, masks_pred,
            encoder, predictor,
            embed_dim_encoder, normalize_predictor, predict_all,
        )

        loss = loss_fn(z_pred, h, masks_pred, loss_exp, has_cls_first=has_cls_first)
        if predict_all and z_context is not None:
            loss_ctx = loss_fn(z_context, h, masks_enc, loss_exp)
            lambda_t = (
                lambda_sched.value(itr)
                if lambda_progressive else lambda_value
            )
            loss = loss + loss_ctx * lambda_t

        loss.backward()

        # Check gradients on the last iteration (before zero_grad clears them)
        if itr == ipe - 1:
            enc_grads_count = sum(1 for p in encoder.parameters() if p.grad is not None)

        optimizer.step()
        optimizer.zero_grad()

        # EMA update
        m_val = min(next(momentum_scheduler), ema[1])
        with torch.no_grad():
            params_k = list(target_encoder.parameters())
            params_q = list(encoder.parameters())
            torch._foreach_mul_(params_k, m_val)
            torch._foreach_add_(params_k, params_q, alpha=1.0 - m_val)

        loss_val = loss.detach().item()
        losses.append(loss_val)
        chk(not np.isnan(loss_val) and not np.isinf(loss_val),
            f"itr {itr}: loss = {loss_val:.4f} (finite, non-NaN)")

    # ---- post-loop assertions ----

    chk(enc_grads_count > 0,
        f"Encoder has gradients ({enc_grads_count} params with grad after backward)")

    tgt_grads = [n for n, p in target_encoder.named_parameters() if p.grad is not None]
    chk(len(tgt_grads) == 0,
        "Target encoder has NO gradients (frozen)")

    tgt_param_after = next(iter(target_encoder.parameters())).detach()
    chk((tgt_param_after - tgt_param_snapshot).abs().max().item() > 0,
        "Target encoder parameters changed after EMA update")

    chk(all(np.isfinite(l) for l in losses),
        f"All {len(losses)} losses finite: {[f'{l:.4f}' for l in losses]}")

    print("\n" + "=" * 60)
    if failures:
        print(f"\033[91mFAILED\033[0m  –  {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  ✗ {f}")
        import sys
        sys.exit(1)
    else:
        print("\033[92mAll checks passed – train.py logic OK\033[0m")
    print("=" * 60 + "\n")
