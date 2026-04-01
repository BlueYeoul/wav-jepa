"""
V-JEPA 2.1 pre-training loop.

Ported from ref/vjepa2/app/vjepa_2_1/train.py
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
from wjepa.masks.collator import MaskCollator
from wjepa.masks.distance import compute_mask_distance
from wjepa.models.utils import apply_masks
from wjepa.models import EMBED_DIMS
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
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    is_causal = cfgs_model.get("is_causal", False)
    pred_is_causal = cfgs_model.get("pred_is_causal", False)
    init_type = cfgs_model.get("init_type", "default")
    img_temporal_dim_size = cfgs_model.get("img_temporal_dim_size", None)
    n_registers = cfgs_model.get("n_registers", 0)
    has_cls_first = cfgs_model.get("has_cls_first", False)
    interpolate_rope = cfgs_model.get("interpolate_rope", False)
    lambda_value_img = cfgs_model.get("lambda_value_img", 0.0)
    lambda_value_vid = cfgs_model.get("lambda_value_vid", 0.0)
    n_registers_predictor = cfgs_model.get("n_registers_predictor", 0)
    lambda_progressive = cfgs_model.get("lambda_progressive", True)
    normalize_predictor = cfgs_model.get("normalize_predictor", False)
    modality_embedding = cfgs_model.get("modality_embedding", False)
    levels_predictor = cfgs_model.get("levels_predictor", 4)

    embed_dim_encoder = EMBED_DIMS.get(model_name)
    if embed_dim_encoder is None:
        raise ValueError(f"Unrecognized model_name: {model_name}")

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)

    # -- IMG DATA (optional)
    cfgs_img_data = args.get("img_data")
    img_rank_ratio = 0.25
    img_mask = None
    if cfgs_img_data is not None:
        img_dataset_type = cfgs_img_data.get("dataset_type", "imagenet")
        img_dataset_paths = cfgs_img_data.get("datasets", [])
        img_dataset_weights = cfgs_img_data.get("datasets_weights", [])
        img_dataset_fpcs = cfgs_img_data.get("dataset_fpcs")
        img_dataset_batch_size = cfgs_img_data.get("batch_size")
        img_rank_ratio = cfgs_img_data.get("rank_ratio", img_rank_ratio)
        img_num_workers = cfgs_img_data.get("num_workers", num_workers)
        img_mask = args.get("img_mask", img_mask)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    shift_by_n = cfgs_loss.get("shift_by_n")
    predict_all = cfgs_loss.get("predict_all", True)
    weight_distance_loss = cfgs_loss.get("weight_distance_loss", False)
    offset_context_loss = cfgs_loss.get("offset_context_loss", False)

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

    # Single-process execution
    world_size, rank = 1, 0
    data_world_size, data_rank = 1, 0
    model_fpcs = dataset_fpcs
    model_cfgs_mask = cfgs_mask
    model_tubelet_size = tubelet_size
    lambda_value = lambda_value_vid

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
    print(len(model_cfgs_mask), model_fpcs)
    encoder, predictor = init_audio_model(
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=int(len(model_cfgs_mask)),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        tubelet_size=model_tubelet_size,
        model_name=model_name,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        is_causal=is_causal,
        pred_is_causal=pred_is_causal,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=predict_all,
        chop_last_n_tokens=shift_by_n,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        n_registers_predictor=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
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
        dataset_fpcs=dataset_fpcs,
    )

    from wjepa.data import init_data

    unsupervised_loader, unsupervised_sampler = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        rank=data_rank,
        world_size=data_world_size,
        datasets_weights=datasets_weights,
        collator=mask_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
    )

    try:
        _dlen = len(unsupervised_loader)
    except Exception:
        try:
            _dlen = unsupervised_loader.num_batches
        except Exception:
            _dlen = -1
    if ipe is None:
        ipe = _dlen
    logger.info(f"batch_size={batch_size}  fpcs={dataset_fpcs}  ipe={ipe}/{_dlen}")

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

    # Models running in single-process mode
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
        if not is_anneal or resume_anneal:
            for _ in range(start_epoch * ipe):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()

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
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    trailing_losses = []
    step_count = 0

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}")

        loss_meter = _AverageMeter()
        iter_time_meter = _AverageMeter()
        gpu_time_meter = _AverageMeter()
        data_time_meter = _AverageMeter()

        for itr in range(ipe):
            itr_start = time.time()

            # -- load batch (with retry) --
            iter_retries, iter_ok = 0, False
            while not iter_ok:
                try:
                    sample = next(loader)
                    iter_ok = True
                except StopIteration:
                    logger.info("Exhausted dataloader; refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    if iter_retries < 5:
                        logger.warning(f"Data load error (retry {iter_retries}): {e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        raise RuntimeError("Exceeded max data retries.") from e

            # -- move to device --
            clips, masks_enc, masks_pred = [], [], []
            for fpc_sample in sample:
                udata, m_enc, m_pred = fpc_sample
                clips.append(udata[0][0].to(device, non_blocking=True))
                masks_enc.append([m.to(device, non_blocking=True) for m in m_enc])
                masks_pred.append([m.to(device, non_blocking=True) for m in m_pred])
            data_time_ms = (time.time() - itr_start) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                gc.collect()

            # ----------------------------------------------------------------
            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips, target_encoder, embed_dim_encoder, levels_predictor)
                    z_pred, z_context = forward_context(
                        clips, masks_enc, masks_pred,
                        encoder, predictor,
                        embed_dim_encoder, normalize_predictor, predict_all,
                        img_temporal_dim_size,
                    )

                    loss_pred = loss_fn(z_pred, h, masks_pred, loss_exp, has_cls_first=has_cls_first)
                    loss = loss_pred

                    if predict_all:
                        d_weights = None
                        if weight_distance_loss:
                            d_weights = compute_mask_distance(
                                masks_pred, masks_enc,
                                seq_len = clips[0].shape[1] if len(clips) > 0 else 0
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

                # Step 2. Backward & step
                run_step = True
                if loss_reg_std_mult is not None:
                    mean_v = np.mean(trailing_losses)
                    std_v = np.std(trailing_losses)
                    bound = mean_v + loss_reg_std_mult * std_v
                    if (
                        loss > bound
                        and epoch > loss_reg_min_epoch
                        and len(trailing_losses) > int(0.5 * loss_reg_num_tracking_steps)
                    ):
                        run_step = False
                        loss.backward()
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
                optimizer.zero_grad()

                # Step 3. EMA update of target encoder
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
                        raise RuntimeError("Loss above bound for too many steps.")

            if itr % log_freq == 0 or itr == ipe - 1:
                logger.info(
                    "[%d, %5d] loss: %.3f  wd: %.2e  lr: %.2e  "
                    "mem: %.0fMB  iter: %.1fms  gpu: %.1fms  data: %.1fms"
                    % (
                        epoch + 1, itr, loss_meter.avg,
                        _new_wd, _new_lr,
                        torch.cuda.max_memory_allocated() / 1024**2,
                        iter_time_meter.avg, gpu_time_meter.avg, data_time_meter.avg,
                    )
                )

            assert not np.isnan(loss), "loss is NaN"

        # -- Save checkpoint
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
# Minimal AverageMeter (avoids dependency on external logging utils)
# ---------------------------------------------------------------------------

class _AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
