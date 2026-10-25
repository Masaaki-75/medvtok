"""Unified MedVTok entry point for stage-1 2D and stage-2 2D/3D training."""

import gc
import glob
import os
import sys
import time

import torch
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.csv_imagefeature_dataset import get_csvimgfeat_dataset
from datasets.csv_volumefeature_dataset import get_csvvolfeat_dataset
from datasets.transforms import build_image_transform
from models.discrim import build_discriminator
from models.medvtok_wrapper import build_medvtok_wrapper
from trainers.optimizer import build_optimizer
from trainers.scheduler import LRScheduler
from trainers.trainer import Trainer as SingleLoaderTrainer
from trainers.trainer import train_one_ep as train_one_ep_single
from trainers.trainer_biloader import BiLoaderTrainer, InfiniteDataLoader
from trainers.trainer_biloader import train_one_ep as train_one_ep_biloader
from trainers.visualizer import setup_visualizer
from utilities import config, dist
from utilities.loss import ClipLoss, SigLipLoss
from utilities.lpips import LPIPS


def maybe_auto_resume(args: config.Args, pattern=("ckpt*.pth", "ep*-it*.pth")):
    if args.resume_from:
        resume = args.resume_from
        print(f"[auto_resume] Load from args.resume @ {resume} ...")
    else:
        patterns = (pattern,) if isinstance(pattern, str) else pattern
        all_ckpt = {
            path for item in patterns
            for path in glob.glob(os.path.join(args.output_dir, item), recursive=False)
        }
        all_ckpt = sorted(all_ckpt, key=os.path.getmtime, reverse=True)
        if not all_ckpt:
            print(f"[auto_resume] NO ckpt found @ {pattern}")
            print("[auto_resume quit]")
            return {}
        resume = all_ckpt[0]
        print(f"[auto_resume] Auto resume from @ {resume} ...")

    print(
        "[auto_resume] Load networks only (w/o optimizers)? "
        f"@ {args.resume_net_only} ..."
    )
    try:
        loaded = torch.load(resume, map_location="cpu")
        dist.barrier()

        if "trainer" not in loaded:
            print(f"[auto_resume] Load BARE weights @ {resume}")
            ckpt = {
                "epoch": 0,
                "iter": 0,
                "args": args,
                "trainer": {"model": loaded},
            }
        else:
            ckpt = loaded
            if args.resume_net_only:
                ckpt["epoch"] = 0
                ckpt["iter"] = 0
                ckpt["args"] = args

        print(
            "[auto_resume success] Resume "
            f"ep{ckpt['epoch']} & it{ckpt['iter']} @ {resume}"
        )
        return ckpt
    except Exception as error:
        print(f"[auto_resume] Failed, {error} @ {resume}")
        return {}


def _maybe_initialize_siglip_encoder(model, args: config.Args, trainer_state: dict):
    """Initialize encoder tensors only; checkpoint restoration takes precedence."""
    if not args.siglip_encoder_init:
        return
    if "model" in trainer_state:
        print("[siglip init] Skipped: restoring model weights from checkpoint.")
        return

    from safetensors.torch import load_file

    state = load_file(args.siglip_encoder_init, device="cpu")
    model.core.encoder.load_state_dict(state, strict=True)
    print(f"[siglip init] Loaded encoder weights from {args.siglip_encoder_init}")


def _configure_training_mode(args: config.Args) -> bool:
    """Normalize mode-specific options and return whether stage 2 is active."""
    use_biloader = bool(getattr(args, "include_3d_data", False))
    if use_biloader:
        if not getattr(args, "train_data_3d", None):
            raise ValueError("Stage 2 requires --train_data_3d.")
        if args.local_bs_3d <= 0:
            args.local_bs_3d = args.local_bs
            print(f"[stage2] local_bs_3d uses local_bs={args.local_bs}")
        print("[mode] Stage 2: alternating 2D/3D bi-loader training")
        return True

    # Preserve the original stage-1 entry point's strict 2D-only behavior.
    overridden = []
    for name, stage1_value in (
        ("dataset_type", "csvimgfeat"),
        ("include_3d_data", False),
        ("include_3d_projector", False),
        ("lslidr", 0.0),
    ):
        if getattr(args, name, stage1_value) != stage1_value:
            overridden.append(name)
        setattr(args, name, stage1_value)
    if overridden:
        print(f"[stage1] Enforced 2D-only options: {', '.join(overridden)}")
    print("[mode] Stage 1: 2D-only single-loader training")
    return False


def main():
    args = config.init_dist_and_get_args()
    use_biloader = _configure_training_mode(args)
    print(f"[args] initial args:\n{args}")
    force_sem_only = getattr(args, "force_sem_only", False)

    ckpt = maybe_auto_resume(args)
    start_iter = ckpt.get("iter", 0)
    start_epoch = ckpt.get("epoch", 0)
    trainer_state = ckpt.get("trainer", {})

    print("[data] Load 2D data...\n")
    if use_biloader:
        if not args.train_data:
            raise ValueError("Stage 2 requires --train_data for its 2D loader.")
        data_2d = get_csvimgfeat_dataset(
            args,
            is_train=True,
            epoch=start_epoch,
            iters=start_iter,
            tokenizer=None,
        )

        print("[data] Load 3D data...\n")
        data_3d = get_csvvolfeat_dataset(
            args,
            is_train=True,
            epoch=start_epoch,
            iters=0,
            tokenizer=None,
        )
        data = {
            "train_2d": data_2d,
            "train_3d": data_3d,
        }
    else:
        data = {}
        if args.train_data:
            data["train"] = get_csvimgfeat_dataset(
                args,
                is_train=True,
                epoch=start_epoch,
                iters=start_iter,
                tokenizer=None,
            )
        if args.val_data:
            data["val"] = get_csvimgfeat_dataset(
                args,
                is_train=False,
                epoch=start_epoch,
                iters=start_iter,
                tokenizer=None,
            )
        if "train" not in data:
            raise ValueError("Stage 1 requires --train_data and a 2D training dataset.")

    model = build_medvtok_wrapper(args)
    if not use_biloader:
        _maybe_initialize_siglip_encoder(model, args, trainer_state)
    disc = build_discriminator(args) if not force_sem_only else nn.Identity()

    if args.lock_text and hasattr(model, "lock_text_tower"):
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm,
            freeze_logit_scale=args.freeze_logit_scale,
        )

    arch = args.model.lower()
    if "noalign" in arch:
        args.lshare = 0

    print(f"[model] Model #params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} (M)")
    print(f"[model] Disc #params {sum(p.numel() for p in disc.parameters()) / 1e6:.2f} (M)")

    model_optim = build_optimizer(args, "model", model)
    disc_optim = build_optimizer(args, "dis", disc) if not force_sem_only else None

    if use_biloader:
        num_2d_batches = data["train_2d"].num_batches
        num_iters_per_epoch = 2 * num_2d_batches
    else:
        num_2d_batches = None
        num_iters_per_epoch = data["train"].num_batches
    max_iter = args.epoch * num_iters_per_epoch
    warmup_iter = args.warmup_ep * num_iters_per_epoch
    disc_max_iter = max_iter - args.disc_start_ep * num_iters_per_epoch
    disc_warmup_iter = args.disc_warmup_ep * num_iters_per_epoch

    model_scheduler = LRScheduler(
        model_optim.optimizer,
        {
            "lr": args.lr,
            "type": args.schedule,
            "start_factor": args.lr_start_ratio,
            "end_factor": args.lr_end_ratio,
            "warmup_iter": warmup_iter,
            "max_iter": max_iter,
        },
    )
    disc_scheduler = (
        LRScheduler(
            disc_optim.optimizer,
            {
                "lr": args.disc_lr,
                "type": args.schedule,
                "start_factor": args.lr_start_ratio,
                "end_factor": args.disc_lr_end_ratio,
                "warmup_iter": disc_warmup_iter,
                "max_iter": disc_max_iter,
            },
        )
        if not force_sem_only
        else None
    )

    contrastive_type = args.contrastive_type.lower()
    if contrastive_type == "clip" or "clip" in arch:
        print(f"[contrastive loss] Using CLIP loss for model {arch}")
        clip_loss = ClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            use_horovod=False,
        )
    else:
        print(f"[contrastive loss] Using SigLIP loss for model {arch}")
        clip_loss = SigLipLoss(
            cache_labels=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            use_horovod=False,
        )

    lpips_loss = LPIPS(args.lpips_path).to(args.device)

    if args.compile_model:
        model = torch.compile(model, backend="inductor")
        disc = torch.compile(disc, backend="inductor") if not force_sem_only else None
        lpips_loss = torch.compile(lpips_loss, backend="inductor")

    if "noalign" in arch or force_sem_only:
        model = DDP(
            model,
            device_ids=[dist.get_local_rank()],
            static_graph=args.ddp_static,
            find_unused_parameters=True,
        )
    else:
        model = DDP(
            model,
            device_ids=[dist.get_local_rank()],
            static_graph=args.ddp_static,
        )
    disc = (
        DDP(disc, device_ids=[dist.get_local_rank()], static_graph=args.ddp_static)
        if not force_sem_only
        else None
    )

    trainer_class = BiLoaderTrainer if use_biloader else SingleLoaderTrainer
    trainer = trainer_class(
        args=args,
        model=model,
        disc=disc,
        model_optim=model_optim,
        disc_optim=disc_optim,
        lpips_loss=lpips_loss,
        clip_loss=clip_loss,
    )

    if trainer_state:
        trainer.load_state_dict(
            trainer_state,
            strict=False,
            resume_net_only=args.resume_net_only,
            ignore_text_params=args.ignore_text_params,
            core_weights_only=args.core_weights_only,
        )

    vis_transform = build_image_transform(args.img_size, is_train=False)
    visualizer = setup_visualizer(args, trainer, vis_transform)

    if args.report_wandb and dist.is_master():
        wandb.init(
            project="medvtok-stage2" if use_biloader else "medvtok-stage1",
            resume="auto",
            save_code=True,
            id=args.run_id,
            name=args.exp_name,
            notes=args.wandb_notes,
            config=args.state_dict(),
        )

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[train] Exp output directory: {args.output_dir}")
    print(f"[train] Start exp at epoch {start_epoch} iter {start_iter}")
    if use_biloader:
        print(
            "[train] Alternating 2D/3D training: "
            f"{num_2d_batches} 2D batches -> {num_iters_per_epoch} total steps/epoch"
        )
        vol_iter = InfiniteDataLoader(data["train_3d"])
        if "vol_epoch" in ckpt:
            vol_iter.set_epoch(ckpt["vol_epoch"])
    else:
        print(f"[train] 2D-only training: {num_iters_per_epoch} steps/epoch")
        vol_iter = None

    stats = None
    for epoch in range(start_epoch, args.epoch):
        gc.collect()
        print(f"[dataloader] set_epoch({epoch})")
        if use_biloader:
            data["train_2d"].set_epoch(epoch)
            vol_iter.set_epoch(epoch)
        else:
            data["train"].set_epoch(epoch)

        start_iter_ep = start_iter if epoch == start_epoch else 0
        print(f"[train] Start training ({epoch})")
        train_one_epoch = train_one_ep_biloader if use_biloader else train_one_ep_single
        train_kwargs = {
            "args": args,
            "data": data,
            "epoch": epoch,
            "trainer": trainer,
            "start_iter": start_iter_ep,
            "model_scheduler": model_scheduler,
            "disc_scheduler": disc_scheduler,
            "visualizer": visualizer,
        }
        if use_biloader:
            train_kwargs["vol_iter"] = vol_iter
        stats = train_one_epoch(**train_kwargs)

    if dist.is_master() and args.save_final_checkpoint:
        ckpt_path = os.path.join(args.output_dir, "ckpt-last.pth")
        checkpoint = {
            "args": args.state_dict(),
            "epoch": args.epoch,
            "iter": 0,
            "trainer": trainer.state_dict(),
        }
        if use_biloader:
            checkpoint["vol_epoch"] = vol_iter.epoch
        torch.save(checkpoint, ckpt_path)
    dist.barrier()

    total_time = f"{(time.time() - start_time) / 60 / 60:.1f}h"
    if stats is None:
        print(f"[train] No epochs were run. Total Training Time: {total_time}")
    else:
        print(
            f"[train] Total Training Time: {total_time}, "
            f"Lg: {stats['Lnll']:.3f}, Ld: {stats['Ld']:.3f}"
        )
    dist.finalize()

    if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(
        sys.stderr, dist.BackupStreamToFile
    ):
        sys.stdout.close()
        sys.stderr.close()


if __name__ == "__main__":
    main()
