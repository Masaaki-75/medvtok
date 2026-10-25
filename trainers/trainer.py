"""Single-loader training entry points used by MedVTok stage 1.

Stage 1 is strictly 2D.  The actual optimization step is shared with the
maintained trainer implementation, while the epoch loop below preserves the
single-loader scheduling and checkpoint semantics of the original stage-1
training code.
"""

import os

import torch

from trainers.logger import MetricLogger, ProfileLogger, SmoothedValue, wandb_log
from trainers.scheduler import LRScheduler
from trainers.trainer_biloader import BiLoaderTrainer
from trainers.visualizer import Visualizer
from utilities import dist


class Trainer(BiLoaderTrainer):
    """The MedVTok trainer restricted to the 2D stage-1 path."""

    def train_step(self, *args, **kwargs):
        # Stage 1 never consumes volumes and therefore never applies SLIDR.
        kwargs["is_3d_step"] = False
        return super().train_step(*args, **kwargs)


def train_one_ep(
    args,
    data: dict,
    epoch: int,
    trainer: Trainer,
    start_iter: int,
    model_scheduler: LRScheduler,
    disc_scheduler: LRScheduler,
    visualizer: Visualizer,
):
    """Train one epoch from the single 2D loader used by stage 1."""
    print("[train_one_ep] fetching 2D data loader")
    dataloader = data["train"].dataloader
    num_iters = data["train"].num_batches
    force_sem_only = getattr(args, "force_sem_only", False)

    metric_logger = MetricLogger(cur_epoch=epoch, total_epoch=args.epoch, delimiter="  ")
    for name in ("glr", "dlr"):
        metric_logger.add_meter(name, SmoothedValue(window_size=1, fmt="{value:.2g}"))
    for name in ("gnm", "dnm"):
        metric_logger.add_meter(name, SmoothedValue(window_size=1, fmt="{median:.2f}"))
    for name in ("L1", "Lnll", "Ld", "Lshare", "Wg"):
        metric_logger.add_meter(name, SmoothedValue(fmt="{median:.3f}"))

    disc_start_iter = args.disc_start_ep * num_iters
    disc_warmup_iter = args.disc_warmup_ep * num_iters

    profile_logger = ProfileLogger(args, log_freq=1000)
    eval_interval = num_iters // args.eval_per_epoch if args.eval_per_epoch > 0 else num_iters
    if eval_interval == 0:
        eval_interval = num_iters
    print(
        f"[train_one_ep] num_iters={num_iters}, "
        f"eval_per_epoch={args.eval_per_epoch}, eval_interval={eval_interval}"
    )

    for cur_iter, sample in metric_logger.monitor_enumerate(
        dataloader, start_iter, num_iters, print_freq=100
    ):
        profile_logger.log(cur_iter)

        global_iter = epoch * num_iters + cur_iter
        disc_global_iter = global_iter - disc_start_iter

        if (cur_iter + 1) % eval_interval == 0 or cur_iter == 0:
            if dist.is_master():
                vis_path = os.path.join(args.output_dir, f"img_{global_iter}.png")
                visualizer.vis(
                    epoch,
                    report_wandb=args.report_wandb,
                    png_path=vis_path,
                )

            if dist.is_master() and cur_iter > 0 and "debug" not in args.exp_name.lower():
                ckpt_path = os.path.join(args.output_dir, f"ep{epoch}-it{cur_iter}.pth")
                torch.save(
                    {
                        "args": args.state_dict(),
                        "epoch": epoch,
                        "iter": cur_iter,
                        "trainer": trainer.state_dict(),
                    },
                    ckpt_path,
                )

            dist.barrier()

        imgs, img_feats, modality_indices = sample
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.to(args.device, non_blocking=True)
        else:
            imgs = [img.to(args.device, non_blocking=True) for img in imgs]

        if img_feats is None:
            pass
        elif isinstance(img_feats, torch.Tensor):
            img_feats = img_feats.to(args.device, non_blocking=True)
        else:
            img_feats = [feature.to(args.device, non_blocking=True) for feature in img_feats]

        modality_indices = modality_indices.to(args.device, non_blocking=True)

        main_lr_stats = list(set(model_scheduler.step(global_iter)))
        disc_lr_stats = (
            list(set(disc_scheduler.step(disc_global_iter)))
            if disc_global_iter >= 0 and hasattr(disc_scheduler, "step")
            else [0]
        )

        stepping = (global_iter + 1) % args.grad_accu == 0
        if force_sem_only:
            warmup_disc_schedule = 0
            fade_blur_schedule = 1
        elif disc_global_iter < 0:
            warmup_disc_schedule = 0
            fade_blur_schedule = 1
        else:
            warmup_disc_schedule = min(1.0, disc_global_iter / disc_warmup_iter)
            fade_blur_schedule = 1 - min(
                1.0, disc_global_iter / (disc_warmup_iter * 2)
            )

        trainer.train_step(
            img=imgs,
            img_feat=img_feats,
            modality_indices=modality_indices,
            global_iter=global_iter,
            stepping=stepping,
            metric_logger=metric_logger,
            warmup_disc_schedule=warmup_disc_schedule,
            fade_blur_schedule=fade_blur_schedule,
            report_wandb=args.report_wandb,
        )

        metric_logger.update(glr=max(main_lr_stats))
        metric_logger.update(dlr=max(disc_lr_stats))

        if args.report_wandb:
            for index, lr in enumerate(main_lr_stats):
                wandb_log(
                    {f"Param_group_{index}_lr": lr},
                    step=global_iter,
                    log_freq=200,
                )
            for index, lr in enumerate(disc_lr_stats):
                wandb_log(
                    {f"Disc_param_group_{index}_lr": lr},
                    step=global_iter,
                    log_freq=200,
                )

    metric_logger.synchronize_between_processes()
    return {name: meter.global_avg for name, meter in metric_logger.meters.items()}
