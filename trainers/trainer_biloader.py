import os
import sys
import math
import time
import datetime
import torch
from warnings import warn
from copy import deepcopy
from pprint import pformat
from typing import Callable, Optional, Tuple
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utilities import config, nan, dist
from utilities.diffaug import DiffAug
from utilities.lpips import LPIPS, tensor_list_to_batch
from utilities.misc import unwrap_model
from utilities.loss import (
    hinge_loss,
    linear_loss,
    softplus_loss,
    cosinesim_loss,
    ClipLoss,
    SigLipLoss,
    compute_expert_alignment_loss,
    slice_differential_regularization_loss,
    total_variation_loss,
    laplacian_loss,
)
from trainers.scheduler import LRScheduler
from trainers.optimizer import AmpOptimizer
from trainers.visualizer import Visualizer
from trainers.logger import MetricLogger, wandb_log, SmoothedValue, ProfileLogger


def get_type(x):
    return x.dtype if hasattr(x, 'dtype') else type(x)


class InfiniteDataLoader:
    """Wraps a DataInfo so its dataloader cycles forever.

    When the underlying iterator is exhausted, increments the epoch
    (for sampler re-shuffling) and restarts.
    """
    def __init__(self, data_info):
        self.data_info = data_info
        self.iterator = iter(data_info.dataloader)
        self.epoch = 0

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.data_info.set_epoch(self.epoch)
            dist.barrier()
            self.iterator = iter(self.data_info.dataloader)
            return next(self.iterator)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.data_info.set_epoch(epoch)
        self.iterator = iter(self.data_info.dataloader)


class BiLoaderTrainer(object):
    def __init__(
        self,
        args: config.Args,
        model: DDP,
        disc: DDP,
        model_optim: AmpOptimizer,
        disc_optim: AmpOptimizer,
        lpips_loss: LPIPS,
        clip_loss: ClipLoss = None,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.disc = disc
        self.model_optim = model_optim
        self.disc_optim = disc_optim
        self.mydebug = getattr(args, 'mydebug', False)
        self.force_sem_only = getattr(args, 'force_sem_only', False)

        self.dcrit = args.dcrit
        self.d_criterion = {
            'hg': hinge_loss, 'hinge': hinge_loss,
            'sp': softplus_loss, 'softplus': softplus_loss,
            'ln': linear_loss, 'lin': linear_loss, 'linear': linear_loss
        }[self.dcrit]
        self.daug = DiffAug(prob=args.disc_aug_prob, cutout=0.2)

        self.wei_expert = args.lexpert
        self.wei_l1 = args.l1
        self.wei_l2 = args.l2
        self.wei_lpips = args.lp
        self.wei_disc = args.ld
        self.wei_kl = args.lkl
        self.wei_slidr = getattr(args, 'lslidr', 0)
        self.slidr_second_order_weight = getattr(args, 'slidr_second_order_weight', 1.0)
        if self.slidr_second_order_weight < 0:
            raise ValueError(
                "slidr_second_order_weight must be non-negative, "
                f"got {self.slidr_second_order_weight}."
            )
        requested_slice_regularization_type = getattr(
            args,
            'slice_regularization_type',
            'slice_differential_regularization_loss',
        ).strip().lower()
        slice_regularization_type_aliases = {
            'slidr': 'slice_differential_regularization_loss',
            'slice_differential_regularization': 'slice_differential_regularization_loss',
            'slice_differential_regularization_loss': 'slice_differential_regularization_loss',
            'tv': 'total_variation_loss',
            'total_variation': 'total_variation_loss',
            'total_variation_loss': 'total_variation_loss',
            'laplacian': 'laplacian_loss',
            'laplacian_loss': 'laplacian_loss',
        }
        if requested_slice_regularization_type not in slice_regularization_type_aliases:
            valid_slice_regularization_types = ', '.join(sorted(slice_regularization_type_aliases))
            raise ValueError(
                f"Unknown slice_regularization_type={requested_slice_regularization_type!r}. "
                f"Expected one of: {valid_slice_regularization_types}."
            )
        self.slice_regularization_type = slice_regularization_type_aliases[
            requested_slice_regularization_type
        ]

        self.lpips_loss = lpips_loss
        self.lp_reso = args.lpr
        self.adapt_wei_disc = args.ld > 0
        self.adapt_type = args.gada

        self.bcr = args.bcr
        if self.bcr > 0:
            self.bcr_strong_aug = DiffAug(prob=1, cutout=args.bcr_cut)

        self.clip_loss = clip_loss
        self.wei_share = args.lshare
        self.grad_ckpt = args.grad_ckpt

        self.dbg_nan = args.dbg_nan
        self.dbg_unused = args.dbg_unused_param
        if self.dbg_nan:
            print(f'[{type(self).__name__}] dbg_nan mode is ON.')
            nan.debug_nan_hook(model)
            if not self.force_sem_only:
                nan.debug_nan_hook(disc)

    def train_step(
        self,
        img,
        img_feat,
        modality_indices,
        global_iter: int,
        stepping: bool,
        metric_logger: MetricLogger,
        warmup_disc_schedule: float,
        fade_blur_schedule: float,
        report_wandb: bool = False,
        is_3d_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor], Optional[float]]:

        if warmup_disc_schedule < 1e-6:
            warmup_disc_schedule = 0
        if fade_blur_schedule < 1e-6:
            fade_blur_schedule = 0
        loggable = global_iter < 2 or global_iter % 50 == 0

        # vae loss
        vae_img = img

        with self.model_optim.amp_ctx:
            output = self.model(img, img_feat, modality_indices, training=True, return_batched_tensor=False)
            rec_img = output['img_rec']
            Lkl = output.get('kl_loss', torch.tensor(0.0)) if self.wei_kl > 0 else torch.tensor(0.0)
            Lslidr = torch.tensor(0.0)

        output_is_list = isinstance(rec_img, (list, tuple)) or isinstance(vae_img, (list, tuple))
        batch_size = len(vae_img) if isinstance(vae_img, (list, tuple)) else vae_img.shape[0]

        if self.force_sem_only:
            L1 = L2 = Lrec = Lslidr = torch.tensor(0.0)
        else:
            if output_is_list:
                L1 = sum([F.l1_loss(rec_img[i], vae_img[i]) for i in range(batch_size)]) / float(batch_size)
                L2 = sum([F.mse_loss(rec_img[i], vae_img[i]) for i in range(batch_size)]) / float(batch_size)
            else:
                L1 = F.l1_loss(rec_img, vae_img)
                L2 = F.mse_loss(rec_img, vae_img)

            if self.wei_slidr > 0 and is_3d_step:
                slidr_inputs, slidr_recons = vae_img, rec_img
                if isinstance(slidr_inputs, (list, tuple)) and torch.is_tensor(slidr_recons):
                    slidr_recons = [slidr_recons[i] for i in range(slidr_recons.shape[0])]
                elif torch.is_tensor(slidr_inputs) and isinstance(slidr_recons, (list, tuple)):
                    slidr_inputs = [slidr_inputs[i] for i in range(slidr_inputs.shape[0])]
                
                if self.slice_regularization_type == 'slice_differential_regularization_loss':
                    Lslidr = slice_differential_regularization_loss(
                        slidr_inputs,
                        slidr_recons,
                        second_order_weight=self.slidr_second_order_weight,
                    )
                elif self.slice_regularization_type == 'total_variation_loss':
                    Lslidr = total_variation_loss(slidr_recons)
                elif self.slice_regularization_type == 'laplacian_loss':
                    Lslidr = laplacian_loss(slidr_recons)
                else:
                    raise RuntimeError(
                        f"Unhandled slice_regularization_type={self.slice_regularization_type!r}."
                    )
                if self.mydebug:
                    print(f"INPUT SHAPES: {[tuple(x.shape) for x in slidr_inputs]}")
                    print(f"RECON SHAPES: {[tuple(x.shape) for x in slidr_recons]}")
                    print(
                        f"LOSS (name={self.slice_regularization_type}, "
                        f"type={type(Lslidr)}): {Lslidr}"
                    )
                
            Lrec = L1 * self.wei_l1 + L2 * self.wei_l2 + Lslidr * self.wei_slidr

        Lpip = torch.tensor(0.0)
        if self.wei_lpips > 0 and (not self.force_sem_only):
            #self.lpips_loss.forward
            Lpip = self.lpips_loss(vae_img, rec_img)  # LPIPS.forward also supports list inputs.

        Lnll = Lrec + self.wei_lpips * Lpip

        # Shared-space contrastive alignment loss.
        Lshare = torch.tensor(0.0)
        if self.wei_share > 0:
            with self.model_optim.amp_ctx:
                shared_losses = self.clip_loss(
                    image_features=output['clip_core_features'],
                    text_features=output['clip_image_features'],
                    logit_scale=output['logit_scale'],
                    logit_bias=output['logit_bias'],
                    output_dict=True
                )
                Lshare = sum(shared_losses.values())

        # Expert-space complementary alignment loss.
        Lexpert = torch.tensor(0.0)
        if self.wei_expert > 0 and "expert_aligned_features" in output and "expert_ddp_anchor" in output:
            with self.model_optim.amp_ctx:
                Lexpert = compute_expert_alignment_loss(output["expert_aligned_features"], img_feat, output["expert_ddp_anchor"])

        # The discriminator and training tricks below are hard to use with varying image sizes.
        # Resize the inputs and reconstructions here so they can be stacked into batch tensors.
        vae_img = tensor_list_to_batch(vae_img, 256, fix_depth=10)
        rec_img = tensor_list_to_batch(rec_img, 256, fix_depth=10)
        
        if warmup_disc_schedule > 0 and (not self.force_sem_only):
            for d in self.disc.parameters():
                d.requires_grad = False
            self.disc.eval()
            with self.disc_optim.amp_ctx:
                Lg = -self.disc(self.daug.aug(rec_img, fade_blur_schedule), grad_ckpt=False).mean()
            self.disc.train()

            wei_g = warmup_disc_schedule * self.wei_disc
            if self.adapt_wei_disc:
                last_layer = unwrap_model(self.model).core.decoder.get_last_param()
                w = (torch.autograd.grad(Lnll, last_layer, retain_graph=True)[0].data.norm() /
                     torch.autograd.grad(Lg, last_layer, retain_graph=True)[0].data.norm().add_(1e-6))
                if self.adapt_type % 10 == 0:
                    w.clamp_(0.0, 1e4)
                elif self.adapt_type % 10 == 1:
                    w.clamp_(0.015, 1e4)
                elif self.adapt_type % 10 == 2:
                    w.clamp_(0.1, 10)
                    w = min(max(w, 0.1), 10)
                elif self.adapt_type % 10 == 3:
                    w.clamp_(0.0, 1e4).sqrt_()
                elif self.adapt_type % 10 == 4:
                    w.clamp_(0.015, 1.5)

                wei_g = wei_g * w

            Lv = Lnll + wei_g * Lg + self.wei_share * Lshare + self.wei_expert * Lexpert + self.wei_kl * Lkl
        else:
            Lv = Lnll + self.wei_share * Lshare + self.wei_expert * Lexpert + self.wei_kl * Lkl
            Lg = wei_g = 0.

        grad_norm_g, scale_log2_g = self.model_optim.backward_clip_step(stepping=stepping, loss=Lv)

        # [discriminator loss]
        if warmup_disc_schedule > 0 and (not self.force_sem_only):
            rec_img = rec_img.data
            for d in self.disc.parameters():
                d.requires_grad = True
            with self.disc_optim.amp_ctx:
                logits_real = self.disc(self.daug.aug(vae_img, fade_blur_schedule), grad_ckpt=self.grad_ckpt).float()
                logits_fake = self.disc(self.daug.aug(rec_img, fade_blur_schedule), grad_ckpt=self.grad_ckpt).float()
            acc_real = (logits_real.data > 0).float().mean().mul_(100)
            acc_fake = (logits_fake.data < 0).float().mean().mul_(100)
            Ld = self.d_criterion(logits_real) + self.d_criterion(-logits_fake)

            if self.bcr:
                with self.disc_optim.amp_ctx:
                    Lbcr = (
                        F.mse_loss(self.disc(self.bcr_strong_aug.aug(vae_img, 0.0), grad_ckpt=self.grad_ckpt).float(), logits_real) +
                        F.mse_loss(self.disc(self.bcr_strong_aug.aug(rec_img, 0.0), grad_ckpt=self.grad_ckpt).float(), logits_fake)
                    ).mul_(self.bcr)
                Ld += Lbcr
            else:
                Lbcr = 0.
            grad_norm_d, scale_log2_d = self.disc_optim.backward_clip_step(stepping=stepping, loss=Ld)
            Ld = Ld.data.clone()
        else:
            Ld = Lbcr = acc_real = acc_fake = grad_norm_d = 0.
            scale_log2_d = None

        if not math.isfinite(Lnll + Ld + wei_g):
            for n, v in zip(['Lrec', 'Lpip', 'Ld', 'wei_g'], [Lrec, Lpip, Ld, wei_g]):
                if not math.isfinite(v):
                    print(f'[rk{dist.get_rank():02d}] {n} is {v}, stopping training!', force=True, flush=True)
            sys.exit(666)

        with torch.no_grad():
            logit_scale = unwrap_model(self.model).core.logit_scale
            if hasattr(logit_scale, 'clamp_'):
                logit_scale.clamp_(0, math.log(100))
            logit_bias = unwrap_model(self.model).core.logit_bias
            if hasattr(logit_bias, 'clamp_'):
                logit_bias.clamp_(-30, 30)

        # [zero_grad]
        if stepping:
            if self.dbg_nan:
                nan.debug_nan_grad(self.model)
                if (not self.force_sem_only):
                    nan.debug_nan_grad(self.disc)
                nan.debug_nan_param(self.model)
                if (not self.force_sem_only):
                    nan.debug_nan_param(self.disc)
            
            if self.dbg_unused:
                ls = []
                for n, p in self.model.named_parameters():
                    # or tuple(p.grad.shape) == (512, 512, 1, 1):
                    if p.grad is None:
                        ls.append(n)

                if (not self.force_sem_only):
                    for n, p in self.disc.named_parameters():
                        if p.grad is None:  # or tuple(p.grad.shape) == (512, 512, 1, 1):
                            ls.append(n)
                
                if len(ls):
                    print(f'[{type(self).__name__}.train_step] unused param: {ls}', flush=True, file=sys.stderr)

            self.model_optim.optimizer.zero_grad(set_to_none=True)
            if (not self.force_sem_only):
                self.disc_optim.optimizer.zero_grad(set_to_none=True)

        # logging
        if loggable:
            metric_logger.update(
                L1=L1, Lnll=Lnll, Ld=Ld, Lkl=Lkl, Lslidr=Lslidr,
                Lshare=Lshare, Lexpert=Lexpert,
                acc_real=acc_real, acc_fake=acc_fake,
                gnm=grad_norm_g, dnm=grad_norm_d,
            )

        if report_wandb:
            log_freq = 50
            wandb_log({'L1': L1}, step=global_iter, log_freq=log_freq)
            # wandb_log({'Lrec': Lrec}, step=global_iter, log_freq=log_freq)
            wandb_log({'Lnll': Lnll}, step=global_iter, log_freq=log_freq)
            wandb_log({'Lshare': Lshare}, step=global_iter, log_freq=log_freq)
            if self.wei_lpips > 0:
                wandb_log({'Lpip': Lpip}, step=global_iter, log_freq=log_freq)
            if self.wei_expert > 0:
                wandb_log({'Lexpert': Lexpert}, step=global_iter, log_freq=log_freq)
            if self.wei_kl > 0:
                wandb_log({'Lkl': Lkl}, step=global_iter, log_freq=log_freq)
            if self.wei_slidr > 0:
                wandb_log({'Lslidr': Lslidr}, step=global_iter, log_freq=log_freq)
            if warmup_disc_schedule > 0 and (not self.force_sem_only):
                wandb_log({'Ldisc': Ld - Lbcr}, step=global_iter, log_freq=log_freq)
                wandb_log({'Lbcr': Lbcr}, step=global_iter, log_freq=log_freq)
                wandb_log({'Lg': Lg}, step=global_iter, log_freq=log_freq)
                wandb_log({'Wei_g': wei_g}, step=global_iter, log_freq=log_freq)
                wandb_log({'Disc_accu_real': acc_real}, step=global_iter, log_freq=log_freq)
                wandb_log({'Disc_accu_fake': acc_fake}, step=global_iter, log_freq=log_freq)
                wandb_log({'Disc_accu_avg': (acc_real + acc_fake) * 0.5}, step=global_iter, log_freq=log_freq)
            if scale_log2_g is not None:
                wandb_log({'Scaler_g': scale_log2_g}, step=global_iter, log_freq=log_freq)
            if scale_log2_d is not None:
                wandb_log({'Scaler_d': scale_log2_d}, step=global_iter, log_freq=log_freq)

            wandb_log({'Gradnorm_g': grad_norm_g}, step=global_iter, log_freq=log_freq)
            wandb_log({'Logit_scale': output['logit_scale']}, step=global_iter, log_freq=log_freq)

            if (not self.force_sem_only):
                wandb_log({'Gradnorm_d': grad_norm_d}, step=global_iter, log_freq=log_freq)
                wandb_log({'Disc_warmup_schedule': warmup_disc_schedule}, step=global_iter, log_freq=log_freq)
                wandb_log({'Disc_fade_blur_schedule': fade_blur_schedule}, step=global_iter, log_freq=log_freq)
        return

    def __repr__(self):
        return (
            f'\n'
            f'[{type(self).__name__}.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[{type(self).__name__}.structure]: {super(BiLoaderTrainer, self).__repr__().replace(BiLoaderTrainer.__name__, "")}'
        )

    def get_config(self):
        return {
            'dcrit': self.dcrit,
            'wei_l1': self.wei_l1,
            'wei_l2': self.wei_l2,
            'wei_lpips': self.wei_lpips,
            'wei_disc': self.wei_disc,
            'wei_share': self.wei_share,
            'bcr': self.bcr,
            'wei_expert': self.wei_expert,
            'wei_kl': self.wei_kl,
        }

    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('model', 'disc', 'model_optim', 'disc_optim'):
            m = getattr(self, k)
            if m is None:
                continue
            if isinstance(m, DDP):
                m = unwrap_model(m)
            if hasattr(m, '_orig_mod'):
                m = m._orig_mod
            
            if hasattr(m, 'state_dict'):
                state[k] = m.state_dict()
        return state
    

    def load_state_dict(self, state, strict=True, resume_net_only=False, ignore_text_params=False, core_weights_only=False):
        keys = ('model', 'disc') if resume_net_only else ('model', 'disc', 'model_optim', 'disc_optim')
        for k in keys:
            m = getattr(self, k)
            if m is not None:
                if isinstance(m, DDP):
                    m = unwrap_model(m)
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                
                if k in state:
                    if k == 'model' and ignore_text_params:  
                        print(f'[{type(self).__name__}.load_state_dict] ignore_text_params @ {ignore_text_params}')
                        # Not loading the text encoder params, since different stage could use different text encoders 
                        state_dict = {k:v for k,v in state[k].items() if not k.startswith('text')}
                    else:
                        state_dict = state[k]

                    print(f'[{type(self).__name__}.load_state_dict] core_weights_only @ {core_weights_only}')
                    if k == 'model' and core_weights_only:
                        ret = m.core.load_state_dict(state_dict, strict=True)
                    else:
                        ret = m.load_state_dict(state_dict, strict=strict)

                    if ret is not None:
                        missing, unexpected = ret
                        print(f'[{type(self).__name__}.load_state_dict] {k} MISSING: {missing}')
                        print(f'[{type(self).__name__}.load_state_dict] {k} UNEXPECTED: {unexpected}')
                        print(f'[{type(self).__name__}.load_state_dict] SUCESSFULLY loaded state_dict!')
                else:
                    print(f'[{type(self).__name__}.load_state_dict] Key {k} is NOT FOUND in state_dict.')
        config: dict = state.pop('config', None)

        if config is not None:
            # Migrate historical checkpoint fields to the MERA loss names.
            config = dict(config)
            for legacy_key, current_key in (
                ('wei_clip', 'wei_share'),
                ('wei_cons', 'wei_expert'),
            ):
                if legacy_key in config:
                    config.setdefault(current_key, config.pop(legacy_key))
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[{type(self).__name__}.load_state_dict] Config mismatch: this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)


def train_one_ep(
    args,
    data: dict,
    epoch: int,
    trainer: BiLoaderTrainer,
    start_iter: int,
    model_scheduler: LRScheduler,
    disc_scheduler: LRScheduler,
    visualizer: Visualizer,
    vol_iter: InfiniteDataLoader,
):
    """Training loop that alternates between 2D image and 3D volume batches.

    Even cur_iter → 2D batch from data['train_2d'].
    Odd  cur_iter → 3D batch from vol_iter (infinite cycling).
    Total steps per epoch = 2 * num_2d_batches.
    """
    dataloader_2d = data['train_2d'].dataloader
    num_2d_batches = data['train_2d'].num_batches
    num_iters = 2 * num_2d_batches
    force_sem_only = getattr(args, 'force_sem_only', False)

    metric_logger = MetricLogger(cur_epoch=epoch, total_epoch=args.epoch, delimiter='  ')
    [metric_logger.add_meter(x, SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ('glr', 'dlr')]
    [metric_logger.add_meter(x, SmoothedValue(window_size=1, fmt='{median:.2f}')) for x in ('gnm', 'dnm')]
    [metric_logger.add_meter(x, SmoothedValue(fmt='{median:.3f}')) for x in ('L1', 'Lnll', 'Ld', 'Lshare', 'Wg')]

    disc_start_iter = args.disc_start_ep * num_iters
    disc_warmup_iter = args.disc_warmup_ep * num_iters

    profile_log_freq = 1000
    profile_logger = ProfileLogger(args, profile_log_freq)
    eval_interval = int(num_iters // args.eval_per_epoch) if args.eval_per_epoch > 0 else num_iters
    if eval_interval == 0:
        eval_interval = num_iters

    print(f"[train_one_ep] num_2d_batches={num_2d_batches}, num_iters={num_iters}, "
          f"eval_per_epoch={args.eval_per_epoch}, eval_interval={eval_interval}")

    iter_2d = iter(dataloader_2d)

    if start_iter > 0:
        num_2d_to_skip = (start_iter + 1) // 2
        print(f"[train_one_ep] Resuming: skipping {num_2d_to_skip} 2D batches")
        for _ in range(num_2d_to_skip):
            try:
                next(iter_2d)
            except StopIteration:
                break

    log_msg = metric_logger.delimiter.join([
        '[Ep]: [{cur_epoch}/{total_epoch}]',
        '[{cur_iter}/{num_iters}]',
        '[{src}]',
        '{meters}',
        'eta: {eta}',
        'iter_time: {iter_time}',
        'data_time: {data_time}',
    ])
    print_freq = 100
    time_stamp = time.time()

    for cur_iter in range(start_iter, num_iters):
        is_3d_step = (cur_iter % 2 == 1)

        if is_3d_step:
            sample = next(vol_iter)
        else:
            try:
                sample = next(iter_2d)
            except StopIteration:
                break

        metric_logger.data_time.update(time.time() - time_stamp)

        profile_logger.log(cur_iter)
        global_iter = epoch * num_iters + cur_iter
        disc_global_iter = global_iter - disc_start_iter

        if (cur_iter + 1) % eval_interval == 0 or cur_iter == 0:
            if dist.is_master():
                vis_path = os.path.join(args.output_dir, f'img_{global_iter}.png')
                visualizer.vis(epoch, report_wandb=args.report_wandb, png_path=vis_path)

            if dist.is_master() and cur_iter > 0 and 'debug' not in args.exp_name.lower():
                ckpt_path = os.path.join(args.output_dir, f'ep{epoch}-it{cur_iter}.pth')
                torch.save({
                    'args': args.state_dict(), 'epoch': epoch, 'iter': cur_iter,
                    'trainer': trainer.state_dict(),
                    'vol_epoch': vol_iter.epoch,
                }, ckpt_path)

            dist.barrier()

        imgs, img_feats, modality_indices = sample
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.to(args.device, non_blocking=True)
        else:
            imgs = [img.to(args.device, non_blocking=True) for img in imgs]
        if isinstance(img_feats, torch.Tensor):
            img_feats = img_feats.to(args.device, non_blocking=True)
        else:
            img_feats = [f.to(args.device, non_blocking=True) for f in img_feats]
        modality_indices = modality_indices.to(args.device, non_blocking=True)

        main_lr_stats = model_scheduler.step(global_iter)
        disc_lr_stats = disc_scheduler.step(disc_global_iter) if (disc_global_iter >= 0 and hasattr(disc_scheduler, 'step')) else [0]
        main_lr_stats = list(set(main_lr_stats))
        disc_lr_stats = list(set(disc_lr_stats))

        stepping = (global_iter + 1) % args.grad_accu == 0
        if force_sem_only:
            warmup_disc_schedule = 0
            fade_blur_schedule = 1
        else:
            warmup_disc_schedule = 0 if disc_global_iter < 0 else min(1.0, disc_global_iter / disc_warmup_iter)
            fade_blur_schedule = 0 if disc_global_iter < 0 else min(1.0, disc_global_iter / (disc_warmup_iter * 2))
            fade_blur_schedule = 1 - fade_blur_schedule

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
            is_3d_step=is_3d_step,
        )

        metric_logger.update(glr=max(main_lr_stats))
        metric_logger.update(dlr=max(disc_lr_stats))

        if args.report_wandb:
            for i, lr in enumerate(main_lr_stats):
                wandb_log({f'Param_group_{i}_lr': lr}, step=global_iter, log_freq=200)
            for i, lr in enumerate(disc_lr_stats):
                wandb_log({f'Disc_param_group_{i}_lr': lr}, step=global_iter, log_freq=200)

        metric_logger.iter_time.update(time.time() - time_stamp)
        if cur_iter % print_freq <= 1:
            eta_iters = (args.epoch - epoch) * num_iters - cur_iter
            eta_seconds = metric_logger.iter_time.global_avg * eta_iters
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            src = '3D' if is_3d_step else '2D'
            print(log_msg.format(
                cur_epoch=epoch,
                total_epoch=args.epoch,
                cur_iter=cur_iter,
                num_iters=num_iters,
                src=src,
                meters=str(metric_logger),
                eta=eta_string,
                iter_time=str(metric_logger.iter_time),
                data_time=str(metric_logger.data_time),
            ), flush=True)
        time_stamp = time.time()

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats
