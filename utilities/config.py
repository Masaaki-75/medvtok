import os
import sys
import torch
import random
import numpy as np
from tap import Tap
from typing import List, Optional, Union
from collections import OrderedDict
from pathlib import Path

from utilities import dist
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT


class Args(Tap):
    model: str = 'medvtok-clip-meanpool' 
    exp_name: str = 'medvtok_exp'
    output_dir: str = str(PROJECT_ROOT / "outputs" / "training")
    init_ckpt_path: str = None
    siglip_encoder_init: str = ''  # Stage-1 encoder-only initialization from converted safetensors
    resume_from: str = ''  # if specified, load this checkpoint; if not, load the latest checkpoint in output_dir (if exists)
    resume_net_only: bool = False
    lpips_path: str = str(PACKAGE_ROOT / "external" / "lpips_with_vgg.pth")
    dino_path: str = str(PACKAGE_ROOT / "external" / "dinov2_vits14_pretrain.pth")
    fid_eval_src: str = ''
    fid_eval_dst: str = ''
    vis_img_dir: str = str(PROJECT_ROOT / "data" / "vis_imgs")
    fid_feature_extractor: str = ""
    clip_pretrain_path: str = ''

    # speed-up
    fp16: bool = False  # whether to use FP16
    bf16: bool = True  # whether to use BF16
    tf32: bool = True  # whether to use TensorFloat32
    compile_model: bool = False  # whether to use torch.compile()
    ddp_static: bool = False  # whether to use static graph in DDP
    grad_ckpt: bool = True  # gradient checkpointing
    grad_accu: int = 1  # gradient accumulation
    device: str = 'cpu' # will be set automatically
    dtype: torch.dtype = torch.float32 # will be set automatically

    # data
    train_data: str = None
    val_data: str = None
    train_root: str = None
    val_root: str = None
    feature_root: str = str(PROJECT_ROOT / "outputs" / "alignment")
    dataset_type: str = 'csvimgfeat'
    csv_img_key: str = 'identifier'
    csv_caption_key: str = 'submodality'
    csv_separator: str = ','
    subset_ratio: float = 1.0
    train_fraction_2d: float = 1.0  # per-modality fraction retained from the 2D training CSV
    train_fraction_3d: float = 1.0  # per-modality fraction retained from the 3D training CSV
    subset_seed: int = 42  # fixed seed for deterministic stratified subsampling
    exclude_modalities: Optional[List[str]] = None  # modality/submodality names removed before training
    img_size: int = 256
    resize_ratio: float = 1.125  # only applicable to 'img' dataset_type
    hflip: bool = False
    workers: int = 8  # 0 disables multiprocessing; negative values are normalized to 0
    train_num_samples: int = None
    train_data_upsampling_factors: str = None
    val_num_samples: int = None
    dataset_resampled: bool = False
    use_aug: bool = False
    prob_flip: float = 0.5  # probability for data augmentation: image flipping
    prob_rot: float = 0.2  # probability for data augmentation: image rotation
    prob_grayscale: float = 0.1  # probability for data augmentation: image grayscale
    prob_jitter: float = 0.0
    image_list_as_batch: bool = True  # if True, return images as list for variable-size support
    feature_list_as_batch: bool = True  # if True, return features as list for variable-dimension support (MEPA)
    diversity_sampling: bool = True  # if True, ensure each batch contains samples from all modality groups
    num_appear_each_modality: int = 1  # minimum samples per modality group in each diverse batch
    keep_ratio: bool = True
    mixed_resolution_training: bool = True


    # model
    embed_dim: int = 1152
    num_query: int = 0
    use_clip_pretrain: bool = False
    patch_size: int = 16
    drop_path: float = 0.1
    text_width: int = 768
    text_heads: int = 12
    text_layers: int = 12
    text_vocab_size: int = 49408
    text_context_length: int = 77

    # CLIP/SigLIP
    local_loss: bool = True
    gather_with_grad: bool = True
    pretrained_clip: str = None
    pretrained_clip_text: str = None
    lock_text: bool = False
    lock_text_unlocked_layers: int = 0
    lock_text_freeze_layer_norm: bool = False
    force_custom_text: bool = False
    force_custom_vision: bool = False
    zeroshot_eval_freq: int = 1

    # discriminator
    dino_depth: int = 12
    dino_kernel_size: int = 9
    disc_norm: str = 'gn'  # gn: group norm, bn: batch norm, sbn: sync batch norm, hbn: hybrid sync batch norm
    disc_aug_prob: float = 1.0
    disc_specnorm: bool = False
    step_disc_every: int = 1
    
    # initialization
    vae_init: float = -0.5  # <0: xavier_normal_(gain=abs(init)); >0: trunc_normal_(std=init)
    disc_init: float = -0.5  # <0: xavier_normal_(gain=abs(init)); >0: trunc_normal_(std=init)

    # optimization
    epoch: int = 1  # number of epochs
    local_bs: int = 64  # batch size per device; if this is specified, --global_bs will be ignored
    global_bs: int = 0  # global batch size (exclusive to --local_bs)
    lr: float = 5e-4  # learning rate
    wd: float = 0.02  # weight decay
    disc_lr: float = 2e-5  # disc lr
    disc_wd: float = 0.2
    grad_clip: float = 10  # <=0 for not using grad clip
    ema: float = 0.9999  # ema ratio
    warmup_iter: int = None
    warmup_ep: float = 0.01  # lr warmup: epochs
    disc_start_ep: float = 0.  # start using disc loss for VAE after xxx epochs;
    disc_warmup_ep: float = 0.03  # disc loss warm up epochs;
    schedule: str = 'cos'  # lr schedule type
    lr_start_ratio: float = 0.  # lr warmup: initial lr ratio
    lr_end_ratio: float = 0.1  # lr schedule: final lr ratio
    disc_lr_end_ratio: float = 0.1
    custom_lr_multiplier: float = None
    optimizer: str = 'adamw'
    optim_eps: float = 1e-6
    fuse_opt: bool = False  # whether to use fused optimizer
    optim_beta: str = '0.9_0.95'  # beta1, beta2 of optimizer
    disc_optim_beta: str = '0.5_0.9'  # beta1, beta2 of disc optimizer

    # loss
    contrastive_type: str = 'siglip'
    l1: float = 0.2  # L1 rec loss weight
    l2: float = 1.0  # L2 rec loss weight
    lp: float = 1.0  # lpips loss weight
    lgram: float = 0.0  # multi-layer VGG Gram-matrix loss weight
    lpr: int = 48    # only calculate lpips >= this image resolution
    ld: float = 0.4  # discriminator loss weight; if <0: NO ADAPTIVE WEIGHT
    lshare: float = 1.0  # shared-space contrastive alignment weight
    lcos: float = 0.0  # cosine loss weight
    lexpert: float = 0.0  # expert-space complementary alignment weight
    lkl: float = 1e-6  # KL loss
    lslidr: float = 0.0  # weight for Slice Differential Regularization (SLIDR)
    slice_regularization_type: str = 'slice_differential_regularization_loss'
    slidr_second_order_weight: float = 1.0
    gada: int = 1
    bcr: float = 4.  # balanced Consistency Regularization, used on small dataset with low reso, StyleSwin: 10.0
    bcr_cut: float = 0.2  # cutout ratio (0.5: half the width)
    dcrit: str = 'hg'  # hg hinge, sp softplus, ln linear

    # wandb log
    report_wandb: bool = True
    wandb_notes: str = None
    run_id: str = None

    # debug
    eval_per_epoch: int = 4
    dbg_unused_param: bool = False
    dbg_nan: bool = False  # 'KEVIN_LOCAL' in os.environ
    seed: int = None
    deterministic: bool = False
    same_seed_for_all_ranks: int = 0  # this is only for distributed sampler

    # extra
    use_biomedclip: bool = False
    vision_as_text: bool = False
    ignore_text_params: bool = False  # Not loading the text encoder params, since different stage could use different text encoders 
    ct_bias: int = 1024  # If CT is stored as biased png
    freeze_logit_scale: bool = False
    core_weights_only: bool = False
    lock_visual_proj: bool = False
    drop_alignment: bool = False
    include_3d_projector: bool = False
    include_3d_data: bool = False
    train_data_3d: str = None
    val_data_3d: str = None
    train_root_3d: str = None
    val_root_3d: str = None
    train_num_samples_3d: int = None
    val_num_samples_3d: int = None
    num_batches_per_epoch_3d: int = None
    volume_dtype: str = 'float32'
    local_bs_3d: int = 1  # per-device batch size for 3D volumes; 0 means use local_bs
    ablation_modality: str = None
    image_encoder_name: str = None
    lora_modules: str = 'qv'
    force_sem_only: bool = False
    padding_mode: str = 'replicate'
    mydebug: bool = False
    enable_visualizer: bool = True
    save_final_checkpoint: bool = True

    def seed_everything(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if self.seed is not None:
            if self.deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
            seed = self.seed + dist.get_rank() * 10000
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:  # for random augmentation
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g

    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        for k in self.class_variables.keys():
            if k not in {'device'}:  # these are not serializable
                d[k] = getattr(self, k)
        return d

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            try:
                setattr(self, k, v)
            except Exception as e:
                print(f'k={k}, v={v}')
                raise e

    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')

    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:  # these are not serializable
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'
    
    def get(self, key: str, default=None):
        return getattr(self, key, default)


def init_dist_and_get_args():
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break

    args = Args(explicit_bool=True).parse_args(known_only=True)
    if args.workers < 0:
        args.workers = 0
    # warn args.extra_args
    if len(args.extra_args) > 0:
        print(f'======================================================================================')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================')
        print(f'======================================================================================\n\n')

    # init torch distributed
    os.makedirs(args.output_dir, exist_ok=True)
    dist.init_distributed_mode(local_out_path=args.output_dir, timeout_minutes=30)

    # set env
    args.set_tf32(args.tf32)
    args.seed_everything()
    args.device = dist.get_device()

    # update args
    if args.local_bs == 0:
        args.local_bs = max(1, round(args.global_bs / args.grad_accu / dist.get_world_size()))
    args.global_bs = args.local_bs * dist.get_world_size()
    if args.fp16 or args.bf16:
        args.dtype = torch.float16 if args.fp16 else torch.bfloat16

    return args
