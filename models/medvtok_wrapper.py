from typing import Optional, Union
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModel

from models.medvtok import build_model
from datasets.csv_imagefeature_dataset import DERM_MODALITIES, CFP_MODALITIES, PATH_MODALITIES, SURG_MODALITIES, CT3D_MODALITIES, MR3D_MODALITIES 
from encoders.conch.builder import get_base_dim as get_base_dim_conch
from encoders.dermlip.builder import get_base_dim as get_base_dim_dermlip
from encoders.hlip.builder import get_base_dim as get_base_dim_hlip
from encoders.medimageinsight.builder import get_base_dim as get_base_dim_mii
from encoders.peskavlp.builder import get_base_dim as get_base_dim_peskavlp
from encoders.retclip.builder import get_base_dim as get_base_dim_retclip
from encoders.spectre.builder import get_base_dim as get_base_dim_spectre



def build_medvtok_wrapper(args):
    arch = args.model
    device = getattr(args, 'device', 'cpu')
    pretrained_path = getattr(args, 'init_ckpt_path', None)
    include_3d_projector = getattr(args, 'include_3d_projector', False)
    ablation_modality = getattr(args, 'ablation_modality', None)
    expert_alignment_weight = getattr(args, 'lexpert', 0)
    use_vision_encoder = getattr(args, 'lshare', 0) > 0
    image_encoder_name = getattr(args, 'image_encoder_name', None)
    if include_3d_projector:
        include_3d_data = getattr(args, 'include_3d_data', False)
        assert include_3d_data, f"Given include_3d_projector=True, we expect include_3d_data=True, got {include_3d_data}"
    
    force_sem_only = getattr(args, 'force_sem_only', False)
    padding_mode = args.padding_mode
    lora_target_modules = []
    for lora_module in ['q', 'v', 'k', 'o']:
        if lora_module in args.lora_modules:
            lora_target_modules.append(f'{lora_module}_proj')

    wrapper_kwargs = dict(
        arch=arch,
        pretrained_path=pretrained_path,
        include_3d_projector=include_3d_projector,
        ablation_modality=ablation_modality,
        expert_alignment_weight=expert_alignment_weight,
        use_vision_encoder=use_vision_encoder,
        image_encoder_name=image_encoder_name,
        force_sem_only=force_sem_only,
        lora_target_modules=lora_target_modules,
        padding_mode=padding_mode,
    )

    model = TokenizerWrapper(**wrapper_kwargs)

    return model.to(device)


class TokenizerWrapper(nn.Module):
    def __init__(
        self, 
        arch='clip', 
        pretrained_path=None, 
        include_3d_projector=False, 
        ablation_modality=None, 
        expert_alignment_weight=0, 
        use_kl=True,
        use_vision_encoder=True,
        image_encoder_name=None,
        force_sem_only=False,
        padding_mode='zero',
        **kwargs,
    ):
        """
        Align tokenizer representations with medical experts in shared and
        expert spaces.
        """
        super().__init__()
        self.use_vision_encoder = use_vision_encoder
        self.core = build_model(
            arch, device='cpu', 
            pretrained_path=pretrained_path, 
            force_sem_only=force_sem_only, 
            padding_mode=padding_mode,
        )
        self.use_kl = use_kl
        self.dim = self.core.dim
        
        self.projectors = None
        self.expert_projectors = None
        self.domain_input_dims = []
        print(f"[WRAPPER] 📖📖📖 Using vision encoder? {use_vision_encoder}.")
        print(f"[WRAPPER] 📖📖📖 Padding mode? {padding_mode}.")
        print(f"[WRAPPER] 📖📖📖 Include 3d projectors? {include_3d_projector}")

        if 'noalign' not in arch and use_vision_encoder:
            self.dim_derm = get_base_dim_dermlip()['raw_dim'] if ablation_modality not in DERM_MODALITIES else get_base_dim_mii()['raw_dim']
            self.dim_cfp = get_base_dim_retclip()['raw_dim'] if ablation_modality not in CFP_MODALITIES else get_base_dim_mii()['raw_dim']
            self.dim_patho = get_base_dim_conch()['raw_dim'] if ablation_modality not in PATH_MODALITIES else get_base_dim_mii()['raw_dim']
            self.dim_surg = get_base_dim_peskavlp()['raw_dim'] if ablation_modality not in SURG_MODALITIES else get_base_dim_mii()['raw_dim']
            
            self.dim_ct = get_base_dim_mii()['raw_dim']
            self.dim_mr = get_base_dim_mii()['raw_dim']
            self.dim_us = get_base_dim_mii()['raw_dim']
            self.dim_xray = get_base_dim_mii()['raw_dim']
            self.dim_other = get_base_dim_mii()['raw_dim']
            self.dim_3dct = get_base_dim_spectre()['raw_dim'] if ablation_modality not in CT3D_MODALITIES else get_base_dim_mii()['raw_dim']
            self.dim_3dmr = get_base_dim_hlip()['raw_dim'] if ablation_modality not in MR3D_MODALITIES else get_base_dim_mii()['raw_dim']

            if image_encoder_name == 'siglip2-so400m-patch16-384':
                self.dim_derm = self.dim_cfp = self.dim_patho = self.dim_surg = 1152
                self.dim_ct = self.dim_mr = self.dim_us = self.dim_xray = self.dim_other = 1152
                self.dim_3dct = self.dim_3dmr = 1152

            # One option is to map tokenizer features into each expert's feature space.
            # However, the expert spaces differ, so CLIP/SigLIP losses cannot compare
            # all features in one shared space. L1 losses would allow distillation,
            # but would not use negative samples to learn well-separated features.
            # 
            # Instead, we project each expert's features into the tokenizer's space.
            # This gives CLIP/SigLIP losses a shared space for comparing features.
            # These projections may lose useful information from the expert features.
            # Simple heads, such as a single nn.Linear layer, may help limit this loss.
            # Warming up the heads to align with MedVTok first is another future option.

            self.domain_input_dims = (
                self.dim_derm,
                self.dim_cfp,
                self.dim_patho,
                self.dim_surg,
                self.dim_ct,
                self.dim_mr,
                self.dim_us,
                self.dim_xray,
                self.dim_other,
            )
            if include_3d_projector:
                self.domain_input_dims = self.domain_input_dims + (self.dim_3dct,)
                self.domain_input_dims = self.domain_input_dims + (self.dim_3dmr,)

            print(f"[WRAPPER] 📖📖📖 Domain feature input dimensions: {self.domain_input_dims}")
            self.projectors = nn.ModuleList([nn.Linear(d, self.dim) for d in self.domain_input_dims])
            if expert_alignment_weight > 0:
                self.expert_projectors = nn.ModuleList([nn.Linear(self.dim, d) for d in self.domain_input_dims])

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Migrate historical checkpoint keys for expert-space projectors.
        legacy_prefix = prefix + "consistency_projectors."
        expert_prefix = prefix + "expert_projectors."
        for key in list(state_dict):
            if key.startswith(legacy_prefix):
                expert_key = expert_prefix + key[len(legacy_prefix):]
                state_dict.setdefault(expert_key, state_dict.pop(key))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, x, features, domains, training=True, return_batched_tensor=False, **kwargs):
        """
        Args:
            x:
              - Tensor: [B, C, H, W]
              - Or list/tuple of length B, each is [1, C, Hi, Wi] (possibly different sizes)
            domains: Long tensor [B,], values in [0, len(self.projectors)-1]
            features: list/tuple of length B, each is [1, Ci].
        """
        if isinstance(x, (list, tuple)):
            batch_size = len(x)
        else:
            batch_size = int(x.shape[0])

        if not torch.is_tensor(domains):
            raise TypeError("domains must be a torch.Tensor of shape [B,].")
        if domains.ndim != 1 or int(domains.shape[0]) != batch_size:
            raise ValueError(f"domains must have shape [B,], got {tuple(domains.shape)} with B={batch_size}.")
        if domains.dtype != torch.long:
            domains = domains.long()

        outputs = self.core(x, training=training, use_kl=self.use_kl, return_batched_tensor=return_batched_tensor)
        core_features = outputs['cls_token']  # a tensor with shape [B, self.dim]
        device = core_features.device
        domains = domains.to(device=device)
        outputs["clip_core_features"] = F.normalize(core_features, dim=-1)

        if self.projectors is not None:
            num_domains = len(self.projectors)
            if torch.any(domains < 0) or torch.any(domains >= num_domains):
                raise ValueError(f"domains has values outside [0, {num_domains-1}].")
            
            # ---- DDP-safe aligned external features ----
            # Always evaluate every projector; only the matching-domain samples contribute to the sum.
            external_features = torch.zeros(batch_size, self.dim, device=device, dtype=core_features.dtype)

            for d, (projector, in_dim) in enumerate(zip(self.projectors, self.domain_input_dims)):
                # projector param dtype/device (important under AMP / mixed precision)
                p = next(projector.parameters())
                align_dtype = p.dtype
                align_device = p.device

                # full input for this projector: [B, in_dim], fill with zeros then scatter real features
                inp = torch.zeros(batch_size, in_dim, device=align_device, dtype=align_dtype)

                mask = (domains == d)  # [B] bool on device
                idx = mask.nonzero(as_tuple=False).squeeze(1)  # [Nd]

                if idx.numel() > 0:
                    # stack features for this domain
                    stacked = torch.stack(
                        [features[i].squeeze().to(device=align_device, dtype=align_dtype) for i in idx.tolist()],
                        dim=0
                    )  # [Nd, in_dim]
                    inp.index_copy_(0, idx.to(align_device), stacked)

                out_d = projector(inp)  # [B, self.dim]  (ALWAYS executed => no unused params)
                out_d = out_d.to(device=device, dtype=core_features.dtype)

                # gate contribution by domain mask (so non-matching samples contribute exactly zero)
                external_features = external_features + out_d * mask.to(core_features.dtype).unsqueeze(1)

            outputs["clip_image_features"] = F.normalize(external_features, dim=-1)

        # ---- DDP-safe expert-space projection ----
        if self.expert_projectors is not None:
            expert_aligned_features = [None] * batch_size
            # DDP-safe: scalar sum ensures all projector params are used in computation graph
            expert_ddp_anchor = torch.zeros(1, device=device, dtype=core_features.dtype)

            for d, (expert_projector, out_dim) in enumerate(zip(self.expert_projectors, self.domain_input_dims)):
                p = next(expert_projector.parameters())
                expert_dtype = p.dtype
                expert_device = p.device

                # Always run on all samples (DDP-safe: ensures no unused params)
                inp = core_features.to(device=expert_device, dtype=expert_dtype)  # [B, self.dim]
                out_d = expert_projector(inp)  # [B, out_dim]
                out_d = out_d.to(device=device, dtype=core_features.dtype)

                mask = (domains == d)
                mask_float = mask.to(core_features.dtype).unsqueeze(1)  # [B, 1]

                # Gate by mask (non-matching samples contribute zero, but computation happened)
                gated = out_d * mask_float  # [B, out_dim]

                # DDP anchor: accumulate to ensure all projector outputs are in computation graph
                expert_ddp_anchor = expert_ddp_anchor + gated.sum()

                # Assign to matching samples
                idx = mask.nonzero(as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    for i in idx.tolist():
                        expert_aligned_features[i] = gated[i:i+1]  # [1, out_dim]

            outputs["expert_aligned_features"] = expert_aligned_features
            outputs["expert_ddp_anchor"] = expert_ddp_anchor  # Can be used as: loss + 0 * anchor

        return outputs

    def img_to_reconstructed_img(self, image):
        return self.core.img_to_reconstructed_img(image)
