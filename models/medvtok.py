import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import Tuple, Optional, Union, List, Literal, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as tfs

from models.utils import (
    seqlens_to_dense, 
    concatenated_to_img_list, concatenated_to_feature_maps,
    images_to_concatenated, batch_tensor_to_concatenated,
    split_by_temporal_batches, reconstruct_from_temporal_slices
)
from models.transformer import ConcatenatedTransformerBlock, LearnedPositionEmbedder, RMSNorm, LayerNorm32


def all_tuples_equal(tuples_list):
    if not tuples_list:
        return True
    first = tuples_list[0]
    return all(t == first for t in tuples_list[1:])



class DiagonalGaussianDistribution:
    """
    Gaussian distribution with diagonal covariance.

    Used for VAE latent sampling.
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        """
        Args:
            parameters: [N, 2*C] or [B, L, 2*C] tensor with mean and logvar concatenated
        """
        self.original_dtype = parameters.dtype
        self.parameters = parameters  #.to(torch.float32)
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.deterministic = deterministic
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self) -> torch.Tensor:
        """Sample from the distribution."""
        sample = torch.randn_like(self.mean, dtype=self.parameters.dtype, device=self.parameters.device)
        x = self.mean + self.std * sample
        return x.to(self.original_dtype)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    def mode(self) -> torch.Tensor:
        """Return the mode (mean) of the distribution."""
        return self.mean.to(self.original_dtype)
    
    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1]
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )
    
    def nll(self, sample: torch.Tensor, dims=None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])

        if dims is None:
            dims = [1, 2, 3] if self.mean.dim() == 4 else [1, 2, 3, 4]
        
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )



class MedVTokEncoder(nn.Module):
    """
    Encoder: transforms input patches to latent representation.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: int,
        mlp_channels: int,
        attn_mode: str = "full",
        qk_rms_norm: bool = False,
        use_bias: bool = False,
        use_rms_norm: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks

        # Input projection (works on [N, C])
        self.input_layer = nn.Linear(in_channels, model_channels, bias=use_bias)
        self.pos_embedder = LearnedPositionEmbedder(model_channels)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ConcatenatedTransformerBlock(
                channels=model_channels,
                num_heads=num_heads,
                mlp_channels=mlp_channels,
                attn_mode=attn_mode,
                use_bias=use_bias,
                use_rope=True,
                qk_rms_norm=qk_rms_norm,
                use_rms_norm=use_rms_norm,
            )
            for _ in range(num_blocks)
        ])

        # Post layer norm
        if use_rms_norm:
            self.post_layernorm = RMSNorm(model_channels)
        else:
            self.post_layernorm = LayerNorm32(model_channels)

    def forward(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        seqlens: List[int],
        kv_cache: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Args:
            feats: [N, in_channels] concatenated features
            coords: [N, 5] concatenated coordinates
            seqlens: List[int] patches per image
            kv_cache: Optional KV cache dict

        Returns:
            h: [N, model_channels] transformed features
            coords: [N, 5] coordinates (unchanged)
            seqlens: List[int] (unchanged)
        """
        # Input projection (works directly on [N, C])
        h = self.input_layer(feats)  # [N, model_channels]

        # Add position embeddings if needed
        if self.pos_embedder is not None:
            pos_emb = self._compute_pos_embedding_concatenated(coords, seqlens)
            h = h + pos_emb.to(h.dtype)

        # Transformer blocks (CONCATENATED - no dense conversion!)
        for block in self.blocks:
            h, coords, seqlens, _ = block(h, coords, seqlens)

        # Post layer norm
        h = self.post_layernorm(h)

        return h, coords, seqlens

    def _compute_pos_embedding_concatenated(
        self,
        coords: torch.Tensor,
        seqlens: List[int],
    ) -> torch.Tensor:
        """
        Compute learned position embeddings for concatenated format.

        Args:
            coords: [N, 5] concatenated coordinates
            seqlens: List[int] sequence lengths

        Returns:
            pos_emb: [N, model_channels]
        """
        N = coords.shape[0]
        device = coords.device
        B = len(seqlens)

        # Get base positional embeddings
        pos_size = self.pos_embedder.position_embedding_size
        base_emb = self.pos_embedder.position_embedding.weight  # [num_patches, C]
        base_emb = base_emb.reshape(pos_size, pos_size, -1)  # [H, W, C]

        # For efficiency, we compute embeddings directly
        # Extract spatial coordinates (x, y)
        x_coords = coords[:, 2].long()  # [N]
        y_coords = coords[:, 3].long()  # [N]

        # Compute spatial shapes per batch
        offset = 0
        all_embeddings = []
        for b, seqlen in enumerate(seqlens):
            end = offset + seqlen
            batch_coords = coords[offset:end]
            batch_x = x_coords[offset:end]
            batch_y = y_coords[offset:end]

            # Get max coords for this batch
            max_x = batch_x.max().item() + 1 if seqlen > 0 else 1
            max_y = batch_y.max().item() + 1 if seqlen > 0 else 1

            # Resize position embeddings for this batch
            resized = F.interpolate(
                base_emb.permute(2, 0, 1).unsqueeze(0).float(),  # [1, C, H, W]
                size=(max_x, max_y),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )  # [1, C, max_x, max_y]
            resized = resized.squeeze(0).permute(1, 2, 0)  # [max_x, max_y, C]
            resized = resized.to(base_emb.dtype)

            # Index into resized embeddings
            batch_emb = resized[batch_x, batch_y]  # [seqlen, C]
            all_embeddings.append(batch_emb)

            offset = end

        return torch.cat(all_embeddings, dim=0)  # [N, C]


class MedVTokDecoder(nn.Module):
    """
    Decoder: transforms latent representation back to output patches.
    """

    def __init__(
        self,
        latent_channels: int,
        model_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: int,
        mlp_channels: int,
        attn_mode: str = "full",
        qk_rms_norm: bool = False,
        use_bias: bool = False,
        use_rms_norm: bool = False,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        # Input projection
        self.input_layer = nn.Linear(latent_channels, model_channels, bias=use_bias)
        self.pos_embedder = LearnedPositionEmbedder(model_channels)

        # Transformer blocks - use CONCATENATED version for efficiency
        self.blocks = nn.ModuleList([
            ConcatenatedTransformerBlock(
                channels=model_channels,
                num_heads=num_heads,
                mlp_channels=mlp_channels,
                attn_mode=attn_mode,
                use_bias=use_bias,
                use_rope=True,
                qk_rms_norm=qk_rms_norm,
                use_rms_norm=use_rms_norm,
            )
            for _ in range(num_blocks)
        ])

        # Output norm and projection
        if use_rms_norm:
            self.out_norm = RMSNorm(model_channels)
        else:
            self.out_norm = LayerNorm32(model_channels)

        # Output layer
        self.out_layer = nn.Linear(model_channels, out_channels, bias=use_bias)

    def forward(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        seqlens: List[int],
        kv_cache: Optional[Dict] = None,
        kv_cache_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], Dict]:
        """
        Args:
            feats: [N, latent_channels] concatenated latent features
            coords: [N, 5] concatenated coordinates
            seqlens: List[int] patches per image
            kv_cache: Optional dict keyed by block index str (e.g. "0", "1", ...)
                      Each value is a per-block KV cache dict.
            kv_cache_size: Number of temporal frames to cache (0 = disabled)

        Returns:
            (h, coords, seqlens, new_kv_cache)
            new_kv_cache: dict keyed by block index str
        """
        if kv_cache is None:
            kv_cache = {}

        # Input projection (works directly on [N, C])
        h = self.input_layer(feats)  # [N, model_channels]

        # Add position embeddings if needed
        if self.pos_embedder is not None:
            pos_emb = self._compute_pos_embedding_concatenated(coords, seqlens)
            h = h + pos_emb.to(h.dtype)

        # Transformer blocks
        new_kv_cache = {}
        for i, block in enumerate(self.blocks):
            block_cache = kv_cache.get(str(i), None)
            h, coords, seqlens, new_block_cache = block(
                h, coords, seqlens,
                kv_cache=block_cache,
                kv_cache_size=kv_cache_size,
            )
            new_kv_cache[str(i)] = new_block_cache

        # Output norm and projection
        h = self.out_norm(h)
        h = self.out_layer(h)

        return h, coords, seqlens, new_kv_cache

    def _compute_pos_embedding_concatenated(
        self,
        coords: torch.Tensor,
        seqlens: List[int],
    ) -> torch.Tensor:
        """
        Compute learned position embeddings for concatenated format.

        Args:
            coords: [N, 5] concatenated coordinates
            seqlens: List[int] sequence lengths

        Returns:
            pos_emb: [N, model_channels]
        """
        N = coords.shape[0]
        device = coords.device
        B = len(seqlens)

        # Get base positional embeddings
        pos_size = self.pos_embedder.position_embedding_size
        base_emb = self.pos_embedder.position_embedding.weight  # [num_patches, C]
        base_emb = base_emb.reshape(pos_size, pos_size, -1)  # [H, W, C]

        # For efficiency, we compute embeddings directly
        # Extract spatial coordinates (x, y)
        x_coords = coords[:, 2].long()  # [N]
        y_coords = coords[:, 3].long()  # [N]

        # Compute spatial shapes per batch
        offset = 0
        all_embeddings = []
        for b, seqlen in enumerate(seqlens):
            end = offset + seqlen
            batch_coords = coords[offset:end]
            batch_x = x_coords[offset:end]
            batch_y = y_coords[offset:end]

            # Get max coords for this batch
            max_x = batch_x.max().item() + 1 if seqlen > 0 else 1
            max_y = batch_y.max().item() + 1 if seqlen > 0 else 1

            # Resize position embeddings for this batch
            resized = F.interpolate(
                base_emb.permute(2, 0, 1).unsqueeze(0).float(),  # [1, C, H, W]
                size=(max_x, max_y),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )  # [1, C, max_x, max_y]
            resized = resized.squeeze(0).permute(1, 2, 0)  # [max_x, max_y, C]
            resized = resized.to(base_emb.dtype)

            # Index into resized embeddings
            batch_emb = resized[batch_x, batch_y]  # [seqlen, C]
            all_embeddings.append(batch_emb)

            offset = end

        return torch.cat(all_embeddings, dim=0)  # [N, C]
    
    def get_last_param(self):
        return self.out_layer.weight



class MeanPoolProjector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projector = nn.Sequential(
            LayerNorm32(dim),
            self.init_mean_pool_projector(nn.Linear(dim, dim, bias=False)),
        )
    
    def init_mean_pool_projector(self, m):
        if isinstance(m, nn.Linear) and m.in_features == m.out_features:
            nn.init.eye_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        return m

    def forward(self, x, mask=None):
        """
        x:    [B, L, D]
        mask: [B, L], True means valid token, False means padding
        """
        if mask is None:
            pooled = x.mean(dim=1)  # [B, D]
        else:
            mask = mask.to(dtype=torch.bool, device=x.device)
            mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)   # [B, L, 1]

            denom = mask_f.sum(dim=1).clamp_min(1.0)        # [B, 1]
            pooled = (x * mask_f).sum(dim=1) / denom        # [B, D]

        return self.projector(pooled)



class MedVTok(nn.Module):
    """
    MedVTok with concatenated tensors.

    Works with (feats, coords, seqlens) format where:
    - feats: [N, D] concatenated features (N = sum(seqlens))
    - coords: [N, 5] concatenated coordinates
    - seqlens: List[int] patches per image
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        in_channels: int = 3072,
        out_channels: int = 3072,
        latent_channels: int = 48,
        encoder_model_channels: int = 1152,
        encoder_num_blocks: int = 27,
        encoder_num_heads: Optional[int] = 16,
        encoder_mlp_channels: int = 4304,
        encoder_attn_mode: str = "full",
        encoder_qk_rms_norm: bool = False,
        encoder_use_bias: bool = True,
        encoder_use_rms_norm: bool = False,
        decoder_model_channels: int = 1152,
        decoder_num_blocks: int = 27,
        decoder_num_heads: Optional[int] = 16,
        decoder_mlp_channels: int = 4304,
        decoder_attn_mode: str = "full",
        decoder_qk_rms_norm: bool = True,
        decoder_use_bias: bool = False,
        decoder_use_rms_norm: bool = True,
        use_decoder: bool = True,
        use_logit_scale_params: bool = True,
        use_logit_bias_params: bool = True,
        padding_mode: str = 'zero',
    ):
        super().__init__()

        # Store configuration
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        self.kv_cache_size = 0
        self.num_sample_frames_batch_size = 16
        # The total decoder context window in temporal frames (before dividing by patch_size[0]).
        # This is the maximum number of temporal positions a single decoder forward pass ever sees,
        # inclusive of both new frames and any cached context from the previous chunk.
        # During training (training=True): this is exactly the chunk size (stride = window, no overlap, no cache). 
        # The model was trained with each 16-frame window decoded independently.
        # During inference (training=False): the window is filled with `num_sample_frames_stride` new frames 
        # plus `kv_cache_size` cached frames from the previous chunk.
        
        self.num_sample_frames_stride = 16
        # Controls how many new frames to advance per decoding step at inference. 
        # With the default stride=16, there's no overlap -- consecutive chunks 
        # tile the video with no shared new frames.
        #
        # Fundamental constraints: 
        # kv_cache_size + num_sample_frames_stride ≤ num_sample_frames_batch_size
        #
        # Long-video smoothing: num_sample_frames_stride = 12, kv_cache_size = 4. 
        # Fill the 16-frame window with 12 new + 4 cached. Only worth trying empirically.
        
        self.dim = encoder_model_channels
        self.padding_mode = padding_mode

        # Encoder
        self.encoder_head = MeanPoolProjector(encoder_model_channels)

        self.encoder = MedVTokEncoder(
            in_channels=in_channels,
            model_channels=encoder_model_channels,
            num_blocks=encoder_num_blocks,
            num_heads=encoder_num_heads or (encoder_model_channels // 64),
            mlp_channels=encoder_mlp_channels,
            attn_mode=encoder_attn_mode,
            qk_rms_norm=encoder_qk_rms_norm,
            use_bias=encoder_use_bias,
            use_rms_norm=encoder_use_rms_norm,
        )
        
        # Projection to latent space (mean + logvar)
        proj_out_channels = 2 * latent_channels
        self.proj = nn.Linear(encoder_model_channels, proj_out_channels, bias=False)

        # Decoder (optional)
        if use_decoder:
            self.decoder = MedVTokDecoder(
                latent_channels=latent_channels,
                model_channels=decoder_model_channels,
                out_channels=out_channels,
                num_blocks=decoder_num_blocks,
                num_heads=decoder_num_heads or (decoder_model_channels // 64),
                mlp_channels=decoder_mlp_channels,
                attn_mode=decoder_attn_mode,
                qk_rms_norm=decoder_qk_rms_norm,
                use_bias=decoder_use_bias,
                use_rms_norm=decoder_use_rms_norm,
            )

        if use_logit_scale_params:
            self.logit_scale = nn.Parameter(torch.tensor([2.0]))
        else:
            self.register_buffer("logit_scale", torch.tensor([2.0]), persistent=False)

        if use_logit_bias_params:
            self.logit_bias = nn.Parameter(torch.tensor([-10.0]))
        else:
            self.register_buffer("logit_bias", torch.tensor([-10.0]), persistent=False)

    def encode(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, List[int]]],
        normalize: bool = False,
        padding_mode: str = 'zero',
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
        """
        Encode input to latent representation.

        Args:
            x: Either:
               - torch.Tensor [B, C, H, W] or [B, T, C, H, W]: Convert to concatenated format
               - Tuple (feats, coords, seqlens): Already in concatenated format
            normalize: Whether to normalize features

        Returns:
            enc_proj_feats: [N, 2*latent_channels] for Gaussian (mean + logvar).
                             N is the number of patches. 
            enc_coords: [N, 5] coordinates.
            enc_seqlens: List[int] patches per image.
            image_feat: [B, encoder_channels] pooled global features per batch.
            enc_full_feats: [N, encoder_channels] encoder output
        """
        # Convert input to concatenated format if needed
        if isinstance(x, torch.Tensor):
            feats, coords, seqlens = batch_tensor_to_concatenated(x, self.patch_size, normalize=normalize, padding_mode=padding_mode)
        else:
            feats, coords, seqlens = x

        # Encode through transformer (concatenated format)
        h, enc_coords, enc_seqlens = self.encoder(feats, coords, seqlens)

        # Convert to dense format for attention pooling head
        h_dense, coords_dense, mask_dense = seqlens_to_dense(h, enc_coords, enc_seqlens)

        # Extract global image features via attention pooling
        image_feat = self.encoder_head(h_dense, mask=mask_dense)

        # Project to latent space (stays in concatenated format)
        enc_proj_feats = self.proj(h)  # [N, 2*latent_channels]

        return enc_proj_feats, enc_coords, enc_seqlens, image_feat, h

    def decode(
        self,
        z: Tuple[torch.Tensor, torch.Tensor, List[int]],
        return_batched_tensor: bool = False,
        training: bool = False,
        data_types = None,
        orig_shapes = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Decode latent representation to output.

        Args:
            z: Tuple (enc_feats, enc_coords, enc_seqlens)
               enc_feats: [N, latent_channels]
               coords: [N, 5]
               seqlens: List[int]
            return_batched_tensor: Whether to return stacked tensor or list
            training: Whether in training mode.
               The distinction between `training=True` and `training=False` for decoding 
               matters only when the model is configured with 
               `kv_cache_size` > 0 and `num_sample_frames_batch_size` > `num_sample_frames_stride`: 
               inference could then use overlapping windows with KV context, while 
               training forcibly disables that for clean gradient flow.
               
            **DDP risk for multiple self.decoder calls per training step**: unused parameters                                                                                                                               
                
            PyTorch DDP (with find_unused_parameters=False, the default) assumes every parameter 
            contributes a gradient after loss.backward(). A problem arises if:         
                - Some temporal chunks are empty (sum(ss) == 0), 
                  which is guarded by `if sum(ss) > 0:` in the decode loop — 
                  those chunks skip self.decoder entirely. 
                - If this causes different DDP ranks to call `self.decoder` 
                  a different number of times (e.g., one GPU's video is short, 
                  another's is long), DDP's synchronization hooks can deadlock or error out.                                                                                                               

            Under `training=True` with non-overlapping **equal-length** videos (same number of 
            temporal chunks on all ranks), this is fine. 
            If video lengths are heterogeneous across DDP ranks, you need either 
            `find_unused_parameters=True` or ensure all ranks call `self.decoder` 
            the same number of times (e.g., pad to the same number of chunks). 

        Returns:
            Either stacked tensor [B, T, H, W, C] or list of tensors
        """
        enc_feats, enc_coords, enc_seqlens = z
        
        frame_batch_size = self.num_sample_frames_batch_size // self.patch_size[0]
        frame_batch_strides = self.num_sample_frames_stride // self.patch_size[0]
        kv_cache_size = self.kv_cache_size // self.patch_size[0]
        
        if training:
            kv_cache_size = 0
            frame_batch_strides = frame_batch_size
        else:
            # No overlap during inference
            frame_batch_size = frame_batch_strides
        
        temporal_slices = split_by_temporal_batches(
            enc_feats, enc_coords, enc_seqlens,
            frame_batch_size, frame_batch_strides,
            adjust_temporal=True, offset=kv_cache_size,
        )

        kv_cache = None
        processed_slices = []
        for (sf, sc, ss, offset) in temporal_slices:
            if sum(ss) > 0:
                df, dc, ds, kv_cache = self.decoder(sf, sc, ss, kv_cache, kv_cache_size)
                processed_slices.append((df, dc, ds, offset))
            else:
                processed_slices.append((sf, sc, ss, offset))
        
        dec_feats, dec_coords, dec_seqlens = reconstruct_from_temporal_slices(processed_slices)  

        # Decode through transformer
        #dec_feats, dec_coords, dec_seqlens = self.decoder(enc_feats, enc_coords, enc_seqlens)
        # enc_feats: tensor [N, 48]
        # dec_feats: tensor [N, 3072]
        # dec_coords: tensor [N, 5]
        # dec_seqlens: list [H1/Ph*W1/Pw, H2/Ph*W2/Pw, ..., HN/Ph*WN/Pw]

        # Convert to image list [T, C, H, W]
        dec_list = concatenated_to_img_list(dec_feats, dec_coords, dec_seqlens, self.patch_size)
        
        # Ensure the output to be [B, C, H, W] (for images) or [B, C, T, H, W] (for videos)
        return self.format_decoded_list(
            dec_list, 
            data_types=data_types,
            return_batched_tensor=return_batched_tensor,
            orig_shapes=orig_shapes
        )

    def reshape_feature_map(self, feats, coords, seqlens):
        """Reshape [N, D] to [B, Ph*Pw, D]"""
        feats = concatenated_to_feature_maps(feats, coords, seqlens)
        try:
            feats = torch.cat(feats, dim=0)  # [B, Ph, Pw, D]
            bs = feats.shape[0]
            dim = feats.shape[-1]
            feats = feats.reshape(bs, -1, dim)
        except Exception as err:
            feats = [f.reshape(f.shape[0], -1, f.shape[-1]) for f in feats]
        return feats
    
    def print_arr(self, x, name=''):
        if isinstance(x, (torch.Tensor, np.ndarray)):
            xmin = x.min().item()
            xmax = x.max().item()
            print(f">>> {name}: shape={tuple(x.shape)}, range=({xmin:.4f}, {xmax:.4f})")
        else:
            s = f">>> {name}: Not a tensor "
            if isinstance(x, (list, tuple)):
                s += f"(type={type(x)}, len={len(x)}): "
                for ii, xx in enumerate(x):
                    if isinstance(xx, (torch.Tensor, np.ndarray)):
                        xxmin = xx.min().item()
                        xxmax = xx.max().item()
                        s += f"\n>>>>>> [{ii}]: shape={tuple(xx.shape)}, range=({xxmin:.4f}, {xxmax:.4f})"
                    elif isinstance(xx, (int, float)):
                        s += f"\n>>>>>> [{ii}]: Not a tensor (type={type(xx)}): {xx}"
                    else:
                        s += f"\n>>>>>> [{ii}]: Not a tensor (type={type(xx)})"

            elif isinstance(x, dict):
                s += f"(type={type(x)}, len={len(x)}): "
                for key, xx in x.items():
                    if isinstance(xx, (torch.Tensor, np.ndarray)):
                        xxmin = xx.min().item()
                        xxmax = xx.max().item()
                        s += f"\n>>>>>> [{key}]: shape={tuple(xx.shape)}, range=({xxmin:.4f}, {xxmax:.4f})"
                    elif isinstance(xx, (int, float)):
                        s += f"\n>>>>>> [{ii}]: Not a tensor (type={type(xx)}): {xx}"
                    else:
                        s += f"\n>>>>>> [{key}]: Not a tensor (type={type(xx)})"
            else:
                s += f"(type={type(x)})"

            print(s)

    def prepare_inputs(self, x, padding_mode='zero'):
        if isinstance(x, (list, tuple)):
            # Assume x is a list of tensors already normalized to [-1, 1]
            feats, coords, seqlens, data_types, orig_shapes = images_to_concatenated(x, self.patch_size, normalize=False, padding_mode=padding_mode)
            return (feats, coords, seqlens), data_types, orig_shapes
        elif isinstance(x, torch.Tensor):
            feats, coords, seqlens, data_types, orig_shapes = batch_tensor_to_concatenated(x, self.patch_size, normalize=False, padding_mode=padding_mode)
            return (feats, coords, seqlens), data_types, orig_shapes
        else:
            raise ValueError(f"Should be either a list of normalized tensors or a batched tensor. Got {type(x)}.")

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, List[int]]],
        sample_posterior: bool = True,
        return_batched_tensor: bool = True,
        training: bool = False,
        decode: bool = True,
        verbose: bool = False,
        use_kl: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass: encode then decode.

        Args:
            x: Input (see encode() for format)
            sample_posterior: Whether to sample from posterior

        Returns:
            Reconstructed output
        """
        padding_mode = self.padding_mode
        x, data_types, orig_shapes = self.prepare_inputs(x, padding_mode=padding_mode)

        # Encode
        enc_proj_feats, coords, seqlens, image_feat, enc_full_feats = self.encode(x, normalize=False, padding_mode=padding_mode)

        # Sample or use mean
        posterior = DiagonalGaussianDistribution(enc_proj_feats)
        latent = posterior.sample() if sample_posterior else posterior.mode()

        if training and use_kl:
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        else:
            kl_loss = latent.new_tensor(0.0)
    
        if verbose:
            # n_patches_i = hi / ph * wi / pw
            # n_patches = n_patches_0 + n_patches_1 + ...
            self.print_arr(x, 'x')                            # (B, 3, H, W) or List[(n_patches, 3072), (n_patches, 5), List]
            self.print_arr(enc_proj_feats, 'enc_proj_feats')  # (n_patches, 96)
            self.print_arr(coords, 'coords')                  # (n_patches, 5)
            self.print_arr(seqlens, 'seqlens')                # value = [n_patches_0, n_patches_1, n_patches_2, ...]
            self.print_arr(image_feat, 'image_feat')          # (B, d=1152)
            self.print_arr(enc_full_feats, 'enc_full_feats')  # (n_patches, 1152)
            self.print_arr(latent, 'latent')                  # (n_patches, l=48)

        # Decode
        if decode:
            dec = self.decode(
                (latent, coords, seqlens), 
                return_batched_tensor=return_batched_tensor, 
                training=training, 
                data_types=data_types,
                orig_shapes=orig_shapes
            )
        else:
            dec = None
        
        enc_full_feats = self.reshape_feature_map(enc_full_feats, coords, seqlens)
        latent = self.reshape_feature_map(latent, coords, seqlens)

        if verbose:
            self.print_arr(dec, 'dec')                                   # (B, 3, H, W) or List[(3, H0, W0), (3, H1, W1), ...]
            self.print_arr(enc_full_feats, 'enc_full_feats (reshaped)')  # (B, n_patches_per_sample, d=1152) or List[(1, n_patches_0, d=1152), (1, n_patches_1, d=1152), ...]
            self.print_arr(latent, 'latent (reshaped)')                  # (B, n_patches_per_sample, l=48) or List[(1, n_patches_0, l=48), (1, n_patches_1, l=48), ...]

        return {
            'recons': dec,                    # (B, 3, H, W) or List[(3, H1, W1), ... (3, Hn, Wn)]
            'img_rec': dec,                   # training-compatible alias
            'feats': image_feat,               # (B, 1152)
            'cls_token': image_feat,           # training-compatible alias
            'patch_tokens': enc_full_feats,    # (B, n_patches_i, 1152) or List[(1, n_patches_0, 1152), ...]
            'latent': latent,                  # (B, s*s, 48) or List[(1, n_patches_0, 48), ...]
            'coords': coords,                  # (n_patches, 5)
            'enc_proj_feats': enc_proj_feats,  # (n_patches, 96) 
            'seqlens': seqlens,                # value = [n_patches_0, n_patches_1, n_patches_2, ...]
            'logit_scale': self.logit_scale.exp(),   # (1,)
            'logit_bias': self.logit_bias,     # (1,)
            'kl_loss': kl_loss,                # tensor scalar
        }
    
    def format_decoded_list(self, raw_decoded_list, data_types, return_batched_tensor=True, orig_shapes=None):
        decoded_list = []
        decoded_shapes = []
        
        for idx, decoded in enumerate(raw_decoded_list):
            orig_t, orig_h, orig_w = orig_shapes[idx]  # T,H,W
            
            if decoded.dim() == 4:
                if decoded.shape[1] == 3:    # [T, C, H, W]
                    decoded = decoded.permute(1, 0, 2, 3)  # [C, T, H, W]
                elif decoded.shape[3] == 3:  # [T, H, W, C]
                    decoded = decoded.permute(3, 0, 1, 2)  # [C, T, H, W]
                elif decoded.shape[0] == 3:  # [C, T, H, W]
                    pass
                else:
                    raise ValueError(f"Given a 4D tensor, we expect its shape to be [T, 3, H, W] or [T, H, W, 3] or [3, T, H, W], but got {decoded.shape}")

                if data_types[idx] == 'image':
                    decoded = decoded[:, 0]  # [C, H, W]
                else:
                    if decoded.shape[1] > orig_t:
                        decoded = decoded[:, :orig_t]

            elif decoded.dim() == 3:
                if decoded.shape[0] == 3:   # [C, H, W]
                    decoded = decoded                   # [C, H, W]
                elif decoded.shape[2] == 3:  # [H, W, C]
                    decoded = decoded.permute(2, 0, 1)  # [C, H, W]
                else:
                    raise ValueError(f"Given a 3D tensor, we expect its shape to be [3, H, W] or [H, W, 3], but got {decoded.shape}")
            else:
                raise ValueError(f"Encounter invalid shape when decoding {idx}-th tensor. Expected 4D or 3D tensors, got {decoded.shape}")
            
            dec_h, dec_w = decoded.shape[-2:]
            if orig_h != dec_h or orig_w != dec_w:
                tmp = decoded.unsqueeze(0) if decoded.dim() == 3 else decoded
                tmp = F.interpolate(
                    tmp, size=(orig_h, orig_w), 
                    mode='bicubic', align_corners=False, antialias=True
                )
                decoded = tmp.squeeze(0) if decoded.dim() == 3 else tmp
            
            decoded_shapes.append(tuple(decoded.shape))
            decoded_list.append(decoded.clip(-1, 1).float())  # <-- this float is very important to make the training run
        
        return_batched_tensor = return_batched_tensor and all_tuples_equal(decoded_shapes)
        return torch.stack(decoded_list, dim=0) if return_batched_tensor else decoded_list



    def img_to_reconstructed_img(self, image):
        return self.forward(
            image,
            sample_posterior=True,
            training=False,
            decode=True,
        )["recons"]

    def freeze_decoder(self):
        self.decoder.eval()
        for parameter in self.decoder.parameters():
            parameter.requires_grad = False


def build_model(arch='medvtok-clip', img_size=256, device='cpu', pretrained_path=None, force_sem_only=False, padding_mode='zero', use_logit_text=False):

    common_kwargs = dict(
        patch_size=[4, 16, 16],
        in_channels=3072,
        out_channels=3072,
        latent_channels=48,
        encoder_model_channels=1152,
        encoder_num_blocks=27,
        encoder_num_heads=16,
        encoder_mlp_channels=4304,
        encoder_attn_mode="full",
        encoder_qk_rms_norm=False,
        encoder_use_bias=True,
        encoder_use_rms_norm=False,
        decoder_model_channels=1152,
        decoder_num_blocks=27,
        decoder_num_heads=16,
        decoder_mlp_channels=4304,
        decoder_attn_mode="full",
        decoder_qk_rms_norm=True,
        decoder_use_bias=False,
        decoder_use_rms_norm=True,
        use_decoder=True,
        padding_mode=padding_mode,
    )

    if 'noalign' in arch:
        common_kwargs.update(dict(use_logit_scale_params=False, use_logit_bias_params=False,))
    elif 'clip' in arch:
        common_kwargs.update(dict(use_logit_scale_params=True, use_logit_bias_params=False,))
    else:
        common_kwargs.update(dict(use_logit_scale_params=True, use_logit_bias_params=True,))

    model = MedVTok(**common_kwargs)

    if pretrained_path:
        sd = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        if 'args' in sd and 'trainer' in sd:  # dev checkpoint format
            sd = {k.replace('core.', ''):v for k,v in sd['trainer']['model'].items() if k.startswith('core.')}
            msg = model.load_state_dict(sd, strict=True)
        else:
            msg = model.load_state_dict(sd, strict=False)
        print(f"✅ Loading checkpoint for {arch} from {pretrained_path}:\n{msg}")
        del sd
    else:
        print(f"⚠️ Using random initialization!")

    if force_sem_only:
        model.freeze_decoder()

    return model.to(device)



def build_transform(image_shape):
    return tfs.Compose([
        tfs.ToTensor(),
        tfs.Resize(image_shape, interpolation=3, antialias=True),
        tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

