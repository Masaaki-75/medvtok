#
# Full attention for dense tensors AND concatenated tensors
# Supports flash_attn, xformers, and PyTorch backends
#

from typing import Optional, Dict, Tuple, Literal, List
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


# Try to import flash_attn
try:
    import flash_attn
    from flash_attn import flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# Try to import xformers
try:
    import xformers.ops as xops
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False

# Get attention backend from environment or use default
ATTN_BACKEND = os.environ.get("ATTN_BACKEND", "auto")  # auto, xformers, flash_attn, pytorch




########################################################
## RoPE
########################################################

class LearnedPositionEmbedder(nn.Module):
    """
    Learned 2D spatial position embeddings with interpolation.

    Used in "learned" and "joint" positional encoding modes.
    """

    def __init__(self, hidden_size: int, num_patches: int = 256):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_patches = num_patches
        self.position_embedding_size = int(self.num_patches ** 0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def infer_spatial_shapes(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.LongTensor:
        """
        Infer spatial shapes from coordinates.

        Args:
            coords: [B, L, 5] coordinate tensor
            mask: [B, L] boolean mask

        Returns:
            [B, 2] tensor with [height, width] for each batch
        """
        B = coords.shape[0]
        device = coords.device

        # VECTORIZED: Compute max per batch without Python loops
        # Use a large negative value for masked positions so they don't affect max
        NEG_INF = -1000000

        # Clone coords and set masked positions to NEG_INF
        masked_coords = coords.clone()
        # Expand mask for broadcasting: [B, L] -> [B, L, 1]
        mask_expanded = mask.unsqueeze(-1)
        masked_coords = torch.where(mask_expanded, masked_coords, torch.full_like(masked_coords, NEG_INF))

        # Get max of x (index 2) and y (index 3) coordinates per batch
        # max returns (values, indices), we only need values
        max_x = masked_coords[:, :, 2].max(dim=1)[0] + 1  # [B]
        max_y = masked_coords[:, :, 3].max(dim=1)[0] + 1  # [B]

        # Handle empty batches (where max would be NEG_INF + 1)
        empty_mask = (max_x < 0)
        max_x = torch.where(empty_mask, torch.ones_like(max_x), max_x)
        max_y = torch.where(empty_mask, torch.ones_like(max_y), max_y)

        spatial_shapes = torch.stack([max_x, max_y], dim=1)  # [B, 2]
        return spatial_shapes

    def resize_positional_embeddings(
        self,
        positional_embeddings: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Resize and extract positional embeddings for given coordinates.

        Args:
            positional_embeddings: [H, W, C] base embeddings
            coords: [B, L, 5] coordinates
            mask: [B, L] boolean mask
            spatial_shapes: [B, 2] spatial shapes per batch

        Returns:
            [B, L, C] resized embeddings
        """
        B, L = coords.shape[:2]
        embed_dim = positional_embeddings.shape[-1]
        device = coords.device
        source_dtype = positional_embeddings.dtype

        # Initialize output
        output = torch.zeros((B, L, embed_dim), device=device, dtype=source_dtype)

        # [H, W, C] -> [1, C, H, W] for interpolation
        pos_emb = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 for CPU
        if pos_emb.device.type == "cpu":
            pos_emb = pos_emb.to(torch.float32)

        for batch_idx in range(B):
            batch_mask = mask[batch_idx]
            if not batch_mask.any():
                continue

            batch_coords = coords[batch_idx]
            height, width = spatial_shapes[batch_idx]

            # Resize positional embeddings for this batch
            batch_pos_emb = F.interpolate(
                pos_emb,
                size=(height.item(), width.item()),
                mode="bilinear",
                align_corners=False,
            )  # [1, C, H, W]

            # [1, C, H, W] -> [H, W, C]
            batch_pos_emb = batch_pos_emb.squeeze(0).permute(1, 2, 0)

            # Extract coordinates
            patch_x = batch_coords[:, 2].long()
            patch_y = batch_coords[:, 3].long()

            # Clamp to valid range
            patch_x = torch.clamp(patch_x, 0, height - 1)
            patch_y = torch.clamp(patch_y, 0, width - 1)

            # Index into resized embeddings
            batch_embeddings = batch_pos_emb[patch_x, patch_y]

            # Cast back to original dtype
            batch_embeddings = batch_embeddings.to(source_dtype)

            # Assign to output
            output[batch_idx] = batch_embeddings

        return output

    def forward(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute positional embeddings.

        Args:
            coords: [B, L, 5] coordinate tensor
            mask: [B, L] boolean mask

        Returns:
            [B, L, C] positional embeddings
        """
        if coords.shape[1] == 0:
            return torch.zeros((coords.shape[0], 0, self.embed_dim), device=coords.device)

        # Get base positional embeddings reshaped to spatial grid
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size,
            self.position_embedding_size,
            -1,
        )

        # Infer spatial shapes from coordinates
        spatial_shapes = self.infer_spatial_shapes(coords, mask)

        # Resize positional embeddings
        resized_embeddings = self.resize_positional_embeddings(
            positional_embeddings,
            coords,
            mask,
            spatial_shapes,
        )

        return resized_embeddings



class RotaryPositionEmbedder(nn.Module):
    """
    Rotary Position Embeddings (RoPE) for 4D coordinates (t, x, y, z).

    Args:
        head_dim: Dimension of each attention head
        pos_cls_token: Position for class token (not used)
    """

    def __init__(self, head_dim: int, pos_cls_token: int = 0):
        super().__init__()
        self.head_dim = head_dim
        self.dim_rope = head_dim // 8  # Each of 4 dims gets head_dim/8 features

        # Precompute frequency bases (not persisted in state_dict)
        self.register_buffer(
            "freqs",
            1.0 / (10000.0 ** (torch.arange(0, self.dim_rope).float() / head_dim)),
            persistent=False
        )

    def compute_freqs_cis(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute RoPE frequencies for given positions.

        Args:
            positions: [..., 4] tensor containing [t, x, y, z] positions

        Returns:
            freqs_cis: [..., 4*dim_rope] complex tensor
        """
        # Ensure freqs is on same device
        freqs = self.freqs.to(positions.device)

        # Flatten positions for outer product
        original_shape = positions.shape[:-1]
        positions_flat = positions.reshape(-1, 4)  # [N, 4]

        # Calculate frequencies for each dimension
        t_freq = torch.outer(positions_flat[:, 0].float(), freqs)  # [N, dim_rope]
        h_freq = torch.outer(positions_flat[:, 1].float(), freqs)  # [N, dim_rope]
        w_freq = torch.outer(positions_flat[:, 2].float(), freqs)  # [N, dim_rope]
        z_freq = torch.outer(positions_flat[:, 3].float(), freqs)  # [N, dim_rope]

        # Convert to complex numbers
        freqs_cis_t = torch.polar(torch.ones_like(t_freq), t_freq)
        freqs_cis_h = torch.polar(torch.ones_like(h_freq), h_freq)
        freqs_cis_w = torch.polar(torch.ones_like(w_freq), w_freq)
        freqs_cis_z = torch.polar(torch.ones_like(z_freq), z_freq)

        # Concatenate all dimensions: [N, 4*dim_rope]
        freqs_cis = torch.cat([freqs_cis_t, freqs_cis_h, freqs_cis_w, freqs_cis_z], dim=-1)

        # Reshape back to original shape
        freqs_cis = freqs_cis.reshape(*original_shape, -1)

        return freqs_cis

    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
        xk_freqs_cis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            xq: Query tensor [..., head_dim]
            xk: Key tensor [..., head_dim]
            freqs_cis: Frequency tensor for queries [..., rope_dim]
            xk_freqs_cis: Frequency tensor for keys (default: same as queries)

        Returns:
            (xq_out, xk_out): Rotated query and key tensors
        """
        if xk_freqs_cis is None:
            xk_freqs_cis = freqs_cis

        # Reshape to pairs for complex number representation
        reshape_xq = xq.to(torch.float32).reshape(*xq.shape[:-1], -1, 2)
        reshape_xk = xk.to(torch.float32).reshape(*xk.shape[:-1], -1, 2)

        # Convert to complex
        xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
        xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])

        # Add head dimension if needed
        if freqs_cis.dim() < xq_.dim():
            freqs_cis = freqs_cis.unsqueeze(-2)
            xk_freqs_cis = xk_freqs_cis.unsqueeze(-2)

        # Apply rotation
        xq_out = xq_ * freqs_cis
        xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1)

        xk_out = xk_ * xk_freqs_cis
        xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1)

        return xq_out.to(xq.dtype), xk_out.to(xk.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        indices: torch.Tensor,
        k_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q: Query tensor [..., head_dim]
            k: Key tensor [..., head_dim]
            indices: Position indices for queries [..., 4]
            k_indices: Position indices for keys (default: same as queries)

        Returns:
            (q_embed, k_embed): Embedded query and key tensors
        """
        # Compute frequencies
        q_freqs = self.compute_freqs_cis(indices)

        if k_indices is None:
            k_freqs = q_freqs
        else:
            k_freqs = self.compute_freqs_cis(k_indices)

        # Apply rotation
        q_embed, k_embed = self.apply_rotary_emb(q, k, freqs_cis=q_freqs, xk_freqs_cis=k_freqs)

        return q_embed, k_embed

    def forward_concatenated(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_indices: torch.Tensor,
        k_indices: torch.Tensor,
        q_seqlens: list,
        k_seqlens: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to concatenated query and key tensors.

        This is the efficient path for concatenated format - no need to split by batch.

        Args:
            q: Query tensor [N, H, D] where N = sum(q_seqlens)
            k: Key tensor [N_k, H, D] where N_k = sum(k_seqlens)
            q_indices: Position indices for queries [N, 4]
            k_indices: Position indices for keys [N_k, 4]
            q_seqlens: List[int] sequence lengths for queries
            k_seqlens: List[int] sequence lengths for keys

        Returns:
            (q_embed, k_embed): Embedded query and key tensors
        """
        # Compute frequencies for all positions at once
        q_freqs = self.compute_freqs_cis(q_indices)  # [N, 4*dim_rope]
        k_freqs = self.compute_freqs_cis(k_indices)  # [N_k, 4*dim_rope]

        # Apply rotation
        # q shape: [N, H, D], freqs shape: [N, 4*dim_rope]
        # Need to unsqueeze freqs to [N, 1, 4*dim_rope] for broadcasting
        q_embed, k_embed = self.apply_rotary_emb(
            q, k,
            freqs_cis=q_freqs.unsqueeze(1),  # [N, 1, rope_dim]
            xk_freqs_cis=k_freqs.unsqueeze(1)  # [N_k, 1, rope_dim]
        )

        return q_embed, k_embed


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    coords: torch.Tensor,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to apply RoPE.

    Args:
        q: Query tensor [..., num_heads, head_dim]
        k: Key tensor [..., num_heads, head_dim]
        coords: Coordinate tensor [..., 4] with [t, x, y, z]
        head_dim: Dimension of attention heads

    Returns:
        (q_rot, k_rot): Rotated query and key tensors
    """
    embedder = RotaryPositionEmbedder(head_dim)
    embedder.freqs = embedder.freqs.to(q.device)

    # coords shape: [..., 4]
    # q/k shape: [..., num_heads, head_dim]
    # Need to expand coords to match q/k
    if coords.dim() < q.dim() - 1:
        # Add dimensions for heads
        for _ in range(q.dim() - coords.dim() - 1):
            coords = coords.unsqueeze(-2)

    return embedder.forward(q, k, coords)



########################################################
## Attention
########################################################



def set_attn_backend(backend: str):
    """
    Set the attention backend.

    Args:
        backend: One of "auto", "xformers", "flash_attn", "pytorch"
    """
    global ATTN_BACKEND
    valid_backends = ["auto", "xformers", "flash_attn", "pytorch"]
    if backend not in valid_backends:
        raise ValueError(f"Invalid backend: {backend}. Must be one of {valid_backends}")
    ATTN_BACKEND = backend


def get_attn_backend() -> str:
    """Get the current attention backend."""
    return ATTN_BACKEND


def dense_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Apply scaled dot-product attention to dense tensors with optional masking.

    Automatically selects the best available backend (xformers > flash_attn > PyTorch),
    or uses the backend specified by ATTN_BACKEND environment variable or set_attn_backend().

    Args:
        q: Query tensor [B, seq_len_q, num_heads, head_dim]
        k: Key tensor [B, seq_len_k, num_heads, head_dim]
        v: Value tensor [B, seq_len_k, num_heads, head_dim]
        mask: Boolean mask [B, seq_len_q] or [B, seq_len_k] (True = valid, False = padded)
        dropout_p: Dropout probability

    Returns:
        out: Output tensor [B, seq_len_q, num_heads, head_dim]
    """
    B, L_q, H, D = q.shape
    _, L_k, _, _ = k.shape

    # Determine which backend to use
    backend = ATTN_BACKEND
    if backend == "auto":
        # Auto-select: prefer xformers > flash_attn > pytorch
        if HAS_XFORMERS and q.is_cuda:
            backend = "xformers"
        elif HAS_FLASH_ATTN and mask is not None and q.is_cuda:
            backend = "flash_attn"
        else:
            backend = "pytorch"

    # Use xformers if selected and available
    if backend == "xformers" and HAS_XFORMERS and q.is_cuda:
        return _xformers_attention(q, k, v, mask, dropout_p)

    # Use flash_attn if selected and available
    elif backend == "flash_attn" and HAS_FLASH_ATTN and mask is not None and q.is_cuda:
        return _flash_attn_with_mask(q, k, v, mask, dropout_p)

    # Fall back to PyTorch
    else:
        return _pytorch_attention(q, k, v, mask, dropout_p)


def _xformers_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Use xformers memory_efficient_attention with BlockDiagonalMask.

    Args:
        q, k, v: [B, seq_len, num_heads, head_dim]
        mask: [B, seq_len] boolean mask (True = valid, False = padded)
        dropout_p: Dropout probability

    Returns:
        out: [B, seq_len, num_heads, head_dim]
    """
    B, L_q, H, D = q.shape
    _, L_k, _, _ = k.shape

    if mask is None:
        # No mask, use standard attention
        # Reshape to [B * seq_len, num_heads, head_dim] for xformers
        q_flat = q.reshape(B * L_q, H, D).unsqueeze(0)  # [1, B*L_q, H, D]
        k_flat = k.reshape(B * L_k, H, D).unsqueeze(0)  # [1, B*L_k, H, D]
        v_flat = v.reshape(B * L_k, H, D).unsqueeze(0)  # [1, B*L_k, H, D]

        out = xops.memory_efficient_attention(q_flat, k_flat, v_flat, p=dropout_p)
        out = out.squeeze(0).reshape(B, L_q, H, D)
        return out

    # With mask, use BlockDiagonalMask
    if mask.dim() == 1:
        # Same mask for all batches
        mask = mask.unsqueeze(0).expand(B, -1)

    # Compute sequence lengths from mask
    q_seqlens = mask[:, :L_q].sum(dim=1).int().tolist()  # List[int]
    k_seqlens = mask[:, :L_k].sum(dim=1).int().tolist()  # List[int]

    # Filter out padded positions and flatten
    q_flat = q.reshape(B * L_q, H, D)[mask[:, :L_q].reshape(-1)]  # [total_valid_q, H, D]
    k_flat = k.reshape(B * L_k, H, D)[mask[:, :L_k].reshape(-1)]  # [total_valid_k, H, D]
    v_flat = v.reshape(B * L_k, H, D)[mask[:, :L_k].reshape(-1)]  # [total_valid_k, H, D]

    # Add batch dimension for xformers
    q_flat = q_flat.unsqueeze(0)  # [1, total_valid_q, H, D]
    k_flat = k_flat.unsqueeze(0)  # [1, total_valid_k, H, D]
    v_flat = v_flat.unsqueeze(0)  # [1, total_valid_k, H, D]

    # Create BlockDiagonalMask
    attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlens, k_seqlens)

    # Apply attention
    out_flat = xops.memory_efficient_attention(
        q_flat, k_flat, v_flat,
        attn_bias=attn_bias,
        p=dropout_p
    )

    # Unflatten and restore original shape with padding
    out_flat = out_flat.squeeze(0)  # [total_valid_q, H, D]

    # Scatter back to original positions
    out = torch.zeros(B * L_q, H, D, dtype=q.dtype, device=q.device)
    out[mask[:, :L_q].reshape(-1)] = out_flat
    out = out.reshape(B, L_q, H, D)

    return out


def _flash_attn_with_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Use flash_attn varlen API with variable-length sequences.

    Args:
        q, k, v: [B, seq_len, num_heads, head_dim]
        mask: [B, seq_len] boolean mask
        dropout_p: Dropout probability

    Returns:
        out: [B, seq_len, num_heads, head_dim]
    """
    B, L_q, H, D = q.shape
    _, L_k, _, _ = k.shape

    # Compute sequence lengths from mask
    if mask.dim() == 1:
        # Same mask for all batches
        mask = mask.unsqueeze(0).expand(B, -1)

    q_seqlens = mask[:, :L_q].sum(dim=1).int()  # [B]
    k_seqlens = mask[:, :L_k].sum(dim=1).int()  # [B] (assume same for k/v)

    # Filter out padded positions
    # Flatten batch dimension
    q_flat = q.reshape(B * L_q, H, D)[mask[:, :L_q].reshape(-1)]  # [total_valid_q, H, D]
    k_flat = k.reshape(B * L_k, H, D)[mask[:, :L_k].reshape(-1)]  # [total_valid_k, H, D]
    v_flat = v.reshape(B * L_k, H, D)[mask[:, :L_k].reshape(-1)]  # [total_valid_k, H, D]

    # Compute cumulative sequence lengths
    cu_seqlens_q = torch.cat([
        torch.tensor([0], device=q.device, dtype=torch.int32),
        q_seqlens.cumsum(0, dtype=torch.int32)
    ])
    cu_seqlens_k = torch.cat([
        torch.tensor([0], device=k.device, dtype=torch.int32),
        k_seqlens.cumsum(0, dtype=torch.int32)
    ])

    # Call flash_attn varlen function
    out_flat = flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=q_seqlens.max().item(),
        max_seqlen_k=k_seqlens.max().item(),
        dropout_p=dropout_p,
    )  # [total_valid_q, H, D]

    # Reconstruct padded format
    out = torch.zeros(B, L_q, H, D, device=q.device, dtype=q.dtype)
    out[mask[:, :L_q]] = out_flat

    return out


def _pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Use PyTorch's scaled_dot_product_attention.

    Args:
        q, k, v: [B, seq_len, num_heads, head_dim]
        mask: [B, seq_len] boolean mask
        dropout_p: Dropout probability

    Returns:
        out: [B, seq_len, num_heads, head_dim]
    """
    B, L_q, H, D = q.shape
    _, L_k, _, _ = k.shape

    # Reshape for attention: [B, H, L, D]
    q = q.transpose(1, 2)  # [B, H, L_q, D]
    k = k.transpose(1, 2)  # [B, H, L_k, D]
    v = v.transpose(1, 2)  # [B, H, L_k, D]

    # Create attention mask if provided
    attn_mask = None
    if mask is not None:
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(B, -1)

        # Create 4D attention mask: [B, 1, L_q, L_k]
        # mask shape: [B, L] where True = valid
        q_mask = mask[:, :L_q]  # [B, L_q]
        k_mask = mask[:, :L_k]  # [B, L_k]

        # Attention mask: [B, 1, L_q, L_k]
        attn_mask = q_mask.unsqueeze(1).unsqueeze(-1) & k_mask.unsqueeze(1).unsqueeze(-2)
        attn_mask = attn_mask.float()

        # Convert to additive mask (0 for valid, -inf for invalid)
        attn_mask = torch.where(attn_mask.bool(), 0.0, float('-inf'))

    # Apply attention
    try:
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
    except Exception as e:
        # Fallback to manual implementation
        out = _manual_attention(q, k, v, attn_mask, dropout_p)

    # Reshape back: [B, H, L_q, D] -> [B, L_q, H, D]
    out = out.transpose(1, 2)

    return out


def _manual_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Manual attention implementation as fallback.

    Args:
        q, k, v: [B, H, L, D]
        attn_mask: [B, 1, L_q, L_k] additive mask
        dropout_p: Dropout probability

    Returns:
        out: [B, H, L_q, D]
    """
    # Compute attention scores
    scale = q.size(-1) ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L_q, L_k]

    # Apply mask
    if attn_mask is not None:
        attn = attn + attn_mask

    # Softmax
    attn = F.softmax(attn, dim=-1)

    # Dropout
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    # Apply to values
    out = torch.matmul(attn, v)  # [B, H, L_q, D]

    return out


# =============================================================================
# CONCATENATED FORMAT ATTENTION (for variable-length sequences without padding)
# =============================================================================

def concatenated_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlens: List[int],
    dropout_p: float = 0.0,
    k_seqlens: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Apply scaled dot-product attention to concatenated (variable-length) tensors.

    This is the EFFICIENT implementation that works directly on concatenated data
    without any padding or scatter/gather operations.

    Uses xformers BlockDiagonalMask or flash_attn varlen APIs for optimal performance.

    Args:
        q: Query tensor [N_q, num_heads, head_dim] where N_q = sum(seqlens)
        k: Key tensor [N_k, num_heads, head_dim] where N_k = sum(k_seqlens or seqlens)
        v: Value tensor [N_k, num_heads, head_dim]
        seqlens: List[int] query sequence lengths for each batch
        dropout_p: Dropout probability
        k_seqlens: Optional[List[int]] key/value sequence lengths (when K/V has cached tokens
                   prepended making it longer than Q). Defaults to seqlens if not provided.

    Returns:
        out: Output tensor [N_q, num_heads, head_dim]
    """
    N, H, D = q.shape
    device = q.device

    if k_seqlens is None:
        k_seqlens = seqlens

    # Determine which backend to use
    backend = ATTN_BACKEND
    if backend == "auto":
        if HAS_XFORMERS and q.is_cuda:
            backend = "xformers"
        elif HAS_FLASH_ATTN and q.is_cuda:
            backend = "flash_attn"
        else:
            backend = "pytorch"

    # Use xformers if selected and available
    if backend == "xformers" and HAS_XFORMERS and q.is_cuda:
        return _xformers_concatenated_attention(q, k, v, seqlens, dropout_p, k_seqlens)

    # Use flash_attn if selected and available
    elif backend == "flash_attn" and HAS_FLASH_ATTN and q.is_cuda:
        return _flash_attn_concatenated_attention(q, k, v, seqlens, dropout_p, k_seqlens)

    # Fall back to PyTorch (needs padding - slower)
    else:
        return _pytorch_concatenated_attention(q, k, v, seqlens, dropout_p, k_seqlens)


def _xformers_concatenated_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlens: List[int],
    dropout_p: float = 0.0,
    k_seqlens: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Use xformers memory_efficient_attention with BlockDiagonalMask.

    This is the most efficient path - no padding, no scatter/gather.

    Args:
        q, k, v: [N, num_heads, head_dim] concatenated tensors
        seqlens: List[int] query sequence lengths
        dropout_p: Dropout probability
        k_seqlens: Optional[List[int]] key/value sequence lengths (for KV cache)

    Returns:
        out: [N_q, num_heads, head_dim]
    """
    if k_seqlens is None:
        k_seqlens = seqlens

    # Add batch dimension for xformers: [N, H, D] -> [1, N, H, D]
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)

    # Create BlockDiagonalMask from sequence lengths
    attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(seqlens, k_seqlens)

    # Apply attention
    out = xops.memory_efficient_attention(
        q, k, v,
        attn_bias=attn_bias,
        p=dropout_p
    )

    # Remove batch dimension: [1, N, H, D] -> [N, H, D]
    return out.squeeze(0)


def _flash_attn_concatenated_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlens: List[int],
    dropout_p: float = 0.0,
    k_seqlens: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Use flash_attn varlen API for concatenated sequences.

    Args:
        q, k, v: [N, num_heads, head_dim] concatenated tensors
        seqlens: List[int] query sequence lengths
        dropout_p: Dropout probability
        k_seqlens: Optional[List[int]] key/value sequence lengths (for KV cache)

    Returns:
        out: [N_q, num_heads, head_dim]
    """
    if k_seqlens is None:
        k_seqlens = seqlens

    device = q.device

    # Compute cumulative query sequence lengths
    q_seqlens_tensor = torch.tensor(seqlens, device=device, dtype=torch.int32)
    cu_seqlens_q = torch.cat([
        torch.tensor([0], device=device, dtype=torch.int32),
        q_seqlens_tensor.cumsum(0, dtype=torch.int32)
    ])

    # Compute cumulative key/value sequence lengths
    k_seqlens_tensor = torch.tensor(k_seqlens, device=device, dtype=torch.int32)
    cu_seqlens_k = torch.cat([
        torch.tensor([0], device=device, dtype=torch.int32),
        k_seqlens_tensor.cumsum(0, dtype=torch.int32)
    ])

    max_seqlen_q = max(seqlens)
    max_seqlen_k = max(k_seqlens)

    # Call flash_attn varlen function
    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=dropout_p,
    )

    return out


def _pytorch_concatenated_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlens: List[int],
    dropout_p: float = 0.0,
    k_seqlens: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    PyTorch fallback for concatenated attention (requires padding).

    This is slower than xformers/flash_attn but works on any device.

    Args:
        q, k, v: [N, num_heads, head_dim]
        seqlens: List[int] query sequence lengths
        dropout_p: Dropout probability
        k_seqlens: Optional[List[int]] key/value sequence lengths (for KV cache)

    Returns:
        out: [N_q, num_heads, head_dim]
    """
    if k_seqlens is None:
        k_seqlens = seqlens

    N_q, H, D = q.shape
    B = len(seqlens)
    max_L_q = max(seqlens)
    max_L_k = max(k_seqlens)
    device = q.device
    dtype = q.dtype

    # Pad Q to [B, max_L_q, H, D]
    q_dense = torch.zeros(B, max_L_q, H, D, dtype=dtype, device=device)
    q_mask = torch.zeros(B, max_L_q, dtype=torch.bool, device=device)
    offset = 0
    for b, seqlen in enumerate(seqlens):
        q_dense[b, :seqlen] = q[offset:offset + seqlen]
        q_mask[b, :seqlen] = True
        offset += seqlen

    # Pad K/V to [B, max_L_k, H, D]
    k_dense = torch.zeros(B, max_L_k, H, D, dtype=dtype, device=device)
    v_dense = torch.zeros(B, max_L_k, H, D, dtype=dtype, device=device)
    kv_mask = torch.zeros(B, max_L_k, dtype=torch.bool, device=device)
    offset = 0
    for b, seqlen in enumerate(k_seqlens):
        k_dense[b, :seqlen] = k[offset:offset + seqlen]
        v_dense[b, :seqlen] = v[offset:offset + seqlen]
        kv_mask[b, :seqlen] = True
        offset += seqlen

    # Reshape for attention: [B, max_L, H, D] -> [B, H, max_L, D]
    q_dense = q_dense.transpose(1, 2)
    k_dense = k_dense.transpose(1, 2)
    v_dense = v_dense.transpose(1, 2)

    # Create attention mask [B, 1, max_L_q, max_L_k]
    attn_mask = q_mask.unsqueeze(1).unsqueeze(-1) & kv_mask.unsqueeze(1).unsqueeze(2)
    attn_mask = torch.where(attn_mask, 0.0, float('-inf'))

    # Apply attention
    try:
        out_dense = F.scaled_dot_product_attention(
            q_dense, k_dense, v_dense,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
    except Exception:
        out_dense = _manual_attention(q_dense, k_dense, v_dense, attn_mask, dropout_p)

    # Reshape back: [B, H, max_L_q, D] -> [B, max_L_q, H, D]
    out_dense = out_dense.transpose(1, 2)

    # Gather back to concatenated format
    out = torch.zeros(N_q, H, D, dtype=dtype, device=device)
    offset = 0
    for b, seqlen in enumerate(seqlens):
        out[offset:offset + seqlen] = out_dense[b, :seqlen]
        offset += seqlen

    return out




class MultiHeadRMSNorm(nn.Module):
    """RMS Normalization for multi-head tensors."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [..., heads, dim]"""
        x_type = x.dtype
        x = x.float()
        x = F.normalize(x, dim=-1)
        return (x * self.gamma * self.scale).to(x_type)


class DenseMultiHeadAttention(nn.Module):
    """
    Multi-head attention working with plain tensors.

    Accepts feats, coords, mask separately instead of wrapped objects.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: str = "full",
        use_bias: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        out_channels: Optional[int] = None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = channels

        assert channels % num_heads == 0
        assert out_channels % num_heads == 0
        assert type in ["self", "cross"]

        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        # Linear projections
        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=use_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=use_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=use_bias)

        # QK normalization
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        # Output projection
        self.to_out = nn.Linear(channels, out_channels, bias=use_bias)

        # RoPE
        if use_rope:
            self.rope = RotaryPositionEmbedder(self.head_dim)

    def forward(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        context_feats: Optional[torch.Tensor] = None,
        context_coords: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        kv_cache_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with plain tensor inputs.

        Args:
            feats: [B, L, C] feature tensor
            coords: [B, L, 5] coordinate tensor
            mask: [B, L] boolean mask
            context_feats/coords/mask: For cross-attention
            kv_cache: Existing KV cache
            kv_cache_size: Number of timesteps to cache

        Returns:
            (output_feats, updated_kv_cache)
        """
        if kv_cache is None:
            kv_cache = {}
        if kv_cache_size is None:
            kv_cache_size = 0

        B, L, C = feats.shape

        # Compute Q, K, V
        if self._type == "self":
            qkv = self.to_qkv(feats)  # [B, L, 3*C]
            qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)  # Each: [B, L, H, D]
            k_coords = coords
            v_coords = coords
            k_mask = mask
            v_mask = mask
        else:
            # Cross-attention
            assert context_feats is not None
            q = self.to_q(feats)
            kv = self.to_kv(context_feats)

            q = q.reshape(B, -1, self.num_heads, self.head_dim)
            kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(dim=2)

            k_coords = context_coords
            v_coords = context_coords
            k_mask = context_mask
            v_mask = context_mask

        # Apply QK normalization (before RoPE, before cache)
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        # Update cache BEFORE RoPE
        new_cache = {}
        if kv_cache_size > 0:
            new_cache = self._update_kv_cache(
                k, v, k_coords, v_coords, k_mask, v_mask, kv_cache_size
            )

        # Prepare KV: concatenate cached + current
        k, v, k_coords, v_coords, k_mask, v_mask = self._prepare_kv_cache(
            k, v, k_coords, v_coords, k_mask, v_mask, kv_cache
        )

        # Apply RoPE if enabled
        if self.use_rope:
            q_indices = coords[:, :, 1:]  # [B, L_q, 4]
            k_indices = k_coords[:, :, 1:]  # [B, L_k, 4]
            q, k = self.rope(q, k, q_indices, k_indices)

        # Apply attention
        out = dense_scaled_dot_product_attention(q, k, v, mask=k_mask)  # [B, L_q, H, D]

        # Output projection
        out = out.reshape(B, -1, self.channels)
        out = self.to_out(out)

        return out, new_cache

    def _prepare_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_coords: torch.Tensor,
        v_coords: torch.Tensor,
        k_mask: torch.Tensor,
        v_mask: torch.Tensor,
        kv_cache: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepend cached K/V to current K/V."""
        cached_k = kv_cache.get("cached_k")
        cached_v = kv_cache.get("cached_v")

        if cached_k is not None and cached_v is not None:
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
            k_coords = torch.cat([kv_cache["cached_k_coords"], k_coords], dim=1)
            v_coords = torch.cat([kv_cache["cached_v_coords"], v_coords], dim=1)
            k_mask = torch.cat([kv_cache["cached_k_mask"], k_mask], dim=1)
            v_mask = torch.cat([kv_cache["cached_v_mask"], v_mask], dim=1)

        return k, v, k_coords, v_coords, k_mask, v_mask

    def _update_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_coords: torch.Tensor,
        v_coords: torch.Tensor,
        k_mask: torch.Tensor,
        v_mask: torch.Tensor,
        kv_cache_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Extract last kv_cache_size timesteps for caching."""
        if kv_cache_size <= 0:
            return {}

        seq_len = k.shape[1]
        if seq_len <= kv_cache_size:
            # Keep everything
            return {
                "cached_k": k.detach(),
                "cached_v": v.detach(),
                "cached_k_coords": k_coords.detach(),
                "cached_v_coords": v_coords.detach(),
                "cached_k_mask": k_mask.detach(),
                "cached_v_mask": v_mask.detach(),
            }
        else:
            # Keep last kv_cache_size positions
            return {
                "cached_k": k[:, -kv_cache_size:].detach(),
                "cached_v": v[:, -kv_cache_size:].detach(),
                "cached_k_coords": k_coords[:, -kv_cache_size:].detach(),
                "cached_v_coords": v_coords[:, -kv_cache_size:].detach(),
                "cached_k_mask": k_mask[:, -kv_cache_size:].detach(),
                "cached_v_mask": v_mask[:, -kv_cache_size:].detach(),
            }


class ConcatenatedMultiHeadAttention(nn.Module):
    """
    Multi-head attention working directly on concatenated format.

    This is the EFFICIENT implementation that avoids dense conversion.
    Data stays in [N, C] format with seqlens tracking batch boundaries.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: str = "full",
        use_bias: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        out_channels: Optional[int] = None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = channels

        assert channels % num_heads == 0
        assert out_channels % num_heads == 0
        assert type in ["self", "cross"]

        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        # Linear projections
        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=use_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=use_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=use_bias)

        # QK normalization
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        # Output projection
        self.to_out = nn.Linear(channels, out_channels, bias=use_bias)

        # RoPE
        if use_rope:
            self.rope = RotaryPositionEmbedder(self.head_dim)

    def forward(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        seqlens: List[int],
        context_feats: Optional[torch.Tensor] = None,
        context_coords: Optional[torch.Tensor] = None,
        context_seqlens: Optional[List[int]] = None,
        kv_cache: Optional[Dict] = None,
        kv_cache_size: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with concatenated tensor inputs.

        Args:
            feats: [N, C] concatenated feature tensor
            coords: [N, 5] concatenated coordinate tensor
            seqlens: List[int] sequence lengths per batch
            context_feats/coords/seqlens: For cross-attention
            kv_cache: Optional existing KV cache dict with keys
                      "cached_k", "cached_v", "cached_coords", "cached_seqlens"
            kv_cache_size: Number of temporal frames to cache (0 = disabled)

        Returns:
            (output_feats, new_cache): [N, out_channels], cache dict
        """
        if kv_cache is None:
            kv_cache = {}

        N, C = feats.shape

        # Compute Q, K, V
        if self._type == "self":
            qkv = self.to_qkv(feats)  # [N, 3*C]
            qkv = qkv.reshape(N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=1)  # Each: [N, H, D]
            k_coords = coords
            k_seqlens = seqlens
        else:
            # Cross-attention
            assert context_feats is not None
            q = self.to_q(feats)
            kv = self.to_kv(context_feats)

            q = q.reshape(N, self.num_heads, self.head_dim)  # [N, H, D]
            kv = kv.reshape(-1, 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(dim=1)  # Each: [N_ctx, H, D]

            k_coords = context_coords
            k_seqlens = context_seqlens

        # Apply QK normalization
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        # Update KV cache BEFORE RoPE (store current K/V for next chunk)
        new_cache = {}
        if kv_cache_size > 0:
            new_cache = self._update_kv_cache_concatenated(k, v, k_coords, k_seqlens, kv_cache_size)

        # Prepend cached K/V to current K/V → get extended k_seqlens
        k, v, k_coords, k_seqlens = self._prepare_kv_cache_concatenated(
            k, v, k_coords, k_seqlens, kv_cache
        )

        # Apply RoPE if enabled
        if self.use_rope:
            q_indices = coords[:, 1:]  # [N, 4]
            k_indices = k_coords[:, 1:]  # [N_k, 4]
            q, k = self.rope.forward_concatenated(q, k, q_indices, k_indices, seqlens, k_seqlens)

        # Apply attention using efficient concatenated implementation
        out = concatenated_scaled_dot_product_attention(
            q, k, v, seqlens, k_seqlens=k_seqlens
        )  # [N, H, D]

        # Output projection
        out = out.reshape(N, self.channels)
        out = self.to_out(out)

        return out, new_cache

    def _update_kv_cache_concatenated(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        coords: torch.Tensor,
        seqlens: List[int],
        kv_cache_size: int,
    ) -> Dict:
        """
        Build new KV cache from current K/V, keeping last kv_cache_size temporal frames.

        Args:
            k: [N_k, H, D] key tensor
            v: [N_k, H, D] value tensor
            coords: [N_k, 5] coordinate tensor
            seqlens: List[int] sequence lengths per batch
            kv_cache_size: Number of temporal frames to retain

        Returns:
            cache dict with "cached_k", "cached_v", "cached_coords", "cached_seqlens"
        """
        device = k.device
        cached_k_list = []
        cached_v_list = []
        cached_coords_list = []
        cached_seqlens = []

        offset = 0
        for seqlen in seqlens:
            batch_k = k[offset:offset + seqlen]        # [s, H, D]
            batch_v = v[offset:offset + seqlen]        # [s, H, D]
            batch_coords = coords[offset:offset + seqlen]  # [s, 5]

            if seqlen > 0:
                # Find max temporal coordinate in this batch
                max_t = batch_coords[:, 1].max().item()
                # Keep tokens where t >= max_t - kv_cache_size + 1
                keep_mask = batch_coords[:, 1] >= (max_t - kv_cache_size + 1)
                cached_k_list.append(batch_k[keep_mask].detach())
                cached_v_list.append(batch_v[keep_mask].detach())
                cached_coords_list.append(batch_coords[keep_mask].detach())
                cached_seqlens.append(keep_mask.sum().item())
            else:
                cached_k_list.append(torch.zeros(0, k.shape[1], k.shape[2], dtype=k.dtype, device=device))
                cached_v_list.append(torch.zeros(0, v.shape[1], v.shape[2], dtype=v.dtype, device=device))
                cached_coords_list.append(torch.zeros(0, 5, dtype=coords.dtype, device=device))
                cached_seqlens.append(0)

            offset += seqlen

        # Concatenate non-empty entries
        non_empty_k = [t for t in cached_k_list if t.shape[0] > 0]
        non_empty_v = [t for t in cached_v_list if t.shape[0] > 0]
        non_empty_c = [t for t in cached_coords_list if t.shape[0] > 0]

        if non_empty_k:
            cat_k = torch.cat(non_empty_k, dim=0)
            cat_v = torch.cat(non_empty_v, dim=0)
            cat_c = torch.cat(non_empty_c, dim=0)
        else:
            H, D = k.shape[1], k.shape[2]
            cat_k = torch.zeros(0, H, D, dtype=k.dtype, device=device)
            cat_v = torch.zeros(0, H, D, dtype=v.dtype, device=device)
            cat_c = torch.zeros(0, 5, dtype=coords.dtype, device=device)

        return {
            "cached_k": cat_k,
            "cached_v": cat_v,
            "cached_coords": cat_c,
            "cached_seqlens": cached_seqlens,
        }

    def _prepare_kv_cache_concatenated(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        coords: torch.Tensor,
        seqlens: List[int],
        kv_cache: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Prepend cached K/V tokens to current K/V.

        Args:
            k: [N_k, H, D] current keys
            v: [N_k, H, D] current values
            coords: [N_k, 5] current coords
            seqlens: List[int] current sequence lengths
            kv_cache: Dict with "cached_k", "cached_v", "cached_coords", "cached_seqlens"

        Returns:
            (k, v, coords, k_seqlens): Extended K/V with cache prepended
        """
        cached_k = kv_cache.get("cached_k")
        if cached_k is None or cached_k.shape[0] == 0:
            return k, v, coords, seqlens

        cached_v = kv_cache["cached_v"]
        cached_coords = kv_cache["cached_coords"]
        cached_seqlens = kv_cache["cached_seqlens"]

        # Build extended k_seqlens = [cache_s_i + curr_s_i, ...]
        new_k_seqlens = [cs + s for cs, s in zip(cached_seqlens, seqlens)]

        # Concatenate cache before current along N dimension
        k_ext = torch.cat([cached_k, k], dim=0)
        v_ext = torch.cat([cached_v, v], dim=0)
        coords_ext = torch.cat([cached_coords, coords], dim=0)

        return k_ext, v_ext, coords_ext, new_k_seqlens


########################################################
## Tranformer Blocks
########################################################


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Args:
        hidden_size: Size of hidden dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize using RMS."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor [..., hidden_size]

        Returns:
            Normalized tensor [..., hidden_size]
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class RMSNorm32(RMSNorm):
    """
    RMSNorm that converts to float32 before forward pass.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        output = output * self.weight.float()
        return output.to(x.dtype)


class LayerNorm32(nn.LayerNorm):
    """
    LayerNorm that converts to float32 before forward pass.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.to(x.dtype)


class GroupNorm32(nn.GroupNorm):
    """
    GroupNorm that converts to float32 before forward pass.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)




class ConcatenatedMLP(nn.Module):
    """
    MLP block for concatenated format: Linear -> Activation -> Linear.

    Uses nn.Sequential for weight compatibility with original checkpoints.
    Works on [N, C] tensors directly.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        activation: nn.Module = nn.GELU(approximate="tanh"),
        use_bias: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features

        # Use Sequential for weight compatibility
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=use_bias),
            activation,
            nn.Linear(hidden_features, out_features, bias=use_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, C] concatenated tensor

        Returns:
            [N, C] tensor
        """
        return self.mlp(x)


class ConcatenatedTransformerBlock(nn.Module):
    """
    Transformer block for concatenated format.

    Works directly on [N, C] tensors with seqlens - NO padding or dense conversion.

    Args:
        channels: Feature dimension
        num_heads: Number of attention heads
        mlp_channels: MLP hidden dimension
        attn_mode: Attention mode (default: "full")
        use_bias: Whether to use bias in linear layers
        use_rope: Whether to use RoPE
        qk_rms_norm: Whether to normalize Q and K
        use_rms_norm: Use RMSNorm instead of LayerNorm
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_channels: int,
        attn_mode: str = "full",
        use_bias: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        use_rms_norm: bool = True,
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.mlp_channels = mlp_channels

        # Normalization layers
        if use_rms_norm:
            self.norm1 = RMSNorm(channels)
            self.norm2 = RMSNorm(channels)
        else:
            self.norm1 = LayerNorm32(channels)
            self.norm2 = LayerNorm32(channels)

        # Attention (concatenated version)
        self.attn = ConcatenatedMultiHeadAttention(
            channels=channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            use_bias=use_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )

        # MLP (works on any shape ending in C)
        self.mlp = ConcatenatedMLP(
            in_features=channels,
            hidden_features=mlp_channels,
            use_bias=use_bias,
        )

    def forward(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        seqlens: List[int],
        kv_cache: Optional[Dict] = None,
        kv_cache_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], Dict]:
        """
        Forward pass with concatenated tensor inputs.

        Args:
            feats: [N, C] concatenated feature tensor
            coords: [N, 5] concatenated coordinate tensor
            seqlens: List[int] sequence lengths per batch
            kv_cache: Optional existing KV cache for this block
            kv_cache_size: Number of temporal frames to cache (0 = disabled)

        Returns:
            (output_feats, coords, seqlens, new_cache)
        """
        # Attention block with residual
        h = self.norm1(feats)
        attn_out, new_cache = self.attn(
            h, coords, seqlens,
            kv_cache=kv_cache,
            kv_cache_size=kv_cache_size,
        )
        feats = feats + attn_out

        # MLP block with residual
        h = self.norm2(feats)
        mlp_out = self.mlp(h)
        feats = feats + mlp_out

        return feats, coords, seqlens, new_cache
