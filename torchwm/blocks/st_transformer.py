import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional


class STSpatialAttention(nn.Module):
    """Spatial attention layer for spatiotemporal transformer.

    Processes video tokens by attending over spatial positions (H*W) within
    each time step independently. Captures within-frame spatial relationships.

    - Input: (B, T, N, C) -- B batches, T time steps, N spatial positions (H*W), C channels
    - Output: (B, T, N, C) -- Same shape, spatially attended features

    **Architecture**

    - QKV projection: Linear(dim, dim*3)
    - Reshape to multi-head attention format
    - Fused scaled dot-product attention (FlashAttention on supported GPUs)
    - Output projection

    Applied to video tokens of shape (B, T, N, C) to capture within-frame
    spatial structure (e.g., object positions).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # QK Normalization (as per Genie paper - improves stability at large scale)
        # Always apply LayerNorm to Q and K per-head (no runtime toggle).
        self.q_norm = nn.LayerNorm(head_dim, elementwise_affine=False, eps=1e-6)
        self.k_norm = nn.LayerNorm(head_dim, elementwise_affine=False, eps=1e-6)

        self.dropout_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C) where T is temporal dim, N is spatial dim (H*W)
        Returns:
            (B, T, N, C)
        """
        B, T, N, C = x.shape

        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # (3, B, heads, T, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply QK normalization (as per Genie paper)
        q = self.q_norm(q)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=self.scale,
        )
        # SDPA returns (B, heads, T, N, head_dim). Merging the heads back means
        # moving that axis next to head_dim: (B, T, N, heads, head_dim) -> (B, T,
        # N, C). A transpose(2, 3) would give (B, heads, N, T, head_dim), whose
        # element count also matches (B, T, N, C) -- so reshaping it raises no
        # error but silently interleaves the head, spatial and temporal axes,
        # which leaks information across time and destroys causality.
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class STTemporalAttention(nn.Module):
    """Temporal attention layer with causal masking for spatiotemporal transformer.

    Processes video tokens by attending over time steps (T) across all spatial
    positions. Uses causal masking to ensure each frame only attends to previous
    frames (important for autoregressive video generation).

    - Input: (B, T, N, C) -- B batches, T time steps, N spatial positions, C channels
    - Output: (B, T, N, C) -- Same shape, temporally attended features

    **Causal masking**

    - Frame t can only attend to frames 0...t-1
    - Prevents information leakage from future frames
    - Essential for autoregressive video generation models

    Applied after STSpatialAttention to model temporal dynamics in the Genie VideoTokenizer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # QK Normalization (as per Genie paper - improves stability at large scale)
        # Always apply LayerNorm to Q and K per-head (no runtime toggle).
        self.q_norm = nn.LayerNorm(head_dim, elementwise_affine=False, eps=1e-6)
        self.k_norm = nn.LayerNorm(head_dim, elementwise_affine=False, eps=1e-6)

        self.dropout_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C) where T is temporal dim, N is spatial dim (H*W)
            causal: whether to apply causal masking
        Returns:
            (B, T, N, C)
        """
        B, T, N, C = x.shape

        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 4, 2, 1, 5)  # (3, B, heads, N, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply QK normalization (as per Genie paper)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Causality comes from SDPA's is_causal path (FlashAttention on supported
        # GPUs), so no explicit T x T mask is built here.
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=causal,
            scale=self.scale,
        )
        # SDPA returns (B, heads, N, T, head_dim); move heads next to head_dim to
        # get (B, T, N, heads, head_dim) before merging into C.
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class STMLP(nn.Module):
    """MLP for ST-Transformer block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class STTransformerBlock(nn.Module):
    """Combined spatiotemporal transformer block with interleaved attention.

    A single block applies:

    1. Spatial attention (within each time frame)
    2. Temporal attention (across frames with causal mask)
    3. MLP projection

    The order is: x -> + SpatialAttn -> + TemporalAttn -> + MLP -> x

    This interleaved design captures both spatial structure and temporal
    dynamics efficiently, used in Genie's VideoTokenizer and DynamicsModel.

    Args:
        dim: Feature dimension (must match patch embedding dimension)
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim = dim * mlp_ratio
        drop, attn_drop: Dropout rates
        drop_path: Stochastic depth rate for drop path regularization
        norm_layer: Normalization layer class (default: nn.LayerNorm)

    **Usage in Genie**::

        # VideoTokenizer encoder (12 layers)
        encoder = STTransformer(
            num_frames=16,
            num_patches_per_frame=256,
            dim=512,
            depth=12,
            num_heads=16
        )
        encoded = encoder(tokens)  # (B, T*N, C)

        # Dynamics model decoder (24 layers)
        decoder = STTransformer(
            num_frames=16,
            num_patches_per_frame=256,
            dim=1024,
            depth=24,
            num_heads=16
        )
        decoded = decoder(tokens)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1_spatial = norm_layer(dim)
        self.norm1_temporal = norm_layer(dim)

        self.attn_spatial = STSpatialAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.attn_temporal = STTemporalAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp = STMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C) or (B, T*H*W, C)
        Returns:
            Same shape as input
        """
        if x.dim() == 3:
            # A flat (B, T*N, C) sequence cannot be split into time and space
            # without knowing one of them. Guessing T=16 silently produced a
            # wrong factorisation for every other sequence length, so require
            # the caller to reshape (STTransformer.forward already does, using
            # its configured num_patches_per_frame).
            raise ValueError(
                "STTransformerBlock expects (B, T, N, C). Got a 3D tensor of "
                f"shape {tuple(x.shape)}; reshape it with the known number of "
                "patches per frame, or call STTransformer, which does this."
            )

        B, T, N, C = x.shape

        # Spatial attention (within each time step)
        x = x + self.drop_path(self.attn_spatial(self.norm1_spatial(x)))

        # Temporal attention (across time steps, with causal mask)
        x = x + self.drop_path(self.attn_temporal(self.norm1_temporal(x)))

        # MLP (single FFW after both spatial and temporal, as per paper)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class STTransformer(nn.Module):
    """Spatiotemporal Transformer for video modeling.

    Contains L spatiotemporal blocks with interleaved spatial and temporal attention.
    """

    def __init__(
        self,
        num_frames: int = 16,
        num_patches_per_frame: int = 256,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_patches_per_frame = num_patches_per_frame
        self.gradient_checkpointing = gradient_checkpointing

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()

        self.blocks = nn.ModuleList(
            [
                STTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T*N, C) where T is num_frames, N is num_patches_per_frame
        Returns:
            (B, T*N, C)
        """
        B, T_N, C = x.shape
        T = T_N // self.num_patches_per_frame
        N = self.num_patches_per_frame

        # Reshape to (B, T, N, C) for ST-attention
        x = x.reshape(B, T, N, C)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.norm(x)

        # Reshape back to (B, T*N, C)
        x = x.reshape(B, T * N, C)

        return x


def create_st_transformer(
    num_frames: int = 16,
    patch_size: int = 4,
    img_size: int = 64,
    dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    gradient_checkpointing: bool = False,
) -> STTransformer:
    """Factory function to create an ST-Transformer."""
    num_patches_per_frame = (img_size // patch_size) ** 2

    return STTransformer(
        num_frames=num_frames,
        num_patches_per_frame=num_patches_per_frame,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        gradient_checkpointing=gradient_checkpointing,
    )
