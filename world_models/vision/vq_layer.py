import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class VectorQuantizer(nn.Module):
    """Vector Quantizer for discrete autoencoder.

    Implements the VQ-VAE quantization from:
    "Neural Discrete Representation Learning" (Van Den Oord et al., 2017)

    Uses exponential moving averages for codebook updates and straight-through
    estimator for gradient flow.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        embedding_dim: int = 512,
        commitment_weight: float = 0.25,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight

        # Codebook: learnable embeddings
        self.codebook = nn.Embedding(vocab_size, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Quantize the input latents.

        Args:
            z: Input tensor of shape (B, C, H, W) or (B, C)

        Returns:
            z_q: Quantized tensor (same shape as input)
            indices: Token indices for each position (B, H, W) or (B,)
            loss: Dictionary containing VQ loss components
        """
        # Reshape for quantization
        original_shape: torch.Size = z.shape

        z_flat: torch.Tensor
        B: int
        C: int
        H: int
        W: int

        if z.dim() == 4:  # (B, C, H, W)
            B, C, H, W = z.shape
            # Flatten spatial dimensions: (B, C, H*W) -> (B, H*W, C)
            z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif z.dim() == 2:  # (B, C)
            B = z.shape[0]
            C = z.shape[1]
            H = 1
            W = 1
            z_flat = z.unsqueeze(1)  # (B, 1, C)
        else:
            raise ValueError(f"Expected 2D or 4D input, got {z.dim()}D")

        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z·e
        z_flat = z_flat.float()
        codebook: torch.Tensor = self.codebook.weight.float()

        d: torch.Tensor = (
            torch.sum(z_flat**2, dim=-1, keepdim=True)
            + torch.sum(codebook**2, dim=-1)
            - 2 * torch.matmul(z_flat, codebook.t())
        )  # (B, H*W, vocab_size)

        # Find nearest codebook entries (indices)
        indices: torch.Tensor = torch.argmin(d, dim=-1)  # (B, H*W) or (B, 1)

        # Get the quantized values
        z_q: torch.Tensor = F.embedding(indices, codebook)  # (B, H*W, C)

        # VQ-VAE loss with a gradient-trained codebook:
        #   codebook loss: pull codebook vectors toward encoder outputs
        #   commitment loss (weighted by beta): pull encoder outputs toward codebook
        codebook_loss: torch.Tensor = F.mse_loss(z_q, z_flat.detach())
        commitment_loss: torch.Tensor = F.mse_loss(z_q.detach(), z_flat)
        vq_loss: torch.Tensor = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator: forward uses z_q, backward copies gradients
        # to the encoder as if z_q == z (argmin is non-differentiable).
        z_q = z_flat + (z_q - z_flat).detach()

        # Perplexity: measure of how many codebook entries are used
        encodings: torch.Tensor = F.one_hot(indices.reshape(-1), self.vocab_size).float()
        avg_probs: torch.Tensor = torch.mean(encodings, dim=0)
        perplexity: torch.Tensor = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        # Reshape back to original spatial dimensions
        if z.dim() == 4:
            z_q = z_q.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            z_q = z_q.squeeze(1)

        indices_reshaped: torch.Tensor = (
            indices.reshape(B, H, W) if z.dim() == 4 else indices.squeeze(-1)
        )

        loss: Dict[str, torch.Tensor] = {
            "vq_loss": vq_loss,
            "perplexity": perplexity,
        }

        return z_q, indices_reshaped, loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices back to embeddings.

        Args:
            indices: Token indices (B, H, W) or (B,)

        Returns:
            Embeddings (B, C, H, W) or (B, C)
        """
        if indices.dim() == 3:  # (B, H, W)
            B, H, W = indices.shape
            z_q = F.embedding(indices, self.codebook.weight)  # (B, H*W, C)
            z_q = z_q.permute(0, 2, 1).reshape(B, -1, H, W)
        else:
            z_q = F.embedding(indices, self.codebook.weight)

        return z_q


class VectorQuantizerEMA(nn.Module):
    """Vector Quantizer with Exponential Moving Average updates.

    Uses EMA updates for the codebook instead of gradient-based updates,
    which leads to more stable training.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        embedding_dim: int = 512,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        # Codebook
        self.codebook = nn.Embedding(vocab_size, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

        # EMA tracking
        # Annotate buffers so the type checker knows these are tensors
        self.register_buffer("ema_cluster_size", torch.zeros(vocab_size))
        self.ema_cluster_size: torch.Tensor = self.ema_cluster_size

        self.register_buffer("ema_embed_avg", self.codebook.weight.data.clone())
        self.ema_embed_avg: torch.Tensor = self.ema_embed_avg

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Quantize with EMA updates."""
        # Flatten spatial dims
        B, C, H, W = z.shape
        z_flat: torch.Tensor = z.permute(0, 2, 3, 1).reshape(B, H * W, C).float()

        # Compute distances
        codebook: torch.Tensor = self.codebook.weight.float()
        d: torch.Tensor = (
            torch.sum(z_flat**2, dim=-1, keepdim=True)
            + torch.sum(codebook**2, dim=-1)
            - 2 * torch.matmul(z_flat, codebook.t())
        )

        indices: torch.Tensor = torch.argmin(d, dim=-1)

        # Quantize: nearest codebook lookup (non-differentiable in indices)
        z_q: torch.Tensor = F.embedding(indices, codebook)  # (B, H*W, C)

        # EMA update (only during training). The codebook is updated by
        # exponential moving averages of the assigned encoder outputs (Van Den
        # Oord et al., 2017), not by gradient descent.
        if self.training:
            with torch.no_grad():
                V = self.vocab_size
                flat_enc: torch.Tensor = F.one_hot(
                    indices.reshape(-1), V
                ).float()  # (N, V)
                z_in: torch.Tensor = z_flat.reshape(-1, C)  # (N, C)

                cluster_counts = flat_enc.sum(dim=0)  # (V,)
                dw = flat_enc.t() @ z_in  # (V, C) sum of z assigned to each code

                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    cluster_counts, alpha=1 - self.ema_decay
                )
                self.ema_embed_avg.mul_(self.ema_decay).add_(
                    dw, alpha=1 - self.ema_decay
                )

                # Laplace smoothing to avoid dividing by zero for unused codes.
                n = self.ema_cluster_size.sum()
                smoothed = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + V * self.epsilon)
                    * n
                )  # (V,)
                self.codebook.weight.data.copy_(
                    self.ema_embed_avg / smoothed.unsqueeze(1)
                )

        # Reshape back to (B, C, H, W)
        z_q = z_q.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Commitment loss: pull encoder outputs toward the codebook (the codebook
        # itself is moved by EMA, so there is no gradient-based codebook loss).
        commitment_loss: torch.Tensor = F.mse_loss(z_q.detach(), z)

        # Straight-through estimator: forward returns z_q, but gradients flow to
        # the encoder as if z_q == z. Without this the encoder receives no
        # reconstruction gradient (argmin is non-differentiable).
        z_q = z + (z_q - z).detach()

        # Perplexity
        encodings_eval: torch.Tensor = F.one_hot(indices, self.vocab_size).float()
        avg_probs: torch.Tensor = torch.mean(
            encodings_eval.reshape(-1, self.vocab_size), dim=0
        )
        perplexity: torch.Tensor = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        loss: Dict[str, torch.Tensor] = {
            "vq_loss": commitment_loss * self.commitment_weight,
            "perplexity": perplexity,
        }

        return z_q, indices.reshape(B, H, W), loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices to embeddings.

        Args:
            indices: Token indices (B, H, W) or (B,)

        Returns:
            Embeddings (B, C, H, W) or (B, C)
        """
        if indices.dim() == 3:  # (B, H, W)
            B, H, W = indices.shape
            indices_flat = indices.reshape(B, -1)  # (B, H*W)
            z_q = F.embedding(indices_flat, self.codebook.weight)  # (B, H*W, C)
            z_q = z_q.permute(0, 2, 1).reshape(B, -1, H, W)
        else:
            z_q = F.embedding(indices, self.codebook.weight)

        return z_q
