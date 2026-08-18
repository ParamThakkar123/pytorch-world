import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Any, Optional, Dict, Tuple, Literal
import numpy as np
from dataclasses import dataclass

from torchwm.configs.serialization import SerializableConfigMixin
from torchwm.models.genie import Genie
from torchwm.models.model_io import save_config_next_to_checkpoint
from torchwm.utils.memory_utils import enable_performance_defaults


@dataclass
class GenieConfig(SerializableConfigMixin):
    """Configuration for Genie training."""

    num_frames: int = 16
    image_size: int = 64
    in_channels: int = 3

    tokenizer_vocab_size: int = 1024
    tokenizer_embedding_dim: int = 32
    tokenizer_encoder_dim: int = 512
    tokenizer_decoder_dim: int = 1024
    tokenizer_encoder_depth: int = 12
    tokenizer_decoder_depth: int = 20

    action_vocab_size: int = 8
    action_embedding_dim: int = 32
    action_encoder_dim: int = 1024
    action_encoder_depth: int = 20
    action_pooling: Literal["mean", "windowed_attention"] = "mean"
    window_attention_heads: int = 1

    dynamics_dim: int = 512
    dynamics_depth: int = 8
    dynamics_num_heads: int = 8

    batch_size: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 5000
    max_steps: int = 125000

    mask_prob_min: float = 0.5
    mask_prob_max: float = 1.0

    sample_temperature: float = 2.0
    maskgit_steps: int = 25


class VideoDataset(Dataset):
    """Dataset for video data."""

    def __init__(
        self, video_paths: list, num_frames: int = 16, image_size: int = 64
    ) -> None:
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        raise NotImplementedError("Implement loading video from paths")


class GenieTrainer:
    """Trainer for Genie model."""

    def __init__(
        self,
        model: nn.Module,
        config: GenieConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = self._create_scheduler()

        if getattr(config, "perf_defaults", True):
            enable_performance_defaults(tf32=bool(getattr(config, "tf32", True)))

        self.use_amp = bool(getattr(config, "use_amp", False))
        self._amp_device = getattr(self.device, "type", str(self.device))
        # bf16 carries fp32's dynamic range, so it needs no loss scaling; fp16
        # does. Only build a scaler for the case that actually requires one.
        self._amp_dtype = (
            torch.bfloat16
            if self._amp_device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            self._amp_device,
            enabled=self.use_amp and self._amp_dtype is torch.float16,
        )

        self.global_step = 0

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Create learning rate scheduler with warmup and cosine decay."""
        warmup_steps = self.config.warmup_steps
        max_steps = self.config.max_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                return 0.5 * (1.0 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor | float | None]:
        """Single training step.

        Args:
            batch: (B, C, T, H, W) video batch

        Returns:
            Dictionary of losses
        """
        self.model.train()
        batch = batch.to(self.device)

        B, C, T, H, W = batch.shape
        mask_prob = (
            torch.rand(1).item()
            * (self.config.mask_prob_max - self.config.mask_prob_min)
            + self.config.mask_prob_min
        )

        with torch.amp.autocast(
            device_type=self._amp_device,
            dtype=self._amp_dtype,
            enabled=self.use_amp,
        ):
            outputs = self.model(batch, mask_prob=mask_prob)

        recon_loss = outputs.get("recon_loss", 0.0)
        vq_loss = outputs.get("vq_loss", 0.0)

        dynamics_loss = outputs.get(
            "dynamics_loss", torch.tensor(0.0, device=self.device)
        )

        z_q_for_dynamics = outputs.get("z_q_for_dynamics", None)
        z_q_for_dynamics_mean = (
            z_q_for_dynamics.mean().item()
            if isinstance(z_q_for_dynamics, torch.Tensor)
            else None
        )

        total_loss = outputs["total_loss"]

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        # Gradients must be unscaled before clipping, or the norm is measured on
        # scaled values and the clip threshold means nothing.
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        self.global_step += 1

        # Normalize all returned metrics to torch.Tensor for consistent typing
        def as_tensor(x: Any) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            try:
                return torch.tensor(float(x))
            except Exception:
                return torch.tensor(float("nan"))

        return {
            "total_loss": as_tensor(total_loss),
            "recon_loss": as_tensor(recon_loss),
            "vq_loss": as_tensor(vq_loss),
            "dynamics_loss": as_tensor(dynamics_loss),
            "learning_rate": torch.tensor(float(self.scheduler.get_last_lr()[0])),
            "z_q_for_dynamics_mean": as_tensor(z_q_for_dynamics_mean),
        }

    def validate(self, val_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Validation step.

        Args:
            val_batch: (B, C, T, H, W) validation video batch

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(val_batch, mask_prob=0.0)
            recon_loss = outputs["tokenizer_loss"].get("recon_loss", 0.0)
            return {
                "val_recon_loss": recon_loss.detach().cpu()
                if isinstance(recon_loss, torch.Tensor)
                else torch.tensor(float(recon_loss)),
            }

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_steps: Optional[int] = None,
        log_interval: int = 100,
        val_interval: int = 1000,
    ) -> None:
        """Full training loop.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            num_steps: Number of training steps (uses config.max_steps if None)
            log_interval: Logging frequency
            val_interval: Validation frequency
        """
        if num_steps is None:
            num_steps = self.config.max_steps

        train_iter = iter(train_dataloader)

        while self.global_step < num_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            batch = batch.to(self.device)

            losses = self.train_step(batch)

            if self.global_step % log_interval == 0:
                print(
                    f"Step {self.global_step}/{num_steps} | "
                    f"Loss: {losses['total_loss']:.4f} | "
                    f"Recon: {losses['recon_loss']:.4f} | "
                    f"VQ: {losses['vq_loss']:.4f} | "
                    f"Dynamics: {losses['dynamics_loss']:.4f} | "
                    f"LR: {losses['learning_rate']:.6f}"
                )

            if val_dataloader is not None and self.global_step % val_interval == 0:
                val_iter = iter(val_dataloader)
                val_batch = next(val_iter)
                val_batch = val_batch.to(self.device)
                val_metrics = self.validate(val_batch)
                print(f"Validation: {val_metrics}")

        print("Training complete!")

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        save_config_next_to_checkpoint(self.config, path)
        torch.save(
            {
                "config": self.config.to_dict(),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]


def create_genie_trainer(
    config: Optional[GenieConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[GenieTrainer, nn.Module]:
    """Factory function to create Genie trainer and model."""
    if config is None:
        config = GenieConfig()

    model = Genie(
        num_frames=config.num_frames,
        image_size=config.image_size,
        in_channels=config.in_channels,
        tokenizer_vocab_size=config.tokenizer_vocab_size,
        tokenizer_embedding_dim=config.tokenizer_embedding_dim,
        action_vocab_size=config.action_vocab_size,
        action_embedding_dim=config.action_embedding_dim,
        dynamics_dim=config.dynamics_dim,
        dynamics_depth=config.dynamics_depth,
        dynamics_num_heads=config.dynamics_num_heads,
        encoder_depth=config.tokenizer_encoder_depth,
        decoder_depth=config.tokenizer_decoder_depth,
        latent_action_depth=config.action_encoder_depth,
        action_pooling=config.action_pooling,
        window_attention_heads=config.window_attention_heads,
    )

    trainer = GenieTrainer(model, config, device)

    return trainer, model


def main(argv: Optional[list[str]] = None) -> None:
    """Console entrypoint for Genie trainer setup.

    The generic ``VideoDataset`` in this module is intentionally abstract, so this
    command provides a discoverable entrypoint for inspecting defaults and
    constructing a trainer. Use a concrete dataset script, such as
    ``scripts/train_genie_tinyworlds.py``, for end-to-end data loading.
    """
    parser = argparse.ArgumentParser(description="Prepare Genie training")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the default Genie max training steps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct the trainer and print its resolved configuration without training.",
    )
    args = parser.parse_args(argv)

    config = GenieConfig()
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    device = torch.device(args.device) if args.device else None
    trainer, _ = create_genie_trainer(config=config, device=device)

    if args.dry_run:
        print(
            "Created GenieTrainer "
            f"on {trainer.device} with max_steps={trainer.config.max_steps}"
        )
        return

    parser.error(
        "Genie requires a concrete video dataset; use --dry-run to validate "
        "trainer construction or scripts/train_genie_tinyworlds.py for an "
        "end-to-end example."
    )


if __name__ == "__main__":
    main()
