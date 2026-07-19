#!/usr/bin/env python3
"""Training script for Genie on TinyWorlds HDF5 dataset."""

import os
import torch
from omegaconf import OmegaConf

from torchwm.configs.genie_config import GenieSmallConfig
from torchwm.training.train_genie import create_genie_trainer
from torchwm.datasets import create_tinyworlds_dataloader


def main():
    cli_cfg = OmegaConf.from_cli()

    dataset = cli_cfg.get("dataset", "SONIC")
    num_frames = int(cli_cfg.get("num_frames", 16))
    image_size = int(cli_cfg.get("image_size", 64))
    batch_size = int(cli_cfg.get("batch_size", 2))
    num_workers = int(cli_cfg.get("num_workers", 4))
    max_steps = int(cli_cfg.get("max_steps", 50000))
    log_interval = int(cli_cfg.get("log_interval", 100))
    val_interval = int(cli_cfg.get("val_interval", 1000))
    learning_rate = float(cli_cfg.get("learning_rate", 1e-4))
    cache_dir = cli_cfg.get("cache_dir", None)
    data_file = cli_cfg.get("data_file", None)
    checkpoint_dir = cli_cfg.get("checkpoint_dir", "checkpoints")
    device_str = cli_cfg.get("device", None)

    device = (
        torch.device(device_str)
        if device_str
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Using device: {device}")

    config = GenieSmallConfig()
    config.num_frames = num_frames
    config.image_size = image_size
    config.batch_size = batch_size
    config.max_steps = max_steps
    config.learning_rate = learning_rate

    print(f"Loading {dataset} dataset...")
    train_dataset, train_loader = create_tinyworlds_dataloader(
        dataset_name=dataset,
        num_frames=num_frames,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        cache_dir=cache_dir,
        download=not data_file,
        data_file=data_file,
    )

    print(f"Dataset: {len(train_dataset)} samples, {len(train_loader)} batches")

    print("Creating Genie model and trainer...")
    trainer, model = create_genie_trainer(config=config, device=device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("Starting training...")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=None,
        num_steps=max_steps,
        log_interval=log_interval,
        val_interval=val_interval,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/genie_{dataset.lower()}_final.pt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
