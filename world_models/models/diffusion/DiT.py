import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any
from einops import rearrange
from world_models.configs.dit_config import DiTConfig as Config
from world_models.models.model_io import (
    apply_config_overrides,
    coerce_config,
    module_summary,
    resolve_pretrained_file,
    save_config_next_to_checkpoint,
)
from world_models.layers.ada_ln_norm import AdaLNNormalization
from world_models.blocks.mhsa import MultiHeadSelfAttention
from world_models.models.diffusion.DDPM import DDPM
from world_models.datasets.cifar10 import make_cifar10
from world_models.datasets.imagenet1k import make_imagenet1k, make_imagefolder
from torchvision.transforms import RandomHorizontalFlip, Compose, ToTensor
from world_models.transforms.image import make_transforms
import time
from torchvision.utils import save_image
import os
from pathlib import Path


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings for diffusion conditioning.

    This function generates positional-style embeddings for diffusion timesteps,
    following the same pattern as transformer positional encodings. The embeddings
    encode the noise level (t) and are used to condition the diffusion model.

    Math:
        embedding[t] = [sin(t/10000^(2i/d)), cos(t/10000^(2i/d))] for i in [0, d/2)

    Args:
        timesteps: Tensor of integer timesteps, shape (B,) or (B, 1)
        dim: Embedding dimension (must be even)

    Returns:
        Tensor of shape (B, dim) with sinusoidal embeddings

    Usage with DiT:
        t = torch.tensor([0, 500, 1000])  # Timesteps
        emb = sinusoidal_time_embedding(t, dim=256)  # (3, 256)

        # Condition the model:
        # - Add to timestep embedding to MLP input
        # - Use AdaLN for adaptive normalization
    """
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(10000.0), half, device=timesteps.device)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class PatchEmbed(nn.Module):
    """Patchify an image into a sequence of learnable patch tokens.

    Used in Vision Transformers (ViT) and DiT to convert 2D images into
    sequences of token embeddings that can be processed by transformers.

    Process:
        1. Conv2d with kernel_size=stride=patch_size extracts non-overlapping patches
        2. Each patch is projected to embed_dim via linear layer (Conv2d)
        3. Learnable positional embeddings are added for spatial information

    Input: (B, C, H, W) images
    Output: (B, N, embed_dim) where N = (H/patch_size) * (W/patch_size)

    Args:
        img_size: Image size (assumes square), e.g., 32 for CIFAR
        patch_size: Size of each patch (typically 4, 8, or 16)
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Output dimension for each patch token

    Usage with DiT:
        patch_embed = PatchEmbed(img_size=32, patch_size=4, in_channels=3, embed_dim=256)
        tokens = patch_embed(images)  # (B, 64, 256) for 32x32 image with patch_size=4
    """

    def __init__(
        self, img_size: int, patch_size: int, in_channels: int, embed_dim: int
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.pos
        return x


class PatchUnEmbed(nn.Module):
    """Reconstruct image-like tensors from patch-token sequences.

    The inverse of `PatchEmbed`, this module reshapes token sequences into
    grids and uses transposed convolution to decode spatial outputs.
    """

    def __init__(
        self, img_size: int, patch_size: int, embed_dim: int, out_channels: int
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.ConvTranspose2d(
            embed_dim, out_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = w = self.img_size // self.patch_size
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Conditioned transformer block used inside the DiT backbone.

    Each block applies adaptive layer-normalized self-attention and MLP
    residual updates conditioned on timestep embeddings.
    """

    def __init__(
        self, d_model: int, n_heads: int, mlp_ratio: float, drop: float, t_dim: int
    ) -> None:
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = AdaLNNormalization(d_model, t_dim)
        self.norm2 = AdaLNNormalization(d_model, t_dim)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(mlp_ratio * d_model), d_model),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x, t_emb)
        attn_out = self.attn(h)
        x = x + attn_out
        h = self.norm2(x, t_emb)
        ff_out = self.ff(h)
        x = x + ff_out
        return x


class DiT(nn.Module):
    """Diffusion Transformer model for image denoising and generation.

    The module maps noisy images and timesteps to predicted noise residuals
    and also provides a classmethod training entrypoint for common datasets.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        d_model: int,
        depth: int,
        heads: int,
        drop: float = 0.0,
        t_dim: int = 256,
    ) -> None:
        super(DiT, self).__init__()
        self.t_dim = t_dim
        self.config = Config(
            IMG_SIZE=img_size,
            CHANNELS=in_channels,
            PATCH=patch_size,
            WIDTH=d_model,
            DEPTH=depth,
            HEADS=heads,
            DROP=drop,
        )
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.GELU(),
            nn.Linear(t_dim, t_dim),
        )
        self.patchify = PatchEmbed(img_size, patch_size, in_channels, d_model)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, heads, mlp_ratio=4.0, drop=drop, t_dim=t_dim)
                for _ in range(depth)
            ]
        )

        self.unpatchify = PatchUnEmbed(img_size, patch_size, d_model, in_channels)
        self.out = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)

        x = self.patchify(x)
        for block in self.transformer_blocks:
            x = block(x, t_emb)
        x = self.unpatchify(x)
        x = self.out(x)
        return x

    @classmethod
    def from_config(
        cls,
        config: Config | dict[str, Any] | str | Path | None = None,
        **overrides: Any,
    ) -> "DiT":
        """Build DiT from a config object, dict, YAML file, or YAML string."""

        args = apply_config_overrides(coerce_config(Config, config), overrides)
        return cls(
            img_size=args.IMG_SIZE,
            patch_size=args.PATCH,
            in_channels=args.CHANNELS,
            d_model=args.WIDTH,
            depth=args.DEPTH,
            heads=args.HEADS,
            drop=args.DROP,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config: Config | dict[str, Any] | str | Path | None = None,
        checkpoint_filename: str | None = None,
        config_filename: str = "config.yaml",
        repo_type: str | None = None,
        revision: str | None = None,
        map_location: str | torch.device | None = None,
        **overrides: Any,
    ) -> "DiT":
        """Load DiT weights from a local path/directory or HF Hub."""

        checkpoint_candidates = (
            (checkpoint_filename,)
            if checkpoint_filename is not None
            else ("dit_model.pth", "model.pt", "pytorch_model.bin", "checkpoint.pt")
        )
        checkpoint_path = resolve_pretrained_file(
            pretrained_model_name_or_path,
            checkpoint_candidates,
            repo_type=repo_type,
            revision=revision,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find a DiT checkpoint for {pretrained_model_name_or_path!r}."
            )
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location or "cpu", weights_only=True
        )
        checkpoint_config = (
            checkpoint.get("config") if isinstance(checkpoint, dict) else None
        )
        if config is None and isinstance(checkpoint_config, dict):
            args = Config.from_dict(checkpoint_config)
        elif config is None:
            config_path = resolve_pretrained_file(
                pretrained_model_name_or_path,
                (config_filename, "dit_config.yaml", "config.yml"),
                repo_type=repo_type,
                revision=revision,
            )
            if config_path is None:
                raise FileNotFoundError(
                    "No config was provided and no config YAML was found beside "
                    f"{pretrained_model_name_or_path!r}."
                )
            args = Config.from_yaml(config_path)
        else:
            args = coerce_config(Config, config)
        model = cls.from_config(apply_config_overrides(args, overrides))
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                "model_state_dict", checkpoint.get("state_dict", checkpoint)
            )
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, path: str | Path) -> None:
        """Save DiT weights and config in a from_pretrained-compatible format."""

        checkpoint_path = Path(path)
        if checkpoint_path.suffix == "":
            checkpoint_path = checkpoint_path / "dit_model.pth"
        save_config_next_to_checkpoint(self.config, checkpoint_path)
        torch.save(
            {"config": self.config.to_dict(), "model_state_dict": self.state_dict()},
            checkpoint_path,
        )

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            param.numel()
            for param in self.parameters()
            if not trainable_only or param.requires_grad
        )

    def summary(self) -> dict[str, Any]:
        return module_summary(
            {
                "t_mlp": self.t_mlp,
                "patchify": self.patchify,
                "transformer_blocks": self.transformer_blocks,
                "unpatchify": self.unpatchify,
            }
        )

    @classmethod
    def train(  # type: ignore[override]
        cls,
        epochs: int,
        dataset: Any,
        batch_size: int = 128,
        lr: float = 2e-4,
        img_size: int = 32,
        channels: int = 3,
        patch: int = 4,
        width: int = 384,
        depth: int = 6,
        heads: int = 6,
        drop: float = 0.1,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        ema: bool = True,
        ema_decay: float = 0.999,
        workdir: str = "./dit_demo",
        root_path: str = "./data",
        image_folder: str | None = None,
        crop_size: int = 224,
        download: bool = True,
        copy_data: bool = False,
        subset_file: str | None = None,
        val_split: float | None = None,
    ) -> None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            print("WARNING: CUDA not available, using CPU")

        if dataset.lower() == "cifar10":
            transform = Compose([RandomHorizontalFlip(), ToTensor()])
        else:
            transform = make_transforms(
                crop_size=crop_size,
                crop_scale=(0.3, 1.0),
                color_jitter=0.5,
                horizontal_flip=True,
                color_distortion=True,
                gaussian_blur=True,
                normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )

        if dataset.lower() == "cifar10":
            _, train_loader, _ = make_cifar10(
                transform=transform,
                batch_size=batch_size,
                collator=None,
                pin_mem=True,
                num_workers=4,
                world_size=1,
                rank=0,
                root_path=root_path,
                drop_last=True,
                train=True,
                download=download,
            )
        elif dataset.lower() == "imagenet":
            _, train_loader, _ = make_imagenet1k(
                transform=transform,
                batch_size=batch_size,
                collator=None,
                pin_mem=True,
                num_workers=4,
                world_size=1,
                rank=0,
                root_path=root_path,
                image_folder=image_folder,
                training=True,
                copy_data=copy_data,
                drop_last=True,
                subset_file=subset_file,
            )
        elif dataset.lower() == "imagefolder":
            _, train_loader, _ = make_imagefolder(
                transform=transform,
                batch_size=batch_size,
                collator=None,
                pin_mem=True,
                num_workers=4,
                world_size=1,
                rank=0,
                root_path=root_path,
                image_folder=image_folder,
                drop_last=True,
                val_split=val_split,
            )
        else:
            raise ValueError(
                f"Unsupported dataset: {dataset}. Supported: cifar10, imagenet, imagefolder"
            )

        ddpm = DDPM(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        ).to(device)

        model = cls(
            img_size=img_size,
            patch_size=patch,
            in_channels=channels,
            d_model=width,
            depth=depth,
            heads=heads,
            drop=drop,
            t_dim=256,
        ).to(device)

        def param_count(model: nn.Module) -> int:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model Parameters: {param_count(model) / 1e6:.2f}M")

        ema_model = None
        if ema:
            import copy

            ema_model = copy.deepcopy(model).to(device).eval()
            for p in ema_model.parameters():
                p.requires_grad = False

        def ema_update(
            m: nn.Module, ema_m: nn.Module, decay: float = ema_decay
        ) -> None:
            with torch.no_grad():
                for p, q in zip(m.parameters(), ema_m.parameters()):
                    q.data.mul_(decay).add_(p.data, alpha=1 - decay)

        opt = torch.optim.AdamW(model.parameters(), lr=lr)

        global_step = 0
        nn.Module.train(model)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            for imgs, _ in train_loader:
                imgs = imgs.to(device)
                b = imgs.size(0)
                t = torch.randint(0, timesteps, (b,), device=device).long()
                noise = torch.randn_like(imgs)
                x_t = ddpm.q_sample(imgs, t, noise)

                pred = model(x_t, t)
                loss = F.mse_loss(pred, noise)

                opt.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if ema_model is not None:
                    ema_update(model, ema_model)

                if global_step % 100 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"Epoch [{epoch}/{epochs}] Step [{global_step}] Loss: {loss.item():.4f} Time Elapsed: {elapsed / 60:.2f} min"
                    )
                    start_time = time.time()

                global_step += 1
        print("Training Complete.")

        os.makedirs(workdir, exist_ok=True)

        model_to_save = ema_model if ema_model is not None else model
        checkpoint_path = Path(workdir) / "dit_model.pth"
        train_config = Config(
            DATASET=dataset,
            BATCH=batch_size,
            EPOCHS=epochs,
            LR=lr,
            IMG_SIZE=img_size,
            CHANNELS=channels,
            PATCH=patch,
            WIDTH=width,
            DEPTH=depth,
            HEADS=heads,
            DROP=drop,
            BETA_START=beta_start,
            BETA_END=beta_end,
            TIMESTEPS=timesteps,
            EMA=ema,
            EMA_DECAY=ema_decay,
            WORKDIR=workdir,
            ROOT_PATH=root_path,
        )
        save_config_next_to_checkpoint(train_config, checkpoint_path)
        torch.save(model_to_save.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

        # Generate new Images
        model_to_sample = ema_model if ema_model is not None else model
        model_to_sample.eval()
        with torch.no_grad():
            samples = ddpm.sample(
                model_to_sample, n=16, img_size=img_size, channels=channels
            )
            os.makedirs(workdir, exist_ok=True)
            save_image((samples + 1) / 2, f"{workdir}/generated_samples.png", nrow=4)
            print(f"Generated samples saved to {workdir}/generated_samples.png")


def create_dit(config: Any = None, **overrides: Any) -> DiT:
    """Create a :class:`DiT` from a ``DiTConfig`` or keyword overrides.

    The public factory API works with config objects, while ``DiT`` itself has a
    compact constructor. This adapter keeps the lower-level model constructor
    unchanged and maps the public config fields onto the expected arguments.
    """

    if config is None:
        config = Config()

    config_fields = set(getattr(config, "__dataclass_fields__", {}))
    config_overrides = {
        key: value for key, value in overrides.items() if key in config_fields
    }
    constructor_overrides = {
        key: value for key, value in overrides.items() if key not in config_fields
    }
    if config_overrides:
        from dataclasses import replace

        config = replace(config, **config_overrides)

    kwargs = {
        "img_size": config.IMG_SIZE,
        "patch_size": config.PATCH,
        "in_channels": config.CHANNELS,
        "d_model": config.WIDTH,
        "depth": config.DEPTH,
        "heads": config.HEADS,
        "drop": config.DROP,
    }
    supported = set(kwargs) | {"t_dim"}
    invalid = sorted(set(constructor_overrides) - supported)
    if invalid:
        raise ValueError(f"Unsupported DiT argument(s): {', '.join(invalid)}")
    kwargs.update(constructor_overrides)
    return DiT(**kwargs)
