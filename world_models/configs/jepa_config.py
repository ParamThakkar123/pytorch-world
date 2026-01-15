import os
from typing import Tuple, Dict, Any


class JEPAConfig:
    """
    Minimal configuration container for JEPA training.
    Converts to the nested dict expected by `train_jepa.main`.
    """

    def __init__(self):
        # meta
        self.use_bfloat16: bool = False
        self.model_name: str = "vit_base"
        self.load_checkpoint: bool = False
        self.read_checkpoint: str | None = None
        self.copy_data: bool = False
        self.pred_depth: int = 6
        self.pred_emb_dim: int = 384

        # data
        self.dataset: str = "imagenet"  # "imagenet" or "imagefolder"
        self.val_split: float | None = (
            None  # optional fraction for val split when using imagefolder
        )
        self.use_gaussian_blur: bool = True
        self.use_horizontal_flip: bool = True
        self.use_color_distortion: bool = True
        self.color_jitter_strength: float = 0.5
        self.batch_size: int = 64
        self.pin_mem: bool = True
        self.num_workers: int = 8
        self.root_path: str = os.environ.get("IMAGENET_ROOT", "/data/imagenet")
        self.image_folder: str = "train"
        self.crop_size: int = 224
        self.crop_scale: Tuple[float, float] = (0.67, 1.0)
        self.download: bool = False  # allow CIFAR10 download if missing

        # mask
        self.allow_overlap: bool = False
        self.patch_size: int = 16
        self.num_enc_masks: int = 1
        self.min_keep: int = 4
        self.enc_mask_scale: Tuple[float, float] = (0.15, 0.2)
        self.num_pred_masks: int = 1
        self.pred_mask_scale: Tuple[float, float] = (0.15, 0.2)
        self.aspect_ratio: Tuple[float, float] = (0.75, 1.5)

        # optimization
        self.ema: Tuple[float, float] = (0.996, 1.0)
        self.ipe_scale: float = 1.0
        self.weight_decay: float = 0.04
        self.final_weight_decay: float = 0.4
        self.epochs: int = 300
        self.warmup: int = 40
        self.start_lr: float = 1e-6
        self.lr: float = 1.5e-4
        self.final_lr: float = 1e-6

        # logging
        self.folder: str = "results/jepa"
        self.write_tag: str = "jepa_run"

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "meta": {
                "use_bfloat16": self.use_bfloat16,
                "model_name": self.model_name,
                "load_checkpoint": self.load_checkpoint,
                "read_checkpoint": self.read_checkpoint,
                "copy_data": self.copy_data,
                "pred_depth": self.pred_depth,
                "pred_emb_dim": self.pred_emb_dim,
            },
            "data": {
                "dataset": self.dataset,
                "val_split": self.val_split,
                "use_gaussian_blur": self.use_gaussian_blur,
                "use_horizontal_flip": self.use_horizontal_flip,
                "use_color_distortion": self.use_color_distortion,
                "color_jitter_strength": self.color_jitter_strength,
                "batch_size": self.batch_size,
                "pin_mem": self.pin_mem,
                "num_workers": self.num_workers,
                "root_path": self.root_path,
                "image_folder": self.image_folder,
                "crop_size": self.crop_size,
                "crop_scale": self.crop_scale,
                "download": self.download,  # new
            },
            "mask": {
                "allow_overlap": self.allow_overlap,
                "patch_size": self.patch_size,
                "num_enc_masks": self.num_enc_masks,
                "min_keep": self.min_keep,
                "enc_mask_scale": self.enc_mask_scale,
                "num_pred_masks": self.num_pred_masks,
                "pred_mask_scale": self.pred_mask_scale,
                "aspect_ratio": self.aspect_ratio,
            },
            "optimization": {
                "ema": self.ema,
                "ipe_scale": self.ipe_scale,
                "weight_decay": self.weight_decay,
                "final_weight_decay": self.final_weight_decay,
                "epochs": self.epochs,
                "warmup": self.warmup,
                "start_lr": self.start_lr,
                "lr": self.lr,
                "final_lr": self.final_lr,
            },
            "logging": {
                "folder": self.folder,
                "write_tag": self.write_tag,
            },
        }
