import torch
import torchvision.transforms as T
from .base import OperatorABC
from typing import Dict, Any


class JEPAOperator(OperatorABC):
    """Operator for JEPA model preprocessing: handles image/video masking and patch processing."""

    def __init__(
        self, image_size: int = 224, patch_size: int = 16, mask_ratio: float = 0.75
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process JEPA inputs: images with masking.

        Expected inputs: {'images': list of PIL Images or tensors, 'mask': optional tensor}
        """
        processed = {}

        # Process images
        if "images" in inputs:
            images = inputs["images"]
            if not isinstance(images, list):
                images = [images]

            # Apply transforms
            tensor_images = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    tensor_images.append(img)
                else:
                    tensor_images.append(self.transform(img))

            processed["images"] = torch.stack(tensor_images)

        # Generate or use mask
        if "mask" in inputs:
            processed["mask"] = inputs["mask"]
        else:
            # Generate random mask
            num_patches = (self.image_size // self.patch_size) ** 2
            mask = torch.rand(num_patches) > self.mask_ratio
            processed["mask"] = mask.float()

        return processed
