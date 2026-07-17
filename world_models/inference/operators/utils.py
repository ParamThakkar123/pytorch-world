import torch
import torchvision.transforms as T
from typing import Any, Union


def normalize_image(
    image: Union[torch.Tensor, Any],
    size: tuple = (224, 224),
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
) -> torch.Tensor:
    """Normalize image tensor or PIL Image."""
    transform = T.Compose(
        [T.Resize(size), T.ToTensor(), T.Normalize(mean=mean, std=std)]
    )

    if isinstance(image, torch.Tensor):
        # Assume already tensor, just normalize
        return T.Normalize(mean=mean, std=std)(image)
    else:
        return transform(image)


def tokenize_text(text: str, max_length: int = 512) -> torch.Tensor:
    """Simple tokenization placeholder - replace with actual tokenizer."""
    # Placeholder: convert to dummy tokens
    tokens = [ord(c) % 1000 for c in text[:max_length]]  # Dummy tokenization
    tokens += [0] * (max_length - len(tokens))
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


def resize_image(image: Union[torch.Tensor, Any], size: tuple) -> torch.Tensor:
    """Resize image to specified size."""
    transform = T.Compose([T.Resize(size), T.ToTensor()])
    if isinstance(image, torch.Tensor):
        return T.Resize(size)(image)
    else:
        return transform(image)
