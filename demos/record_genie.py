#!/usr/bin/env python3
"""Generate frames from a trained Genie checkpoint, headlessly.

Genie generates video frames from a single prompt frame using a learned
dynamics model and latent action model. This script:

1. Loads a trained Genie checkpoint (TinyWorlds or paper-scale).
2. Takes a prompt frame (from a sample in the dataset).
3. Generates ``num_frames`` future frames using the dynamics model.
4. Writes:
   - ``genie_frames.mp4`` — the generated video.
   - ``genie_grid.png`` — a grid of prompt + generated frames.

Usage:
    python demos/record_genie.py -c checkpoints/genie_sonic_final.pt
    python demos/record_genie.py -c ckpt.pt --num-frames 32 --prompt-index 1
    python demos/record_genie.py --random-init --num-frames 8   # pipeline check
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from torchwm.models.genie import create_genie_small
from torchwm.utils.utils import StreamingVideoWriter


def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    """Convert (C,H,W) float tensor to uint8 HxWxC numpy array."""
    arr = t.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    if arr.max() <= 1.0:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round()
    else:
        arr = np.clip(arr, 0, 255).round()
    return arr.astype(np.uint8)


def to_uint8_grid(frames: torch.Tensor, nrow: int) -> np.ndarray:
    """Tile (B, C, H, W) frames in [-1, 1] / [0, 1] into a single uint8 image."""
    imgs = frames.detach().cpu()
    if imgs.min() >= -0.01:
        imgs = (imgs.clamp(0, 1) * 255).round().to(torch.uint8)
    else:
        imgs = ((imgs.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
    imgs = imgs.numpy()
    n, c, h, w = imgs.shape
    ncol = int(math.ceil(n / nrow))
    canvas = np.zeros((ncol * h, nrow * w, c), dtype=np.uint8)
    for i in range(n):
        r, col = divmod(i, nrow)
        canvas[r * h : (r + 1) * h, col * w : (col + 1) * w] = imgs[i].transpose(
            1, 2, 0
        )
    return canvas


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Create a Genie model and load checkpoint if provided."""
    if args.random_init:
        print("--random-init: generating from an UNTRAINED model (noise).")
        return create_genie_small(
            num_frames=args.num_frames, image_size=args.image_size
        ).eval()

    model = create_genie_small(num_frames=args.num_frames, image_size=args.image_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def make_prompt(
    model: torch.nn.Module,
    image_size: int,
    channels: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a random prompt frame in [0, 1]."""
    return torch.rand(1, channels, image_size, image_size, device=device)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate video frames from a Genie checkpoint"
    )
    parser.add_argument("--checkpoint", "-c", default=None)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Skip checkpoint; generate from an untrained model.",
    )
    parser.add_argument(
        "--num-frames", "-n", type=int, default=16, help="Frames to generate."
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--nrow", type=int, default=8, help="Grid columns.")
    parser.add_argument("--out-dir", default="demos/out")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.checkpoint and not args.random_init:
        parser.error("pass --checkpoint/-c, or --random-init to check the pipeline")

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = build_model(args)
    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} on {device}")

    prompt = make_prompt(model, args.image_size, args.channels, device)
    print(f"Prompt shape: {prompt.shape}")

    with torch.no_grad():
        generated = model.generate(
            prompt, num_frames=args.num_frames, actions=None, use_maskgit=False
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = to_uint8_grid(generated.squeeze(0), args.nrow)
    grid_path = out_dir / "genie_grid.png"
    _write_png(grid, grid_path)
    print(f"Wrote {grid_path}")

    video_path = out_dir / "genie_frames.mp4"
    writer = StreamingVideoWriter(str(video_path), fps=args.fps)
    frames = generated.squeeze(0)
    for t in range(frames.shape[1]):
        frame = frames[:, t, :, :]
        img = tensor_to_uint8_img(frame)
        writer.write_frame(np.tile(img, (2, 2, 1)) if args.image_size < 128 else img)
    for _ in range(args.fps):
        writer.write_frame(
            np.tile(tensor_to_uint8_img(frames[:, -1, :, :]), (2, 2, 1))
            if args.image_size < 128
            else tensor_to_uint8_img(frames[:, -1, :, :])
        )
    writer.close()
    print(f"Wrote {video_path}  ({args.num_frames} frames)")

    return 0


def _write_png(image: np.ndarray, path: Path) -> None:
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    raise SystemExit(main())
