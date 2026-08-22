#!/usr/bin/env python3
"""Record a fixed-length DIAMOND demo video without a display.

``scripts/play_diamond.py`` is interactive: it opens a ``cv2`` window and runs
until you press Q. That makes it unusable over SSH on a headless GPU box and
awkward for producing a reproducible clip. This script drives the same agent
non-interactively for a set number of steps and writes the videos to disk.

It produces up to three files:
  real.mp4        - the policy acting in the real environment
  dream.mp4       - the diffusion world model imagining forward from the same
                    conditioning frames, driven by the same policy
  side_by_side.mp4 - both of the above stitched horizontally (real | dream)

Usage:
    python demos/record_diamond.py -c checkpoints/diamond/checkpoint_0.pt
    python demos/record_diamond.py -c ckpt.pt --steps 400 --dream-steps 200 --scale 6
    python demos/record_diamond.py -c ckpt.pt --device cpu --out-dir demos/out
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Running this file directly puts demos/ on sys.path, not the repo root, so the
# `scripts` and `torchwm` packages would not resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from scripts.play_diamond import make_agent
from torchwm.utils.utils import StreamingVideoWriter


def upscale(frame: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour upscale so 64x64 frames are legible in a demo."""
    if scale <= 1:
        return frame
    height, width = frame.shape[:2]
    return cv2.resize(
        frame, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST
    )


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert a float RGB frame in [0, 1] to uint8 without clipping surprises."""
    return (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn a caption into an RGB uint8 frame."""
    out = frame.copy()
    cv2.putText(
        out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
    )
    return out


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> Optional[Path]:
    """Write RGB uint8 frames to ``path``. Returns the path, or None if empty."""
    if not frames:
        return None
    writer = StreamingVideoWriter(str(path), fps=fps)
    for frame in frames:
        writer.write_frame(frame)
    writer.close()
    return path


def rollout(
    agent,
    steps: int,
    dream: bool,
    deterministic: bool,
    scale: int,
) -> tuple[list[np.ndarray], float]:
    """Roll the policy forward in the real env or inside the world model.

    Returns the captured frames and the accumulated reward (always 0.0 in dream
    mode, where there is no environment to score against).
    """
    cfg = agent.config
    device = agent.device

    raw_obs, _ = agent.env.reset()
    obs_history = [raw_obs.astype(np.float32) / 255.0] * cfg.num_conditioning_frames
    action_history: list[int] = []
    policy_hidden = agent.actor_critic.init_hidden(1, device)

    frames: list[np.ndarray] = []
    episode_reward = 0.0
    started = time.time()

    for step in range(steps):
        obs_np = np.stack(obs_history[-cfg.num_conditioning_frames :]).transpose(
            0, 3, 1, 2
        )
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(device)

        with torch.no_grad():
            action, policy_hidden = agent.actor_critic.get_action(
                obs_tensor[:, -1], policy_hidden, deterministic=deterministic
            )

        if dream:
            act_hist = action_history[-cfg.num_conditioning_frames :]
            if len(act_hist) < cfg.num_conditioning_frames:
                act_hist = [0] * (
                    cfg.num_conditioning_frames - len(act_hist)
                ) + act_hist
            act_tensor = torch.tensor(act_hist, device=device).unsqueeze(0)

            with torch.no_grad():
                generated = agent.sampler.sample(
                    model=agent.diffusion_model,
                    shape=(1, 3, cfg.obs_size, cfg.obs_size),
                    device=device,
                    obs_history=obs_tensor,
                    actions=act_tensor,
                )
            next_frame = np.clip(
                generated.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0
            )
            obs_history.append(next_frame)
        else:
            next_raw, reward, done, _ = agent.env.step(action)
            next_frame = next_raw.astype(np.float32) / 255.0
            episode_reward += float(reward)
            obs_history.append(next_frame)
            if done:
                # Keep recording across episode boundaries so the clip stays the
                # requested length rather than ending early.
                raw_obs, _ = agent.env.reset()
                reset_frame = raw_obs.astype(np.float32) / 255.0
                obs_history = [reset_frame] * cfg.num_conditioning_frames
                action_history = []
                policy_hidden = agent.actor_critic.init_hidden(1, device)

        action_history.append(int(action))
        frames.append(upscale(to_uint8(next_frame), scale))

        if (step + 1) % 50 == 0:
            rate = (step + 1) / max(1e-6, time.time() - started)
            print(f"  {'dream' if dream else 'real'}: {step + 1}/{steps} ({rate:.1f} fps)")

    return frames, episode_reward


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record DIAMOND real-env and imagination videos headlessly"
    )
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to a .pt file.")
    parser.add_argument("--game", "-g", default="Breakout-v5", help="Atari game id.")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300, help="Real env frames.")
    parser.add_argument(
        "--dream-steps",
        type=int,
        default=100,
        help="Imagination frames. Diffusion sampling is slow on CPU; set 0 to skip.",
    )
    parser.add_argument(
        "--out-dir", default="demos/out", help="Directory for the written videos."
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--scale", type=int, default=4, help="Nearest-neighbour upscale factor."
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions instead of taking the argmax.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    agent = make_agent(args.checkpoint, args.game, args.device, args.seed)
    print(f"Device: {agent.device}  preset: {agent.config.preset}  game: {args.game}")

    deterministic = not args.stochastic
    written: list[Path] = []

    real_frames: list[np.ndarray] = []
    if args.steps > 0:
        print(f"Recording {args.steps} real-environment frames...")
        real_frames, reward = rollout(
            agent, args.steps, False, deterministic, args.scale
        )
        print(f"  episode reward across clip: {reward:.1f}")
        path = write_video(
            out_dir / "real.mp4", [label(f, "REAL") for f in real_frames], args.fps
        )
        if path:
            written.append(path)

    dream_frames: list[np.ndarray] = []
    if args.dream_steps > 0:
        print(f"Recording {args.dream_steps} imagined frames...")
        dream_frames, _ = rollout(
            agent, args.dream_steps, True, deterministic, args.scale
        )
        path = write_video(
            out_dir / "dream.mp4", [label(f, "DREAM") for f in dream_frames], args.fps
        )
        if path:
            written.append(path)

    if real_frames and dream_frames:
        pairs = min(len(real_frames), len(dream_frames))
        combined = [
            np.concatenate(
                [label(real_frames[i], "REAL"), label(dream_frames[i], "DREAM")], axis=1
            )
            for i in range(pairs)
        ]
        path = write_video(out_dir / "side_by_side.mp4", combined, args.fps)
        if path:
            written.append(path)

    agent.env.close()
    print("\nWrote:")
    for path in written:
        print(f"  {path}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
