#!/usr/bin/env python3
"""Interactively play inside a trained DIAMOND world model.

Two modes (toggle with TAB):
  REAL  - agent or human drives the real Atari environment
  DREAM - agent or human drives inside the diffusion model's imagination

Controls:
  TAB        - toggle REAL / DREAM mode
  R          - reset episode
  Arrow keys - steer (UP / DOWN / LEFT / RIGHT)
  W/A/S/D    - steer (UP / LEFT / DOWN / RIGHT)
  SPACE / X  - FIRE
  Z          - NOOP
  Q / ESC    - quit

When no human key is pressed the actor-critic policy takes over (AGENT control).
Hold a mapped key to take control (HUMAN control).

Usage:
    python scripts/play_diamond.py --checkpoint path/to/checkpoint.pt --game Breakout-v5
    python scripts/play_diamond.py --checkpoint path/to/checkpoint.pt --record gameplay.mp4
"""

import argparse
import time
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

from torchwm.configs.diamond_config import DiamondConfig
from torchwm.training.train_diamond import DiamondAgent
from scripts.play_base import (
    get_action_from_key,
    resolve_checkpoint_path,
    init_video_recorder,
)

ACTION_NAMES = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}


def make_agent(
    checkpoint: str,
    game: str,
    device: Optional[str] = None,
    seed: int = 42,
) -> DiamondAgent:
    ckpt_path = resolve_checkpoint_path(checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    if isinstance(cfg_dict, dict):
        config = DiamondConfig(**cfg_dict)
    else:
        config = cfg_dict

    config.game = game
    config.seed = seed
    if device is not None:
        config.device = device
    config.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config.terminate_on_life_loss = False

    agent = DiamondAgent(config)
    agent.load_checkpoint(ckpt_path)
    agent.actor_critic.eval()
    agent.diffusion_model.eval()
    agent.reward_model.eval()

    return agent


def run_play(
    checkpoint: str,
    game: str = "Breakout-v5",
    device: Optional[str] = None,
    seed: int = 42,
    deterministic: bool = True,
    record: Optional[str] = None,
    record_fps: int = 20,
) -> None:
    agent = make_agent(checkpoint, game, device, seed)
    device_obj = agent.device
    cfg = agent.config

    raw_obs, _ = agent.env.reset()
    norm_obs = raw_obs.astype(np.float32) / 255.0
    obs_history = [norm_obs] * cfg.num_conditioning_frames
    action_history: list[int] = []

    policy_hidden = agent.actor_critic.init_hidden(1, device_obj)
    dream_mode = False
    running = True
    episode_reward = 0.0
    step_count = 0
    fps_counter = 0
    fps_timer = time.time()
    fps_display = 0
    control_mode = "AGENT"

    video_recorder = init_video_recorder(record, fps=record_fps)
    cv2.namedWindow("DIAMOND Play", cv2.WINDOW_NORMAL)

    def build_obs_tensor():
        obs_np = np.stack(obs_history[-cfg.num_conditioning_frames :])
        obs_np = obs_np.transpose(0, 3, 1, 2)
        return torch.from_numpy(obs_np).unsqueeze(0).to(device_obj)

    def build_action_tensor():
        act_hist = action_history[-cfg.num_conditioning_frames :]
        if len(act_hist) < cfg.num_conditioning_frames:
            act_hist = [0] * (cfg.num_conditioning_frames - len(act_hist)) + act_hist
        return torch.tensor(act_hist, device=device_obj).unsqueeze(0)

    def reset_episode():
        nonlocal raw_obs, norm_obs, obs_history, action_history
        nonlocal policy_hidden, episode_reward, step_count
        raw_obs, _ = agent.env.reset()
        norm_obs = raw_obs.astype(np.float32) / 255.0
        obs_history = [norm_obs] * cfg.num_conditioning_frames
        action_history = []
        policy_hidden = agent.actor_critic.init_hidden(1, device_obj)
        episode_reward = 0.0
        step_count = 0

    while running:
        key = cv2.waitKey(16) & 0xFF
        if key == ord("q") or key == 27:
            running = False
            continue

        if key == ord("\t"):
            dream_mode = not dream_mode
            print(f"Switched to {'DREAM' if dream_mode else 'REAL'} mode")
            policy_hidden = agent.actor_critic.init_hidden(1, device_obj)
        if key == ord("r"):
            reset_episode()
            print("Reset episode")

        obs_tensor = build_obs_tensor()

        human_action = get_action_from_key(key)
        if human_action is not None:
            action = human_action
            control_mode = "HUMAN"
            policy_hidden = agent.actor_critic.init_hidden(1, device_obj)
            agent_action = human_action
        else:
            control_mode = "AGENT"
            agent_action, policy_hidden = agent.actor_critic.get_action(
                obs_tensor[:, -1], policy_hidden, deterministic=deterministic
            )
            action = agent_action

        act_tensor = build_action_tensor()

        if dream_mode:
            with torch.no_grad():
                generated = agent.sampler.sample(
                    model=agent.diffusion_model,
                    shape=(1, 3, cfg.obs_size, cfg.obs_size),
                    device=device_obj,
                    obs_history=obs_tensor,
                    actions=act_tensor,
                )

            gen_np = generated.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gen_np = np.clip(gen_np, 0.0, 1.0)
            display_rgb = gen_np
            display_bgr = (gen_np * 255).astype(np.uint8)
            display_bgr = cv2.cvtColor(display_bgr, cv2.COLOR_RGB2BGR)

            obs_history.append(gen_np)

        else:
            next_raw, reward, done, _ = agent.env.step(action)
            next_norm = next_raw.astype(np.float32) / 255.0

            display_rgb = next_raw.astype(np.float32) / 255.0
            display_bgr = next_raw
            if display_bgr.ndim == 3 and display_bgr.shape[2] == 3:
                display_bgr = cv2.cvtColor(display_bgr, cv2.COLOR_RGB2BGR)

            episode_reward += reward
            obs_history.append(next_norm)

            if done:
                print(
                    f"Episode finished. Reward: {episode_reward:.1f}, Steps: {step_count}"
                )
                reset_episode()

        action_history.append(action)
        step_count += 1

        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        action_name = ACTION_NAMES.get(action, str(action))
        mode_label = "DREAM" if dream_mode else "REAL"
        info_lines = [
            f"{mode_label}  {control_mode}  R: {episode_reward:.1f}  Step: {step_count}  FPS: {fps_display}",
            f"Action: {action_name} ({action})",
            "[TAB] toggle  [R] reset  [arrows/WASD] drive  [Q] quit",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(
                display_bgr,
                line,
                (5, 15 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

        cv2.imshow("DIAMOND Play", display_bgr)

        if video_recorder is not None:
            rec_frame = (display_rgb * 255).astype(np.uint8)
            video_recorder.write_frame(rec_frame)

    if video_recorder is not None:
        video_recorder.close()
    cv2.destroyAllWindows()
    agent.env.close()
    print("Exited.")


def main():
    parser = argparse.ArgumentParser(
        description="Play inside a trained DIAMOND world model"
    )
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--game", "-g", default="Breakout-v5")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )
    parser.add_argument(
        "--record", default=None, help="Path to save gameplay video (e.g. gameplay.mp4)"
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=20,
        help="FPS for recorded video (default: 20)",
    )
    args = parser.parse_args()
    run_play(
        checkpoint=args.checkpoint,
        game=args.game,
        device=args.device,
        seed=args.seed,
        deterministic=not args.stochastic,
        record=args.record,
        record_fps=args.record_fps,
    )


if __name__ == "__main__":
    main()
