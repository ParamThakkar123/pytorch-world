import torch
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm
import random
from collections.abc import Sequence
from typing import Optional, cast
from gym.spaces import Discrete, Box

from types import ModuleType
from typing import Optional as _Optional

from world_models.configs.iris_config import IRISConfig
from world_models.experiments import (
    dump_config,
    load_experiment_config,
    update_config_object,
)
from world_models.models.iris_agent import IRISAgent
from world_models.memory.iris_memory import IRISReplayBuffer
from world_models.envs.ale_atari_env import make_atari_env

# Optional OpenCV import at module scope (avoid function-local imports)
cv2: _Optional[ModuleType] = None
try:
    import cv2 as _cv2

    cv2 = _cv2
except Exception:
    cv2 = None


class IRISTrainer:
    """Training loop for IRIS on Atari 100k benchmark."""

    def __init__(
        self,
        game: str = "ALE/Pong-v5",
        device: str = "cuda",
        seed: int = 42,
        config: Optional[IRISConfig] = None,
    ) -> None:
        self.game = game
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.seed = seed

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Config
        self.config = config if config is not None else IRISConfig()

        # Create environment
        self.env = make_atari_env(
            game,
            obs_type="rgb",
            frameskip=4,
            max_episode_steps=27000,  # Standard Atari limit
        )

        # Get action space robustly (Discrete or Box)
        # Declare attribute type for static checkers
        self.action_size: int = 0

        if isinstance(self.env.action_space, Discrete):
            self.action_size = int(self.env.action_space.n)
        elif isinstance(self.env.action_space, Box):
            shape = getattr(self.env.action_space, "shape", None)
            if shape is None:
                raise TypeError("Box action_space has no shape")
            self.action_size = int(np.prod(tuple(shape)))
        else:
            if hasattr(self.env.action_space, "n"):
                self.action_size = int(getattr(self.env.action_space, "n"))
            else:
                raise TypeError(
                    f"Unsupported action_space type: {type(self.env.action_space)}"
                )

        # Create replay buffer
        self.replay_buffer = IRISReplayBuffer(
            size=100000,  # 100k buffer
            obs_shape=(3, 64, 64),  # Resize frames to 64x64
            action_size=self.action_size,
            seq_len=self.config.transformer_timesteps,
            batch_size=self.config.transformer_batch_size,
        )

        # Create agent
        self.agent = IRISAgent(
            config=self.config,
            action_size=self.action_size,
            device=self.device,
        )

        # Metrics: map metric name -> series of numeric values
        self.metrics: defaultdict[str, list[float]] = defaultdict(list)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame: resize to 64x64 and normalize."""
        if cv2 is None:
            raise ImportError(
                "cv2 is required for frame preprocessing. Install opencv-python"
            )

        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_LINEAR)
        frame = frame.astype(np.float32) / 255.0
        return frame.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

    def collect_experience(
        self,
        num_steps: int,
        epsilon: float = 0.01,
    ) -> float:
        """Collect experience from environment.

        Args:
            num_steps: Number of steps to collect
            epsilon: Random action probability

        Returns:
            Mean episode return
        """
        obs, _ = self.env.reset()
        obs = self.preprocess_frame(obs)

        episode_returns = []
        current_return: float = 0.0
        steps_in_episode = 0

        for step in range(num_steps):
            # Choose action
            if np.random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                frame_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                )
                action = self.agent.act(frame_tensor, epsilon=0.0).item()

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            next_obs = self.preprocess_frame(next_obs)

            # Store in replay buffer
            action_one_hot: np.ndarray = np.zeros(self.action_size, dtype=np.float32)
            action_one_hot[action] = 1.0

            self.replay_buffer.add(
                obs,
                action_one_hot,
                float(reward),
                done,
            )

            current_return += float(reward)
            steps_in_episode += 1

            if done:
                episode_returns.append(current_return)
                current_return = 0.0
                steps_in_episode = 0
                obs, _ = self.env.reset()
                obs = self.preprocess_frame(obs)
            else:
                obs = next_obs

        return float(np.mean(episode_returns)) if episode_returns else 0.0

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Get epsilon with decay for better exploration early on
        epsilon = self.get_epsilon(epoch)

        # Phase 1: Collect experience
        mean_return = self.collect_experience(
            num_steps=self.config.env_steps_per_epoch,
            epsilon=epsilon,
        )
        metrics["collection_return"] = mean_return

        # Only update components after warm-start periods
        if epoch >= self.config.start_autoencoder_after:
            # Phase 2: Update autoencoder
            ae_metrics: dict = {}
            for _ in range(self.config.training_steps_per_epoch):
                # Sample random frames
                indices = np.random.randint(
                    0, len(self.replay_buffer), size=self.config.autoencoder_batch_size
                )

                frames = (
                    torch.tensor(
                        np.array([self.replay_buffer.observations[i] for i in indices]),
                        dtype=torch.float32,
                    ).to(self.device)
                    / 255.0
                )

                ae_metrics = self.agent.update_autoencoder(frames)

            metrics["recon_loss"] = ae_metrics.get("recon_loss", 0)
            metrics["vq_loss"] = ae_metrics.get("vq_loss", 0)
            metrics["perplexity"] = ae_metrics.get("perplexity", 0)

        if (
            epoch >= self.config.start_transformer_after
            and len(self.replay_buffer) >= self.config.transformer_timesteps + 1
        ):
            # Phase 3: Update transformer
            obs, acts, rews, terms = self.replay_buffer.sample_sequence()

            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device) / 255.0
            acts_tensor = torch.tensor(acts, dtype=torch.float32).to(self.device)
            rews_tensor = torch.tensor(rews, dtype=torch.float32).to(self.device)
            terms_tensor = torch.tensor(terms, dtype=torch.long).to(self.device)

            tf_metrics = self.agent.update_transformer(
                obs_tensor, acts_tensor, rews_tensor, terms_tensor
            )

            metrics["token_loss"] = tf_metrics.get("token_loss", 0)
            metrics["reward_loss"] = tf_metrics.get("reward_loss", 0)

        if (
            epoch >= self.config.start_actor_critic_after
            and len(self.replay_buffer) >= 50
        ):
            # Phase 4: Update actor-critic in imagination
            # Sample initial frames for imagination
            sample_size = self.config.actor_critic_batch_size
            indices = np.random.randint(0, len(self.replay_buffer), size=sample_size)

            initial_frames = (
                torch.tensor(
                    np.array([self.replay_buffer.observations[i] for i in indices]),
                    dtype=torch.float32,
                ).to(self.device)
                / 255.0
            )

            # Generate imagined trajectories
            imagined = self.agent.imagine_rollout(
                initial_frame=initial_frames,
                horizon=self.config.imagination_horizon,
            )

            # Update policy
            ac_metrics = self.agent.update_actor_critic(imagined)

            metrics["actor_loss"] = ac_metrics.get("actor_loss", 0)
            metrics["value_loss"] = ac_metrics.get("value_loss", 0)
            metrics["entropy"] = ac_metrics.get("entropy", 0)

        self.agent.current_epoch = epoch
        self.agent.global_step += self.config.env_steps_per_epoch

        return metrics

    def get_epsilon(self, epoch: int) -> float:
        """Get exploration epsilon with decay."""
        start_epsilon = 0.5
        min_epsilon = self.config.collect_epsilon
        decay_epochs = 50
        epsilon = max(
            min_epsilon,
            start_epsilon - (start_epsilon - min_epsilon) * epoch / decay_epochs,
        )
        return epsilon

    def evaluate(self, num_episodes: int = 100, render: bool = False) -> dict | tuple:
        """Evaluate agent performance.

        Args:
            num_episodes: Number of evaluation episodes
            render: If True, also return video frames and per-step latent vectors

        Returns:
            If render is False (default): dict with evaluation metrics
            If render is True: tuple (episode_returns_array, videos_list, latents_array)
        """
        episode_returns = []
        videos: list[list[np.ndarray]] = []
        latents_all: list[np.ndarray] = []

        for _ in range(num_episodes):
            raw_obs, _ = self.env.reset()
            obs = self.preprocess_frame(raw_obs)

            episode_return: float = 0.0
            done = False
            frames: list[np.ndarray] = []

            while not done:
                # Prepare frame for policy (CHW, float32, 0-1)
                frame_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                )
                action = self.agent.act(
                    frame_tensor, epsilon=0.0, temperature=self.config.eval_temperature
                ).item()

                next_raw, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store raw frame for video (as HWC uint8 if possible)
                try:
                    frames.append(np.asarray(next_raw))
                except Exception:
                    # Fallback: convert processed obs back to HWC
                    proc = np.asarray(obs)
                    if proc.ndim == 3:
                        # CHW -> HWC
                        frames.append(proc.transpose(1, 2, 0))

                # Compute latent embedding via encoder (quantized embeddings)
                try:
                    proc_frame = self.preprocess_frame(next_raw) if not done else obs
                    with torch.no_grad():
                        ft = (
                            torch.tensor(proc_frame, dtype=torch.float32)
                            .unsqueeze(0)
                            .to(self.device)
                        )
                        z_q, _, _ = self.agent.encoder(ft)
                        # z_q: (B, C, H', W') -> reduce spatial dims and take mean over channels
                        latent = z_q.mean(dim=(2, 3)).squeeze(0).cpu().numpy()
                        latents_all.append(latent.astype(np.float32))
                except Exception:
                    # If encoder fails, skip latent for this step
                    pass

                episode_return += float(reward)
                obs = self.preprocess_frame(next_raw) if not done else obs

            episode_returns.append(episode_return)
            videos.append(frames)

        if render:
            # Stack latents into (N, D) array if any
            if latents_all:
                latents_array = np.vstack(latents_all).astype(np.float32)
            else:
                latents_array = np.empty((0,), dtype=np.float32)
            return np.array(episode_returns), videos, latents_array

        # Non-render fallback: return simple metrics dict for compatibility
        return {
            "eval_mean_return": float(
                np.mean(episode_returns) if episode_returns else 0.0
            ),
            "eval_std_return": float(
                np.std(episode_returns) if episode_returns else 0.0
            ),
            "eval_max_return": float(
                np.max(episode_returns) if episode_returns else 0.0
            ),
            "eval_min_return": float(
                np.min(episode_returns) if episode_returns else 0.0
            ),
        }

    def train(
        self,
        total_epochs: Optional[int] = None,
        eval_interval: int = 50,
        save_dir: str = "checkpoints/iris",
    ) -> None:
        """Full training loop.

        Args:
            total_epochs: Total training epochs
            eval_interval: Evaluate every N epochs
            save_dir: Directory to save checkpoints
        """
        if total_epochs is None:
            total_epochs = self.config.total_epochs

        os.makedirs(save_dir, exist_ok=True)

        print(f"Starting training for {total_epochs} epochs on {self.game}")
        print(f"Action space: {self.action_size}")
        print(f"Device: {self.device}")

        best_eval_return = float("-inf")

        for epoch in tqdm(range(total_epochs), desc="Training"):
            # Train one epoch
            metrics = self.train_epoch(epoch)

            # Log metrics
            for key, value in metrics.items():
                self.metrics[key].append(value)

            # Print progress
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch}:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")

            # Evaluate periodically
            if (
                epoch % eval_interval == 0
                and epoch >= self.config.start_actor_critic_after
            ):
                eval_metrics = cast(
                    dict, self.evaluate(num_episodes=self.config.eval_episodes)
                )
                print(f"\nEvaluation at epoch {epoch}:")
                print(
                    f"  Mean return: {eval_metrics['eval_mean_return']:.2f} +/- {eval_metrics['eval_std_return']:.2f}"
                )

                # Save best model
                if eval_metrics["eval_mean_return"] > best_eval_return:
                    best_eval_return = eval_metrics["eval_mean_return"]
                    save_path = os.path.join(
                        save_dir, f"best_{self.game.split('/')[-1]}.pt"
                    )
                    self.agent.save(save_path)
                    print(f"  Saved best model: {save_path}")

            # Checkpoint periodically
            if epoch % self.config.checkpoint_interval == 0:
                save_path = os.path.join(save_dir, f"checkpoint_{epoch}.pt")
                self.agent.save(save_path)

        print(f"\nTraining complete! Best eval return: {best_eval_return:.2f}")

        return self.metrics  # type: ignore[return-value]


# Overrides that configure the training *run* rather than the model. They are
# not IRISConfig fields, so they must be split out before the config is composed
# -- `update_config_object` is strict and rejects any key it does not recognise.
RUNTIME_OVERRIDE_KEYS = ("game", "device", "seed", "epochs", "save_dir")


def _split_runtime_overrides(
    overrides: Sequence[str],
) -> tuple[dict[str, str], list[str]]:
    """Partition ``key=value`` overrides into runtime options and config fields."""
    runtime: dict[str, str] = {}
    config_overrides: list[str] = []
    for item in overrides:
        key, sep, value = item.partition("=")
        if sep and key.strip() in RUNTIME_OVERRIDE_KEYS:
            runtime[key.strip()] = value.strip()
        else:
            config_overrides.append(item)
    return runtime, config_overrides


def main(argv: list[str] | None = None) -> IRISConfig:
    """Run IRIS training with YAML config files and Hydra dot-list overrides."""
    from world_models.experiments import parse_experiment_args

    args = parse_experiment_args(argv, description="Train IRIS on Atari")

    # Split first: passing `game=`/`device=`/`seed=` through to the strict config
    # loader previously raised ExperimentConfigError, making them unusable.
    runtime, config_overrides = _split_runtime_overrides(args.overrides)

    config = IRISConfig()
    values = load_experiment_config(config, args.config, config_overrides)
    config = update_config_object(config, values)

    game = runtime.get("game", config.env)
    device = runtime.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(runtime.get("seed", 42))
    total_epochs = int(runtime.get("epochs", config.total_epochs))
    save_dir = runtime.get("save_dir", "checkpoints/iris")

    if args.print_config:
        print(dump_config(values))
        return config

    trainer = IRISTrainer(
        game=game,
        device=device,
        seed=seed,
        config=config,
    )

    trainer.train(
        total_epochs=total_epochs,
        save_dir=save_dir,
    )
    return config


if __name__ == "__main__":
    main()
