import os
import torch
import numpy as np
from collections import namedtuple

from world_models.envs.dreamer_envs import Env
from world_models.controller.dreamer_v1_agent import Dreamer
from world_models.memory.dreamer_memory import ExperienceReplay
from world_models.utils.utils import TensorBoardMetrics, save_video

# Configuration for DreamerV1
DreamerConfig = namedtuple(
    "DreamerConfig",
    [
        "device",
        "symbolic",
        "observation_size",
        "action_size",
        "bit_depth",
        "belief_size",
        "state_size",
        "hidden_size",
        "embedding_size",
        "dense_act",
        "cnn_act",
        "reward_scale",
        "pcont",
        "pcont_scale",
        "world_lr",
        "actor_lr",
        "value_lr",
        "free_nats",
        "batch_size",
        "planning_horizon",
        "discount",
        "disclam",
        "temp",
        "with_logprob",
        "grad_clip_norm",
        "expl_amount",
    ],
)


class DreamerV1:
    """
    High-level DreamerV1 wrapper similar to Planet.

    Usage example:
      from world_models.models.dreamer_v1 import DreamerV1
      d = DreamerV1(env='HalfCheetah-v4', symbolic=False)
      d.train(episodes=1000)
    """

    def __init__(
        self,
        env,
        symbolic=False,
        bit_depth=5,
        device=None,
        belief_size=200,
        state_size=30,
        hidden_size=200,
        embedding_size=1024,
        max_episode_length=1000,
        action_repeat=2,
        results_dir=None,
        **kwargs,
    ):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Create environment
        if isinstance(env, str):
            self.env = Env(
                env=env,
                symbolic=symbolic,
                seed=42,
                max_episode_length=max_episode_length,
                action_repeat=action_repeat,
                bit_depth=bit_depth,
            )
        else:
            self.env = env

        # Create agent configuration with more conservative settings
        self.config = DreamerConfig(
            device=self.device,
            symbolic=symbolic,
            observation_size=self.env.observation_size,
            action_size=self.env.action_size,
            bit_depth=bit_depth,
            belief_size=belief_size,
            state_size=state_size,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            dense_act=kwargs.get("dense_act", "elu"),
            cnn_act=kwargs.get("cnn_act", "relu"),
            reward_scale=kwargs.get("reward_scale", 1.0),
            pcont=kwargs.get("pcont", True),
            pcont_scale=kwargs.get("pcont_scale", 10.0),
            world_lr=kwargs.get("world_lr", 6e-4),  # Reduced from 6e-4
            actor_lr=kwargs.get("actor_lr", 8e-5),  # Reduced from 8e-5
            value_lr=kwargs.get("value_lr", 8e-5),  # Reduced from 8e-5
            free_nats=kwargs.get("free_nats", 3.0),
            batch_size=kwargs.get("batch_size", 50),
            planning_horizon=kwargs.get("planning_horizon", 12),  # Reduced from 15
            discount=kwargs.get("discount", 0.99),
            disclam=kwargs.get("disclam", 0.95),
            temp=kwargs.get("temp", 0.5),
            with_logprob=kwargs.get("with_logprob", True),
            grad_clip_norm=kwargs.get("grad_clip_norm", 10.0),  # Reduced from 100.0
            expl_amount=kwargs.get("expl_amount", 0.3),
        )

        # Create agent
        self.agent = Dreamer(self.config)

        # Create experience replay buffer
        self.memory = ExperienceReplay(
            size=kwargs.get("memory_size", 1000000),
            symbolic_env=symbolic,
            observation_size=self.env.observation_size,
            action_size=self.env.action_size,
            bit_depth=bit_depth,
            device=self.device,
        )

        # Results directory
        env_name = env if isinstance(env, str) else "custom_env"
        self.results_dir = results_dir or f"results/dreamer_v1_{env_name}"
        os.makedirs(self.results_dir, exist_ok=True)

        print("DreamerV1 initialized:")
        print(f"  Environment: {env_name}")
        print(f"  Action size: {self.env.action_size}")
        print(f"  Observation size: {self.env.observation_size}")
        print(f"  Device: {self.device}")
        print(f"  Results dir: {self.results_dir}")

    def warmup(self, episodes=5):
        """Collect random episodes to initialize the replay buffer."""
        print(f"Collecting {episodes} random episodes for warmup...")

        for ep in range(episodes):
            self.env.reset()
            episode_reward = 0
            step_count = 0

            while True:
                action = self.env.sample_random_action()
                next_observation, reward, done = self.env.step(action)

                self.memory.append(next_observation, action.cpu(), reward, done)
                episode_reward += reward
                step_count += 1

                if done or step_count >= 1000:
                    break

            print(f"  Episode {ep+1}: {step_count} steps, reward: {episode_reward:.2f}")

        print(
            f"Warmup complete. Buffer size: {self.memory.episodes}"
        )  # Changed from len(self.memory)

    def collect_episode(self, deterministic=False):
        """Collect one episode using current policy."""
        observation = self.env.reset()
        belief = torch.zeros(1, self.config.belief_size, device=self.device)
        state = torch.zeros(1, self.config.state_size, device=self.device)
        action = torch.zeros(1, self.env.action_size, device=self.device)

        episode_reward = 0
        step_count = 0
        frames = []

        while True:
            # Infer state
            belief, state = self.agent.infer_state(
                observation.to(self.device), belief, state, action
            )

            # Select action
            action = self.agent.select_action(
                belief, state, deterministic=deterministic
            )

            # Take step
            next_observation, reward, done = self.env.step(action[0].cpu())

            # Store experience (only during training)
            if not deterministic:
                # detach/clone to avoid sharing storage with current computation graph
                obs_to_store = (
                    next_observation.detach().cpu().clone()
                    if isinstance(next_observation, torch.Tensor)
                    else next_observation
                )
                act_to_store = action.detach().cpu().clone()
                self.memory.append(obs_to_store, act_to_store, reward, done)

            # Save frame for video - postprocess to restore brightness
            if not self.config.symbolic:
                frame = observation.cpu()
                if isinstance(frame, torch.Tensor):
                    frame = frame.numpy()
                frame = np.clip(frame + 0.5, 0.0, 1.0)
                frames.append(frame)

            episode_reward += reward
            step_count += 1
            observation = next_observation

            if done or step_count >= 1000:
                break

        return episode_reward, step_count, frames

    def evaluate(self, episodes=10):
        """Evaluate current policy."""
        print(f"Evaluating policy over {episodes} episodes...")

        # Set to eval mode
        self.agent.transition_model.eval()
        self.agent.observation_model.eval()
        self.agent.reward_model.eval()
        self.agent.encoder.eval()
        self.agent.actor_model.eval()
        self.agent.value_model.eval()

        rewards = []
        lengths = []

        with torch.no_grad():
            for ep in range(episodes):
                reward, length, frames = self.collect_episode(deterministic=True)
                rewards.append(reward)
                lengths.append(length)

                # Save video for first episode
                if ep == 0 and frames:
                    # ensure frames array has shape (T, C, H, W) or (T, H, W, C)
                    import numpy as _np
                    import torch as _torch

                    # stack if list of tensors/arrays
                    if isinstance(frames, list) and len(frames) > 0:
                        first = frames[0]
                        if isinstance(first, _torch.Tensor):
                            frames_arr = _torch.stack(frames, dim=0).cpu()
                        else:
                            frames_arr = _np.stack(frames, axis=0)
                    else:
                        frames_arr = frames

                    # convert to numpy if torch tensor
                    if isinstance(frames_arr, _torch.Tensor):
                        frames_arr = frames_arr.detach().cpu().numpy()

                    # remove singleton extra dim (e.g. (T, 1, C, H, W) -> (T, C, H, W))
                    if frames_arr.ndim == 5 and frames_arr.shape[1] == 1:
                        frames_arr = frames_arr.squeeze(1)

                    # now save (works for (T,C,H,W) or (T,H,W,C))
                    save_video(frames_arr, self.results_dir, "eval_episode")

        # Set back to train mode
        self.agent.transition_model.train()
        self.agent.observation_model.train()
        self.agent.reward_model.train()
        self.agent.encoder.train()
        self.agent.actor_model.train()
        self.agent.value_model.train()

        return {
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "length_mean": np.mean(lengths),
            "episodes": episodes,
        }

    def train(
        self,
        episodes=1000,
        warmup_episodes=5,
        collect_interval=100,
        train_steps_per_collect=50,
        eval_interval=100,
        save_interval=100,
        chunk_size=50,
    ):
        """
        Main training loop.

        Args:
            episodes: Total number of episodes to train
            warmup_episodes: Number of random episodes for warmup
            collect_interval: Steps between training updates
            train_steps_per_collect: Training steps per update
            eval_interval: Episodes between evaluations
            save_interval: Episodes between model saves
            chunk_size: Sequence length for training
        """
        metrics = TensorBoardMetrics(self.results_dir)

        # Warmup
        if self.memory.episodes == 0:  # Changed from len(self.memory) == 0
            self.warmup(warmup_episodes)

        print(f"Starting training for {episodes} episodes...")

        for episode in range(1, episodes + 1):
            # Collect episode
            train_reward, train_length, _ = self.collect_episode(deterministic=False)

            # Training update
            if (
                episode % collect_interval == 0
                and self.memory.episodes >= self.config.batch_size
            ):  # Changed from len(self.memory)
                loss_info = []
                for _ in range(train_steps_per_collect):
                    # Sample batch
                    batch = self.memory.sample(self.config.batch_size, chunk_size)

                    # Update agent
                    losses = self.agent.update_parameters(batch, 1)
                    loss_info.extend(losses)

                # Log training losses
                if loss_info:
                    avg_losses = np.mean(loss_info, axis=0)
                    loss_metrics = {
                        "losses/observation": avg_losses[0],
                        "losses/reward": avg_losses[1],
                        "losses/kl": avg_losses[2],
                        "losses/pcont": avg_losses[3],
                        "losses/actor": avg_losses[4],
                        "losses/value": avg_losses[5],
                    }
                    metrics.update(loss_metrics)

            # Log episode metrics
            episode_metrics = {
                "train/reward": train_reward,
                "train/length": train_length,
                "train/episode": episode,
                "memory/size": self.memory.episodes,  # Changed from len(self.memory)
            }
            metrics.update(episode_metrics)

            print(
                f"Episode {episode}: Reward={train_reward:.2f}, Length={train_length}"
            )

            # Evaluation
            if episode % eval_interval == 0:
                eval_metrics = self.evaluate()
                eval_metrics_prefixed = {
                    f"eval/{k}": v for k, v in eval_metrics.items()
                }
                metrics.update(eval_metrics_prefixed)

                print(
                    f"Eval at episode {episode}: "
                    f"Reward={eval_metrics['reward_mean']:.2f}±{eval_metrics['reward_std']:.2f}"
                )

            # Save model
            if episode % save_interval == 0:
                self.save_checkpoint(episode)

                # collect one deterministic episode (does not append to memory)
                _, _, frames = self.collect_episode(deterministic=True)
                if frames:
                    # stack if list of tensors/arrays
                    if isinstance(frames, list) and len(frames) > 0:
                        first = frames[0]
                        if isinstance(first, torch.Tensor):
                            frames_arr = torch.stack(frames, dim=0).cpu().numpy()
                        else:
                            frames_arr = np.stack(frames, axis=0)
                    else:
                        frames_arr = frames

                    # convert to numpy if torch tensor
                    if isinstance(frames_arr, torch.Tensor):
                        frames_arr = frames_arr.detach().cpu().numpy()

                    # remove singleton extra dim (e.g. (T, 1, C, H, W) -> (T, C, H, W))
                    if frames_arr.ndim == 5 and frames_arr.shape[1] == 1:
                        frames_arr = frames_arr.squeeze(1)

                    # Save with unique name per episode
                    save_video(frames_arr, self.results_dir, f"episode_{episode}")

        print("Training completed!")
        return self.results_dir

    def save_checkpoint(self, episode):
        """Save model checkpoint."""
        checkpoint = {
            "transition_model": self.agent.transition_model.state_dict(),
            "observation_model": self.agent.observation_model.state_dict(),
            "reward_model": self.agent.reward_model.state_dict(),
            "encoder": self.agent.encoder.state_dict(),
            "actor_model": self.agent.actor_model.state_dict(),
            "value_model": self.agent.value_model.state_dict(),
            "world_optimizer": self.agent.world_optimizer.state_dict(),
            "actor_optimizer": self.agent.actor_optimizer.state_dict(),
            "value_optimizer": self.agent.value_optimizer.state_dict(),
            "episode": episode,
            "config": self.config._asdict(),
        }

        checkpoint_path = os.path.join(self.results_dir, f"checkpoint_{episode}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.agent.transition_model.load_state_dict(checkpoint["transition_model"])
        self.agent.observation_model.load_state_dict(checkpoint["observation_model"])
        self.agent.reward_model.load_state_dict(checkpoint["reward_model"])
        self.agent.encoder.load_state_dict(checkpoint["encoder"])
        self.agent.actor_model.load_state_dict(checkpoint["actor_model"])
        self.agent.value_model.load_state_dict(checkpoint["value_model"])

        if "world_optimizer" in checkpoint:
            self.agent.world_optimizer.load_state_dict(checkpoint["world_optimizer"])
            self.agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.agent.value_optimizer.load_state_dict(checkpoint["value_optimizer"])

        print(f"Checkpoint loaded from episode {checkpoint.get('episode', 'unknown')}")

    def close(self):
        """Clean up resources."""
        self.env.close()


# Simple usage example
if __name__ == "__main__":
    # Create and train DreamerV1 on HalfCheetah
    dreamer = DreamerV1(
        env="HalfCheetah-v4",
        symbolic=False,  # Use image observations
        bit_depth=5,
        max_episode_length=1000,
        action_repeat=2,
    )

    # Train the model
    dreamer.train(episodes=1000, warmup_episodes=5, eval_interval=50, save_interval=100)

    # Clean up
    dreamer.close()
