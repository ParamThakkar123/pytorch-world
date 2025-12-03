import os
import numpy as np
import torch
import tqdm
from torchvision.utils import make_grid, save_image

from world_models.memory.dreamer_memory import ExperienceReplay
from world_models.controller.dreamer_v1_agent import Dreamer
from world_models.envs.dreamer_envs import EnvBatcher
from world_models.utils.dreamer_utils import lineplot, write_video


class DreamerV1:
    def __init__(
        self,
        env,
        *,
        device=None,
        bit_depth=5,
        state_size=30,
        belief_size=200,
        embedding_size=1024,
        hidden_size=300,
        memory_size=100000,
        batch_size=50,
        actor_lr=8e-5,
        value_lr=8e-5,
        world_lr=6e-4,
        free_nats=3.0,
        reward_scale=5.0,
        pcont_scale=5.0,
        grad_clip_norm=100.0,
        planning_horizon=15,
        discount=0.99,
        disclam=0.95,
        action_repeat=2,
        results_dir="results/dreamer_v1",
        headless=True,
        render=False,
        seed=1,
        pcont=False,
        with_logprob=False,
        symbolic=False,
        dense_act="elu",
        cnn_act="relu",
        expl_amount=0.3,
    ):
        """
        Minimal, clean, Planet-style DreamerV1 interface.
        """

        self.env = env
        self.seed = seed
        self.bit_depth = bit_depth
        self.render = render
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Observation & Action sizes from env
        self.observation_size = env.observation_size
        self.action_size = env.action_size

        # Replay buffer
        self.replay = ExperienceReplay(
            memory_size,
            False,
            self.observation_size,
            self.action_size,
            bit_depth,
            device,
        )

        # Dreamer agent
        class Args:
            pass

        args = Args()
        args.belief_size = belief_size
        args.state_size = state_size
        args.embedding_size = embedding_size
        args.hidden_size = hidden_size
        args.observation_size = self.observation_size
        args.action_size = self.action_size
        args.batch_size = batch_size
        args.reward_scale = reward_scale
        args.pcont_scale = pcont_scale
        args.bit_depth = bit_depth
        args.world_lr = world_lr
        args.actor_lr = actor_lr
        args.value_lr = value_lr
        args.grad_clip_norm = grad_clip_norm
        args.free_nats = free_nats
        args.discount = discount
        args.disclam = disclam
        args.planning_horizon = planning_horizon
        args.pcont = pcont
        args.with_logprob = with_logprob
        args.device = device
        args.learning_rate_schedule = 0
        args.adam_epsilon = 1e-7
        args.symbolic = symbolic
        args.dense_act = dense_act
        args.cnn_act = cnn_act
        args.expl_amount = expl_amount

        self.agent = Dreamer(args)
        self.action_repeat = action_repeat
        self.max_episode_steps = env.max_episode_length

        self.metrics = {
            "train_rewards": [],
            "test_rewards": [],
            "episodes": [],
            "env_steps": [],
        }

    # -------------------------------------------------------------------------
    # Warmup with random policy
    # -------------------------------------------------------------------------
    def warmup(self, n_episodes=5, random_policy=True):
        print(f"\n[Warmup] Collecting {n_episodes} episodes…")

        for _ in tqdm.tqdm(range(1, n_episodes + 1)):
            _ = self.env.reset()
            done = False

            while not done:
                if random_policy:
                    action = self.env.sample_random_action()
                else:
                    action = torch.zeros(1, self.action_size).to(self.device)

                next_obs, reward, done = self.env.step(action)
                self.replay.append(next_obs, action.cpu(), reward, done)

        print("[Warmup] Done.\n")

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    def train(
        self,
        *,
        epochs=200,
        steps_per_epoch=150,
        batch_size=50,
        chunk_size=50,
        save_every=25,
    ):
        print("\n=== TRAINING DreamerV1 ===")

        for epoch in range(1, epochs + 1):

            # --------------------------
            # Optimize Dreamer world model + actor + value
            # --------------------------
            for _ in tqdm.tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}"):
                batch = self.replay.sample(batch_size, chunk_size)
                self.agent.update_parameters(batch)

            # --------------------------
            # Collect rollout
            # --------------------------
            reward = self._collect_episode()
            self.metrics["train_rewards"].append(reward)
            self.metrics["episodes"].append(epoch)

            # lineplot: training reward curve
            lineplot(
                self.metrics["episodes"],
                self.metrics["train_rewards"],
                "train_reward",
                self.results_dir,
            )

            print(f"[Epoch {epoch}] Reward = {reward}")

            # --------------------------
            # Save model checkpoint
            # --------------------------
            if epoch % save_every == 0:
                self._save_checkpoint(epoch)

                # Also save reconstructed frames (for debugging)
                self._save_reconstruction()

        print("\n=== TRAINING COMPLETE ===")

    # -------------------------------------------------------------------------
    # Collect a training episode
    # -------------------------------------------------------------------------
    def _collect_episode(self):
        obs = self.env.reset()
        total_reward = 0

        belief = torch.zeros(1, self.agent.args.belief_size, device=self.device)
        state = torch.zeros(1, self.agent.args.state_size, device=self.device)
        action = torch.zeros(1, self.action_size, device=self.device)

        for _ in range(self.max_episode_steps // self.action_repeat):
            belief, state = self.agent.infer_state(
                obs.to(self.device), action, belief, state
            )
            action = self.agent.select_action((belief, state), deterministic=False)

            next_obs, reward, done = self.env.step(action.cpu())

            self.replay.append(next_obs, action.cpu(), reward, done)
            total_reward += reward
            obs = next_obs

            if done:
                break

        return total_reward

    # -------------------------------------------------------------------------
    # Save grid reconstruction using make_grid + save_image
    # -------------------------------------------------------------------------
    def _save_reconstruction(self):
        """Save last batch of reconstructed frames for debugging."""
        obs = self.env.reset().unsqueeze(0).to(self.device)
        belief = torch.zeros(1, self.agent.args.belief_size, device=self.device)
        state = torch.zeros(1, self.agent.args.state_size, device=self.device)
        action = torch.zeros(1, self.action_size, device=self.device)

        belief, state = self.agent.infer_state(obs, action, belief, state)
        recon = self.agent.observation_model(belief, state).cpu()

        grid = make_grid(torch.cat([obs.cpu(), recon], dim=3), nrow=1)
        save_image(grid + 0.5, os.path.join(self.results_dir, "reconstruction.png"))

    # -------------------------------------------------------------------------
    # Evaluation using EnvBatcher + write_video
    # -------------------------------------------------------------------------
    def evaluate(self, episodes=5):
        batch_env = EnvBatcher(
            type(self.env),  # same env class
            self.env.init_args,  # args to recreate env
            self.env.init_kwargs,
            episodes,
        )

        frames = []
        rewards = np.zeros(episodes)

        obs = batch_env.reset()

        belief = torch.zeros(episodes, self.agent.args.belief_size, device=self.device)
        state = torch.zeros(episodes, self.agent.args.state_size, device=self.device)
        action = torch.zeros(episodes, self.action_size, device=self.device)

        for _ in tqdm.tqdm(range(self.max_episode_steps // self.action_repeat)):
            belief, state = self.agent.infer_state(
                obs.to(self.device), action, belief, state
            )
            action = self.agent.select_action((belief, state), deterministic=True)

            next_obs, reward, done = batch_env.step(action.cpu())
            rewards += reward.numpy()

            # write video frames
            recon = self.agent.observation_model(belief, state).cpu()
            frame = make_grid(torch.cat([obs, recon], dim=3) + 0.5, nrow=5)
            frames.append(frame.numpy())

            obs = next_obs
            if done.sum().item() == episodes:
                break

        write_video(frames, "eval_rollout", self.results_dir)
        batch_env.close()

        avg_reward = rewards.mean()
        print(f"[Eval] Reward = {avg_reward}")

        return avg_reward

    # -------------------------------------------------------------------------
    # Save model checkpoint
    # -------------------------------------------------------------------------
    def _save_checkpoint(self, epoch):
        ckpt = {
            "transition_model": self.agent.transition_model.state_dict(),
            "observation_model": self.agent.observation_model.state_dict(),
            "reward_model": self.agent.reward_model.state_dict(),
            "encoder": self.agent.encoder.state_dict(),
            "actor_model": self.agent.actor_model.state_dict(),
            "value_model": self.agent.value_model.state_dict(),
        }

        path = os.path.join(self.results_dir, f"model_{epoch}.pth")
        torch.save(ckpt, path)
        print(f"[Checkpoint] Saved model at {path}")
