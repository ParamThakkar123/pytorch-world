from functools import partial
import os
import torch
import numpy as np

from world_models.models.rssm import RecurrentStateSpaceModel
from world_models.controller.rssm_policy import RSSMPolicy
from world_models.controller.rollout_generator import RolloutGenerator
from world_models.utils.utils import (
    TensorBoardMetrics,
    save_video,
    postprocess_img,
    TorchImageEnvWrapper,
)
from world_models.memory.planet_memory import Memory, Episode
from world_models.training.train_planet import train as planet_train


class Planet:
    """
    High-level Planet wrapper.

    Usage example:
      from world_models.models.planet import Planet
      p = Planet(env='CartPole-v1', bit_depth=5)
      p.train(epochs=50)
    """

    def __init__(
        self,
        env,
        bit_depth=5,
        device=None,
        state_size=200,
        latent_size=30,
        embedding_size=1024,
        memory_size=100,
        policy_cfg=None,
        headless=False,
        max_episode_steps=None,
        action_repeats=1,
        results_dir=None,
    ):
        # allow running without opening windows (useful on servers/CI)
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bit_depth = bit_depth

        # if env is a string id, wrap it to the TorchImageEnvWrapper, otherwise assume it's already env-like
        if isinstance(env, str):
            # pass action_repeats into the wrapper (4th positional param in wrapper)
            self.env = TorchImageEnvWrapper(env, bit_depth, None, action_repeats)
        else:
            self.env = env

        # model / optimizer
        # RecurrentStateSpaceModel signatures vary in this repo; using common constructor:
        self.rssm = RecurrentStateSpaceModel(
            self.env.action_size, state_size, latent_size, embedding_size
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.rssm.parameters(), lr=1e-3, eps=1e-4)

        # policy
        policy_cfg = policy_cfg or {}
        self.policy = RSSMPolicy(
            model=self.rssm,
            planning_horizon=policy_cfg.get("planning_horizon", 20),
            num_candidates=policy_cfg.get("num_candidates", 1000),
            num_iterations=policy_cfg.get("num_iterations", 10),
            top_candidates=policy_cfg.get("top_candidates", 100),
            device=self.device,
        )

        env_max_steps = (
            max_episode_steps or getattr(self.env, "max_episode_steps", None) or 100
        )
        self.rollout_gen = RolloutGenerator(
            self.env,
            self.device,
            policy=self.policy,
            episode_gen=lambda: Episode(partial(postprocess_img, depth=self.bit_depth)),
            max_episode_steps=env_max_steps,
        )

        # memory / logging
        self.memory = Memory(memory_size)
        self.summary = None
        # allow user to set results directory at construction time
        self.results_dir = results_dir or "results/planet"

    def warmup(self, n_episodes=1, random_policy=True):
        """Collect n_episodes of rollouts into memory (used as warmup)."""
        eps = self.rollout_gen.rollout_n(n=n_episodes, random_policy=random_policy)
        self.memory.append(eps)

    def train(
        self,
        epochs=100,
        steps_per_epoch=150,
        batch_size=32,
        H=50,
        beta=1.0,
        save_every=25,
        record_grads=False,
        results_dir=None,
    ):
        """
        High-level training loop. Delegates single-step training to the existing `train` function.

        This mirrors the behavior in world_models/training/train_planet.py but wrapped as a class method.
        """
        # allow caller to override results dir for this training run
        if results_dir is not None:
            self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.summary = TensorBoardMetrics(self.results_dir)

        if len(self.memory.episodes) == 0:
            self.warmup(n_episodes=1, random_policy=True)

        for ep in range(epochs):
            metrics = {}
            for _ in range(steps_per_epoch):
                train_metrics = planet_train(
                    self.memory,
                    self.rssm.train(),
                    self.optimizer,
                    self.device,
                    N=batch_size,
                    H=H,
                    beta=beta,
                    grads=record_grads,
                )
                for k, v in train_metrics.items():
                    if k not in metrics:
                        metrics[k] = []
                    if isinstance(v, dict):
                        for ik, iv in v.items():
                            metrics.setdefault(f"{k}/{ik}", []).append(iv)
                    else:
                        metrics[k].append(v)

            compact = {}
            for k, vs in metrics.items():
                try:
                    compact[k] = np.mean(vs)
                except Exception:
                    compact[k] = vs
            self.summary.update(compact)

            self.memory.append(self.rollout_gen.rollout_once(explore=True))
            eval_episode, eval_frames, eval_metrics = self.rollout_gen.rollout_eval()
            self.memory.append(eval_episode)
            save_video(eval_frames, self.results_dir, f"vid_{ep+1}")
            self.summary.update(eval_metrics)

            if (ep + 1) % save_every == 0:
                torch.save(
                    self.rssm.state_dict(),
                    os.path.join(self.results_dir, f"ckpt_{ep+1}.pth"),
                )

        return self.results_dir
