from typing import Any
import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from functools import partial
import os

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from torchwm.utils.utils import (
    preprocess_img,
    bottle,
    TensorBoardMetrics,
    save_video,
    flatten_dict,
    postprocess_img,
    normalize_frames_for_saving,
)
from torchwm.memory.planet_memory import Memory, Episode
from torchwm.models.rssm import RecurrentStateSpaceModel
from torchwm.controller.rssm_policy import RSSMPolicy
from torchwm.controller.rollout_generator import RolloutGenerator


def train(
    memory: Any,
    rssm: Any,
    optimizer: Any,
    device: torch.device,
    N: int = 32,
    H: int = 50,
    beta: float = 1.0,
    grads: bool = False,
) -> dict:
    """
    Training implementation as indicated in:
    Learning Latent Dynamics for Planning from Pixels
    arXiv:1811.04551

    (a.) The Standard Variational Bound Method
        using only single step predictions.
    """
    free_nats = torch.ones(1, device=device) * 3.0
    (x, u, r, t), lengths = memory.sample(N, H, time_first=True)
    x, u, r, t = [
        torch.from_numpy(np.ascontiguousarray(arr)).to(device).float()
        for arr in (x, u, r, t)
    ]
    preprocess_img(x, depth=5)

    e_t = bottle(rssm.encoder, x)
    h_t, s_t = rssm.get_init_state(e_t[0])
    states, priors, posteriors, posterior_samples = [], [], [], []

    for i, a_t in enumerate(torch.unbind(u, dim=0)):
        h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
        states.append(h_t)
        priors.append(rssm.state_prior(h_t))
        posteriors.append(rssm.state_posterior(h_t, e_t[i + 1]))
        m, s = posteriors[-1]
        posterior_samples.append(Normal(m, s).rsample())
        s_t = posterior_samples[-1]

    prior_mean = torch.stack([p[0] for p in priors])
    prior_std = torch.stack([p[1] for p in priors])
    posterior_mean = torch.stack([p[0] for p in posteriors])
    posterior_std = torch.stack([p[1] for p in posteriors])
    prior_dist = Normal(prior_mean, prior_std)
    posterior_dist = Normal(posterior_mean, posterior_std)
    states_stacked = torch.stack(states)
    posterior_samples_stacked = torch.stack(posterior_samples)

    rec_loss = (
        F.mse_loss(
            bottle(rssm.decoder, states_stacked, posterior_samples_stacked),
            x[1:],
            reduction="none",
        )
        .sum((2, 3, 4))
        .mean()
    )

    kld_loss = (
        kl_divergence(posterior_dist, prior_dist).sum(-1).clamp(min=free_nats).mean()
    )

    rew_loss = F.mse_loss(
        bottle(rssm.pred_reward, states_stacked, posterior_samples_stacked), r
    )

    optimizer.zero_grad()
    loss = beta * kld_loss + rec_loss + rew_loss
    loss.backward()  # type: ignore[no-untyped-call]
    nn.utils.clip_grad_norm_(rssm.parameters(), 1000.0, norm_type=2)
    optimizer.step()

    metrics = {
        "losses": {
            "kl": kld_loss.item(),
            "reconstruction": rec_loss.item(),
            "reward_pred": rew_loss.item(),
        },
    }
    if grads:
        metrics["grad_norms"] = {
            k: 0 if v.grad is None else v.grad.norm().item()
            for k, v in rssm.named_parameters()
        }
    return metrics


def main() -> None:
    """Example PlaNet/RSSM training script with rollout collection and evaluation.

    Builds environment/model/policy objects, iteratively trains on replayed
    episodes, and periodically saves videos and checkpoints.
    """
    env: Any = None
    try:
        env = RolloutGenerator
    except Exception:
        pass

    env = RolloutGenerator
    env = __import__(
        "torchwm.utils.utils", fromlist=["TorchImageEnvWrapper"]
    ).TorchImageEnvWrapper("Pendulum-v1", bit_depth=5)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, using CPU")
    rssm_model = RecurrentStateSpaceModel(env.action_size).to(device)
    optimizer = torch.optim.Adam(rssm_model.parameters(), lr=1e-3, eps=1e-4)

    policy = RSSMPolicy(
        model=rssm_model,
        planning_horizon=20,
        num_candidates=1000,
        num_iterations=10,
        top_candidates=100,
        device=device,
    )

    rollout_gen = RolloutGenerator(
        env,
        device,
        policy=policy,
        episode_gen=lambda: Episode(partial(postprocess_img, depth=5)),
        max_episode_steps=100,
    )

    mem = Memory(100)
    mem.append(rollout_gen.rollout_n(1, random_policy=True))
    res_dir = "results/"
    summary = TensorBoardMetrics(f"{res_dir}/")

    for i in trange(2, desc="Epoch", leave=False):
        metrics: dict[str, Any] = {}
        for _ in trange(150, desc="Iter ", leave=False):
            train_metrics = train(mem, rssm_model.train(), optimizer, device)
            for k, v in flatten_dict(train_metrics).items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)
                metrics[f"{k}_mean"] = np.array(metrics[k]).mean()

        summary.update(metrics)
        mem.append([rollout_gen.rollout_once(explore=True)])
        eval_episode, eval_frames, eval_metrics = rollout_gen.rollout_eval()
        mem.append([eval_episode])
        # normalize frames to (T,H,W,3) float in [0,1] before saving
        safe_frames = normalize_frames_for_saving(eval_frames)
        save_video(safe_frames, res_dir, f"vid_{i + 1}")
        summary.update(eval_metrics)

        if (i + 1) % 25 == 0:
            torch.save(rssm_model.state_dict(), f"{res_dir}/ckpt_{i + 1}.pth")

    if os.getenv("TRAIN_RSSM_DEBUG", "0") == "1":
        pdb.set_trace()


if __name__ == "__main__":
    main()
