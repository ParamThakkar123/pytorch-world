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

from world_models.utils.utils import (
    preprocess_img,
    bottle,
    TensorBoardMetrics,
    save_video,
    flatten_dict,
    postprocess_img,
)
from world_models.memory.planet_memory import Memory, Episode
from world_models.models.rssm import RecurrentStateSpaceModel
from world_models.controller.rssm_policy import RSSMPolicy
from world_models.controller.rollout_generator import RolloutGenerator


def train(memory, rssm, optimizer, device, N=32, H=50, beta=1.0, grads=False):
    """
    Training implementation as indicated in:
    Learning Latent Dynamics for Planning from Pixels
    arXiv:1811.04551

    (a.) The Standard Variational Bound Method
        using only single step predictions.
    """
    free_nats = torch.ones(1, device=device) * 3.0
    (x, u, r, t), lengths = memory.sample(N, H, time_first=True)
    x, u, r, t = [torch.tensor(arr).float().to(device) for arr in (x, u, r, t)]
    preprocess_img(x, depth=5)

    e_t = bottle(rssm.encoder, x)
    h_t, s_t = rssm.get_init_state(e_t[0])
    states, priors, posteriors, posterior_samples = [], [], [], []

    for i, a_t in enumerate(torch.unbind(u, dim=0)):
        h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
        states.append(h_t)
        priors.append(rssm.state_prior(h_t))
        posteriors.append(rssm.state_posterior(h_t, e_t[i + 1]))
        posterior_samples.append(Normal(*posteriors[-1]).rsample())
        s_t = posterior_samples[-1]

    prior_dist = Normal(*map(torch.stack, zip(*priors)))
    posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
    states = torch.stack(states)
    posterior_samples = torch.stack(posterior_samples)

    rec_loss = (
        F.mse_loss(
            bottle(rssm.decoder, states, posterior_samples), x[1:], reduction="none"
        )
        .sum((2, 3, 4))
        .mean()
    )

    kld_loss = torch.max(
        kl_divergence(posterior_dist, prior_dist).sum(-1), free_nats
    ).mean()

    rew_loss = F.mse_loss(bottle(rssm.pred_reward, states, posterior_samples), r)

    optimizer.zero_grad()
    loss = beta * kld_loss + rec_loss + rew_loss
    loss.backward()
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


def main():
    env = None
    try:
        env = RolloutGenerator
    except Exception:
        pass

    env = RolloutGenerator
    env = __import__(
        "world_models.utils.utils", fromlist=["TorchImageEnvWrapper"]
    ).TorchImageEnvWrapper("Pendulum-v1", bit_depth=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    for i in trange(10, desc="Epoch", leave=False):
        metrics = {}
        for _ in trange(150, desc="Iter ", leave=False):
            train_metrics = train(mem, rssm_model.train(), optimizer, device)
            for k, v in flatten_dict(train_metrics).items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)
                metrics[f"{k}_mean"] = np.array(metrics[k]).mean()

        summary.update(metrics)
        mem.append(rollout_gen.rollout_once(explore=True))
        eval_episode, eval_frames, eval_metrics = rollout_gen.rollout_eval()
        mem.append(eval_episode)
        save_video(eval_frames, res_dir, f"vid_{i+1}")
        summary.update(eval_metrics)

        if (i + 1) % 25 == 0:
            torch.save(rssm_model.state_dict(), f"{res_dir}/ckpt_{i+1}.pth")

    if os.getenv("TRAIN_RSSM_DEBUG", "0") == "1":
        pdb.set_trace()


if __name__ == "__main__":
    main()
