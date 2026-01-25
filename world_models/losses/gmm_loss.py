import torch
from torch.distributions.normal import Normal


def gmm_loss(
    latent_next_obs: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    logpi: torch.Tensor,
    reduce: bool = True,
):
    latent_next_obs = latent_next_obs.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(latent_next_obs)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs
    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)
    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return -torch.mean(log_prob)
    return -log_prob
