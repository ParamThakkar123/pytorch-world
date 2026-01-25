import torch.nn.functional as F
import torch


def conv_vae_loss_fn(reconst, x, mu, logsigma):
    bce = F.mse_loss(reconst, x, size_average=False)
    kld = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return bce + kld
