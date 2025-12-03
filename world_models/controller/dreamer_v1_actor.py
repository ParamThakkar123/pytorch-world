import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SampleDist:
    """
    After TransformedDistribution, many methods becomes invalid, therefore, we need to approximate them.
    """

    def __init__(self, dist: torch.distributions.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    @property
    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        # print("dist in mode", sample.shape)
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = (
            torch.argmax(logprob, dim=0)
            .reshape(1, batch_size, 1)
            .expand(1, batch_size, feature_size)
        )
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)


class ActorModel(nn.Module):
    def __init__(
        self,
        action_size,
        belief_size,
        state_size,
        hidden_size,
        mean_scale=5,
        min_std=1e-4,
        init_std=5,
        activation_function="elu",
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 2 * action_size)
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale

    def forward(self, belief, state, deterministic=False, with_logprob=False):
        raw_init_std = np.log(np.exp(self.init_std) - 1)
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=-1)))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        hidden = self.fc5(hidden)
        mean, std = torch.chunk(hidden, 2, dim=-1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = F.softplus(std + raw_init_std) + self.min_std
        dist = torch.distributions.Normal(mean, std)
        transform = [torch.distributions.transforms.TanhTransform()]
        dist = torch.distributions.TransformedDistribution(dist, transform)
        dist = torch.distributions.independent.Independent(dist, 1)
        dist = SampleDist(dist)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        if with_logprob:
            logp_pi = dist.log_prob(action)
        else:
            logp_pi = None

        return action, logp_pi
