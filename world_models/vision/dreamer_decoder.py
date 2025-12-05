import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution, Bernoulli
from torch.distributions.independent import Independent
import numpy as np
import torch.distributions as distributions
from torch.distributions import constraints
import torch.nn.functional as F


_str_to_activation = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}


class TanhBijector(distributions.Transform):

    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = constraints.real
        self.codomain = constraints.interval(-1.0, 1.0)

    @property
    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = self.atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2) - x - F.softplus(-2.0 * x))


class ConvDecoder(nn.Module):

    def __init__(self, stoch_size, deter_size, output_shape, activation, depth=32):

        super().__init__()

        self.output_shape = output_shape
        self.depth = depth
        self.kernels = [5, 5, 6, 6]
        self.act_fn = _str_to_activation[activation]

        self.dense = nn.Linear(stoch_size + deter_size, 32 * self.depth)

        layers = []
        for i, kernel_size in enumerate(self.kernels):
            in_ch = (
                32 * self.depth
                if i == 0
                else self.depth * (2 ** (len(self.kernels) - 1 - i))
            )
            out_ch = (
                output_shape[0]
                if i == len(self.kernels) - 1
                else self.depth * (2 ** (len(self.kernels) - 2 - i))
            )
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=2))
            if i != len(self.kernels) - 1:
                layers.append(self.act_fn)

        self.convtranspose = nn.Sequential(*layers)

    def forward(self, features):
        out_batch_shape = features.shape[:-1]
        out = self.dense(features)
        out = torch.reshape(out, [-1, 32 * self.depth, 1, 1])
        out = self.convtranspose(out)
        mean = torch.reshape(out, (*out_batch_shape, *self.output_shape))

        out_dist = Independent(Normal(mean, 1), len(self.output_shape))

        return out_dist


class DenseDecoder(nn.Module):

    def __init__(
        self, stoch_size, deter_size, output_shape, n_layers, units, activation, dist
    ):

        super().__init__()

        self.input_size = stoch_size + deter_size
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.units = units
        self.act_fn = _str_to_activation[activation]
        self.dist = dist

        layers = []

        for i in range(self.n_layers):
            in_ch = self.input_size if i == 0 else self.units
            out_ch = self.units
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(self.act_fn)

        layers.append(nn.Linear(self.units, int(np.prod(self.output_shape))))

        self.model = nn.Sequential(*layers)

    def forward(self, features):

        out = self.model(features)

        if self.dist == "normal":
            return Independent(Normal(out, 1), len(self.output_shape))
        if self.dist == "binary":
            return Independent(Bernoulli(logits=out), len(self.output_shape))
        if self.dist == "none":
            return out

        raise NotImplementedError(self.dist)


class SampleDist:

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        sample = self._dist.rsample(self._samples)
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
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

    def sample(self):
        return self._dist.sample()


class ActionDecoder(nn.Module):

    def __init__(
        self,
        action_size,
        stoch_size,
        deter_size,
        n_layers,
        units,
        activation,
        min_std=1e-4,
        init_std=5,
        mean_scale=5,
    ):

        super().__init__()

        self.action_size = action_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.units = units
        self.act_fn = _str_to_activation[activation]
        self.n_layers = n_layers

        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

        layers = []
        for i in range(self.n_layers):
            in_ch = self.stoch_size + self.deter_size if i == 0 else self.units
            out_ch = self.units
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(self.act_fn)

        layers.append(nn.Linear(self.units, 2 * self.action_size))
        self.action_model = nn.Sequential(*layers)

    def forward(self, features, deter=False):

        out = self.action_model(features)
        mean, std = torch.chunk(out, 2, dim=-1)

        raw_init_std = np.log(np.exp(self._init_std) - 1)
        action_mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
        action_std = F.softplus(std + raw_init_std) + self._min_std

        dist = Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())
        dist = Independent(dist, 1)
        dist = SampleDist(dist)

        if deter:
            return dist.mode()
        else:
            return dist.rsample()

    def add_exploration(self, action, action_noise=0.3):

        return torch.clamp(Normal(action, action_noise).rsample(), -1, 1)
