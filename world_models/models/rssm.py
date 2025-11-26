import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from world_models.vision.encoder import CNNEncoder
from world_models.vision.decoder import CNNDecoder

class RecurrentStateSpaceModel(nn.Module):
    """
    A Recurrent State Space Model (RSSM) for modeling latent dynamics in sequential data.
    """
    def __init__(self, action_size, state_size=200, latent_size=30, hidden_size=200, embed_size=1024, activation_function='relu'):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.act_fn = getattr(F, activation_function)
        self.encoder = CNNEncoder(embed_size)
        self.decoder = CNNDecoder(state_size, latent_size, embed_size)
        self.grucell = nn.GRUCell(state_size, state_size)
        self.lat_act_layer = nn.Linear(latent_size + action_size, state_size)
        self.fc_prior_1 = nn.Linear(state_size, hidden_size)
        self.fc_prior_m = nn.Linear(hidden_size, latent_size)
        self.fc_prior_s = nn.Linear(hidden_size, latent_size)
        self.fc_posterior_1 = nn.Linear(state_size + embed_size, hidden_size)
        self.fc_posterior_m = nn.Linear(hidden_size, latent_size)
        self.fc_posterior_s = nn.Linear(hidden_size, latent_size)
        self.fc_reward_1 = nn.Linear(state_size + latent_size, hidden_size)
        self.fc_reward_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_reward_3 = nn.Linear(hidden_size, 1)

    def get_init_state(self, enc, h_t=None, s_t=None, a_t=None, mean=False):
        """Returns the initial posterior given the observation."""
        N, dev = enc.size(0), enc.device
        h_t = torch.zeros(N, self.state_size).to(dev) if h_t is None else h_t
        s_t = torch.zeros(N, self.latent_size).to(dev) if s_t is None else s_t
        a_t = torch.zeros(N, self.action_size).to(dev) if a_t is None else a_t
        h_tp1 = self.deterministic_state_fwd(h_t, s_t, a_t)
        if mean:
            s_tp1 = self.state_posterior(h_t, enc, sample=True)
        else:
            s_tp1, _ = self.state_posterior(h_t, enc)
        return h_tp1, s_tp1
    
    def deterministic_state_fwd(self, h_t, s_t, a_t):
        """Returns the deterministic state given the previous states and actions"""
        h = torch.cat([s_t, a_t], dim=1)
        h = self.act_fn(self.lat_act_layer(h))
        h_tp1 = self.grucell(h, h_t)
        return h_tp1
    
    def state_prior(self, h_t, sample=False):
        """Returns the prior distribution over the latent state given the deterministic state"""
        z = self.act_fn(self.fc_prior_1(h_t))
        m = self.fc_prior_m(z)
        s = F.softplus(self.fc_prior_s(z)) + 0.1
        if sample:
            return m + torch.rand_like(m) * s
        return m, s
    
    def state_posterior(self, h_t, e_t, sample=False):
        """Returns the state prior given the deterministic state and obs"""
        z = torch.cat([h_t, e_t], dim=1)
        z = self.act_fn(self.fc_posterior_1(z))
        m = self.fc_posterior_m(z)
        s = F.softplus(self.fc_posterior_s(z)) + 0.1
        if sample:
            return m + torch.rand_like(m) * s
        return m, s
    
    def pred_reward(self, h_t, s_t):
        r = self.act_fn(self.fc_reward_1(torch.cat([h_t, s_t], dim=-1)))
        r = self.act_fn(self.fc_reward_2(r))
        r = self.fc_reward_3(r)
        return r.squeeze()
    
    def rollout_prior(self, act, h_t, s_t):
        states, latents = [], []
        for a_t in torch.unbind(act, dim=0):
            h_t = self.deterministic_state_fwd(h_t, s_t, a_t)
            s_t = self.state_prior(h_t)
            states.append(h_t)
            latents.append(s_t)
            Normal(*map(torch.stack, zip(*s_t)))
        return torch.stack(states), torch.stack(latents)