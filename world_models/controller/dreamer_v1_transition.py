import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TransitionModel(nn.Module):
    __constants__ = ["min_std_dev"]

    def __init__(
        self,
        belief_size,
        state_size,
        action_size,
        hidden_size,
        embedding_size,
        activation_function="relu",
        min_std_dev=0.1,
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(
            belief_size + embedding_size, hidden_size
        )
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

    def forward(
        self,
        prev_state: torch.Tensor,
        actions: torch.Tensor,
        prev_belief: torch.Tensor,
        observations: Optional[torch.Tensor] = None,
        nonterminals: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        T = actions.size(0)
        beliefs = []
        prior_states = []
        prior_means = []
        prior_std_devs = []

        if observations is not None:
            post_states = []
            post_means = []
            post_std_devs = []

        belief = prev_belief
        state = prev_state

        for t in range(T):
            # mask for episode boundaries
            if nonterminals is not None and t > 0:
                state = state * nonterminals[t - 1]

            # GRU belief update
            sa_embed = self.act_fn(
                self.fc_embed_state_action(torch.cat([state, actions[t]], dim=1))
            )
            belief = self.rnn(sa_embed, belief)

            # PRIOR p(z_t | belief)
            h_prior = self.act_fn(self.fc_embed_belief_prior(belief))
            prior_mean, prior_std_raw = torch.chunk(
                self.fc_state_prior(h_prior), 2, dim=1
            )
            prior_std = F.softplus(prior_std_raw) + self.min_std_dev
            prior_sample = prior_mean + prior_std * torch.randn_like(prior_mean)

            # Store prior
            prior_means.append(prior_mean)
            prior_std_devs.append(prior_std)
            prior_states.append(prior_sample)
            beliefs.append(belief)

            # POSTERIOR: only if observations present
            if observations is not None:
                obs_t = observations[t]
                h_post = self.act_fn(
                    self.fc_embed_belief_posterior(torch.cat([belief, obs_t], dim=1))
                )
                post_mean, post_std_raw = torch.chunk(
                    self.fc_state_posterior(h_post), 2, dim=1
                )
                post_std = F.softplus(post_std_raw) + self.min_std_dev
                post_sample = post_mean + post_std * torch.randn_like(post_mean)

                post_means.append(post_mean)
                post_std_devs.append(post_std)
                post_states.append(post_sample)

                state = post_sample
            else:
                state = prior_sample

        beliefs = torch.stack(beliefs, dim=0)
        prior_states = torch.stack(prior_states, dim=0)
        prior_means = torch.stack(prior_means, dim=0)
        prior_std_devs = torch.stack(prior_std_devs, dim=0)

        if observations is None:
            return [beliefs, prior_states, prior_means, prior_std_devs]

        post_states = torch.stack(post_states, dim=0)
        post_means = torch.stack(post_means, dim=0)
        post_std_devs = torch.stack(post_std_devs, dim=0)

        return [
            beliefs,
            prior_states,
            prior_means,
            prior_std_devs,
            post_states,
            post_means,
            post_std_devs,
        ]
