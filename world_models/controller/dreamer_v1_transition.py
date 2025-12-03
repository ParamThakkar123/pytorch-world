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
        self.modules = [
            self.fc_embed_state_action,
            self.fc_embed_belief_prior,
            self.fc_state_prior,
            self.fc_embed_belief_posterior,
            self.fc_state_posterior,
        ]

    def forward(
        self,
        prev_state: torch.Tensor,
        actions: torch.Tensor,
        prev_belief: torch.Tensor,
        observations: Optional[torch.Tensor] = None,
        nonterminals: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
                torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        """
        T = actions.size(0) + 1
        (
            beliefs,
            prior_states,
            prior_means,
            prior_std_devs,
            posterior_states,
            posterior_means,
            posterior_std_devs,
        ) = (
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
            [torch.empty(0)] * T,
        )
        beliefs[0], prior_states[0], posterior_states[0] = (
            prev_belief,
            prev_state,
            prev_state,
        )
        for t in range(T - 1):
            _state = prior_states[t] if observations is None else posterior_states[t]
            _state = (
                _state
                if (nonterminals is None or t == 0)
                else _state * nonterminals[t - 1]
            )
            hidden = self.act_fn(
                self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1))
            )
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(
                self.fc_state_prior(hidden), 2, dim=1
            )
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[
                t + 1
            ] * torch.randn_like(prior_means[t + 1])
            if observations is not None:
                t_ = t - 1
                hidden = self.act_fn(
                    self.fc_embed_belief_posterior(
                        torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)
                    )
                )
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(
                    self.fc_state_posterior(hidden), 2, dim=1
                )
                posterior_std_devs[t + 1] = (
                    F.softplus(_posterior_std_dev) + self.min_std_dev
                )
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[
                    t + 1
                ] * torch.randn_like(posterior_means[t + 1])
        hidden = [
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
        ]
        if observations is not None:
            hidden += [
                torch.stack(posterior_states[1:], dim=0),
                torch.stack(posterior_means[1:], dim=0),
                torch.stack(posterior_std_devs[1:], dim=0),
            ]
        return hidden
