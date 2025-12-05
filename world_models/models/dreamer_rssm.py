import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

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


class RSSM(nn.Module):

    def __init__(
        self,
        action_size,
        stoch_size,
        deter_size,
        hidden_size,
        obs_embed_size,
        activation,
    ):

        super().__init__()

        self.action_size = action_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size  # GRU hidden units
        self.hidden_size = hidden_size  # intermediate fc_layers hidden units
        self.embedding_size = obs_embed_size

        self.act_fn = _str_to_activation[activation]
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)

        self.fc_state_action = nn.Linear(
            self.stoch_size + self.action_size, self.deter_size
        )
        self.fc_embed_prior = nn.Linear(self.deter_size, self.hidden_size)
        self.fc_state_prior = nn.Linear(self.hidden_size, 2 * self.stoch_size)
        self.fc_embed_posterior = nn.Linear(
            self.embedding_size + self.deter_size, self.hidden_size
        )
        self.fc_state_posterior = nn.Linear(self.hidden_size, 2 * self.stoch_size)

    def init_state(self, batch_size, device):

        return dict(
            mean=torch.zeros(batch_size, self.stoch_size).to(device),
            std=torch.zeros(batch_size, self.stoch_size).to(device),
            stoch=torch.zeros(batch_size, self.stoch_size).to(device),
            deter=torch.zeros(batch_size, self.deter_size).to(device),
        )

    def get_dist(self, mean, std):

        distribution = distributions.Normal(mean, std)
        distribution = distributions.independent.Independent(distribution, 1)
        return distribution

    def observe_step(self, prev_state, prev_action, obs_embed, nonterm=1.0):

        prior = self.imagine_step(prev_state, prev_action, nonterm)
        posterior_embed = self.act_fn(
            self.fc_embed_posterior(torch.cat([obs_embed, prior["deter"]], dim=-1))
        )
        posterior = self.fc_state_posterior(posterior_embed)
        mean, std = torch.chunk(posterior, 2, dim=-1)
        std = F.softplus(std) + 0.1
        sample = mean + torch.randn_like(mean) * std

        posterior = {"mean": mean, "std": std, "stoch": sample, "deter": prior["deter"]}
        return prior, posterior

    def imagine_step(self, prev_state, prev_action, nonterm=1.0):

        state_action = self.act_fn(
            self.fc_state_action(
                torch.cat([prev_state["stoch"] * nonterm, prev_action], dim=-1)
            )
        )
        deter = self.rnn(state_action, prev_state["deter"] * nonterm)
        prior_embed = self.act_fn(self.fc_embed_prior(deter))
        mean, std = torch.chunk(self.fc_state_prior(prior_embed), 2, dim=-1)
        std = F.softplus(std) + 0.1
        sample = mean + torch.randn_like(mean) * std

        prior = {"mean": mean, "std": std, "stoch": sample, "deter": deter}
        return prior

    def observe_rollout(self, obs_embed, actions, nonterms, prev_state, horizon):

        priors = []
        posteriors = []

        for t in range(horizon):
            prev_action = actions[t] * nonterms[t]
            prior_state, posterior_state = self.observe_step(
                prev_state, prev_action, obs_embed[t], nonterms[t]
            )
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state

        priors = self.stack_states(priors, dim=0)
        posteriors = self.stack_states(posteriors, dim=0)

        return priors, posteriors

    def imagine_rollout(self, actor, prev_state, horizon):

        rssm_state = prev_state
        next_states = []

        for t in range(horizon):
            action = actor(
                torch.cat([rssm_state["stoch"], rssm_state["deter"]], dim=-1).detach()
            )
            rssm_state = self.imagine_step(rssm_state, action)
            next_states.append(rssm_state)

        next_states = self.stack_states(next_states)
        return next_states

    def stack_states(self, states, dim=0):

        return dict(
            mean=torch.stack([state["mean"] for state in states], dim=dim),
            std=torch.stack([state["std"] for state in states], dim=dim),
            stoch=torch.stack([state["stoch"] for state in states], dim=dim),
            deter=torch.stack([state["deter"] for state in states], dim=dim),
        )

    def detach_state(self, state):

        return dict(
            mean=state["mean"].detach(),
            std=state["std"].detach(),
            stoch=state["stoch"].detach(),
            deter=state["deter"].detach(),
        )

    def seq_to_batch(self, state):

        return dict(
            mean=torch.reshape(
                state["mean"],
                (
                    state["mean"].shape[0] * state["mean"].shape[1],
                    *state["mean"].shape[2:],
                ),
            ),
            std=torch.reshape(
                state["std"],
                (
                    state["std"].shape[0] * state["std"].shape[1],
                    *state["std"].shape[2:],
                ),
            ),
            stoch=torch.reshape(
                state["stoch"],
                (
                    state["stoch"].shape[0] * state["stoch"].shape[1],
                    *state["stoch"].shape[2:],
                ),
            ),
            deter=torch.reshape(
                state["deter"],
                (
                    state["deter"].shape[0] * state["deter"].shape[1],
                    *state["deter"].shape[2:],
                ),
            ),
        )
