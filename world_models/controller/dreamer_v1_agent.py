from copy import deepcopy
import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from tqdm import tqdm
from world_models.utils.utils import bottle, cal_returns
from world_models.vision.dreamer_v1_encoder import Encoder
from world_models.observations.dreamer_v1_obs import ObservationModel
from world_models.reward.dreamer_v1_reward import RewardModel
from world_models.controller.dreamer_v1_transition import TransitionModel
from world_models.reward.dreamer_v1_value import ValueModel
from world_models.controller.dreamer_v1_actor import ActorModel
from world_models.controller.dreamer_v1_pcont import PCONTModel


def count_vars(module):
    """count parameters number of module"""
    return sum([np.prod(p.shape) for p in module.parameters()])


class Dreamer:
    def __init__(self, args):
        """
        All paras are passed by args
        :param args: a dict that includes parameters
        """
        super().__init__()
        self.args = args
        # Initialise model parameters randomly
        self.transition_model = TransitionModel(
            args.belief_size,
            args.state_size,
            args.action_size,
            args.hidden_size,
            args.embedding_size,
            args.dense_act,
        ).to(device=args.device)

        self.observation_model = ObservationModel(
            args.symbolic,
            args.observation_size,
            args.belief_size,
            args.state_size,
            args.embedding_size,
            activation_function=(args.dense_act if args.symbolic else args.cnn_act),
        ).to(device=args.device)

        self.reward_model = RewardModel(
            args.belief_size, args.state_size, args.hidden_size, args.dense_act
        ).to(device=args.device)

        self.encoder = Encoder(
            args.symbolic, args.observation_size, args.embedding_size, args.cnn_act
        ).to(device=args.device)

        self.actor_model = ActorModel(
            args.action_size,
            args.belief_size,
            args.state_size,
            args.hidden_size,
            activation_function=args.dense_act,
        ).to(device=args.device)

        self.value_model = ValueModel(
            args.belief_size, args.state_size, args.hidden_size, args.dense_act
        ).to(device=args.device)

        self.pcont_model = PCONTModel(
            args.belief_size, args.state_size, args.hidden_size, args.dense_act
        ).to(device=args.device)

        self.target_value_model = deepcopy(self.value_model)

        for p in self.target_value_model.parameters():
            p.requires_grad = False

        # setup the paras to update
        self.world_param = (
            list(self.transition_model.parameters())
            + list(self.observation_model.parameters())
            + list(self.reward_model.parameters())
            + list(self.encoder.parameters())
        )
        if args.pcont:
            self.world_param += list(self.pcont_model.parameters())

        # setup optimizer
        self.world_optimizer = optim.Adam(self.world_param, lr=args.world_lr)
        self.actor_optimizer = optim.Adam(
            self.actor_model.parameters(), lr=args.actor_lr
        )
        self.value_optimizer = optim.Adam(
            self.value_model.parameters(), lr=args.value_lr
        )

        # setup the free_nat
        self.free_nats = torch.tensor([args.free_nats]).to(device=args.device)

    def process_im(self, image):
        # Resize, put channel first, convert it to a tensor, centre it to [-0.5, 0.5] and add batch dimenstion.

        def preprocess_observation_(observation, bit_depth):
            # Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
            observation.div_(2 ** (8 - bit_depth)).floor_().div_(2**bit_depth).sub_(
                0.5
            )  # Quantise to given bit depth and centre
            observation.add_(
                torch.rand_like(observation).div_(2**bit_depth)
            )  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)

        image = torch.tensor(
            cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(
                2, 0, 1
            ),
            dtype=torch.float32,
        )  # Resize and put channel first

        preprocess_observation_(image, self.args.bit_depth)
        return image.unsqueeze(dim=0)

    def _compute_loss_world(self, state, data):
        (
            beliefs,
            prior_states,
            prior_means,
            prior_std,
            post_states,
            post_means,
            post_std_devs,
        ) = state
        obs, rewards, nonterms = data

        # Observation reconstruction loss
        obs_pred = bottle(self.observation_model, beliefs, post_states)
        obs_loss = (
            ((obs_pred - obs) ** 2)
            .sum(dim=(2, 3, 4) if not self.args.symbolic else 2)
            .mean()
        )

        # Reward prediction loss
        rew_pred = bottle(self.reward_model, beliefs, post_states)
        rew_loss = F.mse_loss(rew_pred, rewards, reduction="mean")

        # KL loss
        prior = Independent(Normal(prior_means, prior_std), 1)
        post = Independent(Normal(post_means, post_std_devs), 1)

        kl = kl_divergence(post, prior)
        kl = torch.max(kl, self.free_nats)  # free nats before averaging
        kl_loss = kl.mean()

        if self.args.pcont:
            pcont_pred = bottle(self.pcont_model, beliefs, post_states)
            # pcont_pred shape: [T*B] needs to be reshaped to [T, B]
            pcont_target = nonterms.squeeze(-1)  # [L, B]
            pcont_loss = F.binary_cross_entropy(pcont_pred, pcont_target)
            pcont_loss = pcont_loss * self.args.pcont_scale
        else:
            pcont_loss = torch.tensor(0.0, device=self.args.device)

        total = (
            obs_loss
            + self.args.reward_scale * rew_loss
            + kl_loss
            + self.args.pcont_scale * pcont_loss
        )
        return total, obs_loss, rew_loss, kl_loss, pcont_loss

    def _compute_loss_actor(self, imag_beliefs, imag_states, imag_ac_logps=None):
        values = bottle(self.value_model, imag_beliefs, imag_states)  # [H+1, B]

        # Predict rewards for steps 1 to H (no reward at t=0)
        rewards = bottle(self.reward_model, imag_beliefs[1:], imag_states[1:])  # [H, B]

        # Predict continuation probability (pcont)
        if self.args.pcont:
            pcont = bottle(
                self.pcont_model, imag_beliefs[1:], imag_states[1:]
            )  # [H, B]
        else:
            pcont = self.args.discount * torch.ones_like(rewards)

        # Bootstrap value from the last imagined state
        bootstrap = values[-1]  # [1, B] — value of final state

        # Compute lambda-returns using your correct cal_returns function
        lambda_returns = cal_returns(
            reward=rewards,  # [H, B, 1]
            value=values[:-1],  # [H, B, 1] — values at steps 0 to H-1
            bootstrap=bootstrap,  # [1, B, 1]
            pcont=pcont,  # [H, B, 1]
            lambda_=self.args.disclam,  # λ from config
        )  # → [H, B, 1]

        # Advantage = lambda_return - value (detach value for stability)
        advantage = lambda_returns - values[:-1].detach()

        # Actor loss: -E[log π(a|s) * A]
        # imag_logps: [H, B] from imagination
        actor_loss = -(imag_ac_logps * advantage).mean()

        return actor_loss

    def _compute_loss_critic(self, imag_beliefs, imag_states, imag_ac_logps=None):

        values = bottle(self.value_model, imag_beliefs, imag_states)  # [H+1, B]
        rewards = bottle(self.reward_model, imag_beliefs[1:], imag_states[1:])  # [H, B]

        if self.args.pcont:
            pcont = bottle(
                self.pcont_model, imag_beliefs[1:], imag_states[1:]
            )  # [H, B]
        else:
            pcont = self.args.discount * torch.ones_like(rewards)

        bootstrap = values[-1]  # [1, B] — value of final state

        lambda_returns = cal_returns(
            reward=rewards,  # [H, B, 1]
            value=values[:-1],  # [H, B, 1] — values at steps 0 to H-1
            bootstrap=bootstrap,  # [1, B, 1]
            pcont=pcont,  # [H, B, 1]
            lambda_=self.args.disclam,  # λ from config
        )  # → [H, B, 1]

        # Value loss: MSE between predicted values (at steps 0 to H-1) and lambda targets
        critic_loss = F.mse_loss(values[:-1], lambda_returns.detach())

        return critic_loss

    def _latent_imagination(self, beliefs, posterior_states, with_logprob=False):
        # Rollout to generate imagined trajectories

        # we always start from the final posterior state of the real-sequence
        start_belief = beliefs[-1]  # [B, D]
        start_state = posterior_states[-1]  # [B, S]

        imag_beliefs = [start_belief]
        imag_states = [start_state]
        imag_logps = [] if with_logprob else None

        for t in range(self.args.planning_horizon):
            action, logp = self.actor_model(
                imag_beliefs[-1],
                imag_states[-1],
                deterministic=False,
                with_logprob=with_logprob,
            )

            if with_logprob:
                imag_logps.append(logp)

            action_seq = action.unsqueeze(0)

            belief_seq, prior_states, _, _ = self.transition_model(
                prev_state=imag_states[-1],  # [B, S]
                actions=action_seq,  # [1, B, A]
                prev_belief=imag_beliefs[-1],  # [B, D]
                observations=None,
                nonterminals=None,
            )

            next_belief = belief_seq[0]
            next_state = prior_states[0]

            imag_beliefs.append(next_belief)
            imag_states.append(next_state)

        imag_beliefs = torch.stack(imag_beliefs, dim=0)  # [H+1, B, D]
        imag_states = torch.stack(imag_states, dim=0)

        if with_logprob:
            imag_logps = torch.stack(imag_logps, dim=0)  # [H, B]
            return imag_beliefs, imag_states, imag_logps

        return imag_beliefs, imag_states, None

    def select_action(self, belief, state, deterministic=False):
        """
        Select an action given observation, belief, and state.
        """
        act, _ = self.actor_model(
            belief, state, deterministic=deterministic, with_logprob=False
        )
        return act

    def infer_state(self, observation, belief, state, action):
        """
        Infer updated belief and state from observation, previous belief/state, and action.
        Uses the transition model to perform posterior update.
        """
        enc = self.encoder(observation).unsqueeze(0)
        act = action.unsqueeze(0)
        nt = torch.ones(observation.shape[0], 1, device=self.args.device).unsqueeze(0)

        # Update belief/state using posterior
        beliefs, _, _, _, post_states, _, _ = self.transition_model(
            state, act, belief, enc, nt
        )
        return beliefs.squeeze(0), post_states.squeeze(0)

    def _soft_update(self, tau=0.01):
        """
        Docstring for _soft_update

        :param self: Description
        :param tau: Description
        """
        for p, tp in zip(
            self.value_model.parameters(), self.target_value_model.parameters()
        ):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * p.data)

    def update_parameters(self, batch, gradient_steps):
        obs, actions, rewards, nonterms = batch
        loss_log = []
        for s in tqdm(range(gradient_steps)):
            init_belief = torch.zeros(
                self.args.batch_size, self.args.belief_size, device=self.args.device
            )
            init_state = torch.zeros(
                self.args.batch_size, self.args.state_size, device=self.args.device
            )

            (
                beliefs,
                prior_states,
                prior_means,
                prior_std_devs,
                post_states,
                post_means,
                post_std_devs,
            ) = self.transition_model(
                init_state,
                actions,
                init_belief,
                bottle(self.encoder, obs),
                nonterms,
            )

            world_total, obs_loss, rew_loss, kl_loss, pcont_loss = (
                self._compute_loss_world(
                    state=(
                        beliefs,
                        prior_states,
                        prior_means,
                        prior_std_devs,
                        post_states,
                        post_means,
                        post_std_devs,
                    ),
                    data=(obs, rewards, nonterms),
                )
            )

            self.world_optimizer.zero_grad()
            world_total.backward()
            nn.utils.clip_grad_norm_(self.world_param, self.args.grad_clip_norm)
            self.world_optimizer.step()

            imag_b, imag_s, imag_logps = self._latent_imagination(
                beliefs, post_states, self.args.with_logprob
            )

            imag_b = imag_b.detach()
            imag_s = imag_s.detach()
            if imag_logps is not None:
                imag_logps = imag_logps.detach()

            # ===== 4. Actor update =====
            actor_loss = self._compute_loss_actor(imag_b, imag_s, imag_logps)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_model.parameters(), self.args.grad_clip_norm
            )
            self.actor_optimizer.step()

            # ===== 5. Critic update =====
            critic_loss = self._compute_loss_critic(imag_b, imag_s, imag_logps)
            self.value_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                self.value_model.parameters(), self.args.grad_clip_norm
            )
            self.value_optimizer.step()

            # ===== 6. Soft update target value =====
            self._soft_update()

            loss_log.append(
                [
                    float(obs_loss.detach()),
                    float(rew_loss.detach()),
                    float(kl_loss.detach()),
                    float(pcont_loss.detach()),
                    float(actor_loss.detach()),
                    float(critic_loss.detach()),
                ]
            )

        return loss_log
