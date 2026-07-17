import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Any, Tuple, Optional
import torch.nn.functional as F
from world_models.utils.logging_utils import setup_logging

from world_models.configs.iris_config import IRISConfig
from world_models.models.model_io import (
    apply_config_overrides,
    coerce_config,
    module_summary,
    parameter_count as count_parameters,
    resolve_pretrained_file,
    save_config_next_to_checkpoint,
)
from world_models.vision.iris_encoder import IRISEncoder
from world_models.vision.iris_decoder import IRISDecoder
from world_models.models.iris_transformer import IRISTransformer
from world_models.controller.iris_policy import (
    CNNFeatureExtractor,
)


def compute_lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discounts: torch.Tensor,
    lambda_coef: float = 0.95,
) -> torch.Tensor:
    """Compute λ-return target for value function training.

    Args:
        rewards: Rewards (B, T)
        values: Value estimates (B, T+1)
        discounts: Discount factors (B, T)
        lambda_coef: Lambda parameter for bootstrapping

    Returns:
        lambda_returns: λ-return targets (B, T)
    """
    T = rewards.shape[1]
    lambda_returns = torch.zeros_like(rewards)

    # Start with the last bootstrapped value
    lambda_returns[:, T - 1] = rewards[:, T - 1] + discounts[:, T - 1] * values[:, T]

    # Compute λ-returns backwards
    for t in range(T - 2, -1, -1):
        lambda_returns[:, t] = rewards[:, t] + discounts[:, t] * (
            (1 - lambda_coef) * values[:, t + 1]
            + lambda_coef * lambda_returns[:, t + 1]
        )

    return lambda_returns


class IRISAgent(nn.Module):
    """Complete IRIS Agent with world model and policy.

    Combines:
    - Discrete autoencoder (encoder + decoder)
    - Transformer world model
    - Actor-Critic for policy and value learning
    """

    def __init__(
        self,
        config: IRISConfig,
        action_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.config = coerce_config(IRISConfig, config)
        config = self.config
        self.action_size = action_size
        self.device = device
        self.logger = setup_logging("IRISAgent")
        self.use_amp = bool(
            getattr(config, "use_amp", True)
            and getattr(device, "type", str(device)) == "cuda"
        )

        # === Discrete Autoencoder ===
        self.encoder = IRISEncoder(
            vocab_size=config.vocab_size,
            tokens_per_frame=config.tokens_per_frame,
            embedding_dim=config.token_embedding_dim,
            in_channels=config.frame_channels,
            base_channels=config.encoder_channels,
            frame_shape=config.get_frame_shape(),
        ).to(device)

        self.decoder = IRISDecoder(
            vocab_size=config.vocab_size,
            embedding_dim=config.token_embedding_dim,
            base_channels=config.decoder_depth,
            out_channels=config.frame_channels,
            frame_shape=config.get_frame_shape(),
        ).to(device)

        # === Transformer World Model ===
        self.transformer = IRISTransformer(
            vocab_size=config.vocab_size,
            tokens_per_frame=config.tokens_per_frame,
            action_size=action_size,
            embed_dim=config.transformer_embed_dim,
            num_layers=config.transformer_layers,
            num_heads=config.transformer_heads,
            dropout=config.transformer_dropout,
            gradient_checkpointing=getattr(config, "gradient_checkpointing", True),
        ).to(device)

        # === Actor-Critic ===
        # Combine actor and critic with shared CNN features
        self.cnn = CNNFeatureExtractor(
            frame_shape=config.get_frame_shape(),
            output_size=config.actor_hidden_size,
        ).to(device)

        self.lstm = nn.LSTM(
            input_size=config.actor_hidden_size,
            hidden_size=config.actor_hidden_size,
            num_layers=config.actor_layers,
            batch_first=True,
        ).to(device)

        self.actor_head = nn.Linear(config.actor_hidden_size, action_size).to(device)
        self.critic_head = nn.Linear(config.actor_hidden_size, 1).to(device)

        # === Optimizers ===
        self._setup_optimizers()

        self.autoencoder_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.transformer_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ac_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # === Training state ===
        self.global_step = 0
        self.current_epoch = 0

    @classmethod
    def from_config(
        cls,
        config: IRISConfig | dict[str, Any] | str | Path | None = None,
        *,
        action_size: int,
        device: torch.device | str | None = None,
        **overrides: Any,
    ) -> "IRISAgent":
        """Build an IRIS agent from a config object, dict, YAML file, or YAML string."""

        args = apply_config_overrides(coerce_config(IRISConfig, config), overrides)
        torch_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        return cls(args, action_size=action_size, device=torch_device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        action_size: int | None = None,
        device: torch.device | str | None = None,
        config: IRISConfig | dict[str, Any] | str | Path | None = None,
        checkpoint_filename: str | None = None,
        config_filename: str = "config.yaml",
        repo_type: str | None = None,
        revision: str | None = None,
        **overrides: Any,
    ) -> "IRISAgent":
        """Load an IRIS agent checkpoint from a local path/directory or HF Hub."""

        checkpoint_candidates = (
            (checkpoint_filename,)
            if checkpoint_filename is not None
            else ("model.pt", "iris.pt", "checkpoint.pt", "ckpt.pt")
        )
        checkpoint_path = resolve_pretrained_file(
            pretrained_model_name_or_path,
            checkpoint_candidates,
            repo_type=repo_type,
            revision=revision,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find an IRIS checkpoint for {pretrained_model_name_or_path!r}."
            )
        map_location = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=True
        )
        checkpoint_config = checkpoint.get("config")
        if config is None and isinstance(checkpoint_config, IRISConfig):
            args = checkpoint_config
        elif config is None and isinstance(checkpoint_config, dict):
            args = IRISConfig.from_dict(checkpoint_config)
        elif config is None:
            config_path = resolve_pretrained_file(
                pretrained_model_name_or_path,
                (config_filename, "iris_config.yaml", "config.yml"),
                repo_type=repo_type,
                revision=revision,
            )
            if config_path is None:
                raise FileNotFoundError(
                    "No config was provided and no config YAML was found beside "
                    f"{pretrained_model_name_or_path!r}."
                )
            args = IRISConfig.from_yaml(config_path)
        else:
            args = coerce_config(IRISConfig, config)
        args = apply_config_overrides(args, overrides)
        resolved_action_size = action_size or checkpoint.get("action_size")
        if resolved_action_size is None:
            raise ValueError(
                "action_size must be provided or present in the checkpoint."
            )
        agent = cls(args, action_size=int(resolved_action_size), device=map_location)
        agent.load(str(checkpoint_path))
        return agent

    def parameter_count(self, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only=trainable_only)

    def summary(self) -> dict[str, Any]:
        return module_summary(
            {
                "encoder": self.encoder,
                "decoder": self.decoder,
                "transformer": self.transformer,
                "cnn": self.cnn,
                "lstm": self.lstm,
                "actor_head": self.actor_head,
                "critic_head": self.critic_head,
            }
        )

    def _setup_optimizers(self) -> None:
        """Setup separate optimizers for each component."""
        # Autoencoder optimizer
        self.autoencoder_opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.config.model_learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )

        # Transformer optimizer
        self.transformer_opt = optim.Adam(
            self.transformer.parameters(),
            lr=self.config.model_learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )

        # Actor-Critic optimizer
        ac_params = (
            list(self.cnn.parameters())
            + list(self.lstm.parameters())
            + list(self.actor_head.parameters())
            + list(self.critic_head.parameters())
        )
        self.ac_opt = optim.Adam(
            ac_params,
            lr=self.config.actor_learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )

    @staticmethod
    def _losses_to_floats(losses: dict[str, torch.Tensor]) -> dict[str, float]:
        keys = list(losses.keys())
        values = torch.stack([losses[key].detach() for key in keys]).cpu().tolist()
        return dict(zip(keys, values))

    def forward_actor_critic(
        self,
        frames: torch.Tensor,  # (B, T, C, H, W)
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through actor-critic.

        Args:
            frames: Input frames (B, T, C, H, W)
            hidden: Optional LSTM hidden state

        Returns:
            action_logits: (B, T, action_size)
            values: (B, T)
            hidden_state: (h, c)
        """
        B, T, C, H, W = frames.shape

        # CNN features
        frames_flat = frames.reshape(B * T, C, H, W)
        features = self.cnn(frames_flat)
        features = features.reshape(B, T, -1)

        # LSTM
        if hidden is None:
            hidden = self._init_lstm_hidden(B)

        lstm_out, hidden = self.lstm(features, hidden)

        # Action and value
        action_logits = self.actor_head(lstm_out)
        values = self.critic_head(lstm_out).squeeze(-1)

        return action_logits, values, hidden

    def _init_lstm_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h = torch.zeros(
            self.config.actor_layers,
            batch_size,
            self.config.actor_hidden_size,
            device=self.device,
        )
        c = torch.zeros(
            self.config.actor_layers,
            batch_size,
            self.config.actor_hidden_size,
            device=self.device,
        )
        return (h, c)

    def act(
        self,
        frame: torch.Tensor,
        epsilon: float = 0.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample action from policy.

        Args:
            frame: Single frame (B, C, H, W)
            epsilon: Random action probability
            temperature: Action distribution temperature

        Returns:
            actions: Selected actions (B,)
        """
        self.eval()
        with torch.no_grad():
            B = frame.shape[0]
            frames = frame.unsqueeze(1)  # (B, 1, C, H, W)

            action_logits, _, _ = self.forward_actor_critic(frames)
            action_logits = action_logits.squeeze(1) / temperature

            # Epsilon-greedy
            if epsilon > 0:
                random_mask = torch.rand(B) < epsilon
                random_actions = torch.randint(
                    0, self.action_size, (B,), device=self.device
                )
                greedy_actions = action_logits.argmax(dim=-1)
                actions = torch.where(random_mask, random_actions, greedy_actions)
            else:
                probs = torch.softmax(action_logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(-1)

        return actions

    def imagine_rollout(
        self,
        initial_frame: torch.Tensor,
        horizon: int = 20,
    ) -> dict:
        """Generate imagined trajectories using world model.

        Args:
            initial_frame: Starting frame (B, C, H, W)
            horizon: Number of steps to imagine

        Returns:
            trajectory: Dictionary with imagined rollout data
        """
        self.eval()
        # Encode initial frame
        with torch.no_grad():
            _, initial_tokens, _ = self.encoder(initial_frame)

        # Lists to store trajectory. The policy consumes reconstructed frames, so
        # every stored frame is a decode of the current tokens (no raw/decoded
        # duplicate at t=0). We collect ``horizon`` (frame, action, reward)
        # triples plus one trailing frame -> ``horizon + 1`` frames total.
        frames_imagined = []
        actions_imagined = []
        rewards_imagined = []

        current_tokens = initial_tokens

        for step in range(horizon):
            # Decode current tokens to get the "observation" the policy sees.
            with torch.no_grad():
                reconstructed_frame = self.decoder(
                    self.encoder.quantizer.decode_indices(current_tokens)
                )
                frames_imagined.append(reconstructed_frame)

            # Get action from policy
            with torch.no_grad():
                action = self.act(reconstructed_frame, epsilon=0.0)

            # Predict next tokens
            with torch.no_grad():
                token_logits, action_hidden = self.transformer.predict_next_tokens(
                    current_tokens, action
                )
                next_tokens = torch.argmax(token_logits, dim=-1)

                # Reshape from (B, K) to (B, H, W) for decode_indices
                tokens_per_dim = int(self.config.tokens_per_frame**0.5)
                next_tokens = next_tokens.reshape(
                    next_tokens.shape[0],
                    tokens_per_dim,
                    tokens_per_dim,
                )

                # Get reward prediction
                reward_pred = self.transformer.reward_head(action_hidden).squeeze(-1)

            actions_imagined.append(action)
            rewards_imagined.append(reward_pred)
            current_tokens = next_tokens

        # Append the final imagined frame so ``frames`` has ``horizon + 1`` entries.
        with torch.no_grad():
            frames_imagined.append(
                self.decoder(self.encoder.quantizer.decode_indices(current_tokens))
            )

        return {
            "frames": torch.stack(frames_imagined, dim=1),  # (B, H+1, C, H, W)
            "actions": (
                torch.stack(actions_imagined, dim=1) if actions_imagined else None
            ),
            "rewards": (
                torch.stack(rewards_imagined, dim=1) if rewards_imagined else None
            ),
        }

    def update_autoencoder(self, frames: torch.Tensor) -> dict:
        """Update discrete autoencoder.

        Args:
            frames: Training frames (B, C, H, W)

        Returns:
            losses: Dictionary of loss values
        """
        self.encoder.train()
        self.decoder.train()

        with torch.amp.autocast(
            device_type=getattr(self.device, "type", str(self.device)),
            enabled=self.use_amp,
        ):
            # Encode
            z_q, indices, vq_loss = self.encoder(frames)

            # Decode
            reconstruction = self.decoder(z_q)

            # Compute losses
            recon_loss = F.l1_loss(reconstruction, frames)
            loss = recon_loss + vq_loss["vq_loss"]

        # Update
        self.autoencoder_opt.zero_grad(set_to_none=True)
        self.autoencoder_scaler.scale(loss).backward()
        self.autoencoder_scaler.unscale_(self.autoencoder_opt)
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            self.config.grad_clip_norm,
        )
        self.autoencoder_scaler.step(self.autoencoder_opt)
        self.autoencoder_scaler.update()

        losses = self._losses_to_floats(
            {
                "recon_loss": recon_loss,
                "vq_loss": vq_loss["vq_loss"],
                "perplexity": vq_loss["perplexity"],
                "total_loss": loss,
            }
        )
        self.logger.debug(f"Autoencoder update: {losses}")
        return losses

    def update_transformer(
        self,
        frames: torch.Tensor,  # (B, T+1, C, H, W)
        actions: torch.Tensor,  # (B, T)
        rewards: torch.Tensor,  # (B, T)
        terminals: torch.Tensor,  # (B, T)
    ) -> dict:
        """Update transformer world model.

        Args:
            frames: Frame sequence
            actions: Actions taken
            rewards: Rewards received
            terminals: Terminal flags

        Returns:
            losses: Dictionary of loss values
        """
        self.transformer.train()

        B, T_plus_1, C, H, W = frames.shape

        # Encode all frames to tokens
        tokens_list = []
        for t in range(T_plus_1):
            _, indices_t, _ = self.encoder(frames[:, t])
            indices_t = indices_t.reshape(B, -1)  # (B, K) flatten spatial dimensions
            tokens_list.append(indices_t)

        tokens = torch.stack(tokens_list, dim=1)  # (B, T+1, K)

        # Convert actions from (B, T, action_size) to (B, T) scalar indices
        if actions.dim() == 3:
            actions = actions.argmax(dim=-1)  # (B, T)

        with torch.amp.autocast(
            device_type=getattr(self.device, "type", str(self.device)),
            enabled=self.use_amp,
        ):
            # Get predictions. The transformer consumes the full T+1 frame
            # sequence (teacher forcing) and predicts frames 1..T.
            token_logits, rewards_pred, terms_pred = self.transformer(
                tokens,  # (B, T+1, K)
                actions,  # (B, T)
            )

            # Token prediction loss
            next_tokens = tokens[:, 1:]  # (B, T, K)
            token_loss = F.cross_entropy(
                token_logits.reshape(-1, self.config.vocab_size),
                next_tokens.reshape(-1),
            )

            # Reward loss (MSE)
            reward_loss = F.mse_loss(rewards_pred, rewards)

            # Termination loss (cross-entropy)
            term_loss = F.cross_entropy(
                terms_pred.reshape(-1, 2),
                terminals.reshape(-1),
            )

            # Total loss
            loss = token_loss + 0.1 * reward_loss + 0.1 * term_loss

        # Update
        self.transformer_opt.zero_grad(set_to_none=True)
        self.transformer_scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        self.transformer_scaler.unscale_(self.transformer_opt)
        nn.utils.clip_grad_norm_(
            self.transformer.parameters(), self.config.grad_clip_norm
        )
        self.transformer_scaler.step(self.transformer_opt)
        self.transformer_scaler.update()

        losses = self._losses_to_floats(
            {
                "token_loss": token_loss,
                "reward_loss": reward_loss,
                "term_loss": term_loss,
                "total_loss": loss,
            }
        )
        self.logger.debug(f"Transformer update: {losses}")
        return losses

    def update_actor_critic(
        self,
        imagined_trajectory: dict,
    ) -> dict:
        """Update actor-critic in imagination.

        Args:
            imagined_trajectory: Dictionary from imagine_rollout

        Returns:
            losses: Dictionary of loss values
        """
        self.train()

        frames = imagined_trajectory["frames"]  # (B, T+1, C, H, W)
        actions = imagined_trajectory["actions"]  # (B, T)
        rewards = imagined_trajectory["rewards"]  # (B, T)

        B, T_plus_1, C, H, W = frames.shape

        with torch.amp.autocast(
            device_type=getattr(self.device, "type", str(self.device)),
            enabled=self.use_amp,
        ):
            # Forward pass
            action_logits, values, _ = self.forward_actor_critic(
                frames[:, :-1]
            )  # (B, T, A), (B, T)

            # Compute log probabilities
            action_dist = torch.softmax(action_logits, dim=-1)
            action_log_probs = torch.log(action_dist + 1e-8)

            # Gather log probs for taken actions
            actions_one_hot = F.one_hot(actions, self.action_size).float()
            taken_log_probs = (action_log_probs * actions_one_hot).sum(dim=-1)  # (B, T)

            # Compute λ-returns
            discounts = torch.full_like(rewards, self.config.discount)
            lambda_returns = compute_lambda_return(
                rewards,
                torch.cat([values, torch.zeros(B, 1, device=self.device)], dim=1),
                discounts,
                self.config.td_lambda,
            )

            # Advantage
            advantages = lambda_returns - values  # (B, T)

            # Actor loss (REINFORCE with baseline)
            actor_loss = -(taken_log_probs * advantages.detach()).mean()

            # Entropy bonus
            entropy = -(action_dist * action_log_probs).sum(dim=-1).mean()
            actor_loss -= self.config.entropy_coef * entropy

            # Critic loss
            value_loss = F.mse_loss(values, lambda_returns.detach())

            # Total loss
            loss = actor_loss + 0.5 * value_loss

        # Update
        self.ac_opt.zero_grad(set_to_none=True)
        self.ac_scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        self.ac_scaler.unscale_(self.ac_opt)
        nn.utils.clip_grad_norm_(
            list(self.cnn.parameters())
            + list(self.lstm.parameters())
            + list(self.actor_head.parameters())
            + list(self.critic_head.parameters()),
            self.config.grad_clip_norm,
        )
        self.ac_scaler.step(self.ac_opt)
        self.ac_scaler.update()

        losses = self._losses_to_floats(
            {
                "actor_loss": actor_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "total_loss": loss,
            }
        )
        self.logger.debug(f"Actor-critic update: {losses}")
        return losses

    def save(self, path: str) -> None:
        """Save agent state."""
        save_config_next_to_checkpoint(self.config, path)
        torch.save(
            {
                "config": self.config.to_dict(),
                "action_size": int(self.action_size),
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "transformer": self.transformer.state_dict(),
                "cnn": self.cnn.state_dict(),
                "lstm": self.lstm.state_dict(),
                "actor_head": self.actor_head.state_dict(),
                "critic_head": self.critic_head.state_dict(),
                "autoencoder_opt": self.autoencoder_opt.state_dict(),
                "transformer_opt": self.transformer_opt.state_dict(),
                "ac_opt": self.ac_opt.state_dict(),
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent state."""
        with torch.serialization.safe_globals([IRISConfig]):
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=True,
            )

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.transformer.load_state_dict(checkpoint["transformer"])
        self.cnn.load_state_dict(checkpoint["cnn"])
        self.lstm.load_state_dict(checkpoint["lstm"])
        self.actor_head.load_state_dict(checkpoint["actor_head"])
        self.critic_head.load_state_dict(checkpoint["critic_head"])

        self.autoencoder_opt.load_state_dict(checkpoint["autoencoder_opt"])
        self.transformer_opt.load_state_dict(checkpoint["transformer_opt"])
        self.ac_opt.load_state_dict(checkpoint["ac_opt"])

        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)
