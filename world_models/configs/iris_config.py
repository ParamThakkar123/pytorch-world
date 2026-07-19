from dataclasses import dataclass

from world_models.configs.serialization import SerializableConfigMixin


@dataclass
class IRISConfig(SerializableConfigMixin):
    """Configuration for IRIS (Imagination with auto-Regression over an Inner Speech)

    Based on paper: "Transformers are Sample-Efficient World Models"
    Implements discrete autoencoder + autoregressive Transformer for sample-efficient RL.
    """

    # === Discrete Autoencoder (VQVAE) ===
    frame_height: int = 64
    frame_width: int = 64
    frame_channels: int = 3

    vocab_size: int = 512
    tokens_per_frame: int = 16
    token_embedding_dim: int = 512

    encoder_channels: int = 64
    encoder_layers: int = 4
    encoder_residual_blocks: int = 2

    decoder_depth: int = 32

    reconstruction_weight: float = 1.0
    commitment_weight: float = 0.25
    perceptual_weight: float = 1.0

    # === Transformer (World Model) ===
    transformer_timesteps: int = 20
    transformer_embed_dim: int = 256
    transformer_layers: int = 10
    transformer_heads: int = 4
    transformer_dropout: float = 0.1

    # === Actor-Critic === (paper Appendix B, Table 6)
    imagination_horizon: int = 20
    discount: float = 0.995
    td_lambda: float = 0.95
    entropy_coef: float = 0.001

    actor_hidden_size: int = 512
    actor_layers: int = 1

    value_hidden_size: int = 512
    value_layers: int = 3

    # === Training ===
    total_epochs: int = 600
    collection_epochs: int = 500
    env_steps_per_epoch: int = 200
    training_steps_per_epoch: int = 200

    model_learning_rate: float = 1e-4
    actor_learning_rate: float = 1e-4
    value_learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    grad_clip_norm: float = 10.0
    use_amp: bool = True
    gradient_checkpointing: bool = True

    collect_epsilon: float = 0.01
    eval_temperature: float = 0.5

    start_autoencoder_after: int = 5
    start_transformer_after: int = 25
    start_actor_critic_after: int = 50

    autoencoder_batch_size: int = 256
    transformer_batch_size: int = 64
    actor_critic_batch_size: int = 64

    # === Atari 100k Benchmark ===
    atari_100k: bool = True
    max_env_steps: int = 100000

    # === Environment ===
    env_backend: str = "gym"
    env: str = "ALE/Pong-v5"
    action_repeat: int = 4

    # === Logging ===
    log_interval: int = 1000
    eval_episodes: int = 100
    checkpoint_interval: int = 50

    def get_frame_shape(self) -> tuple:
        return (self.frame_channels, self.frame_height, self.frame_width)

    def get_autoencoder_config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "tokens_per_frame": self.tokens_per_frame,
            "embedding_dim": self.token_embedding_dim,
            "encoder_channels": self.encoder_channels,
            "encoder_layers": self.encoder_layers,
            "encoder_residual_blocks": self.encoder_residual_blocks,
            "decoder_depth": self.decoder_depth,
            "frame_shape": self.get_frame_shape(),
            "reconstruction_weight": self.reconstruction_weight,
            "commitment_weight": self.commitment_weight,
            "perceptual_weight": self.perceptual_weight,
        }

    def get_transformer_config(self) -> dict:
        return {
            "timesteps": self.transformer_timesteps,
            "embed_dim": self.transformer_embed_dim,
            "layers": self.transformer_layers,
            "heads": self.transformer_heads,
            "dropout": self.transformer_dropout,
            "vocab_size": self.vocab_size,
            "tokens_per_frame": self.tokens_per_frame,
            "action_size": None,
        }

    def get_rl_config(self) -> dict:
        return {
            "imagination_horizon": self.imagination_horizon,
            "discount": self.discount,
            "td_lambda": self.td_lambda,
            "entropy_coef": self.entropy_coef,
            "actor_hidden_size": self.actor_hidden_size,
            "actor_layers": self.actor_layers,
            "value_hidden_size": self.value_hidden_size,
            "value_layers": self.value_layers,
            "frame_shape": self.get_frame_shape(),
        }
