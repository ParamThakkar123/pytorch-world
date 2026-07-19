import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, cast


class IRISTransformer(nn.Module):
    """GPT-like autoregressive Transformer for world modeling.

    Models the dynamics of the environment by predicting, autoregressively over
    an interleaved sequence of frame tokens and actions:

    - Next frame tokens (transition model), one token at a time
    - Rewards
    - Episode termination

    The sequence layout for ``S`` frames and ``S - 1`` actions is::

        z_0^1 ... z_0^K, a_0, z_1^1 ... z_1^K, a_1, ..., z_{S-2}^1 ... z_{S-2}^K,
        a_{S-2}, z_{S-1}^1 ... z_{S-1}^K

    A causal (lower-triangular) attention mask is always applied, so every
    position only attends to itself and preceding positions. The tokens of frame
    ``t + 1`` are predicted starting from the *action* position ``a_t`` (which
    sees the whole of frame ``t`` and the action), then autoregressively from
    each previously predicted token of frame ``t + 1``. This matches the paper's

        z_{t+1}^k ~ p(. | z_{<=t}, a_{<=t}, z_{t+1}^{<k})
    """

    def __init__(
        self,
        vocab_size: int = 512,
        tokens_per_frame: int = 16,
        action_size: int = 18,  # Number of Atari actions
        embed_dim: int = 256,
        num_layers: int = 10,
        num_heads: int = 4,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.tokens_per_frame = tokens_per_frame
        self.action_size = action_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.action_embedding = nn.Embedding(action_size, embed_dim)

        # Positional embeddings
        # Max sequence length: (tokens_per_frame + 1) * timesteps
        # 16 tokens + 1 action per timestep = 17 tokens/timestep
        max_tokens = tokens_per_frame + 1  # tokens + action
        max_seq_len = max_tokens * 50  # Support up to 50 timesteps
        self.max_seq_len = max_seq_len
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)

        # Transformer encoder. GPT-2-style pre-norm blocks (norm_first=True) as
        # described in the paper (layer normalization of the block input).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Output heads
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Token prediction head (for next frame tokens)
        self.token_head = nn.Linear(embed_dim, vocab_size)

        # Reward prediction head
        self.reward_head = nn.Linear(embed_dim, 1)

        # Termination prediction head
        self.termination_head = nn.Linear(
            embed_dim, 2
        )  # Binary: 0=continue, 1=terminal

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with proper scaling."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.action_embedding.weight, std=0.02)

        # Apply special initialization to output heads
        nn.init.zeros_(self.token_head.bias)
        nn.init.zeros_(self.reward_head.bias)
        nn.init.zeros_(self.termination_head.bias)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """Additive lower-triangular causal mask of shape (length, length)."""
        return torch.triu(
            torch.full((length, length), float("-inf"), device=device),
            diagonal=1,
        )

    def _run_transformer(
        self, sequence: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Run transformer layers, checkpointing them during training when enabled."""
        if not (self.gradient_checkpointing and self.training):
            return self.transformer(sequence, mask=mask)

        hidden: torch.Tensor = sequence
        for layer in self.transformer.layers:
            hidden = cast(
                torch.Tensor,
                checkpoint(
                    lambda src, src_mask: layer(src, src_mask=src_mask),
                    hidden,
                    mask,
                    use_reentrant=False,
                ),
            )
        if self.transformer.norm is not None:
            hidden = self.transformer.norm(hidden)
        return hidden

    def _embed_interleaved(
        self,
        frame_tokens: torch.Tensor,  # (B, Tc, K)
        actions: torch.Tensor,  # (B, Tc)
        extra_tokens: Optional[torch.Tensor] = None,  # (B, m)
    ) -> torch.Tensor:
        """Build a positionally-embedded interleaved token/action sequence.

        Produces embeddings for ``[z_0, a_0, ..., z_{Tc-1}, a_{Tc-1}]`` and,
        optionally, ``m`` trailing (partial next-frame) tokens with no following
        action. Used both for the training forward pass and for autoregressive
        generation.
        """
        B, Tc, K = frame_tokens.shape
        E = self.embed_dim

        tok_emb = self.token_embedding(frame_tokens.reshape(B, Tc * K)).reshape(
            B, Tc, K, E
        )
        act_emb = self.action_embedding(actions)  # (B, Tc, E)

        # Interleave each frame block with its action: (B, Tc, K + 1, E)
        blocks = torch.cat([tok_emb, act_emb.unsqueeze(2)], dim=2)
        seq = blocks.reshape(B, Tc * (K + 1), E)

        if extra_tokens is not None and extra_tokens.shape[1] > 0:
            extra_emb = self.token_embedding(extra_tokens)  # (B, m, E)
            seq = torch.cat([seq, extra_emb], dim=1)

        L = seq.shape[1]
        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds positional-embedding capacity "
                f"{self.max_seq_len}."
            )
        return seq + self.pos_embedding[:, :L, :]

    def forward(
        self,
        tokens: torch.Tensor,  # (B, S, K) - S frame token grids
        actions: torch.Tensor,  # (B, S-1) - actions between frames
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced forward pass through the Transformer world model.

        Args:
            tokens: Frame tokens (B, S, K) for S consecutive frames.
            actions: Actions (B, S-1); ``actions[:, t]`` is taken after frame t.

        Returns:
            token_logits: Predictions of frames 1..S-1 (B, S-1, K, vocab_size).
            rewards: Predicted rewards r_0..r_{S-2} (B, S-1).
            terminations: Predicted terminations d_0..d_{S-2} (B, S-1, 2).
        """
        B, S, K = tokens.shape
        if actions.shape[1] != S - 1:
            raise ValueError(
                f"Expected actions of length S-1={S - 1}, got {actions.shape[1]}."
            )

        # Interleave frames 0..S-1 with actions 0..S-2. The sequence ends on the
        # last frame block (z_{S-1}) with no trailing action.
        tok_emb = self.token_embedding(tokens.reshape(B, S * K)).reshape(B, S, K, self.embed_dim)
        act_emb = self.action_embedding(actions)  # (B, S-1, E)

        head_blocks = torch.cat(
            [tok_emb[:, : S - 1], act_emb.unsqueeze(2)], dim=2
        )  # (B, S-1, K+1, E)
        head_blocks = head_blocks.reshape(B, (S - 1) * (K + 1), self.embed_dim)
        last_frame = tok_emb[:, S - 1]  # (B, K, E)
        sequence = torch.cat([head_blocks, last_frame], dim=1)  # (B, L, E)

        L = sequence.shape[1]
        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds positional-embedding capacity "
                f"{self.max_seq_len}."
            )
        sequence = sequence + self.pos_embedding[:, :L, :]

        causal = self._causal_mask(L, tokens.device)
        hidden = self._run_transformer(sequence, causal)
        hidden = self.layer_norm(hidden)

        # For target frame t+1 (t in 0..S-2), the K hidden states that predict its
        # tokens are the contiguous slice starting at action position a_t:
        #   [a_t, z_{t+1}^1, ..., z_{t+1}^{K-1}]  (stride K+1 between t's).
        starts = torch.arange(S - 1, device=tokens.device) * (K + 1) + K  # (S-1,)
        offsets = torch.arange(K, device=tokens.device)  # (K,)
        idx = (starts[:, None] + offsets[None, :]).reshape(-1)  # ((S-1)*K,)

        sel = hidden[:, idx, :].reshape(B, S - 1, K, self.embed_dim)  # (B, S-1, K, E)
        token_logits = self.token_head(sel)  # (B, S-1, K, vocab)

        # Reward / termination are read from the action position a_t, which is the
        # first element of each per-frame slice.
        action_hidden = sel[:, :, 0, :]  # (B, S-1, E)
        rewards = self.reward_head(action_hidden).squeeze(-1)  # (B, S-1)
        terminations = self.termination_head(action_hidden)  # (B, S-1, 2)

        return token_logits, rewards, terminations

    def _normalize_context(
        self, tokens: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Coerce a single-step (frame, action) context to (B, 1, K) / (B, 1)."""
        if tokens.dim() == 3 and tokens.shape[1] != self.tokens_per_frame:
            # (B, H, W) grid of tokens -> (B, K)
            B_grid, H, W = tokens.shape
            tokens = tokens.reshape(B_grid, H * W)
        if tokens.dim() == 2:  # (B, K) -> (B, 1, K)
            tokens = tokens.unsqueeze(1)
        if actions.dim() == 1:  # (B,) -> (B, 1)
            actions = actions.unsqueeze(1)
        return tokens, actions

    @torch.no_grad()
    def _generate_frame(
        self,
        context_tokens: torch.Tensor,  # (B, Tc, K)
        context_actions: torch.Tensor,  # (B, Tc)
        sample: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autoregressively generate the next frame's K tokens.

        Returns:
            step_logits: Per-token logits used at generation (B, K, vocab).
            generated: Sampled/greedy token indices (B, K).
            action_hidden: Hidden state at the final action position (B, E), used
                for reward / termination prediction.
        """
        _, Tc, K = context_tokens.shape
        base_len = Tc * (K + 1)  # sequence ends on the last action a_{Tc-1}

        generated_list: list[torch.Tensor] = []
        logits_list: list[torch.Tensor] = []
        action_hidden: Optional[torch.Tensor] = None

        for k in range(K):
            extra = (
                torch.stack(generated_list, dim=1) if generated_list else None
            )  # (B, k)
            seq = self._embed_interleaved(context_tokens, context_actions, extra)
            L = seq.shape[1]
            causal = self._causal_mask(L, context_tokens.device)
            hidden = self.layer_norm(self._run_transformer(seq, causal))

            if k == 0:
                # Last position of the base sequence is the action a_{Tc-1}.
                action_hidden = hidden[:, base_len - 1, :]

            step_hidden = hidden[:, -1, :]  # predicts the next token
            logits = self.token_head(step_hidden)  # (B, vocab)
            logits_list.append(logits)

            if sample:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
            else:
                next_token = logits.argmax(dim=-1)  # (B,)
            generated_list.append(next_token)

        step_logits = torch.stack(logits_list, dim=1)  # (B, K, vocab)
        generated = torch.stack(generated_list, dim=1)  # (B, K)
        assert action_hidden is not None
        return step_logits, generated, action_hidden

    def predict_next_tokens(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedily predict the next frame tokens autoregressively.

        Args:
            tokens: Current frame tokens (B, K) or (B, H, W).
            actions: Actions taken (B,).

        Returns:
            token_logits: Next frame token logits (B, K, vocab_size). Their argmax
                equals the greedily generated tokens.
            action_hidden: Hidden states for reward prediction (B, embed_dim).
        """
        tokens, actions = self._normalize_context(tokens, actions)
        step_logits, _, action_hidden = self._generate_frame(
            tokens, actions, sample=False
        )
        return step_logits, action_hidden

    def sample_next_tokens(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample next tokens autoregressively from the distribution.

        Args:
            tokens: Current frame tokens (B, K) or (B, H, W).
            actions: Actions taken (B,).
            temperature: Sampling temperature (higher = more random).

        Returns:
            sampled_tokens: Sampled token indices (B, K).
            log_probs: Log probabilities of sampled tokens (B, K).
        """
        tokens, actions = self._normalize_context(tokens, actions)
        step_logits, sampled_tokens, _ = self._generate_frame(
            tokens, actions, sample=True, temperature=temperature
        )
        log_probs = F.log_softmax(step_logits / temperature, dim=-1)
        log_probs = torch.gather(
            log_probs, -1, sampled_tokens.unsqueeze(-1)
        ).squeeze(-1)
        return sampled_tokens, log_probs


class IRISWorldModel(nn.Module):
    """Complete IRIS World Model combining autoencoder and transformer.

    This is the core component that learns environment dynamics entirely
    in the "imaginary" latent space.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        transformer: IRISTransformer,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.transformer = transformer

    def forward(
        self,
        observations: torch.Tensor,  # (B, T+1, C, H, W)
        actions: torch.Tensor,  # (B, T)
    ) -> Tuple[dict, dict]:
        """Full world model forward pass.

        Args:
            observations: Image sequence (B, T+1, C, H, W)
            actions: Actions (B, T)

        Returns:
            predictions: Dictionary with predicted tokens, rewards, terminations
            losses: Dictionary with loss components
        """
        B, T_plus_1, C, H, W = observations.shape
        T = T_plus_1 - 1

        # Encode each frame to tokens
        tokens_list = []
        for t in range(T_plus_1):
            obs_t = observations[:, t]  # (B, C, H, W)
            _, indices_t, _ = self.encoder(obs_t)
            tokens_list.append(indices_t.reshape(B, -1))

        # Stack tokens: (B, T+1, K)
        tokens = torch.stack(tokens_list, dim=1)

        # Get transformer predictions over the full frame sequence. Predictions
        # cover frames 1..T (B, T, K, vocab).
        token_logits, rewards_pred, terminations_pred = self.transformer(
            tokens,  # (B, T+1, K)
            actions,  # (B, T)
        )

        # Decode predictions to images (for visualization)
        decoded_frames_list: list[torch.Tensor] = []
        for t in range(T):
            next_tokens_pred = token_logits[:, t, :, :].argmax(dim=-1)  # Greedy
            decoded_frames_list.append(
                getattr(self.decoder, "decode_from_embeddings")(next_tokens_pred)
            )

        decoded_frames: Optional[torch.Tensor]
        decoded_frames = (
            torch.stack(decoded_frames_list, dim=1) if decoded_frames_list else None
        )

        # Get actual next tokens for loss computation
        next_tokens = tokens[:, 1:]  # (B, T, K)

        # Compute losses
        token_loss = F.cross_entropy(
            token_logits.reshape(-1, self.transformer.vocab_size),
            next_tokens.reshape(-1),
            reduction="mean",
        )

        # Reward and termination losses would be computed with actual labels
        # (These are computed in the training loop)

        predictions = {
            "token_logits": token_logits,
            "rewards": rewards_pred,
            "terminations": terminations_pred,
            "decoded_frames": decoded_frames,
        }

        losses = {
            "token_loss": token_loss,
        }

        return predictions, losses

    def imagine(
        self,
        initial_tokens: torch.Tensor,  # (B, K)
        policy: nn.Module,
        horizon: int = 20,
        temperature: float = 1.0,
    ) -> dict:
        """Generate imagined trajectories.

        Args:
            initial_tokens: Initial frame tokens (B, K)
            policy: Policy network to sample actions
            horizon: Number of steps to imagine
            temperature: Sampling temperature for token prediction

        Returns:
            imagined: Dictionary with imagined trajectories
        """

        # Lists to store trajectory
        tokens_history = [initial_tokens]
        actions_history = []
        rewards_history = []
        terminations_history = []

        # Get initial reconstruction for policy input
        current_tokens = initial_tokens

        for step in range(horizon):
            # Get action from policy (using decoded frame)
            with torch.no_grad():
                decoded_frame = getattr(self.decoder, "decode_from_embeddings")(
                    current_tokens
                )
                action = getattr(policy, "forward")(decoded_frame)
                actions_history.append(action)

            # Predict next tokens
            sampled_tokens, log_probs = self.transformer.sample_next_tokens(
                current_tokens,
                action.squeeze(-1) if action.dim() > 1 else action,
                temperature,
            )

            # Get reward and termination predictions
            with torch.no_grad():
                _, action_hidden = self.transformer.predict_next_tokens(
                    current_tokens, action
                )
                reward = self.transformer.reward_head(action_hidden).mean()
                termination_logits = self.transformer.termination_head(action_hidden)
                termination = torch.softmax(termination_logits, dim=-1)[:, 1]

            tokens_history.append(sampled_tokens)
            rewards_history.append(reward)
            terminations_history.append(termination)

            # Update current tokens
            current_tokens = sampled_tokens

            # Early stopping if terminal
            if termination.mean() > 0.5:
                break

        return {
            "tokens": torch.stack(tokens_history, dim=1),  # (B, H+1, K)
            "actions": torch.stack(actions_history, dim=1) if actions_history else None,
            "rewards": torch.stack(rewards_history, dim=1) if rewards_history else None,
            "terminations": (
                torch.stack(terminations_history, dim=1)
                if terminations_history
                else None
            ),
        }
