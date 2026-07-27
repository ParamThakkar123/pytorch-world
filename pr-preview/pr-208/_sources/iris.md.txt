# IRIS: Transformers for Sample-Efficient World Models

IRIS (Imagination with auto-Regression over an Inner Speech) is an implementation of the paper
"Transformers are Sample-Efficient World Models" (Micheli et al., 2023).

```{contents} Contents
:depth: 3
```

## Overview

IRIS achieves **human-level performance on Atari with only ~2 hours of gameplay** (100k environment steps)
by learning entirely in the imagination of a world model:

1. **Train world model** from real interactions
2. **Generate imagined trajectories** in the latent space
3. **Train policy** purely on imagined data

## Architecture

### High-level diagram

<div class="architecture-diagram" aria-label="IRIS architecture diagram">
  <section class="diagram-section">
    <h3>Discrete Autoencoder</h3>
    <div class="diagram-row">
      <span class="diagram-node info">Encoder CNN 64x64</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">VQ-VAE 512 vocab 16 tokens</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Decoder transposed CNN</span>
    </div>
  </section>
  <section class="diagram-section">
    <h3>Autoregressive Transformer</h3>
    <div class="diagram-row">
      <span class="diagram-node">Latent tokens</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Action token</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Next latent tokens</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Reward and termination heads</span>
    </div>
  </section>
  <section class="diagram-section">
    <h3>Actor-Critic in Imagination</h3>
    <div class="diagram-row">
      <span class="diagram-node success">Actor CNN and LSTM</span>
      <span class="diagram-node success">Critic CNN and LSTM</span>
    </div>
  </section>
</div>

### VQ-VAE: Discrete Autoencoder

Both IRIS and Genie use Vector Quantized Variational Autoencoders (VQ-VAE) to convert
continuous visual observations into discrete token sequences.

```{mermaid}
graph LR
    A["Image x"] --> B["CNN Encoder"]
    B --> C["Continuous z_e(x)"]
    C --> D["Vector Quantization"]
    E["Codebook {e_k}"] --> D
    D --> F["Discrete indices + z_q(x)"]
    F --> G["CNN Decoder"]
    G --> H["Reconstructed x̂"]
```

**Quantization:**

The encoder output `z_e(x)` is mapped to the nearest codebook vector:

```{math}
z_q(x) = e_k, \quad \text{where } k = \arg\min_j \|z_e(x) - e_j\|_2
```

**VQ-VAE Loss:**

```{math}
\mathcal{L}_{\text{VQ}} =
\underbrace{\|\hat{x} - x\|^2}_{\text{reconstruction}}
+ \underbrace{\|\text{sg}[z_e(x)] - e_k\|^2}_{\text{codebook loss}}
+ \beta \cdot \underbrace{\|z_e(x) - \text{sg}[e_k]\|^2}_{\text{commitment loss}}
```

IRIS uses EMA (Exponential Moving Average) for codebook updates instead
of the codebook loss, producing more stable training.

### Discrete Autoencoder Architecture

The encoder maps a 64×64 RGB frame to **16 tokens** from a **512-entry** codebook:

```
Input:  (3, 64, 64)
  └─ Conv2D(3, 64, 3, stride 2, pad 1)   → (64, 32, 32)
  └─ Conv2D(64, 128, 3, stride 2, pad 1) → (128, 16, 16)  + self-attention
  └─ Conv2D(128, 256, 3, stride 2, pad 1) → (256, 8, 8)   + self-attention
  └─ Conv2D(256, 512, 3, stride 2, pad 1) → (512, 4, 4)
  └─ ResBlocks + 1x1 projection to embedding dim
  └─ VQ layer over the 4×4 grid → 16 discrete indices
Output: 16 token indices (4 × 4, each ∈ {0, ..., 511})
```

### Transformer World Model

The transformer is a GPT-style autoregressive model:

```
Params:
  - vocab_size: 512 visual tokens (separate embedding table for the actions)
  - embed_dim: 256
  - num_layers: 10
  - num_heads: 4
  - seq_length: 20 timesteps × (16 tokens + 1 action) = 340 positions

Architecture:
  Token/Action Embedding → Positional Embedding → Causal Transformer Blocks
    → token head (next tokens) + reward head + termination head
```

Reward and termination are predicted by dedicated linear heads (read from the
action position), **not** encoded as extra tokens in the sequence.

**Input sequence format** — frame tokens and actions are interleaved with a
causal mask, and the next frame's tokens are generated autoregressively from the
action position onward:

```
[zₜ_0, zₜ_1, ..., zₜ_15, aₜ] → predict zₜ₊₁_0, then zₜ₊₁_1, ..., zₜ₊₁_15
                              (each conditioned on previously generated tokens)
plus, from the aₜ position:   predict reward rₜ and termination dₜ
```

### Actor-Critic

| Component | Purpose |
|---|---|
| **CNN + LSTM** | Processes reconstructed frames |
| **λ-returns** | Balances bias and variance in value estimation |
| **REINFORCE** | Policy gradient with baseline |
| **Entropy bonus** | Maintains exploration |

### Imagination Rollout

```python
# Imagine H steps: sample tokens autoregressively, decode to frames, feed to actor-critic
for h in range(imagination_horizon):
    tokens = transformer.generate(prev_tokens, action)
    frame = autoencoder.decode(tokens)          # decode to pixels
    action = actor(frame, hidden_state)          # policy
    reward = transformer.reward_head(tokens)     # predicted reward
    hidden_state = lstm(hidden_state, action, tokens)
```

## Training

### Staged training schedule

| Component | Start Epoch | Description |
|-----------|-------------|-------------|
| Autoencoder | 5 | Learn frame compression first |
| Transformer | 25 | Learn dynamics once tokens are good |
| Actor-Critic | 50 | Learn policy in imagination |

### Key Hyperparameters

| Parameter | Value |
|---|---|
| **Frame size** | 64×64 |
| **Tokens per frame** | 16 (from 512 vocabulary) |
| **Transformer sequence length** | 20 timesteps |
| **Imagination horizon** | 20 steps |
| **Discount (γ)** | 0.995 |
| **λ for λ-return** | 0.95 |

## Usage in TorchWM

### Quick start

```python
import torch
import torchwm

agent = torchwm.create_model(
    "iris",
    action_size=4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```

### Using config directly

```python
from torchwm import IRISConfig

config = IRISConfig()

# Autoencoder
config.vocab_size = 512
config.tokens_per_frame = 16

# Transformer
config.transformer_layers = 10
config.transformer_embed_dim = 256

# Training
config.total_epochs = 600
config.env_steps_per_epoch = 200
config.env = "ALE/Pong-v5"
```

### CLI

```bash
torchwm train iris --env ALE/Pong-v5 --device cuda
```

For custom research code:

```bash
python -m world_models.training.train_iris --game "ALE/Pong-v5"
```

See {doc}`configs_reference` for the full IRISConfig field reference with defaults.

## Benchmark Results

| Metric | IRIS (ours) | SPR | DrQ | CURL | SimPLe |
|--------|-------------|-----|-----|------|--------|
| Mean HNS | **1.046** | 0.616 | 0.465 | 0.261 | 0.332 |
| Superhuman games | **10/26** | 6/26 | 3/26 | 2/26 | 1/26 |

## Common Pitfalls

### Codebook collapse

Most codebook entries go unused.

**Fixes:**
- Use EMA codebook updates (default in IRIS)
- Lower commitment loss weight `β`
- Add codebook reset: re-initialize unused codes

### Transformer memory

Sequence length: 16 × 20 = 320 tokens.

**Fixes:**
- Use gradient checkpointing
- Reduce context length

### Slow autoregressive generation

AR token generation is O(tokens) sequential.

**Fixes:**
- Use KV caching for transformer inference
- Reduce the number of imagination steps

## See Also

- {doc}`genie` — extends IRIS with latent actions and video-only training
- {doc}`dreamer` — continuous world model alternative to IRIS

## References

- Micheli, V., Alonso, E., & Fleuret, F. (2023). Transformers are Sample-Efficient World Models. *ICLR 2023.*
- Van Den Oord, A., & Vinyals, O. (2017). Neural Discrete Representation Learning. *NeurIPS 2017.*
