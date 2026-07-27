# JEPA: Joint Embedding Predictive Architecture

JEPA is a self-supervised learning method that learns visual representations by predicting
representations in abstract latent space, without relying on generative modeling or
hand-crafted data augmentations.

Based on paper: [I-JEPA: Image-based Joint Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) (Bardes et al., 2023)

```{contents} Contents
:depth: 3
```

## Overview

I-JEPA learns visual representations **without**:
- Hand-crafted data augmentations (color jitter, grayscale, etc.)
- Negative examples (contrastive learning)
- Pixel-level reconstruction (autoencoders, MAE)

Instead, it predicts the **latent representation** of one image region from
another region using a Vision Transformer (ViT) backbone. The predictor operates
in embedding space, not pixel space, which forces the model to learn
semantically meaningful features.

```{mermaid}
graph TD
    A["Input image x"] --> B["Context encoder f_θ"]
    A --> C["Target encoder f_θ̄ (EMA)"]
    B --> D["Context patches (masked)"]
    C --> E["Target patches"]
    D --> F["Predictor g_φ"]
    E --> G["Target representation sg(y_target)"]
    F --> H["Predicted representation ŷ"]
    H --> I["L2 loss"]
    G --> I
    I --> J["sg: stop-gradient through target encoder"]
```

## Architecture

### High-level diagram

<div class="architecture-diagram" aria-label="JEPA architecture diagram">
  <section class="diagram-section">
    <h3>JEPA Architecture</h3>
    <div class="diagram-row">
      <span class="diagram-node warning">Current frame encoder</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Predictor token</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Predicted representation</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node danger">MSE loss</span>
    </div>
    <div class="diagram-row">
      <span class="diagram-node warning">Future frame frozen encoder</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node">Target representation</span>
      <span class="diagram-arrow">→</span>
      <span class="diagram-node danger">MSE loss</span>
    </div>
  </section>
</div>

### Vision Transformer (ViT)

The backbone encoder in `world_models.models.vit` is a Vision Transformer
following the standard ViT architecture with JEPA-specific modifications.

**Patch embedding:**

The input image `x ∈ ℝ^{3×H×W}` is split into patches of size `P × P`,
producing `N = (H/P) × (W/P)` patches. Each patch is linearly projected to
`embed_dim`:

```{math}
\text{patches} \in \mathbb{R}^{N \times (3 \cdot P^2)} \to
\text{tokens} \in \mathbb{R}^{N \times D}
```

**Transformer blocks:**

Each block consists of:
1. **LayerNorm** → Multi-Head Self-Attention → residual
2. **LayerNorm** → MLP (GELU, 4× hidden) → residual
3. **DropPath** (stochastic depth) regularization during training

**Key architectural details:**
- No class token — all patch tokens are used
- Pre-normalization (LayerNorm before attention and MLP)
- Fixed sin-cos positional embeddings (not learned)

### Target Encoder (EMA)

The target encoder `f_{\bar{θ}}` has the same architecture as the context
encoder `f_θ` but its weights are an **exponential moving average** (EMA) of
the context encoder's weights:

```{math}
\bar{θ} \leftarrow m \cdot \bar{θ} + (1 - m) \cdot θ
```

where `m` is the momentum coefficient (default: cosine schedule from 0.996 to
1.0). The target encoder receives `stop-gradient`.

### Predictor

The predictor `g_φ` is a smaller transformer (default 6 layers, 384 dim) that
predicts target patch representations from context patch representations.

Key design:

| Property | Detail |
|---|---|
| **Lighter than the encoder** | Fewer layers, smaller hidden dim |
| **Positional embeddings for all patches** | The predictor knows which target patches to predict |
| **Mask tokens for target positions** | Learnable embeddings substituted for masked patches |

### Masking

I-JEPA uses **multi-block masking**: random rectangular blocks are masked
rather than individual patches.

```python
config.num_enc_masks = 1           # Number of context blocks
config.enc_mask_scale = (0.15, 0.2)   # Context covers 15-20% of image
config.num_pred_masks = 4          # Number of target blocks
config.pred_mask_scale = (0.15, 0.2)  # Each target is 15-20%
config.aspect_ratio = (0.75, 1.5)     # Block aspect ratio range
```

The predictor sees the context patches and must predict the representation of
each target block's patches. With 4 target blocks and context covering ~15-20%,
most of the image must be predicted from a small visible region.

## Training

### Loss Function

The I-JEPA loss is the L2 distance between predicted and target representations,
averaged over masked patches:

```{math}
\mathcal{L}_{\text{JEPA}} =
\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}}
\left\| g_φ(f_θ(x)_i + \text{mask\_token}, \text{pos}_i) -
\text{sg}(f_{\bar{θ}}(x)_i) \right\|_2^2
```

### Optimization

```{math}
\begin{aligned}
\text{Context encoder: } & θ \leftarrow \text{optimizer}(θ, \nabla_θ \mathcal{L}) \\
\text{Predictor: } & φ \leftarrow \text{optimizer}(φ, \nabla_φ \mathcal{L}) \\
\text{Target encoder: } & \bar{θ} \leftarrow m \cdot \bar{θ} + (1 - m) \cdot θ
\end{aligned}
```

### Learning Rate Schedule

Three-phase schedule:
1. **Warmup** (0 → `warmup_epochs`): Linear increase from 0 to `lr`
2. **Cosine decay** (`warmup_epochs` → `epochs`): Cosine annealing to `min_lr`
3. **Constant**: After `epochs`, remains at `min_lr`

## Usage in TorchWM

### Quick start

```python
import torchwm

agent = torchwm.create_model(
    "jepa",
    dataset="imagenet",
    batch_size=64,
    epochs=100,
)
agent.train()
```

### Using config directly

```python
from torchwm import JEPAAgent, JEPAConfig

cfg = JEPAConfig()
cfg.dataset = "imagenet1k"
cfg.root_path = "/data/imagenet"
cfg.image_folder = "train"
cfg.batch_size = 64
cfg.epochs = 100
cfg.lr = 1.5e-4

agent = JEPAAgent(cfg)
agent.train()
```

### Data pipeline

```python
cfg.dataset = "imagenet1k"     # ImageNet-1K (requires download)
cfg.root_path = "/data/imagenet"

# Or use a generic image folder:
cfg.dataset = "imagefolder"
cfg.root_path = "./my_dataset"
cfg.image_folder = "train"

# Or CIFAR-10 for testing:
cfg.dataset = "cifar10"
cfg.download = True
```

```{note}
JEPA does NOT rely on heavy augmentation like contrastive methods. The
core learning signal comes from the masking prediction task, not from image distortion.
```

### CLI

```bash
torchwm train jepa --dataset imagenet1k --epochs 100 --batch_size 64
```

See {doc}`configs_reference` for the full JEPAConfig field reference with defaults.

## Inference and Downstream Tasks

```python
cfg.eval = True
cfg.read_checkpoint = "./output/checkpoint.pth"
```

### Linear probing protocol

| Method | Top-1 Accuracy (ViT-B/16) |
|--------|---------------------------|
| I-JEPA | 72.4% |
| MAE | 68.5% |
| iBOT | 74.7% |
| DINOv2 | 78.3% |

## I-JEPA vs V-JEPA

| Aspect | I-JEPA (Image) | V-JEPA (Video) |
|--------|----------------|-----------------|
| Input | Single image | Video clip |
| Masking | Spatial block masking | Spatio-temporal tube masking |
| Task | Predict masked patch latents | Predict future frame latents |
| Predictor | Transformer | Spatio-temporal transformer |

## Common Pitfalls

### Predictor collapse

The predictor outputs a constant regardless of input.

**Fixes:**
- Ensure EMA starts close to 1.0 (default: 0.996)
- Verify predictor output variance is non-zero

### Representation collapse

All patches map to nearly identical representations.

**Fixes:**
- Use multi-block masking (not random patch masking)
- Check the feature covariance matrix

### Memory usage

ViT-B/16 with 224×224 creates 196 patch tokens. Batch size 64 requires ~16 GB GPU.

**Tips:**
- Enable `gradient_checkpointing = True`
- Reduce `batch_size` and increase `accum_iter`

### Slow convergence

JEPA requires long warmup (40 epochs) and many total epochs (100–300).

**Tips:**
- Use the cosine schedule for EMA momentum
- Expect 48+ hours on 4× GPUs for ViT-B/16 at 100 epochs

## Comparison to Other Methods

| Method | What it predicts | Approach |
|--------|------------------|----------|
| Autoencoder | Pixels | Reconstruction |
| VAE | Pixels | Generative |
| MAE | Pixels | Masked modeling |
| JEPA | Latents | Predictive coding |
| IRIS | Tokens | Transformer dynamics |

## See Also

- {doc}`iris` — discrete world model using JEPA-style token prediction
- {doc}`vision_guide` — ViT encoder and video tokenizer components

## References

- Bardes, A., Ponce, J., & LeCun, Y. (2023). I-JEPA: Image-based Joint Embedding Predictive Architecture. *arXiv:2301.08243.*
- Assran, M., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *CVPR 2023.*
- Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021.*
