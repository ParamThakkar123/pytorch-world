# Getting Started

## Installation

Install from PyPI:

```bash
pip install torchwm
```

Install from source:

```bash
git clone https://github.com/ParamThakkar123/torchwm.git
cd torchwm
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

## Logging with Weights & Biases and TensorBoard

TorchWM supports logging experiment results to Weights & Biases (WandB) and TensorBoard.

### Weights & Biases

To use WandB logging, set the `WANDB_API_KEY` environment variable (anonymous logins are no longer supported). You can get your key from [wandb.ai](https://wandb.ai/settings).

```python
cfg.enable_wandb = True
cfg.wandb_project = "torchwm"
cfg.wandb_entity = "your-entity"
```

### TensorBoard

Enable TensorBoard logging:

```python
cfg.enable_tensorboard = True
cfg.log_dir = "runs"
```

Logs will be saved to the specified directory and can be viewed with `tensorboard --logdir runs`.

## Quick Start: Friendly API

The recommended entrypoint for common workflows is `torchwm`. It mirrors the
TorchWM implementation package, but gives users short factory helpers for
discovery, model creation, environment creation, and operators.

```python
import torchwm

print(torchwm.list_models())
print(torchwm.list_env_backends())

# Runs on `pip install torchwm[gym]`. Use env="walker-walk" (default backend)
# with `pip install torchwm[dmc]` for DeepMind Control tasks.
agent = torchwm.create_model(
    "dreamer", env="Pendulum-v1", env_backend="gym", total_steps=5_000
)
env = torchwm.make_env("CartPole-v1", backend="gym")
op = torchwm.get_operator("dreamer", image_size=64, action_dim=6)
```

You can still import direct research components from `torchwm` when you
need lower-level control:

```python
from torchwm import DreamerAgent, DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "gym"       # or the default "dmc" with torchwm[dmc] installed
cfg.env = "Pendulum-v1"
agent = DreamerAgent(cfg)
```

## Quick Start: Dreamer

TorchWM implements multiple world model algorithms. Click on each to see detailed documentation:

| Algorithm | Description | Quick Start |
|-----------|-------------|--------------|
| **Dreamer** | Model-based RL with latent dynamics | {doc}`dreamer` |
| **JEPA** | Self-supervised visual representations | {doc}`jepa` |
| **IRIS** | Sample-efficient RL with Transformers | {doc}`iris` |
| **DiT** | Diffusion models with Transformers | {doc}`dit` |

## Quick Start: Inference with Operators

TorchWM now includes standardized operators for preprocessing inputs during inference, making it easy to deploy models consistently.

### What are Operators?

Operators handle input preprocessing: normalizing images, encoding actions, tokenizing text, and generating masks. Each model has a dedicated operator that ensures inputs are in the correct format.

### Basic Usage

```python
import torchwm

# Create operator with config parameters
op = torchwm.get_operator(
    "dreamer",
    image_size=64,  # Image size for Dreamer
    action_dim=6    # Action dimension
)

# Process raw inputs
raw_inputs = {
    'image': your_pil_image_or_tensor,  # PIL Image or torch.Tensor
    'action': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # Action as list
}

# Get processed tensors
processed = op.process(raw_inputs)
# Returns: {'obs': tensor(B, 3, 64, 64), 'action': tensor(B, 6)}
```

### Available Operators

| Operator | Model | Purpose | Key Parameters |
|----------|-------|---------|----------------|
| `DreamerOperator` | Dreamer | Image/action preprocessing | `image_size`, `action_dim` |
| `JEPAOperator` | JEPA | Image masking and patching | `image_size`, `patch_size`, `mask_ratio` |
| `IrisOperator` | IRIS | Sequence tokenization | `seq_length`, `vocab_size` |
| `PlaNetOperator` | PlaNet | State/action transitions | `state_dim`, `action_dim` |

### JEPA Example (Self-Supervised)

```python
from torchwm import JEPAOperator

op = JEPAOperator(image_size=224, patch_size=16, mask_ratio=0.75)
inputs = {'images': [image1, image2]}
result = op(inputs)
# result['images']: stacked normalized images
# result['mask']: random mask for self-supervised learning
```

### IRIS Example (Sequence Processing)

```python
from torchwm import IrisOperator

op = IrisOperator(seq_length=512, vocab_size=32000)
inputs = {'tokens': [101, 2054, 2003, 102]}  # Token sequence
result = op(inputs)
# result['input_ids']: padded token tensor
# result['attention_mask']: attention mask
```

### Integration with Configs

Operators can reuse matching config fields, but operator-only parameters such as
action dimensions should be supplied from the target environment:

```python
import torchwm

cfg = torchwm.create_config("dreamer")
op = torchwm.get_operator(
    "dreamer",
    image_size=cfg.image_size[0],
    action_dim=6,
)
```

### Utilities

For most applications, use `torchwm.get_operator()` for preprocessing. Advanced utility functions remain available to package internals.

```python
import torchwm

op = torchwm.get_operator("jepa", image_size=224, patch_size=16, mask_ratio=0.75)
processed = op.process({"images": [pil_image]})
```

Train a complete world model pipeline (VAE + MDNRNN + Controller) on any Gym environment:

```bash
# Train on CarRacing
python -m world_models.training.train_world_model --env CarRacing-v2

# Train on Pendulum
python -m world_models.training.train_world_model --env Pendulum-v1

# Test trained model
python -m world_models.training.train_world_model --env CarRacing-v2 --test

# Specify action size manually for environments with missing dependencies
python -m world_models.training.train_world_model --env BipedalWalker-v3 --action_size 4
```

Dreamer supports multiple backends through `DreamerConfig.env_backend`; the
top-level `torchwm.make_env()` helper uses the same backend names for standalone
environment creation:

| Backend | Description |
|---|---|
| `dmc` | DeepMind Control Suite tasks (e.g. `walker-walk`) |
| `dmlab` | DeepMind Lab 3D navigation tasks (e.g. `rooms_collect_good_objects_train`) |
| `gym` | Gym/Gymnasium environment IDs or an existing environment instance |
| `mujoco` | Gymnasium MuJoCo task IDs or native MJCF/MJB models |
| `robotics` | Any ID registered by the installed Gymnasium Robotics package |
| `procgen` | Procgen benchmark games such as `coinrun` and `heist` |
| `brax` | JAX/Brax continuous-control environments |
| `unity_mlagents` | Unity ML-Agents executable environments |

## Typical Training Flow

1. Choose an algorithm (Dreamer, JEPA, IRIS, or DiT)
2. Create a config object for that algorithm
3. Override dataset/environment and optimization fields
4. Instantiate the corresponding agent
5. Call `train()` and monitor logs/checkpoints

For complete API details, see {doc}`api_reference`.
