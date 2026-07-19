# Package Overview

TorchWM is organized into focused modules so you can use only the pieces you need.

## Quick Import (Public API)

For applications and examples, prefer the installed package name, `torchwm`. It
mirrors the TorchWM implementation package and exposes the same lazy public
API without importing optional training backends until you use them.

```python
import torchwm

print(torchwm.list_models())
agent = torchwm.create_model("dreamer", env="walker-walk", total_steps=1_000_000)
env = torchwm.make_env("CartPole-v1", backend="gym")
op = torchwm.get_operator("dreamer", image_size=64, action_dim=6)
```

Use `torchwm` for direct component imports as well as factory helpers:

```python
from torchwm import DreamerAgent, DreamerConfig

cfg = DreamerConfig()
cfg.env = "walker-walk"
agent = DreamerAgent(cfg)
```

### Available Exports

| Category | Exports |
|----------|--------|
| **Friendly factories** | `create_config`, `create_model`, `make_env`, `list_models`, `list_env_backends`, `list_envs` |
| **Models / Agents** | `Dreamer`, `DreamerV1`, `DreamerV2`, `DreamerV3`, `DreamerAgent`, `Planet`, `JEPAAgent`, `IRISAgent`, `Genie`, `create_genie`, `DiT`, `create_dit` |
| **State-space models** | `RSSM`, `RecurrentStateSpaceModel`, `DreamerRSSM`, `ModularRSSM`, `create_modular_rssm` |
| **Vision** | `ConvEncoder`, `CNNEncoder`, `IRISEncoder`, `ConvDecoder`, `CNNDecoder`, `DenseDecoder`, `ActionDecoder`, `IRISDecoder`, `VideoTokenizer`, `create_video_tokenizer` |
| **Quantization** | `VectorQuantizer`, `VectorQuantizerEMA` |
| **Configs** | `DreamerConfig`, `JEPAConfig`, `DiTConfig`, `DiamondConfig`, `IRISConfig`, `GenieConfig`, `GenieSmallConfig`, `STTransformerConfig`, `VideoTokenizerConfig`, `LatentActionModelConfig`, `DynamicsModelConfig` |
| **Environments** | `make_atari_env`, `make_gym_env`, `make_mujoco_env`, `make_robotics_env`, `make_brax_env`, `make_procgen_env`, `GymImageEnv`, `ProcgenImageEnv`, `DeepMindControlEnv`, `DMLabEnv`, `make_dmlab_env`, `UnityMLAgentsEnv`, `TimeLimit`, `ActionRepeat`, wrappers, etc. |
| **Memory** | `ReplayBuffer`, `Memory`, `Episode`, `IRISReplayBuffer`, `IRISOnPolicyBuffer` |
| **Operators** | `get_operator`, `DreamerOperator`, `JEPAOperator`, `IrisOperator`, `PlaNetOperator` |
| **Inference** | `DreamerOperator`, `JEPAOperator`, `IrisOperator`, `PlaNetOperator` |
| **Reward / Value** | `RewardModel`, `ValueModel`, `DreamerRewardModel`, `DreamerValueModel` |
| **Controllers** | `RSSMPolicy`, `RolloutGenerator`, `IRISPolicy`, `IRISActor`, `IRISCritic`, `CNNFeatureExtractor` |
| **Transformer blocks** | `STTransformer`, `MultiHeadSelfAttention`, `MultiHeadAttention`, `AdaLNNormalization`, `RMSNorm` |
| **Diffusion** | `DiT`, `DDPM`, `PatchEmbed`, `PatchUnEmbed`, `ActorCriticNetwork`, `RewardTerminationModel` |
| **Genie subcomponents** | `LatentActionModel`, `DynamicsModel`, `create_latent_action_model`, `create_dynamics_model` |
| **Export** | `export_any`, `export_model`, `ExportableAgentMixin` |
| **Registry / plugins** | `register_world_model`, `deregister_world_model`, `register_env_backend`, `deregister_env_backend` |
| **Deprecation** | `deprecated`, `deprecated_class`, `deprecated_function` |
| **Utilities** | `Logger`, `FreezeParameters`, `compute_return`, `preprocess_obs`, `__version__` |

Example usage:

```python
import torchwm

# Training
agent = torchwm.create_model("dreamer", env="walker-walk", total_steps=1_000_000)
agent.train()

# Inference preprocessing
op = torchwm.get_operator("dreamer", image_size=64, action_dim=6)
processed = op.process({"image": image, "action": action})
```

## Core Modules

The module paths below are the public `torchwm.*` surface. They mirror the
internal `world_models` implementation package one-to-one, so every submodule is
importable either way (`from torchwm.models import Dreamer`).

- `torchwm.models`: High-level models and agents (`Dreamer`, `DreamerAgent`, `Planet`, `JEPAAgent`)
- `torchwm.configs`: Configuration containers for Dreamer, JEPA, and diffusion runs
- `torchwm.training`: Script-style training entrypoints for world models (VAE, MDNRNN, Controller, Planet, RSSM, JEPA)

## Environment Integration

- `torchwm.envs`: DMC, Gym/Gymnasium, Atari, MuJoCo, Unity ML-Agents adapters
- `torchwm.envs.wrappers`: Action repeat, normalization, time limits

## World Model Building Blocks

- `torchwm.models.dreamer_rssm`: Recurrent state-space model used by Dreamer
- `torchwm.models.modular_rssm`: Modular RSSM with swappable encoder/decoder/backbone for research experiments
- `torchwm.vision`: Encoders/decoders and action heads for latent dynamics models
- `torchwm.reward`: Reward and value prediction heads
- `torchwm.observations`: Symbolic and visual observation reconstruction modules

## Representation Learning and Diffusion

- `torchwm.models.vit`: Vision Transformer and JEPA predictor components
- `torchwm.models.diffusion`: DDPM scheduler and DiT model implementation
- `torchwm.masks`: Mask collators for JEPA-style context/target masking

## Data and Memory

- `torchwm.datasets`: CIFAR-10, ImageNet-1K, and generic `ImageFolder` dataset loaders
- `torchwm.memory`: Replay buffers for Dreamer and episode-based memory for PlaNet/RSSM

## Utilities

- `torchwm.utils`: Logging, parameter freezing, transforms
- `torchwm.transforms`: Data augmentation pipelines
- `torchwm.benchmarks`: CLI and reporting utilities

## Which API Should I Use?

- End-to-end Dreamer training: `DreamerAgent`
- End-to-end JEPA training: `JEPAAgent`
- World model training scripts: `torchwm.training` modules (e.g., `train_world_model` for VAE+MDNRNN+Controller pipeline)
- Low-level model experimentation: `Dreamer`, `RSSM`, decoder/encoder modules
- Custom world model architectures: `ModularRSSM` with swappable encoder/decoder/backbone
- Custom data pipelines: `make_cifar10`, `make_imagenet1k`, `make_imagefolder`
