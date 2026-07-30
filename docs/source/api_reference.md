# API Reference

This page lists every public class, function, and constant exported from the
`torchwm` top-level package, grouped by category.

```{contents} Contents
:depth: 2
```

## Factory helpers

| Name | Description |
|---|---|
| `create_config(model, **overrides)` | Return a default config dict for a model family, with optional overrides applied. Models: `dreamer`, `jepa`, `iris`, `dit`, `genie`, `planet`. |
| `create_model(model, config=None, **overrides)` | Instantiate a model or high-level agent by canonical name (see {doc}`public_api`). |
| `make_env(env_id, backend="auto", **kwargs)` | Create a TorchWM environment through a named backend. Backends: `dmc`, `gym`, `atari`, `mujoco`, `robotics`, `procgen`, `brax`, `bsuite`, `unity`. |
| `list_models()` | Return canonical model names accepted by `create_model`. |
| `list_env_backends()` | Return backend names accepted by `make_env`. |
| `list_envs(model=None)` | Return known environment IDs, optionally filtered by model family. |
| `get_model_spec(name)` | Return metadata ({py:class}`ModelSpec`) for a model name or alias. |
| `get_env_backend_spec(name)` | Return metadata ({py:class}`EnvBackendSpec`) for an environment backend. |
| `get_operator(model, **params)` | Return an {py:class}`OperatorABC` instance for preprocessing model inputs. |

## Data classes

| Name | Description |
|---|---|
| `ModelSpec` | Named tuple describing a registered model (name, import_path, config_path, aliases, description). |
| `EnvBackendSpec` | Named tuple describing a registered environment backend (name, factory_path, aliases, description). |
| `TensorSpec` | Specification for operator input/output tensors (shape, dtype, name). |
| `OperatorABC` | Abstract base class for inference operators. |

## World model agents

| Name | Source | Description |
|---|---|---|
| `Dreamer` | `world_models.models.dreamer` | Base Dreamer world model (RSSM-based, V1-style). |
| `DreamerV1` | `world_models.models.dreamer_v1` | DreamerV1 (alias for base Dreamer). |
| `DreamerV2` | `world_models.models.dreamer_v2` | DreamerV2 (symlog two-hot heads, balanced KL). |
| `DreamerV3` | `world_models.models.dreamer_v3` | DreamerV3 core model (categorical latents, free bits, return normalization). |
| `DreamerV3Agent` | `world_models.models.dreamer_v3` | High-level DreamerV3 agent with train/evaluate helpers. |
| `CategoricalRSSM` | `world_models.models.categorical_rssm` | DreamerV3 RSSM with straight-through categorical latents. |
| `DreamerAgent` | `world_models.models.dreamer` | High-level Dreamer agent with train/evaluate helpers. |
| `Planet` | `world_models.models.planet` | PlaNet: Deep Planning Network. |
| `JEPAAgent` | `world_models.models.jepa_agent` | I-JEPA agent for self-supervised visual representation learning. |
| `IRISAgent` | `world_models.models.iris_agent` | IRIS agent for sample-efficient RL with Transformers. |
| `Genie` | `world_models.models.genie` | Genie generative interactive environment. |
| `create_genie` | `world_models.models.genie` | Create a Genie model with specified parameters. |
| `create_genie_small` | `world_models.models.genie` | Create Genie-small variant (~50M params). |
| `create_genie_large` | `world_models.models.genie` | Create Genie-large variant (~11B params). |

## Genie subcomponents

| Name | Source | Description |
|---|---|---|
| `LatentActionModel` | `world_models.models.latent_action_model` | Learns latent actions from pairs of video frames. |
| `DynamicsModel` | `world_models.models.dynamics_model` | Transformer-based dynamics for future token prediction. |
| `create_latent_action_model` | `world_models.models.latent_action_model` | Factory for LatentActionModel. |
| `create_dynamics_model` | `world_models.models.dynamics_model` | Factory for DynamicsModel. |

## State-space models

| Name | Source | Description |
|---|---|---|
| `RSSM` | `world_models.models.rssm` | Recurrent State-Space Model (standalone). |
| `RecurrentStateSpaceModel` | `world_models.models.rssm` | Alias for RSSM. |
| `DreamerRSSM` | `world_models.models.dreamer_rssm` | RSSM variant used in Dreamer training loop. |
| `ModularRSSM` | `world_models.models.modular_rssm` | Modular RSSM with swappable encoder/backbone/decoder. |
| `create_modular_rssm` | `world_models.models.modular_rssm` | Factory for ModularRSSM. |

## Diffusion models

| Name | Source | Description |
|---|---|---|
| `DiT` | `world_models.models.diffusion` | Diffusion Transformer model. |
| `create_dit` | `world_models.models.diffusion` | Factory for DiT. |
| `PatchEmbed` | `world_models.models.diffusion` | Image-to-patch embedding layer. |
| `PatchUnEmbed` | `world_models.models.diffusion` | Patch-to-image un-embedding layer. |
| `DDPM` | `world_models.models.diffusion` | Denoising Diffusion Probabilistic Model. |
| `ActorCriticNetwork` | `world_models.models.diffusion` | Actor-critic head for DIAMOND-style RL. |
| `RewardTerminationModel` | `world_models.models.diffusion` | Reward + termination predictor for DIAMOND. |
| `sinusoidal_time_embedding` | `world_models.models.diffusion` | Time-step embedding for diffusion. |

## Vision components

| Name | Source | Description |
|---|---|---|
| `ConvEncoder` | `world_models.vision.dreamer_encoder` | Dreamer convolutional encoder (image → embedding). |
| `CNNEncoder` | `world_models.vision.planet_encoder` | PlaNet CNN encoder (image → embedding). |
| `IRISEncoder` | `world_models.vision.iris_encoder` | IRIS encoder (image → discrete tokens). |
| `ConvDecoder` | `world_models.vision.dreamer_decoder` | Dreamer convolutional decoder (latent → image distribution). |
| `CNNDecoder` | `world_models.vision.planet_decoder` | PlaNet CNN decoder. |
| `DenseDecoder` | `world_models.vision.dreamer_decoder` | MLP decoder for reward/value/discount. |
| `ActionDecoder` | `world_models.vision.dreamer_decoder` | Dreamer policy head (latent → tanh-squashed action). |
| `IRISDecoder` | `world_models.vision.iris_decoder` | IRIS decoder (tokens → image). |
| `VideoTokenizer` | `world_models.vision.video_tokenizer` | Genie VQ-VAE video tokenizer. |
| `create_video_tokenizer` | `world_models.vision.video_tokenizer` | Factory for VideoTokenizer. |
| `VectorQuantizer` | `world_models.vision.vq_layer` | VQ-VAE vector quantization layer. |
| `VectorQuantizerEMA` | `world_models.vision.vq_layer` | VQ-VAE with EMA codebook updates. |
| `TanhBijector` | `world_models.vision.dreamer_decoder` | Tanh bijection for action squashing. |
| `SampleDist` | `world_models.vision.dreamer_decoder` | MC-sampled distribution statistics. |

## Config classes

| Name | Source | Description |
|---|---|---|
| `DreamerConfig` | `world_models.configs.dreamer_config` | Dreamer hyperparameter config. |
| `JEPAConfig` | `world_models.configs.jepa_config` | JEPA hyperparameter config. |
| `DiTConfig` | `world_models.configs.dit_config` | DiT hyperparameter config. |
| `get_dit_config` | `world_models.configs.dit_config` | Factory for DiTConfig with presets. |
| `DiamondConfig` | `world_models.configs.diamond_config` | DIAMOND hyperparameter config. |
| `IRISConfig` | `world_models.configs.iris_config` | IRIS hyperparameter config. |
| `GenieConfig` | `world_models.configs.genie_config` | Genie hyperparameter config. |
| `GenieSmallConfig` | `world_models.configs.genie_config` | Genie-small preset config. |
| `STTransformerConfig` | `world_models.configs.st_transformer_config` | ST-Transformer config. |
| `VideoTokenizerConfig` | `world_models.configs.video_tokenizer_config` | Video tokenizer config. |
| `LatentActionModelConfig` | `world_models.configs.lam_config` | Latent action model config. |
| `DynamicsModelConfig` | `world_models.configs.dynamics_config` | Dynamics model config. |

## Constants

| Name | Description |
|---|---|
| `MODEL_SPECS` | Dict of all built-in model specs (name → {py:class}`ModelSpec`). |
| `ENV_BACKEND_SPECS` | Dict of all built-in environment backend specs. |
| `ATARI_100K_GAMES` | List of Atari 100K benchmark game names. |
| `HUMAN_SCORES` | Dict of human baseline scores for Atari 100K. |
| `RANDOM_SCORES` | Dict of random baseline scores for Atari 100K. |

## Memory / replay buffers

| Name | Source | Description |
|---|---|---|
| `ReplayBuffer` | `world_models.memory.dreamer_memory` | Dreamer ring buffer (transitions → sequences). |
| `Memory` | `world_models.memory.planet_memory` | Episode-based memory for PlaNet. |
| `Episode` | `world_models.memory.planet_memory` | Single episode recording. |
| `IRISReplayBuffer` | `world_models.memory.iris_memory` | Ring buffer for IRIS (uint8 images). |
| `IRISOnPolicyBuffer` | `world_models.memory.iris_memory` | On-policy buffer for episode collection. |

## Environments and wrappers

| Name | Source | Description |
|---|---|---|
| `DeepMindControlEnv` | `world_models.envs.dmc_env` | DeepMind Control Suite adapter. |
| `DMLabEnv` | `world_models.envs.dmlab_env` | DeepMind Lab adapter. |
| `make_dmlab_env` | `world_models.envs.dmlab_env` | Factory for DeepMind Lab. |
| `DMLAB_LEVELS` | `world_models.envs.dmlab_env` | Available DMLab level names. |
| `GymImageEnv` | `world_models.envs.gym_env` | Gymnasium image adapter. |
| `make_gym_env` | `world_models.envs.gym_env` | Factory for GymImageEnv. |
| `MuJoCoImageEnv` | `world_models.envs.mujoco_env` | MuJoCo image adapter. |
| `make_mujoco_env` | `world_models.envs.mujoco_env` | Factory for MuJoCo environments. |
| `MujocoEnv` | `world_models.envs.mujoco_env` | Raw MuJoCo environment wrapper. |
| `BraxImageEnv` | `world_models.envs.brax_env` | Brax image adapter. |
| `make_brax_env` | `world_models.envs.brax_env` | Factory for BraxImageEnv. |
| `BSuiteImageEnv` | `world_models.envs.bsuite_env` | BSuite image adapter. |
| `make_bsuite_env` | `world_models.envs.bsuite_env` | Factory for BSuiteImageEnv. |
| `list_available_bsuite_ids` | `world_models.envs.bsuite_env` | List BSuite environment IDs. |
| `make_atari_env` | `world_models.envs.atari_env` | Factory for Atari ALE environments. |
| `list_available_atari_envs` | `world_models.envs.atari_env` | List available Atari game IDs. |
| `make_atari_vector_env` | `world_models.envs.atari_env` | Factory for vectorized Atari. |
| `make_diamond_atari_env` | `world_models.envs.diamond_atari` | DIAMOND-style Atari preprocessing. |
| `make_procgen_env` | `world_models.envs.procgen_env` | Factory for Procgen environments. |
| `make_robotics_env` | `world_models.envs.robotics_env` | Factory for Gymnasium Robotics. |
| `register_gymnasium_robotics_envs` | `world_models.envs.robotics_env` | Register Robotics envs. |
| `list_gymnasium_robotics_envs` | `world_models.envs.robotics_env` | List installed Robotics envs. |
| `UnityMLAgentsEnv` | `world_models.envs.unity_env` | Unity ML-Agents adapter. |
| `make_unity_mlagents_env` | `world_models.envs.unity_env` | Factory for UnityMLAgentsEnv. |
| `WorldModelEnv` | `world_models.envs.world_model_env` | Environment inside a learned world model. |
| `make_world_model_env` | `world_models.envs.world_model_env` | Factory for WorldModelEnv. |
| `TimeLimit` | `world_models.envs.wrappers` | Episode time limit wrapper. |
| `ActionRepeat` | `world_models.envs.wrappers` | Action repeat wrapper. |
| `NormalizeActions` | `world_models.envs.wrappers` | Action normalization to [-1, 1]. |
| `ObsDict` | `world_models.envs.wrappers` | Observation-to-dict conversion. |
| `OneHotAction` | `world_models.envs.wrappers` | Discrete to one-hot action conversion. |
| `RewardObs` | `world_models.envs.wrappers` | Reward observation injection. |
| `ResizeImage` | `world_models.envs.wrappers` | Image resizing wrapper. |
| `RenderImage` | `world_models.envs.wrappers` | Render-based image observation. |
| `SelectAction` | `world_models.envs.wrappers` | Action selection wrapper. |

## Controllers and policies

| Name | Source | Description |
|---|---|---|
| `RSSMPolicy` | `world_models.controller` | RSSM-based policy for Dreamer. |
| `RolloutGenerator` | `world_models.controller` | Policy rollouts in the environment. |
| `IRISActor` | `world_models.controller` | IRIS actor head. |
| `IRISCritic` | `world_models.controller` | IRIS critic head. |
| `IRISPolicy` | `world_models.controller` | IRIS combined actor-critic policy. |
| `CNNFeatureExtractor` | `world_models.controller` | CNN feature extractor for policy inputs. |

## Inference operators

| Name | Source | Description |
|---|---|---|
| `DreamerOperator` | `world_models.inference.operators` | Dreamer input preprocessing. |
| `JEPAOperator` | `world_models.inference.operators` | JEPA image masking and patching. |
| `IrisOperator` | `world_models.inference.operators` | IRIS sequence tokenization. |
| `PlaNetOperator` | `world_models.inference.operators` | PlaNet action/state preprocessing. |

## Export

| Name | Source | Description |
|---|---|---|
| `export_any(obj, path, format, ...)` | `world_models.export` | Export a model or agent to ONNX/TorchScript/TensorRT. |
| `export_model(module, path, format, ...)` | `world_models.export` | Export a raw nn.Module. |
| `ExportableAgentMixin` | `world_models.export` | Mixin that adds `.export()` to custom agents. |

## Reward and value models

| Name | Source | Description |
|---|---|---|
| `RewardModel` | `world_models.reward` | Base reward model. |
| `ValueModel` | `world_models.reward` | Base value model. |
| `DreamerRewardModel` | `world_models.reward` | Dreamer reward predictor. |
| `DreamerValueModel` | `world_models.reward` | Dreamer value function. |

## Transformer blocks

| Name | Source | Description |
|---|---|---|
| `STTransformer` | `world_models.blocks` | Spatiotemporal Transformer (Genie). |
| `MultiHeadSelfAttention` | `world_models.blocks` | Multi-head self-attention. |
| `MultiHeadAttention` | `world_models.blocks` | Multi-head cross-attention. |
| `AdaLNNormalization` | `world_models.blocks` | Adaptive layer norm for diffusion. |
| `RMSNorm` | `world_models.blocks` | Root mean square layer norm. |

## Plugin registry

| Name | Source | Description |
|---|---|---|
| `register_world_model(name, import_path, ...)` | `world_models.registry` | Register a custom world model architecture. |
| `deregister_world_model(name)` | `world_models.registry` | Remove a registered model. |
| `get_registered_model_spec(name)` | `world_models.registry` | Look up a registered model spec. |
| `list_registered_models()` | `world_models.registry` | List all externally registered model names. |
| `register_env_backend(name, factory_path, ...)` | `world_models.registry` | Register a custom environment backend. |
| `deregister_env_backend(name)` | `world_models.registry` | Remove a registered env backend. |
| `list_registered_env_backends()` | `world_models.registry` | List all registered env backends. |

## Deprecation utilities

| Name | Source | Description |
|---|---|---|
| `deprecated(version, reason)` | `world_models.utils.deprecation` | Decorator to mark functions/classes as deprecated. |
| `deprecated_class(version, alternative)` | `world_models.utils.deprecation` | Shortcut for deprecating a class. |
| `deprecated_function(version, alternative)` | `world_models.utils.deprecation` | Shortcut for deprecating a function. |

## General utilities

| Name | Source | Description |
|---|---|---|
| `Logger` | `world_models.utils` | Logging utility. |
| `FreezeParameters` | `world_models.utils` | Context manager to freeze model parameters. |
| `compute_return(rewards, values, gamma, lambda_)` | `world_models.utils` | Compute GAE or λ-return. |
| `preprocess_obs(obs)` | `world_models.utils` | Observation preprocessing (resize, normalize). |

## Version

| Name | Description |
|---|---|
| `__version__` | Package version string (semver). |
