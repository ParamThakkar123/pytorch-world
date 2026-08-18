# Exported Names by Category

This page lists every public class, function, and constant exported from the
`torchwm` top-level package, grouped by category. For signatures and docstrings
generated from source, see {doc}`api_reference`.

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

## Data classes

| Name | Description |
|---|---|
| `ModelSpec` | Named tuple describing a registered model (name, import_path, config_path, aliases, description). |
| `EnvBackendSpec` | Named tuple describing a registered environment backend (name, factory_path, aliases, description). |

## World model agents

| Name | Source | Description |
|---|---|---|
| `Dreamer` | `torchwm.models.dreamer` | Base Dreamer world model (RSSM-based, V1-style). |
| `DreamerV1` | `torchwm.models.dreamer_v1` | DreamerV1 (alias for base Dreamer). |
| `DreamerV2` | `torchwm.models.dreamer_v2` | DreamerV2 (symlog two-hot heads, balanced KL). |
| `DreamerV3` | `torchwm.models.dreamer` | DreamerV3-style agent (currently mapped to DreamerAgent). |
| `DreamerAgent` | `torchwm.models.dreamer` | High-level Dreamer agent with train/evaluate helpers. |
| `Planet` | `torchwm.models.planet` | PlaNet: Deep Planning Network. |
| `JEPAAgent` | `torchwm.models.jepa_agent` | I-JEPA agent for self-supervised visual representation learning. |
| `IRISAgent` | `torchwm.models.iris_agent` | IRIS agent for sample-efficient RL with Transformers. |
| `Genie` | `torchwm.models.genie` | Genie generative interactive environment. |
| `create_genie` | `torchwm.models.genie` | Create a Genie model with specified parameters. |
| `create_genie_small` | `torchwm.models.genie` | Create Genie-small variant (~50M params). |
| `create_genie_large` | `torchwm.models.genie` | Create Genie-large variant (~11B params). |

## Genie subcomponents

| Name | Source | Description |
|---|---|---|
| `LatentActionModel` | `torchwm.models.latent_action_model` | Learns latent actions from pairs of video frames. |
| `DynamicsModel` | `torchwm.models.dynamics_model` | Transformer-based dynamics for future token prediction. |
| `create_latent_action_model` | `torchwm.models.latent_action_model` | Factory for LatentActionModel. |
| `create_dynamics_model` | `torchwm.models.dynamics_model` | Factory for DynamicsModel. |

## State-space models

| Name | Source | Description |
|---|---|---|
| `RSSM` | `torchwm.models.rssm` | Recurrent State-Space Model (standalone). |
| `RecurrentStateSpaceModel` | `torchwm.models.rssm` | Alias for RSSM. |
| `DreamerRSSM` | `torchwm.models.dreamer_rssm` | RSSM variant used in Dreamer training loop. |
| `ModularRSSM` | `torchwm.models.modular_rssm` | Modular RSSM with swappable encoder/backbone/decoder. |
| `create_modular_rssm` | `torchwm.models.modular_rssm` | Factory for ModularRSSM. |

## Diffusion models

| Name | Source | Description |
|---|---|---|
| `DiT` | `torchwm.models.diffusion` | Diffusion Transformer model. |
| `create_dit` | `torchwm.models.diffusion` | Factory for DiT. |
| `PatchEmbed` | `torchwm.models.diffusion` | Image-to-patch embedding layer. |
| `PatchUnEmbed` | `torchwm.models.diffusion` | Patch-to-image un-embedding layer. |
| `DDPM` | `torchwm.models.diffusion` | Denoising Diffusion Probabilistic Model. |
| `ActorCriticNetwork` | `torchwm.models.diffusion` | Actor-critic head for DIAMOND-style RL. |
| `RewardTerminationModel` | `torchwm.models.diffusion` | Reward + termination predictor for DIAMOND. |
| `sinusoidal_time_embedding` | `torchwm.models.diffusion` | Time-step embedding for diffusion. |

## Vision components

| Name | Source | Description |
|---|---|---|
| `ConvEncoder` | `torchwm.vision.dreamer_encoder` | Dreamer convolutional encoder (image → embedding). |
| `CNNEncoder` | `torchwm.vision.planet_encoder` | PlaNet CNN encoder (image → embedding). |
| `IRISEncoder` | `torchwm.vision.iris_encoder` | IRIS encoder (image → discrete tokens). |
| `ConvDecoder` | `torchwm.vision.dreamer_decoder` | Dreamer convolutional decoder (latent → image distribution). |
| `CNNDecoder` | `torchwm.vision.planet_decoder` | PlaNet CNN decoder. |
| `DenseDecoder` | `torchwm.vision.dreamer_decoder` | MLP decoder for reward/value/discount. |
| `ActionDecoder` | `torchwm.vision.dreamer_decoder` | Dreamer policy head (latent → tanh-squashed action). |
| `IRISDecoder` | `torchwm.vision.iris_decoder` | IRIS decoder (tokens → image). |
| `VideoTokenizer` | `torchwm.vision.video_tokenizer` | Genie VQ-VAE video tokenizer. |
| `create_video_tokenizer` | `torchwm.vision.video_tokenizer` | Factory for VideoTokenizer. |
| `VectorQuantizer` | `torchwm.vision.vq_layer` | VQ-VAE vector quantization layer. |
| `VectorQuantizerEMA` | `torchwm.vision.vq_layer` | VQ-VAE with EMA codebook updates. |
| `TanhBijector` | `torchwm.vision.dreamer_decoder` | Tanh bijection for action squashing. |
| `SampleDist` | `torchwm.vision.dreamer_decoder` | MC-sampled distribution statistics. |

## Config classes

| Name | Source | Description |
|---|---|---|
| `DreamerConfig` | `torchwm.configs.dreamer_config` | Dreamer hyperparameter config. |
| `JEPAConfig` | `torchwm.configs.jepa_config` | JEPA hyperparameter config. |
| `DiTConfig` | `torchwm.configs.dit_config` | DiT hyperparameter config. |
| `get_dit_config` | `torchwm.configs.dit_config` | Factory for DiTConfig with presets. |
| `DiamondConfig` | `torchwm.configs.diamond_config` | DIAMOND hyperparameter config. |
| `IRISConfig` | `torchwm.configs.iris_config` | IRIS hyperparameter config. |
| `GenieConfig` | `torchwm.configs.genie_config` | Genie hyperparameter config. |
| `GenieSmallConfig` | `torchwm.configs.genie_config` | Genie-small preset config. |
| `STTransformerConfig` | `torchwm.configs.st_transformer_config` | ST-Transformer config. |
| `VideoTokenizerConfig` | `torchwm.configs.video_tokenizer_config` | Video tokenizer config. |
| `LatentActionModelConfig` | `torchwm.configs.lam_config` | Latent action model config. |
| `DynamicsModelConfig` | `torchwm.configs.dynamics_config` | Dynamics model config. |

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
| `ReplayBuffer` | `torchwm.memory.dreamer_memory` | Dreamer ring buffer (transitions → sequences). |
| `Memory` | `torchwm.memory.planet_memory` | Episode-based memory for PlaNet. |
| `Episode` | `torchwm.memory.planet_memory` | Single episode recording. |
| `IRISReplayBuffer` | `torchwm.memory.iris_memory` | Ring buffer for IRIS (uint8 images). |
| `IRISOnPolicyBuffer` | `torchwm.memory.iris_memory` | On-policy buffer for episode collection. |

## Environments and wrappers

| Name | Source | Description |
|---|---|---|
| `DeepMindControlEnv` | `torchwm.envs.dmc_env` | DeepMind Control Suite adapter. |
| `DMLabEnv` | `torchwm.envs.dmlab_env` | DeepMind Lab adapter. |
| `make_dmlab_env` | `torchwm.envs.dmlab_env` | Factory for DeepMind Lab. |
| `DMLAB_LEVELS` | `torchwm.envs.dmlab_env` | Available DMLab level names. |
| `GymImageEnv` | `torchwm.envs.gym_env` | Gymnasium image adapter. |
| `make_gym_env` | `torchwm.envs.gym_env` | Factory for GymImageEnv. |
| `MuJoCoImageEnv` | `torchwm.envs.mujoco_env` | MuJoCo image adapter. |
| `make_mujoco_env` | `torchwm.envs.mujoco_env` | Factory for MuJoCo environments. |
| `MujocoEnv` | `torchwm.envs.mujoco_env` | Alias of `MuJoCoImageEnv`. |
| `BraxImageEnv` | `torchwm.envs.brax_env` | Brax image adapter. |
| `make_brax_env` | `torchwm.envs.brax_env` | Factory for BraxImageEnv. |
| `BSuiteImageEnv` | `torchwm.envs.bsuite_env` | BSuite image adapter. |
| `make_bsuite_env` | `torchwm.envs.bsuite_env` | Factory for BSuiteImageEnv. |
| `list_available_bsuite_ids` | `torchwm.envs.bsuite_env` | List BSuite environment IDs. |
| `make_atari_env` | `torchwm.envs.atari_env` | Factory for Atari ALE environments. |
| `list_available_atari_envs` | `torchwm.envs.atari_env` | List available Atari game IDs. |
| `make_atari_vector_env` | `torchwm.envs.atari_env` | Factory for vectorized Atari. |
| `make_diamond_atari_env` | `torchwm.envs.diamond_atari` | DIAMOND-style Atari preprocessing. |
| `make_procgen_env` | `torchwm.envs.procgen_env` | Factory for Procgen environments. |
| `make_robotics_env` | `torchwm.envs.robotics_env` | Factory for Gymnasium Robotics. |
| `register_gymnasium_robotics_envs` | `torchwm.envs.robotics_env` | Register Robotics envs. |
| `list_gymnasium_robotics_envs` | `torchwm.envs.robotics_env` | List installed Robotics envs. |
| `UnityMLAgentsEnv` | `torchwm.envs.unity_env` | Unity ML-Agents adapter. |
| `make_unity_mlagents_env` | `torchwm.envs.unity_env` | Factory for UnityMLAgentsEnv. |
| `WorldModelEnv` | `torchwm.envs.world_model_env` | Environment inside a learned world model. |
| `make_world_model_env` | `torchwm.envs.world_model_env` | Factory for WorldModelEnv. |
| `TimeLimit` | `torchwm.envs.wrappers` | Episode time limit wrapper. |
| `ActionRepeat` | `torchwm.envs.wrappers` | Action repeat wrapper. |
| `NormalizeActions` | `torchwm.envs.wrappers` | Action normalization to [-1, 1]. |
| `ObsDict` | `torchwm.envs.wrappers` | Observation-to-dict conversion. |
| `OneHotAction` | `torchwm.envs.wrappers` | Discrete to one-hot action conversion. |
| `RewardObs` | `torchwm.envs.wrappers` | Reward observation injection. |
| `ResizeImage` | `torchwm.envs.wrappers` | Image resizing wrapper. |
| `RenderImage` | `torchwm.envs.wrappers` | Render-based image observation. |
| `SelectAction` | `torchwm.envs.wrappers` | Action selection wrapper. |

## Controllers and policies

| Name | Source | Description |
|---|---|---|
| `RSSMPolicy` | `torchwm.controller` | RSSM-based policy for Dreamer. |
| `RolloutGenerator` | `torchwm.controller` | Policy rollouts in the environment. |
| `IRISActor` | `torchwm.controller` | IRIS actor head. |
| `IRISCritic` | `torchwm.controller` | IRIS critic head. |
| `IRISPolicy` | `torchwm.controller` | IRIS combined actor-critic policy. |
| `CNNFeatureExtractor` | `torchwm.controller` | CNN feature extractor for policy inputs. |

## Export

| Name | Source | Description |
|---|---|---|
| `export_any(obj, path, format, ...)` | `torchwm.export` | Export a model or agent to ONNX/TorchScript/TensorRT. |
| `export_model(module, path, format, ...)` | `torchwm.export` | Export a raw nn.Module. |
| `ExportableAgentMixin` | `torchwm.export` | Mixin that adds `.export()` to custom agents. |

## Reward and value models

| Name | Source | Description |
|---|---|---|
| `RewardModel` | `torchwm.reward` | Base reward model. |
| `ValueModel` | `torchwm.reward` | Base value model. |
| `DreamerRewardModel` | `torchwm.reward` | Dreamer reward predictor. |
| `DreamerValueModel` | `torchwm.reward` | Dreamer value function. |

## Transformer blocks

| Name | Source | Description |
|---|---|---|
| `STTransformer` | `torchwm.blocks` | Spatiotemporal Transformer (Genie). |
| `MultiHeadSelfAttention` | `torchwm.blocks` | Multi-head self-attention. |
| `MultiHeadAttention` | `torchwm.blocks` | Multi-head cross-attention. |
| `AdaLNNormalization` | `torchwm.blocks` | Adaptive layer norm for diffusion. |
| `RMSNorm` | `torchwm.blocks` | Root mean square layer norm. |

## Plugin registry

| Name | Source | Description |
|---|---|---|
| `register_world_model(name, import_path, ...)` | `torchwm.registry` | Register a custom world model architecture. |
| `deregister_world_model(name)` | `torchwm.registry` | Remove a registered model. |
| `get_registered_model_spec(name)` | `torchwm.registry` | Look up a registered model spec. |
| `list_registered_models()` | `torchwm.registry` | List all externally registered model names. |
| `register_env_backend(name, factory_path, ...)` | `torchwm.registry` | Register a custom environment backend. |
| `deregister_env_backend(name)` | `torchwm.registry` | Remove a registered env backend. |
| `list_registered_env_backends()` | `torchwm.registry` | List all registered env backends. |

## Deprecation utilities

| Name | Source | Description |
|---|---|---|
| `deprecated(version, reason)` | `torchwm.utils.deprecation` | Decorator to mark functions/classes as deprecated. |
| `deprecated_class(version, alternative)` | `torchwm.utils.deprecation` | Shortcut for deprecating a class. |
| `deprecated_function(version, alternative)` | `torchwm.utils.deprecation` | Shortcut for deprecating a function. |

## General utilities

| Name | Source | Description |
|---|---|---|
| `Logger` | `torchwm.utils` | Logging utility. |
| `FreezeParameters` | `torchwm.utils` | Context manager to freeze model parameters. |
| `compute_return(rewards, values, gamma, lambda_)` | `torchwm.utils` | Compute GAE or λ-return. |
| `preprocess_obs(obs)` | `torchwm.utils` | Observation preprocessing (resize, normalize). |

## Version

| Name | Description |
|---|---|
| `__version__` | Package version string (semver). |
