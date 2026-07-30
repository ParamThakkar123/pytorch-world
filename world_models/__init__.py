"""TorchWM public API.

This package keeps imports lightweight while still exposing a friendly top-level
surface.  Common workflows can use the small factory helpers::

    import torchwm

    cfg = torchwm.create_config("dreamer", env="walker-walk")
    agent = torchwm.create_model("dreamer", cfg)
    env = torchwm.make_env("CartPole-v1", backend="gym")

Lower-level research components remain available as lazy top-level exports, for
example ``from torchwm import DreamerAgent, ConvEncoder, ReplayBuffer``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from world_models._version import __version__  # noqa: F401


try:
    from world_models.export import install_export_method as _install_export_method

    _install_export_method()
except ModuleNotFoundError as exc:  # pragma: no cover - torch-free metadata imports
    if exc.name != "torch":
        raise

_API_EXPORTS = {
    "EnvBackendSpec": "world_models.api",
    "ModelSpec": "world_models.api",
    "MODEL_SPECS": "world_models.api",
    "ENV_BACKEND_SPECS": "world_models.api",
    "create_config": "world_models.api",
    "create_model": "world_models.api",
    "get_env_backend_spec": "world_models.api",
    "get_model_spec": "world_models.api",
    "list_env_backends": "world_models.api",
    "list_envs": "world_models.api",
    "list_models": "world_models.api",
    "make_env": "world_models.api",
    "export_any": "world_models.export",
    "export_model": "world_models.export",
    "ExportableAgentMixin": "world_models.export",
}

_LAZY_EXPORTS: dict[str, str] = {
    # Agents and high-level models.
    "Dreamer": "world_models.models",
    "DreamerV1": "world_models.models",
    "DreamerV2": "world_models.models",
    "DreamerV3": "world_models.models",
    "DreamerV3Agent": "world_models.models",
    "DreamerAgent": "world_models.models",
    "Planet": "world_models.models",
    "JEPAAgent": "world_models.models",
    "IRISAgent": "world_models.models",
    "compute_lambda_return": "world_models.models",
    "VisionTransformer": "world_models.models",
    "ModularRSSM": "world_models.models",
    "create_modular_rssm": "world_models.models",
    "Genie": "world_models.models",
    "LatentActionModel": "world_models.models",
    "DynamicsModel": "world_models.models",
    "create_genie": "world_models.models",
    "create_genie_small": "world_models.models",
    "create_genie_large": "world_models.models",
    "create_latent_action_model": "world_models.models",
    "create_dynamics_model": "world_models.models",
    # State-space models.
    "RSSM": "world_models.models",
    "RecurrentStateSpaceModel": "world_models.models",
    "DreamerRSSM": "world_models.models.dreamer_rssm",
    "CategoricalRSSM": "world_models.models",
    # Vision components.
    "DreamerV3Encoder": "world_models.vision",
    "DreamerV3Decoder": "world_models.vision",
    "DreamerV3Head": "world_models.vision",
    "DreamerV3Actor": "world_models.vision",
    "ConvEncoder": "world_models.vision",
    "CNNEncoder": "world_models.vision",
    "ConvDecoder": "world_models.vision",
    "CNNDecoder": "world_models.vision",
    "DenseDecoder": "world_models.vision",
    "ActionDecoder": "world_models.vision",
    "TanhBijector": "world_models.vision",
    "SampleDist": "world_models.vision",
    "IRISEncoder": "world_models.vision",
    "IRISDecoder": "world_models.vision",
    "VideoTokenizer": "world_models.vision",
    "create_video_tokenizer": "world_models.vision",
    "VectorQuantizer": "world_models.vision",
    "VectorQuantizerEMA": "world_models.vision",
    # Memory.
    "ReplayBuffer": "world_models.memory",
    "DreamerV3ReplayBuffer": "world_models.memory",
    "Memory": "world_models.memory",
    "Episode": "world_models.memory",
    "IRISReplayBuffer": "world_models.memory",
    "IRISOnPolicyBuffer": "world_models.memory",
    # Diffusion models.
    "DiT": "world_models.models.diffusion",
    "create_dit": "world_models.models.diffusion",
    "PatchEmbed": "world_models.models.diffusion",
    "PatchUnEmbed": "world_models.models.diffusion",
    "DDPM": "world_models.models.diffusion",
    "ActorCriticNetwork": "world_models.models.diffusion",
    "RewardTerminationModel": "world_models.models.diffusion",
    "sinusoidal_time_embedding": "world_models.models.diffusion",
    # Transformer blocks and layers.
    "STTransformer": "world_models.blocks",
    "MultiHeadSelfAttention": "world_models.blocks",
    "MultiHeadAttention": "world_models.blocks",
    "AdaLNNormalization": "world_models.blocks",
    "RMSNorm": "world_models.blocks",
    # Controllers and policies.
    "RSSMPolicy": "world_models.controller",
    "RolloutGenerator": "world_models.controller",
    "IRISActor": "world_models.controller",
    "IRISCritic": "world_models.controller",
    "IRISPolicy": "world_models.controller",
    "CNNFeatureExtractor": "world_models.controller",
    # Configs.
    "DreamerConfig": "world_models.configs",
    "DreamerV3Config": "world_models.configs",
    "JEPAConfig": "world_models.configs",
    "DiTConfig": "world_models.configs",
    "get_dit_config": "world_models.configs",
    "DiamondConfig": "world_models.configs",
    "IRISConfig": "world_models.configs",
    "GenieConfig": "world_models.configs",
    "GenieSmallConfig": "world_models.configs",
    "STTransformerConfig": "world_models.configs",
    "VideoTokenizerConfig": "world_models.configs",
    "LatentActionModelConfig": "world_models.configs",
    "DynamicsModelConfig": "world_models.configs",
    "ATARI_100K_GAMES": "world_models.configs",
    "HUMAN_SCORES": "world_models.configs",
    "RANDOM_SCORES": "world_models.configs",
    # Environments and wrappers.
    "BSuiteImageEnv": "world_models.envs",
    "make_bsuite_env": "world_models.envs",
    "list_available_bsuite_ids": "world_models.envs",
    "make_atari_env": "world_models.envs",
    "list_available_atari_envs": "world_models.envs",
    "make_atari_vector_env": "world_models.envs",
    "make_diamond_atari_env": "world_models.envs.diamond_atari",
    "MuJoCoImageEnv": "world_models.envs",
    "make_mujoco_env": "world_models.envs",
    "make_mujoco_env_from_config": "world_models.envs",
    "list_gymnasium_robotics_envs": "world_models.envs",
    "make_robotics_env": "world_models.envs",
    "register_gymnasium_robotics_envs": "world_models.envs",
    "GymImageEnv": "world_models.envs",
    "make_gym_env": "world_models.envs",
    "WorldModelEnv": "world_models.envs",
    "make_world_model_env": "world_models.envs",
    "BraxImageEnv": "world_models.envs",
    "make_brax_env": "world_models.envs",
    "DeepMindControlEnv": "world_models.envs",
    "DMLabEnv": "world_models.envs",
    "make_dmlab_env": "world_models.envs",
    "DMLAB_LEVELS": "world_models.envs",
    "UnityMLAgentsEnv": "world_models.envs",
    "make_unity_mlagents_env": "world_models.envs",
    "MujocoEnv": "world_models.envs",
    "TimeLimit": "world_models.envs",
    "ActionRepeat": "world_models.envs",
    "NormalizeActions": "world_models.envs",
    "ObsDict": "world_models.envs",
    "OneHotAction": "world_models.envs",
    "RewardObs": "world_models.envs",
    "ResizeImage": "world_models.envs",
    "RenderImage": "world_models.envs",
    "SelectAction": "world_models.envs",
    # Inference operators.
    "OperatorABC": "world_models.inference.operators",
    "TensorSpec": "world_models.inference.operators",
    "DreamerOperator": "world_models.inference.operators",
    "JEPAOperator": "world_models.inference.operators",
    "IrisOperator": "world_models.inference.operators",
    "PlaNetOperator": "world_models.inference.operators",
    "get_operator": "world_models.inference.operators",
    # Reward/value models.
    "RewardModel": "world_models.reward",
    "ValueModel": "world_models.reward",
    "DreamerRewardModel": "world_models.reward",
    "DreamerValueModel": "world_models.reward",
    # Registry / plugin system.
    "register_world_model": "world_models.registry",
    "deregister_world_model": "world_models.registry",
    "get_registered_model_spec": "world_models.registry",
    "list_registered_models": "world_models.registry",
    "register_env_backend": "world_models.registry",
    "deregister_env_backend": "world_models.registry",
    "list_registered_env_backends": "world_models.registry",
    # Deprecation helpers.
    "deprecated": "world_models.utils.deprecation",
    "deprecated_class": "world_models.utils.deprecation",
    "deprecated_function": "world_models.utils.deprecation",
    # Utilities.
    "Logger": "world_models.utils",
    "FreezeParameters": "world_models.utils",
    "compute_return": "world_models.utils",
    "preprocess_obs": "world_models.utils",
}

_EXPORTS = {**_API_EXPORTS, **_LAZY_EXPORTS}


def __getattr__(name: str) -> Any:
    """Lazily import public symbols on first access."""

    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = ["__version__", *_EXPORTS]
