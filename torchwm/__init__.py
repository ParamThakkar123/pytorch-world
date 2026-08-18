"""TorchWM public API.

This package keeps imports lightweight while still exposing a friendly top-level
surface.  Common workflows can use the small factory helpers::

    import torchwm

    cfg = torchwm.create_config("dreamer", env="walker-walk")
    agent = torchwm.create_model("dreamer", cfg)
    env = torchwm.make_env("CartPole-v1", backend="gym")

Lower-level research components remain available as lazy top-level exports, for
example ``from torchwm import DreamerAgent, ConvEncoder, ReplayBuffer``, and
every implementation submodule is reachable directly::

    from torchwm.models import Dreamer
    from torchwm.training.eval_jepa import jepa_linear_probe
    import torchwm.envs
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from torchwm._version import __version__  # noqa: F401


try:
    from torchwm.export import install_export_method as _install_export_method

    _install_export_method()
except ModuleNotFoundError as exc:  # pragma: no cover - torch-free metadata imports
    if exc.name != "torch":
        raise

_API_EXPORTS = {
    "EnvBackendSpec": "torchwm.api",
    "ModelSpec": "torchwm.api",
    "MODEL_SPECS": "torchwm.api",
    "ENV_BACKEND_SPECS": "torchwm.api",
    "create_config": "torchwm.api",
    "create_model": "torchwm.api",
    "get_env_backend_spec": "torchwm.api",
    "get_model_spec": "torchwm.api",
    "list_env_backends": "torchwm.api",
    "list_envs": "torchwm.api",
    "list_models": "torchwm.api",
    "make_env": "torchwm.api",
    "export_any": "torchwm.export",
    "export_model": "torchwm.export",
    "ExportableAgentMixin": "torchwm.export",
}

_LAZY_EXPORTS: dict[str, str] = {
    # Agents and high-level models.
    "Dreamer": "torchwm.models",
    "DreamerV1": "torchwm.models",
    "DreamerV2": "torchwm.models",
    "DreamerV3": "torchwm.models",
    "DreamerAgent": "torchwm.models",
    "Planet": "torchwm.models",
    "JEPAAgent": "torchwm.models",
    "IRISAgent": "torchwm.models",
    "IRISTransformer": "torchwm.models",
    "IRISWorldModel": "torchwm.models",
    "LPIPSPerceptualLoss": "torchwm.vision",
    "build_perceptual_loss": "torchwm.vision",
    "compute_lambda_return": "torchwm.models",
    "VisionTransformer": "torchwm.models",
    "ModularRSSM": "torchwm.models",
    "create_modular_rssm": "torchwm.models",
    "Genie": "torchwm.models",
    "LatentActionModel": "torchwm.models",
    "DynamicsModel": "torchwm.models",
    "create_genie": "torchwm.models",
    "create_genie_small": "torchwm.models",
    "create_genie_large": "torchwm.models",
    "create_latent_action_model": "torchwm.models",
    "create_dynamics_model": "torchwm.models",
    # State-space models.
    "RSSM": "torchwm.models",
    "RecurrentStateSpaceModel": "torchwm.models",
    # ``dreamer_rssm`` defines the class as ``RSSM``; ``torchwm.models``
    # is what exposes it under the ``DreamerRSSM`` alias.
    "DreamerRSSM": "torchwm.models",
    # Vision components.
    "ConvEncoder": "torchwm.vision",
    "CNNEncoder": "torchwm.vision",
    "ConvDecoder": "torchwm.vision",
    "CNNDecoder": "torchwm.vision",
    "DenseDecoder": "torchwm.vision",
    "ActionDecoder": "torchwm.vision",
    "TanhBijector": "torchwm.vision",
    "SampleDist": "torchwm.vision",
    "IRISEncoder": "torchwm.vision",
    "IRISDecoder": "torchwm.vision",
    "VideoTokenizer": "torchwm.vision",
    "create_video_tokenizer": "torchwm.vision",
    "VectorQuantizer": "torchwm.vision",
    "VectorQuantizerEMA": "torchwm.vision",
    # Memory.
    "ReplayBuffer": "torchwm.memory",
    "Memory": "torchwm.memory",
    "Episode": "torchwm.memory",
    "IRISReplayBuffer": "torchwm.memory",
    "IRISOnPolicyBuffer": "torchwm.memory",
    # Diffusion models.
    # ``DiT`` and ``DDPM`` name both a class and a sibling module. Importing the
    # module binds it over the class on the package, so which one
    # ``torchwm.models.diffusion.DiT`` returns depends on import order -
    # point at the defining modules to make it deterministic.
    "DiT": "torchwm.models.diffusion.DiT",
    "create_dit": "torchwm.models.diffusion",
    "PatchEmbed": "torchwm.models.diffusion",
    "PatchUnEmbed": "torchwm.models.diffusion",
    "DDPM": "torchwm.models.diffusion.DDPM",
    "ActorCriticNetwork": "torchwm.models.diffusion",
    "RewardTerminationModel": "torchwm.models.diffusion",
    "sinusoidal_time_embedding": "torchwm.models.diffusion",
    # Transformer blocks and layers.
    "STTransformer": "torchwm.blocks",
    "MultiHeadSelfAttention": "torchwm.blocks",
    "MultiHeadAttention": "torchwm.blocks",
    "AdaLNNormalization": "torchwm.blocks",
    "RMSNorm": "torchwm.blocks",
    # Controllers and policies.
    "RSSMPolicy": "torchwm.controller",
    "RolloutGenerator": "torchwm.controller",
    "IRISActor": "torchwm.controller",
    "IRISCritic": "torchwm.controller",
    "IRISPolicy": "torchwm.controller",
    "CNNFeatureExtractor": "torchwm.controller",
    # Configs.
    "DreamerConfig": "torchwm.configs",
    "JEPAConfig": "torchwm.configs",
    "DiTConfig": "torchwm.configs",
    "dit_preset_config": "torchwm.configs",
    "list_dit_presets": "torchwm.configs",
    "get_dit_config": "torchwm.configs",
    "DiamondConfig": "torchwm.configs",
    "IRISConfig": "torchwm.configs",
    "GenieConfig": "torchwm.configs",
    "GenieSmallConfig": "torchwm.configs",
    "STTransformerConfig": "torchwm.configs",
    "VideoTokenizerConfig": "torchwm.configs",
    "LatentActionModelConfig": "torchwm.configs",
    "DynamicsModelConfig": "torchwm.configs",
    "ATARI_100K_GAMES": "torchwm.configs",
    "HUMAN_SCORES": "torchwm.configs",
    "RANDOM_SCORES": "torchwm.configs",
    # Environments and wrappers.
    "BSuiteImageEnv": "torchwm.envs",
    "make_bsuite_env": "torchwm.envs",
    "list_available_bsuite_ids": "torchwm.envs",
    "make_atari_env": "torchwm.envs",
    "list_available_atari_envs": "torchwm.envs",
    "make_atari_vector_env": "torchwm.envs",
    "make_diamond_atari_env": "torchwm.envs.diamond_atari",
    "MuJoCoImageEnv": "torchwm.envs",
    "make_mujoco_env": "torchwm.envs",
    "make_mujoco_env_from_config": "torchwm.envs",
    "list_gymnasium_robotics_envs": "torchwm.envs",
    "make_robotics_env": "torchwm.envs",
    "register_gymnasium_robotics_envs": "torchwm.envs",
    "GymImageEnv": "torchwm.envs",
    "make_gym_env": "torchwm.envs",
    "WorldModelEnv": "torchwm.envs",
    "make_world_model_env": "torchwm.envs",
    "BraxImageEnv": "torchwm.envs",
    "make_brax_env": "torchwm.envs",
    "DeepMindControlEnv": "torchwm.envs",
    "DMLabEnv": "torchwm.envs",
    "make_dmlab_env": "torchwm.envs",
    "DMLAB_LEVELS": "torchwm.envs",
    "UnityMLAgentsEnv": "torchwm.envs",
    "make_unity_mlagents_env": "torchwm.envs",
    "MujocoEnv": "torchwm.envs",
    "TimeLimit": "torchwm.envs",
    "ActionRepeat": "torchwm.envs",
    "NormalizeActions": "torchwm.envs",
    "ObsDict": "torchwm.envs",
    "OneHotAction": "torchwm.envs",
    "RewardObs": "torchwm.envs",
    "ResizeImage": "torchwm.envs",
    "RenderImage": "torchwm.envs",
    "SelectAction": "torchwm.envs",
    # I-JEPA evaluation (paper Appendix A.2).
    "jepa_linear_probe": "torchwm.training.eval_jepa",
    "load_jepa_encoder": "torchwm.training.eval_jepa",
    # Reward/value models.
    "RewardModel": "torchwm.reward",
    "ValueModel": "torchwm.reward",
    "DreamerRewardModel": "torchwm.reward",
    "DreamerValueModel": "torchwm.reward",
    # Registry / plugin system.
    "register_world_model": "torchwm.registry",
    "deregister_world_model": "torchwm.registry",
    "get_registered_model_spec": "torchwm.registry",
    "list_registered_models": "torchwm.registry",
    "register_env_backend": "torchwm.registry",
    "deregister_env_backend": "torchwm.registry",
    "list_registered_env_backends": "torchwm.registry",
    # Deprecation helpers.
    "deprecated": "torchwm.utils.deprecation",
    "deprecated_class": "torchwm.utils.deprecation",
    "deprecated_function": "torchwm.utils.deprecation",
    # Utilities.
    "Logger": "torchwm.utils",
    "FreezeParameters": "torchwm.utils",
    "compute_return": "torchwm.utils",
    "preprocess_obs": "torchwm.utils",
    # Performance measurement and tuning.
    "ThroughputMeter": "torchwm.utils.throughput",
    "measure_steps": "torchwm.utils.throughput",
    "tensor_nbytes": "torchwm.utils.throughput",
    "enable_performance_defaults": "torchwm.utils.memory_utils",
    "maybe_compile": "torchwm.utils.memory_utils",
    "to_channels_last": "torchwm.utils.memory_utils",
}

_EXPORTS = {**_API_EXPORTS, **_LAZY_EXPORTS}


# Submodules that are part of the public surface and so must resolve as
# attributes of the package, not only via ``import torchwm.<name>``.
_SUBMODULE_EXPORTS = ("api",)


def __getattr__(name: str) -> Any:
    """Lazily import public symbols on first access."""

    if name in _SUBMODULE_EXPORTS:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
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


__all__ = ["__version__", *_SUBMODULE_EXPORTS, *_EXPORTS]
