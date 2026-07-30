"""Lazy config exports.

Configuration modules can have optional training dependencies, so the package
initializer avoids importing every config eagerly.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DreamerConfig": "world_models.configs.dreamer_config",
    "DreamerV3Config": "world_models.configs.dreamer_v3_config",
    "MODEL_SIZES": "world_models.configs.dreamer_v3_config",
    "JEPAConfig": "world_models.configs.jepa_config",
    "DiTConfig": "world_models.configs.dit_config",
    "get_dit_config": "world_models.configs.dit_config",
    "DiamondConfig": "world_models.configs.diamond_config",
    "IRISConfig": "world_models.configs.iris_config",
    "ATARI_100K_GAMES": "world_models.configs.diamond_config",
    "HUMAN_SCORES": "world_models.configs.diamond_config",
    "RANDOM_SCORES": "world_models.configs.diamond_config",
    "GenieConfig": "world_models.configs.genie_config",
    "GenieSmallConfig": "world_models.configs.genie_config",
    "STTransformerConfig": "world_models.configs.genie_config",
    "VideoTokenizerConfig": "world_models.configs.genie_config",
    "LatentActionModelConfig": "world_models.configs.genie_config",
    "DynamicsModelConfig": "world_models.configs.genie_config",
}


def __getattr__(name: str) -> Any:
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


__all__ = list(_EXPORTS)
