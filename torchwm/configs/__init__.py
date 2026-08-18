"""Lazy config exports.

Configuration modules can have optional training dependencies, so the package
initializer avoids importing every config eagerly.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DreamerConfig": "torchwm.configs.dreamer_config",
    "JEPAConfig": "torchwm.configs.jepa_config",
    "DiTConfig": "torchwm.configs.dit_config",
    "get_dit_config": "torchwm.configs.dit_config",
    "dit_preset_config": "torchwm.configs.dit_config",
    "list_dit_presets": "torchwm.configs.dit_config",
    "DIT_PRESETS": "torchwm.configs.dit_config",
    "DiamondConfig": "torchwm.configs.diamond_config",
    "IRISConfig": "torchwm.configs.iris_config",
    "ATARI_100K_GAMES": "torchwm.configs.diamond_config",
    "HUMAN_SCORES": "torchwm.configs.diamond_config",
    "RANDOM_SCORES": "torchwm.configs.diamond_config",
    "GenieConfig": "torchwm.configs.genie_config",
    "GenieSmallConfig": "torchwm.configs.genie_config",
    "STTransformerConfig": "torchwm.configs.genie_config",
    "VideoTokenizerConfig": "torchwm.configs.genie_config",
    "LatentActionModelConfig": "torchwm.configs.genie_config",
    "DynamicsModelConfig": "torchwm.configs.genie_config",
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
