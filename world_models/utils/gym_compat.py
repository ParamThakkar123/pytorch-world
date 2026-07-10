from __future__ import annotations

from typing import Any


def import_gymnasium() -> Any | None:
    try:
        import gymnasium as gymnasium
    except ModuleNotFoundError:
        return None
    return gymnasium


def import_gym() -> Any:
    gymnasium = import_gymnasium()
    if gymnasium is not None:
        return gymnasium

    try:
        import gym as legacy_gym
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Neither 'gymnasium' nor legacy 'gym' is installed. "
            "Install `torchwm[gym]` to enable gym-compatible environments."
        ) from exc
    return legacy_gym


gym = import_gym()
spaces = gym.spaces


__all__ = ["gym", "spaces", "import_gym", "import_gymnasium"]
