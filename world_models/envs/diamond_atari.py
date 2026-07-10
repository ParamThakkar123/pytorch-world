from __future__ import annotations

import numpy as np
from world_models.envs._contract import finalize_step_info
from typing import Tuple, Dict, Optional, Any


class DiamondAtariWrapper:
    """
    Atari wrapper for DIAMOND following the paper specifications:
    - frameskip: number of frames to skip (default 4)
    - max_noop: maximum number of noop actions at reset (default 30)
    - terminate_on_life_loss: terminate episode when life is lost (default True)
    - reward_clip: clip rewards to [-1, 0, 1] (default True)
    - resize: resize observations to specified size (default 64x64)
    """

    def __init__(
        self,
        env: Any,
        frameskip: int = 4,
        max_noop: int = 30,
        terminate_on_life_loss: bool = True,
        reward_clip: bool = True,
        resize: Optional[Tuple[int, int]] = (64, 64),
        seed: int | None = None,
    ):
        from world_models.utils.gym_compat import spaces

        self.env = env
        self.action_space = env.action_space
        self.frameskip = frameskip
        self.max_noop = max_noop
        self.terminate_on_life_loss = terminate_on_life_loss
        self.reward_clip = reward_clip
        self.resize = resize
        self._rng = np.random.default_rng(seed)

        self.lives = 0
        self._last_lives = 0

        if resize is not None:
            self._height, self._width = resize
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8
            )

    def _apply_frameskip(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Apply frameskip by repeating the action.

        Returns (obs, total_reward, done, info) where `done` is a collapsed
        boolean indicating termination/truncation for older gym APIs.
        """
        total_reward = 0.0
        done = False
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        obs: Any = None

        for _ in range(self.frameskip):
            ret = self.env.step(action)
            # gymnasium returns (obs, reward, terminated, truncated, info)
            if isinstance(ret, tuple) and len(ret) == 5:
                obs, reward, terminated, truncated, info = ret
            else:
                # older gym: (obs, reward, done, info)
                obs, reward, single_done, info = ret
                truncated = (
                    bool(info.get("TimeLimit.truncated", False)) if info else False
                )
                terminated = bool(single_done and not truncated)

            total_reward += float(reward)

            if terminated or truncated:
                done = True
                break

            if self.terminate_on_life_loss:
                # ale attribute may or may not exist depending on backend; runtime
                # checks are used here. Type-checkers don't know about `ale`, so
                # use hasattr guards and ignore the attribute access for mypy.
                if hasattr(self.env, "ale") and hasattr(
                    getattr(self.env, "ale"), "lives"
                ):
                    try:
                        self.lives = self.env.ale.lives()
                    except Exception:
                        # some backends expose lives as attribute or method; ignore failures
                        pass
                    if self.lives < self._last_lives and self.lives > 0:
                        done = True
                        info["life_lost"] = True
                        break

        self._last_lives = self.lives

        if self.reward_clip:
            total_reward = float(np.clip(total_reward, -1, 1))

        assert obs is not None
        info = finalize_step_info(
            info,
            done=done,
            terminated=terminated,
            truncated=truncated,
        )
        return obs, total_reward, done, info

    def seed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def step(self, action: int) -> Any:
        """Step the environment.

        For backwards compatibility with older gym APIs this wrapper returns a
        4-tuple: (obs, reward, done, info). Internally it supports gymnasium's
        5-tuple and collapses (terminated, truncated) into a single `done` bool.
        """
        obs, reward, done, info = self._apply_frameskip(action)

        if self.resize is not None:
            obs = self._resize_obs(obs)

        # Return legacy 4-tuple (obs, reward, done, info)
        return obs, reward, bool(done), info

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        seed = kwargs.get("seed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self.env.reset(**kwargs)

        if self.resize is not None:
            obs = self._resize_obs(obs)

        if self.terminate_on_life_loss:
            if hasattr(self.env, "ale") and hasattr(self.env.ale, "lives"):
                self.lives = self.env.ale.lives()
            else:
                self.lives = 0
            self._last_lives = self.lives

        noops = int(self._rng.integers(1, self.max_noop + 1))
        for _ in range(noops):
            action = self.env.action_space.sample()
            if action == 0:
                # gymnasium env.step returns (obs, reward, terminated, truncated, info)
                step_ret = self.env.step(action)
                if len(step_ret) == 5:
                    obs_step, _, terminated, truncated, _ = step_ret
                    done = bool(terminated or truncated)
                else:
                    # fallback for older gym API
                    obs_step, _, done, _ = step_ret

                if self.resize is not None:
                    obs_step = self._resize_obs(obs_step)

                obs = obs_step

                if done:
                    obs, info = self.env.reset(**kwargs)
                    if self.resize is not None:
                        obs = self._resize_obs(obs)
                    break

        return obs, info

    def _resize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Resize observation to target size."""
        if obs.shape[:2] == (self._height, self._width):
            return obs

        import cv2

        obs = cv2.resize(obs, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return obs.astype(np.uint8)


def make_diamond_atari_env(
    game: str,
    frameskip: int = 4,
    max_noop: int = 30,
    terminate_on_life_loss: bool = True,
    reward_clip: bool = True,
    resize: Tuple[int, int] = (64, 64),
    seed: Optional[int] = None,
) -> DiamondAtariWrapper:
    """
    Create a DIAMOND-compatible Atari environment.

    Args:
        game: Atari game name (e.g., "Breakout-v5")
        frameskip: Number of frames to skip between actions
        max_noop: Maximum number of noop actions at reset
        terminate_on_life_loss: Whether to terminate on life loss
        reward_clip: Whether to clip rewards to [-1, 0, 1]
        resize: Target size for observations
        seed: Random seed

    Returns:
        DiamondAtariWrapper: Configured Atari environment
    """
    from world_models.utils.gym_compat import gym

    env = gym.make(
        game,
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
    )

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    env = DiamondAtariWrapper(
        env=env,
        frameskip=frameskip,
        max_noop=max_noop,
        terminate_on_life_loss=terminate_on_life_loss,
        reward_clip=reward_clip,
        resize=resize,
        seed=seed,
    )

    return env
