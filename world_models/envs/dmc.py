from typing import Any

from world_models.envs._contract import finalize_step_info
from world_models.utils.gym_compat import gym
import numpy as np


class DeepMindControlEnv:
    """Gym-style adapter for DeepMind Control Suite tasks.

    The wrapper exposes DMC observations and actions through Gym spaces and
    adds a rendered RGB image to each observation dict so image-based world
    model pipelines can train consistently across backends.

    Features:
        - Parses domain-task names (e.g., "cheetah-run" -> domain="cheetah", task="run")
        - Automatically handles special cases like "cup" -> "ball_in_cup"
        - Renders RGB images at configurable resolution
        - Returns observations as dict with both state vectors and images

    Args:
        name (str): Environment name in format "domain-task" (e.g., "cheetah-run").
        seed (int): Random seed for environment initialization.
        size (tuple): Target image size as (height, width) (default: (64, 64)).
        camera (int, optional): Camera ID for rendering. Defaults to 0 for most
            domains, 2 for quadruped.

    Attributes:
        observation_space (gym.spaces.Dict): Dict space with state keys and "image".
        action_space (gym.spaces.Box): Continuous action space from DMC spec.

    Example:
        >>> env = DeepMindControlEnv("cheetah-run", seed=0, size=(64, 64))
        >>> obs = env.reset()
        >>> print(obs.keys())  # dict_keys(['position', 'velocity', 'image'])
    """

    def __init__(
        self,
        name: str,
        seed: int,
        size: tuple[int, int] = (64, 64),
        camera: int | None = None,
    ) -> None:
        domain, task = name.split("-", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if isinstance(domain, str):
            from dm_control import suite

            self._env = suite.load(domain, task, task_kwargs={"random": seed})
        else:
            assert task is None
            self._env = domain()
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self) -> gym.spaces.Dict:
        spaces: dict[str, gym.spaces.Space[Any]] = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, (3,) + self._size, dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self) -> gym.spaces.Box:
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs["image"] = self.render().transpose(2, 0, 1).copy()
        reward = time_step.reward or 0
        done = time_step.last()
        vector_observation = np.concatenate(
            [
                np.asarray(value, dtype=np.float32).reshape(-1)
                for value in time_step.observation.values()
            ],
            axis=0,
        )
        info = finalize_step_info(
            {
                "discount": np.array(time_step.discount, np.float32),
                "vector_observation": vector_observation,
            },
            done=done,
            terminated=done,
            truncated=False,
        )
        return obs, float(reward), done, info

    def reset(self) -> dict:
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs["image"] = self.render().transpose(2, 0, 1).copy()
        return obs

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)

