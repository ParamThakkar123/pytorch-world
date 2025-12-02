import torch
import numpy as np
import gymnasium as gym
from world_models.utils.utils import preprocess_img, to_tensor_obs
from world_models.envs.ale_atari_env import list_available_atari_envs

# Constants for action repeats (tuned for different environment types)
ACTION_REPEATS = {
    "cartpole": 8,
    "pendulum": 2,
    "mountaincar": 4,
    "acrobot": 4,
    "lunarlander": 4,
    "bipedal": 2,
    "humanoid": 2,
    "halfcheetah": 2,
    "ant": 2,
    "walker2d": 2,
    "hopper": 2,
    "reacher": 4,
    "pusher": 4,
    "striker": 4,
    "thrower": 4,
}

# List of supported environments
GYM_ENVS = [
    # Classic Control
    "CartPole-v1",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    "Acrobot-v1",
    # Box2D
    "LunarLander-v2",
    "LunarLanderContinuous-v2",
    "BipedalWalker-v3",
    "BipedalWalkerHardcore-v3",
    # MuJoCo
    "Humanoid-v4",
    "HalfCheetah-v4",
    "Ant-v4",
    "Walker2d-v4",
    "Hopper-v4",
    "Reacher-v4",
    "Pusher-v4",
    "Striker-v4",
    "Thrower-v4",
]

ATARI_ENVS = list_available_atari_envs()
print(f"Found {len(ATARI_ENVS)} available Atari environments")

ALL_ENVS = GYM_ENVS + ATARI_ENVS


def _get_action_repeat(env_id):
    """Get recommended action repeat for environment."""
    env_id_lower = env_id.lower()

    for key, repeat in ACTION_REPEATS.items():
        if key in env_id_lower:
            return repeat

    if any(atari in env_id for atari in ["ALE/", "Atari"]):
        return 4  # Standard for Atari
    elif any(
        classic in env_id_lower
        for classic in ["cartpole", "pendulum", "mountaincar", "acrobot"]
    ):
        return 4  # Classic control
    elif any(box2d in env_id_lower for box2d in ["lunar", "bipedal"]):
        return 4  # Box2D
    elif any(
        mujoco in env_id_lower
        for mujoco in ["humanoid", "cheetah", "ant", "walker", "hopper"]
    ):
        return 2  # MuJoCo
    else:
        return 2  # Default


def _images_to_observation(image, bit_depth):
    """Convert image to tensor observation using existing utils."""
    image_tensor = to_tensor_obs(image)

    # Preprocess using existing function
    preprocess_img(image_tensor, bit_depth)
    return image_tensor.unsqueeze(0)  # Add batch dimension


class UniversalGymEnv:
    def __init__(
        self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth
    ):
        self.symbolic = symbolic

        # Try to make environment with render mode
        try:
            self._env = gym.make(env, render_mode="rgb_array" if not symbolic else None)
        except Exception:
            # Fall back to creating env without render_mode for older gym versions or unexpected errors
            self._env = gym.make(env)

        # Set seed
        if hasattr(self._env, "seed"):
            self._env.seed(seed)
        else:
            # For newer gym versions
            self._env.reset(seed=seed)

        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat or _get_action_repeat(env)
        self.bit_depth = bit_depth
        self.env_id = env

        # Print action repeat info
        recommended = _get_action_repeat(env)
        if action_repeat and action_repeat != recommended:
            print(
                f"Using action repeat {action_repeat}; recommended action repeat for {env} is {recommended}"
            )

    def reset(self):
        self.t = 0  # Reset internal timer

        # Handle different gym return formats
        reset_result = self._env.reset()
        if isinstance(reset_result, tuple):
            state, info = reset_result
        else:
            state = reset_result

        if self.symbolic:
            # For state-based environments
            if isinstance(state, np.ndarray):
                return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
            else:
                return torch.tensor([state], dtype=torch.float32).unsqueeze(dim=0)
        else:
            if hasattr(self._env, "render"):
                try:
                    frame = self._env.render()
                    if frame is not None:
                        return _images_to_observation(frame, self.bit_depth)
                except Exception:
                    pass

            # if state is already an image
            if isinstance(state, np.ndarray) and len(state.shape) == 3:
                return _images_to_observation(state, self.bit_depth)
            else:
                # Create synthetic image from state vector
                return self._state_to_image(state)

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        # Handle different action space types
        if hasattr(self._env.action_space, "n"):
            # Discrete action space
            action = int(
                np.clip(
                    action.item() if hasattr(action, "item") else action[0],
                    0,
                    self._env.action_space.n - 1,
                )
            )
        else:
            # Continuous action space
            action = np.array(action).flatten()
            if (
                len(action) == 1
                and hasattr(self._env.action_space, "shape")
                and len(self._env.action_space.shape) > 0
            ):
                action = np.repeat(action, self._env.action_space.shape[0])

        reward = 0
        for k in range(self.action_repeat):
            step_result = self._env.step(action)

            # Handle different gym return formats
            if len(step_result) == 4:
                state, reward_k, done, info = step_result
                truncated = False
            else:  # len == 5 for newer gym
                state, reward_k, terminated, truncated, info = step_result
                done = terminated or truncated

            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t >= self.max_episode_length
            if done:
                break

        if self.symbolic:
            # For state-based environments
            if isinstance(state, np.ndarray):
                observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
            else:
                observation = torch.tensor([state], dtype=torch.float32).unsqueeze(
                    dim=0
                )
        else:
            if hasattr(self._env, "render"):
                try:
                    frame = self._env.render()
                    if frame is not None:
                        observation = _images_to_observation(frame, self.bit_depth)
                    else:
                        observation = self._state_to_image(state)
                except Exception:
                    observation = self._state_to_image(state)
            else:
                observation = self._state_to_image(state)
                observation = self._state_to_image(state)

        return observation, reward, done

    def _state_to_image(self, state):
        """Convert state vector to synthetic 64x64 RGB image."""
        if isinstance(state, (int, float)):
            state = np.array([state])
        elif not isinstance(state, np.ndarray):
            state = np.array(state)

        # Normalize state values
        if state.size > 0:
            state_norm = (state - state.min()) / (state.max() - state.min() + 1e-8)
        else:
            state_norm = np.zeros(1)

        # Create 64x64 image from state
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        # Fill image with patterns based on state values
        for i, val in enumerate(state_norm[: min(len(state_norm), 8)]):
            color_val = int(255 * val)
            x_start = (i % 4) * 16
            y_start = (i // 4) * 32
            image[y_start : y_start + 32, x_start : x_start + 16, i % 3] = color_val

        return _images_to_observation(image, self.bit_depth)

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def observation_size(self):
        if self.symbolic:
            if hasattr(self._env.observation_space, "shape"):
                return (
                    self._env.observation_space.shape[0]
                    if len(self._env.observation_space.shape) > 0
                    else 1
                )
            else:
                return 1
        else:
            return (3, 64, 64)

    @property
    def action_size(self):
        if (
            hasattr(self._env.action_space, "shape")
            and len(self._env.action_space.shape) > 0
        ):
            return self._env.action_space.shape[0]
        elif hasattr(self._env.action_space, "n"):
            return 1  # Discrete action space
        else:
            return 1

    def sample_random_action(self):
        action = self._env.action_space.sample()
        if isinstance(action, np.ndarray):
            return torch.from_numpy(action.astype(np.float32))
        else:
            return torch.tensor([float(action)])


def Env(env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
    """Universal environment factory."""
    return UniversalGymEnv(
        env, symbolic, seed, max_episode_length, action_repeat, bit_depth
    )


def is_supported_env(env_id):
    """Check if environment is supported."""
    return env_id in ALL_ENVS


def list_supported_envs():
    """List all supported environments."""
    return {"gym": GYM_ENVS, "atari": ATARI_ENVS, "all": ALL_ENVS}


# Wrapper for batching environments together
class EnvBatcher:
    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n = n
        self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
        self.dones = [True] * n

    def reset(self):
        observations = [env.reset() for env in self.envs]
        self.dones = [False] * self.n
        return torch.cat(observations)

    def step(self, actions):
        done_mask = torch.nonzero(torch.tensor(self.dones), as_tuple=False)[:, 0]
        observations, rewards, dones = zip(
            *[env.step(action) for env, action in zip(self.envs, actions)]
        )
        dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]
        self.dones = dones
        observations, rewards, dones = (
            torch.cat(observations),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.uint8),
        )
        observations[done_mask] = 0
        rewards[done_mask] = 0
        return observations, rewards, dones

    def close(self):
        [env.close() for env in self.envs]
