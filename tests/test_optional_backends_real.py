import os
import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _assert_image_observation(
    obs: dict[str, np.ndarray], shape: tuple[int, int, int]
) -> None:
    assert set(obs) >= {"image"}
    assert obs["image"].shape == shape
    assert obs["image"].dtype == np.uint8
    assert int(obs["image"].min()) >= 0
    assert int(obs["image"].max()) <= 255


def test_real_brax_image_env_smoke():
    pytest.importorskip("brax")
    from torchwm.envs.brax_env import BraxImageEnv

    env = BraxImageEnv(
        "inverted_pendulum",
        seed=0,
        size=(16, 16),
        jit=False,
        include_state=True,
    )
    try:
        obs = env.reset()
        _assert_image_observation(obs, (3, 16, 16))
        assert obs["state"].dtype == np.float32
        assert env.observation_space.contains(obs)

        next_obs, reward, done, info = env.step(env.action_space.sample())
        _assert_image_observation(next_obs, (3, 16, 16))
        assert next_obs["state"].dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "discount" in info
        assert "action" in info
        assert "vector_observation" in info

        frame = env.render()
        assert frame.ndim == 3
        assert frame.shape[-1] == 3
    finally:
        env.close()


def test_real_robotics_env_smoke():
    pytest.importorskip("gymnasium_robotics")
    from torchwm.envs.robotics_env import (
        list_gymnasium_robotics_envs,
        make_robotics_env,
    )

    env_ids = list_gymnasium_robotics_envs()
    assert env_ids, "Gymnasium Robotics did not register any environments."

    preferred_ids = (
        "FetchReach-v4",
        "FetchReachDense-v4",
        "AdroitHandDoor-v1",
    )
    env_id = next(
        (candidate for candidate in preferred_ids if candidate in env_ids), env_ids[0]
    )
    env = make_robotics_env(env_id, seed=0, size=(16, 16), render_mode="rgb_array")
    try:
        obs = env.reset()
        _assert_image_observation(obs, (3, 16, 16))
        assert env.observation_space.contains(obs)

        next_obs, reward, done, info = env.step(env.action_space.sample())
        _assert_image_observation(next_obs, (3, 16, 16))
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "discount" in info
        assert "action" in info
        assert "vector_observation" in info

        frame = env.render()
        assert frame.ndim == 3
        assert frame.shape[-1] == 3
    finally:
        env.close()


def test_real_mlagents_sdk_importable():
    sdk = pytest.importorskip("mlagents_envs")
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel import (
        EngineConfigurationChannel,
    )

    assert UnityEnvironment is not None
    assert EngineConfigurationChannel is not None
    assert getattr(sdk, "__version__", "")


@pytest.mark.skipif(
    sys.version_info >= (3, 11),
    reason="Upstream Procgen wheels currently support Python 3.10 and below.",
)
def test_real_procgen_backend_if_supported_python():
    pytest.importorskip("procgen")
    from torchwm.envs.procgen_env import ProcgenImageEnv

    env = ProcgenImageEnv("coinrun", seed=0, size=(16, 16), num_levels=1)
    try:
        obs = env.reset()
        _assert_image_observation(obs, (3, 16, 16))
        next_obs, reward, done, info = env.step(env.action_space.sample())
        _assert_image_observation(next_obs, (3, 16, 16))
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "discount" in info
    finally:
        env.close()


@pytest.mark.skipif(
    not os.getenv("TORCHWM_UNITY_BINARY") or not os.getenv("TORCHWM_UNITY_BEHAVIOR"),
    reason="Set TORCHWM_UNITY_BINARY and TORCHWM_UNITY_BEHAVIOR to run against a real Unity build.",
)
def test_real_unity_backend_if_binary_provided():
    from torchwm.envs.unity_env import UnityMLAgentsEnv

    env = UnityMLAgentsEnv(
        file_name=os.environ["TORCHWM_UNITY_BINARY"],
        behavior_name=os.environ["TORCHWM_UNITY_BEHAVIOR"],
        seed=0,
        size=(32, 32),
        no_graphics=True,
        include_state=True,
    )
    try:
        obs = env.reset()
        _assert_image_observation(obs, (3, 32, 32))
        assert env.observation_space.contains(obs)

        next_obs, reward, done, info = env.step(env.action_space.sample())
        _assert_image_observation(next_obs, (3, 32, 32))
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "discount" in info
        assert "action" in info
    finally:
        env.close()
