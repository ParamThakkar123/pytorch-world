import pytest
import numpy as np


gym = pytest.importorskip("gym")

from world_models.envs.dmc import DeepMindControlEnv
from world_models.envs.gym_env import GymImageEnv
from world_models.envs.mujoco_env import MuJoCoImageEnv
from world_models.envs.wrappers import FrameStack, NormalizeActions, TimeLimit


class _ContractFiveTupleEnv:
    def __init__(self, *, terminated=False, truncated=False):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self._terminated = terminated
        self._truncated = truncated

    def reset(self, seed=None):
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), {"seed": seed}

    def step(self, action):
        obs = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        return obs, np.float32(1.25), self._terminated, self._truncated, {}

    def render(self, *args, **kwargs):
        return np.zeros((8, 8, 3), dtype=np.uint8)


class _ContractDictObsEnv:
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(8, 8, 3),
                    dtype=np.uint8,
                ),
                "state": gym.spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(3,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, seed=None):
        return {
            "image": np.full((8, 8, 3), 32, dtype=np.uint8),
            "state": np.array([0.5, -0.5, 1.5], dtype=np.float32),
        }, {"seed": seed}

    def step(self, action):
        return (
            {
                "image": np.full((8, 8, 3), 96, dtype=np.uint8),
                "state": np.array([1.0, 0.0, -1.0], dtype=np.float32),
            },
            0.75,
            False,
            False,
            {},
        )

    def render(self, *args, **kwargs):
        return np.full((8, 8, 3), 96, dtype=np.uint8)


class _TimeLimitBaseEnv:
    def reset(self):
        return {"image": np.zeros((3, 4, 4), dtype=np.uint8)}

    def step(self, action):
        obs = {"image": np.ones((3, 4, 4), dtype=np.uint8)}
        return obs, 0.5, False, {"discount": np.array(1.0, dtype=np.float32)}


class _NormalizeActionBaseEnv:
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=np.array([-2.0, -4.0], dtype=np.float32),
            high=np.array([2.0, 4.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, 2, 2),
                    dtype=np.uint8,
                )
            }
        )
        self.last_action = None

    def reset(self):
        return {"image": np.zeros((3, 2, 2), dtype=np.uint8)}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        return (
            {"image": np.ones((3, 2, 2), dtype=np.uint8)},
            0.25,
            False,
            {"action": self.last_action.copy()},
        )


class _FrameStackBaseEnv:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, 2, 2),
                    dtype=np.uint8,
                ),
                "state": gym.spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )
        self._value = 0

    def reset(self):
        self._value = 1
        return {
            "image": np.full((3, 2, 2), self._value, dtype=np.uint8),
            "state": np.array([0.1], dtype=np.float32),
        }

    def step(self, action):
        self._value += 1
        return (
            {
                "image": np.full((3, 2, 2), self._value, dtype=np.uint8),
                "state": np.array([float(self._value)], dtype=np.float32),
            },
            1.0,
            False,
            {},
        )


class _FakePhysics:
    def render(self, height, width, camera_id=0):
        return np.zeros((height, width, 3), dtype=np.uint8)


class _FakeTimeStep:
    def __init__(self, observation, reward=0.5, discount=0.75, last=False):
        self.observation = observation
        self.reward = reward
        self.discount = discount
        self._last = last

    def last(self):
        return self._last


class _FakeDMCEnv:
    def __init__(self):
        self.physics = _FakePhysics()

    def observation_spec(self):
        return {"position": type("Spec", (), {"shape": (2,)})()}

    def action_spec(self):
        return type(
            "ActionSpec",
            (),
            {
                "minimum": np.array([-1.0, -1.0], dtype=np.float32),
                "maximum": np.array([1.0, 1.0], dtype=np.float32),
            },
        )()

    def reset(self):
        return _FakeTimeStep({"position": np.array([0.0, 1.0], dtype=np.float32)})

    def step(self, action):
        return _FakeTimeStep(
            {"position": np.asarray(action, dtype=np.float32)},
            reward=np.float32(2.0),
            discount=0.0,
            last=True,
        )


class _FakeMjModel:
    nu = 2
    nq = 0
    nv = 0

    def __init__(self):
        self.actuator_ctrlrange = np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=np.float32)
        self.actuator_ctrllimited = np.array([True, True])

    @staticmethod
    def from_xml_string(xml, assets=None):
        return _FakeMjModel()


class _FakeMjData:
    def __init__(self, model):
        self.ctrl = np.zeros((model.nu,), dtype=np.float32)
        self.qpos = np.zeros((0,), dtype=np.float64)
        self.qvel = np.zeros((0,), dtype=np.float64)
        self.time = 0.0


class _FakeRenderer:
    def __init__(self, model, height, width):
        self.height = height
        self.width = width

    def update_scene(self, data, camera=None):
        return None

    def render(self):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8) + 9

    def close(self):
        return None


def test_gym_image_env_contract_preserves_truncation_metadata():
    env = GymImageEnv(_ContractFiveTupleEnv(truncated=True), seed=3, size=(8, 8))

    obs = env.reset()
    next_obs, reward, done, info = env.step(np.array([-1.0, 0.5, 0.25], dtype=np.float32))
    frame = env.render()
    assert np.array_equal(frame, env.render())

    assert set(obs) == {"image"}
    assert set(next_obs) == set(obs)
    assert obs["image"].shape == (3, 8, 8)
    assert obs["image"].dtype == np.uint8
    assert int(obs["image"].min()) >= 0 and int(obs["image"].max()) <= 255
    assert int(next_obs["image"].min()) >= 0 and int(next_obs["image"].max()) <= 255
    assert frame.shape == (8, 8, 3)
    assert frame.dtype == np.uint8
    assert env.observation_space.contains(obs)
    assert env.observation_space.contains(next_obs)
    assert env.action_space.contains(info["action"])
    assert info["executed_action"] == 1
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert done is True
    assert info["terminated"] is False
    assert info["truncated"] is True
    assert info["discount"].dtype == np.float32
    assert "vector_observation" in info
    assert info["vector_observation"].shape == (4,)



def test_gym_image_env_can_include_optional_state_observation():
    env = GymImageEnv(_ContractDictObsEnv(), seed=4, size=(8, 8), include_state=True)

    obs = env.reset()
    next_obs, reward, done, info = env.step(np.array([0.2, -0.3], dtype=np.float32))

    assert set(obs) == {"image", "state"}
    assert set(next_obs) == set(obs)
    assert obs["state"].shape == (3,)
    assert next_obs["state"].shape == (3,)
    assert obs["state"].dtype == np.float32
    assert next_obs["state"].dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert done is False
    assert np.allclose(info["vector_observation"], next_obs["state"])


def test_normalize_actions_clips_to_unit_range_and_preserves_native_execution():
    env = NormalizeActions(_NormalizeActionBaseEnv())

    obs = env.reset()
    next_obs, reward, done, info = env.step(np.array([2.0, -2.0], dtype=np.float32))

    assert obs["image"].shape == (3, 2, 2)
    assert next_obs["image"].shape == (3, 2, 2)
    assert reward == 0.25
    assert done is False
    assert np.array_equal(info["action"], np.array([1.0, -1.0], dtype=np.float32))
    assert np.array_equal(
        info["executed_action"], np.array([2.0, -4.0], dtype=np.float32)
    )
    assert np.array_equal(
        env._env.last_action, np.array([2.0, -4.0], dtype=np.float32)
    )
    assert env.action_space.contains(info["action"])


def test_frame_stack_stacks_chw_images_and_preserves_state():
    env = FrameStack(_FrameStackBaseEnv(), num_frames=3)

    obs = env.reset()
    assert obs["image"].shape == (9, 2, 2)
    assert obs["state"].shape == (1,)
    assert np.all(obs["image"][0:3] == 1)
    assert np.all(obs["image"][3:6] == 1)
    assert np.all(obs["image"][6:9] == 1)

    next_obs, reward, done, info = env.step(0)
    assert reward == 1.0
    assert done is False
    assert isinstance(info, dict)
    assert np.all(next_obs["image"][0:3] == 1)
    assert np.all(next_obs["image"][3:6] == 1)
    assert np.all(next_obs["image"][6:9] == 2)
    assert next_obs["state"][0] == 2.0
    assert env.observation_space.contains(next_obs)


def test_time_limit_marks_wrapper_timeout_as_truncation():
    env = TimeLimit(_TimeLimitBaseEnv(), duration=1)

    env.reset()
    _, reward, done, info = env.step(None)

    assert reward == 0.5
    assert done is True
    assert info["terminated"] is False
    assert info["truncated"] is True
    assert np.asarray(info["discount"]).item() == 1.0


def test_dmc_adapter_matches_contract(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    fake_env = _FakeDMCEnv()
    dm_control = ModuleType("dm_control")
    dm_control.suite = SimpleNamespace(load=lambda domain, task, task_kwargs: fake_env)
    monkeypatch.setitem(sys.modules, "dm_control", dm_control)

    env = DeepMindControlEnv("cartpole-swingup", seed=1, size=(8, 8))
    obs = env.reset()
    frame = env.render()
    next_obs, reward, done, info = env.step(np.array([2.0, -2.0], dtype=np.float32))

    assert set(obs) == {"position", "image"}
    assert set(next_obs) == set(obs)
    assert frame.shape == (8, 8, 3)
    assert frame.shape == (8, 8, 3)
    assert env.observation_space.contains(obs)
    assert env.observation_space.contains(next_obs)
    assert env.action_space.contains(info["action"])
    assert np.array_equal(info["action"], np.array([1.0, -1.0], dtype=np.float32))
    assert np.array_equal(info["executed_action"], np.array([1.0, -1.0], dtype=np.float32))
    assert isinstance(reward, float)
    assert done is True
    assert info["terminated"] is True
    assert info["truncated"] is False
    assert np.asarray(info["discount"]).item() == 0.0
    assert info["vector_observation"].shape == (2,)


def test_mujoco_adapter_adds_discount_and_contract_fields(monkeypatch):
    import sys

    fake_mujoco = type("FakeMujoco", (), {})()
    fake_mujoco.MjModel = _FakeMjModel
    fake_mujoco.MjData = _FakeMjData
    fake_mujoco.Renderer = _FakeRenderer
    fake_mujoco.mj_resetData = lambda model, data: None
    fake_mujoco.mj_forward = lambda model, data: None
    fake_mujoco.mj_step = lambda model, data, nstep=1: setattr(data, "time", data.time + 0.01 * nstep)
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    env = MuJoCoImageEnv(
        xml_string="<mujoco/>",
        size=(8, 10),
        frame_skip=2,
        reward_fn=lambda model, data, action, info: float(action.sum()),
        terminal_fn=lambda model, data, info: data.time > 0.0,
        include_state=True,
    )
    obs = env.reset(seed=5)
    frame = env.render()
    assert np.array_equal(frame, env.render())
    next_obs, reward, done, info = env.step(np.array([0.5, -0.25], dtype=np.float32))

    assert set(obs) == {"image", "state"}
    assert set(next_obs) == set(obs)
    assert obs["state"].shape == (0,)
    assert next_obs["state"].shape == (0,)
    assert frame.shape == (8, 10, 3)
    assert env.observation_space.contains(obs)
    assert env.observation_space.contains(next_obs)
    assert env.action_space.contains(info["action"])
    assert np.array_equal(info["executed_action"], info["action"])
    assert isinstance(reward, float)
    assert done is True
    assert info["terminated"] is True
    assert info["truncated"] is False
    assert np.asarray(info["discount"]).item() == 0.0
    assert info["vector_observation"].shape == (0,)
