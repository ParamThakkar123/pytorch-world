import pytest
import numpy as np
from unittest.mock import Mock, patch

gym = pytest.importorskip("gym")

from world_models.models.dreamer import DreamerAgent
from world_models.models.dreamer_rssm import RSSM
from world_models.configs.dreamer_config import DreamerConfig


class TestDreamerAgent:
    @pytest.fixture
    def config(self):
        config = DreamerConfig()
        config.env = "cartpole_balance"
        config.seed = 42
        config.total_steps = 1000
        config.seed_steps = 1
        config.action_repeat = 1
        config.restore = False
        config.buffer_size = 10
        return config

    @patch("world_models.models.dreamer.make_env")
    @patch("world_models.models.dreamer.Logger")
    def test_initialization(self, mock_logger, mock_make_env, config):
        mock_env = Mock()
        mock_obs_space = Mock()
        mock_obs_space.shape = (3, 64, 64)  # Assuming image shape for DMC env
        mock_env.observation_space = {"image": mock_obs_space}
        mock_action_space = Mock()
        mock_action_space.shape = (2,)  # Assuming action shape
        mock_env.action_space = mock_action_space
        mock_make_env.return_value = mock_env

        config.buffer_size = 10  # Reduce for test
        agent = DreamerAgent(config)

        assert agent.args == config
        assert isinstance(agent.dreamer.rssm, RSSM)
        assert agent.train_env == mock_env
        assert agent.test_env == mock_env
        assert agent.logger == mock_logger.return_value

    @patch("world_models.models.dreamer.make_env")
    @patch("world_models.models.dreamer.Logger")
    def test_evaluate(self, mock_logger, mock_make_env, config):
        mock_env = Mock()
        mock_obs_space = Mock()
        mock_obs_space.shape = (3, 64, 64)
        mock_env.observation_space = {"image": mock_obs_space}
        mock_action_space = Mock()
        mock_action_space.shape = (2,)
        mock_env.action_space = mock_action_space
        mock_make_env.return_value = mock_env

        config.buffer_size = 10  # Reduce for test
        agent = DreamerAgent(config)
        agent.dreamer.evaluate = Mock(
            return_value=(np.array([4.0, 5.0]), np.array([[]]), None)
        )

        agent.evaluate()

        agent.dreamer.evaluate.assert_called_once_with(
            agent.test_env, config.test_episodes, render=True
        )
        mock_logger.return_value.dump_scalars_to_pickle.assert_called_once()
        mock_logger.return_value.log_videos.assert_called_once()

    @patch("world_models.models.dreamer.make_env")
    @patch("world_models.models.dreamer.Logger")
    def test_initialization_with_invalid_arg(self, mock_logger, mock_make_env, config):
        mock_env = Mock()
        mock_obs_space = Mock()
        mock_obs_space.shape = (3, 64, 64)
        mock_env.observation_space = {"image": mock_obs_space}
        mock_action_space = Mock()
        mock_action_space.shape = (2,)
        mock_env.action_space = mock_action_space
        mock_make_env.return_value = mock_env

        with pytest.raises(ValueError, match="Invalid argument: invalid_arg"):
            DreamerAgent(config, invalid_arg="test")


def _mock_image_env():
    mock_env = Mock()
    mock_obs_space = Mock()
    mock_obs_space.shape = (3, 64, 64)
    mock_env.observation_space = {"image": mock_obs_space}
    mock_action_space = Mock()
    mock_action_space.shape = (2,)
    mock_env.action_space = mock_action_space
    return mock_env


class _FrameStackCollectEnv:
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.spec = type("Spec", (), {"max_episode_steps": 5})()
        self._step_count = 0

    def reset(self, seed=None):
        self._step_count = 0
        return np.array([0.0, 0.1, 0.2], dtype=np.float32), {"seed": seed}

    def step(self, action):
        self._step_count += 1
        obs = np.array([0.2, 0.1, 0.0], dtype=np.float32) + self._step_count
        reward = float(self._step_count)
        terminated = self._step_count >= 2
        truncated = False
        return obs.astype(np.float32), reward, terminated, truncated, {}

    def render(self, *args, **kwargs):
        value = 32 + self._step_count * 16
        return np.full((8, 8, 3), value, dtype=np.uint8)

    def close(self):
        pass


class TestDreamerEnvConfig:
    def test_config_defaults_to_single_frame_stack(self):
        config = DreamerConfig()
        assert config.frame_stack == 1

    def test_yaml_roundtrip_preserves_frame_stack(self, tmp_path):
        config = DreamerConfig()
        config.frame_stack = 3
        path = tmp_path / "dreamer.yaml"
        config.to_yaml(path)

        loaded = DreamerConfig.from_yaml(path)
        assert loaded.frame_stack == 3


class TestDreamerFrameStackIntegration:
    @patch("world_models.models.dreamer.Logger")
    def test_collect_random_episodes_uses_stacked_frames_end_to_end(
        self, mock_logger, tmp_path
    ):
        config = DreamerConfig()
        config.env_backend = "gym"
        config.env_instance = _FrameStackCollectEnv()
        config.image_size = (8, 8)
        config.frame_stack = 2
        config.action_repeat = 1
        config.seed_steps = 3
        config.train_seq_len = 2
        config.batch_size = 1
        config.buffer_size = 8
        config.no_gpu = True
        config.use_amp = False
        config.logdir = str(tmp_path / "run")
        config.restore = False
        config.to_yaml = Mock(return_value="")

        agent = DreamerAgent(config)
        rewards = agent.dreamer.collect_random_episodes(agent.train_env, seed_steps=3)

        assert agent.train_env.observation_space["image"].shape == (6, 8, 8)
        assert agent.dreamer.obs_shape == (6, 8, 8)
        assert agent.dreamer.data_buffer.observations.shape == (8, 6, 8, 8)
        assert agent.dreamer.data_buffer.steps == 3
        assert agent.dreamer.data_buffer.episodes == 1
        assert rewards.shape == (2,)
        assert rewards.tolist() == [3.0, 1.0]
        first_obs = agent.dreamer.data_buffer.observations[0]
        second_obs = agent.dreamer.data_buffer.observations[1]
        assert first_obs.shape == (6, 8, 8)
        assert np.array_equal(first_obs[0:3], first_obs[3:6])
        assert np.array_equal(second_obs[0:3], first_obs[3:6])
        assert not np.array_equal(second_obs[3:6], second_obs[0:3])
        mock_logger.assert_called_once()


class TestDreamerConfigSerialization:
    def test_to_dict_and_yaml_roundtrip(self, tmp_path):
        config = DreamerConfig()
        config.env = "cartpole_balance"
        config.seed = 123
        config.image_size = (32, 48)

        config_path = tmp_path / "dreamer.yaml"
        yaml_text = config.to_yaml(config_path)
        loaded = DreamerConfig.from_yaml(config_path)
        loaded_from_text = DreamerConfig.from_yaml(yaml_text)

        assert loaded.to_dict() == config.to_dict()
        assert loaded_from_text.to_dict() == config.to_dict()
        assert loaded.image_size == (32, 48)

    def test_from_dict_rejects_unknown_fields(self):
        with pytest.raises(ValueError, match="Invalid DreamerConfig field: nope"):
            DreamerConfig.from_dict({"nope": 1})


class TestDreamerUXConstructors:
    @patch("world_models.models.dreamer.make_env")
    @patch("world_models.models.dreamer.Logger")
    def test_agent_from_config_yaml_and_summary(
        self, mock_logger, mock_make_env, tmp_path
    ):
        mock_make_env.return_value = _mock_image_env()
        config = DreamerConfig()
        config.buffer_size = 10
        config.logdir = str(tmp_path / "run")
        config_path = tmp_path / "config.yaml"
        config.to_yaml(config_path)

        agent = DreamerAgent.from_config(config_path, seed=99)
        summary = agent.summary()

        assert agent.args.seed == 99
        assert (tmp_path / "run" / "config.yaml").exists()
        assert agent.parameter_count() == summary["total_parameters"]
        assert summary["total_parameters"] > 0
        assert "rssm" in summary["modules"]
        mock_logger.assert_called_once()

    @patch("world_models.models.dreamer.make_env")
    @patch("world_models.models.dreamer.Logger")
    def test_agent_save_and_from_pretrained_local_dir(
        self, mock_logger, mock_make_env, tmp_path
    ):
        mock_make_env.return_value = _mock_image_env()
        config = DreamerConfig()
        config.buffer_size = 10
        config.logdir = str(tmp_path / "run")
        agent = DreamerAgent(config)
        checkpoint_dir = tmp_path / "pretrained"
        checkpoint_path = checkpoint_dir / "model.pt"

        agent.dreamer.save(checkpoint_path)
        loaded = DreamerAgent.from_pretrained(
            checkpoint_dir, logdir=str(tmp_path / "loaded")
        )

        assert (checkpoint_dir / "config.yaml").exists()
        assert (
            loaded.summary()["total_parameters"] == agent.summary()["total_parameters"]
        )

    @patch("world_models.models.dreamer.make_env")
    @patch("world_models.models.dreamer.Logger")
    def test_export_uses_dreamer_actor_default(
        self, mock_logger, mock_make_env, monkeypatch, tmp_path
    ):
        import world_models.export as export_utils

        mock_env = Mock()
        mock_obs_space = Mock()
        mock_obs_space.shape = (3, 64, 64)
        mock_env.observation_space = {"image": mock_obs_space}
        mock_action_space = Mock()
        mock_action_space.shape = (2,)
        mock_env.action_space = mock_action_space
        mock_make_env.return_value = mock_env

        config = DreamerConfig()
        config.buffer_size = 10

        calls = {}

        def fake_export_model(module, path, **kwargs):
            calls["module"] = module
            calls["path"] = path
            calls["kwargs"] = kwargs
            return path

        monkeypatch.setattr(export_utils, "export_model", fake_export_model)
        agent = DreamerAgent(config)
        out_path = tmp_path / "dreamer.onnx"

        returned = agent.export(out_path, format="onnx")

        assert returned == out_path
        assert calls["kwargs"]["format"] == "onnx"
        assert calls["kwargs"]["example_inputs"].shape == (
            1,
            config.stoch_size + config.deter_size,
        )
        assert calls["module"](calls["kwargs"]["example_inputs"]).shape == (1, 2)
