import pytest
from unittest.mock import Mock, patch
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
        config.seed_steps = 100
        config.action_repeat = 1
        config.restore = False
        return config

    @patch("world_models.models.dreamer.make_env")
    @patch("world_models.models.dreamer.Logger")
    def test_initialization(self, mock_logger, mock_make_env, config):
        mock_env = Mock()
        mock_env.observation_space.shape = (4,)
        mock_env.action_space.n = 2
        mock_make_env.return_value = mock_env

        agent = DreamerAgent(config)

        assert agent.config == config
        assert isinstance(agent.rssm, RSSM)
        assert agent.env == mock_env
        assert agent.logger == mock_logger.return_value
