import pytest
from torchwm.configs.dreamer_config import DreamerConfig


@pytest.mark.integration
def test_dreamer_pendulum_no_crash(tmp_path):
    pytest.importorskip("gymnasium")
    from torchwm.models.dreamer import DreamerAgent

    config = DreamerConfig()
    config.env = "Pendulum-v1"
    config.env_backend = "gym"
    config.total_steps = 100
    config.seed_steps = 100
    config.action_repeat = 1
    config.collect_steps = 1
    config.update_steps = 1
    config.batch_size = 1
    config.train_seq_len = 1
    config.imagine_horizon = 1
    config.no_gpu = True
    config.test_interval = 100000
    config.checkpoint_interval = 100000
    config.log_video_freq = -1
    config.seed = 42
    config.logdir = str(tmp_path / "dreamer_integration_test")
    config.enable_wandb = False
    config.enable_tensorboard = False
    config.enable_console_metrics = False
    config.enable_jsonl = False

    agent = DreamerAgent(config)
    agent.train(total_steps=100)
