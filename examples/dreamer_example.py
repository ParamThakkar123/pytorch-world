"""
Example script for training a Dreamer agent on a Gym environment.

This demonstrates how to use the DreamerAgent class for end-to-end training
of a world model-based reinforcement learning agent.
"""

import logging
from omegaconf import OmegaConf
from torchwm.models.dreamer import DreamerAgent
from torchwm.configs.dreamer_config import DreamerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    cli_cfg = OmegaConf.from_cli()

    env = cli_cfg.get("env", "cartpole_balance")
    total_steps = int(cli_cfg.get("total_steps", 10000))
    seed = int(cli_cfg.get("seed", 42))
    logdir = cli_cfg.get("logdir", None)
    device = cli_cfg.get("device", "auto")

    logger.info(f"Training Dreamer on {env}")
    logger.info(f"Total steps: {total_steps}")

    config = DreamerConfig()
    config.env = env
    config.total_steps = total_steps
    config.seed = seed

    if device != "auto":
        config.no_gpu = device == "cpu"

    agent = DreamerAgent(config, logdir=logdir)
    agent.train(total_steps=total_steps)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
