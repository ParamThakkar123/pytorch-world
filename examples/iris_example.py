"""
Example script for training an IRIS agent.

IRIS (Imagination with Recurrent State-Space Models) is a world model-based
reinforcement learning agent that uses a transformer-based architecture
for imagination and planning.
"""

import logging
import torch
from omegaconf import OmegaConf
from torchwm.models.iris_agent import IRISAgent
from torchwm.configs.iris_config import IRISConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    cli_cfg = OmegaConf.from_cli()

    env = cli_cfg.get("env", "cartpole_balance")
    total_steps = int(cli_cfg.get("total_steps", 10000))
    batch_size = int(cli_cfg.get("batch_size", 32))
    device_str = cli_cfg.get("device", "auto")

    logger.info(f"Training IRIS on {env}")

    config = IRISConfig()
    config.env = env
    config.total_steps = total_steps
    config.batch_size = batch_size

    if device_str != "auto":
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = IRISAgent(config, action_size=4, device=device)

    logger.info(
        "IRIS agent created. For full training, integrate with environment data collection."
    )

    initial_frame = torch.randn(2, 3, 64, 64).to(device)
    trajectory = agent.imagine_rollout(initial_frame, horizon=10)
    logger.info(
        f"Imagination rollout completed. Trajectory shapes: {trajectory['frames'].shape}"
    )


if __name__ == "__main__":
    main()
