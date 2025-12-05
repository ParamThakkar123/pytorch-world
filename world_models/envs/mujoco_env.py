from typing import Tuple
import gymnasium as gym


def make_humanoid_env(
    version: str = "v4",
    xml_file: str = "humanoid.xml",
    forward_reward_weight: float = 1.25,
    ctrl_cost_weight: float = 0.1,
    contact_cost_weight: float = 5e-7,
    healthy_reward: float = 5.0,
    terminate_when_unhealthy: bool = True,
    healthy_z_range: Tuple[float, float] = (1.0, 2.0),
    reset_noise_scale: float = 1e-2,
    exclude_current_positions_from_observation: bool = True,
    include_cinert_in_observation: bool = True,
    include_cvel_in_observation: bool = True,
    include_qfrc_actuator_in_observation: bool = True,
    include_cfrc_ext_in_observation: bool = True,
) -> gym.Env:
    """
    Create a Humanoid environment with customizable parameters.

    Args:
        version (str): The version of the Humanoid environment (e.g., "v4").
        xml_file (str): The XML file defining the Humanoid model.
        forward_reward_weight (float): Weight for the forward reward.
        ctrl_cost_weight (float): Weight for the control cost.
        contact_cost_weight (float): Weight for the contact cost.
        healthy_reward (float): Reward for being in a healthy state.
        terminate_when_unhealthy (bool): Whether to terminate the episode when unhealthy.
        healthy_z_range (Tuple[float, float]): The range of z-values considered healthy.
        reset_noise_scale (float): Scale of noise added during environment reset.
        exclude_current_positions_from_observation (bool): Whether to exclude current positions from observations.
        include_cinert_in_observation (bool): Whether to include inertia in observations.
        include_cvel_in_observation (bool): Whether to include velocity in observations.
        include_qfrc_actuator_in_observation (bool): Whether to include actuator forces in observations.
        include_cfrc_ext_in_observation (bool): Whether to include external forces in observations.

    Returns:
        gym.Env: The created Humanoid environment.
    """

    env_id = f"Humanoid-{version}"

    return gym.make(
        env_id,
        xml_file=xml_file,
        forward_reward_weight=forward_reward_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        contact_cost_weight=contact_cost_weight,
        healthy_reward=healthy_reward,
        terminate_when_unhealthy=terminate_when_unhealthy,
        healthy_z_range=healthy_z_range,
        reset_noise_scale=reset_noise_scale,
        exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        include_cinert_in_observation=include_cinert_in_observation,
        include_cvel_in_observation=include_cvel_in_observation,
        include_qfrc_actuator_in_observation=include_qfrc_actuator_in_observation,
        include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
    )


def make_half_cheetah_env(
    version: str = "v4",
    forward_reward_weight: float = 0.1,
    reset_noise_scale: float = 0.1,
    exclude_current_positions_from_observation: bool = True,
    render_mode: str = "rgb_array",
) -> gym.Env:
    """
    Create a HalfCheetah environment with customizable parameters.

    Args:
        version (str): The version of the HalfCheetah environment (e.g., "v4").
        forward_reward_weight (float): Weight for the forward reward.
        reset_noise_scale (float): Scale of noise added during environment reset.
        exclude_current_positions_from_observation (bool): Whether to exclude current positions from observations.
        render_mode (str): The render mode for the environment.

    Returns:
        gym.Env: The created HalfCheetah environment.
    """

    env_id = f"HalfCheetah-{version}"

    return gym.make(
        env_id,
        forward_reward_weight=forward_reward_weight,
        reset_noise_scale=reset_noise_scale,
        exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        render_mode=render_mode,
    )
