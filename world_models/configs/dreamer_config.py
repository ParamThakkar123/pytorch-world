from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from world_models.configs.serialization import SerializableConfigMixin


@dataclass
class DreamerConfig(SerializableConfigMixin):
    """Configuration container for Dreamer training, evaluation, and environment setup.

    This class centralizes environment backend selection (DMC/DMLab/Gym/MuJoCo/Robotics/Unity/Brax),
    model dimensions, replay and optimization settings, logging cadence, and
    checkpoint options consumed by `DreamerAgent`.
    """

    # Environment selection.
    # dmc: DeepMind Control Suite
    # dmlab: DeepMind Lab 3D navigation tasks
    # gym: generic Gym/Gymnasium env IDs or prebuilt env instances
    # mujoco: Gymnasium MuJoCo task IDs or native MuJoCo XML/MJB
    # robotics: Gymnasium Robotics env IDs (including legacy MuJoCo v2/v3)
    # procgen: Procgen procedurally generated benchmark games
    # unity_mlagents: Unity ML-Agents executable
    # brax: JAX/Brax continuous-control environments
    env_backend: str = "dmc"
    env: str = "walker-walk"
    env_instance: Any = None
    image_size: tuple[int, int] = (64, 64)
    gym_render_mode: str = "rgb_array"
    frame_stack: int = 1

    # DeepMind Lab options. dmlab_action_repeat is native DMLab frame
    # repeat; Dreamer action_repeat is still applied by the shared wrapper
    # stack outside the backend adapter.
    dmlab_action_repeat: int = 4
    dmlab_action_set: Any = None
    dmlab_observations: Any = None
    dmlab_config: Any = None
    dmlab_renderer: str = "hardware"

    # Procgen options. Use env values like "coinrun" or "procgen-coinrun-v0".
    procgen_distribution_mode: str = "easy"
    procgen_num_levels: int = 0
    procgen_start_level: Any = None

    # MuJoCo options. Leave mujoco_xml_path unset to auto-detect whether
    # `env` is a Gymnasium task ID or a native MJCF/MJB source.
    mujoco_xml_path: Any = None
    mujoco_xml_string: Any = None
    mujoco_binary_path: Any = None
    mujoco_camera: Any = None
    mujoco_frame_skip: int = 1
    mujoco_reset_noise_scale: float = 0.0

    # Brax options.
    brax_backend: str = "generalized"
    brax_jit: bool = True
    brax_auto_reset: bool = False
    # Suppress noisy optional MuJoCo/MJX Warp import messages emitted during
    # Brax imports. These messages are harmless when Warp is not installed
    # but can clutter logs; enable suppression by default.
    brax_suppress_warp_warnings: bool = True

    # Unity ML-Agents options.
    unity_file_name: Any = None
    unity_behavior_name: Any = None
    unity_worker_id: int = 0
    unity_base_port: int = 5005
    unity_no_graphics: bool = True
    unity_time_scale: float = 20.0
    unity_quality_level: int = 1

    algo: str = "Dreamerv1"
    exp_name: str = "lr1e-3"
    train: bool = True
    evaluate: bool = False
    seed: int = 1
    no_gpu: bool = False
    max_episode_length: int = 1000
    buffer_size: int = 800000
    time_limit: int = 1000
    cnn_activation_function: str = "relu"
    dense_activation_function: str = "elu"
    obs_embed_size: int = 1024
    num_units: int = 400
    deter_size: int = 200
    stoch_size: int = 30
    action_repeat: int = 2
    action_noise: float = 0.3
    total_steps: int = 5_000_000
    seed_steps: int = 5000
    update_steps: int = 100
    collect_steps: int = 1000
    batch_size: int = 50
    train_seq_len: int = 50
    imagine_horizon: int = 15
    use_disc_model: bool = False
    # KL free-nats floor. Once mean KL(posterior || prior) falls below this
    # value the KL term contributes zero gradient, which freezes the RSSM's
    # prior stochastic head -- the only module trained solely by the KL. Since
    # imagination samples `stoch` from that head, a floor set too high leaves
    # the actor optimizing inside an untrained transition model. 3.0 is the
    # DMC-image default; low-entropy observations (e.g. Pendulum) never clear
    # it, so the default here is lowered.
    free_nats: float = 1.0
    discount: float = 0.99
    # Multiplier applied to replayed rewards before they reach the reward head.
    # Dreamer's defaults assume DMC-style per-step rewards in [0, 1]. Envs with
    # much larger magnitudes (Pendulum-v1 spans [-16.3, 0]) inflate returns,
    # value targets, and gradient norms enough that grad clipping fires on
    # nearly every update. Set e.g. reward_scale=0.125 for Pendulum-v1.
    reward_scale: float = 1.0
    td_lambda: float = 0.95
    kl_loss_coeff: float = 1.0
    kl_alpha: float = 0.8
    disc_loss_coeff: float = 10.0
    num_buckets: int = 255
    symlog_range: float = 10.0
    model_learning_rate: float = 6e-4
    actor_learning_rate: float = 8e-5
    value_learning_rate: float = 8e-5
    adam_epsilon: float = 1e-7
    # World-model gradient norms routinely reach ~10^3 on image observations,
    # so a threshold of 100 clipped ~97% of updates and made every step a
    # fixed-size move in the gradient direction.
    grad_clip_norm: float = 1000.0
    # fp16 overflow makes GradScaler silently skip world-model and value
    # updates, and Dreamer is env-bound rather than compute-bound at these
    # model sizes, so AMP buys little. Opt in explicitly if profiling shows it
    # helps.
    use_amp: bool = False
    test: bool = False
    test_interval: int = 10000
    test_episodes: int = 10
    scalar_freq: int = 1_000
    log_video_freq: int = -1
    max_videos_to_save: int = 2
    video_format: str = "gif"  # "gif" or "mp4"
    video_fps: int = 20
    checkpoint_interval: int = 10000
    checkpoint_path: str = ""
    restore: bool = False
    experience_replay: str = ""
    render: bool = False

    # Logging options
    enable_wandb: bool = False
    wandb_project: str = "torchwm"
    wandb_entity: str = ""
    log_dir: str = "runs"
    logdir: Any = None
    # Base directory for DreamerAgent-created relative log directories.
    # If unset, DreamerAgent uses TORCHWM_DATA_DIR or log_dir instead of
    # writing into the package source tree.
    data_dir: Any = None
    log_level: str = "INFO"
    log_file: Any = None
    enable_tensorboard: bool = False
    enable_console_metrics: bool = True
    enable_jsonl: bool = True
    jsonl_filename: str = "metrics.jsonl"
    log_system_stats_freq: int = 1_000
    detect_anomaly: bool = False
