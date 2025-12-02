from world_models.models.planet import Planet
from world_models.envs.mujoco_env import make_half_cheetah_env

env = make_half_cheetah_env(
    version="v4",
    forward_reward_weight=1.0,
    reset_noise_scale=0.01,
    render_mode="rgb_array",
)

planet = Planet(
    env=env,
    bit_depth=5,
    device=None,
    state_size=200,
    latent_size=30,
    embedding_size=1024,
    memory_size=500,
    policy_cfg={
        "planning_horizon": 35,
        "num_candidates": 1000,
        "num_iterations": 10,
        "top_candidates": 100,
    },
    headless=True,
    max_episode_steps=200,
    action_repeats=2,
    results_dir="results/halfcheetah_planet",
)

planet.warmup(n_episodes=10, random_policy=True)

planet.train(
    epochs=200,
    steps_per_epoch=150,
    batch_size=64,
    H=50,
    beta=1.0,
    save_every=25,
    record_grads=False,
)
