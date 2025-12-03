from world_models.envs.dreamer_envs import ControlSuiteEnv
from world_models.models.dreamer_v1 import DreamerV1


# --- Parameters ---
env_name = "walker-walk"
seed = 42
action_repeat = 2  # Recommended for walker is 2
max_episode_length = 1000
bit_depth = 5
batch_size = 50
epochs = 100
steps_per_epoch = 200
save_every = 25

# --- Environment ---
# The ControlSuiteEnv will handle image-based observations when symbolic=False
env = ControlSuiteEnv(
    env=env_name,
    symbolic=False,
    seed=seed,
    max_episode_length=max_episode_length,
    action_repeat=action_repeat,
    bit_depth=bit_depth,
)

# --- DreamerV1 Model ---
model = DreamerV1(
    env,
    bit_depth=bit_depth,
    action_repeat=action_repeat,
    results_dir=f"results/dreamer_v1_{env_name}",
    seed=seed,
    symbolic=False,
    batch_size=batch_size,
)

# --- Training ---
# 1. Collect initial experience with a random policy
model.warmup(n_episodes=5)

# 2. Train the agent
print(f"\nStarting training on {env_name} for {epochs} epochs...")
model.train(
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    save_every=save_every,
    batch_size=batch_size,
)
print("\nTraining finished.")

# 3. Evaluate the trained agent
print("\nEvaluating the final agent...")
model.evaluate(episodes=10)
print("\nEvaluation finished.")
