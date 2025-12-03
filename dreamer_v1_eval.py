from world_models.models.dreamer_v1 import DreamerV1

# Initialize the agent
dreamer = DreamerV1(env="Walker2d-v5", memory_size=10000)

# Path to your checkpoint file
checkpoint_path = "results/dreamer_v1_Walker2d-v5/checkpoint_3000.pth"

# Load the trained weights
dreamer.load_checkpoint(checkpoint_path)

# Evaluate the agent's performance for 1000 episodes
eval_metrics = dreamer.evaluate(episodes=1000)
print(f"Evaluation results: {eval_metrics}")

# You can also generate a video of the agent's performance
_, _, frames = dreamer.collect_episode(deterministic=True)
