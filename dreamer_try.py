from world_models.models.dreamer_v1 import DreamerV1

# Create and train on HalfCheetah
dreamer = DreamerV1(env="Walker2d-v5", memory_size=100000)
dreamer.train(episodes=2000)
