from world_models.models.dreamer_v1 import DreamerV1
from world_models.envs.dreamer_envs import Env

# Create the dm_control environment object
env = Env(
    env=("walker", "walk"),
    symbolic=False,
    seed=0,
    max_episode_length=1000,
    action_repeat=2,
    bit_depth=5,
)

# Create and train on dm_control's walker
dreamer = DreamerV1(env=env, memory_size=10000)
dreamer.train(episodes=3000)
