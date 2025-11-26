from world_models.models.planet import Planet

p = Planet(
    env="CartPole-v1",
    bit_depth=5,
    headless=True,
    max_episode_steps=100,
    action_repeats=1,
    results_dir="my_experiment",
)
p.train(epochs=1)
