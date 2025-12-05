from world_models.models.dreamer import DreamerAgent

agent = DreamerAgent(
    env="walker-walk",
    action_repeat=2,
    exp_name="test_run",
    logdir="my_dreamer_experiment",
)
agent.train(total_steps=100000)
