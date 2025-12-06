from world_models.models.dreamer import DreamerAgent

# point to a saved checkpoint produced by training (see Dreamer.save)
CKPT = "/mnt/e/pytorch-world/world_models/models/data/walker-walk_Dreamerv1_full_config_test_06-12-2025-01-04-15/ckpts/400000_ckpt.pt"

agent = DreamerAgent(
    restore=True,
    checkpoint_path=CKPT,
    env="walker-walk",
    test_episodes=100,
)
# This calls Dreamer.evaluate(...) and saves videos/logs via Logger
agent.evaluate()
