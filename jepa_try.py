from world_models.models.jepa_agent import JEPAAgent

agent = JEPAAgent(
    folder="results/jepa_try",
    write_tag="jepa_try",
)
agent.train()
