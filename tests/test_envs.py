from world_models.envs.dmc import DeepMindControlEnv


class TestDeepMindControlEnv:
    def test_env_creation(self):
        env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_env_step_and_reset(self):
        env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        obs = env.reset()
        assert "image" in obs
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        assert "image" in next_obs
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "discount" in info

    def test_env_render(self):
        env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        env.reset()
        frame = env.render()
        assert frame.shape == (64, 64, 3)
        env.close()


class TestMujocoEnv:
    pass
