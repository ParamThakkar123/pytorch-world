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

    def test_env_seed_reproducibility(self):
        env1 = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        env2 = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert (obs1["image"] == obs2["image"]).all()
        action = env1.action_space.sample()
        next_obs1, _, _, _ = env1.step(action)
        next_obs2, _, _, _ = env2.step(action)
        assert (next_obs1["image"] == next_obs2["image"]).all()

    def test_invalid_env_name(self):
        try:
            DeepMindControlEnv(name="invalid-env", seed=42, size=(64, 64))
        except Exception as e:
            assert isinstance(e, ValueError)

    def test_different_image_sizes(self):
        sizes = [(32, 32), (64, 64), (128, 128)]
        for size in sizes:
            env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=size)
            obs = env.reset()
            assert obs["image"].shape == (size[0], size[1], 3)

    def test_multiple_resets(self):
        env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        for _ in range(5):
            obs = env.reset()
            assert "image" in obs

    def test_step_after_done(self):
        env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        try:
            env.step(env.action_space.sample())
        except Exception as e:
            assert isinstance(e, RuntimeError)

    def test_close_env(self):
        env = DeepMindControlEnv(name="cartpole-swingup", seed=42, size=(64, 64))
        env.reset()
        env.close()
        try:
            env.step(env.action_space.sample())
        except Exception as e:
            assert isinstance(e, RuntimeError)


class TestMujocoEnv:
    pass
