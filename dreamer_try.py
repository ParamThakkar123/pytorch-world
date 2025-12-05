from world_models.configs.dreamer_config import DreamerConfig
from world_models.models.dreamer import DreamerAgent

# Create config and override relevant fields
cfg = DreamerConfig()
cfg.env = "walker-walk"
cfg.action_repeat = 2
cfg.exp_name = "full_config_test"
cfg.total_steps = 400000
cfg.seed = 42
cfg.no_gpu = False
cfg.render = False

# Training / buffer / optimization
cfg.train = True
cfg.evaluate = False
cfg.buffer_size = 800000
cfg.time_limit = 1000
cfg.seed_steps = 5000
cfg.update_steps = 100
cfg.collect_steps = 1000
cfg.batch_size = 50
cfg.train_seq_len = 50
cfg.checkpoint_interval = 10000
cfg.checkpoint_path = ""

# Model / architecture
cfg.obs_embed_size = 1024
cfg.num_units = 400
cfg.deter_size = 200
cfg.stoch_size = 30
cfg.cnn_activation_function = "relu"
cfg.dense_activation_function = "elu"

# Planning / imagination
cfg.imagine_horizon = 15
cfg.use_disc_model = False
cfg.action_noise = 0.3

# Loss / scheduler / hyperparams
cfg.free_nats = 3.0
cfg.discount = 0.99
cfg.td_lambda = 0.95
cfg.kl_loss_coeff = 1.0
cfg.kl_alpha = 0.8
cfg.disc_loss_coeff = 10.0

# Learning rates / optimizer
cfg.model_learning_rate = 6e-4
cfg.actor_learning_rate = 8e-5
cfg.value_learning_rate = 8e-5
cfg.adam_epsilon = 1e-7
cfg.grad_clip_norm = 100.0

# Logging / evaluation
cfg.test_interval = 10000
cfg.test_episodes = 10
cfg.log_video_freq = -1
cfg.max_videos_to_save = 2

# Instantiate agent with the config and run training
agent = DreamerAgent(config=cfg)
agent.train(total_steps=cfg.total_steps)
