class DrivingWorldConfig:
    def __init__(self):
        # Environment and dataset
        self.env = "driving"  # Placeholder for driving environment/dataset
        self.dataset_path = ""  # Path to driving dataset (e.g., images, poses, yaws)
        self.vqvae_checkpoint_path = ""  # Path to pre-trained VQ-VAE model

        # VQ-VAE related
        self.codebook_size = 512
        self.codebook_embed_dim = 256
        self.image_size = (64, 64)
        self.downsample_size = 8

        # Training parameters
        self.pkeep = 0.8
        self.mask_data = 0.1
        self.learning_rate = 1e-4
        self.batch_size = 4
        self.n_epochs = 10
        self.grad_clip_norm = 1.0  # Gradient clipping norm
        self.adam_epsilon = 1e-8  # Adam epsilon

        # Pose and yaw vocabularies
        self.pose_x_vocab_size = 100
        self.pose_y_vocab_size = 100
        self.yaw_vocab_size = 50

        # Transformer architecture
        self.n_layer = 6
        self.n_embd = 512
        self.gpt_type = "models.vid_gpt"  # Module path for GPT model
        self.condition_frames = 3  # Number of condition frames for transformer

        # General settings
        self.seed = 42
        self.no_gpu = False  # If True, use CPU
        self.device = None  # Will be set based on no_gpu

        # Logging and saving
        self.results_dir = "results/driving_world"
        self.logdir = ""  # Specific log directory (auto-generated if empty)
        self.checkpoint_interval = 10  # Save checkpoint every N epochs
        self.checkpoint_path = ""  # Path to load checkpoint for restore
        self.restore = False  # Whether to restore from checkpoint

        # Evaluation
        self.test_episodes = 10  # Number of test episodes for evaluation
        self.render = False  # Whether to render during evaluation

        # Data loading
        self.num_workers = 4  # Number of workers for data loader
        self.pin_memory = True  # Pin memory for data loader

        # Algorithm-specific
        self.algo = "DrivingWorld"
        self.exp_name = "default_run"

        # Additional hyperparameters (can be extended)
        self.free_nats = 3.0  # For KL divergence in future extensions
        self.kl_loss_coeff = 1.0
        self.discount = 0.99  # Discount factor if needed for RL integration

    def update(self, **kwargs):
        """Update config attributes dynamically."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid config attribute: {key}")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"
