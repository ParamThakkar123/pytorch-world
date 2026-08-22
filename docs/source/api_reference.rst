API Reference
=============

This reference is generated from source docstrings and grouped by workflow. Use
:doc:`world_models_guide` for conceptual explanations and this page for exact
classes, functions, and module-level APIs.

Public package surface
----------------------

These modules expose the most common imports and lazy constructors.

Use ``torchwm`` for common workflows::

   import torchwm
   agent = torchwm.create_model("dreamer", env="walker-walk")

**Primary modules:** ``torchwm``, ``torchwm.models``, ``torchwm.configs``, ``torchwm.catalog``, and ``torchwm.envs``.

.. automodule:: torchwm
   :no-index:

.. automodule:: torchwm.api
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.export
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models
   :no-index:

.. automodule:: torchwm.catalog
   :members:
   :undoc-members:
   :show-inheritance:

Model catalog
-------------

Core model families
~~~~~~~~~~~~~~~~~~~

**Key classes:** ``Dreamer``, ``DreamerAgent``, ``RSSM``, ``RecurrentStateSpaceModel``, ``Planet``, ``ModularRSSM``, ``JEPAAgent``, ``VisionTransformer``, ``IRISAgent``, ``IRISTransformer``, ``IRISWorldModel``, ``Genie``, ``LatentActionModel``, and ``DynamicsModel``.

.. automodule:: torchwm.models.dreamer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.dreamer_rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.planet
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.mdrnn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.controller
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.modular_rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.jepa_agent
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.vit
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.iris_agent
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.iris_transformer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.genie
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.latent_action_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.dynamics_model
   :members:
   :undoc-members:
   :show-inheritance:

Diffusion and DIAMOND components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Key classes:** ``DiamondAgent``, ``DDPM``, ``DiT``, ``DiffusionUNet``, ``EDMPreconditioner``, ``EulerSampler``, ``RewardTerminationModel``, and ``ActorCriticNetwork``.

DIAMOND exposes ``DiamondAgent`` from ``torchwm.training.train_diamond``; there is no separate ``DIAMONDAgent`` class name in the package.

.. automodule:: torchwm.models.diffusion
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.diffusion.DDPM
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.diffusion.DiT
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.diffusion.diamond_diffusion
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.diffusion.reward_termination
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.models.diffusion.actor_critic
   :members:
   :undoc-members:
   :show-inheritance:

Vision, tokenization, and layers
--------------------------------

**Key classes:** ``ConvEncoder``, ``ConvDecoder``, ``DenseDecoder``, ``ActionDecoder``, ``CNNEncoder``, ``CNNDecoder``, ``IRISEncoder``, ``IRISDecoder``, ``DiscreteAutoencoder``, ``VectorQuantizer``, ``VectorQuantizerEMA``, ``VideoTokenizer``, ``MultiHeadSelfAttention``, and ``STTransformer``.

.. automodule:: torchwm.vision.VAE.ConvVAE
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.vision.dreamer_encoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.vision.dreamer_decoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.vision.planet_encoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.vision.planet_decoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.vision.iris_encoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.vision.iris_decoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.vision.vq_layer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.vision.video_tokenizer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.blocks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.blocks.mhsa
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.blocks.st_transformer
   :members:
   :undoc-members:
   :show-inheritance:

Configuration objects
---------------------

.. automodule:: torchwm.configs
   :no-index:

.. automodule:: torchwm.configs.wm_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.configs.dreamer_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.configs.jepa_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.configs.iris_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.configs.genie_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.configs.dit_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.configs.diamond_config
   :members:
   :undoc-members:
   :show-inheritance:

Training entry points
---------------------

**Key classes and functions:** ``DiamondAgent``, ``train_diamond``, ``train_dreamer``, ``GenieTrainer``, ``IRISTrainer``, and related training entry points.

.. automodule:: torchwm.training
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.training.train_world_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.training.train_convvae
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.training.train_mdn_rnn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.training.train_controller
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.training.train_jepa
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.training.train_iris
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.training.train_genie
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.training.train_planet
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.training.train_rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.training.train_diamond
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.training.rl_harness
   :members:
   :undoc-members:
   :show-inheritance:

Memory and controllers
----------------------

.. automodule:: torchwm.memory.dreamer_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.memory.planet_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.memory.iris_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.controller.rssm_policy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.controller.iris_policy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.controller.rollout_generator
   :members:
   :undoc-members:
   :show-inheritance:

Datasets, environments, and transforms
--------------------------------------

Environment adapters
~~~~~~~~~~~~~~~~~~~~

The environment APIs below mirror the dedicated environment guide pages: DMC,
DeepMind Lab, Gym/Gymnasium, Atari/ALE, Procgen, MuJoCo, Unity ML-Agents, and vectorization utilities.
DIAMOND-style Atari support is intentionally not listed as an environment
adapter because it is Atari preprocessing rather than a separate environment
family.

.. automodule:: torchwm.envs
   :no-index:

.. automodule:: torchwm.envs.dmc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.envs.dmlab
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.envs.gym_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.envs.ale_atari_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.envs.ale_atari_vector_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.envs.procgen_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.envs.mujoco_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.envs.robotics_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.envs.unity_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.envs.vector_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.envs.wrappers
   :members:
   :undoc-members:
   :show-inheritance:

Atari preprocessing helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These helpers wrap Atari environments for specific training recipes. They are
not separate environment families.

.. automodule:: torchwm.envs.diamond_atari
   :members:
   :undoc-members:
   :show-inheritance:

Datasets and transforms
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: torchwm.datasets.wm_dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.datasets.video_datasets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.datasets.tinyworlds
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.datasets.diamond_dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.datasets.cifar10
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.datasets.imagenet1k
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.datasets.nuplan
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.transforms.image
   :members:
   :undoc-members:
   :show-inheritance:

Masking and JEPA helpers
------------------------

.. automodule:: torchwm.masks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.masks.default
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.masks.multiblock
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.masks.random
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.helpers.jepa_helper
   :members:
   :undoc-members:
   :show-inheritance:

Benchmarks and reports
----------------------

.. automodule:: torchwm.benchmarks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.benchmarks.runner
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.benchmarks.adapters
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.benchmarks.metrics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.benchmarks.reporting
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: torchwm.losses.convae_loss
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.losses.gmm_loss
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.utils.train_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.utils.dreamer_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.utils.jepa_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.utils.data_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.utils.jit_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.utils.memory_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.utils.logging_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchwm.utils.utils
   :members:
   :undoc-members:
   :show-inheritance:
