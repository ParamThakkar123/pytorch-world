# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **DreamerV3** (`create_model("dreamer-v3")`), replacing the placeholder that
  previously aliased `dreamer-v3` to the DreamerV1 agent. Implements the paper's
  robustness techniques: categorical latents with 1% unimix and straight-through
  gradients, separate dynamics/representation KL terms with free bits, symexp
  two-hot reward and critic heads, percentile return normalization with a
  denominator limit, a critic EMA regularizer and replay loss, zero-initialized
  head outputs, and LaProp with adaptive gradient clipping
- `DreamerV3Config` with the paper's Table 4 defaults and the 12M-400M model-size
  presets from Table 3; widths resolve from `model_size` and `update_steps`
  derives from `replay_ratio`
- `CategoricalRSSM`, `DreamerV3Encoder`/`Decoder`/`Head`/`Actor`,
  `DreamerV3ReplayBuffer`, `BlockGRUCell`/`BlockLinear`, and a new
  `world_models.optim` package exporting `LaProp` and `adaptive_grad_clip_` --
  all usable independently of the agent
- Discrete action support in the shared Dreamer `make_env`: environments with a
  discrete action space are now wrapped with `OneHotAction` instead of failing in
  `NormalizeActions`
- `examples/dreamer_v3_example.py` and the {doc}`dreamer_v3` documentation page
- `dmc` optional-dependency extra (`pip install torchwm[dmc]`) that installs
  `dm-control` for the default DeepMind Control backend
- Actionable error from the DMC backend that names the missing `dm_control`
  dependency and points to `torchwm[dmc]` or the gym backend

### Changed
- `DreamerAgent` is now subclassable: `_config_cls` selects the configuration
  class and `_build_core` selects the algorithm core, so the shared environment,
  seeding, logging, and checkpoint loop is reused rather than duplicated
- Passing a base `DreamerConfig` to a subclass agent carries over only the fields
  that were actually changed, so the subclass's tuned defaults are not silently
  overwritten by base-class defaults
- The Dreamer training loop logs any extra metrics an algorithm core publishes via
  `last_metrics` (DreamerV3 reports its individual loss terms, gradient norms, and
  return scale)
- Quick-start examples (README, docs landing page, getting-started guide) now use
  the base-installable `Pendulum-v1` gym backend so they run on
  `pip install torchwm[gym]` out of the box; DMC usage is documented separately

### Fixed
- `create_model("dreamer", env="walker-walk")` no longer raises a bare
  `ModuleNotFoundError`; the DMC dependency is installable and the error is clear
- Broken landing-page example that called `train(env_name=..., total_steps=...)`
  (the `train` method only accepts `total_steps`)
- Env-adapter tests patched non-existent module attributes
  (`world_models.models.dreamer.env_wrapper.*`) and the wrong Gymnasium registry
  reference, causing 10 spurious failures
- Removed stray build artifacts (`nul`, empty `testsdata/` directories) and added
  `.gitignore` guards

## [0.4.2] — 2026-06-20

### Added
- Dreamer integration test for Pendulum-v1 wired into CI
- DIAMOND world model documentation
- `from_pretrained` and `from_config` class methods for Dreamer, IRIS, JEPA, Genie agents
- Gymnasium wrapper for world model environment
- ONNX export support for agents
- `py.typed` marker for PEP 561 type declarations
- PSNR evaluation metric

### Changed
- Restructured dreamer docs with separate V1/V2 theory and examples
- Stripped base deps to minimum, moved extras to optional groups
- Centralized version in `_version.py` as single source of truth
- Improved block exports and testing utilities

### Fixed
- Arbitrary code execution risk in pickle.load (hardened checkpoint/replay deserialization)
- Zero-element tensor reshape crash in ConvEncoder
- Empty sequence edge case in Dreamer training
- 25 GitHub Dependabot vulnerabilities (upgraded 8 packages)
- Dockerfile referencing removed `torchwm_ui` folder
- Replaced debug print calls with proper logging
- MyPy type errors across 40+ files
- Missing imports in `train_jepa.py` (mp, F, DistributedDataParallel)
- CI workflow and docs dependency config
- DreamerConfig documentation field sync

## [0.4.1] — 2026-06-01

### Added
- Modular RSSM with swappable LSTM/Transformer/MLP backbones
- Genie model support (video tokenizer, latent action model, dynamics model)
- Brax environment backend
- BSuite environment backend
- DMLab environment backend
- Procgen environment backend
- Robotics environment backend (gymnasium-robotics)
- Unity ML-Agents environment backend
- Sphinx documentation with auto-deploy to GitHub Pages
- Benchmark runners and reporting utilities
- CLI tools (`torchwm`, `torchwm-train`)

### Changed
- Migrated configs to dataclass style for consistency
- Improved lazy import architecture for faster CLI startup

### Fixed
- Cross-platform memory detection (Windows ctypes + Linux /proc/meminfo + psutil fallback)
- Environment wrapper stack consistency across backends

## [0.4.0] — 2026-05-15

### Added
- Initial public release
- Dreamer (V1/V2) agent implementation
- PlaNet agent implementation
- JEPA self-supervised learning agent
- IRIS sample-efficient RL agent
- DiT (Diffusion Transformer) support
- DIAMOND diffusion world model for Atari
- Core environment backends (DMC, Gym, Atari, MuJoCo)
- Replay buffers (Dreamer, IRIS, PlaNet)
- VQ-VAE and ConvVAE vision components
- HuggingFace Hub checkpoint loading
- TensorBoard and WandB logging integration
