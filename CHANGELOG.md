# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.0] — 2026-07-27

### Added
- Real PEP 561 typing stub: `torchwm/__init__.pyi` now declares every public
  export instead of falling back to `Any`, generated from the export map by
  `python -m tools.gen_type_stub` and verified in CI. `world_models` ships a
  `py.typed` marker so the re-exports stay typed
- `torchwm play -m dreamer` — interactive REAL/DREAM playback for Dreamer
  checkpoints alongside DIAMOND (`scripts/play_dreamer.py`)
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1)
- CI job running the full test suite with every optional backend installed —
  the configuration a contributor gets from `CONTRIBUTING.md`, previously
  untested
- Tests asserting every `__all__` entry resolves, that the stub matches the
  export map, and that the README algorithms table matches the model registry
- Complete `torchwm` public import surface: every implementation submodule is now
  reachable through the friendly namespace (`from torchwm.models import Dreamer`,
  `import torchwm.envs`, ...), not just top-level factory helpers. The internal
  `world_models` package remains importable for backward compatibility
- `dmc` optional-dependency extra (`pip install torchwm[dmc]`) that installs
  `dm-control` for the default DeepMind Control backend
- Actionable error from the DMC backend that names the missing `dm_control`
  dependency and points to `torchwm[dmc]` or the gym backend

### Changed
- README "Supported Algorithms" table now lists all 13 registered models
  (previously 5), keyed by the name `create_model()` accepts
- `viz` extra installs only what it uses (opencv, umap-learn, scikit-learn,
  plotly); the unused FastAPI/uvicorn/starlette/python-multipart entries and the
  duplicated docs dependencies are gone
- CI runs the test matrix on `push` to `main` only, not on every push to a PR
  branch, which ran the whole matrix twice per commit
- Documentation, examples, and scripts now import through the `torchwm` public
  namespace instead of the internal `world_models` package
- Quick-start examples (README, docs landing page, getting-started guide) now use
  the base-installable `Pendulum-v1` gym backend so they run on
  `pip install torchwm[gym]` out of the box; DMC usage is documented separately

### Fixed
- `torchwm.DreamerRSSM` and `torchwm.MujocoEnv` raised `AttributeError` on
  access: the export map pointed `DreamerRSSM` at a module that names the class
  `RSSM`, and `MujocoEnv` had no implementation behind it (it now resolves to
  `MuJoCoImageEnv`)
- README advertised a FastAPI visualization feature that does not exist
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
