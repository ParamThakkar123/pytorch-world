# TorchWM demo runbook

Scripts for producing demo material: train a model, then record video of it
acting and dreaming, or visualise its representations. Everything here wraps
the existing entrypoints — nothing in `torchwm/` or `scripts/` is modified.

## 0. Environment

TorchWM does not pin a PyTorch build. Install the CUDA wheel that matches your
driver before training, or every run silently falls back to CPU:

```bash
# Check what you actually have
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# CUDA 12.6 wheels (pick your own index from pytorch.org/get-started/locally)
uv pip install --reinstall-package torch --reinstall-package torchvision \
    torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

The `torch` wheel is ~2.4 GB. If `uv` reports a network timeout, raise it:
`UV_HTTP_TIMEOUT=900`.

Optional extras the demo paths use:

| Package | Needed for |
|---|---|
| `omegaconf` | `train_iris`, `scripts/smoke_train.py` |
| `tensorboard` | `--tensorboard` on Dreamer runs |
| `dm_control` | `walker-walk` and other DMC tasks (`pip install torchwm[dmc]`) |
| `ale-py` | Atari for DIAMOND/IRIS (already installed) |

## 1. Train

`train_demo.py` runs the stock trainers but overrides the artifact intervals.
This matters: `DreamerConfig` defaults to `checkpoint_interval=10000`,
`test_interval=10000`, `log_video_freq=-1` and `seed_steps=5000`, so a short run
finishes having collected only random seed data and written **no checkpoint and
no video**. The launcher scales those intervals to the run length instead.

```bash
# Dreamer on a Gym task — the lightest end-to-end run
python demos/train_demo.py --algo dreamer --env Pendulum-v1 --env-backend gym

# Dreamer on DeepMind Control (needs the dmc extra)
python demos/train_demo.py --algo dreamer --env walker-walk --env-backend dmc --steps 500000

# DIAMOND on Atari. Use --preset small on GPUs under ~8 GB of VRAM.
python demos/train_demo.py --algo diamond --env Breakout-v5 --preset small --batch-size 16

# IRIS on Atari
python demos/train_demo.py --algo iris --env ALE/Pong-v5 --steps 100

# Genie on TinyWorlds (video prediction)
python demos/train_demo.py --algo genie --steps 5000

# I-JEPA on CIFAR-10 (self-supervised representation learning)
python demos/train_demo.py --algo ijepa --steps 5 --batch-size 32

# See the exact underlying command without running it
python demos/train_demo.py --algo diamond --dry-run

# Forward any extra config override verbatim
python demos/train_demo.py --algo dreamer -- total_steps=1000000 seed=3 use_amp=True
```

Where checkpoints land:

| Algorithm | Path |
|---|---|---|
| Dreamer | `runs/<env>_<algo>_<name>_<timestamp>/ckpts/<step>_ckpt.pt` |
| DIAMOND | `checkpoints/diamond/checkpoint_<epoch>.pt` |
| IRIS | `checkpoints/iris/checkpoint_<epoch>.pt` |
| Genie | `checkpoints/genie_<dataset>_final.pt` (set via `scripts/train_genie_tinyworlds.py`) |
| I-JEPA | `results/jepa_demo/jepa_run-latest.pth.tar` |

Dreamer also writes `metrics.jsonl` and `config.yaml` into its run directory,
which is what you want for plotting learning curves in a slide.

## 1b. State of the checkpoints already in this repo

**Every checkpoint committed to this repo predates the current code and will not
load cleanly.** Verified against `torch 2.13.0+cu126`:

| Checkpoint | Result |
|---|---|
| `checkpoints/diamond/checkpoint_0.pt` | ✗ `RewardTerminationModel` was refactored (`conv_blocks.*` → `stages.*.*`) |
| `checkpoints/diamond/checkpoint_1.pt` | ✗ `DiamondConfig` no longer accepts `num_seeds` |
| `checkpoints/diamond/checkpoint_2.pt` | ✗ same `num_seeds` failure |
| `checkpoints/diamond/ckpt.pt` | ✗ embeds a pickled config; `load_checkpoint` uses `weights_only=True` by design |
| `checkpoints/iris/checkpoint_0.pt` | ~ policy loads via `record_iris.py`; decoder does not (see below) |
| `checkpoints/genie_sonic_final.pt` | ? unpickles, but there is no Genie `play`/`eval` entrypoint |

The IRIS checkpoint is the only one that yields a runnable demo, and only
partially:

- `IRISAgent.load` fails because `IRISDecoder` gained an `index_to_embedding`
  parameter after the checkpoint was written.
- The **policy** path (`forward_actor_critic` → `cnn` → `lstm` → `actor_head`)
  never touches the decoder or tokenizer, so `demos/record_iris.py` loads just
  those four components and records real gameplay.
- It also has to correct the architecture: the checkpoint was trained with
  `actor_layers=4`, but `IRISConfig` now defaults to `1`. The script reads the
  layer count off the LSTM tensors rather than trusting the defaults.
- Caveat: the checkpoint is from **epoch 0, global step 200** — effectively an
  untrained policy (it scores about -10 on Pong). It proves the inference
  pipeline end to end; it is not an impressive result to show anyone.

**Conclusion: a presentable demo needs a fresh training run** (section 1). Once
you have a current checkpoint, sections 2-4 work as written.

## 2. Record inference video (headless)

`scripts/play_diamond.py` and `scripts/play_dreamer.py` are **interactive**:
they open a `cv2` window and run until you press `Q`. That will not work over
SSH on a GPU box, and it cannot produce a fixed-length clip.

`record_diamond.py` drives the same agent non-interactively for a set number of
steps and writes videos to disk:

```bash
python demos/record_diamond.py \
    --checkpoint checkpoints/diamond/checkpoint_0.pt \
    --game Breakout-v5 \
    --steps 300 --dream-steps 100 --scale 4 \
    --out-dir demos/out
```

Outputs `real.mp4`, `dream.mp4` and `side_by_side.mp4` (real | dream). The
side-by-side clip is the single most demo-legible artifact: the same policy,
acting in the real emulator on the left and inside the learned diffusion world
model on the right.

For IRIS, `record_iris.py` records the policy playing the real game at full
Atari resolution:

```bash
python demos/record_iris.py -c checkpoints/iris/checkpoint_0.pt \
    --game ALE/Pong-v5 --episodes 2 --out demos/out/iris_pong.mp4
```

It refuses to record if any policy component fails to load, rather than writing
a video of a half-initialised network.

**DiT** — `record_dit.py` samples images from a trained Diffusion Transformer,
saving a static grid and a denoising trajectory video:

```bash
python demos/record_dit.py -c dit_demo/dit_model.pth --samples 64 --ddim-steps 100
python demos/record_dit.py --random-init                              # pipeline check
```

**Genie** — `record_genie.py` generates video frames from a single prompt frame
using a trained Genie checkpoint, saving a grid and a video:

```bash
python demos/record_genie.py -c checkpoints/genie_sonic_final.pt --num-frames 32
python demos/record_genie.py --random-init --num-frames 8            # pipeline check
```

**I-JEPA** — `record_jepa.py` visualises mask-target prediction from a trained
JEPA encoder+predictor checkpoint, saving a masked-input visualisation and a
similarity heatmap:

```bash
python demos/record_jepa.py -c results/jepa_demo/jepa_run-latest.pth.tar
python demos/record_jepa.py --random-init                            # pipeline check
```

Notes:
- Diffusion sampling is slow. On CPU expect roughly 1 frame/sec, so keep
  `--dream-steps` small (or `0`) unless you are on a GPU.
- Frames are 64x64; `--scale 4` upscales with nearest-neighbour for legibility.
- Recording continues across episode boundaries so the clip is always the
  requested length.

## 3. Interactive demo (needs a desktop session)

For a live, driven-by-hand demo, use the built-in CLI. `TAB` toggles
REAL/DREAM, arrow keys or WASD take control from the policy, `Q` quits.

```bash
torchwm play --model diamond -c checkpoints/diamond/checkpoint_0.pt \
    --game Breakout-v5 --record demos/out/interactive.mp4

torchwm play --model dreamer -c runs/<run>/ckpts/<step>_ckpt.pt --game walker-walk
```

`torchwm play --model dreamer` defaults to `walker-walk`, which requires
`dm_control`. Pass a `--game` your checkpoint was trained on.

## 4. Quantitative results

```bash
# FID / FVD / LPIPS for a DIAMOND world model, plus a real-vs-generated video
torchwm eval --model diamond -c checkpoints/diamond/checkpoint_0.pt \
    --game Breakout-v5 --num-videos 64 --trajectory-length 20 \
    --record demos/out/eval.mp4 --output demos/out/eval.json

# Episode-return benchmark, writes a report to results/bench
torchwm benchmark --agent diamond --game ALE/Breakout-v5 \
    --checkpoint checkpoints/diamond/checkpoint_0.pt --episodes 10 --seeds 3
```

`torchwm benchmark --agent` requires `--checkpoint`; only trained models are
benchmarked. Use `--all-agents` with repeated `--checkpoint-map AGENT=PATH` to
compare adapters on one environment.

## Bugs fixed while building this runbook

All of these were found by actually running the demo paths, and are now fixed in
the repo.

1. **`torchwm play --model diamond` failed with a self-contradictory
   `TypeError: config must be a DiamondConfig ...; got DiamondConfig`.**
   `torchwm/__init__.py` *appended* `_SubmoduleAliasFinder` to `sys.meta_path`.
   Aliasing `torchwm.configs` returns the same module object as
   `torchwm.configs`, whose `__path__` points into `torchwm/`, so the
   default `FileFinder` won for the submodule and executed `diamond_config.py` a
   second time under the name `torchwm.configs.diamond_config` — two distinct
   classes from one file. Fixed by inserting the finder at the *front* of
   `sys.meta_path` and having it decline names backed by a real file under
   `torchwm/` (so `torchwm.cli` still loads normally).
2. **`make_diamond_atari_env` rejected bare game ids.** It forwarded `game`
   straight to `gym.make`, so the `Breakout-v5` stored in every DIAMOND
   checkpoint raised `NameNotFound`. It now accepts either form.
3. **The DIAMOND path never registered the ALE envs**, failing with
   `NamespaceNotFound: Namespace ALE not found`. `make_diamond_atari_env` now
   calls `gym.register_envs(ale_py)`.
4. **`DiamondAtariWrapper` had no `close()`** (nor `render()`); both are now
   forwarded to the wrapped env.
5. **`train_iris`'s `game`/`device`/`seed`/`epochs`/`save_dir` options were
   unreachable.** They were read via `OmegaConf.from_cli` *after*
   `update_config_object(..., strict=True)` had already rejected them as unknown
   `IRISConfig` fields. They are now split out before the config is composed.
   `device` also defaults to CUDA-if-available instead of a hardcoded `"cuda"`.
6. **`torchwm train iris` was a silent no-op.** `train_iris.py` was the only
   trainer missing an `if __name__ == "__main__"` guard, so the CLI's subprocess
   launch ran nothing and exited 0. Guard added.

## Remaining rough edges

- **`torchwm play` requires a display.** Use `demos/record_diamond.py` or
  `demos/record_iris.py` on headless machines.
- **Only DIAMOND has an `eval` entrypoint** (`EVAL_MODULES` in `torchwm/cli.py`
  maps `diamond` only), and `play` covers only `diamond` and `dreamer`.
  `demos/record_genie.py` adds a headless Genie demo path for pre-trained
  checkpoints, and `demos/record_jepa.py` visualises I-JEPA mask prediction.
- **`scripts/smoke_train.py` needs `omegaconf`**, which is not in the base
  dependencies. (`train_iris` no longer requires it.)
