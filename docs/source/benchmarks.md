# Benchmarking World Models

TorchWM includes a lightweight benchmark harness for running standardized
evaluations of trained world-model agents and exporting results that can be used
in experiment logs, reports, and papers.

## Quick Overview

- Preferred CLI entrypoint: `torchwm benchmark`.
- Benchmark adapters live in the TorchWM source tree under `world_models/benchmarks/`.

## Supported adapters

The benchmark CLI currently registers these adapters out of the box:

- `diamond` - DIAMOND diffusion world-model agent
- `iris` - IRIS transformer-based agent
- `dreamerv1` / `dreamerv2` - Dreamer family

Benchmarks are intended for trained models. For single-agent runs, pass a
checkpoint with `--checkpoint`. For multi-agent runs, pass one or more
`--checkpoint-map AGENT=PATH` values, or use `--train-epochs` when you
intentionally want the CLI to train before evaluating.

## TorchWM CLI examples

Run IRIS on Pong for 3 episodes using seed 0 and write the standard result files
to `results/bench`:

```bash
torchwm benchmark \
  --agent iris \
  --game ALE/Pong-v5 \
  --checkpoint checkpoints/iris/pong.pt \
  --seeds 1 \
  --episodes 3
```

Run DIAMOND on Breakout with two explicit seeds and 5 episodes per seed:

```bash
torchwm benchmark \
  --agent diamond \
  --game Breakout-v5 \
  --checkpoint checkpoints/diamond/breakout.pt \
  --seeds 0,1 \
  --episodes 5 \
  --out-dir results/diamond_breakout
```

Run DreamerV2 on a Gym environment:

```bash
torchwm benchmark \
  --agent dreamerv2 \
  --game Pong-v5 \
  --checkpoint checkpoints/dreamerv2/pong.pt \
  --seeds 1 \
  --episodes 10 \
  --device cpu
```


Run DreamerV1 on a DeepMind BSuite diagnostic task:

```bash
torchwm benchmark \
  --agent dreamerv1 \
  --game catch/0 \
  --env-backend bsuite \
  --checkpoint checkpoints/dreamerv1/bsuite_catch.pt \
  --seeds 0,1 \
  --episodes 10 \
  --device cpu
```

The BSuite backend is optional. Install it with `pip install torchwm[bsuite]` or
`pip install bsuite` before running BSuite tasks. TorchWM wraps BSuite's compact
`dm_env` observations as synthetic RGB images so the existing pixel-based
Dreamer benchmark path can evaluate trained world-model agents.

Run all registered adapters on the same game with per-agent checkpoints:

```bash
torchwm benchmark \
  --all-agents \
  --game ALE/Pong-v5 \
  --checkpoint-map iris=checkpoints/iris/pong.pt \
  --checkpoint-map diamond=checkpoints/diamond/pong.pt \
  --checkpoint-map dreamerv1=checkpoints/dreamerv1/pong.pt \
  --checkpoint-map dreamerv2=checkpoints/dreamerv2/pong.pt \
  --seeds 0,1 \
  --episodes 5 \
  --out-dir results/pong_comparison
```

## CLI options

Common `torchwm benchmark` options:

- `--agent AGENT` / `-a AGENT`: run one adapter (`iris`, `diamond`, `dreamerv1`, or `dreamerv2`).
- `--all-agents`: run every registered adapter on the same environment.
- `--game GAME` / `-g GAME`: environment id, such as `ALE/Pong-v5` or a BSuite id like `catch/0`.
- `--checkpoint PATH` / `-c PATH`: checkpoint path for single-agent benchmarks.
- `--checkpoint-map AGENT=PATH`: repeatable per-agent checkpoint mapping for `--all-agents`.
- `--env-backend BACKEND`: optional backend hint forwarded to adapters. Use `bsuite` for DeepMind BSuite ids such as `catch/0`.
- `--seeds SEEDS`: either `N` for seeds `0..N-1`, or a comma-separated list such as `0,1,2`.
- `--episodes N` / `-n N`: number of evaluation episodes per seed.
- `--out-dir DIR`: output directory for report artifacts. The legacy alias `--out_dir` is also accepted.
- `--device DEVICE`: device forwarded to adapters. Defaults to CUDA when available, otherwise CPU.
- `--preset PRESET`: optional adapter/model preset.
- `--train-epochs N`: for `--all-agents`, train first when checkpoint maps are not supplied.

You can also run `torchwm benchmark --help` to see the installed CLI help.

## Python usage

For benchmark runs, prefer the main TorchWM CLI so commands are consistent with
the rest of the package:

```bash
torchwm benchmark --agent iris --game ALE/Pong-v5 --checkpoint checkpoints/iris/pong.pt
```

The command writes benchmark JSON reports under the configured output
directory. Load those reports with standard Python tools when you need custom
analysis:

```py
import json

res = json.load(open("results/bench/benchmark_results.json"))
per_seed = res["aggregate"]["per_seed_means"]
print(per_seed)
```

## Running the Atari 100k benchmark

To run the full Atari 100k benchmark on all configured games with the
centralized benchmark module:

```bash
python -m world_models.benchmarks.atari_100k --benchmark
```

This runs the Atari 100k evaluator from `world_models/benchmarks`, computes
human-normalized scores, and reports aggregate metrics across games and seeds.

## Vectorized environment speed and bottleneck profiling

Use `scripts/benchmark_vector_env_speed.py` to compare TorchWM's multiprocessing
`TorchVectorizedEnv` against a single-threaded loop with the same total number
of environments. The script reports elapsed time, environment steps per second,
and the vectorized speedup ratio.

```bash
python scripts/benchmark_vector_env_speed.py \
  --num-workers 2 \
  --envs-per-worker 4 \
  --steps 500 \
  --out results/bench/vector_env_speed.json
```

By default, the benchmark uses a deterministic synthetic image environment so it
can run without optional Gym/Atari dependencies. To benchmark a real environment,
pass an importable factory in `module:callable` form:

```bash
python scripts/benchmark_vector_env_speed.py \
  --env-factory my_package.envs:make_env \
  --num-workers 4 \
  --envs-per-worker 2 \
  --steps 1000
```

For bottleneck analysis, enable `cProfile` output. The binary `.prof` file can
be inspected with tools such as `snakeviz`, while `--profile-text` writes the top
cumulative call sites for quick terminal review.

```bash
python scripts/benchmark_vector_env_speed.py \
  --steps 500 \
  --profile results/bench/vector_env_speed.prof \
  --profile-text results/bench/vector_env_profile.txt
```

## Outputs

The runner saves these files into the selected `out_dir` (default
`results/bench`):

- `benchmark_results.json` - raw structured results.
- `benchmark_results.csv` - per-seed rows.
- `benchmark_results.md` - human-readable markdown table.
- `benchmark_results.tex` - LaTeX table ready for papers.

Multi-agent runs also write combined reports such as
`combined_benchmark_results.json` and `combined_benchmark_results.csv` in the
root output directory, with per-agent details under subdirectories.

## Computing IQM and bootstrap CIs

The runner stores per-seed means in the JSON under
`aggregate.per_seed_means`. Use your preferred statistics package to compute
IQM and confidence intervals from that array.

## Extending the harness

- Create an adapter in `world_models/benchmarks/adapters.py` that implements:
  - `load_checkpoint(path: str)`
  - `evaluate(num_episodes: int, render: bool = False)` returning `{"episode_returns": list[float]}`
- Register your adapter in `world_models/benchmarks/cli.py` to expose it through `torchwm benchmark`.

## Tests and CI

- Place smoke tests under `world_models/benchmarks/tests/` so CI can run them quickly.
- The repo contains a `mocking_classes.py` helper for building fake agents and environments for fast unit tests.

## Where to start

- Run the examples in `examples/benchmark_iris.py` or use `torchwm benchmark` directly.
- If you need help wiring specific agent configs, use `--device`, `--preset`, and checkpoint options, or call the runner programmatically and pass `extra_kwargs`.
