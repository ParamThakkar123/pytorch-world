Benchmarking utilities for TorchWM
================================

This folder contains a small benchmarking harness to run standardized evaluations
for different world-model agents (DIAMOND, IRIS, DreamerV1/V2) and export
results in formats suitable for research reports (JSON/CSV/Markdown).

Quickstart
----------

1. Run from the CLI (example):

   ```bash
   # run IRIS on Pong for 3 episodes using seed 0
   python -m world_models.benchmarks.cli --agent iris --game ALE/Pong-v5 --seeds 1 --episodes 3
   
   # run DIAMOND on Breakout with two seeds
   python -m world_models.benchmarks.cli --agent diamond --game Breakout-v5 --seeds 0,1 --episodes 5
   
   # run ALL agents on the same game for comparison
   python -m world_models.benchmarks.cli --all-agents --game ALE/Pong-v5 --seeds 2 --episodes 5
   ```

2. Outputs are written to `results/bench` by default (change with `--out_dir`).

Design Notes
------------

- Adapters normalize each agent's `evaluate()` output into a dict with
  `episode_returns: List[float]`. Optional keys include `videos` and `latents`.
- `runner.py` orchestrates per-seed evaluations and exports JSON/CSV/Markdown.
- `metrics.py` contains basic aggregate computations (mean, median, IQM) and a
  simple bootstrap CI helper.

Extending
---------

- Add a new adapter in `adapters.py` implementing `load_checkpoint()` and
  `evaluate(num_episodes, render)` that returns standardized outputs.
- Register the adapter in `cli.py` to expose it via the command-line.

CI / Tests
----------

- Add smoke tests under `world_models/benchmarks/tests/` to validate adapters
  without running full training (the repo contains a `mocking_classes.py` file
  you can use to create lightweight fake agents/environments for CI).

License
-------

Same license as the project: see top-level `LICENSE`.

Vectorized environment speedups
-------------------------------

Use the top-level script below to compare `TorchVectorizedEnv` throughput against
a single-threaded environment loop and optionally collect `cProfile` bottleneck
statistics:

```bash
python scripts/benchmark_vector_env_speed.py \
  --num-workers 2 \
  --envs-per-worker 4 \
  --steps 500 \
  --out results/bench/vector_env_speed.json \
  --profile results/bench/vector_env_speed.prof \
  --profile-text results/bench/vector_env_profile.txt
```

The default synthetic image environment keeps this benchmark dependency-light.
Pass `--env-factory module:callable` to benchmark a project-specific environment
factory with the same single-threaded versus vectorized methodology.
