PYTHON ?= python
export PYTHONPYCACHEPREFIX ?= build/__pycache__

.PHONY: test lint format bench bench-all bench-infer bench-play

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

# Compute benchmark (params, latency, throughput, memory) over every model.
# The shell driver installs everything with uv and runs the sweep with uv run.
# Pass extra flags with BENCH_ARGS, e.g. `make bench BENCH_ARGS="--preset small"`.
bench:
	bash scripts/benchmark_models.sh $(BENCH_ARGS)

bench-all:
	bash scripts/benchmark_models.sh --all --preset small $(BENCH_ARGS)

# Record a video of a trained model playing / generating.
# Example: make bench-infer BENCH_ARGS="--model diamond -c ckpt.pt --game Breakout-v5"
bench-infer:
	bash scripts/benchmark_models.sh --infer $(BENCH_ARGS)

# Interactive play (needs a display). Add --versus to play against the policy.
# Example: make bench-play BENCH_ARGS="--model iris -c ckpt.pt --versus"
bench-play:
	bash scripts/benchmark_models.sh --play $(BENCH_ARGS)

