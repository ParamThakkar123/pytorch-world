#!/usr/bin/env python3
"""Benchmark TorchWM vectorized environment speedups.

Compares a single-threaded list of environments against ``TorchVectorizedEnv``
using the same number of total environments and steps. The default synthetic
image environment is dependency-light and deterministic, which makes it useful
for CI and for profiling framework overhead.
"""

from __future__ import annotations

import argparse
import cProfile
import importlib
import json
import pstats
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from world_models.envs.vector_env import TorchVectorizedEnv


class DiscreteActionSpace:
    """Minimal action-space shim used by the synthetic benchmark env."""

    def __init__(self, n: int) -> None:
        self.n = n

    def sample(self) -> int:
        return int(np.random.randint(self.n))


class BoxObservationSpace:
    """Minimal observation-space shim used by the synthetic benchmark env."""

    def __init__(self, shape: tuple[int, ...], dtype: Any = np.uint8) -> None:
        self.shape = shape
        self.dtype = dtype


class SyntheticImageEnv:
    """Small deterministic image env for vectorization throughput benchmarks."""

    def __init__(self, episode_length: int = 1_000, image_size: int = 64) -> None:
        self.episode_length = episode_length
        self.image_size = image_size
        self.observation_space = {
            "image": BoxObservationSpace((3, image_size, image_size), np.uint8)
        }
        self.action_space = DiscreteActionSpace(4)
        self.seed_value = 0
        self.step_count = 0

    def seed(self, seed: int) -> None:
        self.seed_value = int(seed)

    def _obs(self) -> dict[str, np.ndarray]:
        value = (self.seed_value + self.step_count) % 256
        return {
            "image": np.full(
                (3, self.image_size, self.image_size), value, dtype=np.uint8
            )
        }

    def reset(self) -> dict[str, np.ndarray]:
        self.step_count = 0
        return self._obs()

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        self.step_count += 1
        done = self.step_count >= self.episode_length
        return (
            self._obs(),
            float(np.asarray(action).item()),
            done,
            {"step": self.step_count},
        )

    def render(self) -> np.ndarray:
        return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class ThroughputResult:
    name: str
    total_envs: int
    steps: int
    elapsed_seconds: float
    env_steps_per_second: float


@dataclass(frozen=True)
class BenchmarkReport:
    single_threaded: ThroughputResult
    vectorized: ThroughputResult
    speedup: float


def load_factory(
    factory_path: str | None, image_size: int, episode_length: int
) -> Callable[[], Any]:
    """Load ``module:callable`` env factory or return the synthetic env factory."""
    if factory_path is None:
        return lambda: SyntheticImageEnv(
            episode_length=episode_length, image_size=image_size
        )

    module_name, sep, attr_name = factory_path.partition(":")
    if not sep:
        raise ValueError("--env-factory must use the 'module:callable' format")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    if not callable(factory):
        raise TypeError(f"{factory_path!r} is not callable")
    return factory


def benchmark_single_threaded(
    env_factory: Callable[[], Any], total_envs: int, steps: int
) -> ThroughputResult:
    envs = [env_factory() for _ in range(total_envs)]
    try:
        for env in envs:
            env.reset()
        start = time.perf_counter()
        for _ in range(steps):
            for env in envs:
                env.step(0)
        elapsed = time.perf_counter() - start
    finally:
        for env in envs:
            if hasattr(env, "close"):
                env.close()
    env_steps = total_envs * steps
    return ThroughputResult(
        "single_threaded", total_envs, steps, elapsed, env_steps / elapsed
    )


def benchmark_vectorized(
    env_factory: Callable[[], Any],
    num_workers: int,
    envs_per_worker: int,
    steps: int,
    seed: int,
) -> ThroughputResult:
    vec_env = TorchVectorizedEnv(
        env_factory,
        num_workers=num_workers,
        envs_per_worker=envs_per_worker,
        seed=seed,
    )
    try:
        vec_env.reset_batch()
        actions = torch.zeros(vec_env.total_envs, dtype=torch.long)
        start = time.perf_counter()
        for _ in range(steps):
            vec_env.step_batch(actions)
        elapsed = time.perf_counter() - start
        env_steps = vec_env.total_envs * steps
        return ThroughputResult(
            "vectorized", vec_env.total_envs, steps, elapsed, env_steps / elapsed
        )
    finally:
        vec_env.close()


def run_benchmark(args: argparse.Namespace) -> BenchmarkReport:
    env_factory = load_factory(args.env_factory, args.image_size, args.episode_length)
    total_envs = args.num_workers * args.envs_per_worker
    single = benchmark_single_threaded(env_factory, total_envs, args.steps)
    vector = benchmark_vectorized(
        env_factory, args.num_workers, args.envs_per_worker, args.steps, args.seed
    )
    return BenchmarkReport(
        single, vector, vector.env_steps_per_second / single.env_steps_per_second
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-factory",
        help="Optional 'module:callable' env factory. Defaults to SyntheticImageEnv.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--envs-per-worker", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--episode-length", type=int, default=1_000)
    parser.add_argument(
        "--out", type=Path, default=Path("results/bench/vector_env_speed.json")
    )
    parser.add_argument(
        "--profile", type=Path, help="Write a cProfile .prof file for bottleneck analysis."
    )
    parser.add_argument(
        "--profile-text", type=Path, help="Write top cumulative cProfile stats as text."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile:
        profiler = cProfile.Profile()
        report = profiler.runcall(run_benchmark, args)
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.profile)
        if args.profile_text:
            args.profile_text.parent.mkdir(parents=True, exist_ok=True)
            with args.profile_text.open("w") as fh:
                stats = pstats.Stats(profiler, stream=fh).sort_stats("cumulative")
                stats.print_stats(30)
    else:
        report = run_benchmark(args)

    payload = asdict(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
