import argparse

import pytest

pytest.importorskip("torch")

from scripts.benchmark_vector_env_speed import (
    SyntheticImageEnv,
    benchmark_single_threaded,
    benchmark_vectorized,
    run_benchmark,
)


def test_synthetic_env_benchmarks_report_throughput():
    single = benchmark_single_threaded(
        lambda: SyntheticImageEnv(), total_envs=2, steps=2
    )
    vector = benchmark_vectorized(
        lambda: SyntheticImageEnv(),
        num_workers=1,
        envs_per_worker=2,
        steps=2,
        seed=0,
    )

    assert single.total_envs == 2
    assert vector.total_envs == 2
    assert single.env_steps_per_second > 0
    assert vector.env_steps_per_second > 0


def test_run_benchmark_computes_speedup():
    args = argparse.Namespace(
        env_factory=None,
        image_size=8,
        episode_length=10,
        num_workers=1,
        envs_per_worker=1,
        steps=1,
        seed=0,
    )

    report = run_benchmark(args)

    assert report.single_threaded.name == "single_threaded"
    assert report.vectorized.name == "vectorized"
    assert report.speedup > 0
