"""TorchWM command-line interface.

The installed ``torchwm`` console script and ``python -m torchwm.cli`` both enter through this module. ``tools.cli`` remains as a backward-compatible wrapper.  The CLI intentionally keeps imports lightweight at
startup and only imports optional training, benchmark, dataset, or environment
packages inside the commands that need them.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger("torchwm.cli")

# Keep this mapping cheap to import so ``torchwm models list`` and validation do
# not pull PyTorch or environment packages into every CLI process.
TRAINING_MODULES = {
    "diamond": "world_models.training.train_diamond",
    "dreamer": "world_models.training.train_dreamer",
    "genie": "world_models.training.train_genie",
    "iris": "world_models.training.train_iris",
    "jepa": "world_models.training.train_jepa",
    "planet": "world_models.training.train_planet",
    "rssm": "world_models.training.train_rssm",
    "world-model": "world_models.training.train_world_model",
}

EVAL_MODULES = {
    "diamond": "scripts.eval_diamond",
}

PLAY_MODULES = {
    "diamond": "scripts.play_diamond",
}

BENCHMARK_AGENT_NAMES = ("diamond", "iris", "dreamerv1", "dreamerv2")
BENCHMARK_ENV_BACKENDS = (
    "gym",
    "gymnasium",
    "dmc",
    "mujoco",
    "robotics",
    "bsuite",
    "brax",
    "unity_mlagents",
)


def _echo_error(message: str) -> None:
    """Print user-facing errors to stdout for stable CLI/test output."""
    click.echo(message)


def _parse_benchmark_seeds(value: str) -> list[int]:
    """Parse ``--seeds`` values as either N, a comma list, or one integer."""
    value = value.strip()
    if not value:
        raise ValueError("--seeds cannot be empty")
    if "," in value:
        seeds = [item.strip() for item in value.split(",") if item.strip()]
        if not seeds:
            raise ValueError("--seeds must contain at least one integer")
        return [int(item) for item in seeds]
    if value.isdigit():
        return list(range(int(value)))
    return [int(value)]


def _parse_checkpoint_map(
    values: tuple[str, ...] | list[str] | None,
) -> dict[str, str]:
    """Parse repeated ``--checkpoint-map AGENT=PATH`` options."""
    checkpoints: dict[str, str] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(
                f"Invalid checkpoint map '{value}'. Expected AGENT=PATH, for example iris=checkpoints/iris.pt."
            )
        agent, path = value.split("=", 1)
        agent = agent.strip().lower()
        path = path.strip()
        if not agent or not path:
            raise ValueError(
                f"Invalid checkpoint map '{value}'. Agent and path must both be non-empty."
            )
        checkpoints[agent] = path
    return checkpoints


def _ensure_gym() -> Any:
    """Lazy-import gym, falling back to gymnasium."""
    try:
        return importlib.import_module("gym")
    except Exception:
        return importlib.import_module("gymnasium")


def _ensure_numpy() -> Any | None:
    """Lazy-import numpy and return ``None`` when it is unavailable."""
    try:
        return importlib.import_module("numpy")
    except Exception:
        return None


def _load_catalog() -> dict[str, Any]:
    """Lazy-load the environment catalog used by ``torchwm envs list``."""
    try:
        from world_models.catalog import ENV_BACKENDS
    except Exception:
        logger.debug("Could not import world_models.catalog", exc_info=True)
        return {}
    return ENV_BACKENDS


def _load_benchmark_runtime() -> tuple[Any, Any, Any, Any]:
    """Lazy-load benchmark classes after lightweight CLI validation passes."""
    from world_models.benchmarks.cli import AGENTS
    from world_models.benchmarks.runner import (
        BenchmarkRunner,
        MultiAgentBenchmarkRunner,
    )
    import torch

    return AGENTS, BenchmarkRunner, MultiAgentBenchmarkRunner, torch


@click.group(
    name="torchwm",
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.pass_context
def app(ctx: click.Context) -> None:
    """TorchWM command-line tool."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@app.command("version")
def version() -> None:
    """Show the installed TorchWM package version."""
    try:
        import torchwm as pkg
    except Exception:
        click.echo("torchwm (unknown version)")
        return
    click.echo(getattr(pkg, "__version__", "torchwm (unknown version)"))


@app.group("envs")
def envs_app() -> None:
    """Environment discovery commands."""


@envs_app.command("list")
def envs_list() -> None:
    """List built-in environment backends and example environment ids."""
    catalog = _load_catalog()
    if not catalog:
        click.echo(
            "No environment catalog available (world_models.catalog failed to import)."
        )
        return

    for backend, info in catalog.items():
        click.echo(f"{backend}: {info.get('label', '')}")
        items = info.get("environments", [])
        for env in items[:20]:
            click.echo(f"  - {env}")
        if len(items) > 20:
            click.echo(f"  ... and {len(items) - 20} more")


@app.group("datasets")
def datasets_app() -> None:
    """Dataset inspection and conversion commands."""


@datasets_app.command("list")
@click.argument("path", required=False, type=click.Path(path_type=Path))
def datasets_list(path: Path | None = None) -> None:
    """List dataset entries under PATH, TORCHWM_HOME, or ~/.torchwm."""
    home = Path(os.environ.get("TORCHWM_HOME", Path.home() / ".torchwm"))
    root = path or home
    try:
        has_entries = any(root.iterdir()) if root.exists() else False
    except PermissionError:
        has_entries = False

    if not root.exists() or not has_entries:
        click.echo(f"No datasets found at {root}")
        return

    click.echo(f"Datasets under: {root}")
    for entry in sorted(root.iterdir()):
        click.echo(f"- {entry.name}")


@datasets_app.command("convert")
@click.argument("src", type=click.Path(path_type=Path))
@click.option(
    "--dest-format",
    default="video",
    show_default=True,
    help="Destination format. Only 'video' is currently supported.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    help="Output directory for converted data.",
)
def datasets_convert(src: Path, dest_format: str, out_dir: Path | None) -> None:
    """Convert a simple HDF5/NumPy dataset into MP4 videos."""
    if dest_format != "video":
        _echo_error("Only 'video' dest_format is supported.")
        raise click.exceptions.Exit(1)

    srcp = src
    if not srcp.exists():
        _echo_error(f"Source not found: {srcp}")
        raise click.exceptions.Exit(1)

    numpy = _ensure_numpy()
    if numpy is None:
        _echo_error(
            "Install numpy and the dataset/video optional dependencies to use conversion."
        )
        raise click.exceptions.Exit(1)

    from world_models.datasets.video_datasets import HDF5Dataset, NumPyDataset
    from world_models.utils.utils import save_video

    out = out_dir or (Path.cwd() / "converted_datasets")
    out.mkdir(parents=True, exist_ok=True)

    dataset: HDF5Dataset | NumPyDataset
    if srcp.suffix in {".h5", ".hdf5"}:
        dataset = HDF5Dataset(str(srcp), num_frames=16, image_size=64)
    elif srcp.suffix in {".npz", ".npy"}:
        dataset = NumPyDataset(str(srcp), num_frames=16, image_size=64)
    else:
        _echo_error(
            "Unsupported source format for conversion. Provide .h5, .hdf5, .npz, or .npy."
        )
        raise click.exceptions.Exit(1)

    click.echo(f"Converting {len(dataset)} items from {srcp} -> {out} ...")
    for index in range(len(dataset)):
        item = dataset[index]
        arr = item.numpy() if hasattr(item, "numpy") else item
        if getattr(arr, "ndim", None) == 4 and arr.shape[1] in (1, 3, 4):
            arr = numpy.moveaxis(arr, 1, -1)
        if getattr(arr, "dtype", None) is not None and str(arr.dtype).endswith("uint8"):
            arr = arr.astype("float32") / 255.0  # type: ignore[union-attr]
        elif hasattr(arr, "max") and float(arr.max()) > 1.0:
            arr = arr.astype("float32") / 255.0  # type: ignore[union-attr]
        elif hasattr(arr, "astype"):
            arr = arr.astype("float32")

        out_name = out / f"ep_{index:06d}"
        save_video(arr, str(out_name.parent), out_name.name)
        click.echo(f"Wrote: {out_name}.mp4")


@app.group("models")
def models_app() -> None:
    """Model discovery commands."""


@models_app.command("list")
def models_list() -> None:
    """List supported training entrypoints and exported model names."""
    click.echo("Training entrypoints:")
    for name, module in sorted(TRAINING_MODULES.items()):
        click.echo(f"- {name}: {module}")

    try:
        import world_models.models as wm_models
    except Exception:
        return

    exported = getattr(wm_models, "__all__", None)
    if exported:
        click.echo("\nExported model names:")
        for name in sorted(exported):
            click.echo(f"- {name}")


@app.command("benchmark")
@click.option(
    "--agent",
    "agent",
    "-a",
    help="Agent adapter to evaluate (for example: iris, diamond, dreamerv1, dreamerv2).",
)
@click.option(
    "--all-agents",
    is_flag=True,
    help="Run all registered benchmark adapters on the same environment.",
)
@click.option(
    "--game",
    "game",
    "-g",
    required=True,
    help="Environment id to benchmark, such as ALE/Pong-v5 or the BSuite id catch/0.",
)
@click.option(
    "--env-backend",
    type=click.Choice(BENCHMARK_ENV_BACKENDS, case_sensitive=False),
    help="Environment backend for Dreamer-compatible adapters, for example gym or bsuite.",
)
@click.option(
    "--seeds",
    default="1",
    show_default=True,
    help="Either N (creates seeds 0..N-1) or a comma-separated seed list.",
)
@click.option(
    "--episodes",
    "episodes",
    "-n",
    default=5,
    show_default=True,
    type=int,
    help="Evaluation episodes to run per seed.",
)
@click.option(
    "--checkpoint",
    "checkpoint",
    "-c",
    type=click.Path(path_type=Path),
    help="Checkpoint path for a single-agent benchmark.",
)
@click.option(
    "--checkpoint-map",
    "checkpoint_map",
    multiple=True,
    help="For --all-agents, repeat as AGENT=PATH (for example --checkpoint-map iris=ckpt.pt).",
)
@click.option(
    "--out-dir",
    "out_dir",
    default=Path("results/bench"),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Directory where benchmark reports are written.",
)
@click.option(
    "--out_dir",
    "out_dir_alias",
    type=click.Path(path_type=Path),
    help="Alias for --out-dir.",
)
@click.option(
    "--device",
    help="Device passed to adapters. Defaults to CUDA when available, otherwise CPU.",
)
@click.option(
    "--preset",
    help="Optional adapter/model preset to forward to benchmark adapters.",
)
@click.option(
    "--train-epochs",
    type=int,
    help="For --all-agents only: train agents first for this many epochs if no checkpoints are supplied.",
)
def benchmark(
    agent: str | None,
    all_agents: bool,
    game: str,
    env_backend: str | None,
    seeds: str,
    episodes: int,
    checkpoint: Path | None,
    checkpoint_map: tuple[str, ...],
    out_dir: Path,
    out_dir_alias: Path | None,
    device: str | None,
    preset: str | None,
    train_epochs: int | None,
) -> None:
    """Run TorchWM benchmark evaluations and write report artifacts."""
    try:
        if not all_agents and not agent:
            _echo_error("Either --agent or --all-agents must be specified")
            raise click.exceptions.Exit(1)
        if all_agents and agent:
            _echo_error("--agent and --all-agents cannot be used together")
            raise click.exceptions.Exit(1)

        if agent:
            agent = agent.strip().lower()
            if agent not in BENCHMARK_AGENT_NAMES:
                _echo_error(
                    f"Unknown benchmark agent '{agent}'. Known: {', '.join(sorted(BENCHMARK_AGENT_NAMES))}"
                )
                raise click.exceptions.Exit(1)
            if checkpoint is None:
                _echo_error(
                    "--checkpoint is required when using --agent. Only trained models should be benchmarked."
                )
                raise click.exceptions.Exit(1)

        seed_values = _parse_benchmark_seeds(seeds)
        out_dir = out_dir_alias or out_dir
        checkpoints: dict[str, str] = {}
        if all_agents:
            checkpoints = _parse_checkpoint_map(checkpoint_map)
            unknown = sorted(set(checkpoints).difference(BENCHMARK_AGENT_NAMES))
            if unknown:
                _echo_error(
                    f"Unknown benchmark checkpoint agent(s): {', '.join(unknown)}. Known: {', '.join(sorted(BENCHMARK_AGENT_NAMES))}"
                )
                raise click.exceptions.Exit(1)
            if not checkpoints and train_epochs is None:
                _echo_error(
                    "Provide --checkpoint-map AGENT=PATH at least once, or provide --train-epochs for --all-agents."
                )
                raise click.exceptions.Exit(1)

        agents, runner_cls, multi_runner_cls, torch = _load_benchmark_runtime()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        extra_kwargs = {"device": device, "preset": preset}
        env_spec = {"game": game}
        if env_backend:
            env_spec["env_backend"] = env_backend.lower()

        if all_agents:
            runner = multi_runner_cls(
                adapters=list(agents.values()), out_dir=str(out_dir)
            )
            runner.run_all(
                env_spec=env_spec,
                seeds=seed_values,
                num_episodes=episodes,
                checkpoints=checkpoints or None,
                extra_kwargs=extra_kwargs,
                train_epochs=train_epochs,
            )
            click.echo(f"Multi-agent benchmark finished. Results written to: {out_dir}")
            return

        assert agent is not None
        assert checkpoint is not None
        runner = runner_cls(adapter_cls=agents[agent], out_dir=str(out_dir))
        runner.run(
            env_spec=env_spec,
            seeds=seed_values,
            num_episodes=episodes,
            checkpoint=str(checkpoint),
            extra_kwargs=extra_kwargs,
        )
        click.echo(f"Benchmark finished. Results written to: {out_dir}")
    except ValueError as exc:
        _echo_error(str(exc))
        raise click.exceptions.Exit(1) from exc
    except KeyboardInterrupt:
        _echo_error("Benchmark interrupted by user")
        raise click.exceptions.Exit(1)


@app.command("collect")
@click.option("--env", "env", required=True, help="Environment id (Gym/Atari).")
@click.option(
    "--steps",
    default=1000,
    show_default=True,
    type=int,
    help="Number of environment steps to collect.",
)
@click.option(
    "--out",
    default=Path("./collected.npz"),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Output file path (.npz).",
)
@click.option(
    "--random-policy/--no-random-policy",
    default=True,
    show_default=True,
    help="Use random actions.",
)
def collect(env: str, steps: int, out: Path, random_policy: bool) -> None:
    """Collect environment interactions and save observations/actions/rewards/dones."""
    numpy = _ensure_numpy()
    if numpy is None:
        _echo_error("Please install numpy to use collect")
        raise click.exceptions.Exit(1)
    try:
        gym = _ensure_gym()
    except Exception:
        _echo_error("Please install gym or gymnasium to use collect")
        raise click.exceptions.Exit(1)

    click.echo(f"Creating environment: {env}")
    try:
        try:
            from world_models.envs import make_env

            env_obj = make_env(env)
        except Exception:
            env_obj = gym.make(env)
    except Exception as exc:
        logger.debug("Failed to create env %s: %s", env, exc, exc_info=True)
        _echo_error(f"Failed to create environment: {exc}")
        raise click.exceptions.Exit(1) from exc

    observations: list[Any] = []
    actions: list[Any] = []
    rewards: list[float] = []
    dones: list[bool] = []

    obs = env_obj.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    for _ in range(steps):
        action = (
            env_obj.action_space.sample()
            if random_policy
            else numpy.zeros_like(env_obj.action_space.sample())
        )
        result = env_obj.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            next_obs, reward, terminated, truncated, _info = result
            done = bool(terminated or truncated)
        else:
            next_obs, reward, done, _info = result
            done = bool(done)
        observations.append(numpy.asarray(obs))
        actions.append(numpy.asarray(action))
        rewards.append(float(reward))
        dones.append(done)
        if done:
            obs = env_obj.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
        else:
            obs = next_obs

    click.echo(f"Collected {len(observations)} steps; saving to {out}")
    numpy.savez_compressed(
        str(out),
        observations=numpy.stack(observations),
        actions=numpy.stack(actions),
        rewards=numpy.array(rewards),
        dones=numpy.array(dones),
    )
    click.echo("Saved.")


@app.command("eval")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Choice(list(EVAL_MODULES), case_sensitive=False),
    help="Model type to evaluate",
)
@click.option(
    "--checkpoint", "-c", required=True, help="Path to model checkpoint (.pt file)"
)
@click.option("--game", "-g", default="Breakout-v5", help="Environment/game name")
@click.option(
    "--num-videos", default=256, type=int, help="Number of trajectories to compare"
)
@click.option("--trajectory-length", default=20, type=int, help="Frames per trajectory")
@click.option("--batch-size", default=16, type=int, help="Batch size for generation")
@click.option(
    "--device", default=None, help="Device to run on (default: cuda if available)"
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to save results JSON",
)
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option(
    "--metrics",
    default="fid,fvd,lpips",
    help="Comma-separated metrics to compute (fid,fvd,lpips,psnr)",
)
@click.option(
    "--record",
    default=None,
    type=click.Path(path_type=Path),
    help="Save real + generated videos (e.g. eval.mp4)",
)
def eval_command(
    model: str,
    checkpoint: str,
    game: str,
    num_videos: int,
    trajectory_length: int,
    batch_size: int,
    device: str | None,
    output: Path | None,
    seed: int,
    metrics: str,
    record: Path | None,
) -> None:
    """Evaluate a trained world model with FID/FVD/LPIPS metrics."""
    module_path = EVAL_MODULES[model.lower()]
    mod = importlib.import_module(module_path)

    metrics_list = [m.strip() for m in metrics.split(",") if m.strip()]
    mod.run_eval(
        checkpoint=checkpoint,
        game=game,
        num_videos=num_videos,
        trajectory_length=trajectory_length,
        batch_size=batch_size,
        device=device,
        output=str(output) if output else None,
        seed=seed,
        metrics=metrics_list,
        record=str(record) if record else None,
    )


PLAY_MODEL_NAMES = tuple(sorted(PLAY_MODULES))


@app.command("play")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Choice(PLAY_MODEL_NAMES, case_sensitive=False),
    help="Model type to play",
)
@click.option(
    "--checkpoint", "-c", required=True, help="Path to model checkpoint (.pt file)"
)
@click.option("--game", "-g", default="Breakout-v5", help="Environment/game name")
@click.option(
    "--device", default=None, help="Device to run on (default: cuda if available)"
)
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option(
    "--deterministic/--stochastic",
    default=True,
    help="Use deterministic policy actions",
)
@click.option(
    "--record",
    default=None,
    type=click.Path(path_type=Path),
    help="Save gameplay video (e.g. gameplay.mp4)",
)
@click.option("--record-fps", default=20, type=int, help="FPS for recorded video")
def play_command(
    model: str,
    checkpoint: str,
    game: str,
    device: str | None,
    seed: int,
    deterministic: bool,
    record: Path | None,
    record_fps: int,
) -> None:
    """Interactively play inside a trained world model (real env + dream mode)."""
    module_path = PLAY_MODULES[model.lower()]
    mod = importlib.import_module(module_path)
    mod.run_play(
        checkpoint=checkpoint,
        game=game,
        device=device,
        seed=seed,
        deterministic=deterministic,
        record=str(record) if record else None,
        record_fps=record_fps,
    )


@app.command(
    "train",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("model")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--inproc",
    is_flag=True,
    help="Run training in-process instead of spawning subprocess.",
)
def train(model: str, extra_args: tuple[str, ...], inproc: bool) -> None:
    """Launch a training entrypoint with optional YAML/OmegaConf overrides.

    Examples:
        torchwm train iris --config world_models/configs/experiments/iris.yaml total_epochs=100
        torchwm train jepa optimization.epochs=50 data.batch_size=128
    """
    key = model.strip().lower()
    if key not in TRAINING_MODULES:
        _echo_error(
            f"Unknown model '{model}'. Known: {', '.join(TRAINING_MODULES.keys())}"
        )
        raise click.exceptions.Exit(1)

    module = TRAINING_MODULES[key]
    if inproc:
        try:
            mod = importlib.import_module(module)
            main_fn = getattr(mod, "main", None)
            if callable(main_fn):
                click.echo(f"Running in-process: {module}.main()")
                try:
                    main_fn(list(extra_args))
                except TypeError:
                    main_fn()
                return
            click.echo(
                f"Module {module} has no callable main(); falling back to subprocess"
            )
        except KeyboardInterrupt:
            _echo_error("Training interrupted by user")
            raise click.exceptions.Exit(1)
        except Exception as exc:
            logger.debug("In-process training failed: %s", exc, exc_info=True)
            click.echo("Falling back to subprocess execution")

    cmd = [sys.executable, "-m", module, *extra_args]
    click.echo(f"Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd, check=False, shell=False
        )  # shell=False prevents injection
    except KeyboardInterrupt:
        _echo_error("Training interrupted by user")
        raise click.exceptions.Exit(1)
    raise click.exceptions.Exit(proc.returncode)


def main(*args: Any, **kwargs: Any) -> Any:
    """Backward-compatible callable for imports that use ``torchwm.cli:main``."""
    return app.main(*args, **kwargs)


def run() -> None:
    """Console-script entrypoint used by the installed ``torchwm`` command."""
    app(prog_name="torchwm")


def run_train() -> None:
    """Console-script entrypoint used by the installed ``torchwm-train`` command.

    This is equivalent to ``torchwm train MODEL [ARGS...]`` and dispatches via
    ``TRAINING_MODULES`` so new training modules only need to be added to that
    mapping instead of registering another packaging entry point.
    """
    train.main(prog_name="torchwm-train")


if __name__ == "__main__":
    run()

