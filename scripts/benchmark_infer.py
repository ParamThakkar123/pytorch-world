"""Inference / play front-end for ``scripts/benchmark_models.sh``.

The compute sweep in ``benchmark_models.py`` times synthetic forwards. This
module is the other half: run a **trained** (or randomly initialised) model
and either

* **record** a video of it playing / generating, or
* **play** interactively -- with the policy (keys override it) or against it
  (you drive, the policy's action is shown as the opponent).

Wired from the shell as::

    scripts/benchmark_models.sh --infer --model diamond -c ckpt.pt
    scripts/benchmark_models.sh --play  --model dreamer -c ckpt.pt --versus
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_GAMES = {
    "diamond": "Breakout-v5",
    "dreamer": "walker-walk",
    "iris": "ALE/Pong-v5",
}

RECORD_MODELS = ("diamond", "dreamer", "iris", "genie", "dit", "jepa")
PLAY_MODELS = ("diamond", "dreamer", "iris", "genie")


def _load_demo(name: str) -> Any:
    """Load a file under ``demos/`` -- that directory is not a Python package."""
    path = REPO_ROOT / "demos" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"torchwm_demo_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _python() -> str:
    return sys.executable


def _run(argv: list[str]) -> int:
    print("running:", " ".join(str(part) for part in argv))
    return subprocess.call(argv, cwd=str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("record", "play"),
        default="record",
        help="record: write a video and exit. play: open an interactive window.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="",
        help="diamond, dreamer, iris, genie, dit or jepa.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Alias of --model (the compute sweep uses this flag name).",
    )
    parser.add_argument("--checkpoint", "-c", default=None)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Skip the checkpoint (Genie / DiT / I-JEPA pipeline check only).",
    )
    parser.add_argument("--game", "-g", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "results" / "model_inference"),
        help="Where recorded videos and stills land.",
    )
    parser.add_argument("--steps", type=int, default=300, help="[record] env frames.")
    parser.add_argument(
        "--dream-steps",
        type=int,
        default=100,
        help="[diamond record] imagination frames. 0 skips the dream clip.",
    )
    parser.add_argument("--episodes", type=int, default=2, help="[iris record]")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--record", default=None, help="[play] also save an MP4.")
    parser.add_argument(
        "--control",
        choices=("assist", "human", "versus"),
        default="assist",
        help="assist: play WITH the model (keys override). "
        "human: you always drive. "
        "versus: you drive, the model's action is shown as the opponent.",
    )
    parser.add_argument(
        "--versus",
        action="store_true",
        help="Shortcut for --control versus (play against the policy).",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--list", action="store_true", help="List models and exit.")
    return parser


def resolve_model(args: argparse.Namespace) -> str:
    name = (args.model or args.models or "").strip().lower()
    if "," in name:
        name = name.split(",", 1)[0].strip()
    aliases = {
        "dreamer-v1": "dreamer",
        "dreamerv1": "dreamer",
        "dreamer-v2": "dreamer",
        "dreamerv2": "dreamer",
        "dreamer-v3": "dreamer",
        "ijepa": "jepa",
        "i-jepa": "jepa",
        "genie-small": "genie",
    }
    return aliases.get(name, name)


def list_models() -> None:
    print("record: " + ", ".join(RECORD_MODELS))
    print("play:   " + ", ".join(PLAY_MODELS))
    print()
    print("record = video of the model playing / generating")
    print("play   = interactive window")
    print("         --control assist  (default): play WITH the policy")
    print("         --control versus / --versus: you drive, policy is the opponent HUD")


def record_diamond(args: argparse.Namespace, model: str) -> int:
    script = REPO_ROOT / "demos" / "record_diamond.py"
    argv = [
        _python(),
        str(script),
        "--checkpoint",
        str(args.checkpoint),
        "--game",
        args.game or DEFAULT_GAMES["diamond"],
        "--out-dir",
        args.out_dir,
        "--steps",
        str(args.steps),
        "--dream-steps",
        str(args.dream_steps),
        "--fps",
        str(args.fps),
        "--seed",
        str(args.seed),
    ]
    if args.device:
        argv += ["--device", args.device]
    if args.stochastic:
        argv.append("--stochastic")
    return _run(argv)


def record_iris(args: argparse.Namespace, model: str) -> int:
    script = REPO_ROOT / "demos" / "record_iris.py"
    out = Path(args.out_dir) / "iris.mp4"
    argv = [
        _python(),
        str(script),
        "--checkpoint",
        str(args.checkpoint),
        "--game",
        args.game or DEFAULT_GAMES["iris"],
        "--episodes",
        str(args.episodes),
        "--out",
        str(out),
        "--fps",
        str(args.fps),
    ]
    if args.device:
        argv += ["--device", args.device]
    return _run(argv)


def record_dit(args: argparse.Namespace, model: str) -> int:
    script = REPO_ROOT / "demos" / "record_dit.py"
    argv = [_python(), str(script), "--out-dir", args.out_dir, "--seed", str(args.seed)]
    if args.random_init or not args.checkpoint:
        argv.append("--random-init")
    else:
        argv += ["--checkpoint", str(args.checkpoint)]
    if args.device:
        argv += ["--device", args.device]
    return _run(argv)


def record_genie(args: argparse.Namespace, model: str) -> int:
    script = REPO_ROOT / "demos" / "record_genie.py"
    argv = [
        _python(),
        str(script),
        "--out-dir",
        args.out_dir,
        "--num-frames",
        str(min(args.steps, 32)),
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
    ]
    if args.random_init or not args.checkpoint:
        argv.append("--random-init")
    else:
        argv += ["--checkpoint", str(args.checkpoint)]
    if args.device:
        argv += ["--device", args.device]
    return _run(argv)


def record_jepa(args: argparse.Namespace, model: str) -> int:
    script = REPO_ROOT / "demos" / "record_jepa.py"
    argv = [_python(), str(script), "--out-dir", args.out_dir]
    if args.random_init or not args.checkpoint:
        argv.append("--random-init")
    else:
        argv += ["--checkpoint", str(args.checkpoint)]
    if args.device:
        argv += ["--device", args.device]
    return _run(argv)


def record_dreamer(args: argparse.Namespace, model: str) -> int:
    """Headless Dreamer rollout: policy in the real env, no OpenCV window."""
    import numpy as np
    import torch

    from scripts.play_dreamer import DreamerPlayer, _observation_frame
    from torchwm.utils.utils import StreamingVideoWriter

    game = args.game or DEFAULT_GAMES["dreamer"]
    player = DreamerPlayer(
        str(args.checkpoint), env_name=game, device=args.device, seed=args.seed
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dreamer_real.mp4"
    writer = StreamingVideoWriter(str(path), fps=args.fps)

    obs = player.env.reset()
    state = player.rssm.init_state(1, player.device)
    prev_action = torch.zeros(1, player.action_size, device=player.device)
    reward_sum = 0.0
    for step in range(args.steps):
        frame = _observation_frame(obs)
        writer.write_frame((np.clip(frame, 0, 1) * 255).astype(np.uint8))
        with torch.no_grad():
            _, state = player.rssm.observe_step(
                state, prev_action, player.encode(obs)
            )
            action = player.actor(
                player.features(state), deter=not args.stochastic
            )
        action_np = action[0].cpu().numpy()
        obs, reward, done, info = player.env.step(action_np)
        executed = (
            info["action"]
            if isinstance(info, dict) and "action" in info
            else action_np
        )
        prev_action = torch.tensor(
            np.asarray(executed, dtype=np.float32), device=player.device
        ).unsqueeze(0)
        reward_sum += float(reward)
        if done:
            obs = player.env.reset()
            state = player.rssm.init_state(1, player.device)
            prev_action = torch.zeros(1, player.action_size, device=player.device)
        if (step + 1) % 50 == 0:
            print(f"  real: {step + 1}/{args.steps}")

    writer.close()
    player.env.close()
    print(f"Wrote {path}  (return across clip: {reward_sum:.1f})")
    return 0


def play_diamond(args: argparse.Namespace, control: str) -> int:
    from scripts.play_diamond import run_play

    run_play(
        checkpoint=str(args.checkpoint),
        game=args.game or DEFAULT_GAMES["diamond"],
        device=args.device,
        seed=args.seed,
        deterministic=not args.stochastic,
        record=args.record,
        record_fps=args.fps,
        control=control,
    )
    return 0


def play_dreamer(args: argparse.Namespace, control: str) -> int:
    from scripts.play_dreamer import run_play

    run_play(
        checkpoint=str(args.checkpoint),
        game=args.game or DEFAULT_GAMES["dreamer"],
        device=args.device,
        seed=args.seed,
        deterministic=not args.stochastic,
        record=args.record,
        record_fps=args.fps,
        control=control,
    )
    return 0


def play_iris(args: argparse.Namespace, control: str) -> int:
    """Atari window: assist / human / versus, using the policy-only loader."""
    import cv2
    import numpy as np
    import torch

    iris_demo = _load_demo("record_iris")
    infer_architecture = iris_demo.infer_architecture
    load_policy = iris_demo.load_policy
    preprocess = iris_demo.preprocess
    read_checkpoint = iris_demo.read_checkpoint
    from scripts.play_base import get_action_from_key, init_video_recorder
    from scripts.play_diamond import ACTION_NAMES
    from torchwm.configs.iris_config import IRISConfig
    from torchwm.envs.ale_atari_env import make_atari_env
    from torchwm.models.iris_agent import IRISAgent

    game = args.game or DEFAULT_GAMES["iris"]
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    env = make_atari_env(game, obs_type="rgb", frameskip=4)
    n_actions = int(env.action_space.n)
    ckpt = read_checkpoint(str(args.checkpoint))
    arch = infer_architecture(ckpt)
    config = IRISConfig()
    for field in ("actor_layers", "actor_hidden_size"):
        if field in arch:
            setattr(config, field, arch[field])
    agent = IRISAgent(config=config, action_size=n_actions, device=device)
    report = load_policy(agent, ckpt)
    if report["failed"] or not report["loaded"]:
        print("Policy did not load cleanly; refusing to open play.")
        return 1
    agent.eval()

    video_recorder = init_video_recorder(args.record, fps=args.fps)
    cv2.namedWindow("IRIS Play", cv2.WINDOW_NORMAL)
    obs, _ = env.reset()
    episode_reward = 0.0
    step_count = 0
    running = True
    print("IRIS play: arrows/WASD drive, Q quits. --versus shows the policy's action.")

    while running:
        key = cv2.waitKey(16) & 0xFF
        if key in (ord("q"), 27):
            running = False
            continue
        if key == ord("r"):
            obs, _ = env.reset()
            episode_reward = 0.0
            step_count = 0

        frame = preprocess(obs, config.frame_height)
        tensor = torch.from_numpy(frame).unsqueeze(0).to(device)
        agent_action = int(
            agent.act(tensor, epsilon=0.0, temperature=0.01).item()
        )
        human_action = get_action_from_key(key)
        if control == "assist":
            action = agent_action if human_action is None else human_action
            label = "AGENT" if human_action is None else "HUMAN"
        else:
            action = 0 if human_action is None else human_action
            label = "HUMAN vs AGENT" if control == "versus" else "HUMAN"

        display = obs.astype("uint8")
        if display.ndim == 3 and display.shape[2] == 3:
            display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        else:
            display_bgr = display
        hud = [
            f"{label}  R: {episode_reward:.1f}  Step: {step_count}",
            f"Action: {ACTION_NAMES.get(action, action)}"
            + (
                f"  |  AGENT: {ACTION_NAMES.get(agent_action, agent_action)}"
                if control == "versus"
                else ""
            ),
            "[arrows/WASD] drive  [R] reset  [Q] quit",
        ]
        for i, line in enumerate(hud):
            cv2.putText(
                display_bgr,
                line,
                (5, 15 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )
        cv2.imshow("IRIS Play", display_bgr)
        if video_recorder is not None:
            video_recorder.write_frame(obs.astype(np.uint8))

        obs, reward, terminated, truncated, _ = env.step(int(action))
        episode_reward += float(reward)
        step_count += 1
        if terminated or truncated:
            print(f"Episode finished. Reward: {episode_reward:.1f}")
            obs, _ = env.reset()
            episode_reward = 0.0
            step_count = 0

    if video_recorder is not None:
        video_recorder.close()
    cv2.destroyAllWindows()
    env.close()
    return 0


def play_genie(args: argparse.Namespace, control: str) -> int:
    """Drive Genie with latent-action keys. There is no 'versus' env; 0-7 are you."""
    import cv2
    import numpy as np
    import torch

    genie_demo = _load_demo("record_genie")
    build_model = genie_demo.build_model
    tensor_to_uint8_img = genie_demo.tensor_to_uint8_img
    from scripts.play_base import init_video_recorder

    class _Args:
        checkpoint = args.checkpoint
        random_init = bool(args.random_init or not args.checkpoint)
        num_frames = 8
        image_size = 64

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_model(_Args()).to(device).eval()
    vocab = int(getattr(model, "action_vocab_size", 8) or 8)
    frame = torch.rand(1, 3, 64, 64, device=device)
    history: torch.Tensor | None = None
    action_id = 0
    video_recorder = init_video_recorder(args.record, fps=args.fps)
    cv2.namedWindow("Genie Play", cv2.WINDOW_NORMAL)
    print(
        "Genie play: keys 0-7 / WASD select a latent action, SPACE steps, "
        "R new prompt, Q quits."
    )

    key_to_action = {
        ord("w"): 0,
        ord("s"): 1,
        ord("a"): 2,
        ord("d"): 3,
        ord(" "): None,
    }
    running = True
    while running:
        img = tensor_to_uint8_img(frame[0])
        display = cv2.cvtColor(np.tile(img, (4, 4, 1)), cv2.COLOR_RGB2BGR)
        cv2.putText(
            display,
            f"latent action {action_id}  [0-7/WASD] choose  [SPACE] step  [R] reset  [Q] quit",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )
        cv2.imshow("Genie Play", display)
        if video_recorder is not None:
            video_recorder.write_frame(np.tile(img, (4, 4, 1)))

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            running = False
            continue
        if key == ord("r"):
            frame = torch.rand(1, 3, 64, 64, device=device)
            history = None
            continue
        if ord("0") <= key <= ord("9"):
            action_id = (key - ord("0")) % vocab
        elif key in key_to_action and key_to_action[key] is not None:
            action_id = int(key_to_action[key]) % vocab
        elif key != ord(" "):
            continue

        with torch.no_grad():
            action = torch.tensor([action_id], device=device)
            nxt = model.play(frame, action, history)
            if history is None:
                history = frame.unsqueeze(2)
            history = torch.cat([history, nxt.unsqueeze(2)], dim=2)
            if history.shape[2] > 8:
                history = history[:, :, -8:]
            frame = nxt

    if video_recorder is not None:
        video_recorder.close()
    cv2.destroyAllWindows()
    return 0


RECORDERS = {
    "diamond": record_diamond,
    "dreamer": record_dreamer,
    "iris": record_iris,
    "genie": record_genie,
    "dit": record_dit,
    "jepa": record_jepa,
}

PLAYERS = {
    "diamond": play_diamond,
    "dreamer": play_dreamer,
    "iris": play_iris,
    "genie": play_genie,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("note: ignoring flags not used for inference:", " ".join(unknown))

    if args.list:
        list_models()
        return 0

    model = resolve_model(args)
    if not model:
        parser.error("pass --model diamond|dreamer|iris|genie|dit|jepa (or --list)")

    control = "versus" if args.versus else args.control
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "record":
        if model not in RECORDERS:
            parser.error(f"{model} cannot be recorded. Choose: {', '.join(RECORD_MODELS)}")
        if model in {"diamond", "dreamer", "iris"} and not args.checkpoint:
            parser.error(f"{model} record needs --checkpoint / -c")
        print(f"recording {model} -> {args.out_dir}")
        return RECORDERS[model](args, model)

    if model not in PLAYERS:
        parser.error(
            f"{model} has no interactive play loop. "
            f"Use --infer to record a video. Playable: {', '.join(PLAY_MODELS)}"
        )
    if model != "genie" and not args.checkpoint:
        parser.error(f"{model} play needs --checkpoint / -c")
    print(f"playing {model}  control={control}")
    return PLAYERS[model](args, control)


if __name__ == "__main__":
    raise SystemExit(main())
