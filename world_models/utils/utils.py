import os
import cv2
import gym
import torch
import pickle
import pathlib
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import plotly
from plotly.graph_objs import Scatter, Line

from collections import defaultdict
from world_models.memory.planet_memory import Memory

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

import torch.nn.functional as F


def to_tensor_obs(image):
    """
    Converts the input np img to channel first 64x64 dim torch img.
    """
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    return image


def postprocess_img(image, depth):
    """
    Postprocess an image observation for storage.
    From float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
    """
    image = np.floor((image + 0.5) * 2**depth)
    return np.clip(image * 2 ** (8 - depth), 0, 2**8 - 1).astype(np.uint8)


def preprocess_img(image, depth):
    """
    Preprocesses an observation inplace.
    From float32 Tensor [0, 255] to [-0.5, 0.5]
    Also adds some noise to the observations !!
    """
    image.div_(2 ** (8 - depth)).floor_().div_(2**depth).sub_(0.5)
    image.add_(torch.randn_like(image).div_(2**depth)).clamp_(-0.5, 0.5)


def bottle(func, *tensors):
    """
    Evaluates a func that operates in N x D with inputs of shape N x T x D
    """
    n, t = tensors[0].shape[:2]
    inputs = [x.reshape(n * t, *x.shape[2:]) for x in tensors]
    out = func(*inputs)
    return out.view(n, t, *out.shape[1:])


def get_combined_params(*models):
    """
    Returns the combine parameter list of all the models given as input.
    """
    params = []
    for model in models:
        params.extend(list(model.parameters()))
    return params


def save_video(frames, path, name):
    """
    Saves a video containing frames.

    Accepts frames in either:
      - (T, C, H, W) float in [0,1]
      - (T, H, W, C) float in [0,1]

    Produces {path}/{name}.mp4 and a debug PNG {path}/{name}_debug_frame.png
    with per-channel statistics printed to stdout.
    """
    import numpy as _np

    frames = _np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(
            f"Expected frames with 4 dims (T, C, H, W) or (T, H, W, C), got shape {frames.shape}"
        )

    # detect layout
    if frames.shape[1] in (1, 3, 4):
        # (T, C, H, W)
        is_chw = True
    elif frames.shape[-1] in (1, 3, 4):
        # (T, H, W, C)
        is_chw = False
    else:
        raise ValueError(f"Can't infer channel axis from frames.shape={frames.shape}")

    # convert floats -> uint8 and to HWC format for OpenCV
    if is_chw:
        # (T, C, H, W) -> (T, H, W, C)
        frames_u8 = (frames * 255.0).clip(0, 255).astype("uint8").transpose(0, 2, 3, 1)
    else:
        frames_u8 = (frames * 255.0).clip(0, 255).astype("uint8")

    # Basic per-frame / per-channel sanity checks on first frame
    if frames_u8.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unexpected channel count: {frames_u8.shape[-1]}")

    first = frames_u8[0]
    ch = first.shape[-1]
    stats = {"min": [], "max": [], "mean": []}
    for c in range(ch):
        stats["min"].append(int(first[..., c].min()))
        stats["max"].append(int(first[..., c].max()))
        stats["mean"].append(float(first[..., c].mean()))
    equal_ch = False
    if ch >= 3:
        equal_ch = _np.all(first[..., 0] == first[..., 1]) and _np.all(
            first[..., 1] == first[..., 2]
        )

    out_dir = pathlib.Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_path = out_dir / f"{name}_debug_frame.png"
    try:
        to_write = first
        if first.ndim == 2 or first.shape[-1] == 1:
            to_write = _np.repeat(first[..., None], 3, axis=-1)
        cv2.imwrite(str(debug_path), to_write[..., ::-1])
    except Exception as e:
        print(f"Failed to write debug frame PNG: {e}")

    print(
        f"[save_video] frames.shape={frames.shape} inferred_chw={is_chw} -> written_frame_shape={frames_u8.shape}"
    )
    print(
        f"[save_video] first_frame stats min={stats['min']} max={stats['max']} mean={stats['mean']} equal_rgb_channels={equal_ch}"
    )
    print(f"[save_video] debug PNG saved to: {debug_path}")

    # Write video with OpenCV (expecting HWC uint8 BGR)
    H, W = (None, None)
    if frames_u8.ndim == 4:
        H, W = frames_u8.shape[1], frames_u8.shape[2]
    else:
        raise RuntimeError("Unexpected frames_u8 shape after conversion")

    writer = cv2.VideoWriter(
        str(out_dir / f"{name}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (W, H),
        True,
    )
    try:
        for frame in frames_u8:
            # ensure contiguous HWC uint8
            if not frame.flags["C_CONTIGUOUS"]:
                frame = _np.ascontiguousarray(frame)
            # OpenCV expects BGR
            writer.write(frame[..., ::-1])
    finally:
        writer.release()
    return str(out_dir / f"{name}.mp4")


def combine_videos(
    video_dir, output_name="combined.mp4", pattern="vid_*.mp4", fps=25, resize=True
):
    """
    Combine all videos matching `pattern` in `video_dir` into a single MP4 file.
    Returns the output filepath (string).

    Example:
      combine_videos("results/planet", output_name="all_training.mp4")
    """
    import glob

    files = sorted(glob.glob(os.path.join(video_dir, pattern)))
    if len(files) == 0:
        raise FileNotFoundError(f"No videos found in {video_dir} matching {pattern}")

    # probe first video for size
    cap0 = cv2.VideoCapture(files[0])
    if not cap0.isOpened():
        cap0.release()
        raise RuntimeError(f"Failed to open video {files[0]}")
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = str(pathlib.Path(video_dir) / output_name)
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (width, height), True)

    try:
        for f in files:
            cap = cv2.VideoCapture(f)
            if not cap.isOpened():
                cap.release()
                continue
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # frame is BGR height x width x channels
                if resize and (frame.shape[1] != width or frame.shape[0] != height):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
            cap.release()
    finally:
        writer.release()
    return out_path


def ensure_results_dir_exists(results_dir):
    """
    Simple helper to validate a results directory exists.
    Raises FileNotFoundError if not present.
    """
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")


def save_frames(target, pred_prior, pred_posterior, name, n_rows=5):
    """
    Save side-by-side comparisons of target / prior / posterior predictions.
    Accepts:
      - target: [T+1, C, H, W] or [C, H, W] (torch.Tensor)
      - pred_prior / pred_posterior: [T, C, H, W] or [C, H, W]
    Produces name.png (and ensures output directory exists).
    """

    def ensure_time_dim(x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x.dim() == 3:
            return x.unsqueeze(0)
        return x

    tgt = ensure_time_dim(target).float()
    pp = ensure_time_dim(pred_prior).float()
    ppp = ensure_time_dim(pred_posterior).float()

    T_pred = pp.shape[0]
    T_tgt = max(0, tgt.shape[0] - 1)
    n = min(T_pred, T_tgt if T_tgt > 0 else T_pred)
    if n == 0:
        n = min(T_pred, tgt.shape[0])

    frames = []
    for t in range(n):
        if tgt.shape[0] > 1 and t + 1 < tgt.shape[0]:
            tf = tgt[t + 1]
        else:
            tf = tgt[min(t, tgt.shape[0] - 1)]
        pr = pp[min(t, pp.shape[0] - 1)]
        ppr = ppp[min(t, ppp.shape[0] - 1)]

        H, W = tf.shape[1], tf.shape[2]

        def match_size(img, H, W):
            img4 = img.unsqueeze(0)
            if img4.shape[2] != H or img4.shape[3] != W:
                img4 = F.interpolate(
                    img4, size=(H, W), mode="bilinear", align_corners=False
                )
            return img4.squeeze(0)

        tf = match_size(tf, H, W)
        pr = match_size(pr, H, W)
        ppr = match_size(ppr, H, W)

        def clamp01(x):
            if x.min() < 0 or x.max() > 1:
                return (x - x.min()) / (x.max() - x.min() + 1e-8)
            return x

        tf = clamp01(tf)
        pr = clamp01(pr)
        ppr = clamp01(ppr)

        # concatenate side-by-side along width (dim=2 because [C,H,W])
        row = torch.cat([tf, pr, ppr], dim=2)
        frames.append(row)

    if len(frames) == 0:
        raise RuntimeError(f"No frames to save for {name}")

    grid = make_grid(torch.stack(frames), nrow=n_rows, normalize=False)
    out_dir = os.path.dirname(name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_image(grid, f"{name}.png")


def get_mask(tensor, lengths):
    """
    Generates masks for batches of sequences.

    Accepts:
      - tensor: torch.Tensor or numpy array with shape (N, T, ...) or (N,) (no time dim)
    Returns:
      - mask: torch.Tensor with batch-first layout [N, T, ...] (same device/dtype as input)
    """
    # convert numpy -> torch if needed
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)

    # If tensor has no time axis (e.g. shape [N]), create mask with T = max(lengths)
    if tensor.dim() == 1:
        N = tensor.shape[0]
        T = int(max(lengths))
        mask = torch.zeros((N, T), dtype=tensor.dtype, device=tensor.device)
        for i in range(N):
            mask[i, : int(lengths[i])] = 1.0
        return mask

    # If tensor has time axis as second dim (batch-first: [N, T, ...])
    mask = torch.zeros_like(tensor)
    N = tensor.shape[0]
    for i in range(N):
        mask[i, : int(lengths[i])] = 1.0
    return mask


def load_memory(path, device):
    """
    Loads an experience replay buffer (backwards-compatible with older pickle formats).
    Converts legacy list/.data formats into the current Memory(episodes) object.
    """
    with open(path, "rb") as f:
        memory = pickle.load(f)

    # If file contains a plain list of Episode objects -> wrap into Memory
    if isinstance(memory, list):
        mem = Memory(len(memory))
        mem.append(memory)
        memory = mem

    # If old object had `.data` attribute (legacy) -> convert to Memory
    if hasattr(memory, "data"):
        try:
            mem = Memory(len(memory.data))
            mem.append(memory.data)
            memory = mem
        except Exception:
            # fallback: just attach device to elements
            for e in memory.data:
                setattr(e, "device", device)

    # If object already has `.episodes`, ensure devices are set
    if hasattr(memory, "episodes"):
        for e in memory.episodes:
            setattr(e, "device", device)

    # Final attach device for the container itself
    setattr(memory, "device", device)
    return memory


def flatten_dict(data, sep=".", prefix=""):
    """Flattens a nested dict into a dict.
    eg. {'a': 2, 'b': {'c': 20}} -> {'a': 2, 'b.c': 20}
    """
    x = {}
    for key, val in data.items():
        if isinstance(val, dict):
            x.update(flatten_dict(val, sep=sep, prefix=key))
        else:
            x[f"{prefix}{sep}{key}"] = val
    return x


def normalize_frames_for_saving(frames):
    """
    Ensure frames are in shape (T, H, W, 3) with float values in [0,1].
    Handles inputs in (T, C, H, W) or (T, H, W, C), repeats single-channel -> RGB,
    drops alpha if present, and maps [-0.5,0.5] -> [0,1] when detected.
    """
    import numpy as _np

    frames = _np.asarray(frames).astype(_np.float32)
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D frames array, got shape {frames.shape}")

    if frames.shape[1] in (1, 3, 4):
        frames = frames.transpose(0, 2, 3, 1)
    elif frames.shape[-1] in (1, 3, 4):
        pass
    else:
        raise ValueError(f"Can't infer channel axis from frames.shape={frames.shape}")

    ch = frames.shape[-1]
    if ch == 1:
        frames = _np.repeat(frames, 3, axis=-1)
    elif ch == 4:
        frames = frames[..., :3]

    mn, mx = float(frames.min()), float(frames.max())
    if mn >= -0.6 and mx <= 0.6:
        frames = (frames + 0.5).clip(0.0, 1.0)
    else:
        frames = frames.clip(0.0, 1.0)

    return frames


class TensorBoardMetrics:
    """Plots and (optionally) stores metrics for an experiment."""

    def __init__(self, path):
        self.writer = SummaryWriter(path)
        self.steps = defaultdict(lambda: 0)
        self.summary = {}

    def assign_type(self, key, val):
        if isinstance(val, (list, tuple)):

            def fun(k, x, s):
                self.writer.add_histogram(k, np.array(x), s)

            self.summary[key] = fun
        elif isinstance(val, (np.ndarray, torch.Tensor)):
            self.summary[key] = self.writer.add_histogram
        elif isinstance(val, float) or isinstance(val, int):
            self.summary[key] = self.writer.add_scalar
        else:
            raise ValueError(f"Datatype {type(val)} not allowed")

    def update(self, metrics: dict):
        metrics = flatten_dict(metrics)
        for key_dots, val in metrics.items():
            key = key_dots.replace(".", "/")
            if self.summary.get(key, None) is None:
                self.assign_type(key, val)
            self.summary[key](key, val, self.steps[key])
            self.steps[key] += 1


def apply_model(model, inputs, ignore_dim=None):
    pass


def plot_metrics(metrics, path, prefix):
    os.makedirs(path, exist_ok=True)
    for key, val in metrics.items():
        lineplot(np.arange(len(val)), val, f"{prefix}{key}", path)


def lineplot(xs, ys, title, path="", xaxis="episode"):
    MAX_LINE = Line(color="rgb(0, 132, 180)", dash="dash")
    MIN_LINE = Line(color="rgb(0, 132, 180)", dash="dash")
    NO_LINE = Line(color="rgba(0, 0, 0, 0)")
    MEAN_LINE = Line(color="rgb(0, 172, 237)")
    std_colour = "rgba(29, 202, 255, 0.2)"
    if isinstance(ys, dict):
        data = []
        for key, val in ys.items():
            xs = np.arange(len(val))
            data.append(Scatter(x=xs, y=np.array(val), name=key))
    elif np.asarray(ys, dtype=np.float32).ndim == 2:
        ys = np.asarray(ys, dtype=np.float32)
        ys_mean, ys_std = ys.mean(-1), ys.std(-1)
        ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std
        l_max = Scatter(x=xs, y=ys.max(-1), line=MAX_LINE, name="Max")
        l_min = Scatter(x=xs, y=ys.min(-1), line=MIN_LINE, name="Min")
        l_stu = Scatter(x=xs, y=ys_upper, line=NO_LINE, showlegend=False)
        l_mean = Scatter(
            x=xs,
            y=ys_mean,
            fill="tonexty",
            fillcolor=std_colour,
            line=MEAN_LINE,
            name="Mean",
        )
        l_stl = Scatter(
            x=xs,
            y=ys_lower,
            fill="tonexty",
            fillcolor=std_colour,
            line=NO_LINE,
            name="-1 Std. Dev.",
            showlegend=False,
        )
        data = [l_stu, l_mean, l_stl, l_min, l_max]
    else:
        data = [Scatter(x=xs, y=ys, line=MEAN_LINE)]
    plotly.offline.plot(
        {
            "data": data,
            "layout": dict(title=title, xaxis={"title": xaxis}, yaxis={"title": title}),
        },
        filename=os.path.join(path, title + ".html"),
        auto_open=False,
    )


class TorchImageEnvWrapper:
    """
    Torch Env Wrapper that wraps a gym env and makes interactions using Tensors.
    Also returns observations in image form.
    """

    def __init__(self, env, bit_depth, observation_shape=None, act_rep=2):
        try:
            self.env = gym.make(env, render_mode="rgb_array")
            self._render_mode_supported = True
        except TypeError:
            self.env = gym.make(env)
            self._render_mode_supported = False
        self.bit_depth = bit_depth
        self.action_repeats = act_rep

    def _get_frame(self, last_obs=None):
        """Call env.render robustly across gym versions. Returns ndarray frame or None.
        If rendering fails (OverflowError / pygame Surface) fallback to last_obs or a
        synthesized image from the observation vector.
        """
        frame = None
        try:
            out = self.env.render()
        except Exception:
            out = None
        # gym may return (frame, info)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, np.ndarray):
            frame = out
        else:
            # try alternative render signature
            try:
                out = self.env.render(mode="rgb_array")
                if isinstance(out, tuple):
                    out = out[0]
                if isinstance(out, np.ndarray):
                    frame = out
            except Exception:
                frame = None

        # If frame is still not ndarray, and last_obs is image-like, use that
        if frame is None and last_obs is not None:
            frame = self._obs_to_frame(
                last_obs if not isinstance(last_obs, tuple) else last_obs[0]
            )

        # If still none, synthesize a simple visualization from a 1D state vector
        if frame is None and last_obs is not None:
            try:
                obs = last_obs if not isinstance(last_obs, tuple) else last_obs[0]
                arr = np.asarray(obs)
                # if simple vector -> make a 64x64 RGB gradient from values
                if arr.ndim == 1 or (arr.ndim == 2 and 1 in arr.shape):
                    vals = (
                        (arr.flatten() - arr.min())
                        if arr.max() != arr.min()
                        else arr.flatten()
                    )
                    if vals.size == 0:
                        vals = np.zeros(1)
                    vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
                    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
                    for i, v in enumerate(vals[:8]):  # encode first few dims as bands
                        band = int(255 * v)
                        canvas[:, i * 8 : (i + 1) * 8, :] = band
                    frame = canvas
            except Exception:
                frame = None

        return frame

    def _obs_to_frame(self, obs):
        """If obs already looks like an image (H,W,3) return it, else None."""
        if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[2] in (1, 3, 4):
            return obs
        return None

    def reset(self):
        ret = self.env.reset()
        obs = ret[0] if isinstance(ret, tuple) else ret
        frame = self._obs_to_frame(obs) or self._get_frame(last_obs=obs)
        if frame is None:
            raise RuntimeError(
                "Environment did not provide an RGB frame on reset. "
                "Use an env that supports image rendering or instantiate with render_mode='rgb_array'."
            )
        x = to_tensor_obs(frame)
        preprocess_img(x, self.bit_depth)
        return x

    def step(self, u):
        if isinstance(u, torch.Tensor):
            u_t = u.cpu().detach()
        else:
            u_t = u
        if getattr(self.env.action_space, "n", None) is not None:
            n = int(self.env.action_space.n)
            try:
                val = (
                    float(u_t.item())
                    if isinstance(u_t, torch.Tensor)
                    else float(np.asarray(u_t).reshape(-1)[0])
                )
            except Exception:
                val = float(u_t)
            action = int(np.clip(int(round(val)), 0, n - 1))
        else:
            action = u_t.numpy() if isinstance(u_t, torch.Tensor) else np.asarray(u_t)

        rwds = 0
        last_d = False
        last_i = {}
        last_obs = None
        for _ in range(self.action_repeats):
            ret = self.env.step(action)
            if len(ret) == 4:
                obs, r, d, i = ret
            else:
                obs, r, term, trunc, i = ret
                d = term or trunc
            rwds += r
            last_d = d
            last_i = i
            last_obs = obs
        frame = self._obs_to_frame(
            last_obs if not isinstance(last_obs, tuple) else last_obs[0]
        ) or self._get_frame(last_obs=last_obs)
        if frame is None:
            raise RuntimeError(
                "Environment did not provide an RGB frame on step. "
                "Use an env that supports image rendering or instantiate with render_mode='rgb_array'."
            )
        x = to_tensor_obs(frame)
        preprocess_img(x, self.bit_depth)
        return x, rwds, last_d, last_i

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @property
    def observation_size(self):
        return (3, 64, 64)

    @property
    def action_size(self):
        space = getattr(self.env, "action_space", None)
        if space is None:
            return 1

        shp = getattr(space, "shape", None)
        if shp and len(shp) > 0:
            try:
                prod = int(np.prod(shp))
                return prod if prod > 0 else 1
            except Exception:
                pass

        if getattr(space, "n", None) is not None:
            return 1

        try:
            sample = space.sample()
            arr = np.asarray(sample)
            if arr.ndim == 0:
                return 1
            return int(arr.size)
        except Exception:
            return 1

    def sample_random_action(self):
        return torch.tensor(self.env.action_space.sample())

    @property
    def max_episode_steps(self):
        """Return environment max episode steps (compatible with TimeLimit/spec)."""
        if (
            hasattr(self.env, "_max_episode_steps")
            and self.env._max_episode_steps is not None
        ):
            return self.env._max_episode_steps
        if (
            getattr(self.env, "spec", None) is not None
            and getattr(self.env.spec, "max_episode_steps", None) is not None
        ):
            return int(self.env.spec.max_episode_steps)
        return 1000


def apply_masks(x, masks):
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        all_x.append(torch.gather(x, 1, mask_keep))
    return torch.cat(all_x, dim=0)
