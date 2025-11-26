import numpy as np

from collections import deque
from numpy.random import choice


def _identity(x):
    return x


class Episode:
    """Records the agent's interaction with the environment for a single
    episode. At termination, it converts all the data to Numpy arrays.
    """

    def __init__(self, postprocess_fn=None):
        self.x = []
        self.u = []
        self.t = []
        self.r = []
        self.postprocess_fn = _identity if postprocess_fn is None else postprocess_fn
        self._size = 0

    @property
    def size(self):
        return self._size

    def append(self, obs, act, reward, terminal):
        self._size += 1
        self.x.append(self.postprocess_fn(obs.numpy()))
        self.u.append(act.cpu().numpy())
        self.r.append(reward)
        self.t.append(terminal)

    def terminate(self, obs):
        self.x.append(self.postprocess_fn(obs.numpy()))
        self.x = np.stack(self.x)
        self.u = np.stack(self.u)
        self.r = np.stack(self.r)
        self.t = np.stack(self.t)


class Memory(deque):
    def __init__(self, size=None):
        """Maintains a FIFO list of `size` number of episodes.
        If size is None (e.g. during unpickling or loading legacy files),
        create deques without a fixed maxlen so pickle can restore state.
        """
        maxlen = size if size is not None else None
        self.episodes = deque(maxlen=maxlen)
        self.eps_lengths = deque(maxlen=maxlen)
        if size is not None:
            print(f"Creating memory with len {size} episodes.")

    @property
    def size(self):
        return sum(self.eps_lengths)

    def _append(self, episode: Episode):
        if isinstance(episode, Episode):
            self.episodes.append(episode)
            self.eps_lengths.append(episode.size)
        else:
            raise ValueError("can only append <Episode> or list of <Episode>")

    def append(self, episodes: list[Episode]):
        if isinstance(episodes, Episode):
            episodes = [episodes]
        if isinstance(episodes, list):
            for e in episodes:
                self._append(e)
        else:
            raise ValueError("can only append <Episode> or list of <Episode>")

    def sample(self, batch_size, tracelen=1, time_first=False):
        if len(self.episodes) == 0:
            raise ValueError("Memory is empty; cannot sample.")

        valid_idxs = [i for i, L in enumerate(self.eps_lengths) if L >= tracelen]
        if len(valid_idxs) == 0:
            raise ValueError(
                f"No episodes with length >= {tracelen} available to sample."
            )

        # Quick memory usage estimation to avoid large unexpected allocations.
        # Estimate bytes required for stacked observations: N * (tracelen+1) * prod(obs_shape) * bytes_per_elem
        try:
            sample_ep = self.episodes[valid_idxs[0]]
            if isinstance(sample_ep.x, np.ndarray):
                obs_shape = sample_ep.x.shape[1:]  # (C,H,W)
                obs_dtype = sample_ep.x.dtype
            else:
                obs0 = np.asarray(sample_ep.x[0])
                obs_shape = obs0.shape
                obs_dtype = obs0.dtype
            bytes_per_elem = np.dtype(obs_dtype).itemsize
            est_bytes = int(
                batch_size * (tracelen + 1) * np.prod(obs_shape) * bytes_per_elem
            )
            MAX_BYTES = 200 * 1024 * 1024  # 200 MiB threshold (tunable)
            if est_bytes > MAX_BYTES:
                est_mb = est_bytes / (1024 * 1024)
                raise MemoryError(
                    f"Sampling would allocate ~{est_mb:.1f} MiB for observations "
                    f"(N={batch_size}, H={tracelen}, obs_shape={obs_shape}, dtype={obs_dtype}). "
                    "Reduce batch_size or trace length (H) or downsample images."
                )
        except Exception:
            # If anything goes wrong in estimation, continue and let later code raise the original error.
            pass

        episode_idx = choice(valid_idxs, batch_size)

        init_st_idx = []
        for i in episode_idx:
            max_start = self.eps_lengths[i] - tracelen + 1
            if max_start <= 0:
                init_st_idx.append(0)
            else:
                init_st_idx.append(choice(max_start))

        x, u, r, t = [], [], [], []
        for n, (i, s) in enumerate(zip(episode_idx, init_st_idx)):
            ep = self.episodes[i]
            # ensure arrays (support episodes that weren't .terminate()'d)
            x_arr = np.stack(ep.x) if not isinstance(ep.x, np.ndarray) else ep.x

            # Normalize action arrays to have shape (T, action_dim)
            u_arr = np.stack(ep.u) if not isinstance(ep.u, np.ndarray) else ep.u
            u_arr = np.asarray(u_arr)
            if u_arr.ndim == 1:
                u_arr = u_arr[:, None]  # (T,) -> (T,1)

            # Ensure rewards and terminals are 1D arrays of length T
            r_arr = np.stack(ep.r) if not isinstance(ep.r, np.ndarray) else ep.r
            r_arr = np.asarray(r_arr).reshape(-1)

            t_arr = np.stack(ep.t) if not isinstance(ep.t, np.ndarray) else ep.t
            t_arr = np.asarray(t_arr).reshape(-1)

            x.append(x_arr[s : s + tracelen + 1])
            u.append(u_arr[s : s + tracelen])
            r.append(r_arr[s : s + tracelen])
            t.append(t_arr[s : s + tracelen])
        try:
            if tracelen == 1:
                rets = [np.stack(x)] + [np.stack(i)[:, 0] for i in (u, r, t)]
            else:
                rets = [np.stack(i) for i in (x, u, r, t)]
        except ValueError as exc:

            def shapes(lst):
                return [getattr(a, "shape", np.asarray(a).shape) for a in lst]

            info = {
                "x_shapes": shapes(x),
                "u_shapes": shapes(u),
                "r_shapes": shapes(r),
                "t_shapes": shapes(t),
                "episode_idx": episode_idx.tolist(),
                "start_idx": init_st_idx,
                "eps_lengths_sampled": [self.eps_lengths[i] for i in episode_idx],
            }
            raise ValueError(
                f"Failed to stack sampled segments; shapes info: {info}"
            ) from exc
        if time_first:
            rets = [a.swapaxes(1, 0) for a in rets]
        lengths = np.array([self.eps_lengths[i] for i in episode_idx])
        return rets, lengths
