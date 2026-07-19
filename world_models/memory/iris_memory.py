import numpy as np
from typing import Tuple, Optional, List


class IRISReplayBuffer:
    """Replay buffer for IRIS (Imagination with auto-Regression over an Inner Speech) training.

    Stores (observation, action, reward, terminal) tuples in a ring buffer
    and supports sampling contiguous sequences for world model training.

    Features:
        - Ring buffer with fixed capacity (FIFO eviction when full)
        - Stores uint8 images for memory efficiency
        - Samples sequences with validation to avoid episode boundaries
        - Supports sequence sampling for temporal learning

    Memory Layout:
        - observations: (capacity, C, H, W) uint8
        - actions: (capacity, action_size) float32
        - rewards: (capacity,) float32
        - terminals: (capacity,) float32

    Args:
        size (int): Maximum number of transitions to store.
        obs_shape (tuple): Shape of observations as (C, H, W).
        action_size (int): Dimension of actions.
        seq_len (int): Length of sequences to sample (default: 20).
        batch_size (int): Number of sequences per batch (default: 64).

    Attributes:
        size (int): Buffer capacity.
        obs_shape (tuple): Observation shape.
        action_size (int): Action dimension.
        seq_len (int): Sequence length.
        batch_size (int): Batch size.
        steps (int): Total transitions added.
        episodes (int): Number of episode terminations observed.
    """

    def __init__(
        self,
        size: int,
        obs_shape: Tuple[int, int, int],
        action_size: int,
        seq_len: int = 20,
        batch_size: int = 64,
    ):
        self.size = size
        self.obs_shape = obs_shape  # (C, H, W)
        self.action_size = action_size
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.idx = 0
        self.full = False
        self.steps = 0
        self.episodes = 0

        self.observations = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((size, action_size), dtype=np.float32)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.terminals = np.zeros((size,), dtype=np.float32)

    def add(
        self, obs: np.ndarray, action: np.ndarray, reward: float, terminal: bool
    ) -> None:
        """Add a transition to the buffer.

        Args:
            obs: Observation array with shape (C, H, W).
            action: Action array with shape (action_size,).
            reward: Scalar reward value.
            terminal: Boolean indicating if episode terminated.
        """
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.terminals[self.idx] = float(terminal)

        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1
        self.episodes += 1 if terminal else 0

    def sample_sequence(
        self, seq_len: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of sequences for world model training.

        Returns:
            observations: (batch_size, seq_len+1, C, H, W)
            actions: (batch_size, seq_len, action_size)
            rewards: (batch_size, seq_len)
            terminals: (batch_size, seq_len)
        """
        if seq_len is None:
            seq_len = self.seq_len

        batch_size = self.batch_size
        L = seq_len

        # Sample starting indices
        idxs = self._sample_idxs(L, batch_size)

        # Build sequences
        observations = []
        actions = []
        rewards = []
        terminals = []

        for idx in idxs:
            # Get sequence of observations (L+1 for predicting next frame)
            obs_seq = self.observations[idx : idx + L + 1]
            act_seq = self.actions[idx : idx + L]
            rew_seq = self.rewards[idx : idx + L]
            term_seq = self.terminals[idx : idx + L]

            # Handle wrapping
            if len(obs_seq) < L + 1:
                # Pad by wrapping around
                obs_seq = np.concatenate(
                    [obs_seq, self.observations[: L + 1 - len(obs_seq)]]
                )
                act_seq = np.concatenate([act_seq, self.actions[: L - len(act_seq)]])
                rew_seq = np.concatenate([rew_seq, self.rewards[: L - len(rew_seq)]])
                term_seq = np.concatenate(
                    [term_seq, self.terminals[: L - len(term_seq)]]
                )

            observations.append(obs_seq)
            actions.append(act_seq)
            rewards.append(rew_seq)
            terminals.append(term_seq)

        return (
            np.stack(observations),
            np.stack(actions),
            np.stack(rewards),
            np.stack(terminals),
        )

    def _sample_idxs(self, L: int, n: int) -> np.ndarray:
        """Sample n valid starting indices for sequences of length L."""
        valid_start_range = self.size if self.full else self.idx - L

        if valid_start_range <= 0:
            return np.zeros(n, dtype=int)

        idxs = np.random.randint(0, valid_start_range, size=n)

        # Ensure we don't wrap around terminal states in the middle
        for i in range(n):
            # Check if any terminal in the sequence (excluding last step which is the target)
            for j in range(L - 1):
                if self.terminals[(idxs[i] + j) % self.size] > 0:
                    # Terminal found, need to resample
                    idxs[i] = np.random.randint(0, valid_start_range)
                    break

        return idxs

    def sample_single(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Sample a single transition for online updates."""
        idx = np.random.randint(0, self.size if self.full else self.idx)

        return (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.terminals[idx],
        )

    def __len__(self) -> int:
        return self.size if self.full else self.idx

    @property
    def buffer_capacity(self) -> int:
        """Returns the total capacity of the buffer."""
        return self.size


class IRISOnPolicyBuffer:
    """On-policy buffer for collecting trajectories during environment interaction.

    Used to store the current episode data before adding to the main replay buffer.
    Unlike the main replay buffer, this collects trajectories in a list-based
    structure that's cleared after each episode.

    Useful for:
        - Collecting complete episode trajectories
        - Storing data before batch processing
        - Temporary storage during environment interaction

    Args:
        max_steps (int): Maximum number of steps to store (default: 1000).

    Attributes:
        max_steps (int): Maximum buffer capacity.
        observations (list): List of observations.
        actions (list): List of actions.
        rewards (list): List of rewards.
        terminals (list): List of terminal flags.
    """

    def __init__(self, max_steps: int = 1000):
        self.max_steps = max_steps
        # Typed lists to satisfy static type checkers
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.terminals: List[float] = []

    def add(
        self, obs: np.ndarray, action: np.ndarray, reward: float, terminal: bool
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(float(terminal))

    def clear(self) -> None:
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminals = []

    def get_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self.observations),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.terminals),
        )

    def __len__(self) -> int:
        return len(self.observations)
