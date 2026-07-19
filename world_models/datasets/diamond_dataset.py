import numpy as np
from typing import Dict, Tuple
import torch


class ReplayBuffer:
    """
    Replay buffer for storing environment interactions.
    Stores (observation, action, reward, done, next_observation) tuples.
    """

    def __init__(
        self,
        capacity: int = 1000,
        obs_shape: Tuple[int, int, int] = (64, 64, 3),
        action_dim: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device

        self.observations = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.next_observations = np.zeros((capacity,) + obs_shape, dtype=np.uint8)

        self.position = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
    ) -> None:
        """Add a transition to the buffer."""
        self.observations[self.position] = obs
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.next_observations[self.position] = next_obs

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        obs = (
            torch.from_numpy(self.observations[indices]).float().to(self.device) / 255.0
        )
        # observations stored as H,W,C -> convert to C,H,W
        obs = obs.permute(0, 3, 1, 2)

        next_obs = (
            torch.from_numpy(self.next_observations[indices]).float().to(self.device)
            / 255.0
        )
        next_obs = next_obs.permute(0, 3, 1, 2)

        actions = torch.from_numpy(self.actions[indices]).long().to(self.device)
        if actions.ndim > 1 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": torch.from_numpy(self.rewards[indices]).float().to(self.device),
            "dones": torch.from_numpy(self.dones[indices]).bool().to(self.device),
            "next_obs": next_obs,
        }

    def sample_sequence(
        self,
        batch_size: int,
        sequence_length: int,
        burn_in: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a sequence of transitions for training.

        Args:
            batch_size: Number of sequences to sample
            sequence_length: Total sequence length (burn_in + horizon)
            burn_in: Number of initial frames to use for conditioning

        Returns:
            Dictionary with tensors of shape (batch_size, sequence_length, ...)
        """
        max_start = self.size - sequence_length - 1
        if max_start < 0:
            max_start = 0

        start_indices = np.random.randint(0, max_start + 1, size=batch_size)

        obs_seq = []
        action_seq = []
        reward_seq = []
        done_seq = []
        next_obs_seq = []

        for i in range(batch_size):
            start = start_indices[i]
            indices = np.arange(start, start + sequence_length + 1)

            obs_seq.append(self.observations[indices[:-1]])
            action_seq.append(self.actions[indices[:-1]])
            reward_seq.append(self.rewards[indices[:-1]])
            done_seq.append(self.dones[indices[:-1]])
            next_obs_seq.append(self.next_observations[indices[:-1]])

        obs = torch.from_numpy(np.stack(obs_seq)).float().to(self.device) / 255.0
        # obs: (B, T, H, W, C) -> (B, T, C, H, W)
        obs = obs.permute(0, 1, 4, 2, 3)

        next_obs = (
            torch.from_numpy(np.stack(next_obs_seq)).float().to(self.device) / 255.0
        )
        next_obs = next_obs.permute(0, 1, 4, 2, 3)

        actions = torch.from_numpy(np.stack(action_seq)).long().to(self.device)
        if actions.ndim > 2 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": torch.from_numpy(np.stack(reward_seq)).float().to(self.device),
            "dones": torch.from_numpy(np.stack(done_seq)).bool().to(self.device),
            "next_obs": next_obs,
        }

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= min_size

    def state_dict(self) -> dict:
        """Return a serializable state dict for checkpointing.

        Contains numpy arrays and scalar metadata so it can be saved with
        torch.save or numpy.save.
        """
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "next_observations": self.next_observations,
            "position": int(self.position),
            "size": int(self.size),
            "capacity": int(self.capacity),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state previously produced by `state_dict()`.

        This will resize internal arrays if the saved capacity differs from the
        current buffer capacity.
        """
        obs = state["observations"]
        actions = state["actions"]
        rewards = state["rewards"]
        dones = state["dones"]
        next_obs = state["next_observations"]
        pos = int(state.get("position", 0))
        size = int(state.get("size", 0))

        # allocate arrays with saved capacity shapes
        self.capacity = int(state.get("capacity", obs.shape[0]))
        self.observations = np.zeros((self.capacity,) + self.obs_shape, dtype=np.uint8)
        self.next_observations = np.zeros(
            (self.capacity,) + self.obs_shape, dtype=np.uint8
        )
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)

        # copy available data up to saved size
        n = min(size, obs.shape[0], self.capacity)
        if n > 0:
            self.observations[:n] = obs[:n]
            self.next_observations[:n] = next_obs[:n]
            self.actions[:n] = actions[:n]
            self.rewards[:n] = rewards[:n]
            self.dones[:n] = dones[:n]

        self.position = int(pos) % self.capacity if self.capacity > 0 else 0
        self.size = min(int(size), self.capacity)


class SequenceDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for sampling sequences from the replay buffer.
    Used for training the diffusion world model.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sequence_length: int = 5,  # L (conditioning) + 1 (next frame)
        burn_in: int = 4,
    ):
        self.replay_buffer = replay_buffer
        self.sequence_length = sequence_length
        self.burn_in = burn_in

    def __len__(self) -> int:
        return max(0, self.replay_buffer.size - self.sequence_length)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence starting at idx."""
        indices = np.arange(idx, idx + self.sequence_length + 1)
        # keep numpy arrays separate to avoid mypy inferring ndarray types
        obs_seq_np = self.replay_buffer.observations[indices[:-1]]
        action_seq_np = self.replay_buffer.actions[indices[:-1]]
        reward_seq_np = self.replay_buffer.rewards[indices[:-1]]
        done_seq_np = self.replay_buffer.dones[indices[:-1]]
        next_obs_np = self.replay_buffer.next_observations[indices[-1]]

        # stay on CPU; the training loop batches and transfers to GPU
        obs_seq = torch.from_numpy(obs_seq_np).float() / 255.0
        # (T, H, W, C) -> (T, C, H, W)
        if obs_seq.ndim == 4:
            obs_seq = obs_seq.permute(0, 3, 1, 2)

        next_obs = torch.from_numpy(next_obs_np).float() / 255.0
        # ensure next_obs is (C, H, W)
        if next_obs.ndim == 3:
            next_obs = next_obs.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)

        action_seq = torch.from_numpy(action_seq_np).long()
        if action_seq.ndim > 1 and action_seq.shape[-1] == 1:
            action_seq = action_seq.squeeze(-1)

        rewards = torch.from_numpy(reward_seq_np).float()
        dones = torch.from_numpy(done_seq_np).bool()

        return {
            "obs_seq": obs_seq,
            "action_seq": action_seq,
            "actions": action_seq,  # duplicate key for compatibility
            "rewards": rewards,
            "dones": dones,
            "next_obs": next_obs,
        }
