import shutil
from datetime import datetime
import os
import argparse
import gymnasium as gym
import numpy as np
from multiprocessing import Pool
from torch.utils.data import Dataset
from albumentations.core.composition import Compose
import glob
import torch
from bisect import bisect

"""
Data Generation for World Models Dataset by random rollouts
"""


def rollout(data):
    data_dir, seq_len, rollouts = data
    os.makedirs(data_dir)
    env = gym.make("CarRacing-v2", continuous=False)

    for i in range(rollouts):
        env.reset()
        # get random actions
        actions_rollout = [env.action_space.sample() for _ in range(seq_len)]
        observations_rollout = []
        rewards_rollout = []
        dones_rollout = []

        t = 0
        while True:
            action = actions_rollout[t]
            t += 1

            obs, reward, done, truncated, _ = env.step(action)
            observations_rollout += [obs]
            rewards_rollout += [reward]
            dones_rollout += [done]

            if done or truncated:
                print(f"{data_dir.split('/')[-1]} | End of rollout {i} | {t} frames")
                np.savez(
                    os.path.join(data_dir, f"rollout_{i}"),
                    observations=np.array(observations_rollout),
                    rewards=np.array(rewards_rollout),
                    actions=np.array(actions_rollout),
                    terminals=np.array(dones_rollout),
                )
                break


class RolloutDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Compose,
        train: bool = True,
        buffer_size: int = 100,
        num_test_files: int = 600,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.files = glob.glob(self.root + "/**/*.npz", recursive=True)
        if train:
            self.files = self.files[:-num_test_files]
        else:
            self.files = self.files[-num_test_files:]

        self.cum_size = None
        self.buffer = None
        self.buffer_size = buffer_size
        self.buffer_idx = 0
        self.buffer_fnames = None

    def __len__(self):
        if not self.cum_size:
            print("Load new buffer")
            self.load_next_buffer()
        return self.cum_size[-1]

    def __getitem__(self, idx: int):
        file_idx = bisect(self.cum_size, idx)
        seq_idx = idx - self.cum_size[file_idx]
        data = self.buffer[file_idx]
        return self._get_data(data, seq_idx)

    def _get_data(self, data, idx):
        obs = data["observations"][idx]
        action = data["actions"][idx]
        reward = data["rewards"][idx]
        terminal = data["terminals"][idx]

        if self.transform:
            obs = self.transform(image=obs)["image"]

        obs = torch.tensor(obs).permute(2, 0, 1).float() / 255.0
        action = torch.tensor(action).float()
        reward = torch.tensor(reward).float()
        terminal = torch.tensor(terminal).float()

        return {
            "observation": obs,
            "action": action,
            "reward": reward,
            "terminal": terminal,
        }

    def _data_per_sequence(self, data_length):
        return data_length

    def load_next_buffer(self):
        self.buffer_fnames = self.files[
            self.buffer_idx : self.buffer_idx + self.buffer_size
        ]
        self.buffer_idx += self.buffer_size
        self.buffer_idx = self.buffer_idx % len(self.files)
        self.buffer = []
        self.cum_size = [0]

        for f in self.buffer_fnames:
            with np.load(f) as data:
                self.buffer += [{k: np.copy(v) for k, v in data.items()}]
                self.cum_size += [
                    self.cum_size[-1] + self._data_per_seqence(data["rewards"].shape[0])
                ]


class ObservationDataset(RolloutDataset):
    def _get_data(self, data, idx: int):
        obs = data["observations"][idx]
        if self.transform:
            transformed = self.transform(image=obs)
            obs = transformed["image"]
        obs = torch.tensor(obs).float()
        obs = obs / 255.0
        return obs


class SequenceDataset(RolloutDataset):
    def __init__(
        self,
        root: str,
        transform: Compose,
        train: bool,
        buffer_size: int,
        num_test_files: int,
        seq_len: int,
    ):
        super().__init__(root, transform, train, buffer_size, num_test_files)
        self.seq_len = seq_len

    def _get_data(self, data, idx: int):
        obs_data = data["observations"][idx : idx + self.seq_len]
        if self.transform:
            transformed = [self.transform(image=obs) for obs in obs_data]
            obs_data = [t["image"] for t in transformed]

        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data["actions"][idx + 1 : idx + self.seq_len + 1]
        action = action.astype(np.float32)
        reward = data["rewards"][idx + 1 : idx + self.seq_len + 1]
        terminal = data["terminals"][idx + 1 : idx + self.seq_len + 1].astype(
            np.float32
        )
        return obs, action, reward, terminal, next_obs

    def _data_per_sequence(self, data_length):
        return data_length - self.seq_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollouts", help="number of rollouts", type=int, default=10_000
    )
    parser.add_argument("--threads", help="number of threads", type=int, default=24)
    parser.add_argument("--seq_len", help="sequence length", type=int, default=1000)
    parser.add_argument(
        "--dir", help="output directory", type=str, default="/data/world-models"
    )
    args = parser.parse_args()

    data_folder = (
        args.dir
        + f'/{datetime.now().strftime("%Y-%m-%d")}_rollouts-{args.rollouts}_seqlen-{args.seq_len}'
    )
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
    os.makedirs(data_folder, exist_ok=True)

    reps = args.rollouts // args.threads + 1

    p = Pool(args.threads)
    work = [
        (os.path.join(data_folder, f"thread_{i}"), args.seq_len, reps)
        for i in range(args.threads)
    ]
    print(work)
    p.map(rollout, tuple(work))
