import numpy as np
import cv2
from PIL import Image
from ale_py import ALEInterface
import gym


class Atari:
    LOCK = None
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=4,
        size=(84, 84),
        gray=True,
        noops=0,
        lives="unused",
        sticky=True,
        actions="all",
        length=108000,
        resize="opencv",
        seed=None,
    ):
        assert size[0] == size[1]
        assert lives in ("unused", "discount", "reset")
        assert actions in ("all", "needed")
        assert resize in ("opencv", "pillow")

        # Lock for environments in multiprocessing
        if Atari.LOCK is None:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            Atari.LOCK = ctx.Lock()

        self._repeat = action_repeat
        self._size = size
        self._gray = gray
        self._lives = lives
        self._length = length
        self._noops = noops
        self._sticky = sticky
        self._random = np.random.RandomState(seed)

        # Resize backend
        self._resize = resize
        if resize == "opencv":
            self._cv2 = cv2
        else:
            self._image = Image

        with Atari.LOCK:
            ale = ALEInterface()

        # Sticky action
        ale.setFloat("repeat_action_probability", 0.25 if sticky else 0.0)

        # Random seed
        if seed is not None:
            ale.setInt("random_seed", int(seed))

        # Load ROM
        rom_path = f"{name}.bin"  # you must ensure ROM files exist
        ale.loadROM(rom_path.encode("utf-8"))

        self._ale = ale
        self._actions = (
            ale.getMinimalActionSet()
            if actions == "needed"
            else ale.getLegalActionSet()
        )

        # Create screen buffers
        screen_shape = (ale.getScreenHeight(), ale.getScreenWidth(), 3)
        self._buffer = [
            np.zeros(screen_shape, dtype=np.uint8),
            np.zeros(screen_shape, dtype=np.uint8),
        ]

        self._step = 0
        self._done = True
        self._last_lives = None
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        img_shape = self._size + ((1,) if self._gray else (3,))
        return gym.spaces.Dict({"image": gym.spaces.Box(0, 255, img_shape, np.uint8)})

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reset(self):
        self._ale.reset_game()

        # No-op resets
        if self._noops:
            for _ in range(self._random.randint(self._noops)):
                self._ale.act(0)
                if self._ale.game_over():
                    self._ale.reset_game()

        self._last_lives = self._ale.lives()
        self._step = 0
        self._done = False

        # Fill buffer
        self._screen(self._buffer[0])
        self._buffer[1].fill(0)

        obs, _, _, _ = self._obs(0, is_first=True)
        return obs

    def step(self, action):
        total_reward = 0.0
        dead = False

        action = self._actions[action]

        for i in range(self._repeat):
            reward = self._ale.act(action)
            total_reward += reward

            # capture second-last screen for max-pooling
            if i == self._repeat - 2:
                self._screen(self._buffer[1])

            if self._ale.game_over():
                break

            # Life loss detection
            if self._lives != "unused":
                current_lives = self._ale.lives()
                if current_lives < self._last_lives:
                    dead = True
                    self._last_lives = current_lives
                    break

        self._screen(self._buffer[0])

        self._step += 1
        over = self._ale.game_over()
        self._done = over or (self._length and self._step >= self._length)

        return self._obs(
            total_reward,
            is_last=self._done or (dead and self._lives == "reset"),
            is_terminal=dead or over,
        )

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        # Max over last 2 frames (standard Atari preprocessing)
        np.maximum(self._buffer[0], self._buffer[1], out=self._buffer[0])
        image = self._buffer[0]

        # Resize
        if image.shape[:2] != self._size:
            if self._resize == "opencv":
                image = self._cv2.resize(
                    image, self._size, interpolation=self._cv2.INTER_AREA
                )
            else:
                image = self._image.fromarray(image).resize(self._size, Image.NEAREST)
                image = np.array(image)

        # Grayscale
        if self._gray:
            image = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            image = image[:, :, None]

        return (
            {"image": image, "is_terminal": is_terminal, "is_first": is_first},
            reward,
            is_last,
            {},
        )

    def _screen(self, array):
        self._ale.getScreenRGB(array)

    def close(self):
        pass
