from functools import partial
from torch.optim import Optimizer


class EarlyStopping:
    def __init__(self, mode="min", patience=10, threshold=1e-4, threshold_mode="rel"):
        self.patience = patience
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None
        self.is_better = None
        self.last_epoch = -1
        self._init_is_better(mode, threshold, threshold_mode)
        self._reset()

    def _reset(self):
        self.best = self.mode_worse
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

    @property
    def stop(self):
        return self.num_bad_epochs > self.patience

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == "min" and threshold_mode == "rel":
            rel_epsilon = 1.0 - threshold
            return a < best * rel_epsilon
        elif mode == "min" and threshold_mode == "abs":
            return a < best - threshold
        elif mode == "max" and threshold_mode == "rel":
            rel_epsilon = 1.0 + threshold
            return a > best * rel_epsilon

        return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")
        if mode == "min":
            self.mode_worse = float("inf")
        else:
            self.mode_worse = -float("inf")
        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("is_better",)
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(self.mode, self.threshold, self.threshold_mode)


class ReduceLROnPlateau:
    def __init__(
        self,
        optimizer: Optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        min_lr=0,
        eps=1e-8,
    ):
        self.optimizer = optimizer
        self.factor = factor
        self.min_lr = min_lr
        self.eps = eps
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None
        self.is_better = None
        self.last_epoch = -1
        self._init_is_better(mode, threshold, threshold_mode)
        self._reset()

    def _reset(self):
        self.best = self.mode_worse
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0

    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr

    @property
    def lr(self):
        return [param_group["lr"] for param_group in self.optimizer.param_groups]

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == "min" and threshold_mode == "rel":
            rel_epsilon = 1.0 - threshold
            return a < best * rel_epsilon
        elif mode == "min" and threshold_mode == "abs":
            return a < best - threshold
        elif mode == "max" and threshold_mode == "rel":
            rel_epsilon = 1.0 + threshold
            return a > best * rel_epsilon

        return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")
        if mode == "min":
            self.mode_worse = float("inf")
        else:
            self.mode_worse = -float("inf")
        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("is_better",)
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(self.mode, self.threshold, self.threshold_mode)
