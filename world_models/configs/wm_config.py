from dataclasses import dataclass, field
from typing import Any, Dict


class WMVAEConfig:

    @dataclass
    class _Config:
        height: int
        width: int
        device: str = "cpu"
        train_batch_size: int = 32
        num_epochs: int = 10
        latent_size: int = 32
        data_dir: str = "./data"
        learning_rate: float = 1e-3
        extra: Dict[str, Any] = field(default_factory=dict)

        def validate(self) -> bool:
            assert self.height > 0 and self.width > 0, "height/width must be > 0"
            assert self.train_batch_size > 0, "train_batch_size must be > 0"
            assert self.num_epochs > 0, "num_epochs must be > 0"
            assert self.latent_size > 0, "latent_size must be > 0"
            assert self.learning_rate > 0.0, "learning_rate must be > 0"
            return True

        def to_dict(self) -> Dict[str, Any]:
            d = {k: getattr(self, k) for k in self.__annotations__ if k != "extra"}
            d.update(self.extra)
            return d

    def __init__(self, config_dict: dict):
        known_keys = {
            "height",
            "width",
            "device",
            "train_batch_size",
            "num_epochs",
            "latent_size",
            "data_dir",
            "learning_rate",
        }
        known = {k: v for k, v in config_dict.items() if k in known_keys}
        extra = {k: v for k, v in config_dict.items() if k not in known_keys}
        self._cfg = WMVAEConfig._Config(**known, extra=extra)
        self._cfg.validate()

    def __getattr__(self, item):
        return getattr(self._cfg, item)

    def to_dict(self):
        return self._cfg.to_dict()
