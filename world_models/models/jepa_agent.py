import os
import datetime
from world_models.configs.jepa_config import JEPAConfig
from world_models.training.train_jepa import main as train_jepa_main


class JEPAAgent:
    def __init__(self, config: JEPAConfig | None = None, **kwargs):
        self.cfg = config if config is not None else JEPAConfig()
        for key, val in kwargs.items():
            if key == "logdir":
                self.cfg.folder = val
            elif hasattr(self.cfg, key):
                setattr(self.cfg, key, val)
            else:
                raise ValueError(f"Invalid argument: {key}")
        if not getattr(self.cfg, "write_tag", None):
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.cfg.write_tag = f"jepa_{ts}"
        os.makedirs(self.cfg.folder, exist_ok=True)

    def train(self):
        train_jepa_main(self.cfg)
