# Public API Quick Reference

TorchWM exposes `torchwm` as the friendly public namespace for both
application code and direct component imports. Use it for factory helpers,
operators, model classes, config classes, and environment constructors.

## Common Workflow

```python
import torchwm

# Discover supported factories.
models = torchwm.list_models()
backends = torchwm.list_env_backends()

# Configure and create a model in one step.
agent = torchwm.create_model(
    "dreamer",
    env_backend="dmc",
    env="walker-walk",
    total_steps=1_000_000,
)

# Create standalone environments through a consistent backend selector.
env = torchwm.make_env("CartPole-v1", backend="gym")

# Build preprocessing operators without importing deep modules.
op = torchwm.get_operator("dreamer", image_size=64, action_dim=6)
```

## Factory Helpers

| Helper | Purpose |
|--------|---------|
| `create_config(model, **overrides)` | Build the default config for a model family and apply validated overrides. |
| `create_model(model, config=None, **overrides)` | Instantiate a model or high-level agent by name. Config overrides are applied before construction; unknown fields are passed as constructor arguments when appropriate. |
| `make_env(env_id, backend="auto", **kwargs)` | Create an environment through a named backend such as `gym`, `atari`, `mujoco`, `robotics`, `brax`, or `unity`. |
| `list_models()` | Return canonical names accepted by `create_model`. |
| `list_env_backends()` | Return backend names accepted by `make_env`. |
| `list_envs(model=None)` | Return known environment IDs, optionally filtered by model family. |

## Direct Imports Still Work

The factory API is a convenience layer. Advanced and research workflows can keep
using direct imports from `torchwm`:

```python
from torchwm import DreamerAgent, DreamerConfig, RSSM

cfg = DreamerConfig()
cfg.env = "walker-walk"
agent = DreamerAgent(cfg)
```

Use direct imports when you are composing custom modules, subclassing internals,
or need access to implementation-specific constructors.
