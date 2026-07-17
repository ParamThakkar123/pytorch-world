# Inference Guide

This guide covers how to use trained TorchWM models for inference and deployment.

## Overview

TorchWM provides standardized inference through operators and future pipelines.
For application code, prefer the top-level `torchwm.get_operator()` factory; it
keeps examples short and avoids deep imports.

```{contents} Contents
```

## Loading Trained Models

```python
from torchwm import DreamerAgent

# Load from checkpoint
agent = DreamerAgent.from_pretrained("path/to/checkpoint")
agent.eval()
```

## Using Operators for Preprocessing

See {doc}`operators_guide` for detailed operator usage.

## Basic Inference

### Dreamer

```python
import torch
import torchwm
from torchwm import DreamerAgent

op = torchwm.get_operator("dreamer", image_size=64, action_dim=6)
agent = DreamerAgent.from_pretrained("dreamer_checkpoint")

# Single step prediction
obs = torch.randn(3, 64, 64)
action = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

with torch.no_grad():
    processed = op({'image': obs, 'action': action})
    next_obs, reward = agent.predict(processed)
```

### JEPA

```python
import torch
import torchwm
from torchwm import JEPAAgent

op = torchwm.get_operator("jepa", image_size=224, patch_size=16, mask_ratio=0.75)
agent = JEPAAgent.from_pretrained("jepa_checkpoint")

# Representation learning
images = [torch.randn(3, 224, 224) for _ in range(8)]
processed = op({'images': images})

with torch.no_grad():
    representations = agent.encode(processed)
```

## Rollout and Imagination

Generate imagined trajectories:

```python
# Dreamer imagination
from torchwm import DreamerAgent

agent = DreamerAgent.from_pretrained("dreamer_checkpoint")

initial_obs = torch.randn(3, 64, 64)
horizon = 10

imagined_trajectory = agent.imagine_rollout(initial_obs, horizon)
# Returns dict with imagined observations, actions, rewards
```

## Batch Inference

Process multiple inputs efficiently:

```python
batch_size = 32
obs_batch = torch.randn(batch_size, 3, 64, 64)
action_batch = torch.randn(batch_size, 6)

processed = op({'image': obs_batch, 'action': action_batch})

with torch.no_grad():
    predictions = agent.predict_batch(processed)
```

## GPU Acceleration

Move to GPU for faster inference:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = agent.to(device)
processed = {k: v.to(device) for k, v in processed.items()}

with torch.no_grad():
    output = agent.predict(processed)
```

## Real-time Inference

For interactive applications:

```python
import torch
import torchwm
from torchwm import DreamerAgent

class InferenceServer:
    def __init__(self):
        self.agent = DreamerAgent.from_pretrained("checkpoint").eval()
        self.op = torchwm.get_operator("dreamer", image_size=64, action_dim=6)

    def predict(self, obs, action):
        processed = self.op({'image': obs, 'action': action})
        with torch.no_grad():
            return self.agent.predict(processed)

server = InferenceServer()
```

## Performance Optimization

### JIT Compilation

```python
import torch

agent = torch.jit.script(agent)
```

### Memory Efficient Inference

```python
import torch

with torch.inference_mode():
    output = agent.predict(processed)
```

## Exporting Models

TorchWM installs a deployment-oriented `export()` method once on `torch.nn.Module`, so every model class in the library can be exported with the same API. High-level wrapper agents such as Dreamer and PlaNet use the same exporter for their contained modules:

```python
model.export("model.onnx", format="onnx", example_inputs=example_inputs)
agent.export("agent_actor.onnx", format="onnx")
```

| Format | Alias | Output |
|---|---|---|---|
| `"onnx"` | — | ONNX graph for ONNX Runtime, TensorRT conversion, or other production runtimes |
| `"torchscript"` | `"jit"`, `"ts"` | TorchScript `.pt` file |
| `"tensorrt"` | `"trt"` | Serialized TorchScript TensorRT module (requires `torch-tensorrt`) |

Dreamer exports its deterministic actor by default. The exported Dreamer actor
accepts concatenated latent features with shape `[batch, stoch_size + deter_size]`
and returns actions:

```python
import torch
from torchwm import DreamerAgent

agent = DreamerAgent(env="cartpole_balance")
agent.export("dreamer_actor.onnx", format="onnx")
agent.export("dreamer_actor.pt", format="torchscript")

features = torch.zeros(1, agent.args.stoch_size + agent.args.deter_size)
agent.export(
    "dreamer_actor_dynamic.onnx",
    format="onnx",
    example_inputs=features,
    input_names=["features"],
    output_names=["actions"],
    dynamic_axes={"features": {0: "batch"}, "actions": {0: "batch"}},
)
```

Export individual components by passing `target` when the agent provides more
than one deployable module:

```python
agent.export("dreamer_encoder.onnx", format="onnx", target="obs_encoder")
agent.export("dreamer_reward.pt", format="torchscript", target="reward_model")
```

For any lower-level `torch.nn.Module` model, pass `example_inputs` explicitly if TorchWM cannot infer a safe default:

```python
import torch
import torchwm

genie = torchwm.create_model("genie-small", image_size=32)
video = torch.randn(1, 3, genie.num_frames, genie.image_size, genie.image_size)
genie.export("genie_small.onnx", format="onnx", example_inputs=video)

vit = torchwm.VisionTransformer(img_size=[224])
images = torch.randn(1, 3, 224, 224)
vit.export("vit.onnx", format="onnx", example_inputs=images)
```

Agents that contain multiple deployable modules accept either short target names such as `"obs_encoder"` or fully qualified paths such as `"dreamer.obs_encoder"`. JEPA exports a ViT encoder target by default, while lower-level JEPA `VisionTransformer` modules can be exported directly like any other `torch.nn.Module`.

TensorRT export requires `torch-tensorrt` in the deployment environment:

```python
agent.export("dreamer_actor_trt.pt", format="tensorrt")
```

## Integration Examples

### With Gym Environments

```python
import torchwm
from torchwm import DreamerAgent

env = torchwm.make_env("Pendulum-v1", backend="gym")
agent = DreamerAgent.from_pretrained("pendulum_checkpoint")
op = torchwm.get_operator("dreamer", image_size=64, action_dim=env.action_space.shape[0])

obs, _ = env.reset()
done = False

while not done:
    action = agent.act(obs)  # Get action from agent
    obs, reward, done, _, _ = env.step(action)
```

### With Custom Environments

```python
class CustomEnv:
    def step(self, action):
        # Your environment logic
        return obs, reward, done

env = CustomEnv()
agent = DreamerAgent.from_pretrained("custom_checkpoint")

for episode in range(10):
    obs = env.reset()
    total_reward = 0

    while True:
        processed = op({'image': obs, 'action': action})
        with torch.no_grad():
            next_obs_pred, reward_pred = agent.predict(processed)

        # Use predictions for planning/control
        action = agent.plan(obs, next_obs_pred, reward_pred)
        obs, reward, done = env.step(action)
        total_reward += reward

        if done:
            break

    print(f"Episode {episode}: {total_reward}")
```

## Troubleshooting

### Memory Issues
- Use smaller batch sizes
- Enable gradient checkpointing
- Clear cache: `torch.cuda.empty_cache()`

### Speed Issues
- Move to GPU
- Use JIT compilation
- Batch inputs when possible

### Accuracy Issues
- Ensure proper preprocessing with operators
- Check model loading
- Verify input shapes match training