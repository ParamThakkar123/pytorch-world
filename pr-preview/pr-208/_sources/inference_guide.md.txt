# Inference Guide

This guide covers how to use trained TorchWM models for inference and deployment.

## Overview

TorchWM agents load from a checkpoint, run in `eval()` mode, and take the same
tensors they were trained on. Each model's page documents the observation
layout it expects; this guide covers the mechanics around it.

```{contents} Contents
```

## Loading Trained Models

```python
from torchwm import DreamerAgent

# Load from checkpoint
agent = DreamerAgent.from_pretrained("path/to/checkpoint")
agent.eval()
```

## Basic Inference

### Dreamer

```python
import torch
from torchwm import DreamerAgent

agent = DreamerAgent.from_pretrained("dreamer_checkpoint")

# Single step prediction. Observations are float tensors in [0, 1] shaped
# [batch, channels, height, width]; actions are [batch, action_dim].
obs = torch.rand(1, 3, 64, 64)
action = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])

with torch.no_grad():
    next_obs, reward = agent.predict({"obs": obs, "action": action})
```

### JEPA

I-JEPA is evaluated with a frozen encoder, so inference means extracting
representations rather than rolling out a policy. Load the EMA target-encoder
from a training checkpoint and average-pool its patch tokens, exactly as the
paper's linear-evaluation protocol does:

```python
import torch
from torchwm.training.eval_jepa import load_jepa_encoder, make_eval_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = load_jepa_encoder("results/jepa/jepa_run-latest.pth.tar", device)

transform = make_eval_transforms(crop_size=224)
images = torch.stack([transform(pil_image) for pil_image in batch]).to(device)

with torch.no_grad():
    representations = encoder(images).mean(dim=1)  # [batch, embed_dim]
```

To reproduce the paper's ImageNet linear-probe number, use
`torchwm.training.eval_jepa.jepa_linear_probe` instead, which trains the
linear head on top of these features.

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

with torch.no_grad():
    predictions = agent.predict_batch({"obs": obs_batch, "action": action_batch})
```

## GPU Acceleration

Move to GPU for faster inference:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = agent.to(device)
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    output = agent.predict(inputs)
```

## Real-time Inference

For interactive applications:

```python
import torch
from torchwm import DreamerAgent

class InferenceServer:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.agent = DreamerAgent.from_pretrained("checkpoint").to(self.device).eval()

    def predict(self, obs, action):
        inputs = {
            "obs": obs.to(self.device),
            "action": action.to(self.device),
        }
        with torch.no_grad():
            return self.agent.predict(inputs)

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
    output = agent.predict(inputs)
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
        with torch.no_grad():
            next_obs_pred, reward_pred = agent.predict(
                {"obs": obs, "action": action}
            )

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
- Ensure inputs are normalized the same way as during training
- Check model loading
- Verify input shapes match training