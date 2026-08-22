# Exporting Models for Deployment

TorchWM provides a unified export system that converts trained models and agents
into deployable formats — ONNX, TorchScript, and TensorRT — without requiring
each model to implement its own export logic.

```{contents} Contents
:depth: 3
```

## Overview

The export system is built around three levels of API:

| Level | Function / Method | When to use |
|---|---|---|
| **Module method** | `module.export(path, format, ...)` | Any `torch.nn.Module` — works automatically after importing `torchwm.export` |
| **Agent method** | `agent.export(path, format, ...)` | High-level agents (`DreamerAgent`, `JEPAAgent`, `IRISAgent`) that inherit `ExportableAgentMixin` |
| **Standalone** | `export_any(obj, path, ...)` / `export_model(module, path, ...)` | When you need explicit control over which submodule is exported or want to bypass automatic target resolution |

### Supported formats

| Format | Extension | Use case |
|---|---|---|
| `"onnx"` | `.onnx` | Cross-platform inference, mobile, edge devices, TensorRT conversion |
| `"torchscript"` (aliases: `"jit"`, `"ts"`, `"pt"`, `"script"`) | `.pt` | Serving via LibTorch (C++), no Python dependency at inference time |
| `"tensorrt"` (aliases: `"trt"`) | `.pt` | NVIDIA GPU-optimized inference (requires `torch_tensorrt` package) |

## Quick start

### Exporting any `nn.Module`

Importing `torchwm.export` installs the `.export()` method on every
`torch.nn.Module` instance once:

```python
import torch
import torchwm.export  # installs nn.Module.export

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 10)

    def forward(self, x):
        return self.linear(x)

model = MyModel()
model.eval()

# TorchScript (scripted — works without example_inputs)
model.export("model.pt", format="torchscript")

# ONNX (requires example_inputs)
model.export("model.onnx", format="onnx", example_inputs=torch.zeros(1, 64))
```

### Exporting a trained agent

High-level agents support the same `.export()` method directly:

```python
import torchwm

agent = torchwm.create_model("dreamer", env="walker-walk", total_steps=1000)
# ... train the agent ...

# Export the policy to ONNX
agent.export("policy.onnx", format="onnx")

# Export the RSSM world model to TorchScript
agent.export("rssm.pt", format="torchscript", target="rssm")

# Export the observation encoder to ONNX
agent.export("encoder.onnx", format="onnx", target="obs_encoder")
```

The system automatically resolves which submodule to export and infers the
correct example inputs for each agent type.

## Target resolution

When you call `.export()`, the system needs to decide which `nn.Module` to
serialize. It uses a priority-based resolution strategy:

### 1. Agent-specific defaults

Each agent type has a preferred default target:

| Agent | Default target | Example inputs |
|---|---|---|
| `DreamerAgent` | Policy head (actor) | `[batch, stoch_size + deter_size]` |
| `IRISAgent` | Actor-critic head | `[batch, 1, channels, h, w]` |
| `JEPAAgent` | Vision Transformer encoder | `[batch, 3, crop_size, crop_size]` |

### 2. Explicit `target=` parameter

Override the default by naming a specific submodule:

```python
# Export by attribute path
agent.export("reward.pt", format="torchscript", target="dreamer.reward_model")

# If the attribute name is unique, the short name works
agent.export("value.pt", format="torchscript", target="value_model")
```

If the short name matches multiple modules, the system raises an error and
lists the available fully qualified paths.

### 3. Single-module fallback

If the object is itself an `nn.Module`, it is exported directly. If it
contains exactly one `nn.Module` attribute, that attribute is exported.
If it contains multiple modules, the system picks the first match from this
priority list: `actor`, `policy`, `actor_critic`, `rssm`, `model`,
`world_model`, `encoder`.

## Format-specific details

### ONNX

```python
agent.export(
    "policy.onnx",
    format="onnx",
    input_names=["latent"],
    output_names=["action"],
    dynamic_axes={"latent": {0: "batch"}, "action": {0: "batch"}},
    opset_version=17,        # default
)
```

See the [PyTorch ONNX export docs](https://pytorch.org/docs/stable/onnx.html)
for all supported keyword arguments.

### TorchScript

Two modes, controlled by whether `example_inputs` is provided:

| Mode | When to use | Limitation |
|---|---|---|
| **Tracing** (with `example_inputs`) | Fast export of a fixed forward graph | May not handle dynamic control flow |
| **Scripting** (without `example_inputs`) | Full module graph with control flow | May fail on unsupported Python constructs |

```python
# Trace (requires example_inputs)
agent.export("traced.pt", format="torchscript", example_inputs=torch.zeros(1, 230))

# Script (no example_inputs needed)
agent.export("scripted.pt", format="torchscript")
```

### TensorRT

Requires the optional `torch_tensorrt` package:

```sh
pip install torch_tensorrt
```

```python
agent.export(
    "policy.trt",
    format="tensorrt",
    example_inputs=torch.zeros(1, 230, device="cuda"),
    enabled_precisions={torch.float16},  # FP16 inference
)
```

## Custom agents

If you build a custom agent that is not an `nn.Module`, inherit
`ExportableAgentMixin` to get the `.export()` method:

```python
from torchwm import ExportableAgentMixin

class MyAgent(ExportableAgentMixin):
    def __init__(self):
        self.policy = torch.nn.Linear(64, 5)
        self.encoder = torch.nn.Linear(1024, 64)
```

The mixin will automatically discover `self.policy` and prefer it as the
default target. Pass `target="encoder"` to export a different submodule.

### Custom example input inference

If the auto-inferred example inputs are wrong for your agent, pass them
explicitly:

```python
agent.export(
    "policy.onnx",
    format="onnx",
    example_inputs=torch.zeros(1, 128),  # your custom shape
)
```

Or add inference support by implementing a matching pattern in
`_infer_example_inputs` in `torchwm/export.py`.

## Low-level API

For scripting or batch export, use the standalone functions directly:

```python
from torchwm import export_any, export_model

# export_any resolves the target module from any object
export_any(agent, "policy.onnx", format="onnx")

# export_model exports a raw nn.Module
export_model(agent.policy, "policy.pt", format="torchscript")
```

## Common pitfalls

1. **Missing `.eval()`**: Export always sets the module to eval mode before
   tracing and restores the original mode afterwards. Call `.eval()` manually
   if you are inspecting the exported graph afterwards.

2. **Dynamic control flow with ONNX**: ONNX requires tracing. If your module
   has `if` statements or loops that depend on tensor values, use TorchScript
   instead.

3. **CUDA tensors for TensorRT**: TensorRT export requires example inputs on
   the same device as the module. Pass CUDA tensors as example inputs.

4. **Multiple matches for short names**: If you see `"matched multiple
   modules"`, use the fully qualified path, e.g. `target="dreamer.actor"`
   instead of `target="actor"`.

## See Also

- {doc}`inference_guide` — running exported models in production
- {doc}`public_api` — `export_any` and `export_model` factory functions
