"""Export utilities for production deployment.

The public entry point is ``obj.export(path, format="onnx")``. Importing this
module installs that method on every ``torch.nn.Module`` once, so all TorchWM
models get ONNX, TorchScript, and TensorRT export support without each model
subclassing a TorchWM-specific base class. Non-``nn.Module`` agent wrappers can
inherit :class:`ExportableAgentMixin`, which uses the same resolver and exporter.
"""

from __future__ import annotations

from importlib import import_module, util
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

ExportFormat = Literal["onnx", "torchscript", "tensorrt"]

_FORMAT_ALIASES = {
    "onnx": "onnx",
    "torchscript": "torchscript",
    "torch-script": "torchscript",
    "script": "torchscript",
    "jit": "torchscript",
    "ts": "torchscript",
    "pt": "torchscript",
    "tensorrt": "tensorrt",
    "tensor-rt": "tensorrt",
    "trt": "tensorrt",
}

_PREFERRED_TARGET_SUFFIXES = (
    "actor",
    "policy",
    "actor_critic",
    "rssm",
    "model",
    "world_model",
    "encoder",
)


def _normalize_format(format: str) -> ExportFormat:
    try:
        return _FORMAT_ALIASES[format.strip().lower().replace("_", "-")]  # type: ignore[return-value]
    except KeyError as exc:
        supported = ", ".join(sorted(set(_FORMAT_ALIASES.values())))
        raise ValueError(
            f"Unsupported export format {format!r}. Use one of: {supported}."
        ) from exc


def _as_path(path: str | Path) -> Path:
    export_path = Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    return export_path


def _inputs_to_args(example_inputs: Any) -> tuple[Any, ...]:
    if isinstance(example_inputs, tuple):
        return example_inputs
    if isinstance(example_inputs, list):
        return tuple(example_inputs)
    return (example_inputs,)


def _resolve_attr_path(obj: Any, target: str) -> Any:
    current = obj
    for part in target.split("."):
        if not part:
            continue
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _discover_modules(obj: Any) -> dict[str, nn.Module]:
    modules: dict[str, nn.Module] = {}
    seen: set[int] = set()

    def visit(value: Any, prefix: str) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, nn.Module):
            modules[prefix] = value
            for name, child in value.named_children():
                visit(child, f"{prefix}.{name}" if prefix else name)
            return
        if isinstance(value, dict):
            for name, child in value.items():
                if isinstance(name, str):
                    visit(child, f"{prefix}.{name}" if prefix else name)
            return
        if isinstance(value, (list, tuple)):
            for idx, child in enumerate(value):
                visit(child, f"{prefix}.{idx}" if prefix else str(idx))
            return
        for name, child in vars(value).items() if hasattr(value, "__dict__") else []:
            if name.startswith("_"):
                continue
            if isinstance(child, (nn.Module, dict, list, tuple)) or hasattr(
                child, "__dict__"
            ):
                visit(child, f"{prefix}.{name}" if prefix else name)

    visit(obj, "")
    return {name: module for name, module in modules.items() if name}


class DreamerPolicyExport(nn.Module):
    """Traceable Dreamer policy head used by the generic export resolver."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor(features, deter=True)


class IRISActorCriticExport(nn.Module):
    """Traceable IRIS policy/value head used by the generic export resolver."""

    def __init__(self, agent: Any):
        super().__init__()
        self.agent = agent

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_logits, values, _ = self.agent.forward_actor_critic(frames)
        return action_logits, values


def _dreamer_default(obj: Any, target: str | None) -> nn.Module | None:
    dreamer = getattr(obj, "dreamer", None)
    if dreamer is None:
        return None
    if target in {None, "actor", "dreamer.actor"} and hasattr(dreamer, "actor"):
        return DreamerPolicyExport(dreamer.actor).to(dreamer.device)
    return None


def _iris_default(obj: Any, target: str | None) -> nn.Module | None:
    if target in {None, "actor_critic"} and hasattr(obj, "forward_actor_critic"):
        return IRISActorCriticExport(obj).to(obj.device)
    return None


def _jepa_default(obj: Any, target: str | None) -> nn.Module | None:
    if type(obj).__name__ != "JEPAAgent" or target not in {None, "encoder"}:
        return None
    encoder = getattr(obj, "encoder", None)
    if encoder is not None:
        return encoder
    cfg = obj.cfg
    vit = import_module("torchwm.models.vit")
    factory = getattr(vit, cfg.model_name)
    encoder = factory(img_size=[cfg.crop_size], patch_size=cfg.patch_size)
    setattr(obj, "encoder", encoder)
    return encoder


def _resolve_export_module(obj: Any, target: str | None = None) -> nn.Module:
    for adapter in (_dreamer_default, _iris_default, _jepa_default):
        module = adapter(obj, target)
        if module is not None:
            return module

    if target is not None:
        try:
            module = _resolve_attr_path(obj, target)
        except (AttributeError, KeyError, IndexError):
            modules = _discover_modules(obj)
            matches = [
                module
                for name, module in modules.items()
                if name == target or name.split(".")[-1] == target
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                available = ", ".join(
                    sorted(
                        name
                        for name in modules
                        if name == target or name.split(".")[-1] == target
                    )
                )
                raise ValueError(
                    f"Export target {target!r} matched multiple modules: {available}. "
                    "Use a fully qualified target path."
                )
            raise
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"Export target {target!r} resolved to {type(module).__name__}, not torch.nn.Module."
            )
        return module

    if isinstance(obj, nn.Module):
        return obj

    modules = _discover_modules(obj)
    if not modules:
        raise TypeError(
            f"{type(obj).__name__} does not contain a torch.nn.Module to export. "
            "Attach a module attribute or pass target='path.to.module'."
        )
    if len(modules) == 1:
        return next(iter(modules.values()))
    for suffix in _PREFERRED_TARGET_SUFFIXES:
        for name, module in modules.items():
            if name.split(".")[-1] == suffix:
                return module
    available = ", ".join(sorted(modules))
    raise ValueError(
        f"{type(obj).__name__} contains multiple exportable modules. "
        f"Pass target=<name>; available targets: {available}."
    )


def _infer_example_inputs(
    obj: Any, module: nn.Module, target: str | None
) -> Any | None:
    if hasattr(obj, "dreamer") and target in {None, "actor", "dreamer.actor"}:
        args = obj.args
        return torch.zeros(
            1, args.stoch_size + args.deter_size, device=obj.dreamer.device
        )
    if hasattr(obj, "dreamer") and target in {"obs_encoder", "dreamer.obs_encoder"}:
        return torch.zeros(1, *obj.dreamer.obs_shape, device=obj.dreamer.device)
    if hasattr(obj, "dreamer") and target in {
        "reward_model",
        "value_model",
        "discount_model",
    }:
        args = obj.args
        return torch.zeros(
            1, args.stoch_size + args.deter_size, device=obj.dreamer.device
        )
    if hasattr(obj, "forward_actor_critic") and target in {None, "actor_critic"}:
        frame_shape = obj.config.get_frame_shape()
        return torch.zeros(1, 1, *frame_shape, device=obj.device)
    if type(obj).__name__ == "JEPAAgent" and target in {None, "encoder"}:
        device = next(module.parameters(), torch.empty(0)).device
        return torch.zeros(1, 3, obj.cfg.crop_size, obj.cfg.crop_size, device=device)
    if hasattr(obj, "num_frames") and hasattr(obj, "image_size"):
        device = next(module.parameters(), torch.empty(0)).device
        return torch.zeros(
            1, 3, obj.num_frames, obj.image_size, obj.image_size, device=device
        )
    if (
        hasattr(obj, "env")
        and hasattr(obj, "device")
        and module is getattr(obj, "rssm", None)
    ):
        obs = torch.zeros(1, 2, *obj.env.observation_size, device=obj.device)
        actions = torch.zeros(1, 1, obj.env.action_size, device=obj.device)
        return obs, actions
    return None


class ExportableAgentMixin:
    """Mixin for non-``nn.Module`` agents that delegates to the shared exporter."""

    def export(
        self,
        path: str | Path,
        format: str = "onnx",
        *,
        example_inputs: Any | None = None,
        target: str | None = None,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        opset_version: int = 17,
        **kwargs: Any,
    ) -> Path:
        """Export this agent or one of its contained modules for deployment."""

        return export_any(
            self,
            path,
            format=format,
            example_inputs=example_inputs,
            target=target,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            **kwargs,
        )


def export_any(
    obj: Any,
    path: str | Path,
    format: str = "onnx",
    *,
    example_inputs: Any | None = None,
    target: str | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 17,
    **kwargs: Any,
) -> Path:
    """Export any TorchWM model/agent or a target module contained by it."""

    module = _resolve_export_module(obj, target)
    if example_inputs is None:
        example_inputs = _infer_example_inputs(obj, module, target)
    return export_model(
        module,
        path,
        format=format,
        example_inputs=example_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        **kwargs,
    )


def export_model(
    module: nn.Module,
    path: str | Path,
    format: str = "onnx",
    *,
    example_inputs: Any | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 17,
    **kwargs: Any,
) -> Path:
    """Export a ``torch.nn.Module`` to ONNX, TorchScript, or TensorRT."""

    export_format = _normalize_format(format)
    export_path = _as_path(path)
    was_training = module.training
    module.eval()

    try:
        with torch.inference_mode():
            if export_format == "onnx":
                if example_inputs is None:
                    raise ValueError("example_inputs is required for ONNX export.")
                torch.onnx.export(
                    module,
                    _inputs_to_args(example_inputs),
                    str(export_path),
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    do_constant_folding=kwargs.pop("do_constant_folding", True),
                    **kwargs,
                )
            elif export_format == "torchscript":
                if example_inputs is None:
                    exported = torch.jit.script(module, **kwargs)
                else:
                    exported = torch.jit.trace(  # type: ignore[no-untyped-call]
                        module,
                        _inputs_to_args(example_inputs),
                        strict=kwargs.pop("strict", False),
                        **kwargs,
                    )
                exported.save(str(export_path))
            elif export_format == "tensorrt":
                if example_inputs is None:
                    raise ValueError("example_inputs is required for TensorRT export.")
                if util.find_spec("torch_tensorrt") is None:
                    raise RuntimeError(
                        "TensorRT export requires the optional torch-tensorrt package. "
                        "Install torch-tensorrt in your deployment environment or export ONNX first."
                    )
                torch_tensorrt = import_module("torch_tensorrt")
                trt_inputs = kwargs.pop("inputs", list(_inputs_to_args(example_inputs)))
                enabled_precisions = kwargs.pop("enabled_precisions", {torch.float32})
                compiled = torch_tensorrt.compile(
                    module,
                    ir=kwargs.pop("ir", "ts"),
                    inputs=trt_inputs,
                    enabled_precisions=enabled_precisions,
                    **kwargs,
                )
                torch.jit.save(compiled, str(export_path))
            else:  # pragma: no cover - guarded by _normalize_format.
                raise AssertionError(f"Unhandled export format: {export_format}")
    finally:
        module.train(was_training)

    return export_path


def _module_export(
    self: nn.Module, path: str | Path, format: str = "onnx", **kwargs: Any
) -> Path:
    return export_any(self, path, format=format, **kwargs)


def install_export_method() -> None:
    """Install ``torch.nn.Module.export`` once for every Torch model class."""

    if getattr(nn.Module, "_torchwm_export_installed", False):
        return
    nn.Module.export = _module_export  # type: ignore[attr-defined]
    nn.Module._torchwm_export_installed = True  # type: ignore[attr-defined]


install_export_method()


__all__ = [
    "DreamerPolicyExport",
    "ExportFormat",
    "ExportableAgentMixin",
    "IRISActorCriticExport",
    "export_any",
    "export_model",
    "install_export_method",
]
