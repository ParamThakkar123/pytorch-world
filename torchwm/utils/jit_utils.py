import torch
from typing import Callable


def jit_compile_function(func: Callable) -> Callable:
    """JIT compile a function for performance."""
    try:
        return torch.jit.script(func)
    except Exception as e:
        print(f"JIT compilation failed: {e}")
        return func


def jit_compile_module(module: torch.nn.Module) -> torch.nn.Module:
    """JIT compile a PyTorch module."""
    try:
        return torch.jit.script(module)
    except Exception as e:
        print(f"JIT compilation failed: {e}")
        return module
