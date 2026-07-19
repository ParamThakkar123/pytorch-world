"""Friendly top-level package for TorchWM.

``torchwm`` is the recommended and complete public namespace.  Both the
top-level factory helpers and every implementation submodule are reachable
through it::

    import torchwm

    agent = torchwm.create_model("dreamer", env="walker-walk")

    from torchwm import DreamerAgent, ConvEncoder, ReplayBuffer
    from torchwm.models import Dreamer
    from torchwm.envs import make_gym_env

The real implementation lives in the internal ``world_models`` package.  It
remains importable for backward compatibility, but new code should prefer
``torchwm``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from typing import Any

_INTERNAL = "world_models"

_world_models = importlib.import_module(_INTERNAL)


class _SubmoduleAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``torchwm.<name>`` to the implementation in ``world_models.<name>``.

    Installed at the *end* of ``sys.meta_path`` so genuine ``torchwm`` submodules
    (for example ``torchwm.cli``) are found by the default finders first.  Only
    names that have no real ``torchwm`` module fall through to this alias, at
    which point the matching ``world_models`` submodule is imported lazily and
    registered under the ``torchwm`` name as well.
    """

    _prefix = f"{__name__}."
    _target = f"{_INTERNAL}."

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(self._prefix):
            return None
        return importlib.machinery.ModuleSpec(fullname, self)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Any:
        target_name = self._target + spec.name[len(self._prefix) :]
        module = importlib.import_module(target_name)
        # ``module_from_spec`` unconditionally overwrites ``module.__spec__`` with
        # our alias spec before ``exec_module`` runs.  Because we return the
        # *shared* ``world_models`` module object, that would corrupt its
        # canonical identity (``__spec__.parent`` would no longer match
        # ``__package__``, breaking relative imports).  Stash the real spec so
        # ``exec_module`` can restore it.
        spec._torchwm_canonical_spec = module.__spec__  # type: ignore[attr-defined]
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module: Any) -> None:
        # ``create_module`` returned a fully-initialised ``world_models`` module;
        # restore the canonical spec that ``module_from_spec`` overwrote.
        canonical = getattr(module.__spec__, "_torchwm_canonical_spec", None)
        if canonical is not None:
            module.__spec__ = canonical


if not any(isinstance(finder, _SubmoduleAliasFinder) for finder in sys.meta_path):
    sys.meta_path.append(_SubmoduleAliasFinder())

api = importlib.import_module(f"{_INTERNAL}.api")
sys.modules[f"{__name__}.api"] = api

__version__ = _world_models.__version__
__all__ = [*list(_world_models.__all__), "api"]


def __getattr__(name: str) -> Any:
    try:
        value = getattr(_world_models, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
