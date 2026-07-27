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
import pathlib as _pathlib
import sys
from typing import Any

_INTERNAL = "world_models"

_world_models = importlib.import_module(_INTERNAL)


class _SubmoduleAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``torchwm.<name>`` to the implementation in ``world_models.<name>``.

    Installed at the *front* of ``sys.meta_path``.  Genuine ``torchwm``
    submodules (for example ``torchwm.cli``) are detected by
    :meth:`_genuine_submodule` and declined, so the default finders still load
    them; every other name is aliased to the matching ``world_models`` submodule
    and registered under the ``torchwm`` name as well.

    Being first matters.  Aliasing ``torchwm.configs`` returns the *shared*
    ``world_models.configs`` module object, whose ``__path__`` points into
    ``world_models/``.  If the default path-based finders ran first, they would
    use that ``__path__`` to locate ``configs/diamond_config.py`` and execute it
    a second time under the name ``torchwm.configs.diamond_config``.  That
    yields two distinct classes from one file, so
    ``isinstance(cfg, DiamondConfig)`` fails depending on which namespace the
    caller imported from -- producing errors as self-contradictory as
    ``config must be a DiamondConfig ...; got DiamondConfig``.
    """

    _prefix = f"{__name__}."
    _target = f"{_INTERNAL}."
    # Directory of the real ``torchwm`` package, used to tell genuine submodules
    # apart from names that should be aliased.
    _root = _pathlib.Path(__file__).resolve().parent

    @classmethod
    def _genuine_submodule(cls, fullname: str) -> bool:
        """Return True when ``fullname`` maps to a real file under ``torchwm/``."""
        relative = fullname[len(cls._prefix) :].split(".")
        candidate = cls._root.joinpath(*relative)
        return candidate.with_suffix(".py").is_file() or (
            candidate / "__init__.py"
        ).is_file()

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(self._prefix):
            return None
        if self._genuine_submodule(fullname):
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
    # Must precede the default path-based finders; see the class docstring.
    sys.meta_path.insert(0, _SubmoduleAliasFinder())

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
