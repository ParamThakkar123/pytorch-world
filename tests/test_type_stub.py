"""The shipped stub must cover the whole public surface, and stay in sync.

``torchwm`` ships ``py.typed``, so type checkers trust ``torchwm/__init__.pyi``
completely.  A stale stub is therefore worse than no stub: it silently hands
users ``Any`` for symbols that do exist, or reports missing ones that do not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STUB_PATH = REPO_ROOT / "torchwm" / "__init__.pyi"


def _stub_tree() -> ast.Module:
    return ast.parse(STUB_PATH.read_text(encoding="utf-8"))


def _stub_names() -> set[str]:
    """Names the stub binds - aliased imports plus annotated variables."""

    names: set[str] = set()
    for node in _stub_tree().body:
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_stub_declares_every_public_export():
    import torchwm

    missing = sorted(set(torchwm.__all__) - _stub_names())
    assert not missing, f"symbols missing from {STUB_PATH.name}: {missing}"


def test_stub_is_regenerated_from_the_current_export_map():
    pytest.importorskip("torch")
    from tools.gen_type_stub import main

    assert main(["--check"]) == 0


def test_stub_re_exports_resolve_to_the_runtime_objects():
    # Every ``from X import Y as Z`` in the stub must name the module that
    # really defines the object ``torchwm.Z`` returns, or editors send users to
    # a symbol that is not there.
    import importlib

    import torchwm

    mismatched = []
    for node in _stub_tree().body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        for alias in node.names:
            exported = alias.asname or alias.name
            if exported not in torchwm.__all__:
                continue
            source = importlib.import_module(node.module)
            if getattr(source, alias.name, None) is not getattr(torchwm, exported):
                mismatched.append(f"{node.module}:{alias.name} as {exported}")

    assert not mismatched, f"stub re-exports point at the wrong object: {mismatched}"


def test_both_packages_ship_py_typed():
    # ``torchwm``'s stub re-exports from ``world_models``; without a marker on
    # the implementation package, checkers treat those imports as untyped.
    assert (REPO_ROOT / "torchwm" / "py.typed").exists()
    assert (REPO_ROOT / "world_models" / "py.typed").exists()
