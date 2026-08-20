import tomllib
from pathlib import Path


def _dependency_names(dependencies):
    names = []
    for dependency in dependencies:
        if isinstance(dependency, str):
            names.append(dependency.split(">=", maxsplit=1)[0])
        else:
            names.append(dependency["name"])
    return names


def test_jax_is_brax_optional_dependency_not_core_dependency():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    assert "jax" not in _dependency_names(project["dependencies"])
    assert "jax" in _dependency_names(project["optional-dependencies"]["brax"])


def test_lockfile_keeps_jax_out_of_core_torchwm_dependencies():
    lock = tomllib.loads(Path("uv.lock").read_text())
    torchwm = next(
        package for package in lock["package"] if package["name"] == "torchwm"
    )

    assert "jax" not in _dependency_names(torchwm["dependencies"])
    assert "jax" in _dependency_names(torchwm["optional-dependencies"]["brax"])


def test_console_script_target_packages_are_included_in_setuptools_find():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    scripts = project["project"]["scripts"]
    included_packages = set(project["tool"]["setuptools"]["packages"]["find"]["include"])

    for target in scripts.values():
        module = target.split(":", maxsplit=1)[0]
        top_level_package = module.split(".", maxsplit=1)[0]
        assert top_level_package in included_packages

def test_click_is_core_dependency_for_cli():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    assert "click" in _dependency_names(project["dependencies"])


def test_lockfile_keeps_click_in_core_torchwm_dependencies():
    lock = tomllib.loads(Path("uv.lock").read_text())
    torchwm = next(
        package for package in lock["package"] if package["name"] == "torchwm"
    )

    assert "click" in _dependency_names(torchwm["dependencies"])


def test_gymnasium_and_wandb_are_optional_not_core_dependencies():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    core = _dependency_names(project["dependencies"])
    assert "gymnasium[box2d]" not in project["dependencies"]
    assert "wandb" not in core
    assert "gymnasium[box2d]" in _dependency_names(project["optional-dependencies"]["gym"])
    assert "wandb" in _dependency_names(project["optional-dependencies"]["ml"])


def test_lockfile_keeps_gymnasium_and_wandb_out_of_core_torchwm_dependencies():
    lock = tomllib.loads(Path("uv.lock").read_text())
    torchwm = next(
        package for package in lock["package"] if package["name"] == "torchwm"
    )

    core = _dependency_names(torchwm["dependencies"])
    assert "gymnasium" not in core
    assert "wandb" not in core
    assert "gymnasium" in _dependency_names(torchwm["optional-dependencies"]["gym"])
    assert "wandb" in _dependency_names(torchwm["optional-dependencies"]["ml"])


def test_ml_agents_extra_matches_real_supported_sdk_series():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    assert "mlagents-envs>=0.28.0,<0.29.0" in project["optional-dependencies"]["ml-agents"]
    # The extra must not reinstate a `protobuf<3.21` cap: the whole 3.x line is
    # unpatched for the advisories GitHub reports against `uv.lock`, and no
    # mlagents-envs release that lifts the cap is installable on Python >= 3.11.
    assert "protobuf>=5.29.6" in project["optional-dependencies"]["ml-agents"]


def test_procgen_extra_declares_python_support_boundary():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    assert (
        "procgen>=0.10.7; python_version < '3.11'"
        in project["optional-dependencies"]["procgen"]
    )
