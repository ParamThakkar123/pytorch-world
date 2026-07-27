import re

import pytest

import torchwm
from torchwm import api


def _unresolvable(module):
    """Return ``(name, error)`` for every ``__all__`` entry that cannot be read."""

    broken = []
    for name in module.__all__:
        try:
            getattr(module, name)
        except Exception as exc:  # noqa: BLE001 - the failure mode is the point
            broken.append((name, f"{type(exc).__name__}: {exc}"))
    return broken


def test_every_public_export_resolves():
    # The lazy ``__getattr__`` export map means a typo (wrong module, renamed
    # symbol) stays invisible until a user touches that exact name.  Walk the
    # whole surface so the export map can never drift from the implementation.
    pytest.importorskip("torch")

    broken = _unresolvable(torchwm)
    assert not broken, f"unresolvable torchwm exports: {broken}"


def test_every_internal_public_export_resolves():
    # ``torchwm`` is a lazy alias over ``world_models``, and the export map
    # lives here - check the source of truth directly too.
    pytest.importorskip("torch")
    import world_models

    broken = _unresolvable(world_models)
    assert not broken, f"unresolvable world_models exports: {broken}"


def test_torchwm_all_covers_world_models_all():
    import world_models

    assert set(world_models.__all__) <= set(torchwm.__all__)
    assert "api" in torchwm.__all__


def test_top_level_torchwm_exports_user_facing_factories():
    assert re.match(r"^\d+\.\d+\.\d+$", torchwm.__version__)
    assert torchwm.create_config is api.create_config
    assert "dreamer" in torchwm.list_models()
    assert "gym" in torchwm.list_env_backends()


def test_create_config_accepts_aliases_and_overrides():
    cfg = torchwm.create_config("dreamerv1", env="cartpole-swingup", seed=123)
    assert cfg.env == "cartpole-swingup"
    assert cfg.seed == 123


def test_model_and_backend_specs_resolve_aliases():
    assert torchwm.get_model_spec("i-jepa").name == "jepa"
    assert torchwm.get_env_backend_spec("gymnasium").name == "gym"
    assert torchwm.get_env_backend_spec("wm").name == "world-model"


def test_make_env_dispatches_to_selected_backend(monkeypatch):
    calls = {}

    def fake_loader(import_path):
        calls["import_path"] = import_path

        def factory(env_id, **kwargs):
            return {"env_id": env_id, "kwargs": kwargs}

        return factory

    monkeypatch.setattr(api, "_load_object", fake_loader)
    env = api.make_env("CartPole-v1", backend="gym", render_mode="rgb_array")

    assert calls["import_path"] == "world_models.envs:make_gym_env"
    assert env == {
        "env_id": "CartPole-v1",
        "kwargs": {"render_mode": "rgb_array"},
    }


def test_create_model_for_factory_only_spec_filters_through_signature(monkeypatch):
    spec = api.ModelSpec(
        name="dummy",
        import_path="tests.dummy:create_dummy",
        description="Test-only factory",
    )

    def fake_loader(import_path):
        assert import_path == spec.import_path

        def create_dummy(required, optional=1):
            return {"required": required, "optional": optional}

        return create_dummy

    monkeypatch.setitem(api.MODEL_SPECS, "dummy", spec)
    monkeypatch.setattr(api, "_load_object", fake_loader)

    assert api.create_model("dummy", required=3, optional=5) == {
        "required": 3,
        "optional": 5,
    }


def test_make_env_dispatches_world_model_backend(monkeypatch):
    calls = {}

    def fake_loader(import_path):
        calls["import_path"] = import_path

        def factory(world_model, **kwargs):
            return {"world_model": world_model, "kwargs": kwargs}

        return factory

    model = object()
    monkeypatch.setattr(api, "_load_object", fake_loader)
    env = api.make_env(model, backend="wm", observation_space="obs", action_space="act")

    assert calls["import_path"] == "world_models.envs:make_world_model_env"
    assert env == {
        "world_model": model,
        "kwargs": {"observation_space": "obs", "action_space": "act"},
    }


def test_export_model_torchscript_writes_file(tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    import world_models.export  # noqa: F401 - installs torch.nn.Module.export

    class TinyAgent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

    agent = TinyAgent()
    path = agent.export(
        tmp_path / "tiny.pt",
        format="torchscript",
        example_inputs=torch.zeros(1, 2),
    )

    assert path.exists()
    loaded = torch.jit.load(str(path))
    assert loaded(torch.zeros(1, 2)).shape == (1, 1)


def test_top_level_exports_export_helpers():
    import pytest
    import torchwm

    pytest.importorskip("torch")
    from world_models.export import ExportableAgentMixin, export_any, export_model

    assert torchwm.export_any is export_any
    assert torchwm.export_model is export_model
    assert torchwm.ExportableAgentMixin is ExportableAgentMixin


def test_layer_and_helper_packages_are_importable():
    import world_models.helpers as helpers
    from world_models.layers import AdaLNNormalization, RMSNorm

    assert "load_checkpoint" in dir(helpers)
    assert RMSNorm.__name__ == "RMSNorm"
    assert AdaLNNormalization.__name__ == "AdaLNNormalization"


def test_torchwm_submodules_alias_world_models():
    import world_models.envs
    import world_models.models
    import world_models.utils.deprecation

    import torchwm.envs
    import torchwm.models
    import torchwm.utils.deprecation

    # The friendly ``torchwm.<name>`` surface resolves to the same module object
    # as the internal ``world_models.<name>`` implementation.
    assert torchwm.models is world_models.models
    assert torchwm.envs is world_models.envs
    assert torchwm.utils.deprecation is world_models.utils.deprecation
    # Canonical module identity stays on the internal package.
    assert torchwm.models.__name__ == "world_models.models"


def test_torchwm_submodule_from_imports_resolve():
    from torchwm.envs import make_gym_env
    from torchwm.models import Dreamer
    from torchwm.utils.deprecation import deprecated

    assert Dreamer.__name__ == "Dreamer"
    assert callable(make_gym_env)
    assert callable(deprecated)


def test_torchwm_cli_is_the_real_submodule_not_an_alias():
    # ``torchwm.cli`` is a genuine module shipped in the ``torchwm`` package and
    # must not be shadowed by the ``world_models`` alias finder.
    import torchwm.cli

    assert torchwm.cli.__name__ == "torchwm.cli"
    assert torchwm.cli.__file__.replace("\\", "/").endswith("torchwm/cli.py")


def test_diamond_and_dit_are_registered_in_public_api():
    assert "diamond" in torchwm.list_models()
    assert "dit" in torchwm.list_models()

    diamond_cfg = torchwm.create_config("diamond", game="Pong-v5", seed=11)
    dit_cfg = torchwm.create_config("diffusion-transformer", IMG_SIZE=8, PATCH=4)

    assert diamond_cfg.game == "Pong-v5"
    assert diamond_cfg.seed == 11
    assert dit_cfg.IMG_SIZE == 8
    assert dit_cfg.PATCH == 4
    assert torchwm.get_model_spec("diamond_agent").name == "diamond"
    assert torchwm.get_model_spec("diffusion_transformer").name == "dit"


def test_create_model_uses_dit_config_adapter():
    model = torchwm.create_model(
        "dit",
        IMG_SIZE=8,
        PATCH=4,
        CHANNELS=3,
        WIDTH=16,
        DEPTH=1,
        HEADS=4,
        DROP=0.0,
    )

    assert model.patchify.proj.in_channels == 3
    assert len(model.transformer_blocks) == 1


def test_create_model_dispatches_diamond_agent_with_config(monkeypatch):
    captured = {}

    class FakeDiamondAgent:
        def __init__(self, config):
            captured["config"] = config

    original_loader = api._load_object

    def fake_loader(import_path):
        if import_path == "world_models.training.train_diamond:DiamondAgent":
            return FakeDiamondAgent
        return original_loader(import_path)

    monkeypatch.setattr(api, "_load_object", fake_loader)
    agent = api.create_model("diamond", game="Pong-v5")

    assert isinstance(agent, FakeDiamondAgent)
    assert captured["config"].game == "Pong-v5"
