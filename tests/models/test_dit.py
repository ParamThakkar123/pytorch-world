import torch

from torchwm.configs.dit_config import DiTConfig
from torchwm.models.diffusion.DiT import DiT, PatchEmbed, PatchUnEmbed, create_dit


def test_create_dit_builds_small_model_from_config_and_runs_forward():
    config = DiTConfig(
        IMG_SIZE=8, PATCH=4, CHANNELS=3, WIDTH=16, DEPTH=1, HEADS=4, DROP=0.0
    )
    model = create_dit(config)

    assert isinstance(model, DiT)
    x = torch.randn(2, 3, 8, 8)
    t = torch.tensor([0, 10])
    out = model(x, t)

    # learn_sigma is on by default (paper 3.1), so the model emits 2C channels:
    # the predicted noise followed by the diagonal covariance.
    assert out.shape == (2, 6, 8, 8)


def test_epsilon_only_model_matches_input_shape():
    config = DiTConfig(
        IMG_SIZE=8, PATCH=4, CHANNELS=3, WIDTH=16, DEPTH=1, HEADS=4, DROP=0.0,
        LEARN_SIGMA=False,
    )
    model = create_dit(config)
    x = torch.randn(2, 3, 8, 8)
    out = model(x, torch.tensor([0, 10]))

    assert out.shape == x.shape


def test_eval_mode_works():
    """`.eval()` must not be shadowed by the training-loop entrypoint."""
    model = create_dit(
        DiTConfig(IMG_SIZE=8, PATCH=4, CHANNELS=3, WIDTH=16, DEPTH=1, HEADS=4)
    )
    model.eval()
    assert not model.training
    model.train()
    assert model.training


def test_patch_embed_unembed_round_trip_shape():
    patch = PatchEmbed(img_size=8, patch_size=4, in_channels=3, embed_dim=12)
    unpatch = PatchUnEmbed(img_size=8, patch_size=4, embed_dim=12, out_channels=3)

    tokens = patch(torch.randn(2, 3, 8, 8))
    images = unpatch(tokens)

    assert tokens.shape == (2, 4, 12)
    assert images.shape == (2, 3, 8, 8)
