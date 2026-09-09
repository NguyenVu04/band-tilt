"""The tilt-delta operator's structure: reconstruction, shape, and the identity it starts at.

No Sionna-RT, no GPU, no dataset. Everything here is a hand-built tensor.
"""

from __future__ import annotations

import torch

from src.surrogate.model import (
    _HAAR,
    MAP_SHAPE,
    Film,
    OperatorSpec,
    TiltOperator,
    crop,
    haar_forward,
    haar_inverse,
    pad,
)


def _inputs(batch: int = 2, channels: int = 10):
    """One batch of the four tensors the operator takes."""
    torch.manual_seed(0)
    return (
        torch.randn(batch, channels, *MAP_SHAPE),
        torch.randn(batch, 6),
        torch.randn(batch, 2, *MAP_SHAPE),
        (torch.rand(batch, 1, *MAP_SHAPE) > 0.5).double().float(),
    )


def test_haar_round_trips_exactly_at_every_level() -> None:
    """Orthonormal Haar reconstructs exactly; the wavelet branch relies on it."""
    kernel = _HAAR.repeat(6, 1, 1, 1)
    original = torch.randn(2, 6, 80, 96)

    coarse, details = original, []
    for _ in range(3):
        bands = haar_forward(coarse, kernel)
        coarse, detail = bands[:, :, 0], bands[:, :, 1:]
        details.append(detail)
        assert bands.shape[2] == 4

    for level in reversed(range(3)):
        coarse = haar_inverse(torch.cat([coarse.unsqueeze(2), details[level]], dim=2), kernel)

    assert torch.allclose(coarse, original, atol=1e-5)


def test_haar_halves_both_axes() -> None:
    """One level decimates by two and returns four subbands."""
    kernel = _HAAR.repeat(3, 1, 1, 1)
    bands = haar_forward(torch.randn(1, 3, 80, 96), kernel)
    assert bands.shape == (1, 3, 4, 40, 48)


def test_pad_then_crop_is_the_identity() -> None:
    """The FFT's domain padding must not leak into what the caller sees."""
    x = torch.randn(2, 5, *MAP_SHAPE)
    assert torch.equal(crop(pad(x)), x)
    assert pad(x).shape[-2:] == (80, 96)


def test_film_starts_as_the_identity() -> None:
    """Zero-initialised heads mean an untrained block modulates by nothing."""
    film = Film(width=8, cond_dim=6, embed_dim=16, spatial=True)
    x = torch.randn(2, 8, 12, 12)
    out = film(x, torch.randn(2, 6), torch.randn(2, 16, 12, 12))
    assert torch.allclose(out, x, atol=1e-6)


def test_forward_returns_two_maps_on_the_solved_grid() -> None:
    """The residual and the coverage logit both land back on the solved grid."""
    residual, logit = TiltOperator()(*_inputs())
    assert residual.shape == (2, 1, *MAP_SHAPE)
    assert logit.shape == (2, 1, *MAP_SHAPE)


def test_untrained_model_is_exactly_the_analytic_baseline() -> None:
    """An untrained forward pass must degrade to physics, not to noise."""
    x, cond, cond_map, has_path = _inputs()
    residual, logit = TiltOperator()(x, cond, cond_map, has_path)

    # A zero residual means the prediction is rsrp + delta_gain and nothing else,
    # and the logit is the prior: this tilt change opened and closed no tile.
    assert torch.count_nonzero(residual) == 0
    assert torch.equal(torch.sign(logit), 2.0 * has_path - 1.0)


def test_every_ablation_still_produces_the_right_shape() -> None:
    """Switching a branch off is a config flag, not a different model."""
    for spec in (
        OperatorSpec(use_fno=False),
        OperatorSpec(use_wno=False),
        OperatorSpec(use_spatial_film=False),
        OperatorSpec(use_fno=False, use_wno=False, use_spatial_film=False),
    ):
        residual, logit = TiltOperator(spec)(*_inputs())
        assert residual.shape == (2, 1, *MAP_SHAPE)
        assert logit.shape == (2, 1, *MAP_SHAPE)


def test_dropping_a_branch_drops_its_parameters() -> None:
    """An ablation that changed nothing would measure nothing."""
    full = TiltOperator().n_parameters
    assert TiltOperator(OperatorSpec(use_fno=False)).n_parameters < full
    assert TiltOperator(OperatorSpec(use_wno=False)).n_parameters < full


def test_complex_parameters_are_counted_as_two_floats() -> None:
    """Otherwise the branch the ablation is about reports at half its true size."""
    spectral_only = TiltOperator(OperatorSpec(use_wno=False, use_spatial_film=False))
    elements = sum(p.numel() for p in spectral_only.parameters() if p.requires_grad)
    assert spectral_only.n_parameters > elements


def test_gradients_reach_every_branch() -> None:
    """A branch with no gradient is dead weight the loss can never use."""
    x, cond, cond_map, has_path = _inputs()
    model = TiltOperator()
    # The projection starts at zero, so a residual loss alone has no gradient to
    # give: perturb it first, or this test passes on a disconnected graph.
    torch.nn.init.normal_(model.project.weight, std=0.01)

    residual, _ = model(x, cond, cond_map, has_path)
    residual.square().mean().backward()

    # The FiLM heads are probed at their output layers. Their input layers are
    # genuinely dead at initialisation -- the zero-init output layer blocks the
    # gradient -- and come alive on the first step, so probing those would fail
    # for a reason that is the design rather than a defect.
    for name in (
        "blocks.0.fno.diag_lower",
        "blocks.0.fno.mix_upper",
        "blocks.0.wno.coarse",
        "blocks.0.wno.detail.2",
        "blocks.0.film.scalar.2.weight",
        "blocks.0.film.spatial.weight",
    ):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.count_nonzero(gradient) > 0, name


def test_film_leaves_its_zero_initialisation_after_one_step() -> None:
    """Zero-init must delay the branch, not disable it for the whole run."""
    x, cond, cond_map, has_path = _inputs()
    model = TiltOperator()
    torch.nn.init.normal_(model.project.weight, std=0.01)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(2):
        optimiser.zero_grad()
        residual, _ = model(x, cond, cond_map, has_path)
        (residual - 1.0).square().mean().backward()
        optimiser.step()

    # Two steps: the first moves the output layer off zero, the second is the
    # one that can reach the input layer through it. A branch that stayed dead
    # here would be dead for the whole run.
    assert torch.count_nonzero(model.blocks[0].film.scalar[0].weight.grad) > 0
