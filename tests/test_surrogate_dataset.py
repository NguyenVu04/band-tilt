"""The sweep table, the fitted pattern, and the pair enumeration.

No Sionna-RT and no GPU, so :func:`~src.surrogate.dataset.sweep` and
:func:`~src.surrogate.dataset.check_decomposition` are not exercised here --
both cast rays. Everything below runs on a hand-built sweep whose additive
model holds exactly, which is what gives the pattern fit a known answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.cell import Cell, Tilt
from src.surrogate.dataset import (
    DecompositionReport,
    Encoder,
    Pattern,
    Sweep,
    TiltPairs,
    fit_pattern,
)
from src.surrogate.features import SceneFeatures

BANDS = ("b2600", "b1800")
SHAPE = (8, 10)
TILE_M = 20.0
SCENARIO = "scn_test"


def _true_gain(offset_deg: np.ndarray) -> np.ndarray:
    """A single-lobe vertical pattern, peaked at boresight and floored."""
    return np.maximum(-12.0 * (offset_deg / 10.0) ** 2, -30.0)


def _elevation(n_tx: int) -> np.ndarray:
    """Depression angles spanning the range a 30 m mast actually sees."""
    rng = np.random.default_rng(0)
    return rng.uniform(2.0, 55.0, size=(n_tx, *SHAPE))


def _cells(n_tx: int) -> tuple[Cell, ...]:
    tilt = {band: Tilt(baseline_deg=4.0, bounds_deg=(0.0, 12.0)) for band in BANDS}
    return tuple(
        Cell(name=f"c{i}", x=100.0 * i, y=50.0 * i, z=30.0, azimuth_deg=120.0 * i, tilt=tilt)
        for i in range(n_tx)
    )


def _sweep(elevation: np.ndarray, nan_fraction: float = 0.0) -> tuple[Sweep, np.ndarray]:
    """A sweep obeying ``rsrp = base + gain(elev - tilt)`` exactly, and its base."""
    rng = np.random.default_rng(1)
    n_tx = elevation.shape[0]
    base = rng.uniform(-90.0, -60.0, size=(len(BANDS), n_tx, *SHAPE))
    grids = (np.arange(0.0, 6.5, 0.5), np.arange(0.0, 4.5, 0.5))

    maps = []
    for band, tilts in enumerate(grids):
        rsrp = base[band][None] + _true_gain(elevation[None] - tilts[:, None, None, None])
        if nan_fraction:
            rsrp = np.where(rng.random(rsrp.shape) < nan_fraction, np.nan, rsrp)
        maps.append(rsrp.astype(np.float32))

    return (
        Sweep(
            band_names=BANDS,
            tilts=grids,
            rsrp=tuple(maps),
            tx_names=tuple(f"c{i}" for i in range(n_tx)),
            origin_x=0.0,
            origin_y=0.0,
            tile_size_m=TILE_M,
            scenario_id=SCENARIO,
        ),
        base,
    )


def _features(elevation: np.ndarray) -> SceneFeatures:
    empty = np.zeros(SHAPE)
    return SceneFeatures(
        dsm_m=empty,
        sdf_m=empty,
        facade_azimuth_deg=empty,
        mean_height_m=empty,
        max_height_m=np.full(SHAPE, 15.0),
        mean_sdf_m=np.full(SHAPE, 25.0),
        los_fraction=np.full((elevation.shape[0], *SHAPE), 0.5),
        elevation_deg=elevation,
    )


def test_at_assembles_the_map_a_tilt_vector_names() -> None:
    """Each dimension picks its own slice, cell-major and band-minor."""
    elevation = _elevation(3)
    sweep, _ = _sweep(elevation)
    # Dimension order is cell-major, band-minor, as TiltSpace declares it.
    vector = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 0.5])

    built = sweep.at(vector)

    assert built.shape == (2, 3, *SHAPE)
    for tx, (first, second) in enumerate([(0.0, 1.0), (2.0, 3.0), (4.0, 0.5)]):
        assert np.allclose(built[0, tx], sweep.rsrp[0][sweep.index_of(0, first), tx])
        assert np.allclose(built[1, tx], sweep.rsrp[1][sweep.index_of(1, second), tx])


def test_a_tilt_off_the_grid_is_an_error_not_an_interpolation() -> None:
    """The sweep is an exact table, so a near miss is a caller bug."""
    sweep, _ = _sweep(_elevation(2))
    with pytest.raises(ValueError, match="not on the b2600 sweep grid"):
        sweep.index_of(0, 0.3)


def test_fit_recovers_the_pattern_that_generated_the_sweep() -> None:
    """Differencing along tilt cancels the propagation term exactly."""
    elevation = _elevation(3)
    sweep, _ = _sweep(elevation)

    pattern = fit_pattern(sweep, elevation)

    # Only the angles the sweep actually visited are identified; outside them
    # the fit is an extrapolation and says nothing.
    visited = (elevation.min() - sweep.tilts[0].max(), elevation.max())
    inside = (pattern.offset_deg >= visited[0]) & (pattern.offset_deg <= visited[1])
    expected = _true_gain(pattern.offset_deg[inside])
    assert np.allclose(
        pattern.gain_db[0][inside] - pattern.gain_db[0][inside].max(),
        expected - expected.max(),
        atol=0.5,
    )


def test_fit_reconstructs_the_sweep_it_was_fitted_to() -> None:
    """The gauge splits the constant arbitrarily, so the sum is what must hold."""
    elevation = _elevation(3)
    sweep, _ = _sweep(elevation, nan_fraction=0.3)

    pattern = fit_pattern(sweep, elevation)

    # The gauge splits the constant between the two terms arbitrarily, so what
    # has to hold is the sum, not either half.
    for band, tilts in enumerate(sweep.tilts):
        for index, tilt in enumerate(tilts):
            predicted = pattern.reference(band, elevation, float(tilt))
            actual = sweep.rsrp[band][index]
            finite = np.isfinite(actual)
            assert np.abs(predicted[finite] - actual[finite]).max() < 1.0


def test_pattern_delta_is_zero_for_no_change_and_antisymmetric() -> None:
    """A tilt change and its reverse are the same correction, negated."""
    elevation = _elevation(2)
    sweep, _ = _sweep(elevation)
    pattern = fit_pattern(sweep, elevation)

    assert np.allclose(pattern.delta(0, elevation, 3.0, 3.0), 0.0)
    assert np.allclose(
        pattern.delta(0, elevation, 1.0, 4.0), -pattern.delta(0, elevation, 4.0, 1.0)
    )


def test_pattern_peaks_at_zero_db() -> None:
    """The gauge puts boresight at 0 dB, so base_db reads as the level there."""
    elevation = _elevation(2)
    sweep, _ = _sweep(elevation)
    pattern = fit_pattern(sweep, elevation)
    assert np.allclose(pattern.gain_db.max(axis=1), 0.0)


def _pairs(split: str) -> TiltPairs:
    elevation = _elevation(3)
    sweep, _ = _sweep(elevation, nan_fraction=0.3)
    return TiltPairs(sweep, _features(elevation), fit_pattern(sweep, elevation), _cells(3), split)


def test_the_two_splits_never_share_a_pair() -> None:
    """A pair in both splits would make the test score meaningless."""
    train = {tuple(row) for row in _pairs("train").index}
    test = {tuple(row) for row in _pairs("test").index}
    assert train and test
    assert not train & test


def test_train_moves_between_whole_degrees_and_test_lands_between_them() -> None:
    """The split is on the tilt axis: the model is tested on tilts never solved."""
    train, test = _pairs("train"), _pairs("test")

    for band, _tx, before, after in train.index:
        for tilt in (train.sweep.tilts[band][before], train.sweep.tilts[band][after]):
            assert float(tilt).is_integer()

    for band, _tx, before, after in test.index:
        assert float(test.sweep.tilts[band][before]).is_integer()
        assert not float(test.sweep.tilts[band][after]).is_integer()


def test_a_pair_never_moves_a_tilt_to_itself() -> None:
    """A no-op pair carries no supervision and would dilute the loss."""
    index = _pairs("train").index
    assert not any(before == after for _band, _tx, before, after in index)


def test_an_item_carries_every_tensor_the_model_and_loss_need() -> None:
    """Shapes the operator and the three loss terms index into."""
    pairs = _pairs("train")
    item = pairs[0]

    assert item["x"].shape == (len(TiltPairs.channels), *SHAPE)
    assert item["cond"].shape == (3 + len(BANDS),)
    assert item["cond_map"].shape == (2, *SHAPE)
    for key in ("has_path", "baseline", "target", "target_mask"):
        assert item[key].shape == (1, *SHAPE), key
    assert all(tensor.dtype.is_floating_point for tensor in item.values())


def test_every_channel_is_finite_even_where_no_path_reached_the_tile() -> None:
    """Two thirds of a slice has no path; a NaN there would poison the batch."""
    pairs = _pairs("train")
    for index in (0, len(pairs) // 2, len(pairs) - 1):
        item = pairs[index]
        assert item["x"].isfinite().all()
        assert item["baseline"].isfinite().all()
        assert item["target"].isfinite().all()


def test_the_coverage_channel_matches_the_map_it_describes() -> None:
    """The mask head's prior is signed by this channel, so it must be the source's."""
    pairs = _pairs("train")
    band, tx, before, _after = pairs.index[0]
    expected = np.isfinite(pairs.sweep.rsrp[band][before, tx])
    assert np.array_equal(pairs[0]["has_path"].numpy()[0].astype(bool), expected)


def test_the_baseline_is_the_input_map_plus_the_analytic_correction() -> None:
    """The model predicts a residual over this, so it has to be exactly that sum."""
    pairs = _pairs("train")
    item = pairs[0]
    band, _tx, before, after = pairs.index[0]
    tilts = pairs.sweep.tilts[band]

    # The stack's first channel is the filled map, normalised; undoing that has
    # to land back on the baseline once the pattern's correction is added.
    filled = item["x"][0].numpy() * 20.0 - 100.0
    correction = item["x"][2].numpy() * 10.0
    assert np.allclose(filled + correction, item["baseline"].numpy()[0], atol=1e-2)
    assert np.allclose(
        correction,
        pairs.pattern.delta(
            band, pairs.elevation[pairs.index[0][1]], float(tilts[before]), float(tilts[after])
        ),
        atol=1e-2,
    )


def test_a_bad_split_name_is_rejected() -> None:
    """A typo must not silently produce an empty dataset."""
    elevation = _elevation(2)
    sweep, _ = _sweep(elevation)
    with pytest.raises(ValueError, match="split must be"):
        TiltPairs(sweep, _features(elevation), fit_pattern(sweep, elevation), _cells(2), "valid")


def test_reference_is_finite_where_no_tilt_ever_found_a_path() -> None:
    """The fill has to be defined even where the propagation term is not."""
    elevation = _elevation(2)
    sweep, _ = _sweep(elevation, nan_fraction=1.0)
    base = np.full((len(BANDS), 2, *SHAPE), np.nan)
    pattern = Pattern(
        band_names=BANDS,
        offset_deg=np.arange(-30.0, 90.0, 0.5),
        gain_db=np.zeros((len(BANDS), len(np.arange(-30.0, 90.0, 0.5)))),
        base_db=base,
    )
    assert np.isfinite(pattern.reference(0, elevation, 4.0)).all()


def _encoder(n_tx: int = 3) -> Encoder:
    """An encoder over the synthetic sweep, for the batched-encoding tests."""
    elevation = _elevation(n_tx)
    sweep, _ = _sweep(elevation, nan_fraction=0.3)
    return Encoder.build(sweep, _features(elevation), fit_pattern(sweep, elevation), _cells(n_tx))


def test_encode_batch_matches_one_pair_at_a_time() -> None:
    """The batched path is the only implementation, so a row of it must be the row.

    Mixed bands and transmitters on purpose: the failure this guards is a
    transposed or mis-paired index, which a batch of one band or one
    transmitter cannot detect.
    """
    encoder = _encoder()
    rng = np.random.default_rng(3)
    triples = [(0, 2, 1.0, 3.5), (1, 0, 4.0, 2.0), (0, 1, 6.0, 0.5), (1, 2, 3.0, 3.0)]
    sources = rng.normal(-90.0, 8.0, size=(len(triples), *SHAPE))
    sources[rng.random(sources.shape) < 0.3] = np.nan

    batch = encoder.encode_batch(
        np.array([band for band, _, _, _ in triples]),
        np.array([tx for _, tx, _, _ in triples]),
        sources,
        np.array([before for _, _, before, _ in triples]),
        np.array([after for _, _, _, after in triples]),
    )

    for row, (band, tx, before, after) in enumerate(triples):
        single = encoder.assemble(band, tx, sources[row], before, after)
        for key, value in single.items():
            assert np.array_equal(batch[key][row], value, equal_nan=True), f"{key} row {row}"


def test_encode_batch_pairs_the_transmitter_with_its_own_band() -> None:
    """Two rows sharing a band but not a transmitter must not share a level.

    ``base_db`` is indexed by the pair. Indexing it by band alone and slicing
    afterwards is what the batched path replaced, and it is the mistake that
    would leave every row carrying transmitter zero's propagation term.
    """
    encoder = _encoder()
    source = np.full(SHAPE, -85.0)
    batch = encoder.encode_batch(
        np.array([0, 0]),
        np.array([0, 2]),
        np.stack([source, source]),
        np.array([2.0, 2.0]),
        np.array([4.0, 4.0]),
    )
    assert not np.allclose(batch["x"][0, 3], batch["x"][1, 3])  # elevation channel


def test_gain_is_interpolated_with_each_row_own_band_curve() -> None:
    """Grouping the rows by band must not let one band's curve reach another's."""
    encoder = _encoder()
    angles = np.tile(np.linspace(0.0, 40.0, SHAPE[0] * SHAPE[1]).reshape(SHAPE), (2, 1, 1))
    grouped = encoder._gain(np.array([0, 1]), angles)
    assert np.array_equal(grouped[0], encoder.pattern.gain(0, angles[0]))
    assert np.array_equal(grouped[1], encoder.pattern.gain(1, angles[1]))


def _report(others: float, repeat: float, moved: float) -> DecompositionReport:
    return DecompositionReport(
        moved_cell="c0",
        band=BANDS[0],
        max_abs_delta_db=others,
        repeat_max_abs_delta_db=repeat,
        moved_max_abs_delta_db=moved,
    )


def test_decomposition_holds_reads_the_untouched_map_against_repeat_noise() -> None:
    """The verdict is relative: the same numbers pass or fail on the noise floor.

    ``RadioMapSolver`` accumulates with atomic adds, so a sound solver still
    differs in the last bits between two solves of one tilt. A fixed tolerance
    would either hide real coupling or fail on a machine that is merely noisier.
    """
    assert _report(others=1e-6, repeat=1e-5, moved=3.0).holds
    assert not _report(others=1e-3, repeat=1e-5, moved=3.0).holds
    # Bit-identical solves: the strict test the noise floor generalises.
    assert _report(others=0.0, repeat=0.0, moved=3.0).holds
    # A null result -- nothing moved at all -- is a failure, not a pass.
    assert not _report(others=0.0, repeat=0.0, moved=0.0).holds
