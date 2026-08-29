"""Tests for the five KPIs — the highest-value tests in the project.

The KPIs define the objective. Every optimizer, the surrogate's training labels
and the final validation all read them from :mod:`src.kpi`, so an error here is
not a wrong number in one place: it is a wrong objective, optimised
successfully, and reported with confidence.

None of these tests need Sionna-RT
----------------------------------
The KPI functions take a plain RSRP array, so the whole objective is testable
against a six-by-five fixture with values worked out by hand. That is the
payoff for keeping :mod:`src.kpi` free of any simulator dependency.

The expected values come from the fixture docstring, not from the code
-----------------------------------------------------------------------
``tests/conftest.py::rsrp_grid`` documents its KPIs and shows how each grid cell
was constructed to produce them. The assertions below use those literals. A test
that computes its expectation with the function under test asserts only that the
implementation agrees with itself.

Every test is skipped until ``src/kpi/`` is implemented; the skip list is the
implementation checklist.
"""

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig

from src.kpi import band_priority, coverage, serving, vector

pytestmark = pytest.mark.skip(reason="implement src/kpi/ first")


def test_hole_rate_matches_hand_computed_value(rsrp_grid: np.ndarray, cfg: DictConfig) -> None:
    """One of five grid cells is a hole, so the rate is 20%."""
    r_max = serving.max_rsrp(rsrp_grid)
    assert coverage.hole_rate(r_max, cfg) == pytest.approx(20.0)


def test_weak_rate_matches_hand_computed_value(rsrp_grid: np.ndarray, cfg: DictConfig) -> None:
    """One of five grid cells is weak, so the rate is 20%."""
    r_max = serving.max_rsrp(rsrp_grid)
    assert coverage.weak_rate(r_max, cfg) == pytest.approx(20.0)


def test_hole_and_weak_are_disjoint(rsrp_grid: np.ndarray, cfg: DictConfig) -> None:
    """A location cannot be both a hole and weak coverage.

    The hole and weak thresholds define half-open bands that partition the
    grid. An inclusive boundary on both sides double-counts the threshold value,
    which shows up as the two rates summing past 100.
    """
    r_max = serving.max_rsrp(rsrp_grid)
    assert coverage.hole_rate(r_max, cfg) + coverage.weak_rate(r_max, cfg) <= 100.0


def test_hole_threshold_is_inclusive(cfg: DictConfig) -> None:
    """RSRP exactly at ``hole_dbm`` is a hole, not weak coverage.

    The hole condition is ``R_max <= hole_dbm``. Off-by-one
    at the boundary silently moves locations between the two highest-priority
    KPIs.
    """
    r_max = np.array([cfg.kpi.hole_dbm])
    assert coverage.hole_rate(r_max, cfg) == pytest.approx(100.0)
    assert coverage.weak_rate(r_max, cfg) == pytest.approx(0.0)


def test_overlap_rate_matches_hand_computed_value(
    rsrp_grid: np.ndarray, cell_bands: pd.DataFrame, cfg: DictConfig
) -> None:
    """Two of five grid cells have at least one overlapping neighbour."""
    per_cell = serving.cell_rsrp(rsrp_grid, cell_bands, cfg)
    n_ov = coverage.overlap_neighbors(per_cell, serving.serving_cell(per_cell), cfg)
    assert coverage.overlap_rate(n_ov) == pytest.approx(40.0)


def test_mean_overlap_neighbors_matches_hand_computed_value(
    rsrp_grid: np.ndarray, cell_bands: pd.DataFrame, cfg: DictConfig
) -> None:
    """One overlapping cell has one neighbour and one has two, so the mean is 1.5.

    The denominator is the number of overlapping locations, not the grid size
    Dividing by the grid instead gives 0.6 here, and
    makes this KPI a rescaled overlap rate carrying no extra information.
    """
    per_cell = serving.cell_rsrp(rsrp_grid, cell_bands, cfg)
    n_ov = coverage.overlap_neighbors(per_cell, serving.serving_cell(per_cell), cfg)
    assert coverage.mean_overlap_neighbors(n_ov) == pytest.approx(1.5)


def test_serving_cell_is_excluded_from_its_own_overlap_count(
    rsrp_grid: np.ndarray, cell_bands: pd.DataFrame, cfg: DictConfig
) -> None:
    """A cell does not overlap itself.

    The serving cell trivially satisfies the margin condition against itself, so
    forgetting to exclude it adds exactly one to every covered location — which
    still looks like a plausible result.
    """
    per_cell = serving.cell_rsrp(rsrp_grid, cell_bands, cfg)
    n_ov = coverage.overlap_neighbors(per_cell, serving.serving_cell(per_cell), cfg)
    assert n_ov[2] == 0


def test_uncovered_location_contributes_no_overlap(
    rsrp_grid: np.ndarray, cell_bands: pd.DataFrame, cfg: DictConfig
) -> None:
    """A coverage hole has nothing to overlap with.

    Grid cell 0 is ``-inf`` everywhere, so the serving-cell difference is
    ``nan``. The coverage condition must gate the count rather than the
    comparison deciding it.
    """
    per_cell = serving.cell_rsrp(rsrp_grid, cell_bands, cfg)
    n_ov = coverage.overlap_neighbors(per_cell, serving.serving_cell(per_cell), cfg)
    assert n_ov[0] == 0


def test_mean_overlap_neighbors_returns_zero_when_nothing_overlaps(cfg: DictConfig) -> None:
    """No overlap is the best case, not an undefined one.

    Returning ``nan`` from an empty denominator makes the KPI unorderable, and
    an optimizer comparing candidates cannot rank it.
    """
    assert coverage.mean_overlap_neighbors(np.zeros(5)) == pytest.approx(0.0)


def test_dominant_band_is_the_strongest_pair_not_the_strongest_cell(
    rsrp_grid: np.ndarray, cell_bands: pd.DataFrame
) -> None:
    """The dominant band is the argmax over cell-band pairs directly.

    Grid cells 1 and 4 are dominated by a low band while their serving cell's
    strongest carrier differs, so implementing this via the serving cell gives
    a different answer.
    """
    b_star = serving.dominant_band(rsrp_grid, cell_bands)
    bands = cell_bands["band"].to_numpy()
    assert list(bands[b_star][1:]) == ["low", "high", "high", "low"]


def test_band_priority_score_matches_hand_computed_value(
    rsrp_grid: np.ndarray,
    cell_bands: pd.DataFrame,
    ue_density_vector: np.ndarray,
    cfg: DictConfig,
) -> None:
    """The UE-weighted average of dominant-band weights is 2.4 for this fixture."""
    b_star = serving.dominant_band(rsrp_grid, cell_bands)
    weights = band_priority.band_weights(cell_bands, cfg)
    score = band_priority.band_priority_score(b_star, ue_density_vector, weights)
    assert score == pytest.approx(2.4)


def test_band_priority_score_ignores_empty_grid_cells(
    rsrp_grid: np.ndarray,
    cell_bands: pd.DataFrame,
    ue_density_vector: np.ndarray,
    cfg: DictConfig,
) -> None:
    """A grid cell with no UE observations must not move the score.

    Otherwise the optimizer is rewarded for band coordination in empty areas,
    which is exactly what the UE weighting exists to prevent.
    """
    b_star = serving.dominant_band(rsrp_grid, cell_bands)
    weights = band_priority.band_weights(cell_bands, cfg)
    baseline = band_priority.band_priority_score(b_star, ue_density_vector, weights)

    flipped = b_star.copy()
    flipped[0] = 1 - flipped[0]
    assert band_priority.band_priority_score(flipped, ue_density_vector, weights) == pytest.approx(
        baseline
    )


def test_zero_weight_is_rejected(cell_bands: pd.DataFrame, cfg: DictConfig) -> None:
    """The Band Priority Score requires strictly positive band weights.

    A zero or negative weight makes the score non-monotonic in band quality, so
    the objective stops meaning what ``priority_weight`` says it means.
    """
    broken = cfg.copy()
    broken.radio.bands.low.priority_weight = 0.0
    with pytest.raises(ValueError):
        band_priority.band_weights(cell_bands, broken)


def test_kpi_vector_returns_every_named_kpi(
    rsrp_grid: np.ndarray,
    cell_bands: pd.DataFrame,
    ue_density_vector: np.ndarray,
    cfg: DictConfig,
) -> None:
    """The single entry point returns all five, keyed by name.

    Every consumer stacks these into a table, which only works if the key set
    never varies with the input.
    """
    kpis = vector.kpi_vector(rsrp_grid, ue_density_vector, cell_bands, cfg)
    assert set(kpis) == set(vector.KPI_NAMES)


def test_lexicographic_prefers_lower_hole_rate_despite_worse_everything_else(
    cfg: DictConfig,
) -> None:
    """Hole rate outranks every other objective — see ``cfg.kpi.order``.

    This is the assertion that pins down the whole priority. A weighted sum can
    always be made to lose it by trading enough of the other four.
    """
    better_holes = dict.fromkeys(vector.KPI_NAMES, 50.0) | {"hole_rate": 1.0}
    worse_holes = dict.fromkeys(vector.KPI_NAMES, 0.0) | {"hole_rate": 10.0}
    assert vector.lexicographic_better(better_holes, worse_holes, cfg)
    assert not vector.lexicographic_better(worse_holes, better_holes, cfg)


def test_lexicographic_falls_through_to_overlap_within_tolerance(cfg: DictConfig) -> None:
    """A hole-rate difference inside the tolerance is a tie, so overlap decides.

    Without slack the comparison never reaches the second objective, because two
    continuous KPIs essentially never tie exactly — and the lexicographic order
    degenerates into single-objective optimization on hole rate.
    """
    slack = float(cfg.kpi.tolerance.hole_rate) / 2.0
    left = dict.fromkeys(vector.KPI_NAMES, 0.0) | {"hole_rate": 5.0, "overlap_rate": 10.0}
    right = dict.fromkeys(vector.KPI_NAMES, 0.0) | {"hole_rate": 5.0 + slack, "overlap_rate": 20.0}
    assert vector.lexicographic_better(left, right, cfg)


def test_scalarize_rejects_weights_that_contradict_the_priority(cfg: DictConfig) -> None:
    """``lambda_H > lambda_O > lambda_W`` is checked, not assumed.

    Weights that violate the ordering produce an
    objective which silently contradicts the stated priority while every
    individual KPI still looks correct.
    """
    broken = cfg.copy()
    broken.kpi.weights.hole_rate = 0.1
    broken.kpi.weights.weak_rate = 9.0
    kpis = dict.fromkeys(vector.KPI_NAMES, 1.0)
    with pytest.raises(ValueError):
        vector.scalarize(kpis, broken)


def test_normalize_orients_every_kpi_so_larger_is_better(cfg: DictConfig) -> None:
    """The one maximised KPI must stop being a special case after normalisation.

    Band Priority Score is maximised and the other four are minimised. Handling
    that anywhere other than here is how a second, local sign flip ends up
    minimising the one KPI that should grow.
    """
    good = dict.fromkeys(vector.KPI_NAMES, 0.0) | {"band_priority_score": 3.0}
    bad = dict.fromkeys(vector.KPI_NAMES, 10.0) | {"band_priority_score": 1.0}
    n_good = vector.normalize(good, cfg)
    n_bad = vector.normalize(bad, cfg)
    assert all(n_good[name] >= n_bad[name] for name in vector.KPI_NAMES)
