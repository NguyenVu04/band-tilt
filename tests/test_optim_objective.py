"""The KPI vector, the sign convention, and the objective that picks a winner."""

from __future__ import annotations

import math

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.data import demand
from src.optim.objective import (
    KPI_NAMES,
    MAXIMISED,
    MEASURE_NAMES,
    KpiVector,
    best_by_objective,
    objective,
    tile_share,
)

# Band labels for a map built by _map, in its band-axis order. 'hi' is the
# preferred band, so it is the one lambda is counted on wherever it is covered.
_BANDS = ("hi", "lo")


@pytest.fixture
def cfg():
    """The thresholds and the band preference the objective reads."""
    return OmegaConf.create(
        {
            "kpi": {
                "hole_dbm": -120.0,
                "overlap_margin_db": 6.0,
                "capacity": {"band_preference": list(_BANDS)},
            },
        }
    )


def _kpi(**overrides: float) -> KpiVector:
    """A middling measurement, with named fields overridden."""
    values = {
        "hole_rate": 0.10,
        "overlap_rate": 0.30,
        "overlap_neighbor_mean": 0.45,
        "weak_rate": 0.10,
        "rsrp_p05_dbm": -105.0,
        "rsrp_p50_dbm": -95.0,
        "sinr_p05_db": -3.0,
        "sinr_p50_db": 8.0,
        "served_rate": 0.20,
        "prb_utilisation_max": 0.62,
        "load_imbalance": 0.44,
        "objective": 0.50,
    }
    return KpiVector(**{**values, **overrides})


def _u(multiplicity: float) -> float:
    """The utility of a tile served by that many cells: ``lambda e^(1 - lambda)``."""
    return multiplicity * math.exp(1.0 - multiplicity)


def _map(values: list[list[float]]) -> np.ndarray:
    """A one-tile radio map from nested ``[band][tx]`` RSRP lists."""
    return np.array(values, dtype=float)[:, :, None, None]


def _score(rsrp: np.ndarray, cfg, share: np.ndarray | None = None) -> float:
    """Score a map built by :func:`_map`, labelling its bands in axis order."""
    if share is None:
        share = np.ones(rsrp.shape[-2:])
    return objective(rsrp, _BANDS[: rsrp.shape[0]], cfg, share)


def test_priority_order_is_the_adr_order() -> None:
    """Reordering this changes which column of every table is which."""
    assert KPI_NAMES == (
        "hole_rate",
        "overlap_rate",
        "overlap_neighbor_mean",
        "weak_rate",
        "rsrp_p05_dbm",
        "rsrp_p50_dbm",
        "sinr_p05_db",
        "sinr_p50_db",
        "served_rate",
        "prb_utilisation_max",
        "load_imbalance",
    )
    assert MEASURE_NAMES == (*KPI_NAMES, "objective")


def test_only_the_quality_service_and_objective_measures_are_maximised() -> None:
    """The usual place a sign error hides: the load measures are minimised."""
    assert MAXIMISED == {
        "rsrp_p05_dbm",
        "rsrp_p50_dbm",
        "sinr_p05_db",
        "sinr_p50_db",
        "served_rate",
        "objective",
    }


def test_as_dict_round_trips_through_from_mapping() -> None:
    """The shape run.json records and the evaluation stage reads back."""
    kpi = _kpi(hole_rate=0.123)
    assert KpiVector.from_mapping(kpi.as_dict()) == kpi


def test_from_mapping_names_a_missing_measure() -> None:
    """An incomplete measurement must not become a silent zero."""
    values = _kpi().as_dict()
    del values["objective"]
    with pytest.raises(KeyError, match="objective"):
        KpiVector.from_mapping(values)


def test_band_labels_must_match_the_map(cfg) -> None:
    """Mislabelled bands would rank the wrong layers against each other."""
    with pytest.raises(ValueError, match="band labels"):
        objective(_map([[-100.0]]), ["hi", "lo"], cfg, np.ones((1, 1)))


def test_a_band_absent_from_the_preference_raises(cfg) -> None:
    """A band with no rank has no place in the priority order, so lambda is undefined."""
    with pytest.raises(ValueError, match="band_preference"):
        objective(_map([[-100.0]]), ["mid"], cfg, np.ones((1, 1)))


# --- lambda, the count the objective scores ---------------------------------


def test_one_dominant_cell_scores_the_maximum(cfg) -> None:
    """The whole point of the objective: exactly one server is worth 1.0."""
    assert _score(_map([[-90.0, -130.0]]), cfg) == pytest.approx(1.0)


def test_a_second_cell_inside_the_margin_costs_a_quarter(cfg) -> None:
    """A neighbour 4 dB down is inside the 6 dB margin; one 7 dB down is not."""
    assert _score(_map([[-100.0, -104.0]]), cfg) == pytest.approx(_u(2.0))
    assert _score(_map([[-100.0, -107.0]]), cfg) == pytest.approx(1.0)


def test_a_third_cell_costs_more_than_the_second(cfg) -> None:
    """The utility keeps falling, so the search never trades one crowd for a worse one."""
    two = _score(_map([[-100.0, -104.0, -130.0]]), cfg)
    three = _score(_map([[-100.0, -104.0, -105.0]]), cfg)
    assert three == pytest.approx(_u(3.0))
    assert three < two < 1.0


def test_the_margin_is_inclusive(cfg) -> None:
    """A neighbour exactly 6 dB down overlaps; 6.01 dB down does not."""
    assert _score(_map([[-100.0, -106.0]]), cfg) == pytest.approx(_u(2.0))
    assert _score(_map([[-100.0, -106.01]]), cfg) == pytest.approx(1.0)


def test_a_neighbour_below_the_hole_threshold_does_not_count(cfg) -> None:
    """Within the margin but unusable: -122 dBm serves nobody, so it crowds nobody."""
    assert _score(_map([[-118.0, -122.0]]), cfg) == pytest.approx(1.0)


def test_the_preferred_band_is_scored_even_where_another_is_stronger(cfg) -> None:
    """Priority, not strength, picks the band, as the serving rule does.

    'hi' is barely covered and alone; 'lo' is 49 dB stronger and crowded. The
    tile would be served on 'hi', so it scores 'hi'.
    """
    assert _score(_map([[-119.0, -130.0], [-70.0, -71.0]]), cfg) == pytest.approx(1.0)


def test_the_next_band_is_scored_where_the_preferred_one_is_a_hole(cfg) -> None:
    """With 'hi' below the threshold the tile falls through to 'lo', crowding and all."""
    assert _score(_map([[-130.0, -130.0], [-70.0, -71.0]]), cfg) == pytest.approx(_u(2.0))


def test_a_tile_no_band_covers_scores_zero(cfg) -> None:
    """A hole is worth nothing, however close to the threshold it comes."""
    assert _score(_map([[-130.0], [-121.0]]), cfg) == pytest.approx(0.0)


def test_a_no_path_tile_scores_zero(cfg) -> None:
    """No path is -inf, which is a hole on its own."""
    assert _score(_map([[np.nan, np.nan]]), cfg) == pytest.approx(0.0)


def test_the_objective_is_bounded_by_one(cfg) -> None:
    """Every tile served by exactly one cell is the best a map can do."""
    rsrp = np.array([[[[-90.0, -95.0], [-92.0, -91.0]]]], dtype=float)
    rsrp = np.concatenate([rsrp, np.full_like(rsrp, -130.0)], axis=1)
    assert _score(rsrp, cfg, np.ones((2, 2))) == pytest.approx(1.0)


# --- the demand weights -----------------------------------------------------


def _two_tiles() -> np.ndarray:
    """One band, one cell, two tiles: one served cleanly, one with no path."""
    return np.array([[[[-100.0, np.nan]]]], dtype=float)


def test_the_busiest_tile_weighs_twice_the_quietest(cfg) -> None:
    """The weight is 1 + r: demand doubles a tile at most, and silences none."""
    rsrp = _two_tiles()
    # Demand on the served tile: weights 2 and 1, so J = (2 * 1 + 1 * 0) / 3.
    assert _score(rsrp, cfg, np.array([[1.0, 0.0]])) == pytest.approx(2.0 / 3.0)
    # Demand on the hole instead: weights 1 and 2, so the hole costs more.
    assert _score(rsrp, cfg, np.array([[0.0, 1.0]])) == pytest.approx(1.0 / 3.0)


def test_ground_the_mdt_never_saw_is_still_scored(cfg) -> None:
    """An unreported tile weighs 1, not 0, so a hole out there is not free."""
    assert _score(_two_tiles(), cfg, np.zeros((1, 2)) + 1e-9) == pytest.approx(0.5)


def test_the_share_need_not_be_normalised(cfg) -> None:
    """Only the ratio to the busiest tile is read, so any scaling of one map is one J."""
    counts = _score(_two_tiles(), cfg, np.array([[3.0, 1.0]]))
    share = _score(_two_tiles(), cfg, np.array([[0.75, 0.25]]))
    assert counts == pytest.approx(share)
    # weights 2 and 1 + 1/3, so J = 2 / (10/3).
    assert counts == pytest.approx(0.6)


def test_a_share_that_does_not_cover_the_grid_raises(cfg) -> None:
    """Silently weighting the wrong tiles is the failure this rules out."""
    with pytest.raises(ValueError, match="cover it exactly"):
        _score(_two_tiles(), cfg, np.ones((1, 3)))


def test_a_demand_map_that_is_everywhere_zero_raises(cfg) -> None:
    """An empty demand map is a broken artifact, not a weightless grid."""
    with pytest.raises(ValueError, match="zero everywhere"):
        _score(_two_tiles(), cfg, np.zeros((1, 2)))


def test_the_objective_reads_the_demand_map_when_given_no_share(cfg, tmp_path) -> None:
    """The path every caller but a test takes: the demand comes off disk."""
    path = tmp_path / "demand.npz"
    demand.save({demand.SHARE: np.array([[1.0, 0.0]])}, path)
    cfg.data = {"output": {"demand_file": str(path)}}
    assert tile_share(cfg, (1, 2)).tolist() == [[1.0, 0.0]]
    assert objective(_two_tiles(), ["hi"], cfg) == pytest.approx(2.0 / 3.0)


def test_a_demand_map_on_another_grid_is_refused(cfg, tmp_path) -> None:
    """Two artifacts built on different grids must not be silently combined."""
    path = tmp_path / "demand.npz"
    demand.save({demand.SHARE: np.ones((4, 4))}, path)
    cfg.data = {"output": {"demand_file": str(path)}}
    with pytest.raises(ValueError, match="different grids"):
        tile_share(cfg, (1, 2))


def test_a_missing_demand_map_names_the_stage_that_builds_it(cfg, tmp_path) -> None:
    """The objective cannot be evaluated before preprocessing has run."""
    cfg.data = {"output": {"demand_file": str(tmp_path / "absent.npz")}}
    with pytest.raises(FileNotFoundError, match="task preprocess"):
        tile_share(cfg, (1, 2))


# --- selection -------------------------------------------------------------


def test_the_pick_ignores_every_reported_kpi() -> None:
    """A better rate with a lower objective does not win."""
    moved = _kpi(hole_rate=0.0, overlap_rate=0.0, served_rate=1.0, objective=0.49)
    assert best_by_objective([_kpi(), moved]) == 0


def test_the_highest_objective_wins() -> None:
    """A higher objective is picked."""
    assert best_by_objective([_kpi(), _kpi(objective=0.60)]) == 1


def test_an_equal_objective_keeps_the_earlier_configuration() -> None:
    """The incumbent holds unless a candidate actually scores higher."""
    assert best_by_objective([_kpi(), _kpi()]) == 0


def test_choosing_from_nothing_raises() -> None:
    """An empty run has no winner to report."""
    with pytest.raises(ValueError, match="no candidates"):
        best_by_objective([])
