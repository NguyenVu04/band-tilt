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
    ObjectiveSpec,
    best_by_objective,
    objective,
    tile_share,
)

# alpha 0 everywhere, so a test that passes no share reads the tile-uniform
# weights and the per-band terms differ only in their own layers.
_OBJECTIVE = {"tau_r_db": 10.0, "beta": 1.0, "alpha": {"hi": 0.0, "lo": 0.0}}

# Band labels for a map built by _map, in its band-axis order.
_BANDS = ("hi", "lo")


@pytest.fixture
def cfg():
    """The objective's parameters and the KPI thresholds it shares."""
    return OmegaConf.create(
        {
            "kpi": {
                "hole_dbm": -120.0,
                "overlap_margin_db": 6.0,
                "objective": dict(_OBJECTIVE),
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


def _sp(x: float) -> float:
    """The coverage utility: softplus of the threshold margin in tau_R units."""
    return math.log1p(math.exp(x))


def _map(values: list[list[float]]) -> np.ndarray:
    """A one-tile radio map from nested ``[band][tx]`` RSRP lists."""
    return np.array(values, dtype=float)[:, :, None, None]


def _even(rsrp: np.ndarray) -> np.ndarray:
    """Equal share on every tile; with alpha 0 the weights are equal regardless."""
    return np.ones(rsrp.shape[-2:])


def _score(rsrp: np.ndarray, cfg, share: np.ndarray | None = None) -> float:
    """Score a map built by :func:`_map`, labelling its bands in axis order."""
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


@pytest.mark.parametrize(
    ("key", "value"),
    [("tau_r_db", 0.0), ("beta", -0.1)],
)
def test_unusable_objective_parameters_raise(cfg, key, value) -> None:
    """Each bound keeps a band's term finite and non-negative."""
    cfg.kpi.objective[key] = value
    with pytest.raises(ValueError, match=key):
        ObjectiveSpec.from_config(cfg, ["hi"])


def test_an_alpha_outside_the_unit_interval_raises(cfg) -> None:
    """Outside [0, 1] the weights stop being a blend of two distributions."""
    cfg.kpi.objective.alpha.hi = 1.5
    with pytest.raises(ValueError, match="alpha"):
        ObjectiveSpec.from_config(cfg, ["hi"])


def test_a_band_with_no_alpha_raises(cfg) -> None:
    """A band silently defaulting to a weighting nobody recorded is the failure here."""
    with pytest.raises(ValueError, match="mid"):
        ObjectiveSpec.from_config(cfg, ["mid"])


def test_band_labels_must_match_the_map(cfg) -> None:
    """Mislabelled bands would apply the wrong alpha to the wrong layers."""
    with pytest.raises(ValueError, match="band labels"):
        objective(_map([[-100.0]]), ["hi", "lo"], cfg, np.ones((1, 1)))


def test_objective_is_ln_two_on_the_coverage_threshold(cfg) -> None:
    """At R_s = T_cov softplus is ln 2, and a server not above T_cov has no neighbours."""
    rsrp = _map([[-120.0, -121.0]])
    assert _score(rsrp, cfg, _even(rsrp)) == pytest.approx(math.log(2.0))


def test_objective_discounts_each_co_band_neighbour_by_q_ov(cfg) -> None:
    """A neighbour 4 dB down is inside the 6 dB margin; one 7 dB down is not."""
    inside = _score(_map([[-100.0, -104.0]]), cfg, np.ones((1, 1)))
    outside = _score(_map([[-100.0, -107.0]]), cfg, np.ones((1, 1)))
    covered = _sp(2.0)
    assert inside == pytest.approx(covered * math.exp(-1.0))
    assert outside == pytest.approx(covered)


def test_each_band_is_scored_on_its_own_strongest_transmitter(cfg) -> None:
    """Two bands 1 dB apart are two terms, not one band-collapsed maximum.

    Overlap stays co-band with it: neither cell is the other's neighbour.
    """
    rsrp = _map([[-100.0], [-101.0]])
    expected = _sp(2.0) + _sp(1.9)
    assert _score(rsrp, cfg, _even(rsrp)) == pytest.approx(expected)


def test_a_hole_on_one_band_is_not_hidden_by_another_band_covering_it(cfg) -> None:
    """The regression the per-band split exists to prevent.

    A band with no path scores zero on its own term however strong the other
    band is at that tile, so the hole costs a whole band's contribution.
    """
    covered = _sp(3.0)
    assert _score(_map([[-90.0], [np.nan]]), cfg, np.ones((1, 1))) == pytest.approx(covered)
    assert _score(_map([[-90.0], [-90.0]]), cfg, np.ones((1, 1))) == pytest.approx(2.0 * covered)


def test_objective_counts_crowding_on_a_band_that_does_not_serve(cfg) -> None:
    """'hi' serves alone at -90 dBm; on 'lo' tx1 is 2 dB below tx0, so m = 1.

    The 'lo' neighbour is judged against the 'lo' server, and 'lo' carries its
    own coverage term rather than being hidden behind 'hi'.
    """
    rsrp = _map([[-90.0, -130.0], [-100.0, -102.0]])
    expected = _sp(3.0) + _sp(2.0) * math.exp(-1.0)
    assert _score(rsrp, cfg, _even(rsrp)) == pytest.approx(expected)


def test_objective_scores_a_no_path_tile_zero(cfg) -> None:
    """No path is R_s = -inf: no coverage utility at all."""
    rsrp = _map([[np.nan, np.nan]])
    assert _score(rsrp, cfg, _even(rsrp)) == pytest.approx(0.0)


# --- the per-band tile weights ---------------------------------------------


def _two_tiles() -> np.ndarray:
    """One band, one cell, two tiles: one covered at T_cov, one with no path."""
    return np.array([[[[-120.0, np.nan]]]], dtype=float)


def test_alpha_zero_weights_every_tile_equally(cfg) -> None:
    """A coverage layer spends its effort on the map, not on where the UEs were."""
    rsrp = _two_tiles()
    assert _score(rsrp, cfg, np.array([[1.0, 0.0]])) == pytest.approx(0.5 * _sp(0.0))
    assert _score(rsrp, cfg, np.array([[0.0, 1.0]])) == pytest.approx(0.5 * _sp(0.0))


def test_alpha_one_weights_purely_by_the_demand_share(cfg) -> None:
    """A capacity layer scores only where the MDT reported."""
    cfg.kpi.objective.alpha.hi = 1.0
    rsrp = _two_tiles()
    assert _score(rsrp, cfg, np.array([[1.0, 1.0]])) == pytest.approx(0.5 * _sp(0.0))
    assert _score(rsrp, cfg, np.array([[1.0, 0.0]])) == pytest.approx(_sp(0.0))
    assert _score(rsrp, cfg, np.array([[0.0, 1.0]])) == pytest.approx(0.0)


def test_alpha_between_the_two_blends_them(cfg) -> None:
    """Half of each at alpha = 0.5: w = (1 - alpha) / n + alpha * p."""
    cfg.kpi.objective.alpha.hi = 0.5
    assert _score(_two_tiles(), cfg, np.array([[1.0, 0.0]])) == pytest.approx(0.75 * _sp(0.0))


def test_each_band_blends_with_its_own_alpha(cfg) -> None:
    """The point of a per-band alpha: one map, two weightings of it."""
    cfg.kpi.objective.alpha.hi = 0.0
    cfg.kpi.objective.alpha.lo = 1.0
    rsrp = np.array([[[[-120.0, np.nan]]], [[[-120.0, np.nan]]]], dtype=float)
    # 'hi' averages the two tiles; 'lo' sees only the reported one.
    assert _score(rsrp, cfg, np.array([[1.0, 0.0]])) == pytest.approx(1.5 * _sp(0.0))


def test_the_share_need_not_be_normalised(cfg) -> None:
    """It is normalised here, so a raw count raster works as well as a share."""
    cfg.kpi.objective.alpha.hi = 1.0
    assert _score(_two_tiles(), cfg, np.array([[3.0, 1.0]])) == pytest.approx(0.75 * _sp(0.0))


def test_a_share_that_does_not_cover_the_grid_raises(cfg) -> None:
    """Silently weighting the wrong tiles is the failure this rules out."""
    with pytest.raises(ValueError, match="cover it exactly"):
        _score(_two_tiles(), cfg, np.ones((1, 3)))


def test_a_share_summing_to_nothing_raises(cfg) -> None:
    """An empty demand map is a broken artifact, not an objective of zero."""
    with pytest.raises(ValueError, match="sum to zero"):
        _score(_two_tiles(), cfg, np.zeros((1, 2)))


def test_the_objective_reads_the_demand_map_when_given_no_share(cfg, tmp_path) -> None:
    """The path every caller but a test takes: the share comes off disk."""
    path = tmp_path / "demand.npz"
    demand.save({demand.SHARE: np.array([[1.0, 0.0]])}, path)
    cfg.data = {"output": {"demand_file": str(path)}}
    cfg.kpi.objective.alpha.hi = 1.0
    assert tile_share(cfg, (1, 2)).tolist() == [[1.0, 0.0]]
    assert objective(_two_tiles(), ["hi"], cfg) == pytest.approx(_sp(0.0))


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
