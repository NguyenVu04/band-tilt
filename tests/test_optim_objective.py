"""The KPI vector, the sign convention, and the objective that picks a winner."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.optim.objective import (
    KPI_NAMES,
    MAXIMISED,
    MEASURE_NAMES,
    OBJECTIVE_NAMES,
    KpiVector,
    ObjectiveSpec,
    best_by_score,
    cvar,
    j_load,
    j_radio,
    score,
)

_OBJECTIVE = {"tau_r_db": 10.0, "beta": 1.0, "rho_0": 0.5, "alpha": 0.5, "gamma": 0.5}


@pytest.fixture
def cfg():
    """One band, one cell of 10 PRBs; each UE at 0 dB SINR needs exactly one PRB."""
    return OmegaConf.create(
        {
            "kpi": {
                "hole_dbm": -120.0,
                "overlap_margin_db": 6.0,
                "objective": dict(_OBJECTIVE),
                "capacity": {
                    "band_preference": ["hi"],
                    "rsrp_threshold_dbm": -120.0,
                    "max_admission_utilisation": 1.0,
                    "throughput_per_ue_bps": 180_000.0,
                    "bands": {"hi": {"scs_hz": 15000}},
                },
            },
            "simulation": {
                "seed": 0,
                "transmitters": {
                    "cells": [
                        {
                            "name": "c0",
                            "x": 0.0,
                            "y": 0.0,
                            "z": 30.0,
                            "azimuth_deg": 0.0,
                            "tilt": {},
                            "max_prb": {"hi": 10},
                        }
                    ]
                },
            },
        }
    )


def _kpi(**overrides: float) -> KpiVector:
    """A middling measurement, with named fields overridden."""
    values = {
        "hole_rate": 0.10,
        "overlap_rate": 0.30,
        "served_ratio": 0.20,
        "weak_rate": 0.10,
        "edge_rsrp_dbm": -105.0,
        "j_radio": 0.50,
        "j_load": 0.50,
    }
    return KpiVector(**{**values, **overrides})


def _map(values: list[list[float]]) -> np.ndarray:
    """A one-tile radio map from nested ``[band][tx]`` RSRP lists."""
    return np.array(values, dtype=float)[:, :, None, None]


def test_priority_order_is_the_adr_order() -> None:
    """Reordering this changes which column of every table is which."""
    assert KPI_NAMES == ("hole_rate", "overlap_rate", "served_ratio", "weak_rate", "edge_rsrp_dbm")
    assert MEASURE_NAMES == KPI_NAMES + OBJECTIVE_NAMES == KPI_NAMES + ("j_radio", "j_load")


def test_only_the_ue_edge_and_objective_measures_are_maximised() -> None:
    """The usual place a sign error hides."""
    assert MAXIMISED == {"served_ratio", "edge_rsrp_dbm", "j_radio", "j_load"}


def test_as_dict_round_trips_through_from_mapping() -> None:
    """The shape run.json records and the evaluation stage reads back."""
    kpi = _kpi(hole_rate=0.123)
    assert KpiVector.from_mapping(kpi.as_dict()) == kpi


def test_from_mapping_names_a_missing_measure() -> None:
    """An incomplete measurement must not become a silent zero."""
    values = _kpi().as_dict()
    del values["j_load"]
    with pytest.raises(KeyError, match="j_load"):
        KpiVector.from_mapping(values)


@pytest.mark.parametrize(
    ("key", "value"),
    [("tau_r_db", 0.0), ("beta", -0.1), ("rho_0", 1.0), ("alpha", 1.0), ("gamma", 1.5)],
)
def test_unusable_objective_parameters_raise(cfg, key, value) -> None:
    """Each bound keeps its term finite and inside [0, 1]."""
    cfg.kpi.objective[key] = value
    with pytest.raises(ValueError, match=key):
        ObjectiveSpec.from_config(cfg)


def test_j_radio_is_half_on_the_coverage_threshold(cfg) -> None:
    """At R_s = T_cov the sigmoid is a half, and a server not above T_cov has no neighbours."""
    assert j_radio(_map([[-120.0, -121.0]]), cfg) == pytest.approx(0.5)


def test_j_radio_discounts_each_co_band_neighbour_by_q_ov(cfg) -> None:
    """A neighbour 4 dB down is inside the 6 dB margin; one 7 dB down is not."""
    inside = j_radio(_map([[-100.0, -104.0]]), cfg)
    outside = j_radio(_map([[-100.0, -107.0]]), cfg)
    covered = 1.0 / (1.0 + math.exp(-2.0))
    assert inside == pytest.approx(covered * math.exp(-1.0))
    assert outside == pytest.approx(covered)


def test_j_radio_ignores_a_close_layer_on_another_band(cfg) -> None:
    """Overlap is co-band: the other band's cell 1 dB down is not a neighbour."""
    assert j_radio(_map([[-100.0], [-101.0]]), cfg) == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))


def test_j_radio_counts_crowding_on_a_band_that_does_not_serve(cfg) -> None:
    """'hi' serves alone at -90 dBm; on 'lo' tx1 is 2 dB below tx0, so m = 1.

    The 'lo' neighbour is judged against the 'lo' server, not against R_s, which
    it is 12 dB below.
    """
    rsrp = _map([[-90.0, -130.0], [-100.0, -102.0]])
    assert j_radio(rsrp, cfg) == pytest.approx(1.0 / (1.0 + math.exp(-3.0)) * math.exp(-1.0))


def test_j_radio_scores_a_no_path_tile_zero(cfg) -> None:
    """No path is R_s = -inf: no coverage utility at all."""
    assert j_radio(_map([[np.nan, np.nan]]), cfg) == pytest.approx(0.0)


def test_cvar_is_the_upper_tail_mean() -> None:
    """An alpha of 0.5 keeps the top two of four values; an alpha of 0 keeps all."""
    assert cvar(np.array([1.0, 4.0, 2.0, 3.0]), 0.5) == pytest.approx(3.5)
    assert cvar(np.array([1.0, 4.0, 2.0, 3.0]), 0.0) == pytest.approx(2.5)


def test_j_load_reads_the_tail_of_utilisation_over_rho_0(cfg) -> None:
    """9 UEs then 1 UE on a 10-PRB cell: rho 0.9 and 0.1, excess 0.4 and 0.

    alpha = 0.5 keeps the worse interval, so J_load = 1 - 0.4 / (1 - 0.5).
    """
    rsrp = np.array([[[[-90.0]]]])
    sinr = np.zeros_like(rsrp)
    ue = pd.DataFrame({"t_index": [0] * 9 + [1], "tile_row": [0] * 10, "tile_col": [0] * 10})
    assert j_load(rsrp, sinr, ["hi"], ue, cfg) == pytest.approx(0.2)
    cfg.kpi.objective.rho_0 = 0.95
    assert j_load(rsrp, sinr, ["hi"], ue, cfg) == pytest.approx(1.0)


def test_the_score_is_the_gamma_weighted_geometric_mean(cfg) -> None:
    """A gamma of 0.5 is the geometric mean; a gamma of 1 ignores load entirely."""
    kpi = _kpi(j_radio=0.81, j_load=0.25)
    assert score([kpi], cfg)[0] == pytest.approx(math.sqrt(0.81 * 0.25))
    cfg.kpi.objective.gamma = 1.0
    assert score([kpi], cfg)[0] == pytest.approx(0.81)


def test_the_score_ignores_every_reported_kpi(cfg) -> None:
    """Moving a rate must not move the objective."""
    moved = _kpi(hole_rate=0.99, overlap_rate=0.99, served_ratio=0.01, edge_rsrp_dbm=-60.0)
    assert score([moved], cfg)[0] == pytest.approx(score([_kpi()], cfg)[0])


def test_the_highest_score_wins(cfg) -> None:
    """A gain in one term with nothing lost elsewhere raises the score."""
    assert best_by_score([_kpi(), _kpi(j_load=0.60)], cfg) == 1


def test_an_equal_score_keeps_the_earlier_configuration(cfg) -> None:
    """The incumbent holds unless a candidate actually scores higher."""
    assert best_by_score([_kpi(), _kpi()], cfg) == 0


def test_choosing_from_nothing_raises(cfg) -> None:
    """An empty run has no winner to report."""
    with pytest.raises(ValueError, match="no candidates"):
        best_by_score([], cfg)
