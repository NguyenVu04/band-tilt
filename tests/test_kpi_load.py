"""The cell-load KPIs, on assignments small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.kpi.load import load_imbalance, prb_by_cell_interval, prb_utilisation_max, utilisation


def _served(rows: list[tuple[int, int, int, float]]) -> pd.DataFrame:
    """A serving assignment from ``(t_index, band, tx, prb_per_ue)`` tuples."""
    return pd.DataFrame(rows, columns=["t_index", "band", "tx", "prb_per_ue"])


def test_prb_by_cell_interval_sums_each_cell_band_within_each_interval() -> None:
    """Two UEs share a cell-band in interval 0; interval 1 starts it empty again."""
    served = _served([(0, 0, 1, 2.0), (0, 0, 1, 3.0), (0, 1, 0, 4.0), (1, 0, 1, 1.0)])
    t_values, prb = prb_by_cell_interval(served, n_band=2, n_tx=2)
    assert t_values.tolist() == [0, 1]
    assert prb.shape == (2, 2, 2)
    assert prb[0, 0, 1] == pytest.approx(5.0)
    assert prb[0, 1, 0] == pytest.approx(4.0)
    assert prb[1, 0, 1] == pytest.approx(1.0)
    assert prb.sum() == pytest.approx(10.0)


def test_a_blocked_ue_loads_nothing_but_still_marks_its_interval() -> None:
    """Only admitted UEs load a cell-band, and an idle interval is still an interval."""
    served = _served([(0, -1, -1, 6.0), (1, 0, 0, 2.0)])
    t_values, prb = prb_by_cell_interval(served, n_band=1, n_tx=1)
    assert t_values.tolist() == [0, 1]
    assert prb[:, 0, 0].tolist() == pytest.approx([0.0, 2.0])


def test_utilisation_is_nan_where_a_cell_band_has_no_prbs() -> None:
    """A limit of zero carries no traffic; a ratio over it would sort to the top."""
    prb = np.array([[[4.0, 0.0]]])
    share = utilisation(prb, np.array([[10.0, 0.0]]))
    assert share[0, 0, 0] == pytest.approx(0.4)
    assert np.isnan(share[0, 0, 1])


def test_prb_utilisation_max_is_the_busiest_cell_band_in_the_busiest_interval() -> None:
    """One number, over every cell-band and every interval."""
    prb = np.array([[[2.0, 8.0]], [[6.0, 1.0]]])
    assert prb_utilisation_max(prb, np.array([[10.0, 10.0]])) == pytest.approx(0.8)


def test_prb_utilisation_max_of_a_network_with_no_prbs_is_nan() -> None:
    """Nothing can be loaded, so there is no share to report."""
    assert np.isnan(prb_utilisation_max(np.zeros((1, 1, 1)), np.zeros((1, 1))))


def test_load_imbalance_is_zero_when_every_cell_band_carries_the_same_share() -> None:
    """Equal shares of different limits still balance: the measure is scale-free."""
    prb = np.array([[[5.0, 10.0]]])
    assert load_imbalance(prb, np.array([[10.0, 20.0]])) == pytest.approx(0.0)


def test_load_imbalance_rises_as_the_traffic_concentrates() -> None:
    """Two cell-bands, all the load on one: std over mean of (0.8, 0) is 1."""
    prb = np.array([[[8.0, 0.0]]])
    assert load_imbalance(prb, np.array([[10.0, 10.0]])) == pytest.approx(1.0)


def test_load_imbalance_averages_each_cell_band_over_every_interval() -> None:
    """An idle interval counts as zero load, not as a missing sample."""
    prb = np.array([[[10.0, 0.0]], [[0.0, 0.0]]])
    # Means are 0.5 and 0, so std over mean is 1 whatever the timing.
    assert load_imbalance(prb, np.array([[10.0, 10.0]])) == pytest.approx(1.0)


def test_load_imbalance_of_an_idle_network_is_nan() -> None:
    """A ratio of spread to nothing says nothing about balance."""
    assert np.isnan(load_imbalance(np.zeros((2, 1, 2)), np.array([[10.0, 10.0]])))
