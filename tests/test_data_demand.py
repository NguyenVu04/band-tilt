"""The demand map: the per-tile median, the KDE, and what the objective reads."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data import demand


@pytest.fixture
def cfg():
    """A KDE with no smoothing and no uniform floor, so a test reads the medians."""
    return OmegaConf.create({"data": {"demand": {"bandwidth_m": 0.0, "uniform_share": 0.0}}})


def _mdt(rows: list[tuple[int, int, int, float]]) -> pd.DataFrame:
    """An MDT from ``(t_index, tile_row, tile_col, prb_per_ue)`` tuples."""
    return pd.DataFrame(rows, columns=["t_index", "tile_row", "tile_col", "prb_per_ue"])


# --- the median ------------------------------------------------------------


def test_reports_sharing_a_tile_and_interval_are_summed_before_the_median() -> None:
    """Two UEs on one tile at one instant ask for both their PRBs, not the larger."""
    mdt = _mdt([(0, 0, 0, 2.0), (0, 0, 0, 3.0)])
    assert demand.median_prb_per_tile(mdt, (1, 1))[0, 0] == pytest.approx(5.0)


def test_the_median_is_taken_over_the_intervals_the_tile_was_reported_in() -> None:
    """Three occupied intervals of a much longer horizon: the median is of those three.

    Over every interval in the horizon the median would be zero, which is the
    whole reason this basis is the one chosen.
    """
    mdt = _mdt([(0, 0, 0, 1.0), (5, 0, 0, 4.0), (99, 0, 0, 10.0)])
    assert demand.median_prb_per_tile(mdt, (1, 1))[0, 0] == pytest.approx(4.0)


def test_a_tile_the_mdt_never_reported_is_zero() -> None:
    """Zero is a statement about the sample, and the KDE is what fills it in."""
    raster = demand.median_prb_per_tile(_mdt([(0, 1, 1, 6.0)]), (2, 2))
    assert raster.tolist() == [[0.0, 0.0], [0.0, 6.0]]


def test_an_empty_mdt_gives_an_empty_map() -> None:
    """No reports is a legitimate state; nothing to raise about."""
    empty = pd.DataFrame(columns=["t_index", "tile_row", "tile_col", "prb_per_ue"])
    assert demand.median_prb_per_tile(empty, (2, 2)).sum() == 0.0


def test_a_report_off_the_grid_is_refused() -> None:
    """The MDT and the radio map would be describing different scenarios."""
    with pytest.raises(ValueError, match="different grids"):
        demand.median_prb_per_tile(_mdt([(0, 0, 5, 1.0)]), (2, 2))


# --- the weights -----------------------------------------------------------


def test_the_weights_sum_to_one(cfg) -> None:
    """The objective divides by their sum, and every table reads them as shares."""
    spec = demand.DemandSpec.from_config(cfg)
    weight = demand.kde_weights(np.array([[1.0, 3.0], [0.0, 0.0]]), 20.0, spec)
    assert weight.sum() == pytest.approx(1.0)
    assert weight.ravel().tolist() == pytest.approx([0.25, 0.75, 0.0, 0.0])


def test_smoothing_moves_weight_onto_tiles_with_no_report_of_their_own(cfg) -> None:
    """The point of the KDE: a tile beside a hotspot is not weightless."""
    cfg.data.demand.bandwidth_m = 20.0
    spec = demand.DemandSpec.from_config(cfg)
    median = np.zeros((1, 5))
    median[0, 2] = 10.0
    weight = demand.kde_weights(median, 20.0, spec)
    assert weight.sum() == pytest.approx(1.0)
    assert weight[0, 1] > 0.0
    assert weight[0, 2] > weight[0, 1] > weight[0, 0]


def test_an_empty_demand_map_degrades_to_equal_weights(cfg) -> None:
    """No measured traffic means no reason to prefer one tile, not a division by zero."""
    spec = demand.DemandSpec.from_config(cfg)
    weight = demand.kde_weights(np.zeros((2, 2)), 20.0, spec)
    assert weight.ravel().tolist() == pytest.approx([0.25] * 4)


def test_the_uniform_share_puts_a_floor_under_every_tile(cfg) -> None:
    """Ground the MDT never reported keeps a fraction of the influence."""
    cfg.data.demand.uniform_share = 0.5
    spec = demand.DemandSpec.from_config(cfg)
    weight = demand.kde_weights(np.array([[4.0, 0.0]]), 20.0, spec)
    assert weight.sum() == pytest.approx(1.0)
    assert weight.ravel().tolist() == pytest.approx([0.75, 0.25])


@pytest.mark.parametrize(("key", "value"), [("bandwidth_m", -1.0), ("uniform_share", 1.5)])
def test_unusable_kde_settings_raise(cfg, key, value) -> None:
    """Each bound keeps the weights non-negative and summing to one."""
    cfg.data.demand[key] = value
    with pytest.raises(ValueError, match=key):
        demand.DemandSpec.from_config(cfg)


def test_a_config_with_no_demand_block_raises(cfg) -> None:
    """The objective cannot fall back to a default nobody recorded."""
    del cfg.data.demand
    with pytest.raises(ValueError, match="no `demand` block"):
        demand.DemandSpec.from_config(cfg)


# --- the artifact ----------------------------------------------------------


def test_the_artifact_round_trips_through_disk(cfg, tmp_path) -> None:
    """What preprocessing writes is what the objective reads back."""
    arrays = demand.build(_mdt([(0, 0, 1, 8.0)]), (1, 2), 20.0, cfg)
    cfg.data.output = {"demand_file": str(demand.save(arrays, tmp_path / "demand.npz"))}
    assert arrays[demand.MEDIAN_PRB].tolist() == [[0.0, 8.0]]
    assert demand.load_weights(cfg).ravel().tolist() == pytest.approx([0.0, 1.0])


def test_a_rebuilt_artifact_is_read_again(cfg, tmp_path) -> None:
    """Nothing is cached, so a rebuild inside the clock's resolution is still picked up."""
    path = tmp_path / "demand.npz"
    cfg.data.output = {"demand_file": str(path)}
    demand.save({demand.WEIGHT: np.array([[1.0, 0.0]])}, path)
    assert demand.load_weights(cfg).tolist() == [[1.0, 0.0]]
    demand.save({demand.WEIGHT: np.array([[0.0, 1.0]])}, path)
    assert demand.load_weights(cfg).tolist() == [[0.0, 1.0]]
