"""The demand map: the per-tile report counts, the share, and what the objective reads."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data import demand


def _mdt(rows: list[tuple[int, int]]) -> pd.DataFrame:
    """An MDT from ``(tile_row, tile_col)`` tuples, one per report."""
    return pd.DataFrame(rows, columns=["tile_row", "tile_col"])


# --- the counts ------------------------------------------------------------


def test_every_report_on_a_tile_is_counted() -> None:
    """Rows, not intervals: three reports on one tile are three, whenever they arrived."""
    assert demand.reports_per_tile(_mdt([(0, 0), (0, 0), (0, 0)]), (1, 1))[0, 0] == 3.0


def test_reports_are_not_de_duplicated() -> None:
    """Identical rows are distinct UEs here; the simulator redraws positions per interval."""
    assert demand.reports_per_tile(_mdt([(0, 0), (0, 0)]), (1, 1))[0, 0] == 2.0


def test_a_tile_the_mdt_never_reported_is_zero() -> None:
    """Zero is a statement about the sample; alpha is what decides whether it matters."""
    raster = demand.reports_per_tile(_mdt([(1, 1), (1, 1)]), (2, 2))
    assert raster.tolist() == [[0.0, 0.0], [0.0, 2.0]]


def test_an_empty_mdt_gives_an_empty_map() -> None:
    """No reports is a legitimate state; nothing to raise about."""
    empty = pd.DataFrame(columns=["tile_row", "tile_col"])
    assert demand.reports_per_tile(empty, (2, 2)).sum() == 0.0


def test_a_report_off_the_grid_is_refused() -> None:
    """The MDT and the radio map would be describing different scenarios."""
    with pytest.raises(ValueError, match="different grids"):
        demand.reports_per_tile(_mdt([(0, 5)]), (2, 2))


# --- the share -------------------------------------------------------------


def test_the_share_sums_to_one() -> None:
    """The objective blends it against a uniform term, and both are read as shares."""
    share = demand.share(np.array([[1.0, 3.0], [0.0, 0.0]]))
    assert share.sum() == pytest.approx(1.0)
    assert share.ravel().tolist() == pytest.approx([0.25, 0.75, 0.0, 0.0])


def test_an_empty_demand_map_degrades_to_equal_shares() -> None:
    """No measured traffic means no reason to prefer one tile, not a division by zero."""
    assert demand.share(np.zeros((2, 2))).ravel().tolist() == pytest.approx([0.25] * 4)


def test_an_unreported_tile_keeps_no_share_of_its_own() -> None:
    """There is no smoothing: a tile beside a hotspot is weightless unless alpha < 1."""
    share = demand.share(np.array([[0.0, 0.0, 10.0, 0.0, 0.0]]))
    assert share.ravel().tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0])


# --- the artifact ----------------------------------------------------------


def test_the_artifact_round_trips_through_disk(tmp_path) -> None:
    """What preprocessing writes is what the objective reads back."""
    arrays = demand.build(_mdt([(0, 1), (0, 1)]), (1, 2), 20.0)
    cfg = OmegaConf.create(
        {"data": {"output": {"demand_file": str(demand.save(arrays, tmp_path / "demand.npz"))}}}
    )
    assert arrays[demand.REPORTS].tolist() == [[0.0, 2.0]]
    assert demand.load_share(cfg).ravel().tolist() == pytest.approx([0.0, 1.0])


def test_a_rebuilt_artifact_is_read_again(tmp_path) -> None:
    """Nothing is cached, so a rebuild inside the clock's resolution is still picked up."""
    path = tmp_path / "demand.npz"
    cfg = OmegaConf.create({"data": {"output": {"demand_file": str(path)}}})
    demand.save({demand.SHARE: np.array([[1.0, 0.0]])}, path)
    assert demand.load_share(cfg).tolist() == [[1.0, 0.0]]
    demand.save({demand.SHARE: np.array([[0.0, 1.0]])}, path)
    assert demand.load_share(cfg).tolist() == [[0.0, 1.0]]


def test_a_missing_demand_map_names_the_stage_that_builds_it(tmp_path) -> None:
    """A search that starts without one must say which task to run."""
    cfg = OmegaConf.create({"data": {"output": {"demand_file": str(tmp_path / "gone.npz")}}})
    with pytest.raises(FileNotFoundError, match="task preprocess"):
        demand.load_share(cfg)
