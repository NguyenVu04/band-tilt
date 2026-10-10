"""The KPI definitions, and the reductions they share."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.kpi import throughput_mean_mbps, throughput_percentile_mbps
from src.kpi.capacity import _tile_index, finite, max_rsrp, serve_intervals
from src.kpi.overlap import overlap_neighbors
from src.optim.objective import map_kpis
from tests.conftest import write_sectors


@pytest.fixture
def cfg(tmp_path):
    """The thresholds the KPIs read, without composing the whole config."""
    return OmegaConf.create(
        {
            "kpi": {
                "hole_dbm": -120.0,
                "weak_dbm": -90.0,
                "overlap_margin_db": 6.0,
                "capacity": {"max_admission_utilisation": 1.0},
            },
            "simulation": {
                "radio_map": {"bands": [{"name": n, "scs_hz": 15000} for n in ("hi", "lo")]},
                "input": {
                    "sectors_file": write_sectors(
                        tmp_path,
                        [
                            {
                                "name": "c0",
                                "x": 0.0,
                                "y": 0.0,
                                "z": 30.0,
                                "azimuth_deg": 0.0,
                                "tilt": {},
                                "max_prb": {"hi": 100, "lo": 100},
                            }
                        ],
                    )
                },
            },
        }
    )


def _map(values: list[list[list[float]]]) -> np.ndarray:
    """A radio map from nested ``[band][tx]`` lists of per-tile values.

    Each innermost list becomes the single row of a ``1 x n`` grid, so every
    fixture below reads as a table of sector-band layers against locations.
    """
    return np.array(values, dtype=float)[:, :, None, :]


def _sinr(rsrp: np.ndarray) -> np.ndarray:
    """A SINR map of 0 dB wherever ``rsrp`` has a path: one PRB carries 180 kbit/s."""
    return np.where(np.isfinite(rsrp), 0.0, np.nan)


def _ue(rows: list[dict[str, float]]) -> pd.DataFrame:
    """A UE frame carrying only the columns the KPIs read."""
    return pd.DataFrame(rows)


def _served(rsrp: np.ndarray, ue: pd.DataFrame, cfg) -> pd.DataFrame:
    """The serving assignment the UE KPIs reduce."""
    return serve_intervals(rsrp, _sinr(rsrp), ["hi", "lo"], ue, cfg)


def test_hole_and_weak_thresholds_are_inclusive_upper_bounds(cfg) -> None:
    """A tile exactly on a threshold falls in the class below it."""
    rsrp = _map([[[-120.0, -119.9, -90.0, -89.9]]])
    kpis = map_kpis(rsrp, _sinr(rsrp), cfg)
    assert kpis["hole_rate"] == pytest.approx(0.25)
    # -119.9 and -90.0 are weak; -89.9 is strong and -120.0 is a hole.
    assert kpis["weak_rate"] == pytest.approx(0.5)


def test_a_tile_no_transmitter_reaches_is_a_hole(cfg) -> None:
    """NaN is the ray tracer's no-path marker, not a missing value to skip."""
    rsrp = _map([[[np.nan, -80.0]]])
    kpis = map_kpis(rsrp, _sinr(rsrp), cfg)
    assert kpis["hole_rate"] == pytest.approx(0.5)
    assert kpis["weak_rate"] == pytest.approx(0.0)


def test_a_map_reaching_nothing_is_entirely_holes(cfg) -> None:
    """The soft hole term rests on this: no path must never read as coverage."""
    rsrp = _map([[[np.nan, np.nan]]])
    assert map_kpis(rsrp, _sinr(rsrp), cfg)["hole_rate"] == pytest.approx(1.0)


def test_rsrp_percentile_ignores_the_locations_with_no_coverage(cfg) -> None:
    """Conditional on coverage by design: a hole has no serving RSRP to report."""
    # No path, exactly on the hole threshold, then three covered tiles. The
    # first two are excluded, so the samples are -100, -90 and -80 only.
    rsrp = _map([[[np.nan, -120.0, -100.0, -90.0, -80.0]]])
    kpis = map_kpis(rsrp, _sinr(rsrp), cfg)
    assert kpis["rsrp_p05_dbm"] == pytest.approx(-99.0)
    assert kpis["rsrp_p50_dbm"] == pytest.approx(-90.0)


def test_rsrp_percentile_of_a_dead_map_is_minus_infinity(cfg) -> None:
    """Total outage has to order below every configuration that covers something."""
    rsrp = _map([[[np.nan, np.nan]]])
    assert map_kpis(rsrp, _sinr(rsrp), cfg)["rsrp_p05_dbm"] == -np.inf


def test_sinr_percentile_reads_the_layer_the_rsrp_percentile_reads(cfg) -> None:
    """The best server on tile 0 is band 'lo', so its SINR is the one reported."""
    rsrp = _map([[[-100.0, -95.0]], [[-80.0, -130.0]]])
    sinr = _map([[[3.0, 9.0]], [[12.0, -5.0]]])
    # Tile 0: 'lo' at -80 serves, SINR 12. Tile 1: 'hi' at -95 serves, SINR 9.
    assert map_kpis(rsrp, sinr, cfg)["sinr_p50_db"] == pytest.approx(10.5)


def test_sinr_percentile_of_a_dead_map_is_minus_infinity(cfg) -> None:
    """Nothing covered, nothing to take a percentile of."""
    rsrp = _map([[[np.nan, np.nan]]])
    assert map_kpis(rsrp, _sinr(rsrp), cfg)["sinr_p50_db"] == -np.inf


def test_overlap_counts_within_each_band_and_sums_across_them(cfg) -> None:
    """The co-band rule: a strong other-band layer is not an overlapping neighbour.

    Tile 0 has two transmitters within the margin on each band, so each band
    contributes one neighbour beyond its own serving sector. Tile 1 has one
    reachable transmitter per band and the two bands sit 20 dB apart - close
    enough to overlap if bands were compared against each other, which the
    co-band rule does not do.
    """
    rsrp = _map(
        [
            [[-80.0, -80.0], [-84.0, -130.0]],
            [[-90.0, -100.0], [-94.0, -130.0]],
        ]
    )
    assert overlap_neighbors(rsrp, cfg).tolist() == [[2, 0]]


def test_an_uncovered_band_contributes_no_neighbours(cfg) -> None:
    """Subtracting the serving sector must not take an uncovered band below zero."""
    rsrp = _map([[[-130.0], [-130.0]], [[-80.0], [-140.0]]])
    assert overlap_neighbors(rsrp, cfg).tolist() == [[0]]


def test_overlap_neighbor_mean_averages_over_covered_tiles_only(cfg) -> None:
    """Tile 0 carries two neighbours, tile 1 is a hole: the mean is 2, not 1."""
    rsrp = _map(
        [
            [[-80.0, -130.0], [-84.0, -130.0]],
            [[-90.0, -130.0], [-94.0, -130.0]],
        ]
    )
    assert overlap_neighbors(rsrp, cfg).tolist() == [[2, 0]]
    assert map_kpis(rsrp, _sinr(rsrp), cfg)["overlap_neighbor_mean"] == pytest.approx(2.0)


def test_overlap_neighbor_mean_of_a_dead_map_is_nan(cfg) -> None:
    """No coverage is not the same statement as no crowding."""
    rsrp = _map([[[np.nan, np.nan]]])
    assert np.isnan(map_kpis(rsrp, _sinr(rsrp), cfg)["overlap_neighbor_mean"])


def test_the_per_band_counts_are_what_the_total_sums(cfg) -> None:
    """A band's slice counts that band alone, which is how the per-layer KPIs read it."""
    rsrp = _map(
        [
            [[-80.0, -80.0], [-84.0, -130.0]],
            [[-90.0, -100.0], [-94.0, -130.0]],
        ]
    )
    per_band = np.stack([overlap_neighbors(rsrp[b : b + 1], cfg) for b in range(2)])
    assert per_band.tolist() == [[[1, 0]], [[1, 0]]]
    assert per_band.sum(axis=0).tolist() == overlap_neighbors(rsrp, cfg).tolist()


def test_tile_index_rejects_a_ue_off_the_map() -> None:
    """A UE outside the grid means the UE table and the map are different scenarios."""
    with pytest.raises(ValueError, match="different grids"):
        _tile_index(_ue([{"tile_row": 0, "tile_col": 5}]), (1, 4))


def test_every_covered_ue_is_served(cfg) -> None:
    """Three UEs on covered tiles: every one gets a layer, because nobody is refused."""
    rsrp = _map([[[-95.0, -105.0]], [[-70.0, -85.0]]])
    ue = _ue(
        [{"t_index": 0, "t_s": 0.0, "tile_row": 0, "tile_col": 0}] * 2
        + [{"t_index": 0, "t_s": 0.0, "tile_row": 0, "tile_col": 1}]
    )
    assert (_served(rsrp, ue, cfg)["band"] >= 0).all()


def test_a_ue_on_a_hole_is_served_by_nobody_at_zero_throughput(cfg) -> None:
    """Tile 1 is heard only at or below -120 dBm, so no layer may serve it."""
    rsrp = _map([[[-80.0, -120.0]], [[-90.0, -140.0]]])
    ue = _ue(
        [
            {"t_index": 0, "t_s": 0.0, "tile_row": 0, "tile_col": 0},
            {"t_index": 0, "t_s": 0.0, "tile_row": 0, "tile_col": 1},
        ]
    )
    served = _served(rsrp, ue, cfg)
    assert served["band"].tolist()[1] == -1
    assert served["estimated_throughput_mbps"].tolist()[1] == 0.0


def test_throughput_statistics_count_an_unserved_ue_as_zero(cfg) -> None:
    """Two UEs split 100 PRBs on tile 0 and a third is on a hole.

    At 0 dB one PRB carries 180 kbit/s, so each served UE gets 9 Mbps and the
    hole UE 0: the statistics are over 0, 9 and 9.
    """
    rsrp = _map([[[-80.0, -130.0]], [[np.nan, np.nan]]])
    ue = _ue(
        [{"t_index": 0, "t_s": 0.0, "tile_row": 0, "tile_col": 0}] * 2
        + [{"t_index": 0, "t_s": 0.0, "tile_row": 0, "tile_col": 1}]
    )
    served = _served(rsrp, ue, cfg)
    assert throughput_percentile_mbps(served, 5.0) == pytest.approx(0.9)
    assert throughput_percentile_mbps(served, 50.0) == pytest.approx(9.0)
    assert throughput_mean_mbps(served) == pytest.approx(6.0)


def test_throughput_statistics_are_zero_without_a_report() -> None:
    """No report has nothing to average; it reads as a total outage."""
    served = pd.DataFrame(columns=["band", "estimated_throughput_mbps"])
    assert throughput_percentile_mbps(served, 5.0) == 0.0
    assert throughput_mean_mbps(served) == 0.0


def test_max_rsrp_skips_no_path_layers_and_marks_unreached_tiles_minus_infinity() -> None:
    """It must equal the strongest finite layer, as ``finite(rsrp).max`` defines it."""
    rsrp = np.random.default_rng(0).uniform(-150.0, -60.0, size=(3, 4, 5, 6))
    rsrp[rsrp < -110.0] = np.nan
    rsrp[:, :, 0, 0] = np.nan
    best = max_rsrp(rsrp)
    assert best[0, 0] == -np.inf
    np.testing.assert_array_equal(best, finite(rsrp).max(axis=(0, 1)))
