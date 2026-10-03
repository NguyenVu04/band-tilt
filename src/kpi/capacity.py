"""Serving-cell choice and the throughput each UE gets.

The one serving rule in the project, read by the UE KPIs and the per-cell-band
tables. Within an interval UEs connect one at a time. A UE's candidates are the
cell-bands above ``kpi.hole_dbm``; it joins the one where its equal share of the
usable PRBs, ``kpi.capacity.max_admission_utilisation`` of ``max_prb`` split over
the UEs already there plus itself, carries the most throughput. Nobody is
refused: a UE with no candidate is the only one not served.

Its estimated throughput is read after the interval's last UE has connected,
at the equal share of its cell-band's final UE count, so a UE's figure falls as
later UEs join the same cell-band.

UEs connect in ``t_s`` order. Two UEs reporting at the same instant are taken
strongest RSRP first, over every layer at the UE, and row order breaks what
remains.

SINR is an input, never computed here: the solver's own, from the radio map
(:func:`src.simulation.radio.solve_band`).

:func:`max_rsrp`, the strongest layer at each location (the best server), and
:func:`serving_sinr`, that layer's SINR, also live here: the hole, weak and
signal-quality KPIs read them. Neither is the serving rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.core.cell import read_cells

# TS 38.211 4.4.4.1: one resource block is 12 consecutive subcarriers.
_SUBCARRIERS_PER_PRB = 12


@dataclass(frozen=True)
class CapacitySpec:
    """The serving and PRB settings, aligned to the radio map's bands and cells.

    Attributes:
        min_rsrp_dbm: A layer at or below this is never a candidate.
        max_admission_utilisation: Share of ``max_prb`` shared among a
            cell-band's UEs.
        prb_bandwidth_hz: Bandwidth of one PRB per band, shape ``[n_band]``.
        max_prb: PRB limit per cell-band, shape ``[n_band, n_tx]``.
    """

    min_rsrp_dbm: float
    max_admission_utilisation: float
    prb_bandwidth_hz: np.ndarray
    max_prb: np.ndarray

    @property
    def pool_prb(self) -> np.ndarray:
        """PRBs shared among a cell-band's UEs, ``[n_band, n_tx]``."""
        return self.max_admission_utilisation * self.max_prb

    @classmethod
    def from_config(cls, cfg: DictConfig, band_labels: Sequence[str], n_tx: int) -> CapacitySpec:
        """Read ``kpi.capacity``, ``kpi.hole_dbm``, the cells and each band's ``scs_hz``.

        The cells are read from ``simulation.input.cells_file`` in table order, which
        is the radio map's tx axis: :func:`src.simulation.radio.solve` writes
        them in that order and preprocessing checks ``tx_name`` against it. ``scs_hz`` is read
        from ``simulation.radio_map.bands``, the same value the solver's noise
        bandwidth uses.

        Raises:
            ValueError: When a band has no ``simulation.radio_map.bands`` entry,
                the config holds other than ``n_tx`` cells, or
                ``max_admission_utilisation`` is outside ``(0, 1]``.
            KeyError: When a cell has no ``max_prb`` for a band.
        """
        admission = float(cfg.kpi.capacity.max_admission_utilisation)
        if not 0.0 < admission <= 1.0:
            raise ValueError(
                f"kpi.capacity.max_admission_utilisation must be in (0, 1], got {admission}"
            )
        bands = {str(entry.name): entry for entry in cfg.simulation.radio_map.bands}
        missing = [label for label in band_labels if label not in bands]
        if missing:
            raise ValueError(f"No simulation.radio_map.bands entry for {', '.join(missing)}.")
        cells = read_cells(cfg.simulation.input.cells_file)
        if len(cells) != n_tx:
            raise ValueError(
                f"{cfg.simulation.input.cells_file} holds {len(cells)} cells for a radio map with "
                f"{n_tx} transmitters."
            )
        return cls(
            min_rsrp_dbm=float(cfg.kpi.hole_dbm),
            max_admission_utilisation=admission,
            prb_bandwidth_hz=_prb_bandwidth_hz(
                [float(bands[label].scs_hz) for label in band_labels]
            ),
            max_prb=np.array(
                [[float(cell.max_prb_for(label)) for cell in cells] for label in band_labels]
            ),
        )


def finite(rsrp: np.ndarray) -> np.ndarray:
    """RSRP with the ray tracer's no-path NaN replaced by ``-inf``.

    Substituting once, here, is what lets every threshold comparison downstream
    run without a NaN special case: an unreachable location compares as a hole
    on its own.
    """
    return np.where(np.isfinite(rsrp), rsrp, -np.inf)


def max_rsrp(rsrp: np.ndarray) -> np.ndarray:
    """Strongest signal at each location, over every cell-band layer.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``, NaN where
            no path was found.

    Returns:
        ``R_max(g)``, shape ``[n_rows, n_cols]``, ``-inf`` where nothing is
        received.
    """
    # fmax skips NaN, so the no-path layers need no full-map -inf copy first.
    best = np.fmax.reduce(np.reshape(rsrp, (-1, *np.shape(rsrp)[-2:])), axis=0)
    return np.where(np.isnan(best), -np.inf, best)


def covered(rsrp: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Mask of locations receiving something above ``cfg.kpi.hole_dbm``: not a hole."""
    return covered_best(max_rsrp(rsrp), cfg)


def covered_best(r_max: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """:func:`covered` from a :func:`max_rsrp` already taken."""
    return r_max > float(cfg.kpi.hole_dbm)


def serving_sinr(rsrp: np.ndarray, sinr: np.ndarray) -> np.ndarray:
    """SINR of the strongest layer at each location.

    The layer :func:`max_rsrp` reports, read out of the solver's own SINR, so
    the RSRP and SINR percentiles describe the same cell-band at every tile.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``, NaN where
            no path was found.
        sinr: The solver's SINR in dB, same shape.

    Returns:
        ``[n_rows, n_cols]`` in dB. Where nothing is received every layer ties
        at ``-inf`` and the value is whatever the first layer holds, usually
        NaN; callers mask on coverage before reading it.
    """
    layers = finite(rsrp).reshape(-1, *rsrp.shape[-2:])
    best = layers.argmax(axis=0)[None]
    return np.take_along_axis(np.asarray(sinr, dtype=float).reshape(layers.shape), best, axis=0)[0]


def _tile_index(ue: pd.DataFrame, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The UE table's ``(row, col)`` tile indices, checked against the map's grid.

    Args:
        ue: The UE table, one row per UE per interval, carrying ``tile_row``
            and ``tile_col``.
        shape: The radio map's ``(n_rows, n_cols)``.

    Returns:
        ``(row, col)``, each of shape ``[n_report]``, ready to index a
        ``[n_rows, n_cols]`` raster.

    Raises:
        ValueError: When a UE falls outside the map's grid, which means the UE table
            and the radio map were built on different grids.
    """
    n_rows, n_cols = shape
    row = ue["tile_row"].to_numpy()
    col = ue["tile_col"].to_numpy()
    # No UE is outside no grid. Without this, min() on the empty array raises
    # numpy's "zero-size array to reduction" instead of anything a caller can act on.
    if not len(row):
        return row, col
    if row.min() < 0 or row.max() >= n_rows or col.min() < 0 or col.max() >= n_cols:
        raise ValueError(
            f"UE tiles span rows {row.min()}..{row.max()} cols {col.min()}..{col.max()}, "
            f"outside the radio map's {n_rows} x {n_cols} grid. The two were built on "
            "different grids."
        )
    return row, col


# The throughput below is a Shannon bound, not an NR link adaptation model.
# Three deviations from 3GPP, all in the optimistic direction, kept because the
# search needs a quantity that is smooth in tilt and this one is:
#
# - No modulation and coding ceiling or floor. log2(1 + SINR) has neither, so a
#   high-SINR UE is credited more than the top MCS carries and a UE below the
#   lowest schedulable rate is credited a small positive rate, not zero.
# - The rate basis is the nominal RB bandwidth, 12 * SCS. The UE data rate of
#   TS 38.306 4.1.2 uses the symbol rate 12 / T_s^mu, T_s^mu = 1e-3 / (14 * 2^mu),
#   and scales by (1 - OH) with OH = 0.14 for downlink FR1.
# - One layer, no MIMO: no v_Layers factor and no R_max.
#
# SINR is the solver's per-RE value. Signal, interference and noise are all flat
# across the carrier, so it holds on every PRB alike and frequency-selective
# fading does not appear. The PRB limits themselves are 3GPP:
# max_prb per cell-band is N_RB from TS 38.101-1 Table 5.3.2-1.


def spectral_efficiency(sinr: np.ndarray) -> np.ndarray:
    """Shannon spectral efficiency ``log2(1 + SINR)`` in bit/s/Hz, SINR in dB."""
    return np.log2(1.0 + 10.0 ** (np.asarray(sinr, dtype=float) / 10.0))


def _prb_bandwidth_hz(scs_hz: float | Sequence[float] | np.ndarray) -> np.ndarray:
    """Bandwidth of one PRB: 12 subcarriers of ``scs_hz``."""
    return _SUBCARRIERS_PER_PRB * np.asarray(scs_hz, dtype=float)


def _prb_rate_bps(sinr: np.ndarray, bandwidth_hz: float | np.ndarray) -> np.ndarray:
    """Throughput of one PRB, ``B_PRB * log2(1 + SINR)``, in bit/s."""
    return np.asarray(bandwidth_hz, dtype=float) * spectral_efficiency(sinr)


def _select_serving(
    rsrp: np.ndarray, sinr: np.ndarray, t_s: np.ndarray, spec: CapacitySpec
) -> tuple[np.ndarray, np.ndarray]:
    """Connect one interval's UEs, each to the cell-band that gives it the most throughput.

    UEs are taken in ``t_s`` order, simultaneous ones strongest RSRP first over
    every layer at the UE, and the order given breaks what remains. Each takes
    the candidate maximising ``pool_prb / (n + 1) * rate``, ``n`` the UEs
    already there; a tie keeps the lower flat layer index.

    Args:
        rsrp: ``[n_ue, n_band, n_tx]`` RSRP at each UE's location.
        sinr: ``[n_ue, n_band, n_tx]`` SINR in dB at the same locations.
        t_s: ``[n_ue]`` report time of each UE, in seconds.
        spec: The capacity settings.

    Returns:
        ``(layer, throughput_bps)`` per UE: the flat ``band * n_tx + tx``
        index, ``-1`` where no layer is above ``min_rsrp_dbm``, and the
        throughput at the interval's final UE count, NaN where not served.
    """
    n_ue = rsrp.shape[0]
    rate = _prb_rate_bps(sinr, spec.prb_bandwidth_hz[None, :, None]).reshape(n_ue, -1)
    pool = spec.pool_prb.ravel()
    # NaN compares False, so no-path layers drop out here too.
    candidate = (rsrp.reshape(n_ue, -1) > spec.min_rsrp_dbm) & np.isfinite(rate)

    layer = np.full(n_ue, -1)
    count = np.zeros(pool.size)
    strongest = finite(rsrp).reshape(n_ue, -1).max(axis=1) if n_ue else np.empty(0)
    # np.lexsort sorts by the last key first, and is stable, so row order breaks
    # a UE pair tied on both time and strength.
    # The choice stays a loop because each UE sees the counts the earlier ones left.
    for ue in np.lexsort((-strongest, t_s)):
        if not candidate[ue].any():
            continue
        offer = np.where(candidate[ue], pool / (count + 1.0) * rate[ue], -np.inf)
        layer[ue] = int(np.argmax(offer))
        count[layer[ue]] += 1.0

    throughput = np.full(n_ue, np.nan)
    served = np.flatnonzero(layer >= 0)
    chosen = layer[served]
    throughput[served] = pool[chosen] / count[chosen] * rate[served, chosen]
    return layer, throughput


def serve_rows(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    t_index: np.ndarray,
    t_s: np.ndarray,
    spec: CapacitySpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run :func:`_select_serving` once per interval.

    Args:
        rsrp: ``[n_ue, n_band, n_tx]`` RSRP each UE sees.
        sinr: ``[n_ue, n_band, n_tx]`` SINR in dB at the same UEs.
        t_index: Interval of each UE; UEs share PRBs only within one.
        t_s: Report time of each UE; sets the connect order inside an interval.
        spec: The capacity settings.

    Returns:
        ``(band, tx, throughput_bps)`` per UE, in input order; band and tx are
        ``-1`` and throughput NaN where no layer is above ``min_rsrp_dbm``.
    """
    n_tx = rsrp.shape[2]
    layer = np.full(len(t_index), -1)
    throughput = np.full(len(t_index), np.nan)
    # One stable sort groups the intervals, keeping row order inside each,
    # rather than a full scan of t_index per interval.
    order = np.argsort(t_index, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(t_index[order]) != 0])
    for at in np.split(order, starts[1:]) if order.size else ():
        layer[at], throughput[at] = _select_serving(rsrp[at], sinr[at], t_s[at], spec)
    band, tx = np.divmod(layer, n_tx)
    unserved = layer < 0
    band[unserved], tx[unserved] = -1, -1
    return band, tx, throughput


def serve_intervals(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    band_labels: Sequence[str],
    ue: pd.DataFrame,
    cfg: DictConfig,
    spec: CapacitySpec | None = None,
) -> pd.DataFrame:
    """Serve every UE row from the radio map, via :func:`serve_rows`.

    Args:
        rsrp: Radio map in dBm, ``[n_band, n_tx, n_rows, n_cols]``, read at
            each UE's tile, so the result follows the tilts.
        sinr: The solver's SINR in dB, same shape and axes as ``rsrp``.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        ue: Supplies ``t_index``, ``t_s``, ``tile_row`` and ``tile_col`` only.
        cfg: Composed config; see :meth:`CapacitySpec.from_config`.
        spec: The capacity model already read from ``cfg``, to skip re-reading
            it when serving many maps of the same network.

    Returns:
        One row per UE row, index aligned: ``t_index``, ``tile_row``,
        ``tile_col``, ``band``, ``tx`` (``-1`` when not served), ``sinr_db``
        and ``estimated_throughput_mbps`` at the serving cell-band, NaN when
        not served.

    Raises:
        ValueError: When the UE table is off the map's grid or the config does not
            cover the map's bands and cells.
    """
    if spec is None:
        spec = CapacitySpec.from_config(cfg, band_labels, rsrp.shape[1])
    row, col = _tile_index(ue, rsrp.shape[-2:])
    t_index = ue["t_index"].to_numpy()

    band, tx, throughput = serve_rows(
        rsrp[:, :, row, col].transpose(2, 0, 1),
        sinr[:, :, row, col].transpose(2, 0, 1),
        t_index,
        ue["t_s"].to_numpy(dtype=float),
        spec,
    )
    out = pd.DataFrame({"t_index": t_index, "tile_row": row, "tile_col": col}, index=ue.index)
    out["band"], out["tx"] = band, tx
    served = band >= 0
    out["sinr_db"] = np.nan
    out.loc[served, "sinr_db"] = sinr[band[served], tx[served], row[served], col[served]]
    out["estimated_throughput_mbps"] = throughput / 1e6
    return out
