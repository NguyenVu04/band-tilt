"""Serving-cell choice and PRB demand.

The one serving rule in the project, read by the served rate and the load
measures: prefer bands in ``kpi.capacity.band_preference`` order while
the band's strongest cell clears ``kpi.capacity.rsrp_threshold_dbm``, else take
the strongest cell-band. That fallback is unreachable while
``kpi.capacity.rsrp_threshold_dbm`` equals ``kpi.hole_dbm``, as the committed
config sets them: a layer below the threshold is not a candidate at all, so
band preference always decides. A cell-band admits a UE only while the UE's PRBs
and the PRBs already on the cell-band together stay at or under
``kpi.capacity.max_admission_utilisation`` of ``max_prb``; otherwise the UE
passes to the next candidate in that same ranking. The cap is therefore a
ceiling on the resulting load and not a gate on the load before admission: no
cell-band ever ends an interval above that share of ``max_prb``. A layer at or
below ``kpi.hole_dbm`` is never a candidate.

Within an interval UEs are admitted in ``t_s`` order: a cell fills in the order
its reports arrive, not best-first. Two UEs reporting at the same instant are
taken strongest RSRP first, over every layer at the UE, and row order breaks
what remains.

SINR is an input, never computed here: the solver's own, from the radio map
(:func:`src.simulation.radio.solve_band`). PRBs are kept fractional: an average
over an interval, and smooth in tilt.

:func:`max_rsrp`, the strongest layer at each location, and
:func:`serving_sinr`, that layer's SINR, also live here: the hole, weak and
percentile KPIs read them. Neither is the serving rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.core.cell import Cell

# TS 38.211 4.4.4.1: one resource block is 12 consecutive subcarriers.
_SUBCARRIERS_PER_PRB = 12


def band_rank(cfg: DictConfig, band_labels: Sequence[str]) -> np.ndarray:
    """Serving preference of each band, 0 most preferred, aligned to ``band_labels``.

    The serving rule and :func:`src.evaluation.maps.serving_band` both rank
    bands by ``kpi.capacity.band_preference``, so it is read in one place. The
    objective does not.

    Raises:
        ValueError: When a band is absent from ``band_preference``, which would
            leave its rank undefined.
    """
    preference = [str(label) for label in cfg.kpi.capacity.band_preference]
    missing = [label for label in band_labels if label not in preference]
    if missing:
        raise ValueError(f"No kpi.capacity.band_preference entry for {', '.join(missing)}.")
    return np.array([preference.index(label) for label in band_labels])


@dataclass(frozen=True)
class CapacitySpec:
    """The serving and PRB settings, aligned to the radio map's bands and cells.

    Attributes:
        band_rank: Preference per band, 0 most preferred, shape ``[n_band]``.
        rsrp_threshold_dbm: Below this a band is skipped for the next one.
        max_admission_utilisation: Share of ``max_prb`` a cell-band's load may
            not exceed; an admission that would carry it past this is refused.
        min_rsrp_dbm: A layer at or below this is never a candidate.
        throughput_per_ue_bps: Assumed throughput each UE requires.
        scs_hz: Subcarrier spacing per band, shape ``[n_band]``.
        max_prb: PRB limit per cell-band, shape ``[n_band, n_tx]``.
    """

    band_rank: np.ndarray
    rsrp_threshold_dbm: float
    max_admission_utilisation: float
    min_rsrp_dbm: float
    throughput_per_ue_bps: float
    scs_hz: np.ndarray
    max_prb: np.ndarray

    @classmethod
    def from_config(cls, cfg: DictConfig, band_labels: Sequence[str], n_tx: int) -> CapacitySpec:
        """Read ``kpi.capacity``, ``kpi.hole_dbm``, the cells and each band's ``scs_hz``.

        The cells are taken in config order, which is the radio map's tx axis:
        :func:`src.simulation.radio.solve` writes them in that order and
        preprocessing checks ``tx_name`` against the config. ``scs_hz`` is read
        from ``simulation.radio_map.bands``, the same value the solver's noise
        bandwidth uses.

        Raises:
            ValueError: As :func:`band_rank`, or when a band has no
                ``simulation.radio_map.bands`` entry, the config holds other
                than ``n_tx`` cells, or ``max_admission_utilisation`` is
                outside ``(0, 1]``.
            KeyError: When a cell has no ``max_prb`` for a band.
        """
        capacity = cfg.kpi.capacity
        admission = float(capacity.max_admission_utilisation)
        if not 0.0 < admission <= 1.0:
            raise ValueError(
                f"kpi.capacity.max_admission_utilisation must be in (0, 1], got {admission}"
            )
        rank = band_rank(cfg, band_labels)
        bands = {str(entry.name): entry for entry in cfg.simulation.radio_map.bands}
        missing = [label for label in band_labels if label not in bands]
        if missing:
            raise ValueError(f"No simulation.radio_map.bands entry for {', '.join(missing)}.")
        cells = [Cell.from_config(entry) for entry in cfg.simulation.transmitters.cells]
        if len(cells) != n_tx:
            raise ValueError(
                f"simulation.transmitters.cells holds {len(cells)} cells for a radio map with "
                f"{n_tx} transmitters."
            )
        return cls(
            band_rank=rank,
            rsrp_threshold_dbm=float(capacity.rsrp_threshold_dbm),
            max_admission_utilisation=admission,
            min_rsrp_dbm=float(cfg.kpi.hole_dbm),
            throughput_per_ue_bps=float(capacity.throughput_per_ue_bps),
            scs_hz=np.array([float(bands[label].scs_hz) for label in band_labels]),
            max_prb=np.array(
                [[float(cell.max_prb_for(label)) for cell in cells] for label in band_labels]
            ),
        )


@dataclass(frozen=True)
class _Serving:
    """One interval's assignment.

    Attributes:
        band: Serving band index per UE, ``-1`` where blocked or unreachable.
        tx: Serving transmitter index per UE, ``-1`` likewise.
        prb_per_ue: PRBs each UE needs at its serving cell-band; when blocked,
            at the first candidate whose need is finite. NaN when the UE has no
            such candidate, whether because no layer reaches it or because every
            layer that does would need unbounded PRBs at its SINR.
        load: PRBs assigned per cell-band, shape ``[n_band, n_tx]``.
    """

    band: np.ndarray
    tx: np.ndarray
    prb_per_ue: np.ndarray
    load: np.ndarray


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
    return finite(rsrp).max(axis=(0, 1))


def covered(rsrp: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Mask of locations receiving something above ``cfg.kpi.hole_dbm``: not a hole."""
    return max_rsrp(rsrp) > float(cfg.kpi.hole_dbm)


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


# The PRB requirement below is a Shannon bound, not an NR link adaptation model.
# Three deviations from 3GPP, all in the optimistic direction, kept because the
# search needs a quantity that is smooth in tilt and this one is:
#
# - No modulation and coding ceiling or floor. log2(1 + SINR) has neither, so a
#   high-SINR UE is charged too few PRBs and a UE below the lowest schedulable
#   rate is charged a finite number rather than being refused.
# - The rate basis is the nominal RB bandwidth, 12 * SCS. The UE data rate of
#   TS 38.306 4.1.2 uses the symbol rate 12 / T_s^mu, T_s^mu = 1e-3 / (14 * 2^mu),
#   and scales by (1 - OH) with OH = 0.14 for downlink FR1.
# - One layer, no MIMO: no v_Layers factor and no R_max.
#
# SINR is the solver's per-RE value. Signal, interference and noise are all flat
# across the carrier, so it holds on every PRB alike and frequency-selective
# fading does not appear. The PRB limits themselves are 3GPP:
# max_prb per cell-band is N_RB from TS 38.101-1 Table 5.3.2-1.


def _spectral_efficiency(sinr: np.ndarray) -> np.ndarray:
    """Shannon spectral efficiency ``log2(1 + SINR)`` in bit/s/Hz."""
    return np.log2(1.0 + 10.0 ** (np.asarray(sinr, dtype=float) / 10.0))


def _prb_bandwidth_hz(scs_hz: float | np.ndarray) -> np.ndarray:
    """Bandwidth of one PRB: 12 subcarriers of ``scs_hz``."""
    return _SUBCARRIERS_PER_PRB * np.asarray(scs_hz, dtype=float)


def _prb_rate_bps(sinr: np.ndarray, bandwidth_hz: float | np.ndarray) -> np.ndarray:
    """Throughput of one PRB, ``B_PRB * log2(1 + SINR)``, in bit/s."""
    return np.asarray(bandwidth_hz, dtype=float) * _spectral_efficiency(sinr)


def _prb_per_ue(per_ue_bps: float, rate_bps: float | np.ndarray) -> np.ndarray:
    """PRBs one UE needs at a given per-PRB rate; ``inf`` where the rate is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(per_ue_bps, np.asarray(rate_bps, dtype=float))


def _candidate_orders(
    rsrp: np.ndarray, band_rank: np.ndarray, threshold_dbm: float, min_rsrp_dbm: float = -np.inf
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-bands each location may be served by, most preferred first.

    Args:
        rsrp: ``[n_loc, n_band, n_tx]`` RSRP in dBm, NaN where no path.
        band_rank: Preference per band, 0 most preferred.
        threshold_dbm: The RSRP a layer needs to be taken on band preference.
        min_rsrp_dbm: A layer at or below this is left out.

    Returns:
        ``(order, heard)``, each ``[n_loc, n_band * n_tx]``. ``order`` holds flat
        layer indices: layers at or above the threshold by band preference then
        RSRP, followed by the rest by RSRP. ``heard`` marks, in that same order,
        the entries that are candidates at all; no-path layers and layers at or
        below ``min_rsrp_dbm`` sort last and are unmarked.
    """
    flat = rsrp.reshape(rsrp.shape[0], rsrp.shape[1] * rsrp.shape[2])
    band = np.repeat(np.arange(rsrp.shape[1]), rsrp.shape[2])
    # NaN compares False, so no-path layers drop out here too.
    heard = flat > min_rsrp_dbm
    above = heard & (flat >= threshold_dbm)
    strength = np.where(heard, -flat, np.inf)
    # np.lexsort sorts by the last key first, and is stable, so equal keys keep
    # flat-index order.
    order = np.lexsort((strength, np.where(above, band_rank[band], 0), ~above), axis=-1)
    return order, np.take_along_axis(heard, order, axis=-1)


def _candidate_order(
    rsrp: np.ndarray, band_rank: np.ndarray, threshold_dbm: float, min_rsrp_dbm: float = -np.inf
) -> np.ndarray:
    """:func:`_candidate_orders` at one ``[n_band, n_tx]`` location, candidates only.

    The first entry is the rule's choice before capacity.
    """
    order, heard = _candidate_orders(rsrp[None], band_rank, threshold_dbm, min_rsrp_dbm)
    return order[0][heard[0]]


def _select_serving(
    rsrp: np.ndarray, sinr: np.ndarray, t_s: np.ndarray, spec: CapacitySpec
) -> _Serving:
    """Assign one interval's UEs to cell-bands under the PRB limits.

    UEs are taken in ``t_s`` order, simultaneous ones strongest RSRP first over
    every layer at the UE, and the order given breaks what remains; each walks
    its :func:`_candidate_order` and takes the first cell-band where its PRBs fit
    under ``max_admission_utilisation`` of ``max_prb``, counting the load already
    there. A cell-band therefore never passes that share.

    Args:
        rsrp: ``[n_ue, n_band, n_tx]`` RSRP at each UE's location.
        sinr: ``[n_ue, n_band, n_tx]`` SINR in dB at the same locations.
        t_s: ``[n_ue]`` report time of each UE, in seconds.
        spec: The capacity settings.
    """
    n_ue, n_band, n_tx = rsrp.shape
    bandwidth = _prb_bandwidth_hz(spec.scs_hz)[None, :, None]
    need = _prb_per_ue(spec.throughput_per_ue_bps, _prb_rate_bps(sinr, bandwidth))
    limit = spec.max_prb.ravel()

    band = np.full(n_ue, -1)
    tx = np.full(n_ue, -1)
    per_ue = np.full(n_ue, np.nan)
    load = np.zeros(n_band * n_tx)
    strongest = finite(rsrp).reshape(n_ue, -1).max(axis=1) if n_ue else np.empty(0)
    # np.lexsort sorts by the last key first, and is stable, so row order breaks
    # a UE pair tied on both time and strength.
    admission_order = np.lexsort((-strongest, t_s))
    orders, heard = _candidate_orders(
        rsrp, spec.band_rank, spec.rsrp_threshold_dbm, spec.min_rsrp_dbm
    )
    need = need.reshape(n_ue, n_band * n_tx)
    # The ranking is vectorised above; admission stays a loop because each UE
    # sees the load the earlier ones left.
    for ue in admission_order:
        ue_need = need[ue]
        order = orders[ue][heard[ue]]
        order = order[np.isfinite(ue_need[order])]
        if order.size == 0:
            continue
        ceiling = spec.max_admission_utilisation * limit[order]
        fits = order[load[order] + ue_need[order] <= ceiling]
        if fits.size == 0:
            per_ue[ue] = ue_need[order[0]]
            continue
        chosen = fits[0]
        load[chosen] += ue_need[chosen]
        per_ue[ue] = ue_need[chosen]
        band[ue], tx[ue] = divmod(int(chosen), n_tx)
    return _Serving(band=band, tx=tx, prb_per_ue=per_ue, load=load.reshape(n_band, n_tx))


def serve_rows(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    t_index: np.ndarray,
    t_s: np.ndarray,
    spec: CapacitySpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run :func:`_select_serving` once per interval.

    Args:
        rsrp: ``[n_ue, n_band, n_tx]`` RSRP each UE sees, clean or reported.
        sinr: ``[n_ue, n_band, n_tx]`` SINR in dB at the same UEs.
        t_index: Interval of each UE; UEs compete for PRBs only within one.
        t_s: Report time of each UE; sets the admission order inside an interval.
        spec: The capacity settings.

    Returns:
        ``(band, tx, prb_per_ue)`` per UE, in input order, as :class:`_Serving`
        holds them.
    """
    band = np.full(len(t_index), -1)
    tx = np.full(len(t_index), -1)
    per_ue = np.full(len(t_index), np.nan)
    for value in np.unique(t_index):
        at = np.flatnonzero(t_index == value)
        serving = _select_serving(rsrp[at], sinr[at], t_s[at], spec)
        band[at], tx[at], per_ue[at] = serving.band, serving.tx, serving.prb_per_ue
    return band, tx, per_ue


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
        ``tile_col``, ``band``, ``tx``, ``prb_per_ue``, ``sinr_db`` (at the
        serving cell-band, NaN when blocked).

    Raises:
        ValueError: When the UE table is off the map's grid or the config does not
            cover the map's bands and cells.
    """
    if spec is None:
        spec = CapacitySpec.from_config(cfg, band_labels, rsrp.shape[1])
    row, col = _tile_index(ue, rsrp.shape[-2:])
    t_index = ue["t_index"].to_numpy()

    band, tx, per_ue = serve_rows(
        rsrp[:, :, row, col].transpose(2, 0, 1),
        sinr[:, :, row, col].transpose(2, 0, 1),
        t_index,
        ue["t_s"].to_numpy(dtype=float),
        spec,
    )
    out = pd.DataFrame({"t_index": t_index, "tile_row": row, "tile_col": col}, index=ue.index)
    out["band"], out["tx"], out["prb_per_ue"] = band, tx, per_ue
    served = band >= 0
    out["sinr_db"] = np.nan
    out.loc[served, "sinr_db"] = sinr[band[served], tx[served], row[served], col[served]]
    return out


def prb_by_interval(
    t_index: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    prb_per_ue: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """PRBs required per tile in each interval: the sum of its UEs' PRBs.

    A UE with no reachable cell-band (NaN) adds nothing.

    Returns:
        ``(t_values, prb)``: the sorted intervals present, and ``prb`` shaped
        ``[n_t, n_rows, n_cols]`` aligned to them.
    """
    n_rows, n_cols = shape
    size = n_rows * n_cols
    t_values, t_pos = np.unique(t_index, return_inverse=True)
    # int64 first: the processed UE table stores tiles as int16, and row * n_cols overflows it.
    flat = t_pos * size + np.asarray(row, dtype=np.int64) * n_cols + np.asarray(col, dtype=np.int64)
    prb = np.bincount(flat, weights=np.nan_to_num(prb_per_ue), minlength=len(t_values) * size)
    return t_values, prb.reshape(len(t_values), n_rows, n_cols)


def demand_prb(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    band_labels: Sequence[str],
    ue: pd.DataFrame,
    cfg: DictConfig,
) -> np.ndarray:
    """PRBs required per tile in its busiest interval, ``[n_rows, n_cols]``.

    A blocked UE still counts at its first choice: this is demand, not what
    was served.

    Raises:
        ValueError: As :func:`serve_intervals`.
    """
    served = serve_intervals(rsrp, sinr, band_labels, ue, cfg)
    _, prb = prb_by_interval(
        served["t_index"].to_numpy(),
        served["tile_row"].to_numpy(),
        served["tile_col"].to_numpy(),
        served["prb_per_ue"].to_numpy(),
        rsrp.shape[-2:],
    )
    return prb.max(axis=0)
