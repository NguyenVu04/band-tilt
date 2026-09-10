"""Serving-cell choice and PRB demand - a diagnostic beside the KPIs, not one.

A temporary serving rule: prefer bands in ``kpi.band_priority`` order while the
band's strongest cell clears ``kpi.capacity.rsrp_threshold_dbm``, else take the
strongest cell-band; a cell-band out of PRBs passes the UE to the next candidate
in that same ranking.

SINR assumes every co-band transmitter is fully loaded, so it is a lower bound
on what a scheduler would see. PRBs are kept fractional: an average over an
interval, and smooth in tilt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.kpi.tiles import tile_index

# TS 38.211 4.4.4.1: one resource block is 12 consecutive subcarriers.
SUBCARRIERS_PER_PRB = 12

# Thermal noise density kT at 290 K.
_THERMAL_DBM_PER_HZ = -174.0


@dataclass(frozen=True)
class CapacitySpec:
    """``kpi.capacity`` and the band preference, aligned to the radio map's bands.

    Attributes:
        band_rank: Preference per band, 0 most preferred, shape ``[n_band]``.
        rsrp_threshold_dbm: Below this a band is skipped for the next one.
        throughput_per_ue_bps: Average throughput each UE requires.
        noise_figure_db: UE receiver noise figure.
        scs_hz: Subcarrier spacing per band, shape ``[n_band]``.
        n_prb: PRB limit per cell-band, per band, shape ``[n_band]``.
    """

    band_rank: np.ndarray
    rsrp_threshold_dbm: float
    throughput_per_ue_bps: float
    noise_figure_db: float
    scs_hz: np.ndarray
    n_prb: np.ndarray

    @property
    def noise_dbm(self) -> np.ndarray:
        """Noise power per resource element, per band."""
        return noise_per_re_dbm(self.scs_hz, self.noise_figure_db)

    @classmethod
    def from_config(cls, cfg: DictConfig, band_labels: Sequence[str]) -> CapacitySpec:
        """Read ``kpi.capacity`` and ``kpi.band_priority``.

        Raises:
            ValueError: When a band has no priority weight or no capacity entry.
        """
        capacity = cfg.kpi.capacity
        missing = [
            label
            for label in band_labels
            if label not in cfg.kpi.band_priority or label not in capacity.bands
        ]
        if missing:
            raise ValueError(
                f"No kpi.band_priority or kpi.capacity.bands entry for {', '.join(missing)}."
            )
        weight = np.array([float(cfg.kpi.band_priority[label]) for label in band_labels])
        return cls(
            band_rank=np.argsort(np.argsort(-weight, kind="stable"), kind="stable"),
            rsrp_threshold_dbm=float(capacity.rsrp_threshold_dbm),
            throughput_per_ue_bps=float(capacity.throughput_per_ue_bps),
            noise_figure_db=float(capacity.noise_figure_db),
            scs_hz=np.array([float(capacity.bands[label].scs_hz) for label in band_labels]),
            n_prb=np.array([float(capacity.bands[label].n_prb) for label in band_labels]),
        )


@dataclass(frozen=True)
class Serving:
    """One interval's assignment.

    Attributes:
        band: Serving band index per UE, ``-1`` where blocked or unreachable.
        tx: Serving transmitter index per UE, ``-1`` likewise.
        prb_per_ue: PRBs each UE needs at its serving cell-band, or at its
            first choice when blocked; NaN when no cell-band reaches it.
        load: PRBs assigned per cell-band, shape ``[n_band, n_tx]``.
    """

    band: np.ndarray
    tx: np.ndarray
    prb_per_ue: np.ndarray
    load: np.ndarray


def noise_per_re_dbm(scs_hz: float | np.ndarray, noise_figure_db: float) -> np.ndarray:
    """Thermal noise plus noise figure over one subcarrier, in dBm."""
    return _THERMAL_DBM_PER_HZ + 10.0 * np.log10(np.asarray(scs_hz, dtype=float)) + noise_figure_db


def sinr_db(rsrp: np.ndarray, noise_dbm: float | np.ndarray) -> np.ndarray:
    """Co-band SINR of every cell-band layer, with full-load interference.

    Args:
        rsrp: RSRP in dBm, ``[n_band, n_tx, ...]``, NaN where no path.
        noise_dbm: Noise per resource element, scalar or one per band.

    Returns:
        Same shape as ``rsrp``: each layer against every other transmitter on
        its band plus noise. ``-inf`` where the layer has no path.
    """
    power = np.where(np.isfinite(rsrp), 10.0 ** (rsrp / 10.0), 0.0)
    noise_shape = (-1,) + (1,) * (rsrp.ndim - 1)
    noise = 10.0 ** (np.asarray(noise_dbm, dtype=float).reshape(noise_shape) / 10.0)
    interference = power.sum(axis=1, keepdims=True) - power
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(power / (interference + noise))


def spectral_efficiency(sinr: np.ndarray) -> np.ndarray:
    """Shannon spectral efficiency ``log2(1 + SINR)`` in bit/s/Hz."""
    return np.log2(1.0 + 10.0 ** (np.asarray(sinr, dtype=float) / 10.0))


def prb_bandwidth_hz(scs_hz: float | np.ndarray) -> np.ndarray:
    """Bandwidth of one PRB: 12 subcarriers of ``scs_hz``."""
    return SUBCARRIERS_PER_PRB * np.asarray(scs_hz, dtype=float)


def prb_rate_bps(sinr: np.ndarray, bandwidth_hz: float | np.ndarray) -> np.ndarray:
    """Throughput of one PRB, ``B_PRB * log2(1 + SINR)``, in bit/s."""
    return np.asarray(bandwidth_hz, dtype=float) * spectral_efficiency(sinr)


def required_throughput_bps(n_ue: float | np.ndarray, per_ue_bps: float) -> np.ndarray:
    """Throughput a tile requires: its UE count times the per-UE requirement."""
    return np.asarray(n_ue, dtype=float) * per_ue_bps


def prb_per_ue(per_ue_bps: float, rate_bps: float | np.ndarray) -> np.ndarray:
    """PRBs one UE needs at a given per-PRB rate; ``inf`` where the rate is zero."""
    with np.errstate(divide="ignore"):
        return np.divide(per_ue_bps, np.asarray(rate_bps, dtype=float))


def prb_required(n_ue: float | np.ndarray, per_ue_prb: float | np.ndarray) -> np.ndarray:
    """PRBs a tile requires: its UE count times the PRBs per UE."""
    return np.asarray(n_ue, dtype=float) * np.asarray(per_ue_prb, dtype=float)


def candidate_order(rsrp: np.ndarray, band_rank: np.ndarray, threshold_dbm: float) -> np.ndarray:
    """Cell-bands one location may be served by, most preferred first.

    Args:
        rsrp: ``[n_band, n_tx]`` RSRP in dBm at one location, NaN where no path.
        band_rank: Preference per band, 0 most preferred.
        threshold_dbm: The RSRP a layer needs to be taken on band preference.

    Returns:
        Flat indices into ``rsrp``: layers at or above the threshold by band
        preference then RSRP, followed by the rest by RSRP. No-path layers are
        left out. The first entry is the rule's choice before capacity.
    """
    flat = rsrp.ravel()
    band = np.repeat(np.arange(rsrp.shape[0]), rsrp.shape[1])
    heard = np.isfinite(flat)
    above = heard & (flat >= threshold_dbm)
    strength = np.where(heard, -flat, np.inf)
    # np.lexsort sorts by the last key first.
    order = np.lexsort((strength, np.where(above, band_rank[band], 0), ~above))
    return order[heard[order]]


def select_serving(rsrp: np.ndarray, sinr: np.ndarray, spec: CapacitySpec) -> Serving:
    """Assign one interval's UEs to cell-bands under the PRB limits.

    UEs are taken in the order given; each walks its :func:`candidate_order`
    and takes the first cell-band with room for it.

    Args:
        rsrp: ``[n_ue, n_band, n_tx]`` RSRP at each UE's location.
        sinr: ``[n_ue, n_band, n_tx]`` SINR in dB at the same locations.
        spec: The capacity settings.
    """
    n_ue, n_band, n_tx = rsrp.shape
    bandwidth = prb_bandwidth_hz(spec.scs_hz)[None, :, None]
    need = prb_per_ue(spec.throughput_per_ue_bps, prb_rate_bps(sinr, bandwidth))
    limit = np.repeat(spec.n_prb, n_tx)

    band = np.full(n_ue, -1)
    tx = np.full(n_ue, -1)
    per_ue = np.full(n_ue, np.nan)
    load = np.zeros(n_band * n_tx)
    # ponytail: Python loop over UEs; vectorise if this enters the search loop.
    for ue in range(n_ue):
        ue_need = need[ue].ravel()
        order = candidate_order(rsrp[ue], spec.band_rank, spec.rsrp_threshold_dbm)
        order = order[np.isfinite(ue_need[order])]
        if order.size == 0:
            continue
        fits = order[load[order] + ue_need[order] <= limit[order]]
        if fits.size == 0:
            per_ue[ue] = ue_need[order[0]]
            continue
        chosen = fits[0]
        load[chosen] += ue_need[chosen]
        per_ue[ue] = ue_need[chosen]
        band[ue], tx[ue] = divmod(int(chosen), n_tx)
    return Serving(band=band, tx=tx, prb_per_ue=per_ue, load=load.reshape(n_band, n_tx))


def serve_intervals(
    rsrp: np.ndarray, band_labels: Sequence[str], mdt: pd.DataFrame, cfg: DictConfig
) -> pd.DataFrame:
    """Run :func:`select_serving` for every interval of the MDT.

    Args:
        rsrp: Radio map in dBm, ``[n_band, n_tx, n_rows, n_cols]``. RSRP and
            SINR are read at each UE's tile, so the result follows the tilts.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        mdt: Supplies ``t_index``, ``tile_row`` and ``tile_col`` only.
        cfg: Composed config; reads ``kpi.capacity`` and ``kpi.band_priority``.

    Returns:
        One row per MDT row, index aligned: ``t_index``, ``tile_row``,
        ``tile_col``, ``band``, ``tx``, ``sinr_db`` (at the serving or first
        choice), ``prb_per_ue``.

    Raises:
        ValueError: When the MDT is off the map's grid or a band has no
            capacity entry.
    """
    spec = CapacitySpec.from_config(cfg, band_labels)
    row, col = tile_index(mdt, rsrp.shape[-2:])
    sinr = sinr_db(rsrp, spec.noise_dbm)
    t_index = mdt["t_index"].to_numpy()

    out = pd.DataFrame({"t_index": t_index, "tile_row": row, "tile_col": col}, index=mdt.index)
    band = np.full(len(mdt), -1)
    tx = np.full(len(mdt), -1)
    per_ue = np.full(len(mdt), np.nan)
    for value in np.unique(t_index):
        at = np.flatnonzero(t_index == value)
        r, c = row[at], col[at]
        serving = select_serving(
            rsrp[:, :, r, c].transpose(2, 0, 1), sinr[:, :, r, c].transpose(2, 0, 1), spec
        )
        band[at], tx[at], per_ue[at] = serving.band, serving.tx, serving.prb_per_ue
    out["band"], out["tx"], out["prb_per_ue"] = band, tx, per_ue
    served = band >= 0
    out["sinr_db"] = np.nan
    out.loc[served, "sinr_db"] = sinr[band[served], tx[served], row[served], col[served]]
    return out


def demand_prb(
    rsrp: np.ndarray, band_labels: Sequence[str], mdt: pd.DataFrame, cfg: DictConfig
) -> np.ndarray:
    """PRBs required per tile in its busiest interval, ``[n_rows, n_cols]``.

    Per interval, a tile needs the sum of its UEs' PRBs - its UE count times
    PRBs per UE when all share a serving cell-band. A blocked UE still counts at
    its first choice: this is demand, not what was served. The raster keeps the
    maximum over intervals.

    Raises:
        ValueError: As :func:`serve_intervals`.
    """
    n_rows, n_cols = rsrp.shape[-2:]
    served = serve_intervals(rsrp, band_labels, mdt, cfg)
    flat = served["tile_row"].to_numpy() * n_cols + served["tile_col"].to_numpy()
    weight = np.nan_to_num(served["prb_per_ue"].to_numpy())
    t_index = served["t_index"].to_numpy()
    peak = np.zeros(n_rows * n_cols)
    for value in np.unique(t_index):
        at = t_index == value
        per_tile = np.bincount(flat[at], weights=weight[at], minlength=n_rows * n_cols)
        peak = np.maximum(peak, per_tile)
    return peak.reshape(n_rows, n_cols)
