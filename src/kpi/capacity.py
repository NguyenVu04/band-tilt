"""Serving-cell choice and PRB demand.

The one serving rule in the project, read by the MDT stage, the Band Priority
Score and the demand map: prefer bands in ``kpi.band_priority`` order while the
band's strongest cell clears ``kpi.capacity.rsrp_threshold_dbm``, else take the
strongest cell-band; a cell-band out of PRBs (``max_prb`` on the cell) passes
the UE to the next candidate in that same ranking.

:func:`sinr_db` is the project's only definition of SINR, derived from RSRP
alone: every co-band transmitter fully loaded, plus ``k * T * B`` over the band
bandwidth. No map or report stores SINR, so a ray-traced map and a synthetic
report score by the same rule. PRBs are kept
fractional: an average over an interval, and smooth in tilt.

:func:`max_rsrp`, the strongest layer at each location, also lives here: the
hole and weak rates and the Band Priority Score's hole gate read it. It is not
the serving rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.constants import Boltzmann

from src.core.cell import Cell

# TS 38.211 4.4.4.1: one resource block is 12 consecutive subcarriers.
_SUBCARRIERS_PER_PRB = 12


@dataclass(frozen=True)
class CapacitySpec:
    """The serving and PRB settings, aligned to the radio map's bands and cells.

    Attributes:
        band_rank: Preference per band, 0 most preferred, shape ``[n_band]``.
        rsrp_threshold_dbm: Below this a band is skipped for the next one.
        throughput_per_ue_bps: Assumed throughput each UE requires.
        scs_hz: Subcarrier spacing per band, shape ``[n_band]``.
        max_prb: PRB limit per cell-band, shape ``[n_band, n_tx]``.
        noise_dbm: Thermal noise ``k * T * B`` per band, shape ``[n_band]``.
    """

    band_rank: np.ndarray
    rsrp_threshold_dbm: float
    throughput_per_ue_bps: float
    scs_hz: np.ndarray
    max_prb: np.ndarray
    noise_dbm: np.ndarray

    @classmethod
    def from_config(cls, cfg: DictConfig, band_labels: Sequence[str], n_tx: int) -> CapacitySpec:
        """Read ``kpi.capacity``, ``kpi.band_priority`` and the simulation's cells and bands.

        The cells are taken in config order, which is the radio map's tx axis:
        :func:`src.simulation.radio.solve` writes them in that order and
        preprocessing checks ``tx_name`` against the config.

        Raises:
            ValueError: When a band has no priority weight, capacity entry or
                bandwidth, or the config holds other than ``n_tx`` cells.
            KeyError: When a cell has no ``max_prb`` for a band.
        """
        capacity = cfg.kpi.capacity
        radio_map = cfg.simulation.radio_map
        bandwidth = {str(entry.name): float(entry.bandwidth) for entry in radio_map.bands}
        missing = [
            label
            for label in band_labels
            if label not in cfg.kpi.band_priority
            or label not in capacity.bands
            or label not in bandwidth
        ]
        if missing:
            raise ValueError(
                "No kpi.band_priority, kpi.capacity.bands or simulation.radio_map.bands entry "
                f"for {', '.join(missing)}."
            )
        cells = [Cell.from_config(entry) for entry in cfg.simulation.transmitters.cells]
        if len(cells) != n_tx:
            raise ValueError(
                f"simulation.transmitters.cells holds {len(cells)} cells for a radio map with "
                f"{n_tx} transmitters."
            )
        weight = np.array([float(cfg.kpi.band_priority[label]) for label in band_labels])
        return cls(
            band_rank=np.argsort(np.argsort(-weight, kind="stable"), kind="stable"),
            rsrp_threshold_dbm=float(capacity.rsrp_threshold_dbm),
            throughput_per_ue_bps=float(capacity.throughput_per_ue_bps),
            scs_hz=np.array([float(capacity.bands[label].scs_hz) for label in band_labels]),
            max_prb=np.array(
                [[float(cell.max_prb_for(label)) for cell in cells] for label in band_labels]
            ),
            noise_dbm=_thermal_noise_dbm(
                float(radio_map.temperature), np.array([bandwidth[label] for label in band_labels])
            ),
        )


@dataclass(frozen=True)
class _Serving:
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


def _tile_index(mdt: pd.DataFrame, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The MDT's ``(row, col)`` tile indices, checked against the map's grid.

    Args:
        mdt: Synthetic MDT, one row per UE per interval, carrying ``tile_row``
            and ``tile_col``.
        shape: The radio map's ``(n_rows, n_cols)``.

    Returns:
        ``(row, col)``, each of shape ``[n_report]``, ready to index a
        ``[n_rows, n_cols]`` raster.

    Raises:
        ValueError: When a UE falls outside the map's grid, which means the MDT
            and the radio map were built on different grids.
    """
    n_rows, n_cols = shape
    row = mdt["tile_row"].to_numpy()
    col = mdt["tile_col"].to_numpy()
    if row.min() < 0 or row.max() >= n_rows or col.min() < 0 or col.max() >= n_cols:
        raise ValueError(
            f"MDT tiles span rows {row.min()}..{row.max()} cols {col.min()}..{col.max()}, "
            f"outside the radio map's {n_rows} x {n_cols} grid. The two were built on "
            "different grids."
        )
    return row, col


def _thermal_noise_dbm(temperature_k: float, bandwidth_hz: float | np.ndarray) -> np.ndarray:
    """Thermal noise ``k * T * B`` in dBm, as ``sionna.rt.Scene.thermal_noise_power``."""
    power_w = Boltzmann * temperature_k * np.asarray(bandwidth_hz, dtype=float)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(power_w) + 30.0


def sinr_db(rsrp: np.ndarray, noise_dbm: float | np.ndarray) -> np.ndarray:
    """Co-band SINR of every cell-band layer, with full-load interference.

    Args:
        rsrp: RSRP in dBm, ``[n_band, n_tx, ...]``, NaN where no path.
        noise_dbm: Noise power, scalar or one per band.

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


def _spectral_efficiency(sinr: np.ndarray) -> np.ndarray:
    """Shannon spectral efficiency ``log2(1 + SINR)`` in bit/s/Hz."""
    return np.log2(1.0 + 10.0 ** (np.asarray(sinr, dtype=float) / 10.0))


def _prb_bandwidth_hz(scs_hz: float | np.ndarray) -> np.ndarray:
    """Bandwidth of one PRB: 12 subcarriers of ``scs_hz``."""
    return _SUBCARRIERS_PER_PRB * np.asarray(scs_hz, dtype=float)


def _prb_rate_bps(sinr: np.ndarray, bandwidth_hz: float | np.ndarray) -> np.ndarray:
    """Throughput of one PRB, ``B_PRB * log2(1 + SINR)``, in bit/s."""
    return np.asarray(bandwidth_hz, dtype=float) * _spectral_efficiency(sinr)


def _required_throughput_bps(n_ue: float | np.ndarray, per_ue_bps: float) -> np.ndarray:
    """Throughput a tile requires: its UE count times the per-UE requirement."""
    return np.asarray(n_ue, dtype=float) * per_ue_bps


def _prb_per_ue(per_ue_bps: float, rate_bps: float | np.ndarray) -> np.ndarray:
    """PRBs one UE needs at a given per-PRB rate; ``inf`` where the rate is zero."""
    with np.errstate(divide="ignore"):
        return np.divide(per_ue_bps, np.asarray(rate_bps, dtype=float))


def _prb_required(n_ue: float | np.ndarray, per_ue_prb: float | np.ndarray) -> np.ndarray:
    """PRBs a tile requires: its UE count times the PRBs per UE."""
    return np.asarray(n_ue, dtype=float) * np.asarray(per_ue_prb, dtype=float)


def _candidate_order(rsrp: np.ndarray, band_rank: np.ndarray, threshold_dbm: float) -> np.ndarray:
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


def _select_serving(rsrp: np.ndarray, sinr: np.ndarray, spec: CapacitySpec) -> _Serving:
    """Assign one interval's UEs to cell-bands under the PRB limits.

    UEs are taken in the order given; each walks its :func:`_candidate_order`
    and takes the first cell-band with room for it.

    Args:
        rsrp: ``[n_ue, n_band, n_tx]`` RSRP at each UE's location.
        sinr: ``[n_ue, n_band, n_tx]`` SINR in dB at the same locations.
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
    # Python loop over UEs; vectorise if this enters the search loop.
    for ue in range(n_ue):
        ue_need = need[ue].ravel()
        order = _candidate_order(rsrp[ue], spec.band_rank, spec.rsrp_threshold_dbm)
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
    return _Serving(band=band, tx=tx, prb_per_ue=per_ue, load=load.reshape(n_band, n_tx))


def serve_rows(
    rsrp: np.ndarray, sinr: np.ndarray, t_index: np.ndarray, spec: CapacitySpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run :func:`_select_serving` once per interval.

    Args:
        rsrp: ``[n_ue, n_band, n_tx]`` RSRP each UE sees, clean or reported.
        sinr: ``[n_ue, n_band, n_tx]`` SINR in dB at the same UEs.
        t_index: Interval of each UE; UEs compete for PRBs only within one.
        spec: The capacity settings.

    Returns:
        ``(band, tx, prb_per_ue)`` per UE, as :class:`_Serving` holds them.
    """
    band = np.full(len(t_index), -1)
    tx = np.full(len(t_index), -1)
    per_ue = np.full(len(t_index), np.nan)
    for value in np.unique(t_index):
        at = np.flatnonzero(t_index == value)
        serving = _select_serving(rsrp[at], sinr[at], spec)
        band[at], tx[at], per_ue[at] = serving.band, serving.tx, serving.prb_per_ue
    return band, tx, per_ue


def serve_intervals(
    rsrp: np.ndarray, band_labels: Sequence[str], mdt: pd.DataFrame, cfg: DictConfig
) -> pd.DataFrame:
    """Serve every MDT row from the radio map, via :func:`serve_rows`.

    Args:
        rsrp: Radio map in dBm, ``[n_band, n_tx, n_rows, n_cols]``. RSRP is
            read at each UE's tile and SINR recomputed from it, so the result
            follows the tilts.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        mdt: Supplies ``t_index``, ``tile_row`` and ``tile_col`` only.
        cfg: Composed config; see :meth:`CapacitySpec.from_config`.

    Returns:
        One row per MDT row, index aligned: ``t_index``, ``tile_row``,
        ``tile_col``, ``band``, ``tx``, ``sinr_db`` (at the serving cell-band,
        NaN when blocked), ``prb_per_ue``.

    Raises:
        ValueError: When the MDT is off the map's grid or the config does not
            cover the map's bands and cells.
    """
    spec = CapacitySpec.from_config(cfg, band_labels, rsrp.shape[1])
    row, col = _tile_index(mdt, rsrp.shape[-2:])
    sinr = sinr_db(rsrp, spec.noise_dbm)
    t_index = mdt["t_index"].to_numpy()

    band, tx, per_ue = serve_rows(
        rsrp[:, :, row, col].transpose(2, 0, 1),
        sinr[:, :, row, col].transpose(2, 0, 1),
        t_index,
        spec,
    )
    out = pd.DataFrame({"t_index": t_index, "tile_row": row, "tile_col": col}, index=mdt.index)
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
    flat = t_pos * size + row * n_cols + col
    prb = np.bincount(flat, weights=np.nan_to_num(prb_per_ue), minlength=len(t_values) * size)
    return t_values, prb.reshape(len(t_values), n_rows, n_cols)


def demand_prb(
    rsrp: np.ndarray, band_labels: Sequence[str], mdt: pd.DataFrame, cfg: DictConfig
) -> np.ndarray:
    """PRBs required per tile in its busiest interval, ``[n_rows, n_cols]``.

    A blocked UE still counts at its first choice: this is demand, not what
    was served.

    Raises:
        ValueError: As :func:`serve_intervals`.
    """
    served = serve_intervals(rsrp, band_labels, mdt, cfg)
    _, prb = prb_by_interval(
        served["t_index"].to_numpy(),
        served["tile_row"].to_numpy(),
        served["tile_col"].to_numpy(),
        served["prb_per_ue"].to_numpy(),
        rsrp.shape[-2:],
    )
    return prb.max(axis=0)
