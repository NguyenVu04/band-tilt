"""The KPI vector, its sign convention, and the objective that picks one winner.

The KPI definitions live in :mod:`src.kpi` and are not restated here. What this
module adds is what an optimizer needs around them: one value object carrying
every measurement, the orientation that turns them into "larger is better", and
the objective (docs/adr/0009-effective-coverage-objective.md):

    J = sum_g w_g * lambda_g * exp(1 - lambda_g) / sum_g w_g

``lambda_g`` is :func:`src.kpi.overlap.serving_multiplicity`: how many cells
effectively serve tile ``g``, counted on the band that would serve it.
``lambda * exp(1 - lambda)`` peaks at exactly 1 where one cell dominates, falls
to 0.74 where a second sits within the overlap margin, and is 0 where nothing
covers the tile. So J is the demand-weighted share of the grid served cleanly by
one cell, bounded in ``[0, 1]``.

``w_g = 1 + r_g`` weights each tile by demand, ``r_g`` being its MDT reports
over the busiest tile's (:mod:`src.data.demand`). Ground the MDT never saw still
weighs 1, so a hole out there is scored, just not doubly.

The objective has no parameters of its own: it reads ``kpi.hole_dbm``,
``kpi.overlap_margin_db`` and ``kpi.capacity.band_preference``, each of which
the KPIs or the serving rule already define.

:data:`KPI_NAMES` are reported and no selection reads them; ``objective`` is
stored beside them, so a history is ranked without re-reading a radio map.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data import demand
from src.kpi.capacity import CapacitySpec, serve_intervals
from src.kpi.hole import hole_rate
from src.kpi.load import load_imbalance, prb_by_cell_interval, prb_utilisation_max
from src.kpi.overlap import overlap_neighbor_mean, overlap_rate, serving_multiplicity
from src.kpi.quality import (
    LOW_PERCENTILE,
    MEDIAN_PERCENTILE,
    rsrp_percentile_dbm,
    sinr_percentile_db,
)
from src.kpi.served import served_rate
from src.kpi.weak import weak_rate

# What a deployment reads, in ADR 0007's reporting order: where coverage fails,
# how crowded it is, how strong it is, whether the traffic got served, and how
# the load sits.
KPI_NAMES = (
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

# Everything measured per candidate: the column order of every table.
MEASURE_NAMES = (*KPI_NAMES, "objective")

# The measures where larger is better. Named once, so no call site re-decides a
# sign; the rest are minimised.
MAXIMISED = frozenset(
    {
        "served_rate",
        "rsrp_p05_dbm",
        "rsrp_p50_dbm",
        "sinr_p05_db",
        "sinr_p50_db",
        "objective",
    }
)


@dataclass(frozen=True)
class KpiVector:
    """One configuration's measurement: the KPIs, then the objective.

    Attributes:
        hole_rate: Share of the grid receiving nothing above ``kpi.hole_dbm``,
            counting locations the ray tracer found no path to at all.
        overlap_rate: Share of the grid with at least one overlapping neighbour.
        overlap_neighbor_mean: Overlapping co-band neighbours per covered tile,
            the severity behind that rate's incidence.
        weak_rate: Share of the grid covered but below ``kpi.weak_dbm``.
        rsrp_p05_dbm: Cell-edge serving RSRP over covered locations only, so it
            is read beside ``hole_rate``.
        rsrp_p50_dbm: Median serving RSRP over the same locations.
        sinr_p05_db: Cell-edge best-server SINR over the same locations.
        sinr_p50_db: Median best-server SINR over the same locations.
        served_rate: Share of UE reports admitted to a cell-band.
        prb_utilisation_max: Highest load any cell-band reaches in any interval,
            as a share of its limit.
        load_imbalance: Coefficient of variation of cell-band utilisation.
        objective: See :func:`objective`. In ``[0, 1]``.
    """

    hole_rate: float
    overlap_rate: float
    overlap_neighbor_mean: float
    weak_rate: float
    rsrp_p05_dbm: float
    rsrp_p50_dbm: float
    sinr_p05_db: float
    sinr_p50_db: float
    served_rate: float
    prb_utilisation_max: float
    load_imbalance: float
    objective: float

    def as_dict(self) -> dict[str, float]:
        """The values keyed by name, as ``run.json`` records them."""
        return {name: float(value) for name, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, values: dict[str, float]) -> KpiVector:
        """Build from a name-keyed mapping.

        Raises:
            KeyError: When a measure is absent, which means a trial was
                completed with an incomplete measurement, or the record predates
                the current measure set.
        """
        missing = [name for name in MEASURE_NAMES if name not in values]
        if missing:
            raise KeyError(f"no value for {', '.join(missing)}")
        return cls(**{name: float(values[name]) for name in MEASURE_NAMES})


def tile_share(cfg: DictConfig, shape: tuple[int, int]) -> np.ndarray:
    """The demand map's tile shares, checked against the radio map's grid.

    Side effect: reads ``data.output.demand_file``; see
    :func:`src.data.demand.load_share`.

    Raises:
        FileNotFoundError: When the demand map has not been built.
        ValueError: When it was built on another grid, which would silently
            weight the wrong tiles.
    """
    share = demand.load_share(cfg)
    if share.shape != tuple(shape):
        raise ValueError(
            f"the demand map is {share.shape[0]} x {share.shape[1]} but the radio map is "
            f"{shape[0]} x {shape[1]}. The two were built on different grids; re-run "
            "`task preprocess`."
        )
    return share


def objective(
    rsrp: np.ndarray,
    band_labels: Sequence[str],
    cfg: DictConfig,
    share: np.ndarray | None = None,
) -> float:
    """Demand-weighted share of the grid served cleanly by exactly one cell.

    ``sum_g w_g * lambda_g * exp(1 - lambda_g) / sum_g w_g``, over every tile of
    the grid. ``lambda_g`` is :func:`src.kpi.overlap.serving_multiplicity` and
    ``w_g = 1 + r_g``, with ``r_g`` the tile's demand relative to the busiest
    tile's.

    A tile scores its full 1 only when one cell clears ``kpi.hole_dbm`` on the
    band that would serve it and nothing else on that band comes within
    ``kpi.overlap_margin_db``. A second cell inside the margin costs it a
    quarter, a third nearly two thirds. A hole scores 0, and so does a tile the
    ray tracer found no path to.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``.
        band_labels: Band names aligned to axis 0 of ``rsrp``; they carry
            ``kpi.capacity.band_preference`` onto the array.
        cfg: Composed config; see :func:`src.kpi.overlap.serving_multiplicity`.
        share: Tile demand, ``[n_rows, n_cols]``. Read from
            ``data.output.demand_file`` when None, which is what every caller
            but a test does. Only each tile's ratio to the largest is read, so
            any positive scaling of one map gives one J.

    Returns:
        A value in ``[0, 1]``, reaching 1 only if every tile of the grid is
        served by exactly one cell. Maximised.

    Raises:
        FileNotFoundError: When ``share`` is None and the demand map has not
            been built.
        ValueError: When ``band_labels`` does not match axis 0 of ``rsrp``, the
            demand does not cover the radio map's grid, or it is zero
            everywhere, leaving nothing to weight by.
    """
    multiplicity = serving_multiplicity(rsrp, cfg, band_labels)
    grid = rsrp.shape[-2:]
    p = tile_share(cfg, grid) if share is None else np.asarray(share, float)
    if p.shape != grid:
        raise ValueError(
            f"the tile shares are {p.shape} for a {grid} grid; they must cover it exactly."
        )
    peak = float(p.max())
    if not peak > 0:
        raise ValueError("the tile demand is zero everywhere, so there is nothing to weight by.")
    weight = 1.0 + p / peak
    utility = multiplicity * np.exp(1.0 - multiplicity)
    return float((weight * utility).sum() / weight.sum())


def evaluate_kpis(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    band_labels: Sequence[str],
    ue: pd.DataFrame,
    cfg: DictConfig,
    share: np.ndarray | None = None,
) -> KpiVector:
    """Measure one radio map on every KPI and the objective.

    The UE table is served once, here, and the served rate and both load
    measures read that one assignment: serving is the expensive half of a
    measurement, and running it three times would treble what a search pays per
    candidate on top of the ray tracing.

    Args:
        rsrp: RSRP in dBm, shape ``[n_band, n_tx, n_rows, n_cols]``, NaN where
            no path was found.
        sinr: The solver's SINR in dB, same shape as ``rsrp``.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        ue: The UE table; ``t_index``, ``t_s``, ``tile_row`` and ``tile_col``
            place the UEs the served rate counts.
        cfg: Composed config; the measures read ``cfg.kpi``.
        share: Passed to :func:`objective`.

    Raises:
        ValueError: When ``band_labels`` does not match axis 0 of ``rsrp``, or
            as :func:`src.kpi.capacity.serve_intervals` and :func:`objective`.
    """
    if len(band_labels) != rsrp.shape[0]:
        raise ValueError(
            f"{len(band_labels)} band labels for a radio map with {rsrp.shape[0]} bands."
        )
    spec = CapacitySpec.from_config(cfg, band_labels, rsrp.shape[1])
    served = serve_intervals(rsrp, sinr, band_labels, ue, cfg)
    _t_values, prb = prb_by_cell_interval(served, rsrp.shape[0], rsrp.shape[1])

    return KpiVector(
        hole_rate=hole_rate(rsrp, cfg),
        overlap_rate=overlap_rate(rsrp, cfg),
        overlap_neighbor_mean=overlap_neighbor_mean(rsrp, cfg),
        weak_rate=weak_rate(rsrp, cfg),
        rsrp_p05_dbm=rsrp_percentile_dbm(rsrp, cfg, LOW_PERCENTILE),
        rsrp_p50_dbm=rsrp_percentile_dbm(rsrp, cfg, MEDIAN_PERCENTILE),
        sinr_p05_db=sinr_percentile_db(rsrp, sinr, cfg, LOW_PERCENTILE),
        sinr_p50_db=sinr_percentile_db(rsrp, sinr, cfg, MEDIAN_PERCENTILE),
        served_rate=served_rate(served),
        prb_utilisation_max=prb_utilisation_max(prb, spec.max_prb),
        load_imbalance=load_imbalance(prb, spec.max_prb),
        objective=objective(rsrp, band_labels, cfg, share),
    )


def best_by_objective(kpis: Sequence[KpiVector]) -> int:
    """Index of the highest ``objective``.

    A tie resolves to the earlier index, so the incumbent holds unless a
    candidate actually scores higher.

    Raises:
        ValueError: When ``kpis`` is empty.
    """
    if not kpis:
        raise ValueError("no candidates to choose from")
    return int(np.argmax([kpi.objective for kpi in kpis]))
