"""Select the MDT: the UEs the serving rule admits at the committed tilts.

MDT is a subset of the UE population, not a second draw. A UE enters it when
:func:`src.kpi.capacity.serve_intervals` admits it on the baseline radio map,
because only a served UE reports measurements. The search scores the load term
and the served ratio on MDT; evaluation scores every measure on all UEs.

``rsrp_dbm`` is the serving cell-band's RSRP at the UE's tile. It is kept for
reference and plots; no score reads it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.kpi.capacity import serve_intervals
from src.simulation.sample import CSV_COLUMNS
from src.tracking import log_stage

MDT_COLUMNS = (*CSV_COLUMNS, "rsrp_dbm")


def select(
    rsrp: np.ndarray,
    sinr: np.ndarray,
    band_labels: Sequence[str],
    ue: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    """The served UE rows, with the RSRP of the cell-band serving each.

    Args:
        rsrp: Radio map in dBm, ``[n_band, n_tx, n_rows, n_cols]``.
        sinr: The solver's SINR in dB, same shape.
        band_labels: Band names aligned to axis 0 of ``rsrp``.
        ue: The UE table with :data:`src.simulation.sample.CSV_COLUMNS`.
        cfg: Composed config; see :meth:`src.kpi.capacity.CapacitySpec.from_config`.

    Returns:
        :data:`MDT_COLUMNS`, one row per admitted UE, in the UE table's order.

    Raises:
        ValueError: As :func:`src.kpi.capacity.serve_intervals`.
    """
    served = serve_intervals(rsrp, sinr, band_labels, ue, cfg)
    band = served["band"].to_numpy()
    on = band >= 0
    mdt = ue.loc[on, list(CSV_COLUMNS)].reset_index(drop=True)
    mdt["rsrp_dbm"] = rsrp[
        band[on],
        served["tx"].to_numpy()[on],
        served["tile_row"].to_numpy()[on],
        served["tile_col"].to_numpy()[on],
    ]
    return mdt


def build(cfg: DictConfig) -> Path:
    """Serve the UE table on the baseline radio map and write the MDT. Returns its path.

    Reads ``simulation.output.ue_file`` and ``radio_map_file``; writes
    ``simulation.output.mdt_file``.

    Raises:
        FileNotFoundError: When the scenario or radio stage has not been run.
        ValueError: As :func:`select`.
    """
    output = cfg.simulation.output
    ue_path, map_path = Path(output.ue_file), Path(output.radio_map_file)
    for path, stage in ((ue_path, "scenario"), (map_path, "radio")):
        if not path.is_file():
            raise FileNotFoundError(f"No {path}. Run `task simulation:{stage}` first.")

    ue = pd.read_csv(ue_path)
    with np.load(map_path, allow_pickle=False) as archive:
        rsrp = archive["rsrp_dbm"].astype(float)
        sinr = archive["sinr_db"].astype(float)
        band_labels = [str(label) for label in archive["band_label"]]

    mdt = select(rsrp, sinr, band_labels, ue, cfg)
    path = Path(output.mdt_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Millimetre precision, as the UE table, so MDT rows match UE rows exactly.
    mdt.to_csv(path, index=False, float_format="%.3f")

    print(f"ue:  {len(ue)} rows")
    print(f"mdt: {len(mdt)} rows served ({len(mdt) / len(ue):.1%})  ->  {path}")
    return path


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build the MDT. Entry point for ``task simulation:mdt``."""
    log_stage(cfg, "simulation_mdt", groups=["simulation", "kpi"], outputs=[build(cfg)])


if __name__ == "__main__":
    main()
