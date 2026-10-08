"""Live what-if: move antenna tilts, ray-trace them, and read the KPIs against today's.

Run with ``task demo``. Needs a CUDA GPU, the ``rt`` and ``demo`` extras, and the
outputs of ``task simulation`` and ``task preprocess``. Published solutions under
``optim.output.deliverable_dir`` are offered as starting points when present.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import load_config
from src.core.sector import read_sectors, site_frame
from src.evaluation import compare, plots
from src.evaluation import runs as run_store
from src.evaluation.export import readable
from src.kpi.capacity import max_rsrp
from src.optim.evaluator import Evaluator
from src.optim.history import History
from src.optim.objective import OBJECTIVE_NAMES, KpiVector
from src.utils.plotting import label

ROOT = Path(__file__).resolve().parents[1]
CURRENT = "Current network"


@st.cache_resource(show_spinner="Loading the scene and the UE table...")
def evaluator() -> tuple[Evaluator, threading.Lock]:
    """One evaluator for every session; the lock keeps one ray trace on the GPU at a time."""
    return Evaluator(load_config(root=ROOT), keep_rsrp=True), threading.Lock()


@st.cache_resource
def baseline_map() -> dict[str, np.ndarray]:
    """The archived baseline radio map, for the grid extent of every plot."""
    return run_store.baseline_map(evaluator()[0].cfg)


@st.cache_data(max_entries=32, show_spinner="Ray-tracing every band...")
def solve(tilt: tuple[float, ...]) -> tuple[KpiVector, np.ndarray, float]:
    """KPIs, best-server RSRP map and ray-tracing seconds for one tilt vector."""
    model, lock = evaluator()
    with lock:
        result = model.evaluate(np.array(tilt))
    return result.kpi, max_rsrp(result.rsrp), result.seconds


def starting_points(cfg, space) -> dict[str, np.ndarray]:
    """The incumbent, then every published solution of every method."""
    points = {CURRENT: space.baseline}
    index = pd.MultiIndex.from_tuples(space.pairs, names=["sector", "band"])
    for path in sorted(Path(ROOT, cfg.optim.output.deliverable_dir).glob("tilt_options_*.csv")):
        method = path.stem.removeprefix("tilt_options_")
        options = pd.read_csv(path)
        for solution, group in options.groupby("solution"):
            tilt = group.set_index(["sector", "band"])["optimized_tilt_deg"].reindex(index)
            tag = " (recommended)" if group["recommended"].any() else ""
            points[f"{label(method)} solution {solution}{tag}"] = tilt.to_numpy(dtype=float)
    return points


st.set_page_config(page_title="Band-tilt what-if", layout="wide")
st.title("Multi-band tilt what-if")

model, _lock = evaluator()
cfg, space = model.cfg, model.space
points = starting_points(cfg, space)

with st.sidebar:
    start_name = st.selectbox("Start from", list(points))
    st.caption("Shift every sector on a band by the same amount, then fine-tune per sector below.")
    shifts = {
        band: st.slider(f"{label(band)} shift [deg]", -15.0, 15.0, 0.0, 0.25)
        for band in space.band_names
    }

start = points[start_name]
shift = np.array([shifts[band] for _sector, band in space.pairs])
proposal = space.clip(start + shift)

grid = pd.DataFrame(
    proposal.reshape(len(space.sectors), len(space.band_names)),
    index=[sector.name for sector in space.sectors],
    columns=list(space.band_names),
)
st.subheader("Tilt per sector and band [deg]")
st.caption(
    f"Bounds {space.lower.min():g} to {space.upper.max():g} deg; values outside a sector's own "
    "bounds are clipped. Moving a slider or the start point resets the edits."
)
edited = st.data_editor(grid, key=f"grid-{start_name}-{sorted(shifts.items())}")
proposal = space.clip(edited.to_numpy(dtype=float).reshape(-1))

if st.button("Ray-trace this configuration", type="primary"):
    st.session_state["evaluated"] = tuple(proposal.round(4))

if "evaluated" not in st.session_state:
    st.info("Set the tilts, then ray-trace. The current network is traced once for reference.")
    st.stop()

tilt = st.session_state["evaluated"]
before_kpi, before_rsrp, _ = solve(tuple(space.baseline.round(4)))
after_kpi, after_rsrp, seconds = solve(tilt)

st.subheader("Result")
*objective_columns, middle, right = st.columns(len(OBJECTIVE_NAMES) + 2)
for column, name in zip(objective_columns, OBJECTIVE_NAMES, strict=True):
    after, before = getattr(after_kpi, name), getattr(before_kpi, name)
    column.metric(f"{label(name)} (maximise)", f"{after:.4g}", f"{after - before:+.4g}")
moved = np.abs(np.array(tilt) - space.baseline) > 1e-9
middle.metric("Sector-bands moved", f"{int(moved.sum())} of {space.n_dim}")
right.metric("Ray tracing", f"{seconds:.1f} s")

table = compare.delta_table(before_kpi, after_kpi).rename(
    columns={"before": "Current", "after": "What-if", "delta": "Change"}
)
st.dataframe(readable(table), hide_index=True, width="stretch")

st.pyplot(
    plots.coverage_maps(
        before_rsrp,
        after_rsrp,
        baseline_map(),
        cfg,
        sectors=site_frame(read_sectors(cfg.simulation.input.sectors_file)),
        name="what-if",
    ),
    clear_figure=True,
)

if moved.any():
    change = History(space).tilt_table(np.array(tilt))
    st.subheader("Tilt change against the current network")
    st.dataframe(readable(change[moved]), hide_index=True, width="stretch")
