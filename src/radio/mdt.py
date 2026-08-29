"""Synthetic MDT — evaluate RSRP along UE trajectories. PROJECT.md section 9.2.

.. note::
   **Blocked on the multi-band cell configuration.** ``src.radio.cell_band.
   build_table`` cannot form ``(cell, band)`` pairs without a band, carrier
   frequency or transmit power column, and none of those exist in
   ``data/external/gcell_conf.csv`` today. **Do not fabricate band data to
   make this module run** — see CLAUDE.md, *Known gaps*. Everything below is
   the intended shape, written so the gap is explicit rather than silent, the
   same way ``notebooks/01_mobility.ipynb`` was before
   :mod:`src.mobility` existed.

   ``src.mobility`` is unblocked and produces exactly the input this module
   needs: a trajectory frame with ``ue_id``, ``t``, ``sim_x``, ``sim_y``,
   ``ue_height`` in the Sionna-RT scene local frame (see
   :func:`src.mobility.simulate.tidy`). This module's signatures accept that
   frame directly.

Why per-position ray tracing, not a radio map
------------------------------------------------
PROJECT.md section 9.2 evaluates propagation *at each UE position and time*.
``src.radio.radiomap.compute_radiomap`` instead samples a fixed plane on the
``cfg.radio.grid`` lattice — cheap because it is solved once and reused for
every KPI evaluation, but a UE rarely sits exactly on a grid point. Reading
its RSRP off the nearest grid cell instead of ray-tracing its actual position
is precisely the substitution PROJECT.md section 25.4 names as a failure
mode: it silently replaces the measurement the spec asks for with computing
the wrong thing that happens to look similar. So this module uses
``sionna.rt.PathSolver`` directly, with one ``Receiver`` per UE sample, not
``RadioMapSolver``.

Cost
-----
The development GPU is a 4 GB laptop part (RTX 3050 Ti). ``PathSolver`` holds
the scene, its acceleration structure, and every traced path for every
attached ``Receiver`` in VRAM simultaneously — unlike a radio map, whose
memory cost is fixed by the grid regardless of how many UE samples it is
later read at. A trajectory set is on the order of 10^4-10^5 positions (see
``configs/mobility.yaml``'s ``sampling`` comment), so solving them all
attached to the scene at once will not fit. :func:`receiver_batches` is not
an optimisation to skip under time pressure; it is the constraint that makes
this module runnable at all on this hardware.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def receiver_batches(n_positions: int, cfg: DictConfig) -> Iterator[slice]:
    """Yield index slices small enough to hold as attached receivers at once.

    Args:
        n_positions: Total number of UE positions to evaluate — typically
            ``len(trajectories)``.
        cfg: Composed config; uses ``cfg.radio.mdt.receiver_batch`` (not yet
            declared in ``configs/radio.yaml`` — see the Notes below).

    Returns:
        Slices into the positions array, covering ``range(n_positions)``
        exactly once, in order.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Batch size is a VRAM measurement, not a guess: it must be set by
        running :func:`evaluate_positions` against this scene at increasing
        batch sizes until it stops fitting, then backing off. That is why
        ``cfg.radio.mdt.receiver_batch`` does not exist yet in
        ``configs/radio.yaml`` — a ``<placeholder>`` key nothing reads adds
        no safety, and a guessed literal would be exactly the kind of
        unmeasured number CLAUDE.md's *Never guess* rule forbids. Add the key,
        with a value obtained by measurement against the real scene, when
        this module is implemented.

        The last batch is short; do not pad it to a fixed size.

    Example:
        >>> list(receiver_batches(2500, cfg))  # doctest: +SKIP
        [slice(0, 512), slice(512, 1024), ..., slice(2048, 2500)]
    """
    # TODO(1): read the batch size from cfg.radio.mdt.receiver_batch — never a literal
    # TODO(2): yield slice objects, not copies, so the caller indexes `positions`
    # TODO(3): the last batch is short; do not pad it
    raise NotImplementedError("src.radio.mdt.receiver_batches")


def evaluate_positions(
    scene: Any,
    table: pd.DataFrame,
    positions: np.ndarray,
    tilt: np.ndarray,
    cfg: DictConfig,
) -> np.ndarray:
    """Ray-trace RSRP at a batch of UE positions, for every cell-band.

    Args:
        scene: The scene from :func:`src.radio.scene.load_scene`, with
            transmitters already attached
            (:func:`src.radio.scene.add_transmitters`) and oriented at
            ``tilt`` (:func:`src.radio.radiomap.set_tilt`) — called once,
            outside any batch loop, since orientation does not change between
            positions.
        table: The cell-band table from
            :func:`src.radio.cell_band.build_table`.
        positions: ``(n_positions, 3)`` — ``(sim_x, sim_y, ue_height)`` in the
            scene local frame, in the row order the result must preserve.
            :func:`src.mobility.simulate.tidy`'s ``sim_x``/``sim_y``/
            ``ue_height`` columns, stacked, is the intended input.
        tilt: The absolute tilt vector already applied to ``scene``, in
            ``table`` order. Not re-applied here; passed through only so the
            caller's intent is visible in the signature and in any error this
            function raises.
        cfg: Composed config; uses ``cfg.radio.mdt.receiver_batch`` (via
            :func:`receiver_batches`).

    Returns:
        ``(n_positions, len(table))`` dBm, in the same row order as
        ``positions`` and the same column order as ``table`` — matching the
        ``(n_cell_bands, |G|)`` axis convention of
        :func:`src.radio.radiomap.compute_radiomap`, transposed because here
        the positions are not a regular grid. ``-inf`` where no path reaches
        the receiver, never a sentinel number (same contract as
        :mod:`src.radio.radiomap`).

    Raises:
        NotImplementedError: Always — implement this module first.
        ImportError: Once implemented, when Sionna-RT is not installed — name
            ``uv sync --extra rt``.

    Notes:
        Batches over :func:`receiver_batches`; within a batch, attach one
        ``sionna.rt.Receiver`` per row of ``positions``, solve with
        ``sionna.rt.PathSolver`` (not ``RadioMapSolver`` — see the module
        docstring), then **remove the batch's receivers from the scene**
        before the next batch. Sionna keeps attached objects resident, and on
        4 GB of VRAM the batch size is the only knob that keeps a solve
        resident at all.

    Example:
        >>> rsrp = evaluate_positions(scene, table, positions, tilt, cfg)
        >>> rsrp.shape
        (2500, 26)
    """
    # TODO(1): import sionna.rt inside the function; on ImportError name `uv sync --extra rt`
    # TODO(2): BLOCKED — `table` cannot be built until the multi-band cell
    #          configuration arrives. Do not fabricate carrier_hz, tx_power_dbm
    #          or a band column to make this run (CLAUDE.md, Known gaps).
    # TODO(3): for each slice from receiver_batches(len(positions), cfg): attach
    #          one sionna.rt.Receiver per row, named from the row index so the
    #          result can be reassembled without a join
    # TODO(4): solve the batch with sionna.rt.PathSolver
    # TODO(5): sum path powers per (transmitter, receiver), add the band's
    #          tx_power_dbm, convert to dBm; write -inf where no path reached
    #          the receiver — matching the array contract in src.radio.radiomap
    # TODO(6): remove the batch's receivers from the scene before the next batch
    # TODO(7): reassemble batches in the input row order and return; never re-sort
    raise NotImplementedError("src.radio.mdt.evaluate_positions")


def build_mdt(
    trajectories: pd.DataFrame,
    scene: Any,
    table: pd.DataFrame,
    tilt: np.ndarray,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Assemble the synthetic MDT record — PROJECT.md section 9.3.

    Args:
        trajectories: UE positions to evaluate, in the scene local frame —
            :func:`src.mobility.simulate.tidy`'s output, typically after
            :func:`src.mobility.frame.clip_to_scene`. Carries ``scenario_id``,
            ``ue_id``, ``date``, ``sim_x``, ``sim_y``, ``ue_height``.
        scene: The scene from :func:`src.radio.scene.load_scene`, with
            transmitters attached and oriented at ``tilt``.
        table: The cell-band table from
            :func:`src.radio.cell_band.build_table`.
        tilt: The absolute tilt vector already applied to ``scene``.
        cfg: Composed config.

    Returns:
        One row per ``(trajectory sample, cell-band)`` pair above the
        reportable RSRP floor, with columns matching the section 9.3 record —
        ``ue_id``, ``sim_x``, ``sim_y``, ``date``, ``gcell_id``, ``band``,
        ``rsrp``, ``ue_height`` — plus ``scenario_id``, carried through from
        ``trajectories`` because ``src.data.split``'s scenario-level method
        needs it on every MDT record, not only on the trajectory it was
        derived from.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Calls :func:`evaluate_positions` once over the whole of
        ``trajectories`` (which internally batches — see
        :func:`receiver_batches`), then reshapes the resulting
        ``(n_positions, len(table))`` array into one row per non-``-inf``
        entry. A record is a *measurement*: a position is not a synthetic MDT
        row against a cell-band it cannot receive at all.

    Example:
        >>> mdt = build_mdt(trajectories, scene, table, tilt, cfg)
        >>> sorted(mdt.columns)
        ['band', 'date', 'gcell_id', 'rsrp', 'scenario_id', 'sim_x', 'sim_y',
         'ue_height', 'ue_id']
    """
    # TODO(1): BLOCKED — see evaluate_positions; do not fabricate band data
    # TODO(2): positions = trajectories[["sim_x", "sim_y", "ue_height"]].to_numpy()
    # TODO(3): rsrp = evaluate_positions(scene, table, positions, tilt, cfg)
    # TODO(4): melt to one row per (position, cell-band) with rsrp > -inf
    # TODO(5): join gcell_id/band from `table`, and ue_id/sim_x/sim_y/date/
    #          ue_height/scenario_id from `trajectories`, by position index
    # TODO(6): return columns matching configs/data.yaml schema.mdt, plus
    #          scenario_id
    raise NotImplementedError("src.radio.mdt.build_mdt")
