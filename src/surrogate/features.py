"""Turn scenario features and a tilt configuration into the surrogate input tensor.

The surrogate learns ``f_sur: (x, tilt) -> R_hat`` (PROJECT.md section 11).
``tilt`` is fixed by the problem — one absolute tilt per cell-band — but what
goes into ``x`` is an open choice, declared in ``cfg.surrogate.features``.

This is the only module allowed to fit on data
----------------------------------------------
Everything in :mod:`src.data` is deliberately model-agnostic: it applies hard
rules and never learns a threshold. Scalers, encoders and any other fitted
transform live here instead, and are fitted on the surrogate TRAIN partition
only. A scaler fitted on the full D_sur has seen the held-out configurations,
and the reported surrogate error is then optimistic for a reason that is hard to
find later.

Fit once, then transform
------------------------
The fitted state travels with the model artifact, not with the code. A
surrogate loaded in notebook 05a or 05b must transform new tilt configurations
exactly as it did during training — refitting at inference time silently changes
the input distribution and the predictions with it. The surrogate is frozen for
the whole optimization phase (PROJECT.md section 11.4), so this transform is
fixed from the moment it is accepted.

What is worth including
-----------------------
Tilt alone makes the surrogate memorise configurations rather than learn the
geometry, and it cannot generalise to a network whose cells have moved. Cell
geometry, band identity, UE density and the neighbour structure are all
candidates (PROJECT.md section 14.1 lists the analogous set for the MARL state).
Each is a config switch, so the ablation is a sweep rather than a rewrite.

Scene features are the ones the sim-to-reality study depends on. PROJECT.md
section 12 perturbs buildings, dimensions and materials between scenarios; a
surrogate whose input never describes the environment cannot generalise across
those perturbations, it can only memorise the scenario it trained on — and the
held-out-scenario error in ``cfg.surrogate.acceptance`` is then measuring
nothing.
"""

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def build_state(table: pd.DataFrame, rho: np.ndarray, cfg: DictConfig) -> dict:
    """Assemble the state s that does not vary with the tilt configuration.

    Args:
        table: The cell-band table from
            :func:`src.radio.cell_band.build_table`.
        rho: UE density per grid cell, built from the training split.
        cfg: Composed config; uses ``cfg.surrogate.features``.

    Returns:
        A mapping of the enabled state components.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Computed once and reused for every configuration in D_sur. Cell
        positions, azimuths, band identities and UE density are all constant
        across the dataset — recomputing them per row is pure waste on the
        largest loop in the project.

    Example:
        >>> state = build_state(table, rho, cfg)
    """
    # TODO(1): read cfg.surrogate.features to decide which components to include
    # TODO(2): cell_geometry -> sim_x, sim_y, antenna_height, azimuth per cell-band
    # TODO(3): band_identity -> band index and its priority weight
    # TODO(4): neighbor_graph -> which cells are near enough to overlap
    # TODO(5): ue_density -> rho, asserted aligned to the evaluation grid
    raise NotImplementedError("src.surrogate.features.build_state")


def fit(train_df: pd.DataFrame, state: dict, cfg: DictConfig) -> Any:
    """Fit any learned transform on the surrogate training partition only.

    Args:
        train_df: The train partition from :func:`src.surrogate.dataset.split`.
        state: The state mapping from :func:`build_state`.
        cfg: Composed config; uses ``cfg.surrogate.features``.

    Returns:
        The fitted transformer, to be saved with the model artifact.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Only the train partition. Passing the full D_sur here is the single most
        likely way to make the surrogate look better than it is.

    Example:
        >>> transformer = fit(train_df, state, cfg)
    """
    # TODO(1): assert train_df is the train partition, not the full dataset
    # TODO(2): fit scalers over the tilt columns and any continuous state
    # TODO(3): return the fitted object for saving alongside the model
    raise NotImplementedError("src.surrogate.features.fit")


def transform(tilt: np.ndarray, state: dict, transformer: Any) -> np.ndarray:
    """Build the surrogate input tensor for one or many tilt configurations.

    Args:
        tilt: Shape ``(n_cell_bands,)`` for one configuration or
            ``(n, n_cell_bands)`` for a batch, in degrees.
        state: The state mapping from :func:`build_state`.
        transformer: The fitted transformer from :func:`fit`.

    Returns:
        The feature tensor the model consumes.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, when ``tilt`` does not match the
            cell-band count the transformer was fitted with.

    Notes:
        Accept a single configuration and a batch through the same path. Both
        optimizers call this in their inner loop — BO on a handful of
        candidates, MARL on every environment step — and a per-configuration
        Python loop is the difference between an experiment that finishes and
        one that does not.

        Check the width. A tilt vector of the wrong length is the symptom of a
        band added to ``configs/radio.yaml`` after the surrogate was trained,
        and it must fail loudly rather than broadcast.

    Example:
        >>> x = transform(tilt, state, transformer)
    """
    # TODO(1): raise ValueError when tilt width disagrees with the fitted width
    # TODO(2): reshape a single configuration to a batch of one
    # TODO(3): concatenate the scaled tilt with the enabled state components
    raise NotImplementedError("src.surrogate.features.transform")
