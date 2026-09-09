"""Score a tilt vector with the learned operator instead of the ray tracer.

Satisfies :class:`src.optim.evaluator.ObjectiveEvaluator`, so every search in
``src/optim/methods/`` takes one of these in place of the ray-traced
:class:`~src.optim.evaluator.Evaluator` without changing a line: same ``space``,
same ``evaluate``, same :class:`~src.optim.evaluator.EvaluationResult`, and the
KPIs come from the same :func:`src.optim.objective.evaluate_kpis`.

The 36 dimensions are predicted in one batch rather than one at a time. Each is
its own slice of the map, which is what makes the batch legitimate and the
whole evaluation a single forward pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from src.optim.evaluator import EvaluationResult
from src.optim.objective import evaluate_kpis
from src.optim.space import TiltSpace
from src.surrogate.dataset import Encoder, Sweep
from src.surrogate.model import TiltOperator


@dataclass
class SurrogateEvaluator:
    """Predict a radio map for a tilt vector, then score it exactly as usual.

    The anchor is the map at the committed tilts, taken from the sweep rather
    than from ``data/interim/radio_map.npz`` so that anchor and target share one
    solver seed and one material draw -- the difference between them is then the
    tilt change and nothing else.

    Attributes:
        cfg: The composed config, read for the KPI thresholds and the MDT path.
        encoder: The channel assembly the weights were fitted to.
        model: The trained operator. Put in eval mode on construction.
        device: Where the forward pass runs.
        keep_rsrp: Whether each result carries its predicted map.
        analytic_only: Skip the network and return the analytic re-embedding.
            The zero-parameter baseline, scored through the same path so the
            comparison is like for like.
    """

    cfg: DictConfig
    encoder: Encoder
    model: TiltOperator
    device: str = "cpu"
    keep_rsrp: bool = False
    analytic_only: bool = False

    space: TiltSpace = field(init=False)
    n_calls: int = field(init=False, default=0)
    total_seconds: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        """Read the tilt space, load the MDT, and park the model on the device."""
        self.space = TiltSpace.from_config(self.cfg)
        self._mdt = pd.read_parquet(self.cfg.data.output.mdt_file)
        self._anchor = self.sweep.at(self.space.baseline)
        self.model = self.model.to(self.device).eval()

    @property
    def sweep(self) -> Sweep:
        """The sweep the anchor and the tilt grids come from."""
        return self.encoder.sweep

    @property
    def band_labels(self) -> tuple[str, ...]:
        """Band names in the radio map's band-axis order."""
        return self.sweep.band_names

    def __enter__(self) -> SurrogateEvaluator:
        """Return the evaluator, ready to score.

        A context manager only so this is a drop-in for the ray-traced
        evaluator, which has GPU scene state to release. There is nothing here
        that has to be freed.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Nothing to release."""

    def predict(self, tilt_deg: np.ndarray) -> np.ndarray:
        """The predicted radio map, ``[n_band, n_tx, n_rows, n_cols]`` in dBm.

        ``nan`` where the coverage head says no path reaches the tile, matching
        what :func:`src.simulation.radio.solve_band` writes, so the KPIs need no
        special case for a predicted map.

        Raises:
            ValueError: When the vector is the wrong length or leaves the box.
        """
        values = self.space.clip(np.asarray(tilt_deg, dtype=float).reshape(-1))
        self.space.to_cells(values)

        n_band, n_tx = len(self.band_labels), len(self.sweep.tx_names)
        pairs = [
            (band, tx, float(values[tx * n_band + band]))
            for band in range(n_band)
            for tx in range(n_tx)
        ]
        batch = [
            self.encoder.assemble(
                band, tx, self._anchor[band, tx], float(self.space.baseline[tx * n_band + band]), to
            )
            for band, tx, to in pairs
        ]
        stacked = {
            key: torch.from_numpy(
                np.ascontiguousarray(np.stack([item[key] for item in batch]), dtype=np.float32)
            ).to(self.device)
            for key in ("x", "cond", "cond_map", "has_path", "baseline")
        }

        with torch.no_grad():
            if self.analytic_only:
                # The baseline keeps the anchor's coverage: re-embedding the
                # pattern cannot open a path the ray tracer never found.
                predicted = stacked["baseline"]
                covered = stacked["has_path"] > 0.5
            else:
                residual, logit = self.model(
                    stacked["x"], stacked["cond"], stacked["cond_map"], stacked["has_path"]
                )
                predicted = stacked["baseline"] + residual
                covered = logit > 0.0
            out = torch.where(covered, predicted, torch.nan).squeeze(1).cpu().numpy()

        rsrp = np.empty((n_band, n_tx, *self.sweep.shape), dtype=np.float32)
        for index, (band, tx, _to) in enumerate(pairs):
            rsrp[band, tx] = out[index]
        return rsrp

    def evaluate(self, tilt_deg: np.ndarray) -> EvaluationResult:
        """Predict this tilt vector's map and score it on all five KPIs."""
        started = time.perf_counter()
        rsrp = self.predict(tilt_deg)
        seconds = time.perf_counter() - started

        kpi = evaluate_kpis(rsrp, self.band_labels, self._mdt, self.cfg)
        self.n_calls += 1
        self.total_seconds += seconds
        return EvaluationResult(
            tilt_deg=np.asarray(tilt_deg, dtype=float).reshape(-1),
            kpi=kpi,
            seconds=seconds,
            rsrp=rsrp if self.keep_rsrp else None,
        )

    @classmethod
    def from_checkpoint(
        cls,
        cfg: DictConfig,
        path: str | Path | None = None,
        device: str = "cpu",
        **kwargs: object,
    ) -> SurrogateEvaluator:
        """Rebuild an evaluator from what :mod:`src.surrogate.train` wrote.

        Reads the sweep, the scene channels and the fitted pattern the
        checkpoint names, so a model can never be paired with an encoding it was
        not fitted against.
        """
        from src.surrogate.train import load_checkpoint

        model, encoder = load_checkpoint(cfg, path, device)
        return cls(cfg=cfg, encoder=encoder, model=model, device=device, **kwargs)
