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
from src.optim.objective import SURROGATE, evaluate_kpis
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

    @property
    def scenario_id(self) -> str:
        """The scenario the sweep behind this model was solved in."""
        return self.sweep.scenario_id

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
        # Band-major over the 36 slices, so the model's output reshapes straight
        # back onto the map's [band, tx] axes.
        bands = np.repeat(np.arange(n_band), n_tx)
        txs = np.tile(np.arange(n_tx), n_band)
        # TiltSpace is ordered cell-major and band-minor, which is not the order
        # a radio map's axes are in.
        dimension = txs * n_band + bands

        encoded = self.encoder.encode_batch(
            bands,
            txs,
            self._anchor[bands, txs],
            self.space.baseline[dimension],
            values[dimension],
        )
        stacked = {
            key: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(self.device)
            for key, value in encoded.items()
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

        return out.reshape(n_band, n_tx, *self.sweep.shape).astype(np.float32)

    def evaluate(self, tilt_deg: np.ndarray) -> EvaluationResult:
        """Predict this tilt vector's map and score it on all four KPIs."""
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
            source=SURROGATE,
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


def from_config(cfg: DictConfig, **kwargs: object) -> SurrogateEvaluator:
    """Build the evaluator ``optim.search`` names, or say exactly what is missing.

    The entry point :mod:`src.optim.run` uses. Both checks below fail loudly
    rather than degrading: a search is cheap enough to repeat and expensive
    enough to waste.

    Raises:
        FileNotFoundError: When no operator has been trained, naming the two
            tasks that produce one.
        ValueError: When the sweep the operator was fitted to belongs to a
            different scenario than this config describes. The search would
            then explore one world and :mod:`src.optim.report` measure another,
            and every artifact would still look entirely plausible.
    """
    from src.simulation import scenario as scenario_module

    path = Path(cfg.optim.search.model_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"No trained surrogate at {path}. The search scores candidates with it, so run "
            "`task surrogate:dataset` then `task surrogate:train` first."
        )

    evaluator = SurrogateEvaluator.from_checkpoint(
        cfg, path, str(cfg.optim.search.device), **kwargs
    )
    expected = scenario_module.scenario_id(cfg)
    if evaluator.scenario_id != expected:
        raise ValueError(
            f"{path} was fitted on scenario {evaluator.scenario_id}, but this config is "
            f"{expected}. Re-run `task surrogate:dataset` and `task surrogate:train` against "
            "the current scenario before searching."
        )
    return evaluator
