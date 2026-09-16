"""Soft thresholds: the sigmoid that turns a KPI into a desirability.

Lives here rather than in :mod:`src.optim.objective` because both sides need it
and the dependency only runs one way: :mod:`src.kpi` knows nothing of the
optimizer. The objective softens its scalar KPIs at score time; the served term
softens per tile at measure time (:func:`src.kpi.served.served_desirability`),
and neither may use a different curve from the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig
from scipy.special import expit


@dataclass(frozen=True)
class SoftSpec:
    """Where one KPI's desirability turns, and how sharply.

    Attributes:
        target: The KPI value scoring exactly 0.5. Calibrate it near the
            incumbent's measurement, because that is where the sigmoid responds
            most and so has the most to say about a small change.
        temperature: The KPI units the curve turns over. Set it from the KPI's
            measured solver noise (``kpi.tolerance``), so noise barely moves the
            desirability while a real improvement crosses the band.
    """

    target: float
    temperature: float


def soft_spec(cfg: DictConfig, name: str) -> SoftSpec:
    """Read ``kpi.soft.<name>``.

    Raises:
        ValueError: When ``kpi.soft`` or the entry is absent, or the target is
            not finite, or the temperature is not finite and positive. There is
            no safe default: a temperature of 1.0 on a rate that moves by
            thousandths makes the sigmoid indistinguishable from a straight
            line, silently reverting the objective to the weighted sum it
            replaced.
    """
    block = cfg.kpi.get("soft")
    if block is None:
        raise ValueError(
            "configs/kpi.yaml has no `soft` block. The objective needs a target and "
            "temperature per KPI it scores."
        )
    if name not in block:
        raise ValueError(f"kpi.soft has no entry for {name}")

    entry = block[name]
    target, temperature = float(entry["target"]), float(entry["temperature"])
    if not np.isfinite(target):
        raise ValueError(f"kpi.soft.{name}.target must be finite")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"kpi.soft.{name}.temperature must be finite and positive")
    return SoftSpec(target=target, temperature=temperature)


def soften(values: np.ndarray, spec: SoftSpec, *, maximise: bool) -> np.ndarray:
    """Map KPI values to desirabilities in ``(0, 1)``; larger is better.

    ``sigmoid(s * (value - target) / temperature)``, with ``s`` positive for a
    maximised KPI and negative for a minimised one. One is comfortably past the
    target, zero far short of it, and a half exactly on it.

    The curve is strictly monotone, so it cannot reorder candidates on a single
    KPI. What it changes is the exchange rate between KPIs: past its target a
    KPI saturates and stops paying, so the search spends its moves on whichever
    KPI is still short. That is the point of the formulation.
    """
    sign = 1.0 if maximise else -1.0
    return expit(sign * (np.asarray(values, dtype=float) - spec.target) / spec.temperature)
