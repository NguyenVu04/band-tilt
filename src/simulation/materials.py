"""Frequency-static radio materials, and their per-scenario perturbation.

The bundled scenes ship ITU materials, whose permittivity and conductivity are
recomputed from ITU-R P.2040 every time the scene's carrier changes. That is
convenient and, for this project, twice wrong:

- **It forbids sub-GHz carriers.** P.2040 Table 3 publishes coefficients for
  concrete, brick, marble and metal only from 1 GHz up, and sionna-rt raises
  rather than extrapolating outside a published range. Every low band a
  coverage layer would actually use sits below that floor.
- **It discards perturbations.** Setting a permittivity and then setting the
  frequency recomputes the value straight back, silently.

So the ITU materials are replaced with plain ones carrying no frequency-update
callback, evaluated here at each band's own frequency. The ITU model is applied
with its validity range deliberately not enforced. That extrapolation is mild:
the exponent on frequency is zero for every material these scenes use, so
permittivity is unchanged from its published value and only conductivity moves,
smoothly. It is still outside the range P.2040 validates, and a report using a
sub-GHz band has to say so.

The coefficients are read from sionna-rt rather than transcribed, so there is
no second copy of the ITU table to drift out of step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import DictConfig


@dataclass(frozen=True)
class MaterialSpec:
    """How far a scenario's materials may stray from their ITU values.

    Each range is a multiplier drawn uniformly and applied to the ITU value,
    except ``scattering_coefficient``, which is an absolute value because ITU
    does not define one.

    Attributes:
        relative_permittivity_scale: Multiplier range for permittivity.
        conductivity_scale: Multiplier range for conductivity.
        scattering_coefficient: Absolute range for the scattering coefficient.
    """

    relative_permittivity_scale: tuple[float, float]
    conductivity_scale: tuple[float, float]
    scattering_coefficient: tuple[float, float]

    @classmethod
    def from_config(cls, cfg: DictConfig) -> MaterialSpec:
        """Read ``simulation.materials.perturbation``."""
        perturbation = cfg.simulation.materials.perturbation
        return cls(
            relative_permittivity_scale=_pair(perturbation.relative_permittivity_scale),
            conductivity_scale=_pair(perturbation.conductivity_scale),
            scattering_coefficient=_pair(perturbation.scattering_coefficient),
        )


def evaluate(name: str, frequency_hz: float) -> tuple[float, float]:
    """Relative permittivity and conductivity [S/m] of an ITU material.

    Implements ITU-R P.2040 Section 2.1.4: ``eta_r = a * f_GHz**b`` and
    ``sigma = c * f_GHz**d``. Where the frequency falls outside every published
    range for the material, the nearest range's coefficients are extrapolated
    rather than raising — which is the whole point of this module.

    Raises:
        ImportError: When sionna-rt's ITU table cannot be imported.
        ValueError: When ``name`` is not an ITU material.
    """
    table = _itu_table()
    if name not in table:
        raise ValueError(f"{name!r} is not an ITU material. Known: {', '.join(sorted(table))}.")

    f_ghz = frequency_hz / 1e9
    ranges = table[name]
    # Distance from the frequency to each published interval; zero when inside.
    lo, hi = min(ranges, key=lambda r: max(r[0] - f_ghz, f_ghz - r[1], 0.0))
    a, b, c, d = ranges[(lo, hi)]
    return float(a * f_ghz**b), float(c * f_ghz**d)


def install(
    scene: Any,
    frequency_hz: float,
    spec: MaterialSpec,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    """Make the scene's materials frequency-static, perturbed, and valid here.

    Each material keeps its identity — concrete stays concrete — but its
    frequency-update callback is switched off and its properties are set to the
    ITU values for ``frequency_hz``, scaled by this scenario's draw. Returns
    the installed ``(permittivity, conductivity, scattering)`` per material.

    **Call this before setting** ``scene.frequency``. The frequency setter
    invokes ``frequency_update()`` on every registered material, so a material
    that still holds its ITU callback will both raise on a carrier outside its
    published range and overwrite the values set here.

    Materials are mutated in place rather than replaced. Building new ones and
    reassigning every object leaves the originals registered but unused, and
    the frequency setter walks the registry rather than the objects, so the
    originals would still raise; they also cannot be unregistered while the
    scene believes they are in use.

    The draw depends only on ``seed`` and the material names, so every band of
    one scenario sees the same materials — a scenario is one world, not one per
    band — and re-running for another band recomputes from the ITU value rather
    than compounding the previous scaling.

    Note that ``scattering_coefficient`` only reaches the result when the
    solver runs with diffuse reflection enabled; it is otherwise inert.
    """
    import mitsuba as mi

    rng = np.random.default_rng(seed)
    installed: dict[str, tuple[float, float, float]] = {}
    for name in sorted(str(key) for key in scene.radio_materials):
        material = scene.get(name)
        permittivity, conductivity = evaluate(name, frequency_hz)
        permittivity *= rng.uniform(*spec.relative_permittivity_scale)
        conductivity *= rng.uniform(*spec.conductivity_scale)
        scattering = rng.uniform(*spec.scattering_coefficient)

        material.frequency_update_callback = None
        material.relative_permittivity = mi.Float(permittivity)
        material.conductivity = mi.Float(conductivity)
        material.scattering_coefficient = mi.Float(scattering)
        installed[name] = (permittivity, conductivity, scattering)
    return installed


def _itu_table() -> dict[str, dict[tuple[float, float], tuple[float, float, float, float]]]:
    """sionna-rt's copy of ITU-R P.2040 Table 3.

    Raises:
        ImportError: When the table cannot be imported. It lives in a private
            sionna-rt module, so an upgrade can move it.
    """
    try:
        from sionna.rt.radio_materials.itu import ITU_MATERIALS_PROPERTIES
    except ImportError as exc:  # pragma: no cover - depends on the installed sionna-rt
        raise ImportError(
            "Could not import ITU_MATERIALS_PROPERTIES from "
            "sionna.rt.radio_materials.itu. It is a private module, so a sionna-rt "
            "upgrade may have moved it; this module needs the P.2040 coefficients."
        ) from exc
    return ITU_MATERIALS_PROPERTIES


def _pair(values: Any) -> tuple[float, float]:
    """Read a two-element config range."""
    low, high = (float(value) for value in values)
    return low, high
