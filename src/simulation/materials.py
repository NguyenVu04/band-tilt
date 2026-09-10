"""Install frequency-static ITU radio materials."""

from __future__ import annotations

from typing import Any


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
    # Zero inside an interval; otherwise choose the nearest interval.
    lo, hi = min(ranges, key=lambda r: max(r[0] - f_ghz, f_ghz - r[1], 0.0))
    a, b, c, d = ranges[(lo, hi)]
    return float(a * f_ghz**b), float(c * f_ghz**d)


def install(scene: Any, frequency_hz: float) -> dict[str, tuple[float, float]]:
    """Make the scene's materials frequency-static and valid at ``frequency_hz``.

    Each material keeps its identity — concrete stays concrete — but its
    frequency-update callback is switched off and its properties are set to the
    ITU values for ``frequency_hz``, with no scattering. Returns the installed
    ``(permittivity, conductivity)`` per material.

    **Call this before setting** ``scene.frequency``. The frequency setter
    invokes ``frequency_update()`` on every registered material, so a material
    that still holds its ITU callback will raise on a carrier outside its
    published range, such as 700 MHz.

    Materials are mutated in place rather than replaced. Building new ones and
    reassigning every object leaves the originals registered but unused, and
    the frequency setter walks the registry rather than the objects, so the
    originals would still raise; they also cannot be unregistered while the
    scene believes they are in use.
    """
    import mitsuba as mi

    installed: dict[str, tuple[float, float]] = {}
    for name in sorted(str(key) for key in scene.radio_materials):
        material = scene.get(name)
        permittivity, conductivity = evaluate(name, frequency_hz)

        material.frequency_update_callback = None
        material.relative_permittivity = mi.Float(permittivity)
        material.conductivity = mi.Float(conductivity)
        material.scattering_coefficient = mi.Float(0.0)
        installed[name] = (permittivity, conductivity)
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
