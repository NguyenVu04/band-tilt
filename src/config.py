"""Config loading and validation helpers.

Scripts get their config from the ``@hydra.main`` decorator. Notebooks cannot —
the decorator takes over the process entry point — so they use the Compose API
through :func:`load_config` instead. Both paths read the same files under
``configs/``, so a notebook and a script always see the same settings.

What must NOT go here
---------------------
Defaults and values. Every tunable lives in ``configs/*.yaml`` so that Git
history records what produced a result. This module only loads and checks.

Why validation is worth the code
--------------------------------
Several config files still carry ``<placeholder>`` values, because PROJECT.md
section 30 leaves those parameters open until they are fixed experimentally. A
placeholder that reaches a numeric call site does not raise — it produces a
string comparison, a silent cast, or a KPI computed against the wrong
threshold. :func:`validate_config` is what turns that into an early, loud
failure.
"""

from pathlib import Path

from omegaconf import DictConfig

#: Absolute path to the configs directory, resolved from this file's location
#: so notebooks work regardless of the current working directory.
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Compose the Hydra config for use outside a ``@hydra.main`` entry point.

    Args:
        config_name: Name of the root config file in ``configs/``, without the
            ``.yaml`` suffix.
        overrides: Hydra override strings, e.g. ``["optim=marl", "seed=7"]``.

    Returns:
        The composed config, namespaced as ``cfg.data``, ``cfg.radio``,
        ``cfg.kpi``, ``cfg.surrogate`` and ``cfg.optim``.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Use :func:`hydra.initialize_config_dir` with :data:`CONFIG_DIR` rather
        than ``initialize(config_path=...)``: the relative form resolves against
        the caller's file, which breaks when the same helper is called from
        ``notebooks/`` and from ``tests/``.

    Example:
        >>> cfg = load_config(overrides=["optim=marl"])
        >>> cfg.kpi.hole_dbm
        -120.0
    """
    # TODO(1): with initialize_config_dir(str(CONFIG_DIR), version_base=None):
    # TODO(2):     cfg = compose(config_name=config_name, overrides=overrides or [])
    # TODO(3): call validate_config(cfg) before returning
    raise NotImplementedError("src.config.load_config")


def validate_config(cfg: DictConfig) -> None:
    """Fail fast on configs that would silently produce unusable results.

    Args:
        cfg: The composed config.

    Raises:
        NotImplementedError: Always — implement this module first.
        ValueError: Once implemented, when a required setting is missing or
            inconsistent.

    Notes:
        Each of these fails late and confusingly if it is not caught here:

        - Split fractions ``cfg.data.split.test_size`` / ``val_size`` in
          ``(0, 1)``, and ``split.group_col`` declared in the MDT schema.
        - KPI thresholds ordered ``hole_dbm < weak_dbm``. Reversed, every
          location classifies as a hole and the optimizer chases a constant.
        - Every band in ``cfg.radio.bands`` has ``tilt.min < tilt.max`` and a
          strictly positive ``priority_weight`` (PROJECT.md section 4.7).
        - ``cfg.kpi.order`` equals :data:`src.kpi.vector.KPI_NAMES`. The
          config declares the priority and the module indexes by position, so a
          divergence silently scores every candidate against the wrong
          objective (PROJECT.md section 5).
        - Scalarization weights ordered ``hole > overlap > overlap_neighbors >
          bps > weak`` when ``cfg.kpi.mode`` is ``scalarized`` (PROJECT.md
          section 25.3) — the weights can otherwise contradict the stated
          priority without any error.
        - No value anywhere in the tree still matches ``<...>``.

    Example:
        >>> validate_config(cfg)
    """
    # TODO(1): check data.split.test_size and data.split.val_size are in (0, 1)
    # TODO(2): check data.split.group_col exists in data.schema.mdt.columns
    # TODO(3): check kpi.hole_dbm < kpi.weak_dbm
    # TODO(4): check every radio.bands[b] has tilt.min < tilt.max and priority_weight > 0
    # TODO(5): check tuple(kpi.order) == src.kpi.vector.KPI_NAMES
    # TODO(6): if kpi.mode == "scalarized", check weights follow kpi.order
    # TODO(7): walk the whole tree and reject any remaining "<placeholder>" string
    raise NotImplementedError("src.config.validate_config")
