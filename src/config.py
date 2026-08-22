"""Config loading and validation helpers.

Scripts get their config from the ``@hydra.main`` decorator. Notebooks cannot —
the decorator takes over the process entry point — so they use the Compose API
through :func:`load_config` instead. Both paths read the same files under
``configs/``, so a notebook and a script always see the same settings.

What must NOT go here
---------------------
Defaults and values. Every tunable lives in ``configs/*.yaml`` so that Git
history records what produced a result. This module only loads and checks.
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
        overrides: Hydra override strings, e.g. ``["models=model_a", "seed=7"]``.

    Returns:
        The composed config.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Use :func:`hydra.initialize_config_dir` with :data:`CONFIG_DIR` rather
        than ``initialize(config_path=...)``: the relative form resolves against
        the caller's file, which breaks when the same helper is called from
        ``notebooks/`` and from ``tests/``.

    Example:
        >>> cfg = load_config(overrides=["models=model_a"])
        >>> cfg.split.test_size
        0.2
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
        Worth checking here, because each of these fails late and confusingly:
        the split fractions are in ``(0, 1)``; ``split.group_col`` is declared
        in ``schema.columns`` when a group-aware method is selected;
        ``stratify_col`` is set if and only if the method needs it; no
        placeholder ``<...>`` values were left in the config.
    """
    # TODO(1): check split.test_size and split.val_size are in (0, 1)
    # TODO(2): check split.group_col exists in schema.columns for group methods
    # TODO(3): check no value still matches the "<placeholder>" pattern
    raise NotImplementedError("src.config.validate_config")
