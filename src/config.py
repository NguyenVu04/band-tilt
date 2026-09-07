"""Compose the Hydra config outside a ``@hydra.main`` entry point.

Every stage runs under ``@hydra.main``, which composes the config for it. A
notebook has no such decorator, so it composes the same config through here
rather than restating any of it.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

_CONFIG_DIR = "configs"
_CONFIG_NAME = "config"


def load_config(overrides: list[str] | None = None, root: Path | None = None) -> DictConfig:
    """Compose ``configs/config.yaml``, taking the overrides the CLI takes.

    Args:
        overrides: Hydra overrides, such as ``["seed=7", "simulation.grid.tile_size_m=40"]``.
        root: Directory holding ``configs/``. Defaults to the working directory,
            which the notebooks change to the project root.

    Returns:
        The same config a stage's ``@hydra.main`` would receive.

    Raises:
        FileNotFoundError: When ``root`` holds no ``configs/`` directory.
    """
    directory = (root or Path.cwd()) / _CONFIG_DIR
    if not directory.is_dir():
        raise FileNotFoundError(
            f"No {directory}. Run from the project root, or pass root=<project root>."
        )
    # Hydra refuses to initialise twice, which re-running a notebook cell is.
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(directory.resolve())):
        return compose(config_name=_CONFIG_NAME, overrides=list(overrides or ()))
