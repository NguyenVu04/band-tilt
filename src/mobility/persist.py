"""Write trajectories and their manifest under ``data/interim/``, and read them back.

One scenario, one directory: ``<output.dir>/scenario_id=<id>/``, holding
``part-0.parquet`` (the trajectory frame) and ``manifest.json`` (provenance —
see :func:`src.data.scenario.build_manifest`). Hive-style partition naming so
``pyarrow.dataset`` — and any tool that understands the convention — reads
``scenario_id`` back as a column without a manual filter, and so a scenario's
identity survives the file being copied out of its directory (the
``scenario_id`` column on every row is the same information, redundantly).

``data/`` is DVC-tracked and never committed; this module writes under it but
does not run ``dvc add`` — that remains a manual step (``task dvc:repro`` /
``dvc add``), same as every other stage in this project.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig


def trajectory_dir(cfg: DictConfig, sid: str) -> Path:
    """The directory one scenario's trajectories and manifest live in.

    Args:
        cfg: Composed config; uses ``cfg.mobility.output.dir``.
        sid: The scenario id.

    Returns:
        ``<output.dir>/scenario_id=<sid>/``.

    Example:
        >>> trajectory_dir(cfg, "scn_a1b2c3d4e5f6a7b8")
        PosixPath('data/interim/trajectories/scenario_id=scn_a1b2c3d4e5f6a7b8')
    """
    return Path(cfg.mobility.output.dir) / f"scenario_id={sid}"


def save_trajectories(df: pd.DataFrame, manifest: dict[str, Any], cfg: DictConfig) -> Path:
    """Write a scenario's trajectory frame and manifest to disk.

    Args:
        df: The trajectory frame — :func:`src.mobility.simulate.tidy`'s
            output, after :func:`src.mobility.frame.clip_to_scene`. Must
            carry a ``scenario_id`` column matching ``manifest["scenario_id"]``.
        manifest: From :func:`src.data.scenario.build_manifest`; must have a
            ``scenario_id`` key.
        cfg: Composed config; uses ``cfg.mobility.output.dir``.

    Returns:
        The scenario's directory (see :func:`trajectory_dir`).

    Raises:
        ValueError: When ``df["scenario_id"]`` is not uniformly
            ``manifest["scenario_id"]`` — writing a mismatched pair would
            make the partition name and the row contents disagree about the
            scenario's identity.

    Example:
        >>> save_trajectories(clipped, manifest, cfg)
        PosixPath('data/interim/trajectories/scenario_id=scn_...')
    """
    sid = manifest["scenario_id"]
    if not df.empty and not (df["scenario_id"] == sid).all():
        raise ValueError(
            f"df['scenario_id'] does not match manifest['scenario_id']={sid!r} "
            "for every row; refusing to write a partition whose name and "
            "contents disagree."
        )

    out_dir = trajectory_dir(cfg, sid)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "part-0.parquet", index=False)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return out_dir


def load_trajectories(cfg: DictConfig, sid: str | None = None) -> pd.DataFrame:
    """Load one scenario's trajectories, or every scenario's, concatenated.

    Args:
        cfg: Composed config; uses ``cfg.mobility.output.dir``.
        sid: A specific scenario id, or ``None`` to load every scenario found
            under ``cfg.mobility.output.dir``.

    Returns:
        The trajectory frame(s), concatenated when ``sid`` is ``None``.

    Raises:
        FileNotFoundError: When the requested scenario (or, for ``sid=None``,
            any scenario at all) has no ``part-0.parquet`` under
            ``cfg.mobility.output.dir``.

    Example:
        >>> load_trajectories(cfg, "scn_a1b2c3d4e5f6a7b8").columns.tolist()
        ['scenario_id', 'ue_id', 't', ...]
    """
    base = Path(cfg.mobility.output.dir)
    if sid is not None:
        part = trajectory_dir(cfg, sid) / "part-0.parquet"
        if not part.is_file():
            raise FileNotFoundError(f"No trajectories at {part}")
        return pd.read_parquet(part)

    parts = sorted(base.glob("scenario_id=*/part-0.parquet"))
    if not parts:
        raise FileNotFoundError(f"No trajectories found under {base}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def load_manifest(cfg: DictConfig, sid: str) -> dict[str, Any]:
    """Load one scenario's manifest.

    Args:
        cfg: Composed config; uses ``cfg.mobility.output.dir``.
        sid: The scenario id.

    Returns:
        The manifest, as written by :func:`save_trajectories`.

    Raises:
        FileNotFoundError: When the scenario has no ``manifest.json``.

    Example:
        >>> load_manifest(cfg, "scn_a1b2c3d4e5f6a7b8")["seeds"]
        {'trips': 42, 'route': 43, 'sumo': 44}
    """
    path = trajectory_dir(cfg, sid) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def discard_sumo_inputs(sumo_dir: str | Path) -> None:
    """Remove the intermediate SUMO input directory for one scenario.

    Args:
        sumo_dir: The ``sumo/`` subdirectory under a scenario's output
            directory (trips, routes, the raw FCD file) — see
            :func:`src.mobility.simulate.generate`.

    Notes:
        Called when ``cfg.mobility.output.keep_sumo_inputs`` is ``False``
        (the default): every file under ``sumo_dir`` is regenerable from the
        manifest's seeds and the config, so keeping it is a convenience for
        inspecting or diffing a route set, not a requirement for
        reproducing one.

    Example:
        >>> discard_sumo_inputs(trajectory_dir(cfg, sid) / "sumo")
    """
    path = Path(sumo_dir)
    if path.is_dir():
        shutil.rmtree(path)
