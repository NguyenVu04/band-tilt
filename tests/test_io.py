"""Tests for artifact serialisation.

The invariant under test is the project's serving contract: whatever is saved
must come back able to ``transform`` and ``predict`` without refitting.
"""

from pathlib import Path

import pytest

from src.utils import io

pytestmark = pytest.mark.skip(reason="implement src/utils/io.py first")


def test_artifact_round_trips(tmp_path: Path) -> None:
    """What was saved is what is loaded."""
    payload = {"preprocessor": {"fitted": True}, "model": {"weights": [1, 2, 3]}}
    path = io.save_artifact(payload, tmp_path / "model_a.pkl")
    assert io.load_artifact(path) == payload


def test_save_creates_missing_directories(tmp_path: Path) -> None:
    """Saving into a fresh directory works without a manual mkdir.

    ``models/`` is gitignored, so it may well not exist on a clean clone.
    """
    path = io.save_artifact({"model": 1}, tmp_path / "nested" / "dir" / "model_a.pkl")
    assert Path(path).exists()


def test_artifact_contains_preprocessor_and_model(tmp_path: Path) -> None:
    """Both halves are stored together.

    A model saved without its fitted preprocessing cannot be served the way it
    was trained, and the resulting skew shows up as wrong predictions rather
    than as an error.
    """
    payload = {"preprocessor": object(), "model": object()}
    path = io.save_artifact(payload, tmp_path / "model_a.pkl")
    loaded = io.load_artifact(path)
    assert {"preprocessor", "model"} <= set(loaded)


def test_metadata_is_readable(tmp_path: Path) -> None:
    """Provenance can be read back without unpickling the model.

    The serving layer's health endpoint needs the version, not the weights.
    """
    path = io.save_artifact({"model": 1}, tmp_path / "model_a.pkl", metadata={"version": "v1"})
    assert io.artifact_metadata(path)["version"] == "v1"


def test_missing_artifact_raises_file_not_found(tmp_path: Path) -> None:
    """A missing artifact fails with a clear error, not a pickle error."""
    with pytest.raises(FileNotFoundError):
        io.load_artifact(tmp_path / "absent.pkl")
