"""Tests for artifact serialisation.

The invariant under test: whatever is saved must come back able to
``transform`` and ``predict`` without refitting. A surrogate reloaded without
its feature transformer does not raise — it receives tilts on a different scale
from the one it was trained on and mispredicts silently, which then propagates
into an optimization result.
"""

from pathlib import Path

import pytest

from src.utils import io

pytestmark = pytest.mark.skip(reason="implement src/utils/io.py first")


def test_artifact_round_trips(tmp_path: Path) -> None:
    """What was saved is what is loaded."""
    payload = {"transformer": {"fitted": True}, "model": {"weights": [1, 2, 3]}}
    path = io.save_artifact(payload, tmp_path / "surrogate.pkl")
    assert io.load_artifact(path) == payload


def test_save_creates_missing_parent_directories(tmp_path: Path) -> None:
    """Saving into a fresh run directory must not require an mkdir at the call site."""
    path = io.save_artifact({"model": 1}, tmp_path / "nested" / "run" / "surrogate.pkl")
    assert Path(path).exists()


def test_payload_carries_the_transformer_with_the_model(tmp_path: Path) -> None:
    """Both halves travel together, or the reloaded surrogate mispredicts."""
    payload = {"transformer": {"fitted": True}, "model": {"weights": [1]}}
    loaded = io.load_artifact(io.save_artifact(payload, tmp_path / "surrogate.pkl"))
    assert "transformer" in loaded
    assert "model" in loaded


def test_metadata_is_readable_without_unpickling(tmp_path: Path) -> None:
    """Provenance must be inspectable without executing the payload.

    Unpickling to find out what produced an artifact means trusting it first,
    and it fails outright when the environment no longer has the classes.
    """
    path = io.save_artifact({"model": 1}, tmp_path / "surrogate.pkl", metadata={"seed": 42})
    assert io.artifact_metadata(path)["seed"] == 42


def test_metadata_records_the_cell_band_ordering(tmp_path: Path) -> None:
    """A theta vector cannot be interpreted without the column order it used.

    Adding a band to ``configs/radio.yaml`` after training makes every stored
    column index wrong, and the mismatch is invisible unless the ordering was
    saved alongside.
    """
    order = ["cell_a|high", "cell_a|low", "cell_b|high", "cell_b|low"]
    path = io.save_artifact({"model": 1}, tmp_path / "surrogate.pkl", metadata={"order": order})
    assert io.artifact_metadata(path)["order"] == order


def test_missing_artifact_raises(tmp_path: Path) -> None:
    """A missing artifact fails immediately, not at first prediction."""
    with pytest.raises(FileNotFoundError):
        io.load_artifact(tmp_path / "absent.pkl")
