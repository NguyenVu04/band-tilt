"""Fit the tilt-delta operator to the sweep, and write what reloads it.

The model predicts a residual over the analytic re-embedding, so the loss is on
what the physics gets wrong rather than on the map itself. Two terms carry the
supervision and a third regularises it:

``level``
    Huber on dB wherever the *target* has a path -- not wherever both ends do.
    A tile the tilt change opened has no source level and would be dropped by
    the stricter mask, which is exactly the tile the coverage head exists for.
``coverage``
    Cross-entropy on whether a path reaches the tile after the change.
``cycle``
    The operator run backwards on its own prediction has to return the map it
    started from. Free supervision, and a physical identity rather than a
    penalty chosen for convenience.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.optim.space import TiltSpace
from src.surrogate.dataset import (
    DB_CENTRE,
    DB_SCALE,
    GAIN_SCALE,
    Encoder,
    Sweep,
    TiltPairs,
    fit_pattern,
    load_features,
)
from src.surrogate.model import OperatorSpec, TiltOperator
from src.tracking import log_stage


@dataclass(frozen=True)
class TrainSpec:
    """The knobs training reads.

    Attributes:
        epochs: Passes over the training pairs.
        batch_size: Pairs per step.
        learning_rate: Adam's initial rate; cosine-annealed to zero.
        huber_delta: Where the level loss turns linear, in dB. Above a few dB
            an error is a shadow the model missed, not a level it mis-scaled,
            and squaring those lets a handful of tiles set the gradient.
        coverage_weight: Weight on the coverage cross-entropy.
        cycle_weight: Weight on the reverse-consistency term.
        device: Where to train.
        seed: Seeds the initialisation and the shuffling.
    """

    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 2e-3
    huber_delta: float = 3.0
    coverage_weight: float = 0.2
    cycle_weight: float = 0.1
    device: str = "cuda"
    seed: int = 42

    @classmethod
    def from_config(cls, cfg: DictConfig) -> TrainSpec:
        """Read ``surrogate.train``."""
        return cls(**{field: cfg.surrogate.train[field] for field in cls.__dataclass_fields__})


@dataclass(frozen=True)
class EpochReport:
    """What one epoch cost, on both splits."""

    epoch: int
    train_loss: float
    train_mae_db: float
    test_mae_db: float
    test_coverage_f1: float
    seconds: float


def losses(
    model: TiltOperator, batch: dict[str, torch.Tensor], spec: TrainSpec
) -> tuple[torch.Tensor, dict[str, float]]:
    """The total loss for one batch, and the parts it is made of."""
    residual, logit = model(batch["x"], batch["cond"], batch["cond_map"], batch["has_path"])
    predicted = batch["baseline"] + residual
    target_mask = batch["target_mask"]

    level = _masked_huber(predicted, batch["target"], target_mask, spec.huber_delta)
    coverage = torch.nn.functional.binary_cross_entropy_with_logits(logit, target_mask)

    reverse_residual, _ = model(*_reverse(batch, predicted, torch.sigmoid(logit)))
    returned = predicted - _delta_gain(batch) + reverse_residual
    source = batch["x"][:, :1] * DB_SCALE + DB_CENTRE
    cycle = _masked_huber(returned, source, batch["has_path"] * target_mask, spec.huber_delta)

    total = level + spec.coverage_weight * coverage + spec.cycle_weight * cycle
    return total, {
        "level": level.detach().item(),
        "coverage": coverage.detach().item(),
        "cycle": cycle.detach().item(),
    }


def _delta_gain(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """The analytic correction the input stack carries, back in dB."""
    return batch["x"][:, 2:3] * GAIN_SCALE


def _reverse(
    batch: dict[str, torch.Tensor], predicted: torch.Tensor, coverage: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """The same batch with the tilt change turned around.

    Built from tensors rather than re-encoded through :class:`Encoder`, which
    is what keeps the cycle term differentiable: the reverse of a tilt change
    reuses every static channel, negates the analytic correction, and swaps the
    two angle maps. Only the map being moved from is different, and that is the
    forward prediction itself.
    """
    x = batch["x"].clone()
    x[:, :1] = (predicted - DB_CENTRE) / DB_SCALE
    x[:, 1:2] = coverage
    x[:, 2:3] = -x[:, 2:3]

    cond = batch["cond"].clone()
    cond[:, 0], cond[:, 1] = batch["cond"][:, 1], batch["cond"][:, 0]
    cond[:, 2] = -batch["cond"][:, 2]

    return x, cond, batch["cond_map"].flip(1), coverage


def _masked_huber(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float
) -> torch.Tensor:
    """Huber over the masked tiles only, safe when a batch has none.

    The loss runs on every tile and the mask is applied after, which does about
    13% more arithmetic than it needs to. Left that way deliberately: measured
    at 0.75 ms of a 4.3 s training step, so the waste is a tenth of a
    millisecond. Skipping it means indexing by the mask, which forces a device
    synchronisation and a dynamic shape, and on a tensor this small that is
    likely to cost more than it saves.
    """
    per_tile = torch.nn.functional.huber_loss(predicted, target, reduction="none", delta=delta)
    return (per_tile * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def score(model: TiltOperator, loader: DataLoader, device: str) -> tuple[float, float]:
    """Mean absolute dB error over covered tiles, and the coverage head's F1."""
    model.eval()
    error = weight = 0.0
    true_positive = predicted_positive = actual_positive = 0.0

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        residual, logit = model(batch["x"], batch["cond"], batch["cond_map"], batch["has_path"])
        mask = batch["target_mask"]

        absolute = ((batch["baseline"] + residual) - batch["target"]).abs()
        error += float((absolute * mask).sum())
        weight += float(mask.sum())

        hit = (logit > 0.0).float()
        true_positive += float((hit * mask).sum())
        predicted_positive += float(hit.sum())
        actual_positive += float(mask.sum())

    model.train()
    denominator = predicted_positive + actual_positive
    return (
        error / max(weight, 1.0),
        2.0 * true_positive / denominator if denominator else 0.0,
    )


def fit(
    train_pairs: TiltPairs,
    test_pairs: TiltPairs,
    spec: TrainSpec,
    operator: OperatorSpec,
) -> tuple[TiltOperator, list[EpochReport]]:
    """Train the operator and report both splits after every epoch."""
    torch.manual_seed(spec.seed)
    device = spec.device if torch.cuda.is_available() or spec.device == "cpu" else "cpu"
    if device != spec.device:
        print(f"WARNING no CUDA device; training on {device}")

    model = TiltOperator(operator).to(device).train()
    print(f"operator: {model.n_parameters} parameters, {len(train_pairs)} training pairs")

    train_loader = DataLoader(train_pairs, batch_size=spec.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_pairs, batch_size=spec.batch_size)
    optimiser = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(spec.epochs * (len(train_loader) or 1), 1)
    )

    history = []
    for epoch in range(spec.epochs):
        started = time.perf_counter()
        running = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimiser.zero_grad()
            total, _parts = losses(model, batch, spec)
            total.backward()
            optimiser.step()
            schedule.step()
            running += float(total)

        train_mae, _ = score(model, train_loader, device)
        test_mae, test_f1 = score(model, test_loader, device)
        report = EpochReport(
            epoch=epoch,
            train_loss=running / max(len(train_loader), 1),
            train_mae_db=train_mae,
            test_mae_db=test_mae,
            test_coverage_f1=test_f1,
            seconds=time.perf_counter() - started,
        )
        history.append(report)
        print(
            f"epoch {epoch:>3}  loss {report.train_loss:8.4f}  "
            f"train MAE {train_mae:6.3f} dB  test MAE {test_mae:6.3f} dB  "
            f"coverage F1 {test_f1:5.3f}  {report.seconds:5.1f}s"
        )

    return model.cpu(), history


def build_pairs(cfg: DictConfig) -> tuple[TiltPairs, TiltPairs, Encoder]:
    """Read the sweep and the scene channels, fit the pattern, split the pairs.

    Raises:
        FileNotFoundError: When the sweep or the scene channels are missing,
            which means notebook 03a has not been run.
    """
    sweep_path = Path(cfg.surrogate.output.sweep_file)
    feature_path = Path(cfg.surrogate.output.feature_file)
    for path in (sweep_path, feature_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"No surrogate artifact at {path}. Run `task surrogate:dataset` first."
            )

    sweep_data = Sweep.load(sweep_path)
    features = load_features(feature_path)
    pattern = fit_pattern(sweep_data, features.elevation_deg)
    cells = TiltSpace.from_config(cfg).cells
    train_pairs = TiltPairs(sweep_data, features, pattern, cells, "train")
    test_pairs = TiltPairs(sweep_data, features, pattern, cells, "test")
    return train_pairs, test_pairs, train_pairs.encoder


def train(cfg: DictConfig) -> Path:
    """Fit the operator and write the checkpoint. The script form of notebook 03b."""
    spec = TrainSpec.from_config(cfg)
    train_pairs, test_pairs, _encoder = build_pairs(cfg)
    operator = OperatorSpec(
        in_channels=len(TiltPairs.channels),
        # Two tilts, their difference, and which band this is.
        cond_dim=3 + len(train_pairs.sweep.band_names),
        **{field: cfg.surrogate.model[field] for field in cfg.surrogate.model},
    )
    model, history = fit(train_pairs, test_pairs, spec, operator)

    path = Path(cfg.surrogate.output.model_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "operator": asdict(operator),
            "train": asdict(spec),
            "history": [asdict(report) for report in history],
            # Named rather than embedded: the encoding must come from the same
            # sweep the weights were fitted to, and a path that no longer
            # resolves is a loud failure where a stale copy would be a silent one.
            "sweep_file": str(cfg.surrogate.output.sweep_file),
            "feature_file": str(cfg.surrogate.output.feature_file),
        },
        path,
    )
    print(f"surrogate: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_checkpoint(
    cfg: DictConfig, path: str | Path | None = None, device: str = "cpu"
) -> tuple[TiltOperator, Encoder]:
    """Rebuild the operator and the encoding it was fitted with."""
    checkpoint = torch.load(
        Path(path or cfg.surrogate.output.model_file), map_location=device, weights_only=False
    )
    model = TiltOperator(OperatorSpec(**checkpoint["operator"]))
    model.load_state_dict(checkpoint["state_dict"])

    sweep_data = Sweep.load(checkpoint["sweep_file"])
    features = load_features(checkpoint["feature_file"])
    pattern = fit_pattern(sweep_data, features.elevation_deg)
    encoder = Encoder.build(sweep_data, features, pattern, TiltSpace.from_config(cfg).cells)
    return model, encoder


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for ``task surrogate:train``."""
    path = train(cfg)
    epochs = torch.load(path, map_location="cpu", weights_only=False)["history"]
    log_stage(
        cfg,
        "surrogate_train",
        groups=["surrogate"],
        step_metrics=[{k: v for k, v in e.items() if k != "epoch"} for e in epochs],
        artifacts=[path],
    )


if __name__ == "__main__":
    main()
