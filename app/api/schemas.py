"""Request and response models — the service's public contract.

These schemas are the boundary between callers and the model. Keep them
stable and explicit: renaming a field here breaks clients, and accepting a
loosely typed payload pushes validation failures into the model, where they
surface as strange predictions rather than a clear 422.

Mirror the feature contract in ``configs/data.yaml``. When a hard bound is
declared there, express it here too (``ge``/``le``) so invalid input is
rejected at the edge rather than silently scored.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """One record to score.

    Attributes:
        feature_a: Replace with the real features. Constraints should match the
            hard bounds declared in ``configs/data.yaml``.
    """

    feature_a: float = Field(
        ...,
        description="<what this feature is, and its unit>",
        # ge=<lower-bound>, le=<upper-bound>,   # mirror configs/data.yaml
    )
    # TODO(1): add one field per model input, with description and bounds
    # TODO(2): add a model_config example so /docs shows a valid payload


class BatchPredictionRequest(BaseModel):
    """Several records scored in one call.

    Attributes:
        records: The records to score.
    """

    records: list[PredictionRequest]


class PredictionResponse(BaseModel):
    """The model's answer for one record.

    Attributes:
        prediction: The predicted value or label.
        model_version: Which artifact produced it — read from the artifact
            metadata, so a response can always be traced back to a model.
    """

    prediction: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    """Predictions for a batch, in request order.

    Attributes:
        predictions: One entry per input record.
    """

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Liveness and readiness information.

    Attributes:
        status: ``"ok"`` when the service can serve predictions.
        model_loaded: Whether the artifact is in memory.
        model_version: The loaded artifact's version, when known.
    """

    status: str
    model_loaded: bool
    model_version: str | None = None
