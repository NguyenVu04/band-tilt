"""FastAPI entry point and routes.

Run locally with ``task api`` (``uvicorn app.api.main:app --reload``) and open
http://localhost:8000/docs for the generated API documentation.

Routes stay thin: validate, delegate to :mod:`app.api.inference`, return a
schema. Anything resembling data logic belongs in ``src``, so that serving and
training cannot diverge.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI

from app.api.dependencies import get_model_service
from app.api.inference import ModelService
from app.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup and release it at shutdown.

    Args:
        app: The application instance.

    Yields:
        Control back to the server while it runs.

    Notes:
        Loading here rather than on first request means a missing or corrupt
        artifact fails the deployment immediately — the failure mode you want.
    """
    # TODO(1): get_model_service()  # warm the cache, fail fast on a bad artifact
    yield
    # TODO(2): release resources if the model holds any (GPU memory, sessions)


app = FastAPI(
    title="<project-name> API",
    description="Prediction service. See PROJECT.md for the project conventions.",
    version="0.1.0",
    lifespan=lifespan,
)

ModelDep = Annotated[ModelService, Depends(get_model_service)]


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(service: ModelDep) -> HealthResponse:
    """Report whether the service can serve predictions.

    Args:
        service: Injected model service.

    Returns:
        Status, whether the artifact is loaded, and its version.

    Raises:
        NotImplementedError: Always — implement this route first.

    Notes:
        Include the model version: knowing *which* model a deployment is
        serving is what makes an incident diagnosable.
    """
    # TODO(1): return HealthResponse(status="ok", model_loaded=service.is_loaded, ...)
    raise NotImplementedError("app.api.main.health")


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(request: PredictionRequest, service: ModelDep) -> PredictionResponse:
    """Score a single record.

    Args:
        request: The validated input record.
        service: Injected model service.

    Returns:
        The prediction and the model version that produced it.

    Raises:
        NotImplementedError: Always — implement this route first.
    """
    # TODO(1): frame = to_frame([request.model_dump()])
    # TODO(2): value = service.predict(frame)[0]
    # TODO(3): return PredictionResponse(prediction=value, model_version=service.version)
    raise NotImplementedError("app.api.main.predict")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(request: BatchPredictionRequest, service: ModelDep) -> BatchPredictionResponse:
    """Score several records in one call.

    Args:
        request: The validated input records.
        service: Injected model service.

    Returns:
        One prediction per input record, in request order.

    Raises:
        NotImplementedError: Always — implement this route first.

    Notes:
        Transform and predict over the whole frame at once; per-record loops
        give up most of the throughput a batch endpoint exists for. Cap the
        batch size so one request cannot exhaust memory.
    """
    # TODO(1): build one frame from all records and predict in a single call
    raise NotImplementedError("app.api.main.predict_batch")
