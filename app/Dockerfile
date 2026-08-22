# Serving image for the FastAPI service.
#
# Build and run from the project root (the build context must include src/):
#   task docker:build
#   task docker:run
#
# Two stages so the final image carries the virtualenv but not the build tools
# or the uv cache.

# --- build ------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Pinned so image builds are reproducible; bump deliberately.
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Dependency layer first: it only changes when the manifest changes, so source
# edits do not re-resolve the environment.
COPY pyproject.toml README.md ./
COPY uv.lock* ./
RUN uv sync --extra app --no-install-project

# Then the code.
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
RUN uv sync --extra app --no-editable

# --- runtime ----------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Never run the service as root.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /srv

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/src /srv/src
COPY --from=builder /build/app /srv/app
COPY --from=builder /build/configs /srv/configs

# Model artifacts are DVC-tracked, not baked into the image. Mount them, or add
# a `dvc pull` step here with credentials supplied at build time:
#   docker run -v ./models:/srv/models ...
COPY models/ /srv/models/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
