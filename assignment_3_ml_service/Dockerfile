# Dockerfile
# ----------
# Packages the already-trained model + FastAPI app into a container that
# serves predictions. Training happens OUTSIDE Docker (via `python
# src/train.py` on the host) -- this image only serves the model that
# training already produced. That keeps the image small and fast to build,
# and matches how the assignment describes the service: train once,
# persist the pipeline, then deploy the artifact.

FROM python:3.12-slim

# Prevents Python from writing .pyc files and buffering stdout/stderr --
# both make container logs behave more predictably.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first, separately from application code. Docker
# caches this layer, so re-building after only changing api/main.py (for
# example) doesn't reinstall every package from scratch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the running service actually needs: the API code, the
# monitoring module, the persisted model artifact, and the static frontend.
# Training scripts, notebooks, and raw data are intentionally NOT copied
# in -- see .dockerignore -- since the service doesn't need them to run.
COPY api/ ./api/
COPY monitoring/ ./monitoring/
COPY model/ ./model/
COPY static/ ./static/

EXPOSE 8000

# No secrets or credentials are baked into this image -- the model doesn't
# need an API key, database password, or similar to run.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
