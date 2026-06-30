# syntax=docker/dockerfile:1
#
# Single-stage image that runs the full credit-risk benchmark.
# The default command trains all models, evaluates them, and writes the
# report + figures to /app/reports -- mount a volume there to retrieve them.
#
#   docker build -t credit-risk-pipeline .
#   docker run --rm -v "$(pwd)/reports:/app/reports" credit-risk-pipeline
#
# To run against the real Kaggle data, also mount it into /app/data:
#   docker run --rm \
#     -v "$(pwd)/data:/app/data" \
#     -v "$(pwd)/reports:/app/reports" \
#     credit-risk-pipeline

FROM python:3.12-slim

# Avoid interactive prompts and keep Python output unbuffered for clean logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install dependencies first so layer caching skips reinstalls on code edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project source and tests.
COPY src/ ./src/
COPY tests/ ./tests/

# Default: run the benchmark. Override with e.g. `pytest` to run tests:
#   docker run --rm credit-risk-pipeline python -m pytest tests/ -q
CMD ["python", "-m", "credit_risk.train"]
