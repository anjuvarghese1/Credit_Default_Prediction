# syntax=docker/dockerfile:1
# Single-stage image that runs the full credit-risk benchmark
#   docker build -t credit-risk-pipeline .
#   docker run --rm -v "$(pwd)/reports:/app/reports" credit-risk-pipeline

# To run against the real Kaggle data, also mount it into /app/data:
#   docker run --rm \
#     -v "$(pwd)/data:/app/data" \
#     -v "$(pwd)/reports:/app/reports" \
#     credit-risk-pipeline

FROM python:3.12-slim

# Avoid interactive prompts and keep Python output unbuffered for clean logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# install dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the project source and tests
COPY src/ ./src/
COPY tests/ ./tests/

#   docker run --rm credit-risk-pipeline python -m pytest tests/ -q
CMD ["python", "-m", "credit_risk.train"]
