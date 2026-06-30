# Convenience targets. Run `make help` for the list.

.PHONY: help install train test docker-build docker-run clean

help:
	@echo "Targets:"
	@echo "  install       Install dependencies into the active environment"
	@echo "  train         Run the full benchmark (data -> models -> report)"
	@echo "  test          Run the test suite"
	@echo "  docker-build  Build the Docker image"
	@echo "  docker-run    Run the benchmark in Docker, writing to ./reports"
	@echo "  clean         Remove generated artifacts and caches"

install:
	pip install -r requirements.txt

train:
	PYTHONPATH=src python -m credit_risk.train

test:
	PYTHONPATH=src python -m pytest tests/ -q

docker-build:
	docker build -t credit-risk-pipeline .

docker-run:
	docker run --rm \
		-v "$(PWD)/data:/app/data" \
		-v "$(PWD)/reports:/app/reports" \
		-v "$(PWD)/models:/app/models" \
		credit-risk-pipeline

clean:
	rm -rf reports/figures/*.png reports/*.md reports/*.json models/*.joblib
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
