# ═══════════════════════════════════════════════════════════════
#  AirFair Vista — Makefile
#  Usage: make <target>
#
#  Targets:
#    install       Install all Python dependencies
#    train         Run the full training pipeline
#    train-large   Train on 500k rows
#    app           Launch the Streamlit app locally
#    test          Run all pytest tests
#    test-fast     Run only unit tests (skip pipeline integration test)
#    docker-build  Build the Docker image
#    docker-run    Run the container on port 8501
#    docker-up     docker-compose up --build (recommended)
#    docker-down   docker-compose down
#    clean         Remove generated data, model artefacts, and caches
#    clean-models  Remove only model artefacts (keep data)
#    lint          Run flake8 linter on src/
#    help          Show this help message
# ═══════════════════════════════════════════════════════════════

.PHONY: install train train-large app test test-fast \
        docker-build docker-run docker-up docker-down \
        clean clean-models lint help

# ── Python / pip ──────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Training ──────────────────────────────────────────────────
train:
	python train_model.py

train-large:
	python train_model.py --rows 500000

train-force:
	python train_model.py --force

# ── App ───────────────────────────────────────────────────────
app:
	streamlit run src/frontend/app.py

# ── Tests ─────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/test_generator.py tests/test_features.py tests/test_preprocessor.py \
	       -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ── Docker ────────────────────────────────────────────────────
docker-build:
	docker build -t airfair-vista:latest .

docker-run:
	docker run -p 8501:8501 \
	    -v $(shell pwd)/models:/app/models \
	    -v $(shell pwd)/data:/app/data \
	    -v $(shell pwd)/logs:/app/logs \
	    airfair-vista:latest

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

# ── Clean ─────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f data/flight_price_synthetic.csv
	rm -f data/flight_price_combined.csv
	rm -f models/model.pkl models/encoders.pkl models/features.pkl models/model_meta.json
	rm -f logs/training.log

clean-models:
	rm -f models/model.pkl models/encoders.pkl models/features.pkl models/model_meta.json

# ── Lint ──────────────────────────────────────────────────────
lint:
	flake8 src/ train_model.py --max-line-length=120 --ignore=E501,W503

# ── Help ──────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  AirFair Vista — available make targets"
	@echo "  ────────────────────────────────────────"
	@echo "  install       Install Python dependencies"
	@echo "  train         Train model (100k rows default)"
	@echo "  train-large   Train model (500k rows)"
	@echo "  train-force   Force re-generate data and retrain"
	@echo "  app           Launch Streamlit app locally"
	@echo "  test          Run all tests"
	@echo "  test-fast     Run unit tests only (skip integration)"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run Docker container"
	@echo "  docker-up     docker-compose up --build"
	@echo "  docker-down   docker-compose down"
	@echo "  clean         Remove all generated files"
	@echo "  clean-models  Remove model artefacts only"
	@echo "  lint          Run flake8 linter"
	@echo ""
