# ═══════════════════════════════════════════════════════════════
#  AirFair Vista — Root Makefile
#  All commands run from airfair_v3/
# ═══════════════════════════════════════════════════════════════

.PHONY: install-ml install-app install train train-large train-force \
        app up down build logs clean clean-models help

# ── Install ───────────────────────────────────────────────────
install-ml:
	cd ml_pipeline && pip install -r requirements.txt

install-app:
	cd streamlit_app && pip install -r requirements.txt

install: install-ml install-app

# ── Training ──────────────────────────────────────────────────
train:
	cd ml_pipeline && python train.py

train-large:
	cd ml_pipeline && python train.py --rows 500000

train-force:
	cd ml_pipeline && python train.py --force

train-skip-eda:
	cd ml_pipeline && python train.py --skip-eda

# ── App ───────────────────────────────────────────────────────
app:
	cd streamlit_app && streamlit run app.py

# ── Docker ────────────────────────────────────────────────────
build:
	docker-compose build

up:
	docker-compose up --build

up-detached:
	docker-compose up --build -d

down:
	docker-compose down

logs:
	docker-compose logs -f

logs-ml:
	docker-compose logs -f ml_pipeline

logs-app:
	docker-compose logs -f streamlit_app

retrain:
	docker-compose run --rm ml_pipeline python train.py --force
	docker-compose restart streamlit_app

# ── Clean ─────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f ml_pipeline/data/processed/*.csv
	rm -f ml_pipeline/models/model.pkl
	rm -f ml_pipeline/models/encoders.pkl
	rm -f ml_pipeline/models/features.pkl
	rm -f ml_pipeline/models/model_meta.json
	rm -f ml_pipeline/reports/*.png
	rm -f ml_pipeline/reports/insights.json
	rm -f ml_pipeline/logs/*.log

clean-models:
	rm -f ml_pipeline/models/model.pkl
	rm -f ml_pipeline/models/encoders.pkl
	rm -f ml_pipeline/models/features.pkl
	rm -f ml_pipeline/models/model_meta.json

# ── Help ──────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  AirFair Vista — available commands"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  make install        Install all Python dependencies"
	@echo "  make install-ml     Install ML pipeline deps only"
	@echo "  make install-app    Install Streamlit app deps only"
	@echo ""
	@echo "  make train          Train model (100k rows)"
	@echo "  make train-large    Train model (500k rows)"
	@echo "  make train-force    Force re-generate data + retrain"
	@echo "  make train-skip-eda Train without re-running EDA"
	@echo ""
	@echo "  make app            Launch Streamlit app locally"
	@echo ""
	@echo "  make up             Docker: build + run full stack"
	@echo "  make up-detached    Docker: run in background"
	@echo "  make down           Docker: stop all containers"
	@echo "  make logs           Docker: stream all logs"
	@echo "  make retrain        Docker: retrain + restart app"
	@echo ""
	@echo "  make clean          Remove all generated files"
	@echo "  make clean-models   Remove model artefacts only"
	@echo ""
