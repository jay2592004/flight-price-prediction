# ═══════════════════════════════════════════════════════════════
#  AirFair Vista — Production Dockerfile
#  Multi-stage build:
#    Stage 1 (builder) — install Python deps into a venv
#    Stage 2 (runtime) — copy venv + source, run training if
#                        models/ is empty, then serve Streamlit
# ═══════════════════════════════════════════════════════════════

# ── Stage 1: builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed by some packages (lightgbm, shap)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies into a virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

# libgomp1 required at runtime by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application source (data CSV copied separately via volume or COPY)
COPY src/           ./src/
COPY train_model.py ./train_model.py
COPY requirements.txt ./requirements.txt
COPY .streamlit/    ./.streamlit/

# Create required directories
RUN mkdir -p models data logs

# Copy dataset if present (can also be mounted via docker volume)
COPY data/ ./data/

# Expose Streamlit port
EXPOSE 8501

# Healthcheck — Streamlit responds on /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Entrypoint:
#   1. Train the model if models/model.pkl does not exist
#   2. Start Streamlit
CMD ["sh", "-c", \
     "[ ! -f models/model.pkl ] && echo 'Training model...' && python train_model.py; \
      echo 'Starting Streamlit...'; \
      streamlit run src/frontend/app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false"]
