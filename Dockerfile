# Production Dockerfile for OTA Machine Learning Service
FROM python:3.12-slim AS base

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code and assets
COPY src/ ./src/
COPY static/ ./static/
COPY presentation/ ./presentation/

# Generate data and train ML models
RUN python -m src.data.generator && \
    python -m src.models.train_ranking && \
    python -m src.models.recommender

# Expose port (default 8000)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1

# Start production uvicorn server (supports dynamic cloud PORT)
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
