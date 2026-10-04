# syntax=docker/dockerfile:1
#
# Kautilya - offline forensic analysis platform (Linux).
#
# Build needs network access once (pip + npm). The running container makes NO
# outbound connections: all models, reference lists and the UI are baked in or
# mounted. Verify with:  docker run --rm --network none kautilya python scripts/verify_offline.py
#
#   docker build -t kautilya .
#   docker run --rm -p 8000:8000 -v "$PWD/data:/app/data:ro" -v kautilya-state:/app/state kautilya

# ---- 1. Build the React UI ------------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. Runtime -----------------------------------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KAUTILYA_HOST=0.0.0.0 \
    KAUTILYA_PORT=8000 \
    KAUTILYA_DATA_DIR=/app/data \
    KAUTILYA_GEOIP_DIR=/app/geoip \
    KAUTILYA_CASE_DB=/app/state/cases.db

WORKDIR /app
COPY requirements-pinned.txt ./
RUN pip install --no-cache-dir -r requirements-pinned.txt

COPY pyproject.toml README.md ./
COPY backend ./backend
COPY scripts ./scripts
COPY reports ./reports
COPY --from=frontend /build/dist ./frontend/dist

# Non-root user; data is mounted read-only, state (case DB) is a writable volume.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/state /app/geoip \
    && chown -R app:app /app/state
USER app

VOLUME ["/app/state"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import os,sys,urllib.request as u; p=os.environ.get('PORT') or os.environ.get('KAUTILYA_PORT') or '8000'; sys.exit(0 if u.urlopen(f'http://127.0.0.1:{p}/health', timeout=3).status == 200 else 1)"

CMD ["python", "-m", "backend.main"]
