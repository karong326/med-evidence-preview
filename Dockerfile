ARG FRONTEND_IMAGE=node:22-alpine@sha256:1314fb30ec37592d5845e5a160d522c0e680972aa534549940b8d73bde88481a
ARG PYTHON_IMAGE=python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
FROM ${FRONTEND_IMAGE} AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/index.html frontend/vite.config.js ./
COPY frontend/src ./src
COPY frontend/public ./public
RUN npm run build

FROM ${PYTHON_IMAGE}
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    XDG_CACHE_HOME=/tmp/cache HOME=/tmp \
    APP_HOST=0.0.0.0 APP_PORT=8001 DATA_ROOT=/state
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 postgresql-client-17 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && install -d -o 10001 -g 10001 /state /state/raw /state/uploads /state/chroma /state/prompts
COPY pyproject.toml README.md alembic.ini ./
COPY src ./src
COPY migrations ./migrations
COPY scripts/check_environment.py scripts/maintenance.py ./scripts/
COPY --from=frontend-build /build/dist ./frontend/dist
RUN python -m pip install --no-cache-dir --timeout 120 torch==2.6.0 \
      --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install --no-cache-dir . \
    && python -m pip check
ENV PYTHONPATH=/app/src
USER 10001:10001
EXPOSE 8001
CMD ["python", "-m", "med_evidence_agentic_rag.main"]
