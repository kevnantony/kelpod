# syntax=docker/dockerfile:1.7
# ZLLF CPU image: api + worker (+ `zllf run`) with the FAKE / CPU hardware profiles.
#
#   docker build -f deploy/docker/Dockerfile.app -t zllf-app:dev .        # from the repo root
#   docker run --rm --network none --read-only --tmpfs /tmp zllf-app:dev --help
#
# Build-time network is allowed (build host). Run-time is offline: Appendix A env vars below,
# no package index (PIP_NO_INDEX), models only from the read-only /data/models mount.
# CPU-only torch: see deploy/scripts/install_cpu_env.sh.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.12
ARG NODE_IMAGE=node:22-bookworm-slim
# web-build: build the UI from web/ (default). web-prebuilt: copy an existing web/dist from the
# build context (faster; `cd web && npm run build` first).
ARG WEB_STAGE=web-build

FROM ${UV_IMAGE} AS uv

# --------------------------------------------------------------------------- web UI
FROM ${NODE_IMAGE} AS web-build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build && mkdir -p /out && cp -r dist /out/dist

FROM scratch AS web-prebuilt
COPY web/dist /out/dist

FROM ${WEB_STAGE} AS web

# --------------------------------------------------------------------------- builder
FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12
WORKDIR /app

# 1) third-party deps only (cached layer; changes only with uv.lock)
COPY pyproject.toml uv.lock README.md ./
COPY deploy/scripts/install_cpu_env.sh /usr/local/bin/install_cpu_env.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    install_cpu_env.sh --project /app --venv /opt/venv --no-project-install

# 2) the project itself (editable: zllf.core.config resolves configs/ relative to src/)
COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps -e /app \
 && /opt/venv/bin/python -m compileall -q /app/src

# --------------------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="zllf-app" \
      org.opencontainers.image.description="Zero-Leak LoRA Factory - CPU api/worker image" \
      org.opencontainers.image.source="kelpod"

# Appendix A (DEVELOPMENT_PLAN) offline/telemetry env + read-only-root friendliness.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_DO_NOT_TRACK=1 \
    DO_NOT_TRACK=1 \
    RAY_USAGE_STATS_ENABLED=0 \
    WANDB_MODE=disabled \
    WANDB_DISABLED=true \
    GRADIO_ANALYTICS_ENABLED=False \
    TOKENIZERS_PARALLELISM=false \
    DOCLING_DISABLE_TELEMETRY=1 \
    PIP_NO_INDEX=1 \
    UV_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/models \
    HF_HUB_CACHE=/data/models/hub \
    ZLLF_DATA_DIR=/data \
    ZLLF_MODELS_DIR=/data/models/zllf \
    ZLLF_DOCLING_ARTIFACTS_PATH=/data/models/docling \
    HOME=/tmp/home \
    XDG_CACHE_HOME=/tmp/cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    PATH=/opt/venv/bin:$PATH

RUN groupadd --system --gid 10001 zllf \
 && useradd --system --uid 10001 --gid zllf --home-dir /nonexistent --shell /usr/sbin/nologin zllf \
 && mkdir -p /data/models /data/runs \
 && chown -R zllf:zllf /data

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app
COPY --from=web /out/dist /app/web/dist
# Small synthetic fixture corpus so the egress self-test runs inside the image.
COPY tests/fixtures /app/tests/fixtures
WORKDIR /app

USER 10001:10001
VOLUME ["/data"]
EXPOSE 8000
# API health endpoint (zllf.api.routes.system); probe is stdlib-only.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/api/system/health', timeout=4); sys.exit(0)"]
ENTRYPOINT ["zllf"]
CMD ["--help"]
