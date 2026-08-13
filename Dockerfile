# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY requirements-build.lock ./
RUN python -m pip install --no-cache-dir --require-hashes -r requirements-build.lock

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip wheel --no-cache-dir --no-deps --no-build-isolation --wheel-dir /wheels .

FROM python:3.12-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PINGPONG_DATA_DIR=/data \
    PINGPONG_HOST=0.0.0.0 \
    PINGPONG_PORT=8000

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.lock ./
RUN python -m pip install --no-cache-dir --require-hashes -r requirements.lock

COPY --from=builder /wheels/*.whl /tmp/wheels/
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels \
    && groupadd --gid 10001 pingpong \
    && useradd --uid 10001 --gid 10001 --system --no-create-home pingpong \
    && mkdir -p /data \
    && chown pingpong:pingpong /data

ARG APP_VERSION=1.2.3
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="Ping-Pong Auto Highlight" \
      org.opencontainers.image.description="Local-first table-tennis point highlight reels with NVIDIA NVDEC/NVENC support" \
      org.opencontainers.image.source="https://github.com/momonong/pingpong-auto-highlight" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

USER pingpong

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"]

CMD ["pingpong-highlight", "serve"]
