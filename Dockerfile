# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

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

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir . \
    && groupadd --gid 10001 pingpong \
    && useradd --uid 10001 --gid 10001 --system --no-create-home pingpong \
    && mkdir -p /data \
    && chown pingpong:pingpong /data

USER pingpong

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"]

CMD ["pingpong-highlight", "serve"]
