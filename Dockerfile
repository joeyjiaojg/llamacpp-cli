# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget curl git build-essential ca-certificates numactl && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Download llama.cpp binary — BEFORE copying source so this layer is cached
# independently of code changes.
#
# Two-level caching:
#   1. Docker layer cache: layer only rebuilds when ARG changes (URL change)
#   2. BuildKit cache mount (/var/cache/llamacpp): tarball persists on the
#      host between builds so even --no-cache skips the GitHub download.
# ---------------------------------------------------------------------------
ARG LLAMACPP_RELEASE_URL=https://github.com/ggml-org/llama.cpp/releases/download/b9371/llama-b9371-bin-ubuntu-x64.tar.gz
RUN --mount=type=cache,target=/var/cache/llamacpp \
    FILENAME=$(basename "${LLAMACPP_RELEASE_URL}") && \
    CACHED="/var/cache/llamacpp/${FILENAME}" && \
    if [ ! -f "${CACHED}" ]; then \
        echo "==> Downloading ${FILENAME}..." && \
        wget -q -O "${CACHED}.tmp" "${LLAMACPP_RELEASE_URL}" && \
        mv "${CACHED}.tmp" "${CACHED}"; \
    else \
        echo "==> Cache hit: ${FILENAME} ($(du -sh ${CACHED} | cut -f1))"; \
    fi && \
    mkdir -p /data/llamacpp/.llamacpp/bin && \
    tar -xzf "${CACHED}" -C /data/llamacpp/.llamacpp/bin && \
    FIRST=$(ls /data/llamacpp/.llamacpp/bin | head -1) && \
    if [ -d "/data/llamacpp/.llamacpp/bin/${FIRST}" ]; then \
        mv /data/llamacpp/.llamacpp/bin/${FIRST}/* /data/llamacpp/.llamacpp/bin/ && \
        rmdir /data/llamacpp/.llamacpp/bin/${FIRST}; \
    fi && \
    chmod +x /data/llamacpp/.llamacpp/bin/llama-* && \
    echo "==> Installed: $(ls /data/llamacpp/.llamacpp/bin/ | tr '\n' ' ')"

WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies — layer cached until pyproject.toml changes.
# Copy only pyproject.toml (no src/) so code edits don't bust this layer.
# ---------------------------------------------------------------------------
COPY pyproject.toml ./
# Dummy src/ structure so pip install -e . resolves the package metadata
RUN mkdir -p src/llamacpp_cli && touch src/llamacpp_cli/__init__.py && \
    pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Copy actual source — only layers below here rebuild on code changes.
# ---------------------------------------------------------------------------
COPY src/ src/
RUN pip install --no-cache-dir --no-deps -e .

RUN mkdir -p /data/llamacpp/.llamacpp/models

ENV LLAMACPP_HOME=/data/llamacpp/.llamacpp
ENV PATH="${LLAMACPP_HOME}/bin:${PATH}"
ENV LLAMACPP_AUTO_INSTALL=false
ENV LLAMACPP_RELEASE_URL=${LLAMACPP_RELEASE_URL}
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["llamacpp", "--help"]
