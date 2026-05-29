FROM python:3.11-slim

# Auto-detect network: TCP-connect to deb.debian.org:80 with 2s timeout.
# If slow/unreachable, switch to Tsinghua mirror for apt and pip.
# Override with: --build-arg MIRROR_CHECK=skip  (force default mirrors)
#                --build-arg MIRROR_CHECK=china  (force China mirrors)
ARG MIRROR_CHECK=auto
RUN USE_CHINA=0 && \
    if [ "${MIRROR_CHECK}" = "china" ]; then \
        USE_CHINA=1 && echo "==> Forced China mirrors"; \
    elif [ "${MIRROR_CHECK}" = "auto" ]; then \
        RTT=$(python3 -c " \
import socket, time; \
t=time.time(); \
s=socket.socket(); \
s.settimeout(2); \
ok=0; \
try: s.connect(('deb.debian.org', 80)); ok=1 \
except: pass; \
s.close(); \
print(int((time.time()-t)*1000) if ok else 9999)" 2>/dev/null || echo 9999) && \
        echo "==> deb.debian.org TCP connect: ${RTT}ms" && \
        if [ "${RTT}" -gt 200 ]; then USE_CHINA=1; fi; \
    fi && \
    if [ "${USE_CHINA}" = "1" ]; then \
        echo "==> Using Tsinghua mirrors for apt + pip" && \
        sed -i \
            's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g; \
             s|http://security.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' \
            /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
        sed -i \
            's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g; \
             s|http://security.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' \
            /etc/apt/sources.list 2>/dev/null || true; \
    else \
        echo "==> Using default mirrors"; \
    fi && \
    echo "${USE_CHINA}" > /tmp/use_china

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget curl git build-essential ca-certificates numactl && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Download llama.cpp binary — BEFORE copying source so this layer is cached
# independently of code changes. The layer only rebuilds when LLAMACPP_RELEASE_URL
# changes (not on source edits), so repeated make build-backend is fast.
# ---------------------------------------------------------------------------
ARG LLAMACPP_RELEASE_URL=https://github.com/ggml-org/llama.cpp/releases/download/b9371/llama-b9371-bin-ubuntu-x64.tar.gz
RUN FILENAME=$(basename "${LLAMACPP_RELEASE_URL}") && \
    echo "==> Downloading ${FILENAME}..." && \
    wget -q -O "/tmp/${FILENAME}" "${LLAMACPP_RELEASE_URL}" && \
    mkdir -p /data/llamacpp/.llamacpp/bin && \
    tar -xzf "/tmp/${FILENAME}" -C /data/llamacpp/.llamacpp/bin && \
    FIRST=$(ls /data/llamacpp/.llamacpp/bin | head -1) && \
    if [ -d "/data/llamacpp/.llamacpp/bin/${FIRST}" ]; then \
        mv /data/llamacpp/.llamacpp/bin/${FIRST}/* /data/llamacpp/.llamacpp/bin/ && \
        rmdir /data/llamacpp/.llamacpp/bin/${FIRST}; \
    fi && \
    chmod +x /data/llamacpp/.llamacpp/bin/llama-* && \
    rm "/tmp/${FILENAME}" && \
    echo "==> Installed: $(ls /data/llamacpp/.llamacpp/bin/ | tr '\n' ' ')"

WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies — layer cached until pyproject.toml changes.
# Use China pip mirror when CHINA=1.
# ---------------------------------------------------------------------------
COPY pyproject.toml ./
RUN mkdir -p src/llamacpp_cli && touch src/llamacpp_cli/__init__.py && \
    if [ "$(cat /tmp/use_china 2>/dev/null)" = "1" ]; then \
        pip install --no-cache-dir -e . \
            -i https://pypi.tuna.tsinghua.edu.cn/simple \
            --trusted-host pypi.tuna.tsinghua.edu.cn; \
    else \
        pip install --no-cache-dir -e .; \
    fi

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
