FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        curl \
        git \
        build-essential \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ src/

# Install Python package (README.md is optional for pip install -e)
RUN pip install --no-cache-dir -e .

# Create llamacpp home directory
RUN mkdir -p /data/llamacpp/.llamacpp/models \
             /data/llamacpp/.llamacpp/bin

# Set environment variables
ENV LLAMACPP_HOME=/data/llamacpp/.llamacpp
ENV PATH="${LLAMACPP_HOME}/bin:${PATH}"
ENV LLAMACPP_AUTO_INSTALL=true

# Pre-install llama.cpp during build to avoid GitHub API rate limits at runtime
# This downloads the llama.cpp binary from GitHub releases
RUN llamacpp install || echo "llama.cpp install failed - will retry at runtime"

# Expose ports
# 8000: llama.cpp server
# 8080: load balancer proxy
EXPOSE 8000 8080

# Default command (can be overridden in docker-compose)
CMD ["llamacpp", "--help"]
