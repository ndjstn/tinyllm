# Multi-stage Dockerfile for TinyLLM
# Production-ready with security best practices

# Stage 1: Builder - Install dependencies
FROM python:3.11-slim AS builder

# Install system dependencies and uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add uv to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Create a minimal project structure for dependency installation
RUN mkdir -p src/tinyllm && \
    touch src/tinyllm/__init__.py

# Install dependencies using uv (fast!)
RUN uv pip install --system --no-cache-dir \
    pydantic>=2.0.0 \
    pydantic-settings>=2.0.0 \
    httpx>=0.25.0 \
    aiohttp>=3.9.0 \
    aiosqlite>=0.19.0 \
    langgraph>=0.2.0 \
    langchain-core>=0.3.0 \
    pyyaml>=6.0.0 \
    rich>=13.0.0 \
    typer>=0.12.0 \
    structlog>=24.0.0 \
    redis>=5.0.0 \
    prometheus-client>=0.19.0 \
    textual>=0.89.0

# Stage 2: Runtime - Create minimal production image
FROM python:3.11-slim AS runtime

# Install runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r tinyllm && \
    useradd -r -g tinyllm -u 1000 -m -s /bin/bash tinyllm

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=tinyllm:tinyllm src/ ./src/
COPY --chown=tinyllm:tinyllm graphs/ ./graphs/
COPY --chown=tinyllm:tinyllm prompts/ ./prompts/
COPY --chown=tinyllm:tinyllm pyproject.toml ./
COPY --chown=tinyllm:tinyllm README.md ./

# Install the package
RUN pip install --no-cache-dir -e .

# Copy entrypoint script
COPY --chown=tinyllm:tinyllm scripts/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/config /app/.tinyllm && \
    chown -R tinyllm:tinyllm /app

# Switch to non-root user
USER tinyllm

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TINYLLM_DATA_DIR=/app/data \
    TINYLLM_CONFIG_DIR=/app/config \
    TINYLLM_LOG_LEVEL=INFO \
    TINYLLM_LOG_FORMAT=json

# Expose ports
# 8000: Default application port (can be used for API if implemented)
# 9090: Prometheus metrics port
EXPOSE 8000 9090

# Health check using the CLI health command
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD tinyllm health --json || exit 1

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command - can be overridden
CMD ["tinyllm", "doctor"]
