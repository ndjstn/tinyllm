#!/bin/bash
set -e

# TinyLLM Docker Entrypoint Script
# Handles initialization, dependency checks, and service startup

echo "==> Starting TinyLLM container..."

# Environment variable defaults
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
METRICS_ENABLED="${METRICS_ENABLED:-true}"
METRICS_PORT="${METRICS_PORT:-9090}"
TINYLLM_LOG_LEVEL="${TINYLLM_LOG_LEVEL:-INFO}"
TINYLLM_LOG_FORMAT="${TINYLLM_LOG_FORMAT:-json}"

# Function to wait for a service to be ready
wait_for_service() {
    local service_name=$1
    local service_host=$2
    local service_port=$3
    local max_attempts=${4:-30}
    local attempt=1

    echo "==> Waiting for $service_name at $service_host:$service_port..."

    while [ $attempt -le $max_attempts ]; do
        if nc -z "$service_host" "$service_port" 2>/dev/null; then
            echo "==> $service_name is ready!"
            return 0
        fi

        echo "    Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "ERROR: $service_name failed to become ready after $max_attempts attempts"
    return 1
}

# Function to wait for Ollama API
wait_for_ollama() {
    local max_attempts=${1:-30}
    local attempt=1
    local ollama_url="${OLLAMA_HOST}/api/tags"

    echo "==> Waiting for Ollama API at $ollama_url..."

    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$ollama_url" >/dev/null 2>&1; then
            echo "==> Ollama API is ready!"
            return 0
        fi

        echo "    Attempt $attempt/$max_attempts: Ollama API not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "WARNING: Ollama API is not responding. The application may not work correctly."
    return 1
}

# Function to start metrics server
start_metrics_server() {
    if [ "$METRICS_ENABLED" = "true" ]; then
        echo "==> Metrics collection enabled on port $METRICS_PORT"
        # The metrics server is started by the application itself via prometheus_client
        # We just export the configuration here
        export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
        mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
    else
        echo "==> Metrics collection disabled"
    fi
}

# Function to initialize configuration
initialize_config() {
    if [ ! -f "$TINYLLM_CONFIG_DIR/tinyllm.yaml" ]; then
        echo "==> Initializing default configuration..."
        tinyllm init --config "$TINYLLM_CONFIG_DIR"
    else
        echo "==> Using existing configuration at $TINYLLM_CONFIG_DIR"
    fi
}

# Parse OLLAMA_HOST to extract hostname and port
if [[ "$OLLAMA_HOST" =~ ^https?://([^:]+):?([0-9]+)?$ ]]; then
    OLLAMA_HOSTNAME="${BASH_REMATCH[1]}"
    OLLAMA_PORT="${BASH_REMATCH[2]:-11434}"
else
    OLLAMA_HOSTNAME="localhost"
    OLLAMA_PORT="11434"
fi

# Wait for dependencies (only if not localhost - in docker-compose)
if [ "$OLLAMA_HOSTNAME" != "localhost" ] && [ "$OLLAMA_HOSTNAME" != "127.0.0.1" ]; then
    # We're in a container orchestration setup
    echo "==> Detected orchestrated environment"

    # Wait for Ollama
    wait_for_service "Ollama" "$OLLAMA_HOSTNAME" "$OLLAMA_PORT" 60
    wait_for_ollama 30 || echo "WARNING: Continuing without Ollama connection"

    # Wait for Redis (if configured)
    if [ -n "$REDIS_HOST" ] && [ "$REDIS_HOST" != "localhost" ] && [ "$REDIS_HOST" != "127.0.0.1" ]; then
        wait_for_service "Redis" "$REDIS_HOST" "$REDIS_PORT" 30 || echo "WARNING: Continuing without Redis connection"
    fi
else
    echo "==> Standalone mode (localhost dependencies)"
fi

# Initialize configuration if needed
initialize_config

# Start metrics server configuration
start_metrics_server

# Health check before starting
echo "==> Running health check..."
if tinyllm health --json; then
    echo "==> Health check passed!"
else
    echo "WARNING: Health check failed, but continuing..."
fi

# Log startup information
echo "==> TinyLLM Configuration:"
echo "    Log Level: $TINYLLM_LOG_LEVEL"
echo "    Log Format: $TINYLLM_LOG_FORMAT"
echo "    Ollama Host: $OLLAMA_HOST"
echo "    Redis: $REDIS_HOST:$REDIS_PORT"
echo "    Metrics Enabled: $METRICS_ENABLED"
echo "    Data Directory: $TINYLLM_DATA_DIR"
echo "    Config Directory: $TINYLLM_CONFIG_DIR"
echo ""

# Execute the main command
echo "==> Starting: $*"
echo ""

exec "$@"
