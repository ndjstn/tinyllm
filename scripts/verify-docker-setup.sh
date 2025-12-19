#!/bin/bash
# TinyLLM Docker Setup Verification Script
# Checks that all components are properly configured before deployment

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================"
echo "TinyLLM Docker Setup Verification"
echo "============================================"
echo ""

# Check if running from project root
if [ ! -f "Dockerfile" ] || [ ! -f "docker-compose.yaml" ]; then
    echo -e "${RED}ERROR: Please run this script from the TinyLLM project root${NC}"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print check result
print_check() {
    local name="$1"
    local status="$2"
    local message="$3"

    if [ "$status" = "ok" ]; then
        echo -e "${GREEN}✓${NC} $name"
    elif [ "$status" = "warn" ]; then
        echo -e "${YELLOW}⚠${NC} $name: $message"
    else
        echo -e "${RED}✗${NC} $name: $message"
    fi
}

# Check Docker
echo "Checking prerequisites..."
if command_exists docker; then
    docker_version=$(docker --version | awk '{print $3}' | sed 's/,//')
    print_check "Docker installed" "ok" "$docker_version"
else
    print_check "Docker installed" "error" "Docker not found. Install from https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
if command_exists docker-compose || docker compose version >/dev/null 2>&1; then
    if command_exists docker-compose; then
        compose_version=$(docker-compose --version | awk '{print $4}' | sed 's/,//')
    else
        compose_version=$(docker compose version --short)
    fi
    print_check "Docker Compose installed" "ok" "$compose_version"
else
    print_check "Docker Compose installed" "error" "Docker Compose not found"
    exit 1
fi

# Check Docker daemon
if docker info >/dev/null 2>&1; then
    print_check "Docker daemon running" "ok"
else
    print_check "Docker daemon running" "error" "Docker daemon not running. Start Docker first."
    exit 1
fi

echo ""
echo "Checking Docker resources..."

# Check available memory
total_mem=$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo 0)
if [ "$total_mem" -gt 0 ]; then
    mem_gb=$((total_mem / 1024 / 1024 / 1024))
    if [ "$mem_gb" -ge 16 ]; then
        print_check "Docker memory (${mem_gb}GB)" "ok"
    else
        print_check "Docker memory (${mem_gb}GB)" "warn" "16GB+ recommended"
    fi
fi

# Check disk space
available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$available_space" -ge 50 ]; then
    print_check "Disk space (${available_space}GB available)" "ok"
else
    print_check "Disk space (${available_space}GB available)" "warn" "50GB+ recommended"
fi

echo ""
echo "Checking required files..."

# Check required files
files=(
    "Dockerfile"
    "docker-compose.yaml"
    ".dockerignore"
    ".env.example"
    "scripts/docker-entrypoint.sh"
    "docker/prometheus/prometheus.yml"
    "docker/grafana/provisioning/datasources/prometheus.yml"
    "docker/grafana/provisioning/dashboards/dashboard.yml"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        print_check "$file" "ok"
    else
        print_check "$file" "error" "File not found"
        exit 1
    fi
done

# Check if .env exists
echo ""
echo "Checking configuration..."
if [ -f ".env" ]; then
    print_check ".env file exists" "ok"
else
    print_check ".env file exists" "warn" "Run 'cp .env.example .env' to create it"
fi

# Check if entrypoint is executable
if [ -x "scripts/docker-entrypoint.sh" ]; then
    print_check "docker-entrypoint.sh executable" "ok"
else
    print_check "docker-entrypoint.sh executable" "warn" "Run 'chmod +x scripts/docker-entrypoint.sh'"
    chmod +x scripts/docker-entrypoint.sh 2>/dev/null || true
fi

# Check for GPU support (optional)
echo ""
echo "Checking optional features..."
if command_exists nvidia-smi; then
    if docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
        print_check "NVIDIA GPU support" "ok" "GPU available"
    else
        print_check "NVIDIA GPU support" "warn" "GPU detected but Docker GPU support not configured"
    fi
else
    print_check "NVIDIA GPU support" "warn" "No GPU detected (optional)"
fi

# Check port availability
echo ""
echo "Checking port availability..."
ports=(8000 9090 9091 3000 6379 11434)
port_names=("TinyLLM" "Metrics" "Prometheus" "Grafana" "Redis" "Ollama")

for i in "${!ports[@]}"; do
    port="${ports[$i]}"
    name="${port_names[$i]}"

    if ! nc -z localhost "$port" 2>/dev/null; then
        print_check "Port $port ($name)" "ok" "Available"
    else
        print_check "Port $port ($name)" "warn" "Port in use"
    fi
done

# Summary
echo ""
echo "============================================"
echo "Verification Summary"
echo "============================================"
echo ""

# Check if Docker Compose file is valid
if docker-compose config >/dev/null 2>&1 || docker compose config >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Docker configuration is valid${NC}"
else
    echo -e "${RED}✗ Docker configuration has errors${NC}"
    exit 1
fi

echo ""
echo "Next steps:"
echo "  1. Review and customize .env file"
echo "  2. Run: make docker-up"
echo "  3. Pull models: make docker-pull-models"
echo "  4. Check health: make docker-health"
echo ""
echo "For more information, see DOCKER_QUICKSTART.md"
echo ""
