# TinyLLM Docker Deployment Guide

This guide covers running TinyLLM in Docker containers for development and production deployments.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Services](#services)
- [Usage](#usage)
- [Production Deployment](#production-deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Security Best Practices](#security-best-practices)

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 16GB RAM minimum (32GB+ recommended)
- 50GB disk space
- Optional: NVIDIA GPU with Docker GPU support for accelerated inference

### Basic Setup

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your preferences (optional):
```bash
# Review and customize settings
nano .env
```

3. Start all services:
```bash
docker-compose up -d
```

4. Check service health:
```bash
docker-compose ps
docker-compose logs -f tinyllm
```

5. Pull Ollama models (required):
```bash
docker-compose exec ollama ollama pull qwen2.5:0.5b
docker-compose exec ollama ollama pull qwen2.5:3b
docker-compose exec ollama ollama pull granite-code:3b
```

6. Verify TinyLLM health:
```bash
docker-compose exec tinyllm tinyllm health
```

## Architecture

The Docker Compose stack includes:

```
┌─────────────────────────────────────────────────────────────┐
│                     TinyLLM Stack                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐     ┌──────────────┐                      │
│  │   TinyLLM    │────▶│    Ollama    │  LLM Backend         │
│  │     App      │     │  (LLMs)      │                      │
│  └──────┬───────┘     └──────────────┘                      │
│         │                                                    │
│         ├──────────────┐                                     │
│         ▼              ▼                                     │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │    Redis     │  │  Prometheus  │  Cache & Metrics       │
│  │   (Cache)    │  │  (Metrics)   │                        │
│  └──────────────┘  └──────┬───────┘                        │
│                            │                                 │
│                            ▼                                 │
│                     ┌──────────────┐                        │
│                     │   Grafana    │  Visualization         │
│                     │ (Dashboard)  │                        │
│                     └──────────────┘                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Network Layout

All services run on an isolated bridge network (`tinyllm-network`):
- Internal DNS resolution between containers
- No external exposure except mapped ports
- Secure inter-service communication

### Port Mappings

| Service | Container Port | Host Port | Purpose |
|---------|---------------|-----------|---------|
| TinyLLM | 8000 | 8000 | Application API |
| TinyLLM | 9090 | 9090 | Prometheus metrics |
| Ollama | 11434 | 11434 | Ollama API |
| Redis | 6379 | 6379 | Redis cache |
| Prometheus | 9090 | 9091 | Prometheus UI |
| Grafana | 3000 | 3000 | Grafana dashboard |

## Configuration

### Environment Variables

Key configuration options in `.env`:

#### Application
```bash
TINYLLM_LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
TINYLLM_LOG_FORMAT=json         # json, console
METRICS_ENABLED=true            # Enable Prometheus metrics
```

#### Ollama
```bash
OLLAMA_HOST=http://ollama:11434
OLLAMA_TIMEOUT_MS=30000
OLLAMA_MAX_RETRIES=3
```

#### Redis
```bash
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=                 # Set in production!
```

#### Monitoring
```bash
GRAFANA_USER=admin              # Change in production!
GRAFANA_PASSWORD=admin          # Change in production!
```

### Volume Mounts

Persistent data is stored in named volumes:

```bash
# View volumes
docker volume ls | grep tinyllm

# Inspect a volume
docker volume inspect tinyllm-data
```

| Volume | Purpose | Backup Priority |
|--------|---------|----------------|
| `tinyllm-data` | Application data | High |
| `tinyllm-config` | Configuration files | High |
| `tinyllm-versions` | Graph versions | Medium |
| `ollama-models` | Downloaded models | Medium (large) |
| `redis-data` | Cache data | Low |
| `prometheus-data` | Metrics (7d retention) | Low |
| `grafana-data` | Dashboards | Medium |

## Services

### TinyLLM Application

Main application container running the TinyLLM neural network.

```bash
# View logs
docker-compose logs -f tinyllm

# Execute commands
docker-compose exec tinyllm tinyllm doctor
docker-compose exec tinyllm tinyllm run "What is 2+2?"

# Interactive shell
docker-compose exec tinyllm bash
```

### Ollama (LLM Backend)

Serves the language models.

```bash
# List models
docker-compose exec ollama ollama list

# Pull a new model
docker-compose exec ollama ollama pull llama2:7b

# Remove a model
docker-compose exec ollama ollama rm llama2:7b

# Check service
curl http://localhost:11434/api/tags
```

### Redis (Cache)

Provides caching and messaging capabilities.

```bash
# Connect to Redis CLI
docker-compose exec redis redis-cli

# Monitor commands
docker-compose exec redis redis-cli MONITOR

# Check memory usage
docker-compose exec redis redis-cli INFO memory
```

### Prometheus (Metrics)

Collects and stores metrics from TinyLLM.

- Web UI: http://localhost:9091
- Metrics endpoint: http://localhost:9090/metrics

```bash
# Query metrics
curl http://localhost:9090/metrics | grep tinyllm
```

### Grafana (Visualization)

Provides dashboards for monitoring.

- Web UI: http://localhost:3000
- Default credentials: admin/admin (change on first login!)

## Usage

### Running Queries

```bash
# Simple query
docker-compose exec tinyllm tinyllm run "Calculate 15 * 23"

# With trace output
docker-compose exec tinyllm tinyllm run --trace "Write Python code to reverse a string"

# Custom graph
docker-compose exec tinyllm tinyllm run --graph /app/graphs/custom.yaml "Your query"
```

### Interactive Chat

```bash
docker-compose exec tinyllm tinyllm chat --model qwen2.5:3b
```

### Health Checks

```bash
# Check all services
docker-compose ps

# TinyLLM health
docker-compose exec tinyllm tinyllm health

# JSON output
docker-compose exec tinyllm tinyllm health --json
```

### Model Management

```bash
# List available models
docker-compose exec ollama ollama list

# Pull recommended models
docker-compose exec ollama ollama pull qwen2.5:0.5b    # Router
docker-compose exec ollama ollama pull qwen2.5:3b      # Specialist
docker-compose exec ollama ollama pull granite-code:3b # Code
```

### Accessing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f tinyllm
docker-compose logs -f ollama

# Last 100 lines
docker-compose logs --tail=100 tinyllm
```

## Production Deployment

### Security Checklist

- [ ] Change default Grafana credentials
- [ ] Set Redis password
- [ ] Use secrets management (Docker Swarm secrets or Kubernetes)
- [ ] Enable HTTPS/TLS for external endpoints
- [ ] Restrict network access with firewall rules
- [ ] Use read-only root filesystem where possible
- [ ] Implement proper backup strategy
- [ ] Enable audit logging
- [ ] Scan images for vulnerabilities

### Resource Limits

The docker-compose.yaml includes sensible defaults. Adjust based on your workload:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
    reservations:
      cpus: '1'
      memory: 1G
```

### GPU Support

For NVIDIA GPU acceleration:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Uncomment GPU sections in `docker-compose.yaml`:
```yaml
# For Ollama service
runtime: nvidia
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

3. Verify GPU access:
```bash
docker-compose exec ollama nvidia-smi
```

### High Availability

For production HA setup:

1. Use Docker Swarm or Kubernetes
2. Deploy multiple TinyLLM replicas
3. Use external Redis cluster
4. Use persistent storage (NFS, cloud volumes)
5. Set up load balancer
6. Configure health checks and auto-restart

### Backup Strategy

```bash
# Backup volumes
docker run --rm -v tinyllm-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/tinyllm-data-backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v tinyllm-data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/tinyllm-data-backup.tar.gz -C /data
```

## Monitoring

### Prometheus Metrics

Access Prometheus at http://localhost:9091

Key metrics to monitor:
```promql
# Request rate
rate(tinyllm_requests_total[5m])

# Request latency (p95)
histogram_quantile(0.95, rate(tinyllm_request_latency_seconds_bucket[5m]))

# Error rate
rate(tinyllm_errors_total[5m])

# Token usage
rate(tinyllm_tokens_total[5m])
```

### Grafana Dashboards

Access Grafana at http://localhost:3000

Default dashboards include:
- System overview
- Request metrics
- Model performance
- Error tracking

### Health Monitoring

Set up health check endpoints:

```bash
# Readiness probe (for load balancers)
curl http://localhost:8000/health

# Liveness probe
docker-compose exec tinyllm tinyllm health --json
```

## Troubleshooting

### Common Issues

#### 1. Ollama not responding

```bash
# Check Ollama logs
docker-compose logs ollama

# Verify Ollama is running
docker-compose ps ollama

# Test Ollama API
curl http://localhost:11434/api/tags

# Restart Ollama
docker-compose restart ollama
```

#### 2. Out of memory errors

```bash
# Check memory usage
docker stats

# Increase memory limits in docker-compose.yaml
# Or reduce concurrent requests
```

#### 3. Models not found

```bash
# Pull missing models
docker-compose exec ollama ollama pull qwen2.5:3b

# List available models
docker-compose exec ollama ollama list
```

#### 4. Permission errors

```bash
# Fix volume permissions
docker-compose exec tinyllm chown -R tinyllm:tinyllm /app/data
```

#### 5. Network connectivity issues

```bash
# Check network
docker network inspect tinyllm-network

# Recreate network
docker-compose down
docker-compose up -d
```

### Debug Mode

Enable debug logging:

```bash
# Temporary (environment variable)
docker-compose exec -e TINYLLM_LOG_LEVEL=DEBUG tinyllm tinyllm run "test"

# Permanent (edit .env)
TINYLLM_LOG_LEVEL=DEBUG
docker-compose up -d
```

### Container Shell Access

```bash
# Access TinyLLM container
docker-compose exec tinyllm bash

# Access as root (for debugging)
docker-compose exec -u root tinyllm bash

# Access Ollama container
docker-compose exec ollama bash
```

## Cleanup

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data!)
docker-compose down -v

# Remove only stopped containers
docker-compose rm
```

### Prune Docker Resources

```bash
# Remove unused containers, networks, images
docker system prune -a

# Remove volumes (careful!)
docker volume prune
```

## Advanced Topics

### Custom Graph Definitions

Mount custom graphs:

```yaml
volumes:
  - ./my-graphs:/app/graphs:ro
```

### Multi-Stage Deployments

Use different compose files for different environments:

```bash
# Development
docker-compose -f docker-compose.yaml up

# Production
docker-compose -f docker-compose.yaml -f docker-compose.prod.yaml up
```

### Scaling Services

```bash
# Scale TinyLLM instances
docker-compose up -d --scale tinyllm=3
```

### Integration with CI/CD

```bash
# Build and test in CI
docker build -t tinyllm:test .
docker run tinyllm:test pytest

# Push to registry
docker tag tinyllm:test registry.example.com/tinyllm:latest
docker push registry.example.com/tinyllm:latest
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/ndjstn/tinyllm/issues
- Documentation: https://github.com/ndjstn/tinyllm#readme
