# TinyLLM Docker Quick Start

Get TinyLLM running in Docker in under 5 minutes!

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 16GB RAM minimum
- 50GB disk space

## Quick Start (3 Commands)

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Start the stack
make docker-up

# 3. Pull models (required)
make docker-pull-models
```

That's it! TinyLLM is now running.

## Verify Installation

```bash
# Check all services are healthy
make docker-health

# Run a test query
docker-compose exec tinyllm tinyllm run "What is 2+2?"
```

## Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Metrics | http://localhost:9090 | - |
| Prometheus | http://localhost:9091 | - |
| Grafana | http://localhost:3000 | admin/admin |
| Ollama API | http://localhost:11434 | - |

## Common Commands

```bash
# View logs
make docker-logs

# Stop services
make docker-down

# Restart services
make docker-down && make docker-up

# Open shell in container
make docker-shell

# Run tests
make docker-test

# Backup data
make docker-backup
```

## Interactive Chat

```bash
docker-compose exec tinyllm tinyllm chat --model qwen2.5:3b
```

Type your messages and press Enter. Type 'quit' to exit.

## Running Queries

```bash
# Simple query
docker-compose exec tinyllm tinyllm run "Your question here"

# With execution trace
docker-compose exec tinyllm tinyllm run --trace "Write Python code to sort a list"
```

## Managing Models

```bash
# List installed models
docker-compose exec ollama ollama list

# Pull a new model
docker-compose exec ollama ollama pull llama2:7b

# Remove a model
docker-compose exec ollama ollama rm llama2:7b
```

## Troubleshooting

### Services won't start

```bash
# Check logs
make docker-logs

# Restart everything
make docker-down
make docker-up
```

### Ollama can't connect

```bash
# Check Ollama service
docker-compose ps ollama
docker-compose logs ollama

# Restart Ollama
docker-compose restart ollama
```

### Out of memory

```bash
# Check resource usage
docker stats

# Reduce concurrent requests or increase Docker memory limits
```

### Models not found

```bash
# Pull missing models
make docker-pull-models

# Or pull specific model
docker-compose exec ollama ollama pull qwen2.5:3b
```

## GPU Support (Optional)

If you have an NVIDIA GPU:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Edit `docker-compose.yaml` and uncomment the GPU sections under the `ollama` service

3. Restart:
```bash
make docker-down
make docker-up
```

4. Verify:
```bash
docker-compose exec ollama nvidia-smi
```

## Clean Up

```bash
# Stop services (keeps data)
make docker-down

# Remove everything including data
make docker-clean
```

## Production Notes

Before deploying to production:

- [ ] Change Grafana password in `.env`
- [ ] Set Redis password in `.env`
- [ ] Review resource limits in `docker-compose.yaml`
- [ ] Set up HTTPS/TLS termination
- [ ] Configure backups (use `make docker-backup`)
- [ ] Set up monitoring alerts

## Need Help?

- Full documentation: [DOCKER.md](./DOCKER.md)
- Project README: [README.md](./README.md)
- Issues: https://github.com/ndjstn/tinyllm/issues
