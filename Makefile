.PHONY: test test-unit test-integration test-stress test-load test-chaos test-perf test-cov test-cov-gate clean install help
.PHONY: docker-build docker-up docker-down docker-logs docker-clean docker-pull-models
.PHONY: docker-health docker-shell docker-test docker-backup docker-restore

help:
	@echo "TinyLLM Development Makefile"
	@echo ""
	@echo "Development Commands:"
	@echo "  make install            Install dependencies in virtual environment"
	@echo "  make test               Run all tests"
	@echo "  make test-unit          Run unit tests only"
	@echo "  make test-integration   Run integration tests only"
	@echo "  make test-stress        Run stress tests only"
	@echo "  make test-load          Run load tests only"
	@echo "  make test-chaos         Run chaos tests only"
	@echo "  make test-perf          Run performance tests only"
	@echo "  make test-cov           Run tests with coverage report"
	@echo "  make test-cov-gate      Run tests with coverage gate (requires >=80%)"
	@echo "  make clean              Remove cache and build artifacts"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build       Build Docker images"
	@echo "  make docker-up          Start all Docker services"
	@echo "  make docker-down        Stop all Docker services"
	@echo "  make docker-logs        View Docker logs (all services)"
	@echo "  make docker-clean       Clean up Docker resources"
	@echo "  make docker-pull-models Pull recommended Ollama models"
	@echo "  make docker-health      Check health of all services"
	@echo "  make docker-shell       Open shell in TinyLLM container"
	@echo "  make docker-test        Run tests in Docker"
	@echo "  make docker-backup      Backup Docker volumes"
	@echo "  make docker-restore     Restore Docker volumes from backup"
	@echo ""

install:
	.venv/bin/python -m pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest tests/ -v

test-unit:
	.venv/bin/python -m pytest tests/unit/ -v

test-integration:
	.venv/bin/python -m pytest tests/integration/ -v

test-stress:
	.venv/bin/python -m pytest tests/stress/ -v

test-load:
	.venv/bin/python -m pytest tests/load/ -v -m load

test-chaos:
	.venv/bin/python -m pytest tests/chaos/ -v -m chaos

test-perf:
	.venv/bin/python -m pytest tests/perf/ -v -m perf

test-cov:
	.venv/bin/python -m pytest tests/ --cov=src/tinyllm --cov-report=html --cov-report=term

test-cov-gate:
	.venv/bin/python scripts/check_coverage.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

# ============================================================================
# Docker Commands
# ============================================================================

docker-build:
	@echo "Building TinyLLM Docker image..."
	docker-compose build

docker-up:
	@echo "Starting TinyLLM stack..."
	docker-compose up -d
	@echo "Waiting for services to start..."
	@sleep 5
	@echo ""
	@echo "Services started! Access points:"
	@echo "  - TinyLLM:    http://localhost:8000"
	@echo "  - Metrics:    http://localhost:9090"
	@echo "  - Prometheus: http://localhost:9091"
	@echo "  - Grafana:    http://localhost:3000 (admin/admin)"
	@echo "  - Ollama:     http://localhost:11434"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Pull models: make docker-pull-models"
	@echo "  2. Check health: make docker-health"
	@echo "  3. View logs: make docker-logs"

docker-down:
	@echo "Stopping TinyLLM stack..."
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v
	@echo "WARNING: This will remove all volumes and data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v --remove-orphans; \
		docker system prune -f; \
		echo "Cleanup complete!"; \
	else \
		echo "Cleanup cancelled."; \
	fi

docker-pull-models:
	@echo "Pulling recommended Ollama models..."
	@echo "This may take a while depending on your internet connection..."
	docker-compose exec ollama ollama pull qwen2.5:0.5b
	docker-compose exec ollama ollama pull qwen2.5:3b
	docker-compose exec ollama ollama pull granite-code:3b
	@echo ""
	@echo "Models installed! List them with:"
	@echo "  docker-compose exec ollama ollama list"

docker-health:
	@echo "Checking service health..."
	@echo ""
	@echo "=== Docker Compose Services ==="
	@docker-compose ps
	@echo ""
	@echo "=== TinyLLM Health ==="
	@docker-compose exec tinyllm tinyllm health || echo "TinyLLM health check failed"
	@echo ""
	@echo "=== Ollama Status ==="
	@docker-compose exec ollama ollama list || echo "Ollama not available"

docker-shell:
	@echo "Opening shell in TinyLLM container..."
	docker-compose exec tinyllm bash

docker-test:
	@echo "Running tests in Docker container..."
	docker-compose exec tinyllm pytest tests/ -v

docker-backup:
	@echo "Backing up Docker volumes..."
	@mkdir -p backups
	@docker run --rm \
		-v tinyllm-data:/data \
		-v $(PWD)/backups:/backup \
		alpine tar czf /backup/tinyllm-data-$$(date +%Y%m%d-%H%M%S).tar.gz -C /data .
	@docker run --rm \
		-v tinyllm-config:/data \
		-v $(PWD)/backups:/backup \
		alpine tar czf /backup/tinyllm-config-$$(date +%Y%m%d-%H%M%S).tar.gz -C /data .
	@echo "Backups created in ./backups/"

docker-restore:
	@echo "Available backups:"
	@ls -lh backups/*.tar.gz 2>/dev/null || echo "No backups found"
	@echo ""
	@echo "To restore, run:"
	@echo "  docker run --rm -v tinyllm-data:/data -v \$$(PWD)/backups:/backup alpine tar xzf /backup/BACKUP_FILE.tar.gz -C /data"
