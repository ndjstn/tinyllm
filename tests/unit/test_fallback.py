"""Unit tests for fallback model strategies."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tinyllm.models.fallback import (
    FallbackClient,
    FallbackConfig,
    FallbackStrategy,
    HealthTracker,
    ModelHealth,
)
from tinyllm.models.client import GenerateResponse


@pytest.fixture
def mock_response():
    """Create a mock GenerateResponse."""
    return GenerateResponse(
        model="test-model",
        created_at="2024-01-01T00:00:00Z",
        response="Test response",
        done=True,
        total_duration=1000000,
        load_duration=500000,
        prompt_eval_count=10,
        prompt_eval_duration=300000,
        eval_count=20,
        eval_duration=200000,
    )


@pytest.fixture
def fallback_config():
    """Create a test fallback config."""
    return FallbackConfig(
        primary_model="primary-model",
        fallback_models=["fallback-1", "fallback-2"],
        timeout_ms=5000,
        strategy=FallbackStrategy.SEQUENTIAL,
    )


class TestModelHealth:
    """Test ModelHealth tracking."""

    def test_initial_state(self):
        """Test initial health state."""
        health = ModelHealth(model_name="test-model")
        assert health.success_count == 0
        assert health.failure_count == 0
        assert health.total_requests == 0
        assert health.success_rate == 1.0  # Start optimistic
        assert health.is_healthy is True

    def test_record_success(self):
        """Test recording successful requests."""
        health = ModelHealth(model_name="test-model")
        health.record_success(latency_ms=100.0)

        assert health.success_count == 1
        assert health.failure_count == 0
        assert health.success_rate == 1.0
        assert health.average_latency_ms == 100.0
        assert health.consecutive_failures == 0
        assert health.is_healthy is True

    def test_record_failure(self):
        """Test recording failed requests."""
        health = ModelHealth(model_name="test-model")
        health.record_failure()

        assert health.success_count == 0
        assert health.failure_count == 1
        assert health.success_rate == 0.0
        assert health.consecutive_failures == 1
        assert health.is_healthy is True  # Still healthy after 1 failure

    def test_becomes_unhealthy(self):
        """Test model becomes unhealthy after consecutive failures."""
        health = ModelHealth(model_name="test-model")
        health.record_failure()
        health.record_failure()
        health.record_failure()

        assert health.consecutive_failures == 3
        assert health.is_healthy is False

    def test_recovery_after_success(self):
        """Test model recovers after successful request."""
        health = ModelHealth(model_name="test-model")
        health.record_failure()
        health.record_failure()
        health.record_success(latency_ms=100.0)

        assert health.consecutive_failures == 0
        assert health.is_healthy is True

    def test_mixed_requests(self):
        """Test success rate with mixed successes and failures."""
        health = ModelHealth(model_name="test-model")
        health.record_success(latency_ms=100.0)
        health.record_success(latency_ms=200.0)
        health.record_failure()
        health.record_success(latency_ms=150.0)

        assert health.total_requests == 4
        assert health.success_count == 3
        assert health.failure_count == 1
        assert health.success_rate == 0.75
        assert health.average_latency_ms == 150.0  # (100 + 200 + 150) / 3


class TestHealthTracker:
    """Test HealthTracker functionality."""

    @pytest.mark.asyncio
    async def test_record_success(self):
        """Test recording success updates health."""
        tracker = HealthTracker()
        await tracker.record_success("model-1", latency_ms=100.0)

        health = tracker.get_health("model-1")
        assert health.success_count == 1
        assert health.average_latency_ms == 100.0

    @pytest.mark.asyncio
    async def test_record_failure(self):
        """Test recording failure updates health."""
        tracker = HealthTracker()
        await tracker.record_failure("model-1", error="timeout")

        health = tracker.get_health("model-1")
        assert health.failure_count == 1

    def test_is_healthy(self):
        """Test checking if model is healthy."""
        tracker = HealthTracker()
        assert tracker.is_healthy("new-model") is True

        health = tracker.get_health("new-model")
        health.record_failure()
        health.record_failure()
        health.record_failure()

        assert tracker.is_healthy("new-model") is False

    def test_get_models_by_health(self):
        """Test sorting models by health."""
        tracker = HealthTracker()

        # Model 1: Good success rate, low latency
        h1 = tracker.get_health("model-1")
        h1.record_success(100.0)
        h1.record_success(100.0)

        # Model 2: Lower success rate
        h2 = tracker.get_health("model-2")
        h2.record_success(100.0)
        h2.record_failure()

        # Model 3: Unhealthy
        h3 = tracker.get_health("model-3")
        h3.record_failure()
        h3.record_failure()
        h3.record_failure()

        ordered = tracker.get_models_by_health(["model-1", "model-2", "model-3"])
        assert ordered[0] == "model-1"  # Best
        assert ordered[1] == "model-2"  # Medium
        assert ordered[2] == "model-3"  # Unhealthy

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        tracker = HealthTracker()

        h1 = tracker.get_health("model-1")
        h1.record_success(100.0)

        h2 = tracker.get_health("model-2")
        h2.record_failure()

        metrics = tracker.get_all_metrics()
        assert "model-1" in metrics
        assert "model-2" in metrics
        assert metrics["model-1"]["success_count"] == 1
        assert metrics["model-2"]["failure_count"] == 1


class TestFallbackClient:
    """Test FallbackClient functionality."""

    @pytest.mark.asyncio
    async def test_sequential_success_primary(self, fallback_config, mock_response):
        """Test sequential strategy succeeds with primary model."""
        client = FallbackClient(config=fallback_config)

        # Mock the _try_model method to succeed on first try
        with patch.object(
            client, "_try_model", new_callable=AsyncMock
        ) as mock_try:
            mock_try.return_value = mock_response

            result = await client.generate(prompt="test prompt")

            assert result.model_used == "primary-model"
            assert result.fallback_occurred is False
            assert mock_try.call_count == 1

    @pytest.mark.asyncio
    async def test_sequential_fallback_on_failure(self, fallback_config, mock_response):
        """Test sequential strategy falls back on primary failure."""
        client = FallbackClient(config=fallback_config)

        # Mock to fail on primary, succeed on fallback
        call_count = 0

        async def mock_try_model(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Primary failed")
            return mock_response

        with patch.object(client, "_try_model", side_effect=mock_try_model):
            result = await client.generate(prompt="test prompt")

            assert result.model_used == "fallback-1"
            assert result.fallback_occurred is True
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_sequential_all_models_fail(self, fallback_config):
        """Test sequential strategy when all models fail."""
        client = FallbackClient(config=fallback_config)

        with patch.object(
            client, "_try_model", new_callable=AsyncMock
        ) as mock_try:
            mock_try.side_effect = RuntimeError("All failed")

            with pytest.raises(RuntimeError, match="All models failed"):
                await client.generate(prompt="test prompt")

    @pytest.mark.asyncio
    async def test_skip_unhealthy_models(self, fallback_config, mock_response):
        """Test skipping unhealthy models."""
        client = FallbackClient(config=fallback_config)

        # Mark primary as unhealthy
        primary_health = client._health_tracker.get_health("primary-model")
        primary_health.record_failure()
        primary_health.record_failure()
        primary_health.record_failure()

        with patch.object(
            client, "_try_model", new_callable=AsyncMock
        ) as mock_try:
            mock_try.return_value = mock_response

            result = await client.generate(prompt="test prompt")

            # Should skip primary and go to fallback-1
            assert result.model_used == "fallback-1"
            assert mock_try.call_count == 1

    @pytest.mark.asyncio
    async def test_fastest_strategy(self, mock_response):
        """Test fastest strategy races all models."""
        config = FallbackConfig(
            primary_model="primary-model",
            fallback_models=["fallback-1"],
            strategy=FallbackStrategy.FASTEST,
        )
        client = FallbackClient(config=config)

        # Mock to make fallback-1 respond faster
        async def mock_try_model(model, *args, **kwargs):
            if model == "primary-model":
                await asyncio.sleep(1.0)  # Slow
            else:
                await asyncio.sleep(0.1)  # Fast
            return mock_response

        with patch.object(client, "_try_model", side_effect=mock_try_model):
            result = await client.generate(prompt="test prompt")

            # Should use faster model
            assert result.model_used in ["primary-model", "fallback-1"]

    @pytest.mark.asyncio
    async def test_load_balanced_strategy(self, mock_response):
        """Test load balanced strategy uses health metrics."""
        config = FallbackConfig(
            primary_model="primary-model",
            fallback_models=["fallback-1"],
            strategy=FallbackStrategy.LOAD_BALANCED,
        )
        client = FallbackClient(config=config)

        # Make fallback-1 healthier than primary
        primary_health = client._health_tracker.get_health("primary-model")
        primary_health.record_failure()
        primary_health.record_failure()

        fallback_health = client._health_tracker.get_health("fallback-1")
        fallback_health.record_success(100.0)
        fallback_health.record_success(100.0)

        with patch.object(
            client, "_try_model", new_callable=AsyncMock
        ) as mock_try:
            mock_try.return_value = mock_response

            result = await client.generate(prompt="test prompt")

            # Should prefer healthier model
            assert result.model_used == "fallback-1"

    def test_get_health_metrics(self, fallback_config):
        """Test getting health metrics."""
        client = FallbackClient(config=fallback_config)

        # Record some activity
        h1 = client._health_tracker.get_health("primary-model")
        h1.record_success(100.0)
        h1.record_success(150.0)

        h2 = client._health_tracker.get_health("fallback-1")
        h2.record_failure()

        metrics = client.get_health_metrics()

        assert "per_model" in metrics
        assert "overall" in metrics
        assert metrics["overall"]["total_requests"] == 3
        assert metrics["overall"]["total_successes"] == 2

    def test_get_fallback_statistics(self, fallback_config):
        """Test getting fallback statistics."""
        client = FallbackClient(config=fallback_config)

        # Record some activity
        h1 = client._health_tracker.get_health("primary-model")
        h1.record_success(100.0)

        h2 = client._health_tracker.get_health("fallback-1")
        h2.record_success(150.0)

        stats = client.get_fallback_statistics()

        assert "total_requests" in stats
        assert "fallback_requests" in stats
        assert stats["fallback_rate"] > 0  # Some fallback occurred


class TestFallbackConfig:
    """Test FallbackConfig validation."""

    def test_default_config(self):
        """Test default configuration."""
        config = FallbackConfig(primary_model="test-model")

        assert config.primary_model == "test-model"
        assert config.fallback_models == []
        assert config.strategy == FallbackStrategy.SEQUENTIAL
        assert config.timeout_ms == 30000

    def test_custom_config(self):
        """Test custom configuration."""
        config = FallbackConfig(
            primary_model="primary",
            fallback_models=["fb1", "fb2", "fb3"],
            timeout_ms=60000,
            strategy=FallbackStrategy.FASTEST,
        )

        assert config.primary_model == "primary"
        assert len(config.fallback_models) == 3
        assert config.timeout_ms == 60000
        assert config.strategy == FallbackStrategy.FASTEST

    def test_retry_on_errors(self):
        """Test retry error configuration."""
        config = FallbackConfig(
            primary_model="test",
            retry_on_errors=["timeout", "connection", "rate_limit"],
        )

        assert "timeout" in config.retry_on_errors
        assert "connection" in config.retry_on_errors
