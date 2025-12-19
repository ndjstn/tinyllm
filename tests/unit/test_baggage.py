"""Tests for OpenTelemetry baggage propagation."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_telemetry_enabled():
    """Mock telemetry as enabled."""
    with patch("tinyllm.telemetry.is_telemetry_enabled", return_value=True):
        yield


@pytest.fixture
def mock_baggage():
    """Mock baggage module."""
    mock = MagicMock()
    with patch("tinyllm.telemetry.baggage", mock):
        yield mock


class TestBaggageFunctions:
    """Test baggage propagation functions."""

    def test_set_baggage(self, mock_telemetry_enabled, mock_baggage):
        """Test setting baggage."""
        from tinyllm.telemetry import set_baggage

        set_baggage("user_id", "12345")

        mock_baggage.set_baggage.assert_called_once_with("user_id", "12345")

    def test_set_baggage_disabled(self):
        """Test setting baggage when telemetry is disabled."""
        from tinyllm.telemetry import set_baggage

        with patch("tinyllm.telemetry.is_telemetry_enabled", return_value=False):
            # Should not raise error
            set_baggage("user_id", "12345")

    def test_get_baggage(self, mock_telemetry_enabled, mock_baggage):
        """Test getting baggage."""
        from tinyllm.telemetry import get_baggage

        mock_baggage.get_baggage.return_value = "12345"

        result = get_baggage("user_id")

        assert result == "12345"
        mock_baggage.get_baggage.assert_called_once_with("user_id")

    def test_get_baggage_disabled(self):
        """Test getting baggage when telemetry is disabled."""
        from tinyllm.telemetry import get_baggage

        with patch("tinyllm.telemetry.is_telemetry_enabled", return_value=False):
            result = get_baggage("user_id")
            assert result is None

    def test_get_all_baggage(self, mock_telemetry_enabled, mock_baggage):
        """Test getting all baggage."""
        from tinyllm.telemetry import get_all_baggage

        mock_baggage.get_all.return_value = {"user_id": "12345", "tenant_id": "acme"}

        result = get_all_baggage()

        assert result == {"user_id": "12345", "tenant_id": "acme"}
        mock_baggage.get_all.assert_called_once()

    def test_get_all_baggage_empty(self, mock_telemetry_enabled, mock_baggage):
        """Test getting all baggage when empty."""
        from tinyllm.telemetry import get_all_baggage

        mock_baggage.get_all.return_value = None

        result = get_all_baggage()

        assert result == {}

    def test_remove_baggage(self, mock_telemetry_enabled, mock_baggage):
        """Test removing baggage."""
        from tinyllm.telemetry import remove_baggage

        remove_baggage("user_id")

        mock_baggage.remove_baggage.assert_called_once_with("user_id")

    def test_clear_baggage(self, mock_telemetry_enabled, mock_baggage):
        """Test clearing all baggage."""
        from tinyllm.telemetry import clear_baggage

        clear_baggage()

        mock_baggage.clear.assert_called_once()

    def test_inject_baggage_into_headers(self, mock_telemetry_enabled):
        """Test injecting baggage into headers."""
        from tinyllm.telemetry import inject_baggage_into_headers

        with patch("opentelemetry.propagate.inject") as mock_inject:
            headers = {"Content-Type": "application/json"}
            result = inject_baggage_into_headers(headers)

            assert result == headers
            mock_inject.assert_called_once_with(headers)

    def test_inject_baggage_into_headers_empty(self, mock_telemetry_enabled):
        """Test injecting baggage into empty headers."""
        from tinyllm.telemetry import inject_baggage_into_headers

        with patch("opentelemetry.propagate.inject") as mock_inject:
            result = inject_baggage_into_headers()

            assert isinstance(result, dict)
            mock_inject.assert_called_once()

    def test_inject_baggage_disabled(self):
        """Test injecting baggage when telemetry is disabled."""
        from tinyllm.telemetry import inject_baggage_into_headers

        with patch("tinyllm.telemetry.is_telemetry_enabled", return_value=False):
            headers = {"Content-Type": "application/json"}
            result = inject_baggage_into_headers(headers)

            assert result == headers

    def test_extract_baggage_from_headers(self, mock_telemetry_enabled, mock_baggage):
        """Test extracting baggage from headers."""
        from tinyllm.telemetry import extract_baggage_from_headers

        mock_context = MagicMock()
        mock_baggage.get_all.return_value = {"user_id": "12345"}

        with patch("opentelemetry.propagate.extract", return_value=mock_context):
            headers = {"baggage": "user_id=12345"}
            result = extract_baggage_from_headers(headers)

            assert result == {"user_id": "12345"}

    def test_extract_baggage_from_headers_empty(self, mock_telemetry_enabled, mock_baggage):
        """Test extracting baggage from empty headers."""
        from tinyllm.telemetry import extract_baggage_from_headers

        mock_context = MagicMock()
        mock_baggage.get_all.return_value = {}

        with patch("opentelemetry.propagate.extract", return_value=mock_context):
            result = extract_baggage_from_headers({})

            assert result == {}

    def test_extract_baggage_disabled(self):
        """Test extracting baggage when telemetry is disabled."""
        from tinyllm.telemetry import extract_baggage_from_headers

        with patch("tinyllm.telemetry.is_telemetry_enabled", return_value=False):
            result = extract_baggage_from_headers({"baggage": "user_id=12345"})
            assert result == {}


class TestBaggageContext:
    """Test baggage context manager."""

    def test_baggage_context(self, mock_telemetry_enabled, mock_baggage):
        """Test baggage context manager."""
        from tinyllm.telemetry import baggage_context

        # Mock get_all_baggage to return empty initially
        mock_baggage.get_all.return_value = {}

        with baggage_context(user_id="12345", tenant_id="acme"):
            # Verify baggage was set
            assert mock_baggage.set_baggage.call_count >= 2

        # Verify baggage was cleared
        mock_baggage.clear.assert_called()

    def test_baggage_context_restores_original(self, mock_telemetry_enabled, mock_baggage):
        """Test that baggage context restores original baggage."""
        from tinyllm.telemetry import baggage_context

        # Mock existing baggage
        mock_baggage.get_all.return_value = {"existing": "value"}

        with baggage_context(user_id="12345"):
            pass

        # Should restore original baggage
        mock_baggage.clear.assert_called()

    def test_baggage_context_disabled(self):
        """Test baggage context when telemetry is disabled."""
        from tinyllm.telemetry import baggage_context

        with patch("tinyllm.telemetry.is_telemetry_enabled", return_value=False):
            # Should not raise error
            with baggage_context(user_id="12345"):
                pass

    def test_baggage_context_exception(self, mock_telemetry_enabled, mock_baggage):
        """Test that baggage context cleans up on exception."""
        from tinyllm.telemetry import baggage_context

        mock_baggage.get_all.return_value = {}

        with pytest.raises(ValueError):
            with baggage_context(user_id="12345"):
                raise ValueError("Test error")

        # Should still clean up
        mock_baggage.clear.assert_called()


class TestBaggageErrorHandling:
    """Test error handling in baggage functions."""

    def test_set_baggage_error(self, mock_telemetry_enabled, mock_baggage):
        """Test error handling in set_baggage."""
        from tinyllm.telemetry import set_baggage

        mock_baggage.set_baggage.side_effect = Exception("Test error")

        # Should not raise, just log warning
        set_baggage("user_id", "12345")

    def test_get_baggage_error(self, mock_telemetry_enabled, mock_baggage):
        """Test error handling in get_baggage."""
        from tinyllm.telemetry import get_baggage

        mock_baggage.get_baggage.side_effect = Exception("Test error")

        result = get_baggage("user_id")
        assert result is None

    def test_get_all_baggage_error(self, mock_telemetry_enabled, mock_baggage):
        """Test error handling in get_all_baggage."""
        from tinyllm.telemetry import get_all_baggage

        mock_baggage.get_all.side_effect = Exception("Test error")

        result = get_all_baggage()
        assert result == {}

    def test_inject_baggage_error(self, mock_telemetry_enabled):
        """Test error handling in inject_baggage_into_headers."""
        from tinyllm.telemetry import inject_baggage_into_headers

        with patch("opentelemetry.propagate.inject", side_effect=Exception("Test error")):
            headers = {"Content-Type": "application/json"}
            result = inject_baggage_into_headers(headers)

            # Should return headers unchanged
            assert result == headers

    def test_extract_baggage_error(self, mock_telemetry_enabled):
        """Test error handling in extract_baggage_from_headers."""
        from tinyllm.telemetry import extract_baggage_from_headers

        with patch("opentelemetry.propagate.extract", side_effect=Exception("Test error")):
            result = extract_baggage_from_headers({"baggage": "user_id=12345"})

            # Should return empty dict
            assert result == {}
