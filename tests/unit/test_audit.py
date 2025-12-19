"""Tests for audit logging."""

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tinyllm.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditSeverity,
    audit_event,
    get_audit_logger,
)


class TestAuditEvent:
    """Test AuditEvent model."""

    def test_create_event(self):
        """Test creating an audit event."""
        event = AuditEvent(
            event_id="test-001",
            event_type=AuditEventType.GRAPH_EXECUTED,
            timestamp=datetime.now(timezone.utc).isoformat(),
            action="execute_graph",
            status="success",
            severity=AuditSeverity.INFO,
            actor_id="user-123",
            resource_type="graph",
            resource_id="multi_domain",
            message="Graph executed successfully",
        )

        assert event.event_id == "test-001"
        assert event.event_type == AuditEventType.GRAPH_EXECUTED
        assert event.action == "execute_graph"
        assert event.status == "success"
        assert event.severity == AuditSeverity.INFO
        assert event.actor_id == "user-123"
        assert event.resource_type == "graph"
        assert event.resource_id == "multi_domain"

    def test_compute_hash(self):
        """Test computing event hash."""
        event = AuditEvent(
            event_id="test-001",
            event_type=AuditEventType.MODEL_REQUEST,
            timestamp="2025-01-01T00:00:00Z",
            action="generate",
            status="success",
        )

        hash1 = event.compute_hash()
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters

        # Same event produces same hash
        hash2 = event.compute_hash()
        assert hash1 == hash2

        # Different event produces different hash
        event2 = AuditEvent(
            event_id="test-002",
            event_type=AuditEventType.MODEL_REQUEST,
            timestamp="2025-01-01T00:00:00Z",
            action="generate",
            status="success",
        )
        hash3 = event2.compute_hash()
        assert hash1 != hash3

    def test_compute_hash_with_chain(self):
        """Test computing hash with previous hash."""
        event = AuditEvent(
            event_id="test-001",
            event_type=AuditEventType.GRAPH_EXECUTED,
            timestamp="2025-01-01T00:00:00Z",
            action="execute",
            status="success",
        )

        hash1 = event.compute_hash()
        hash2 = event.compute_hash(previous_hash="abc123")

        # Hash should be different with previous hash
        assert hash1 != hash2


class TestAuditLogger:
    """Test AuditLogger."""

    def test_init_with_file(self):
        """Test initializing logger with file."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            logger = AuditLogger(log_file=log_file)

            assert logger.log_file == log_file
            assert logger.enable_chain_verification is True
            assert logger._event_count == 0

    def test_log_event(self):
        """Test logging an event."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            logger = AuditLogger(log_file=log_file, buffer_size=1)

            event = logger.log_event(
                event_type=AuditEventType.GRAPH_EXECUTED,
                action="execute_graph",
                status="success",
                actor_id="user-123",
                resource_type="graph",
                resource_id="multi_domain",
                message="Test execution",
            )

            assert event.event_type == AuditEventType.GRAPH_EXECUTED
            assert event.action == "execute_graph"
            assert event.status == "success"
            assert event.actor_id == "user-123"
            assert event.event_hash is not None

            # Event should be flushed to file
            assert log_file.exists()

    def test_hash_chain(self):
        """Test hash chain integrity."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            logger = AuditLogger(log_file=log_file, buffer_size=10)

            # Log multiple events
            event1 = logger.log_event(
                event_type=AuditEventType.SYSTEM_START,
                action="start_system",
                status="success",
            )
            event2 = logger.log_event(
                event_type=AuditEventType.GRAPH_EXECUTED,
                action="execute_graph",
                status="success",
            )
            event3 = logger.log_event(
                event_type=AuditEventType.SYSTEM_STOP,
                action="stop_system",
                status="success",
            )

            # Verify chain
            assert event1.previous_hash is None
            assert event2.previous_hash == event1.event_hash
            assert event3.previous_hash == event2.event_hash

    def test_verify_chain(self):
        """Test verifying hash chain."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            logger = AuditLogger(log_file=log_file)

            # Create valid chain
            events = [
                logger.log_event(
                    event_type=AuditEventType.GRAPH_EXECUTED,
                    action=f"action-{i}",
                    status="success",
                )
                for i in range(5)
            ]

            # Verify valid chain
            is_valid, error = logger.verify_chain(events)
            assert is_valid is True
            assert error is None

    def test_verify_broken_chain(self):
        """Test detecting broken hash chain."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            logger = AuditLogger(log_file=log_file)

            # Create events
            events = [
                logger.log_event(
                    event_type=AuditEventType.GRAPH_EXECUTED,
                    action=f"action-{i}",
                    status="success",
                )
                for i in range(3)
            ]

            # Break the chain
            events[1].previous_hash = "invalid_hash"

            # Verify should fail
            is_valid, error = logger.verify_chain(events)
            assert is_valid is False
            assert "hash chain broken" in error.lower()

    def test_flush(self):
        """Test flushing events to disk."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            logger = AuditLogger(log_file=log_file, buffer_size=100)

            # Log events
            for i in range(5):
                logger.log_event(
                    event_type=AuditEventType.GRAPH_EXECUTED,
                    action=f"action-{i}",
                    status="success",
                )

            assert len(logger._buffer) == 5

            # Flush
            logger.flush()
            assert len(logger._buffer) == 0
            assert log_file.exists()

            # Verify file contents
            with open(log_file) as f:
                lines = f.readlines()
                assert len(lines) == 5

    def test_read_events(self):
        """Test reading events from file."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            logger = AuditLogger(log_file=log_file, buffer_size=1)

            # Log events
            logger.log_event(
                event_type=AuditEventType.GRAPH_EXECUTED,
                action="execute_graph",
                status="success",
                actor_id="user-123",
                resource_id="graph-1",
            )
            logger.log_event(
                event_type=AuditEventType.MODEL_REQUEST,
                action="generate",
                status="success",
                actor_id="user-456",
                resource_id="model-1",
            )
            logger.log_event(
                event_type=AuditEventType.GRAPH_EXECUTED,
                action="execute_graph",
                status="failure",
                actor_id="user-123",
                resource_id="graph-2",
            )

            # Read all events
            events = logger.read_events()
            assert len(events) == 3

            # Filter by event type
            graph_events = logger.read_events(event_type=AuditEventType.GRAPH_EXECUTED)
            assert len(graph_events) == 2

            # Filter by actor
            user_events = logger.read_events(actor_id="user-123")
            assert len(user_events) == 2

            # Filter by resource
            resource_events = logger.read_events(resource_id="graph-1")
            assert len(resource_events) == 1

    def test_read_events_with_limit(self):
        """Test reading events with limit."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            logger = AuditLogger(log_file=log_file, buffer_size=1)

            # Log events
            for i in range(10):
                logger.log_event(
                    event_type=AuditEventType.GRAPH_EXECUTED,
                    action=f"action-{i}",
                    status="success",
                )

            # Read with limit
            events = logger.read_events(limit=5)
            assert len(events) == 5

    def test_get_statistics(self):
        """Test getting audit log statistics."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            logger = AuditLogger(log_file=log_file, buffer_size=1)

            # Log events with different types and severities
            logger.log_event(
                event_type=AuditEventType.GRAPH_EXECUTED,
                action="execute",
                status="success",
                severity=AuditSeverity.INFO,
            )
            logger.log_event(
                event_type=AuditEventType.GRAPH_EXECUTED,
                action="execute",
                status="success",
                severity=AuditSeverity.INFO,
            )
            logger.log_event(
                event_type=AuditEventType.SYSTEM_ERROR,
                action="error",
                status="failure",
                severity=AuditSeverity.ERROR,
            )

            stats = logger.get_statistics()
            assert stats["total_events"] == 3
            assert stats["event_types"]["graph.executed"] == 2
            assert stats["event_types"]["system.error"] == 1
            assert stats["severities"]["info"] == 2
            assert stats["severities"]["error"] == 1
            assert stats["file_size_bytes"] > 0

    def test_severity_levels(self):
        """Test different severity levels."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            logger = AuditLogger(log_file=log_file)

            severities = [
                AuditSeverity.DEBUG,
                AuditSeverity.INFO,
                AuditSeverity.NOTICE,
                AuditSeverity.WARNING,
                AuditSeverity.ERROR,
                AuditSeverity.CRITICAL,
                AuditSeverity.ALERT,
            ]

            for severity in severities:
                event = logger.log_event(
                    event_type=AuditEventType.SYSTEM_ERROR,
                    action="test",
                    status="success",
                    severity=severity,
                )
                assert event.severity == severity


class TestAuditEventTypes:
    """Test audit event types."""

    def test_all_event_types(self):
        """Test that all event types are valid."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            logger = AuditLogger(log_file=log_file)

            # Test a representative sample of event types
            test_types = [
                AuditEventType.AUTH_SUCCESS,
                AuditEventType.GRAPH_EXECUTED,
                AuditEventType.MODEL_REQUEST,
                AuditEventType.DATA_EXPORT,
                AuditEventType.SECURITY_VIOLATION,
                AuditEventType.COMPLIANCE_PII_ACCESS,
            ]

            for event_type in test_types:
                event = logger.log_event(
                    event_type=event_type,
                    action=f"test_{event_type}",
                    status="success",
                )
                assert event.event_type == event_type


class TestGlobalAuditLogger:
    """Test global audit logger functions."""

    def test_get_audit_logger(self):
        """Test getting global audit logger."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            logger1 = get_audit_logger(log_file=log_file)
            logger2 = get_audit_logger(log_file=log_file)

            # Should return same instance
            assert logger1 is logger2

    def test_audit_event_helper(self):
        """Test audit_event helper function."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            # Initialize global logger
            get_audit_logger(log_file=log_file)

            event = audit_event(
                event_type=AuditEventType.GRAPH_EXECUTED,
                action="execute_graph",
                status="success",
                actor_id="test-user",
            )

            assert event.event_type == AuditEventType.GRAPH_EXECUTED
            assert event.action == "execute_graph"
            assert event.actor_id == "test-user"
