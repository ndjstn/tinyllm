"""Audit logging for TinyLLM.

This module provides comprehensive audit logging for security-relevant events,
compliance tracking, and operational transparency. Audit logs are immutable
and include cryptographic integrity verification.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.logging import get_logger

logger = get_logger(__name__, component="audit")


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication & Authorization
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    AUTH_TOKEN_CREATED = "auth.token.created"
    AUTH_TOKEN_REVOKED = "auth.token.revoked"
    ACCESS_GRANTED = "access.granted"
    ACCESS_DENIED = "access.denied"

    # Configuration Changes
    CONFIG_CHANGED = "config.changed"
    CONFIG_LOADED = "config.loaded"
    CONFIG_EXPORTED = "config.exported"

    # Graph Operations
    GRAPH_CREATED = "graph.created"
    GRAPH_MODIFIED = "graph.modified"
    GRAPH_DELETED = "graph.deleted"
    GRAPH_EXECUTED = "graph.executed"

    # Node Operations
    NODE_CREATED = "node.created"
    NODE_MODIFIED = "node.modified"
    NODE_DELETED = "node.deleted"
    NODE_EXECUTED = "node.executed"

    # Model Operations
    MODEL_LOADED = "model.loaded"
    MODEL_SWITCHED = "model.switched"
    MODEL_REQUEST = "model.request"
    MODEL_RESPONSE = "model.response"

    # Data Operations
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"

    # System Operations
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    SYSTEM_CONFIG_RELOAD = "system.config.reload"

    # Security Events
    SECURITY_VIOLATION = "security.violation"
    SECURITY_RATE_LIMIT = "security.rate_limit"
    SECURITY_INJECTION_DETECTED = "security.injection.detected"

    # Compliance Events
    COMPLIANCE_PII_ACCESS = "compliance.pii.access"
    COMPLIANCE_DATA_RETENTION = "compliance.data.retention"
    COMPLIANCE_EXPORT = "compliance.export"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    ALERT = "alert"


class AuditEvent(BaseModel):
    """An immutable audit event record."""

    # Event identification
    event_id: str = Field(..., description="Unique event identifier")
    event_type: AuditEventType = Field(..., description="Type of event")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    severity: AuditSeverity = Field(default=AuditSeverity.INFO)

    # Actor information
    actor_id: Optional[str] = Field(None, description="ID of user/service that triggered event")
    actor_type: Optional[str] = Field(None, description="Type of actor (user, service, system)")
    actor_ip: Optional[str] = Field(None, description="IP address of actor")

    # Resource information
    resource_type: Optional[str] = Field(None, description="Type of resource affected")
    resource_id: Optional[str] = Field(None, description="ID of resource affected")

    # Event details
    action: str = Field(..., description="Action performed")
    status: str = Field(..., description="Result status (success, failure, partial)")
    message: Optional[str] = Field(None, description="Human-readable message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional event details")

    # Context
    trace_id: Optional[str] = Field(None, description="Trace ID for correlation")
    session_id: Optional[str] = Field(None, description="Session ID")
    graph_id: Optional[str] = Field(None, description="Graph ID if applicable")
    node_id: Optional[str] = Field(None, description="Node ID if applicable")

    # Integrity
    previous_hash: Optional[str] = Field(None, description="Hash of previous event (chain)")
    event_hash: Optional[str] = Field(None, description="Hash of this event")

    def compute_hash(self, previous_hash: Optional[str] = None) -> str:
        """Compute cryptographic hash of this event.

        Args:
            previous_hash: Hash of previous event for chaining.

        Returns:
            SHA-256 hash of event data.
        """
        # Create canonical representation
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "actor_id": self.actor_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "status": self.status,
            "message": self.message,
            "previous_hash": previous_hash,
        }

        # Create deterministic JSON
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class AuditLogger:
    """Audit logger with integrity verification and persistence."""

    def __init__(
        self,
        log_file: Optional[Path] = None,
        enable_chain_verification: bool = True,
        buffer_size: int = 100,
    ):
        """Initialize audit logger.

        Args:
            log_file: Path to audit log file. If None, logs to stdout only.
            enable_chain_verification: Enable cryptographic chain verification.
            buffer_size: Number of events to buffer before flushing.
        """
        self.log_file = log_file
        self.enable_chain_verification = enable_chain_verification
        self.buffer_size = buffer_size
        self._buffer: List[AuditEvent] = []
        self._last_hash: Optional[str] = None
        self._event_count = 0

        # Ensure log directory exists
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "audit_logger_initialized",
            log_file=str(log_file) if log_file else None,
            chain_verification=enable_chain_verification,
        )

    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        status: str = "success",
        severity: AuditSeverity = AuditSeverity.INFO,
        actor_id: Optional[str] = None,
        actor_type: Optional[str] = None,
        actor_ip: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        graph_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> AuditEvent:
        """Log an audit event.

        Args:
            event_type: Type of event.
            action: Action performed.
            status: Result status.
            severity: Event severity.
            actor_id: ID of actor.
            actor_type: Type of actor.
            actor_ip: IP address.
            resource_type: Type of resource.
            resource_id: ID of resource.
            message: Human-readable message.
            details: Additional details.
            trace_id: Trace ID for correlation.
            session_id: Session ID.
            graph_id: Graph ID.
            node_id: Node ID.

        Returns:
            The created audit event.
        """
        # Generate event ID
        self._event_count += 1
        event_id = f"audit-{int(time.time() * 1000)}-{self._event_count}"

        # Create event
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            severity=severity,
            actor_id=actor_id,
            actor_type=actor_type,
            actor_ip=actor_ip,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            status=status,
            message=message,
            details=details,
            trace_id=trace_id,
            session_id=session_id,
            graph_id=graph_id,
            node_id=node_id,
            previous_hash=self._last_hash,
        )

        # Compute hash for integrity
        if self.enable_chain_verification:
            event.event_hash = event.compute_hash(self._last_hash)
            self._last_hash = event.event_hash

        # Log to structured logger
        bound_logger = logger.bind(
            event_id=event.event_id,
            event_type=event.event_type,
            severity=event.severity,
            actor_id=event.actor_id,
            resource_type=event.resource_type,
            resource_id=event.resource_id,
            trace_id=event.trace_id,
        )

        # Use appropriate log method based on severity
        log_data = {
            "action": event.action,
            "status": event.status,
            "message": event.message,
        }

        if event.severity in (AuditSeverity.DEBUG,):
            bound_logger.debug("audit_event", **log_data)
        elif event.severity in (AuditSeverity.INFO, AuditSeverity.NOTICE):
            bound_logger.info("audit_event", **log_data)
        elif event.severity == AuditSeverity.WARNING:
            bound_logger.warning("audit_event", **log_data)
        elif event.severity in (AuditSeverity.ERROR,):
            bound_logger.error("audit_event", **log_data)
        else:  # CRITICAL, ALERT
            bound_logger.critical("audit_event", **log_data)

        # Add to buffer
        self._buffer.append(event)

        # Flush if buffer is full
        if len(self._buffer) >= self.buffer_size:
            self.flush()

        return event

    def flush(self) -> None:
        """Flush buffered events to disk."""
        if not self._buffer:
            return

        if self.log_file:
            with open(self.log_file, "a") as f:
                for event in self._buffer:
                    f.write(event.model_dump_json() + "\n")

        logger.debug("audit_events_flushed", count=len(self._buffer))
        self._buffer.clear()

    def verify_chain(self, events: List[AuditEvent]) -> tuple[bool, Optional[str]]:
        """Verify integrity of audit event chain.

        Args:
            events: List of events to verify (in chronological order).

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not self.enable_chain_verification:
            return True, None

        if not events:
            return True, None

        previous_hash = None
        for i, event in enumerate(events):
            # Verify hash chain
            if event.previous_hash != previous_hash:
                return False, f"Event {i} ({event.event_id}): hash chain broken"

            # Verify event hash
            expected_hash = event.compute_hash(previous_hash)
            if event.event_hash != expected_hash:
                return False, f"Event {i} ({event.event_id}): hash mismatch"

            previous_hash = event.event_hash

        return True, None

    def read_events(
        self,
        event_type: Optional[AuditEventType] = None,
        actor_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[AuditEvent]:
        """Read and filter audit events from log file.

        Args:
            event_type: Filter by event type.
            actor_id: Filter by actor ID.
            resource_id: Filter by resource ID.
            trace_id: Filter by trace ID.
            start_time: Filter by start time.
            end_time: Filter by end time.
            limit: Maximum number of events to return.

        Returns:
            List of matching audit events.
        """
        if not self.log_file or not self.log_file.exists():
            return []

        events: List[AuditEvent] = []

        with open(self.log_file) as f:
            for line in f:
                try:
                    event = AuditEvent.model_validate_json(line.strip())

                    # Apply filters
                    if event_type and event.event_type != event_type:
                        continue
                    if actor_id and event.actor_id != actor_id:
                        continue
                    if resource_id and event.resource_id != resource_id:
                        continue
                    if trace_id and event.trace_id != trace_id:
                        continue

                    # Time filtering
                    if start_time or end_time:
                        event_time = datetime.fromisoformat(event.timestamp)
                        if start_time and event_time < start_time:
                            continue
                        if end_time and event_time > end_time:
                            continue

                    events.append(event)

                    if limit and len(events) >= limit:
                        break

                except Exception as e:
                    logger.warning("invalid_audit_event", error=str(e))

        return events

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics.

        Returns:
            Dictionary with statistics.
        """
        if not self.log_file or not self.log_file.exists():
            return {
                "total_events": 0,
                "file_size_bytes": 0,
                "chain_verification_enabled": self.enable_chain_verification,
            }

        total_events = 0
        event_types: Dict[str, int] = {}
        severities: Dict[str, int] = {}

        with open(self.log_file) as f:
            for line in f:
                try:
                    event = AuditEvent.model_validate_json(line.strip())
                    total_events += 1
                    event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
                    severities[event.severity] = severities.get(event.severity, 0) + 1
                except Exception:
                    pass

        return {
            "total_events": total_events,
            "file_size_bytes": self.log_file.stat().st_size,
            "event_types": event_types,
            "severities": severities,
            "chain_verification_enabled": self.enable_chain_verification,
        }

    def close(self) -> None:
        """Close audit logger and flush remaining events."""
        self.flush()
        logger.info("audit_logger_closed", events_logged=self._event_count)


# Global audit logger instance
_global_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(
    log_file: Optional[Path] = None,
    enable_chain_verification: bool = True,
) -> AuditLogger:
    """Get or create global audit logger instance.

    Args:
        log_file: Path to audit log file.
        enable_chain_verification: Enable hash chain verification.

    Returns:
        Global AuditLogger instance.
    """
    global _global_audit_logger
    if _global_audit_logger is None:
        if log_file is None:
            log_file = Path("logs/audit.log")
        _global_audit_logger = AuditLogger(
            log_file=log_file,
            enable_chain_verification=enable_chain_verification,
        )
    return _global_audit_logger


def audit_event(
    event_type: AuditEventType,
    action: str,
    **kwargs,
) -> AuditEvent:
    """Convenience function to log an audit event.

    Args:
        event_type: Type of event.
        action: Action performed.
        **kwargs: Additional event parameters.

    Returns:
        The created audit event.
    """
    logger = get_audit_logger()
    return logger.log_event(event_type, action, **kwargs)
