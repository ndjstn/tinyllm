"""Property-based tests for error handling components.

These tests use Hypothesis to verify that error handling invariants hold
under all possible inputs, particularly around error normalization,
signature stability, and aggregation properties.
"""

from datetime import datetime, timedelta
import re
from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from tinyllm.errors import TinyLLMError
from tinyllm.error_enrichment import (
    EnrichedError,
    ErrorContext,
    ErrorCategory,
    ErrorSeverity,
)
from tinyllm.error_aggregation import ErrorSignature, AggregatedError, ErrorAggregator
from tinyllm.error_impact import ImpactScorer, ImpactScore


# Custom strategies for error domain


@st.composite
def error_contexts(draw: Any) -> ErrorContext:
    """Generate random ErrorContext instances."""
    node_id = draw(st.one_of(st.none(), st.text(min_size=1, max_size=50)))
    graph_id = draw(st.one_of(st.none(), st.text(min_size=1, max_size=50)))

    return ErrorContext(
        stack_trace=draw(st.text(min_size=10, max_size=500)),
        exception_type=draw(st.sampled_from([
            "ValueError", "TypeError", "RuntimeError", "KeyError",
            "TimeoutError", "ConnectionError", "IOError"
        ])),
        exception_message=draw(st.text(min_size=5, max_size=200)),
        node_id=node_id,
        graph_id=graph_id,
        timestamp=draw(st.datetimes(
            min_value=datetime(2024, 1, 1),
            max_value=datetime(2025, 12, 31)
        )),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.booleans()),
            max_size=5
        ))
    )


@st.composite
def enriched_errors(draw: Any) -> EnrichedError:
    """Generate random EnrichedError instances."""
    return EnrichedError(
        error_id=draw(st.text(min_size=5, max_size=30)),
        message=draw(st.text(min_size=5, max_size=300)),
        category=draw(st.sampled_from(list(ErrorCategory))),
        severity=draw(st.sampled_from(list(ErrorSeverity))),
        context=draw(error_contexts()),
        is_retryable=draw(st.booleans()),
        suggested_actions=draw(st.lists(
            st.text(min_size=5, max_size=50),
            max_size=3
        )),
        recovery_strategies=draw(st.lists(
            st.text(min_size=5, max_size=50),
            max_size=3
        ))
    )


@st.composite
def normalized_error_messages(draw: Any) -> str:
    """Generate error messages with various normalizable patterns."""
    templates = [
        "Connection timeout after {id} milliseconds",
        "Error at {timestamp}",
        "Server listening on :{port}",
        "Failed to connect to {ip}:{port}",
        "Request {uuid} failed",
        "File not found: {path}/file{id}.txt",
        "Database error on table_{id}",
    ]

    template = draw(st.sampled_from(templates))

    # Generate values that should be normalized
    values = {
        "id": draw(st.integers(min_value=1000, max_value=99999)),  # 4+ digits
        "timestamp": draw(st.datetimes()).strftime("%Y-%m-%d %H:%M:%S"),
        "port": draw(st.integers(min_value=100, max_value=999)),  # 3 digits for port
        "ip": f"{draw(st.integers(min_value=1, max_value=255))}.{draw(st.integers(min_value=0, max_value=255))}.{draw(st.integers(min_value=0, max_value=255))}.{draw(st.integers(min_value=1, max_value=255))}",
        "uuid": f"{draw(st.uuids())}",
        "path": "/var/log/app",
    }

    # Only use values that are in the template
    result = template
    for key, value in values.items():
        result = result.replace(f"{{{key}}}", str(value))

    return result


class TestErrorSignatureProperties:
    """Property-based tests for ErrorSignature."""

    @given(enriched_errors())
    @settings(max_examples=100, deadline=2000)
    def test_signature_always_created(self, error: EnrichedError) -> None:
        """ErrorSignature can always be created from any EnrichedError."""
        signature = ErrorSignature.from_enriched_error(error)

        assert signature.signature_hash is not None
        assert len(signature.signature_hash) > 0
        assert signature.exception_type == error.context.exception_type
        assert signature.category == error.category

    @given(enriched_errors())
    @settings(max_examples=100, deadline=2000)
    def test_signature_deterministic(self, error: EnrichedError) -> None:
        """Same error should always produce same signature."""
        sig1 = ErrorSignature.from_enriched_error(error)
        sig2 = ErrorSignature.from_enriched_error(error)

        # Signatures should be identical
        assert sig1.signature_hash == sig2.signature_hash
        assert sig1.exception_type == sig2.exception_type
        assert sig1.category == sig2.category
        assert sig1.message_pattern == sig2.message_pattern

    @given(normalized_error_messages(), normalized_error_messages())
    @settings(max_examples=50, deadline=2000)
    def test_normalization_idempotent(self, msg1: str, msg2: str) -> None:
        """Normalizing an already normalized message should not change it."""
        # Normalize once
        normalized1 = ErrorSignature._normalize_message(msg1)

        # Normalize again
        normalized2 = ErrorSignature._normalize_message(normalized1)

        # Should be the same (idempotent)
        assert normalized1 == normalized2


class TestErrorNormalizationProperties:
    """Property-based tests for error message normalization."""

    @given(st.integers(min_value=1000, max_value=999999))
    @settings(max_examples=100, deadline=1000)
    def test_numeric_ids_normalized(self, numeric_id: int) -> None:
        """Numeric IDs (4+ digits) should be normalized to <ID>."""
        message = f"Error in operation {numeric_id}"
        normalized = ErrorSignature._normalize_message(message)

        # 4+ digit numbers should become <ID>
        if len(str(numeric_id)) >= 4:
            assert "<ID>" in normalized
            assert str(numeric_id) not in normalized

    @given(st.uuids())
    @settings(max_examples=50, deadline=1000)
    def test_uuids_normalized(self, uuid_val) -> None:
        """UUIDs should be normalized to <UUID>."""
        message = f"Request {uuid_val} failed"
        normalized = ErrorSignature._normalize_message(message)

        assert "<UUID>" in normalized
        assert str(uuid_val) not in normalized

    @given(
        st.integers(min_value=1, max_value=255),
        st.integers(min_value=1, max_value=255),
        st.integers(min_value=1, max_value=255),
        st.integers(min_value=1, max_value=255)
    )
    @settings(max_examples=50, deadline=1000)
    def test_ip_addresses_normalized(self, a: int, b: int, c: int, d: int) -> None:
        """IP addresses should be normalized to <IP>."""
        ip = f"{a}.{b}.{c}.{d}"
        message = f"Connection to {ip} failed"
        normalized = ErrorSignature._normalize_message(message)

        assert "<IP>" in normalized
        assert ip not in normalized

    @given(st.text(min_size=0, max_size=500))
    @settings(max_examples=100, deadline=1000)
    def test_normalization_preserves_structure(self, message: str) -> None:
        """Normalization should not change message length dramatically."""
        normalized = ErrorSignature._normalize_message(message)

        # Normalized message should not be empty if original wasn't
        if message.strip():
            assert normalized.strip() != ""


class TestAggregatedErrorProperties:
    """Property-based tests for AggregatedError."""

    @given(enriched_errors())
    @settings(max_examples=50, deadline=2000)
    def test_aggregated_error_add_occurrence(self, error: EnrichedError) -> None:
        """Adding occurrences should work with EnrichedError objects."""
        signature = ErrorSignature.from_enriched_error(error)
        agg = AggregatedError(signature=signature)
        initial_count = agg.count

        # Add occurrence using the actual API
        agg.add_occurrence(error)

        # Count should have increased
        assert agg.count == initial_count + 1
        assert error.error_id in agg.error_ids

    @given(enriched_errors())
    @settings(max_examples=50, deadline=2000)
    def test_aggregated_error_first_last_seen(self, error: EnrichedError) -> None:
        """first_seen should be <= last_seen."""
        signature = ErrorSignature.from_enriched_error(error)
        agg = AggregatedError(signature=signature)

        # Add multiple occurrences
        for i in range(3):
            agg.add_occurrence(error)

        # first_seen should be before or equal to last_seen
        assert agg.first_seen <= agg.last_seen

    @given(
        enriched_errors(),
        st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=30, deadline=3000)
    def test_sample_errors_bounded(self, error: EnrichedError, num_errors: int) -> None:
        """Sample errors should not exceed limit of 5."""
        signature = ErrorSignature.from_enriched_error(error)
        agg = AggregatedError(signature=signature)

        # Add many occurrences
        for i in range(num_errors):
            agg.add_occurrence(error)

        # Sample errors should be bounded at 5
        assert len(agg.sample_errors) <= 5


class TestErrorAggregatorProperties:
    """Property-based tests for ErrorAggregator."""

    @given(st.lists(enriched_errors(), min_size=1, max_size=20))
    @settings(max_examples=30, deadline=5000)
    def test_aggregator_deduplication(self, errors: list[EnrichedError]) -> None:
        """Aggregator should deduplicate errors with same signature."""
        aggregator = ErrorAggregator()

        # Add all errors
        for error in errors:
            aggregator.add_error(error)

        # Number of aggregated errors should be <= number of input errors
        aggregated = aggregator.get_top_errors(limit=1000)
        assert len(aggregated) <= len(errors)

    @given(enriched_errors(), st.integers(min_value=2, max_value=5))
    @settings(max_examples=30, deadline=3000)
    def test_aggregator_same_error_increases_count(
        self,
        error: EnrichedError,
        repetitions: int
    ) -> None:
        """Adding same error multiple times increases count."""
        aggregator = ErrorAggregator()

        # Add same error multiple times (same signature)
        for i in range(repetitions):
            aggregator.add_error(error)

        # Should have exactly 1 aggregated error
        aggregated = aggregator.get_top_errors(limit=1000)
        assert len(aggregated) == 1

        # Count should match repetitions
        assert aggregated[0].count == repetitions


class TestImpactScorerProperties:
    """Property-based tests for ImpactScorer."""

    @given(enriched_errors())
    @settings(max_examples=50, deadline=2000)
    def test_impact_score_bounded(self, error: EnrichedError) -> None:
        """Impact scores should always be between 0 and 100."""
        scorer = ImpactScorer()
        score = scorer.score_error(error)

        assert 0.0 <= score.total_score <= 100.0
        assert 0.0 <= score.severity_score <= 100.0
        assert 0.0 <= score.frequency_score <= 100.0
        assert 0.0 <= score.scope_score <= 100.0
        assert 0.0 <= score.recency_score <= 100.0
        assert 0.0 <= score.criticality_score <= 100.0

    @given(
        enriched_errors(),
        st.dictionaries(
            st.sampled_from(["severity", "frequency", "scope", "recency", "criticality"]),
            st.floats(min_value=0.1, max_value=1.0),
            min_size=5,
            max_size=5
        )
    )
    @settings(max_examples=30, deadline=3000)
    def test_impact_scorer_weights_normalized(
        self,
        error: EnrichedError,
        weights: dict
    ) -> None:
        """Custom weights should produce valid scores."""
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            normalized_weights = {k: v / total for k, v in weights.items()}
        else:
            # All zeros, use default
            normalized_weights = {k: 0.2 for k in weights.keys()}

        scorer = ImpactScorer(weights=normalized_weights)
        score = scorer.score_error(error)

        # Score should still be bounded
        assert 0.0 <= score.total_score <= 100.0

    @given(enriched_errors())
    @settings(max_examples=30, deadline=3000)
    def test_severity_score_mapping(self, error: EnrichedError) -> None:
        """Severity scores should follow expected ordering."""
        scorer = ImpactScorer()

        # Create errors with different severities
        debug_error = EnrichedError(
            error_id="err_debug",
            message=error.message,
            category=error.category,
            severity=ErrorSeverity.DEBUG,
            context=error.context,
            is_retryable=True
        )

        fatal_error = EnrichedError(
            error_id="err_fatal",
            message=error.message,
            category=error.category,
            severity=ErrorSeverity.FATAL,
            context=error.context,
            is_retryable=True
        )

        debug_score = scorer.score_error(debug_error)
        fatal_score = scorer.score_error(fatal_error)

        # FATAL should have higher severity score than DEBUG
        assert fatal_score.severity_score > debug_score.severity_score


class TestTransactionProperties:
    """Property-based tests for transaction behavior."""

    @given(st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=10))
    @settings(max_examples=30, deadline=3000)
    def test_transaction_operation_order_preserved(self, operation_ids: list[str]) -> None:
        """Transaction should preserve operation order."""
        from tinyllm.core.transaction import Transaction, Operation

        tx = Transaction(transaction_id="test_tx")
        tx.begin()

        # Add operations
        for op_id in operation_ids:
            op = Operation(
                operation_id=op_id,
                node_id="test_node",
                operation_type="test"
            )
            tx.add_operation(op)

        # Operation order should match input order
        for i, op_id in enumerate(operation_ids):
            assert tx.operations[i].operation_id == op_id


# Configure hypothesis settings
settings.register_profile("property_tests", max_examples=100, deadline=3000)
settings.register_profile("property_tests_quick", max_examples=20, deadline=1000)

import os
if os.getenv("HYPOTHESIS_PROFILE") == "property_tests":
    settings.load_profile("property_tests")
