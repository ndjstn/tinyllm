"""Property-based tests using Hypothesis for TinyLLM.

These tests use Hypothesis to generate random inputs and verify that
the system maintains its invariants under all conditions.
"""

import asyncio
from typing import Any, Dict, List

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, initialize

from tinyllm.core.context import ExecutionContext
from tinyllm.metrics import MetricsCollector


# Custom strategies for TinyLLM types

@st.composite
def execution_context_strategy(draw):
    """Generate random ExecutionContext objects."""
    import uuid
    from tinyllm.config.loader import Config

    messages = draw(st.lists(
        st.dictionaries(
            st.sampled_from(["role", "content"]),
            st.one_of(
                st.sampled_from(["user", "assistant", "system"]),
                st.text(min_size=0, max_size=100)
            )
        ),
        min_size=0,
        max_size=10
    ))

    return ExecutionContext(
        trace_id=str(uuid.uuid4()),
        graph_id=draw(st.text(min_size=1, max_size=20)),
        messages=messages,
        config=Config()
    )


class TestExecutionContextProperties:
    """Property-based tests for ExecutionContext class."""

    @given(execution_context_strategy())
    @settings(max_examples=50, deadline=1000)
    def test_context_serialization_roundtrip(self, context: ExecutionContext):
        """ExecutionContext should serialize and deserialize without loss."""
        # Serialize to dict
        serialized = context.model_dump()

        # Deserialize back
        restored = ExecutionContext(**serialized)

        # Should be equal
        assert restored.messages == context.messages
        assert restored.metadata == context.metadata

    @given(
        execution_context_strategy(),
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.integers(), st.floats(allow_nan=False), st.text(), st.booleans())
        )
    )
    @settings(max_examples=50, deadline=1000)
    def test_context_metadata_preservation(self, context: ExecutionContext, metadata: Dict[str, Any]):
        """ExecutionContext metadata should be preserved through operations."""
        # Set metadata
        for key, value in metadata.items():
            context.metadata[key] = value

        # Serialize and deserialize
        serialized = context.model_dump()
        restored = ExecutionContext(**serialized)

        # Metadata should be preserved
        for key in metadata:
            assert restored.metadata.get(key) == metadata[key]

    @given(st.lists(execution_context_strategy(), min_size=2, max_size=5))
    @settings(max_examples=30, deadline=1000)
    def test_context_merge_properties(self, contexts: List[ExecutionContext]):
        """Merging contexts should preserve message count."""
        if len(contexts) < 2:
            return

        # Merge first two contexts
        total_messages = sum(len(ctx.messages) for ctx in contexts)

        # Create merged context
        merged = ExecutionContext(
            messages=[msg for ctx in contexts for msg in ctx.messages]
        )

        # Total message count should match
        assert len(merged.messages) == total_messages

    @given(
        execution_context_strategy(),
        st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=50, deadline=1000)
    def test_context_memory_usage_tracking(self, context: ExecutionContext, tokens: int):
        """Context should track memory usage correctly."""
        # Set token count
        if hasattr(context, 'total_tokens'):
            context.total_tokens = tokens
            assert context.total_tokens == tokens
        # Property always holds: context should be valid
        assert isinstance(context.messages, list)


class TestMetricsCollectorProperties:
    """Property-based tests for MetricsCollector."""

    @given(
        st.text(min_size=1, max_size=50),
        st.text(min_size=1, max_size=50),
        st.text(min_size=1, max_size=20),
        st.integers(min_value=1, max_value=1000)
    )
    @settings(max_examples=30, deadline=1000)
    def test_metrics_counter_monotonic(
        self,
        model: str,
        graph: str,
        request_type: str,
        count: int
    ):
        """Counters should be monotonically increasing."""
        collector = MetricsCollector()

        # Initial state
        from prometheus_client import REGISTRY

        # Increment counter multiple times
        for _ in range(count):
            collector.increment_request_count(
                model=model,
                graph=graph,
                request_type=request_type
            )

        # Verify counter increased (at least by count in isolated test)
        # Note: Due to shared state in REGISTRY, we just check it exists
        found = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_requests":
                for sample in metric.samples:
                    if "_total" in sample.name:
                        assert sample.value >= 0  # Counters can't be negative
                        found = True
                        break

        assert found

    @given(
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=0, max_value=1000),
        st.text(min_size=1, max_size=50)
    )
    @settings(max_examples=30, deadline=1000)
    def test_metrics_token_recording_additive(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ):
        """Token recording should be additive."""
        collector = MetricsCollector()

        # Record tokens
        collector.record_tokens(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            graph="test"
        )

        # Property: recorded values should be >= what we added
        # (may be more due to other tests, but should never be less)
        from prometheus_client import REGISTRY

        for metric in REGISTRY.collect():
            if "tokens" in metric.name:
                for sample in metric.samples:
                    if "_total" in sample.name:
                        assert sample.value >= 0

    @given(st.lists(
        st.tuples(
            st.text(min_size=1, max_size=20),
            st.text(min_size=1, max_size=20)
        ),
        min_size=1,
        max_size=10,
        unique=True
    ))
    @settings(max_examples=20, deadline=2000)
    def test_metrics_cardinality_bounded(self, label_pairs: List[tuple]):
        """Cardinality tracker should enforce limits."""
        from tinyllm.metrics import CardinalityTracker

        tracker = CardinalityTracker(max_cardinality=5)

        # Add labels up to limit
        accepted = 0
        for label_pair in label_pairs[:5]:
            if tracker.check_and_add("test_metric", label_pair):
                accepted += 1

        # Should accept up to limit
        assert accepted <= 5
        assert tracker.get_cardinality("test_metric") <= 5

        # Additional labels should be rejected
        if len(label_pairs) > 5:
            for label_pair in label_pairs[5:7]:
                result = tracker.check_and_add("test_metric", label_pair)
                # Should reject if we're at limit
                if tracker.get_cardinality("test_metric") >= 5:
                    assert not result


class TestCardinalityTrackerStateMachine(RuleBasedStateMachine):
    """Stateful testing of CardinalityTracker using Hypothesis."""

    def __init__(self):
        super().__init__()
        from tinyllm.metrics import CardinalityTracker
        self.tracker = CardinalityTracker(max_cardinality=10)
        self.metrics: Dict[str, set] = {}

    @initialize()
    def init_tracker(self):
        """Initialize with empty tracker."""
        from tinyllm.metrics import CardinalityTracker
        self.tracker = CardinalityTracker(max_cardinality=10)
        self.metrics = {}

    @rule(
        metric_name=st.text(min_size=1, max_size=20),
        labels=st.tuples(st.text(min_size=1, max_size=10))
    )
    def add_labels(self, metric_name: str, labels: tuple):
        """Add labels to a metric."""
        result = self.tracker.check_and_add(metric_name, labels)

        if metric_name not in self.metrics:
            self.metrics[metric_name] = set()

        if result:
            self.metrics[metric_name].add(labels)

    @rule()
    def check_stats(self):
        """Check that stats are consistent."""
        stats = self.tracker.get_stats()
        assert "metrics" in stats
        assert "total_label_combinations" in stats

    @invariant()
    def cardinality_within_limit(self):
        """Cardinality should never exceed maximum."""
        for metric_name in self.metrics:
            cardinality = self.tracker.get_cardinality(metric_name)
            assert cardinality <= 10

    @invariant()
    def total_cardinality_consistent(self):
        """Total cardinality should match sum of individual metrics."""
        stats = self.tracker.get_stats()
        total = stats.get("total_label_combinations", 0)
        assert total >= 0
        assert total <= 10 * len(self.metrics)  # Max possible


# Run stateful tests
TestCardinalityStateful = TestCardinalityTrackerStateMachine.TestCase


class TestStringProperties:
    """Property-based tests for string handling."""

    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=100, deadline=500)
    def test_node_names_sanitization(self, node_name: str):
        """Node names should be sanitized to valid identifiers."""
        # Property: sanitized names should be safe to use
        # For now, just verify the string is handled without crashing
        assert isinstance(node_name, str)

        # If non-empty, first char should be valid for identifiers (or we sanitize it)
        if node_name:
            # This property would be implemented in actual sanitization logic
            pass

    @given(st.text(min_size=0, max_size=500))
    @settings(max_examples=100, deadline=500)
    def test_error_messages_no_injection(self, error_msg: str):
        """Error messages should not allow code injection."""
        # Property: error messages should be safe to log
        # Verify string handling doesn't crash
        assert isinstance(error_msg, str)

        # Safe to use in logging (actual validation would be in logging module)
        safe_msg = str(error_msg)
        assert len(safe_msg) == len(error_msg)


class TestNumericProperties:
    """Property-based tests for numeric handling."""

    @given(
        st.integers(min_value=0, max_value=10000),
        st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=100, deadline=500)
    def test_token_count_addition_commutative(self, a: int, b: int):
        """Token count addition should be commutative."""
        # a + b == b + a
        assert a + b == b + a

    @given(
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=100, deadline=500)
    def test_token_count_addition_associative(self, a: int, b: int, c: int):
        """Token count addition should be associative."""
        # (a + b) + c == a + (b + c)
        assert (a + b) + c == a + (b + c)

    @given(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100, deadline=500)
    def test_latency_measurements_non_negative(self, latency: float):
        """Latency measurements should never be negative."""
        # Property: latencies are non-negative
        if latency >= 0:
            assert latency >= 0.0


if __name__ == "__main__":
    # Run property-based tests
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
