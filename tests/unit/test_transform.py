"""Tests for the transform node."""

import json
import pytest

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.message import Message, MessagePayload
from tinyllm.nodes.transform import (
    TransformNode,
    TransformType,
    TransformSpec,
    TransformPipeline,
    TransformResult,
    TransformNodeConfig,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_context():
    """Create a mock execution context."""
    class MockContext:
        trace_id = "test-trace"
        memory = None
    return MockContext()


@pytest.fixture
def base_message():
    """Create a base message for testing."""
    return Message(
        trace_id="trace-1",
        source_node="entry",
        payload=MessagePayload(content="Hello World", task="test task"),
    )


def make_transform_node(transforms: list) -> TransformNode:
    """Helper to create transform nodes."""
    definition = NodeDefinition(
        id="transform_test",
        type=NodeType.TRANSFORM,
        config={"transforms": transforms},
    )
    return TransformNode(definition)


# =============================================================================
# TRANSFORM SPEC TESTS
# =============================================================================


class TestTransformSpec:
    """Tests for TransformSpec model."""

    def test_create_basic_spec(self):
        """Test creating a basic transform spec."""
        spec = TransformSpec(
            transform_type=TransformType.UPPERCASE,
            params={},
        )
        assert spec.transform_type == TransformType.UPPERCASE
        assert spec.params == {}

    def test_spec_with_params(self):
        """Test spec with required params."""
        spec = TransformSpec(
            transform_type=TransformType.TRUNCATE,
            params={"max_length": 100},
        )
        assert spec.params["max_length"] == 100

    def test_truncate_requires_max_length(self):
        """Test that truncate requires max_length param."""
        with pytest.raises(ValueError, match="requires param 'max_length'"):
            TransformSpec(
                transform_type=TransformType.TRUNCATE,
                params={},
            )

    def test_regex_extract_requires_pattern(self):
        """Test that regex_extract requires pattern."""
        with pytest.raises(ValueError, match="requires param 'pattern'"):
            TransformSpec(
                transform_type=TransformType.REGEX_EXTRACT,
                params={},
            )

    def test_regex_replace_requires_both_params(self):
        """Test that regex_replace requires pattern and replacement."""
        with pytest.raises(ValueError):
            TransformSpec(
                transform_type=TransformType.REGEX_REPLACE,
                params={"pattern": ".*"},  # Missing replacement
            )

    def test_template_requires_template(self):
        """Test that template requires template param."""
        with pytest.raises(ValueError, match="requires param 'template'"):
            TransformSpec(
                transform_type=TransformType.TEMPLATE,
                params={},
            )

    def test_spec_is_frozen(self):
        """Test that spec is immutable."""
        spec = TransformSpec(
            transform_type=TransformType.UPPERCASE,
            params={},
        )
        with pytest.raises(Exception):
            spec.transform_type = TransformType.LOWERCASE


# =============================================================================
# TRANSFORM PIPELINE TESTS
# =============================================================================


class TestTransformPipeline:
    """Tests for TransformPipeline model."""

    def test_create_pipeline(self):
        """Test creating a transform pipeline."""
        pipeline = TransformPipeline(
            transforms=[
                TransformSpec(transform_type=TransformType.STRIP, params={}),
                TransformSpec(transform_type=TransformType.UPPERCASE, params={}),
            ]
        )
        assert len(pipeline.transforms) == 2

    def test_pipeline_requires_transforms(self):
        """Test that pipeline requires at least one transform."""
        with pytest.raises(ValueError):
            TransformPipeline(transforms=[])

    def test_pipeline_stop_on_error_default(self):
        """Test default stop_on_error is True."""
        pipeline = TransformPipeline(
            transforms=[
                TransformSpec(transform_type=TransformType.STRIP, params={}),
            ]
        )
        assert pipeline.stop_on_error is True


# =============================================================================
# TEXT TRANSFORMATION TESTS
# =============================================================================


class TestTextTransformations:
    """Tests for text transformation types."""

    @pytest.mark.asyncio
    async def test_uppercase(self, base_message, mock_context):
        """Test uppercase transformation."""
        node = make_transform_node([{"type": "uppercase"}])
        result = await node.execute(base_message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == "HELLO WORLD"

    @pytest.mark.asyncio
    async def test_lowercase(self, base_message, mock_context):
        """Test lowercase transformation."""
        node = make_transform_node([{"type": "lowercase"}])
        result = await node.execute(base_message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == "hello world"

    @pytest.mark.asyncio
    async def test_strip(self, mock_context):
        """Test strip transformation."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content="  hello  "),
        )
        node = make_transform_node([{"type": "strip"}])
        result = await node.execute(message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == "hello"

    @pytest.mark.asyncio
    async def test_truncate(self, mock_context):
        """Test truncate transformation."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content="This is a very long text"),
        )
        node = make_transform_node([
            {"type": "truncate", "params": {"max_length": 10}}
        ])
        result = await node.execute(message, mock_context)

        assert result.success
        output = result.output_messages[0].payload.content
        assert len(output) == 10
        assert output.endswith("...")

    @pytest.mark.asyncio
    async def test_truncate_short_text(self, mock_context):
        """Test truncate with text shorter than max_length."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content="short"),
        )
        node = make_transform_node([
            {"type": "truncate", "params": {"max_length": 100}}
        ])
        result = await node.execute(message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == "short"

    @pytest.mark.asyncio
    async def test_truncate_custom_suffix(self, mock_context):
        """Test truncate with custom suffix."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content="This is a very long text"),
        )
        node = make_transform_node([
            {"type": "truncate", "params": {"max_length": 15, "suffix": "[more]"}}
        ])
        result = await node.execute(message, mock_context)

        assert result.success
        output = result.output_messages[0].payload.content
        assert output.endswith("[more]")


# =============================================================================
# JSON TRANSFORMATION TESTS
# =============================================================================


class TestJsonTransformations:
    """Tests for JSON transformation types."""

    @pytest.mark.asyncio
    async def test_json_extract(self, mock_context):
        """Test JSON path extraction."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content='{"name": "John", "age": 30}'),
        )
        node = make_transform_node([
            {"type": "json_extract", "params": {"path": "name"}}
        ])
        result = await node.execute(message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == "John"

    @pytest.mark.asyncio
    async def test_json_extract_nested(self, mock_context):
        """Test nested JSON path extraction."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content='{"user": {"name": "John"}}'),
        )
        node = make_transform_node([
            {"type": "json_extract", "params": {"path": "user.name"}}
        ])
        result = await node.execute(message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == "John"

    @pytest.mark.asyncio
    async def test_json_extract_array(self, mock_context):
        """Test JSON array extraction."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content='{"items": ["a", "b", "c"]}'),
        )
        node = make_transform_node([
            {"type": "json_extract", "params": {"path": "items.1"}}
        ])
        result = await node.execute(message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == "b"

    @pytest.mark.asyncio
    async def test_json_wrap(self, base_message, mock_context):
        """Test JSON wrapping."""
        node = make_transform_node([
            {"type": "json_wrap", "params": {"key": "message"}}
        ])
        result = await node.execute(base_message, mock_context)

        assert result.success
        output = json.loads(result.output_messages[0].payload.content)
        assert output["message"] == "Hello World"

    @pytest.mark.asyncio
    async def test_json_wrap_default_key(self, base_message, mock_context):
        """Test JSON wrapping with default key."""
        node = make_transform_node([{"type": "json_wrap", "params": {}}])
        result = await node.execute(base_message, mock_context)

        assert result.success
        output = json.loads(result.output_messages[0].payload.content)
        assert output["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_json_parse_valid(self, mock_context):
        """Test JSON parse validation with valid JSON."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content='{"valid": true}'),
        )
        node = make_transform_node([{"type": "json_parse", "params": {}}])
        result = await node.execute(message, mock_context)

        assert result.success

    @pytest.mark.asyncio
    async def test_json_parse_invalid(self, mock_context):
        """Test JSON parse validation with invalid JSON."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content="not valid json"),
        )
        node = make_transform_node([{"type": "json_parse", "params": {}}])
        result = await node.execute(message, mock_context)

        assert not result.success

    @pytest.mark.asyncio
    async def test_json_stringify(self, mock_context):
        """Test JSON stringify."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content='{"a":1,"b":2}'),
        )
        node = make_transform_node([{"type": "json_stringify", "params": {}}])
        result = await node.execute(message, mock_context)

        assert result.success
        # Should be pretty printed
        assert "\n" in result.output_messages[0].payload.content


# =============================================================================
# REGEX TRANSFORMATION TESTS
# =============================================================================


class TestRegexTransformations:
    """Tests for regex transformation types."""

    @pytest.mark.asyncio
    async def test_regex_extract(self, mock_context):
        """Test regex extraction."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content="The answer is 42."),
        )
        node = make_transform_node([
            {"type": "regex_extract", "params": {"pattern": r"\d+"}}
        ])
        result = await node.execute(message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == "42"

    @pytest.mark.asyncio
    async def test_regex_extract_group(self, mock_context):
        """Test regex extraction with capture group."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content="Name: John Smith"),
        )
        node = make_transform_node([
            {"type": "regex_extract", "params": {"pattern": r"Name: (\w+)", "group": 1}}
        ])
        result = await node.execute(message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == "John"

    @pytest.mark.asyncio
    async def test_regex_extract_no_match(self, mock_context):
        """Test regex extraction with no match."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content="No numbers here"),
        )
        node = make_transform_node([
            {"type": "regex_extract", "params": {"pattern": r"\d+"}}
        ])
        result = await node.execute(message, mock_context)

        assert not result.success

    @pytest.mark.asyncio
    async def test_regex_replace(self, mock_context):
        """Test regex replacement."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content="Hello 123 World 456"),
        )
        node = make_transform_node([
            {"type": "regex_replace", "params": {"pattern": r"\d+", "replacement": "X"}}
        ])
        result = await node.execute(message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == "Hello X World X"


# =============================================================================
# STRUCTURAL TRANSFORMATION TESTS
# =============================================================================


class TestStructuralTransformations:
    """Tests for structural transformation types."""

    @pytest.mark.asyncio
    async def test_template(self, base_message, mock_context):
        """Test template transformation."""
        node = make_transform_node([
            {"type": "template", "params": {"template": "Message: {content}"}}
        ])
        result = await node.execute(base_message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == "Message: Hello World"

    @pytest.mark.asyncio
    async def test_split(self, mock_context):
        """Test split transformation."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content="a,b,c"),
        )
        node = make_transform_node([
            {"type": "split", "params": {"separator": ","}}
        ])
        result = await node.execute(message, mock_context)

        assert result.success
        output = json.loads(result.output_messages[0].payload.content)
        assert output == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_split_with_index(self, mock_context):
        """Test split with specific index."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content="a,b,c"),
        )
        node = make_transform_node([
            {"type": "split", "params": {"separator": ",", "index": 1}}
        ])
        result = await node.execute(message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == "b"

    @pytest.mark.asyncio
    async def test_join(self, mock_context):
        """Test join transformation."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content='["a", "b", "c"]'),
        )
        node = make_transform_node([
            {"type": "join", "params": {"separator": "-"}}
        ])
        result = await node.execute(message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == "a-b-c"


# =============================================================================
# PIPELINE TESTS
# =============================================================================


class TestTransformPipelines:
    """Tests for transform pipelines."""

    @pytest.mark.asyncio
    async def test_multiple_transforms(self, mock_context):
        """Test pipeline with multiple transforms."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content="  hello world  "),
        )
        node = make_transform_node([
            {"type": "strip"},
            {"type": "uppercase"},
        ])
        result = await node.execute(message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == "HELLO WORLD"

    @pytest.mark.asyncio
    async def test_pipeline_stop_on_error(self, mock_context):
        """Test pipeline stops on error when configured."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content="no numbers"),
        )
        definition = NodeDefinition(
            id="transform_test",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [
                    {"type": "regex_extract", "params": {"pattern": r"\d+"}},
                    {"type": "uppercase"},
                ],
                "stop_on_error": True,
            },
        )
        node = TransformNode(definition)
        result = await node.execute(message, mock_context)

        assert not result.success
        assert "REGEX_EXTRACT" in result.error

    @pytest.mark.asyncio
    async def test_complex_pipeline(self, mock_context):
        """Test complex pipeline."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content='{"text": "  hello  "}'),
        )
        node = make_transform_node([
            {"type": "json_extract", "params": {"path": "text"}},
            {"type": "strip"},
            {"type": "uppercase"},
            {"type": "json_wrap", "params": {"key": "result"}},
        ])
        result = await node.execute(message, mock_context)

        assert result.success
        output = json.loads(result.output_messages[0].payload.content)
        assert output["result"] == "HELLO"


# =============================================================================
# METADATA TESTS
# =============================================================================


class TestTransformMetadata:
    """Tests for transform node metadata."""

    @pytest.mark.asyncio
    async def test_metadata_includes_transform_count(self, base_message, mock_context):
        """Test that metadata includes transform count."""
        node = make_transform_node([
            {"type": "strip"},
            {"type": "uppercase"},
        ])
        result = await node.execute(base_message, mock_context)

        assert result.metadata["transforms_applied"] == 2

    @pytest.mark.asyncio
    async def test_metadata_includes_lengths(self, base_message, mock_context):
        """Test that metadata includes lengths."""
        node = make_transform_node([{"type": "uppercase"}])
        result = await node.execute(base_message, mock_context)

        assert "original_length" in result.metadata
        assert "output_length" in result.metadata

    @pytest.mark.asyncio
    async def test_output_payload_metadata(self, base_message, mock_context):
        """Test output message payload metadata."""
        node = make_transform_node([{"type": "uppercase"}])
        result = await node.execute(base_message, mock_context)

        payload = result.output_messages[0].payload
        assert payload.metadata["transformed"] is True
        assert payload.metadata["transforms_applied"] == 1


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_content(self, mock_context):
        """Test handling empty content."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(content=""),
        )
        node = make_transform_node([{"type": "uppercase"}])
        result = await node.execute(message, mock_context)

        assert result.success
        assert result.output_messages[0].payload.content == ""

    @pytest.mark.asyncio
    async def test_uses_task_if_no_content(self, mock_context):
        """Test that task is used if content is empty."""
        message = Message(
            trace_id="trace-1",
            source_node="entry",
            payload=MessagePayload(task="test task", content=""),
        )
        # When content is empty string, it's falsy so task is used
        # Actually the code uses `content or task or ""` so empty string is falsy
        node = make_transform_node([{"type": "uppercase"}])
        result = await node.execute(message, mock_context)

        assert result.success
        # Empty string is falsy, so it falls through to task
        assert result.output_messages[0].payload.content == "TEST TASK"

    @pytest.mark.asyncio
    async def test_custom_transform_not_implemented(self, base_message, mock_context):
        """Test that custom transform raises NotImplementedError."""
        node = make_transform_node([{"type": "custom", "params": {}}])
        result = await node.execute(base_message, mock_context)

        assert not result.success
        assert "not yet supported" in result.error.lower()
