"""Tests for test data generators."""

import pytest

from tinyllm.config.graph import GraphDefinition, NodeDefinition, NodeType
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message
from tinyllm.models.client import GenerateResponse
from tests.generators import (
    ContextGenerator,
    GraphGenerator,
    MessageGenerator,
    NodeGenerator,
    RandomDataGenerator,
    ResponseGenerator,
    context,
    graph,
    message,
    messages,
    node,
    nodes,
    response,
    responses,
)


class TestMessageGenerator:
    """Tests for MessageGenerator."""

    def test_generate_message(self):
        """Test generating a single message."""
        msg = MessageGenerator.generate()

        assert isinstance(msg, Message)
        assert msg.trace_id is not None
        assert msg.source_node is not None
        assert msg.payload.task is not None
        assert msg.payload.content is not None

    def test_generate_message_with_params(self):
        """Test generating message with custom parameters."""
        msg = MessageGenerator.generate(
            trace_id="custom-trace",
            source_node="custom-node",
            task="custom task",
            content="custom content",
            metadata={"key": "value"},
        )

        assert msg.trace_id == "custom-trace"
        assert msg.source_node == "custom-node"
        assert msg.payload.task == "custom task"
        assert msg.payload.content == "custom content"
        assert msg.payload.metadata["key"] == "value"

    def test_batch_messages(self):
        """Test generating batch of messages."""
        msgs = MessageGenerator.batch(5)

        assert len(msgs) == 5
        for msg in msgs:
            assert isinstance(msg, Message)

    def test_batch_messages_with_same_trace(self):
        """Test batch with same trace ID."""
        trace_id = "same-trace"
        msgs = MessageGenerator.batch(3, trace_id=trace_id)

        assert all(msg.trace_id == trace_id for msg in msgs)

    def test_random_task(self):
        """Test random task generation."""
        task = MessageGenerator.random_task()

        assert isinstance(task, str)
        assert len(task) > 0

    def test_random_content(self):
        """Test random content generation."""
        content = MessageGenerator.random_content()

        assert isinstance(content, str)
        assert len(content.split()) >= 5

    def test_random_content_with_bounds(self):
        """Test random content with word count bounds."""
        content = MessageGenerator.random_content(min_words=10, max_words=10)
        words = content.split()

        assert len(words) == 10

    def test_convenience_function(self):
        """Test convenience function."""
        msg = message()
        assert isinstance(msg, Message)

        msgs = messages(3)
        assert len(msgs) == 3


class TestNodeGenerator:
    """Tests for NodeGenerator."""

    def test_generate_node(self):
        """Test generating a single node."""
        node_def = NodeGenerator.generate(NodeType.MODEL)

        assert isinstance(node_def, NodeDefinition)
        assert node_def.type == NodeType.MODEL
        assert node_def.id is not None
        assert node_def.config is not None

    def test_generate_node_with_id(self):
        """Test generating node with custom ID."""
        node_def = NodeGenerator.generate(
            NodeType.MODEL,
            node_id="custom-node",
        )

        assert node_def.id == "custom-node"

    def test_generate_node_with_config(self):
        """Test generating node with custom config."""
        config = {"model": "custom-model", "temperature": 0.5}
        node_def = NodeGenerator.generate(
            NodeType.MODEL,
            config=config,
        )

        assert node_def.config["model"] == "custom-model"
        assert node_def.config["temperature"] == 0.5

    def test_default_config_for_all_types(self):
        """Test that default config exists for all node types."""
        for node_type in NodeType:
            config = NodeGenerator.default_config(node_type)
            assert isinstance(config, dict)

    def test_batch_nodes(self):
        """Test generating batch of nodes."""
        node_defs = NodeGenerator.batch(5, node_type=NodeType.MODEL)

        assert len(node_defs) == 5
        for node_def in node_defs:
            assert isinstance(node_def, NodeDefinition)
            assert node_def.type == NodeType.MODEL

    def test_batch_random_types(self):
        """Test batch with random node types."""
        node_defs = NodeGenerator.batch(10)

        assert len(node_defs) == 10
        types = {nd.type for nd in node_defs}
        # Should have some variety
        assert len(types) >= 1

    def test_convenience_function(self):
        """Test convenience functions."""
        node_def = node(NodeType.MODEL)
        assert isinstance(node_def, NodeDefinition)

        node_defs = nodes(3, node_type=NodeType.TOOL)
        assert len(node_defs) == 3
        assert all(nd.type == NodeType.TOOL for nd in node_defs)


class TestGraphGenerator:
    """Tests for GraphGenerator."""

    def test_generate_graph(self):
        """Test generating a graph."""
        graph_def = GraphGenerator.generate()

        assert isinstance(graph_def, GraphDefinition)
        assert graph_def.id is not None
        assert len(graph_def.nodes) >= 2  # At least entry and exit
        assert len(graph_def.edges) > 0

    def test_generate_graph_with_nodes(self):
        """Test generating graph with specific node count."""
        graph_def = GraphGenerator.generate(num_nodes=5)

        # 5 intermediate + entry + exit = 7 total
        assert len(graph_def.nodes) == 7

    def test_graph_has_entry_and_exit(self):
        """Test that generated graph has entry and exit nodes."""
        graph_def = GraphGenerator.generate()

        node_ids = {n.id for n in graph_def.nodes}
        assert "entry" in node_ids
        assert "exit" in node_ids

    def test_linear_graph(self):
        """Test linear graph generation."""
        graph_def = GraphGenerator.linear(num_nodes=3)

        assert isinstance(graph_def, GraphDefinition)
        assert len(graph_def.nodes) == 5  # entry + 3 + exit

    def test_branching_graph(self):
        """Test branching graph generation."""
        graph_def = GraphGenerator.branching()

        assert isinstance(graph_def, GraphDefinition)
        # Should have router and multiple branches
        node_types = [n.type for n in graph_def.nodes]
        assert NodeType.ROUTER in node_types

    def test_parallel_graph(self):
        """Test parallel graph generation."""
        graph_def = GraphGenerator.parallel()

        assert isinstance(graph_def, GraphDefinition)
        # Should have fanout node
        node_types = [n.type for n in graph_def.nodes]
        assert NodeType.FANOUT in node_types

    def test_convenience_function(self):
        """Test convenience function."""
        graph_def = graph(num_nodes=2)
        assert isinstance(graph_def, GraphDefinition)


class TestContextGenerator:
    """Tests for ContextGenerator."""

    def test_generate_context(self):
        """Test generating execution context."""
        ctx = ContextGenerator.generate()

        assert isinstance(ctx, ExecutionContext)
        assert ctx.trace_id is not None
        assert ctx.graph_id is not None
        assert ctx.config is not None

    def test_generate_context_with_params(self):
        """Test generating context with custom parameters."""
        ctx = ContextGenerator.generate(
            trace_id="custom-trace",
            graph_id="custom-graph",
            variables={"key": "value"},
        )

        assert ctx.trace_id == "custom-trace"
        assert ctx.graph_id == "custom-graph"
        assert ctx.variables["key"] == "value"

    def test_convenience_function(self):
        """Test convenience function."""
        ctx = context()
        assert isinstance(ctx, ExecutionContext)


class TestResponseGenerator:
    """Tests for ResponseGenerator."""

    def test_generate_response(self):
        """Test generating model response."""
        resp = ResponseGenerator.generate()

        assert isinstance(resp, GenerateResponse)
        assert resp.model is not None
        assert resp.response is not None
        assert resp.done is True

    def test_generate_response_with_params(self):
        """Test generating response with custom parameters."""
        resp = ResponseGenerator.generate(
            model="custom-model",
            response_text="custom response",
            prompt_tokens=50,
            completion_tokens=25,
        )

        assert resp.model == "custom-model"
        assert resp.response == "custom response"
        assert resp.prompt_eval_count == 50
        assert resp.eval_count == 25

    def test_random_response(self):
        """Test random response generation."""
        resp_text = ResponseGenerator.random_response()

        assert isinstance(resp_text, str)
        assert len(resp_text) > 0

    def test_batch_responses(self):
        """Test generating batch of responses."""
        resps = ResponseGenerator.batch(5)

        assert len(resps) == 5
        for resp in resps:
            assert isinstance(resp, GenerateResponse)

    def test_convenience_function(self):
        """Test convenience functions."""
        resp = response()
        assert isinstance(resp, GenerateResponse)

        resps = responses(3)
        assert len(resps) == 3


class TestRandomDataGenerator:
    """Tests for RandomDataGenerator."""

    def test_string(self):
        """Test random string generation."""
        s = RandomDataGenerator.string(length=20)

        assert isinstance(s, str)
        assert len(s) == 20

    def test_alphanumeric(self):
        """Test alphanumeric string generation."""
        s = RandomDataGenerator.alphanumeric(length=15)

        assert isinstance(s, str)
        assert len(s) == 15
        assert s.isalnum()

    def test_integer(self):
        """Test random integer generation."""
        i = RandomDataGenerator.integer(min_val=10, max_val=20)

        assert isinstance(i, int)
        assert 10 <= i <= 20

    def test_float_value(self):
        """Test random float generation."""
        f = RandomDataGenerator.float_value(min_val=0.5, max_val=1.5)

        assert isinstance(f, float)
        assert 0.5 <= f <= 1.5

    def test_boolean(self):
        """Test random boolean generation."""
        b = RandomDataGenerator.boolean()

        assert isinstance(b, bool)

    def test_timestamp(self):
        """Test random timestamp generation."""
        from datetime import datetime

        ts = RandomDataGenerator.timestamp()

        assert isinstance(ts, datetime)

    def test_email(self):
        """Test random email generation."""
        email = RandomDataGenerator.email()

        assert isinstance(email, str)
        assert "@" in email
        assert "." in email

    def test_url(self):
        """Test random URL generation."""
        url = RandomDataGenerator.url()

        assert isinstance(url, str)
        assert url.startswith("https://")

    def test_dict_data(self):
        """Test random dict generation."""
        data = RandomDataGenerator.dict_data(num_keys=5)

        assert isinstance(data, dict)
        assert len(data) == 5

    def test_dict_data_with_keys(self):
        """Test dict with custom keys."""
        keys = ["key1", "key2", "key3"]
        data = RandomDataGenerator.dict_data(keys=keys)

        assert len(data) == 3
        assert all(k in data for k in keys)

    def test_list_data(self):
        """Test random list generation."""
        lst = RandomDataGenerator.list_data(length=10, item_type="int")

        assert isinstance(lst, list)
        assert len(lst) == 10
        assert all(isinstance(item, int) for item in lst)

    def test_list_data_types(self):
        """Test list generation for all types."""
        for item_type in ["string", "int", "float", "bool"]:
            lst = RandomDataGenerator.list_data(length=5, item_type=item_type)
            assert len(lst) == 5


class TestGeneratorIntegration:
    """Integration tests for generators."""

    def test_generate_complete_test_scenario(self):
        """Test generating a complete test scenario."""
        # Create a graph
        graph_def = GraphGenerator.linear(num_nodes=2)

        # Create messages
        msgs = MessageGenerator.batch(3, trace_id="test-trace")

        # Create context
        ctx = ContextGenerator.generate(
            trace_id="test-trace",
            graph_id=graph_def.id,
        )

        # Create responses
        resps = ResponseGenerator.batch(2)

        # Verify everything is consistent
        assert all(msg.trace_id == "test-trace" for msg in msgs)
        assert ctx.trace_id == "test-trace"
        assert ctx.graph_id == graph_def.id
        assert len(resps) == 2

    def test_batch_generation_performance(self):
        """Test that batch generation is reasonably fast."""
        import time

        start = time.time()

        # Generate large batches
        MessageGenerator.batch(100)
        NodeGenerator.batch(50)
        ResponseGenerator.batch(100)

        duration = time.time() - start

        # Should complete in under 1 second
        assert duration < 1.0

    def test_uniqueness_of_ids(self):
        """Test that generated IDs are unique."""
        msgs = MessageGenerator.batch(100)
        trace_ids = [msg.trace_id for msg in msgs]

        # All trace IDs should be unique
        assert len(set(trace_ids)) == len(trace_ids)

    def test_reproducibility_with_seed(self):
        """Test that random generation can be seeded."""
        import random

        # Set seed
        random.seed(42)
        data1 = RandomDataGenerator.dict_data(num_keys=5)

        # Reset seed
        random.seed(42)
        data2 = RandomDataGenerator.dict_data(num_keys=5)

        # Should generate same data
        assert data1.keys() == data2.keys()
