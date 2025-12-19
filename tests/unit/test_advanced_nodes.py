"""Tests for advanced node implementations (Router, Model, Tool, Gate)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.registry import NodeRegistry
from tinyllm.nodes.router import RouterNode, RouteDefinition, CompoundRoute
from tinyllm.nodes.model import ModelNode
from tinyllm.nodes.tool import ToolNode
from tinyllm.nodes.gate import GateNode, GateCondition


@pytest.fixture
def execution_context():
    """Create a test execution context."""
    return ExecutionContext(
        trace_id="test-trace-123",
        graph_id="test-graph",
        config=Config(),
    )


@pytest.fixture
def sample_message():
    """Create a sample test message."""
    return Message(
        trace_id="test-trace-123",
        source_node="test",
        payload=MessagePayload(
            task="Write a Python function to calculate factorial",
            content="factorial function",
        ),
    )


class TestRouterNode:
    """Tests for RouterNode."""

    def test_router_node_registered(self):
        """Test RouterNode is registered."""
        assert NodeRegistry.is_registered(NodeType.ROUTER)

    def test_create_router_node(self):
        """Test creating a router node from definition."""
        definition = NodeDefinition(
            id="router.test",
            type=NodeType.ROUTER,
            config={
                "model": "qwen2.5:0.5b",
                "routes": [
                    {"name": "code", "description": "Code tasks", "target": "model.code"},
                    {"name": "math", "description": "Math tasks", "target": "model.math"},
                ],
            },
        )
        node = RouterNode(definition)
        assert node.id == "router.test"
        assert len(node.router_config.routes) == 2

    def test_router_multi_label_config(self):
        """Test multi-label configuration."""
        definition = NodeDefinition(
            id="router.multi",
            type=NodeType.ROUTER,
            config={
                "multi_label": True,
                "max_labels": 3,
                "fanout_enabled": True,
                "routes": [
                    {"name": "code", "description": "Code", "target": "code", "priority": 2},
                    {"name": "math", "description": "Math", "target": "math", "priority": 1},
                ],
            },
        )
        node = RouterNode(definition)
        assert node.router_config.multi_label is True
        assert node.router_config.max_labels == 3
        assert node.router_config.fanout_enabled is True

    def test_router_compound_routes(self):
        """Test compound route configuration."""
        definition = NodeDefinition(
            id="router.compound",
            type=NodeType.ROUTER,
            config={
                "multi_label": True,
                "routes": [
                    {"name": "code", "description": "Code", "target": "code"},
                    {"name": "math", "description": "Math", "target": "math"},
                ],
                "compound_routes": [
                    {"domains": ["code", "math"], "target": "code_math", "priority": 10},
                ],
            },
        )
        node = RouterNode(definition)
        assert len(node.router_config.compound_routes) == 1
        assert "code" in node.router_config.compound_routes[0].domains

    def test_parse_route(self):
        """Test route parsing."""
        definition = NodeDefinition(
            id="router.test",
            type=NodeType.ROUTER,
            config={"routes": []},
        )
        node = RouterNode(definition)

        assert node._parse_route("code") == "code"
        assert node._parse_route("CODE") == "code"
        assert node._parse_route("code\n") == "code"
        assert node._parse_route("code!@#") == "code"

    def test_parse_multi_labels(self):
        """Test multi-label parsing."""
        definition = NodeDefinition(
            id="router.test",
            type=NodeType.ROUTER,
            config={
                "routes": [
                    {"name": "code", "description": "Code", "target": "code"},
                    {"name": "math", "description": "Math", "target": "math"},
                ],
            },
        )
        node = RouterNode(definition)

        # Comma-separated
        labels = node._parse_multi_labels("code, math")
        assert "code" in labels
        assert "math" in labels

    def test_find_compound_route(self):
        """Test compound route matching."""
        definition = NodeDefinition(
            id="router.test",
            type=NodeType.ROUTER,
            config={
                "routes": [
                    {"name": "code", "description": "Code", "target": "code"},
                    {"name": "math", "description": "Math", "target": "math"},
                ],
                "compound_routes": [
                    {"domains": ["code", "math"], "target": "code_math", "priority": 10},
                ],
            },
        )
        node = RouterNode(definition)

        # Should match compound route
        match = node._find_compound_route({"code", "math"})
        assert match is not None
        assert match.target == "code_math"

        # Should not match with only one domain
        no_match = node._find_compound_route({"code"})
        assert no_match is None


class TestModelNode:
    """Tests for ModelNode."""

    def test_model_node_registered(self):
        """Test ModelNode is registered."""
        assert NodeRegistry.is_registered(NodeType.MODEL)

    def test_create_model_node(self):
        """Test creating a model node."""
        definition = NodeDefinition(
            id="model.test",
            type=NodeType.MODEL,
            config={
                "model": "qwen2.5:3b",
                "temperature": 0.5,
                "system_prompt": "You are helpful.",
            },
        )
        node = ModelNode(definition)
        assert node.id == "model.test"
        assert node.model_config.model == "qwen2.5:3b"
        assert node.model_config.temperature == 0.5

    def test_model_node_default_config(self):
        """Test model node default configuration."""
        definition = NodeDefinition(
            id="model.default",
            type=NodeType.MODEL,
            config={},
        )
        node = ModelNode(definition)
        assert node.model_config.model == "qwen2.5:3b"
        assert node.model_config.temperature == 0.7
        assert node.model_config.stream is False

    def test_build_prompt(self, sample_message):
        """Test prompt building."""
        definition = NodeDefinition(
            id="model.test",
            type=NodeType.MODEL,
            config={},
        )
        node = ModelNode(definition)
        prompt = node._build_prompt(sample_message)
        assert "factorial" in prompt


class TestToolNode:
    """Tests for ToolNode."""

    def test_tool_node_registered(self):
        """Test ToolNode is registered."""
        assert NodeRegistry.is_registered(NodeType.TOOL)

    def test_create_tool_node(self):
        """Test creating a tool node."""
        definition = NodeDefinition(
            id="tool.calc",
            type=NodeType.TOOL,
            config={
                "tool_id": "calculator",
                "continue_on_error": True,
            },
        )
        node = ToolNode(definition)
        assert node.id == "tool.calc"
        assert node.tool_config.tool_id == "calculator"
        assert node.tool_config.continue_on_error is True

    def test_extract_tool_input(self):
        """Test tool input extraction."""
        definition = NodeDefinition(
            id="tool.test",
            type=NodeType.TOOL,
            config={
                "tool_id": "calculator",
                "input_mapping": {"expression": "content"},
            },
        )
        node = ToolNode(definition)
        message = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="2 + 2"),
        )

        tool_input = node._extract_tool_input(message)
        assert tool_input.get("expression") == "2 + 2"

    def test_get_nested_field(self):
        """Test nested field extraction."""
        definition = NodeDefinition(
            id="tool.test",
            type=NodeType.TOOL,
            config={"tool_id": "calculator"},
        )
        node = ToolNode(definition)

        data = {"level1": {"level2": {"value": 42}}}
        assert node._get_nested_field(data, "level1.level2.value") == 42
        assert node._get_nested_field(data, "nonexistent") is None


class TestGateNode:
    """Tests for GateNode."""

    def test_gate_node_registered(self):
        """Test GateNode is registered."""
        assert NodeRegistry.is_registered(NodeType.GATE)

    def test_create_gate_node(self):
        """Test creating a gate node."""
        definition = NodeDefinition(
            id="gate.test",
            type=NodeType.GATE,
            config={
                "mode": "expression",
                "conditions": [
                    {"name": "has_code", "expression": "'code' in content", "target": "code.handler"},
                ],
                "default_target": "general.handler",
            },
        )
        node = GateNode(definition)
        assert node.id == "gate.test"
        assert node.gate_config.mode == "expression"
        assert len(node.gate_config.conditions) == 1

    def test_gate_modes(self):
        """Test valid gate modes."""
        for mode in ["expression", "llm", "hybrid"]:
            definition = NodeDefinition(
                id=f"gate.{mode}",
                type=NodeType.GATE,
                config={"mode": mode},
            )
            node = GateNode(definition)
            assert node.gate_config.mode == mode

    def test_build_eval_context(self, sample_message, execution_context):
        """Test evaluation context building."""
        definition = NodeDefinition(
            id="gate.test",
            type=NodeType.GATE,
            config={"mode": "expression"},
        )
        node = GateNode(definition)

        ctx = node._build_eval_context(sample_message, execution_context)
        assert "content" in ctx
        assert "task" in ctx
        assert "len" in ctx  # Utility function

    @pytest.mark.asyncio
    async def test_expression_evaluation(self, sample_message, execution_context):
        """Test expression-based gate evaluation."""
        definition = NodeDefinition(
            id="gate.test",
            type=NodeType.GATE,
            config={
                "mode": "expression",
                "conditions": [
                    {"name": "has_factorial", "expression": "'factorial' in content", "target": "math.handler"},
                ],
                "default_target": "general.handler",
            },
        )
        node = GateNode(definition)

        result = await node._evaluate_expressions(sample_message, execution_context)
        assert result.success is True
        assert "math.handler" in result.next_nodes

    @pytest.mark.asyncio
    async def test_default_target_fallback(self, execution_context):
        """Test fallback to default target."""
        definition = NodeDefinition(
            id="gate.test",
            type=NodeType.GATE,
            config={
                "mode": "expression",
                "conditions": [
                    {"name": "never_match", "expression": "False", "target": "never"},
                ],
                "default_target": "default.handler",
            },
        )
        node = GateNode(definition)

        message = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="something"),
        )

        result = await node._evaluate_expressions(message, execution_context)
        assert result.success is True
        assert "default.handler" in result.next_nodes


class TestNodeRegistration:
    """Tests for comprehensive node registration."""

    def test_all_node_types_registered(self):
        """Test that all expected node types are registered."""
        expected_types = [
            NodeType.ENTRY,
            NodeType.EXIT,
            NodeType.ROUTER,
            NodeType.MODEL,
            NodeType.TOOL,
            NodeType.GATE,
        ]
        for node_type in expected_types:
            assert NodeRegistry.is_registered(node_type), f"{node_type} not registered"

    def test_create_nodes_from_registry(self):
        """Test creating all node types from registry."""
        definitions = [
            NodeDefinition(id="entry.test", type=NodeType.ENTRY, config={}),
            NodeDefinition(id="exit.test", type=NodeType.EXIT, config={}),
            NodeDefinition(id="router.test", type=NodeType.ROUTER, config={"routes": []}),
            NodeDefinition(id="model.test", type=NodeType.MODEL, config={}),
            NodeDefinition(id="tool.test", type=NodeType.TOOL, config={"tool_id": "calculator"}),
            NodeDefinition(id="gate.test", type=NodeType.GATE, config={"mode": "expression"}),
        ]

        for defn in definitions:
            node = NodeRegistry.create(defn)
            assert node is not None
            assert node.id == defn.id
