"""Test data generators for TinyLLM testing.

This module provides factories and generators for creating test data,
including messages, graphs, nodes, and execution contexts.
"""

import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from tinyllm.config.graph import (
    EdgeDefinition,
    GraphDefinition,
    NodeDefinition,
    NodeType,
)
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessagePayload
from tinyllm.models.client import GenerateResponse


class MessageGenerator:
    """Generate test messages."""

    @staticmethod
    def generate(
        trace_id: Optional[str] = None,
        source_node: Optional[str] = None,
        task: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Generate a test message.

        Args:
            trace_id: Trace ID (generated if None).
            source_node: Source node ID (generated if None).
            task: Task content (generated if None).
            content: Message content (generated if None).
            metadata: Additional metadata.

        Returns:
            Message instance.
        """
        return Message(
            trace_id=trace_id or f"trace-{uuid4().hex[:8]}",
            source_node=source_node or f"node-{uuid4().hex[:8]}",
            payload=MessagePayload(
                task=task or MessageGenerator.random_task(),
                content=content or MessageGenerator.random_content(),
                metadata=metadata or {},
            ),
        )

    @staticmethod
    def batch(count: int, **kwargs) -> List[Message]:
        """Generate a batch of messages.

        Args:
            count: Number of messages to generate.
            **kwargs: Arguments passed to generate().

        Returns:
            List of Message instances.
        """
        return [MessageGenerator.generate(**kwargs) for _ in range(count)]

    @staticmethod
    def random_task() -> str:
        """Generate a random task description."""
        tasks = [
            "Write a Python function to calculate factorial",
            "Explain quantum computing in simple terms",
            "Calculate the sum of 1 to 100",
            "Translate 'hello' to Spanish",
            "Generate a haiku about coding",
            "Debug this code snippet",
            "Optimize database query performance",
            "Create a REST API endpoint",
            "Write unit tests for authentication",
            "Design a caching strategy",
        ]
        return random.choice(tasks)

    @staticmethod
    def random_content(min_words: int = 5, max_words: int = 20) -> str:
        """Generate random content.

        Args:
            min_words: Minimum number of words.
            max_words: Maximum number of words.

        Returns:
            Random content string.
        """
        words = [
            "test", "data", "example", "sample", "mock",
            "python", "code", "function", "class", "method",
            "execute", "process", "generate", "compute", "analyze",
            "result", "output", "input", "value", "parameter",
        ]
        num_words = random.randint(min_words, max_words)
        return " ".join(random.choices(words, k=num_words))


class NodeGenerator:
    """Generate test nodes."""

    @staticmethod
    def generate(
        node_type: NodeType,
        node_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeDefinition:
        """Generate a node definition.

        Args:
            node_type: Type of node to generate.
            node_id: Node ID (generated if None).
            config: Node configuration (generated if None).

        Returns:
            NodeDefinition instance.
        """
        if node_id is None:
            node_id = f"{node_type.value}.{uuid4().hex[:8]}"

        if config is None:
            config = NodeGenerator.default_config(node_type)

        return NodeDefinition(
            id=node_id,
            type=node_type,
            config=config,
        )

    @staticmethod
    def default_config(node_type: NodeType) -> Dict[str, Any]:
        """Get default config for a node type.

        Args:
            node_type: Type of node.

        Returns:
            Default configuration dictionary.
        """
        configs = {
            NodeType.ENTRY: {},
            NodeType.EXIT: {"status": "success"},
            NodeType.MODEL: {
                "model": "qwen2.5:3b",
                "temperature": 0.7,
                "max_tokens": 1000,
            },
            NodeType.ROUTER: {
                "model": "qwen2.5:0.5b",
                "routes": [
                    {
                        "name": "code",
                        "description": "Code-related tasks",
                        "target": "code.specialist",
                    },
                    {
                        "name": "general",
                        "description": "General tasks",
                        "target": "general.specialist",
                    },
                ],
            },
            NodeType.TOOL: {"tool_id": "calculator"},
            NodeType.GATE: {
                "model": "qwen2.5:3b",
                "pass_threshold": 0.7,
            },
            NodeType.TRANSFORM: {
                "transforms": [{"type": "strip"}],
            },
            NodeType.LOOP: {
                "max_iterations": 3,
                "condition": "continue",
            },
            NodeType.FANOUT: {
                "target_nodes": ["node1", "node2"],
                "mode": "parallel",
            },
            NodeType.TIMEOUT: {
                "timeout_ms": 5000,
                "inner_node": "test.inner",
            },
            NodeType.REASONING: {
                "model": "qwen2.5:3b",
                "num_steps": 3,
            },
        }
        return configs.get(node_type, {})

    @staticmethod
    def batch(
        count: int,
        node_type: Optional[NodeType] = None,
        **kwargs
    ) -> List[NodeDefinition]:
        """Generate a batch of nodes.

        Args:
            count: Number of nodes to generate.
            node_type: Type of nodes (random if None).
            **kwargs: Arguments passed to generate().

        Returns:
            List of NodeDefinition instances.
        """
        nodes = []
        for _ in range(count):
            ntype = node_type or random.choice(list(NodeType))
            nodes.append(NodeGenerator.generate(ntype, **kwargs))
        return nodes


class GraphGenerator:
    """Generate test graphs."""

    @staticmethod
    def generate(
        graph_id: Optional[str] = None,
        num_nodes: int = 3,
        num_edges: Optional[int] = None,
    ) -> GraphDefinition:
        """Generate a graph definition.

        Args:
            graph_id: Graph ID (generated if None).
            num_nodes: Number of nodes (excluding entry/exit).
            num_edges: Number of edges (auto-calculated if None).

        Returns:
            GraphDefinition instance.
        """
        if graph_id is None:
            graph_id = f"graph-{uuid4().hex[:8]}"

        # Create entry and exit nodes
        nodes = [
            NodeDefinition(id="entry", type=NodeType.ENTRY, config={}),
            NodeDefinition(id="exit", type=NodeType.EXIT, config={"status": "success"}),
        ]

        # Create intermediate nodes
        for i in range(num_nodes):
            node = NodeGenerator.generate(
                node_type=NodeType.MODEL,
                node_id=f"node{i}",
            )
            nodes.append(node)

        # Create edges (simple linear flow by default)
        edges = []

        # Entry to first node
        edges.append(EdgeDefinition(
            source="entry",
            target=nodes[2].id if len(nodes) > 2 else "exit",
            condition=None,
        ))

        # Connect intermediate nodes
        for i in range(2, len(nodes) - 1):
            target = nodes[i + 1].id if i + 1 < len(nodes) - 1 else "exit"
            edges.append(EdgeDefinition(
                source=nodes[i].id,
                target=target,
                condition=None,
            ))

        return GraphDefinition(
            id=graph_id,
            nodes=nodes,
            edges=edges,
        )

    @staticmethod
    def linear(num_nodes: int = 3) -> GraphDefinition:
        """Generate a linear graph (entry -> n1 -> n2 -> ... -> exit).

        Args:
            num_nodes: Number of intermediate nodes.

        Returns:
            GraphDefinition instance.
        """
        return GraphGenerator.generate(num_nodes=num_nodes)

    @staticmethod
    def branching() -> GraphDefinition:
        """Generate a branching graph with conditional routing.

        Returns:
            GraphDefinition instance.
        """
        nodes = [
            NodeDefinition(id="entry", type=NodeType.ENTRY, config={}),
            NodeDefinition(
                id="router",
                type=NodeType.ROUTER,
                config={
                    "model": "qwen2.5:0.5b",
                    "routes": [
                        {
                            "name": "code",
                            "description": "Code tasks",
                            "target": "code_specialist",
                        },
                        {
                            "name": "math",
                            "description": "Math tasks",
                            "target": "math_specialist",
                        },
                    ],
                },
            ),
            NodeDefinition(
                id="code_specialist",
                type=NodeType.MODEL,
                config={"model": "granite-code:3b"},
            ),
            NodeDefinition(
                id="math_specialist",
                type=NodeType.MODEL,
                config={"model": "qwen2.5:3b"},
            ),
            NodeDefinition(id="exit", type=NodeType.EXIT, config={"status": "success"}),
        ]

        edges = [
            EdgeDefinition(source="entry", target="router", condition=None),
            EdgeDefinition(source="router", target="code_specialist", condition="code"),
            EdgeDefinition(source="router", target="math_specialist", condition="math"),
            EdgeDefinition(source="code_specialist", target="exit", condition=None),
            EdgeDefinition(source="math_specialist", target="exit", condition=None),
        ]

        return GraphDefinition(
            id="branching-graph",
            nodes=nodes,
            edges=edges,
        )

    @staticmethod
    def parallel() -> GraphDefinition:
        """Generate a graph with parallel execution.

        Returns:
            GraphDefinition instance.
        """
        nodes = [
            NodeDefinition(id="entry", type=NodeType.ENTRY, config={}),
            NodeDefinition(
                id="fanout",
                type=NodeType.FANOUT,
                config={
                    "target_nodes": ["worker1", "worker2", "worker3"],
                    "mode": "parallel",
                },
            ),
            NodeDefinition(
                id="worker1",
                type=NodeType.MODEL,
                config={"model": "qwen2.5:3b"},
            ),
            NodeDefinition(
                id="worker2",
                type=NodeType.MODEL,
                config={"model": "qwen2.5:3b"},
            ),
            NodeDefinition(
                id="worker3",
                type=NodeType.MODEL,
                config={"model": "qwen2.5:3b"},
            ),
            NodeDefinition(id="exit", type=NodeType.EXIT, config={"status": "success"}),
        ]

        edges = [
            EdgeDefinition(source="entry", target="fanout", condition=None),
            EdgeDefinition(source="fanout", target="worker1", condition=None),
            EdgeDefinition(source="fanout", target="worker2", condition=None),
            EdgeDefinition(source="fanout", target="worker3", condition=None),
            EdgeDefinition(source="worker1", target="exit", condition=None),
            EdgeDefinition(source="worker2", target="exit", condition=None),
            EdgeDefinition(source="worker3", target="exit", condition=None),
        ]

        return GraphDefinition(
            id="parallel-graph",
            nodes=nodes,
            edges=edges,
        )


class ContextGenerator:
    """Generate execution contexts."""

    @staticmethod
    def generate(
        trace_id: Optional[str] = None,
        graph_id: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> ExecutionContext:
        """Generate an execution context.

        Args:
            trace_id: Trace ID (generated if None).
            graph_id: Graph ID (generated if None).
            variables: Context variables.

        Returns:
            ExecutionContext instance.
        """
        from tinyllm.config.loader import Config

        return ExecutionContext(
            trace_id=trace_id or f"trace-{uuid4().hex[:8]}",
            graph_id=graph_id or f"graph-{uuid4().hex[:8]}",
            config=Config(),
            variables=variables or {},
        )


class ResponseGenerator:
    """Generate model responses."""

    @staticmethod
    def generate(
        model: str = "qwen2.5:3b",
        response_text: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> GenerateResponse:
        """Generate a model response.

        Args:
            model: Model name.
            response_text: Response text (generated if None).
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.

        Returns:
            GenerateResponse instance.
        """
        if response_text is None:
            response_text = ResponseGenerator.random_response()

        if prompt_tokens is None:
            prompt_tokens = random.randint(10, 100)

        if completion_tokens is None:
            completion_tokens = len(response_text.split())

        return GenerateResponse(
            model=model,
            created_at=datetime.utcnow().isoformat(),
            response=response_text,
            done=True,
            total_duration=random.randint(100, 500) * 1_000_000,  # nanoseconds
            load_duration=random.randint(5, 20) * 1_000_000,
            prompt_eval_count=prompt_tokens,
            prompt_eval_duration=random.randint(20, 100) * 1_000_000,
            eval_count=completion_tokens,
            eval_duration=random.randint(50, 300) * 1_000_000,
        )

    @staticmethod
    def random_response() -> str:
        """Generate a random response text."""
        responses = [
            "The result is 42.",
            "Here's a Python function:\n```python\ndef example():\n    return True\n```",
            "Based on the context, I would recommend...",
            "The answer to your question is yes.",
            "Let me break this down into steps: 1) First, 2) Then, 3) Finally.",
            "This is an interesting problem that requires...",
            "According to the documentation...",
            "The correct approach would be to...",
            "I've analyzed the data and found that...",
            "The optimal solution is...",
        ]
        return random.choice(responses)

    @staticmethod
    def batch(count: int, **kwargs) -> List[GenerateResponse]:
        """Generate a batch of responses.

        Args:
            count: Number of responses to generate.
            **kwargs: Arguments passed to generate().

        Returns:
            List of GenerateResponse instances.
        """
        return [ResponseGenerator.generate(**kwargs) for _ in range(count)]


class RandomDataGenerator:
    """Generate random data for testing."""

    @staticmethod
    def string(length: int = 10, charset: str = string.ascii_letters) -> str:
        """Generate a random string.

        Args:
            length: Length of string.
            charset: Character set to use.

        Returns:
            Random string.
        """
        return "".join(random.choices(charset, k=length))

    @staticmethod
    def alphanumeric(length: int = 10) -> str:
        """Generate a random alphanumeric string.

        Args:
            length: Length of string.

        Returns:
            Random alphanumeric string.
        """
        return RandomDataGenerator.string(length, string.ascii_letters + string.digits)

    @staticmethod
    def integer(min_val: int = 0, max_val: int = 1000) -> int:
        """Generate a random integer.

        Args:
            min_val: Minimum value.
            max_val: Maximum value.

        Returns:
            Random integer.
        """
        return random.randint(min_val, max_val)

    @staticmethod
    def float_value(min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Generate a random float.

        Args:
            min_val: Minimum value.
            max_val: Maximum value.

        Returns:
            Random float.
        """
        return random.uniform(min_val, max_val)

    @staticmethod
    def boolean() -> bool:
        """Generate a random boolean.

        Returns:
            Random boolean.
        """
        return random.choice([True, False])

    @staticmethod
    def timestamp(
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> datetime:
        """Generate a random timestamp.

        Args:
            start: Start time (defaults to 30 days ago).
            end: End time (defaults to now).

        Returns:
            Random datetime.
        """
        if end is None:
            end = datetime.utcnow()
        if start is None:
            start = end - timedelta(days=30)

        delta = end - start
        random_seconds = random.randint(0, int(delta.total_seconds()))
        return start + timedelta(seconds=random_seconds)

    @staticmethod
    def email() -> str:
        """Generate a random email address.

        Returns:
            Random email address.
        """
        username = RandomDataGenerator.alphanumeric(8).lower()
        domain = RandomDataGenerator.alphanumeric(6).lower()
        return f"{username}@{domain}.com"

    @staticmethod
    def url() -> str:
        """Generate a random URL.

        Returns:
            Random URL.
        """
        domain = RandomDataGenerator.alphanumeric(8).lower()
        path = RandomDataGenerator.alphanumeric(10).lower()
        return f"https://{domain}.com/{path}"

    @staticmethod
    def dict_data(
        keys: Optional[List[str]] = None,
        num_keys: int = 5,
    ) -> Dict[str, Any]:
        """Generate a random dictionary.

        Args:
            keys: List of keys to use (generated if None).
            num_keys: Number of keys if keys is None.

        Returns:
            Random dictionary.
        """
        if keys is None:
            keys = [f"key{i}" for i in range(num_keys)]

        data = {}
        for key in keys:
            value_type = random.choice(["string", "int", "float", "bool"])
            if value_type == "string":
                data[key] = RandomDataGenerator.string()
            elif value_type == "int":
                data[key] = RandomDataGenerator.integer()
            elif value_type == "float":
                data[key] = RandomDataGenerator.float_value()
            else:
                data[key] = RandomDataGenerator.boolean()

        return data

    @staticmethod
    def list_data(length: int = 5, item_type: str = "string") -> List[Any]:
        """Generate a random list.

        Args:
            length: Length of list.
            item_type: Type of items ("string", "int", "float", "bool").

        Returns:
            Random list.
        """
        generators = {
            "string": RandomDataGenerator.string,
            "int": RandomDataGenerator.integer,
            "float": RandomDataGenerator.float_value,
            "bool": RandomDataGenerator.boolean,
        }

        generator = generators.get(item_type, RandomDataGenerator.string)
        return [generator() for _ in range(length)]


# Convenience functions

def message(**kwargs) -> Message:
    """Generate a test message.

    Args:
        **kwargs: Arguments passed to MessageGenerator.generate().

    Returns:
        Message instance.
    """
    return MessageGenerator.generate(**kwargs)


def messages(count: int, **kwargs) -> List[Message]:
    """Generate test messages.

    Args:
        count: Number of messages.
        **kwargs: Arguments passed to MessageGenerator.generate().

    Returns:
        List of Message instances.
    """
    return MessageGenerator.batch(count, **kwargs)


def node(node_type: NodeType, **kwargs) -> NodeDefinition:
    """Generate a test node.

    Args:
        node_type: Type of node.
        **kwargs: Arguments passed to NodeGenerator.generate().

    Returns:
        NodeDefinition instance.
    """
    return NodeGenerator.generate(node_type, **kwargs)


def nodes(count: int, **kwargs) -> List[NodeDefinition]:
    """Generate test nodes.

    Args:
        count: Number of nodes.
        **kwargs: Arguments passed to NodeGenerator.batch().

    Returns:
        List of NodeDefinition instances.
    """
    return NodeGenerator.batch(count, **kwargs)


def graph(**kwargs) -> GraphDefinition:
    """Generate a test graph.

    Args:
        **kwargs: Arguments passed to GraphGenerator.generate().

    Returns:
        GraphDefinition instance.
    """
    return GraphGenerator.generate(**kwargs)


def context(**kwargs) -> ExecutionContext:
    """Generate an execution context.

    Args:
        **kwargs: Arguments passed to ContextGenerator.generate().

    Returns:
        ExecutionContext instance.
    """
    return ContextGenerator.generate(**kwargs)


def response(**kwargs) -> GenerateResponse:
    """Generate a model response.

    Args:
        **kwargs: Arguments passed to ResponseGenerator.generate().

    Returns:
        GenerateResponse instance.
    """
    return ResponseGenerator.generate(**kwargs)


def responses(count: int, **kwargs) -> List[GenerateResponse]:
    """Generate model responses.

    Args:
        count: Number of responses.
        **kwargs: Arguments passed to ResponseGenerator.generate().

    Returns:
        List of GenerateResponse instances.
    """
    return ResponseGenerator.batch(count, **kwargs)
