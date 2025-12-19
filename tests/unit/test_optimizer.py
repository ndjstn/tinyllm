"""Tests for graph optimizer."""

import pytest

from tinyllm.config.graph import EdgeDefinition, GraphDefinition, NodeDefinition, NodeType
from tinyllm.core.graph import Graph
from tinyllm.core.optimizer import GraphOptimizer, ExecutionPlan
from tinyllm.nodes.transform import TransformNode


@pytest.fixture
def simple_linear_graph():
    """Create a simple linear graph."""
    graph_def = GraphDefinition(
        id="linear",
        version="1.0.0",
        name="Linear Test Graph",
        nodes=[
            NodeDefinition(id="input", type=NodeType.TRANSFORM, config={}),
            NodeDefinition(id="process", type=NodeType.TRANSFORM, config={}),
            NodeDefinition(id="output", type=NodeType.TRANSFORM, config={}),
        ],
        edges=[
            EdgeDefinition(from_node="input", to_node="process"),
            EdgeDefinition(from_node="process", to_node="output"),
        ],
        entry_points=["input"],
        exit_points=["output"],
    )

    graph = Graph(graph_def)
    for node_def in graph_def.nodes:
        graph.add_node(TransformNode(node_def))

    return graph


@pytest.fixture
def branching_graph():
    """Create a graph with conditional branches."""
    graph_def = GraphDefinition(
        id="branching",
        version="1.0.0",
        name="Branching Test Graph",
        nodes=[
            NodeDefinition(id="start", type=NodeType.TRANSFORM, config={}),
            NodeDefinition(id="branch_a", type=NodeType.TRANSFORM, config={}),
            NodeDefinition(id="branch_b", type=NodeType.TRANSFORM, config={}),
            NodeDefinition(id="end", type=NodeType.TRANSFORM, config={}),
        ],
        edges=[
            EdgeDefinition(from_node="start", to_node="branch_a", condition="route == 'a'"),
            EdgeDefinition(from_node="start", to_node="branch_b", condition="route == 'b'"),
            EdgeDefinition(from_node="branch_a", to_node="end"),
            EdgeDefinition(from_node="branch_b", to_node="end"),
        ],
        entry_points=["start"],
        exit_points=["end"],
    )

    graph = Graph(graph_def)
    for node_def in graph_def.nodes:
        graph.add_node(TransformNode(node_def))

    return graph


@pytest.fixture
def parallel_graph():
    """Create a graph with parallel execution paths."""
    graph_def = GraphDefinition(
        id="parallel",
        version="1.0.0",
        name="Parallel Test Graph",
        nodes=[
            NodeDefinition(id="start", type=NodeType.TRANSFORM, config={}),
            NodeDefinition(id="parallel_1", type=NodeType.TRANSFORM, config={}),
            NodeDefinition(id="parallel_2", type=NodeType.TRANSFORM, config={}),
            NodeDefinition(id="parallel_3", type=NodeType.TRANSFORM, config={}),
            NodeDefinition(id="end", type=NodeType.TRANSFORM, config={}),
        ],
        edges=[
            EdgeDefinition(from_node="start", to_node="parallel_1"),
            EdgeDefinition(from_node="start", to_node="parallel_2"),
            EdgeDefinition(from_node="start", to_node="parallel_3"),
            EdgeDefinition(from_node="parallel_1", to_node="end"),
            EdgeDefinition(from_node="parallel_2", to_node="end"),
            EdgeDefinition(from_node="parallel_3", to_node="end"),
        ],
        entry_points=["start"],
        exit_points=["end"],
    )

    graph = Graph(graph_def)
    for node_def in graph_def.nodes:
        graph.add_node(TransformNode(node_def))

    return graph


@pytest.fixture
def graph_with_unreachable():
    """Create a graph with unreachable nodes."""
    graph_def = GraphDefinition(
        id="unreachable",
        version="1.0.0",
        name="Unreachable Test Graph",
        nodes=[
            NodeDefinition(id="start", type=NodeType.TRANSFORM, config={}),
            NodeDefinition(id="reachable", type=NodeType.TRANSFORM, config={}),
            NodeDefinition(id="unreachable1", type=NodeType.TRANSFORM, config={}),
            NodeDefinition(id="unreachable2", type=NodeType.TRANSFORM, config={}),
            NodeDefinition(id="end", type=NodeType.TRANSFORM, config={}),
        ],
        edges=[
            EdgeDefinition(from_node="start", to_node="reachable"),
            EdgeDefinition(from_node="reachable", to_node="end"),
            # unreachable1 and unreachable2 have no path from start
        ],
        entry_points=["start"],
        exit_points=["end"],
    )

    graph = Graph(graph_def)
    for node_def in graph_def.nodes:
        graph.add_node(TransformNode(node_def))

    return graph


def test_optimizer_initialization():
    """Test optimizer can be initialized."""
    optimizer = GraphOptimizer()
    assert optimizer is not None
    assert optimizer._plan_cache == {}


def test_optimize_simple_graph(simple_linear_graph):
    """Test optimizing a simple linear graph."""
    optimizer = GraphOptimizer()
    result = optimizer.optimize(simple_linear_graph)

    assert result.original_nodes == 3
    assert result.optimized_nodes == 3
    assert len(result.passes) > 0
    assert result.execution_plan_hash is not None


def test_dead_code_elimination(graph_with_unreachable):
    """Test dead code elimination identifies unreachable nodes."""
    optimizer = GraphOptimizer()
    result = optimizer.optimize(graph_with_unreachable)

    # Should identify 2 unreachable nodes
    dead_code_pass = next(
        (p for p in result.passes if p.pass_name == "dead_code_elimination"), None
    )
    assert dead_code_pass is not None
    assert dead_code_pass.applied is True
    assert dead_code_pass.nodes_removed == 2


def test_node_fusion_detection(simple_linear_graph):
    """Test node fusion identifies sequential nodes."""
    optimizer = GraphOptimizer()
    result = optimizer.optimize(simple_linear_graph, aggressive=True)

    # Should identify fusable nodes
    fusion_pass = next((p for p in result.passes if p.pass_name == "node_fusion"), None)
    assert fusion_pass is not None
    # Linear graph should have fusable pairs


def test_lazy_evaluation_detection(branching_graph):
    """Test lazy evaluation identifies conditional branches."""
    optimizer = GraphOptimizer()
    result = optimizer.optimize(branching_graph)

    # Should identify lazy branches
    lazy_pass = next((p for p in result.passes if p.pass_name == "lazy_evaluation"), None)
    assert lazy_pass is not None
    assert lazy_pass.applied is True


def test_execution_plan_compilation(simple_linear_graph):
    """Test execution plan compilation."""
    optimizer = GraphOptimizer()
    plan = optimizer.compile_execution_plan(simple_linear_graph)

    assert plan.graph_id == "linear"
    assert len(plan.nodes_in_order) == 3
    assert plan.nodes_in_order == ["input", "process", "output"]
    assert plan.get_hash() is not None


def test_execution_plan_caching(simple_linear_graph):
    """Test execution plan caching."""
    optimizer = GraphOptimizer()

    # First compilation
    plan1 = optimizer.compile_execution_plan(simple_linear_graph)

    # Second compilation should use cache
    plan2 = optimizer.compile_execution_plan(simple_linear_graph)

    assert plan1.get_hash() == plan2.get_hash()
    assert len(optimizer._plan_cache) == 1


def test_parallel_group_identification(parallel_graph):
    """Test identification of parallel execution groups."""
    optimizer = GraphOptimizer()
    plan = optimizer.compile_execution_plan(parallel_graph)

    # Should identify parallel_1, parallel_2, parallel_3 as parallel
    assert len(plan.parallel_groups) > 0

    # Check if parallel nodes are grouped
    parallel_nodes = {"parallel_1", "parallel_2", "parallel_3"}
    found_parallel_group = False
    for group in plan.parallel_groups:
        if set(group) == parallel_nodes:
            found_parallel_group = True
            break

    assert found_parallel_group


def test_fused_node_groups(simple_linear_graph):
    """Test identification of fusable node groups."""
    optimizer = GraphOptimizer()
    plan = optimizer.compile_execution_plan(simple_linear_graph)

    # Linear graph should have at least one fusion group
    assert len(plan.fused_nodes) > 0

    # Check that fused nodes are sequential
    for group_id, nodes in plan.fused_nodes.items():
        assert len(nodes) > 1


def test_lazy_branches_identification(branching_graph):
    """Test identification of lazy evaluation branches."""
    optimizer = GraphOptimizer()
    plan = optimizer.compile_execution_plan(branching_graph)

    # Both branches should be marked as lazy
    assert len(plan.lazy_branches) > 0
    assert "branch_a" in plan.lazy_branches or "branch_b" in plan.lazy_branches


def test_execution_plan_hash_consistency():
    """Test execution plan hash is consistent."""
    plan = ExecutionPlan(
        graph_id="test",
        nodes_in_order=["a", "b", "c"],
        parallel_groups=[["b", "c"]],
        fused_nodes={"f1": ["a", "b"]},
        lazy_branches={"c"},
    )

    hash1 = plan.get_hash()
    hash2 = plan.get_hash()

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex digest


def test_execution_plan_hash_differs():
    """Test different plans have different hashes."""
    plan1 = ExecutionPlan(graph_id="test", nodes_in_order=["a", "b", "c"])
    plan2 = ExecutionPlan(graph_id="test", nodes_in_order=["c", "b", "a"])

    assert plan1.get_hash() != plan2.get_hash()


def test_get_plan_from_cache(simple_linear_graph):
    """Test retrieving plan from cache."""
    optimizer = GraphOptimizer()

    # Compile and cache
    plan1 = optimizer.compile_execution_plan(simple_linear_graph)

    # Retrieve from cache
    plan2 = optimizer.get_plan("linear")

    assert plan2 is not None
    assert plan2.graph_id == "linear"


def test_clear_cache(simple_linear_graph):
    """Test clearing the execution plan cache."""
    optimizer = GraphOptimizer()

    # Add some plans
    optimizer.compile_execution_plan(simple_linear_graph)
    assert len(optimizer._plan_cache) == 1

    # Clear cache
    optimizer.clear_cache()
    assert len(optimizer._plan_cache) == 0


def test_cache_stats(simple_linear_graph, branching_graph):
    """Test cache statistics."""
    optimizer = GraphOptimizer()

    optimizer.compile_execution_plan(simple_linear_graph)
    optimizer.compile_execution_plan(branching_graph)

    stats = optimizer.get_cache_stats()
    assert stats["cached_plans"] == 2
    assert len(stats["cache_keys"]) == 2


def test_optimize_with_aggressive_mode(simple_linear_graph):
    """Test aggressive optimization mode."""
    optimizer = GraphOptimizer()

    # Non-aggressive
    result1 = optimizer.optimize(simple_linear_graph, aggressive=False)

    # Aggressive
    result2 = optimizer.optimize(simple_linear_graph, aggressive=True)

    # Aggressive should run more passes
    assert len(result2.passes) >= len(result1.passes)


def test_optimization_result_structure():
    """Test optimization result has correct structure."""
    optimizer = GraphOptimizer()

    graph_def = GraphDefinition(
        id="test",
        version="1.0",
        nodes=[NodeDefinition(id="n1", type=NodeType.TRANSFORM, config={})],
        edges=[],
        entry_points=["n1"],
        exit_points=["n1"],
    )
    graph = Graph(graph_def)
    graph.add_node(TransformNode(graph_def.nodes[0]))

    result = optimizer.optimize(graph)

    assert result.original_nodes >= 0
    assert result.optimized_nodes >= 0
    assert isinstance(result.passes, list)
    assert result.execution_plan_hash is not None


def test_topological_order_with_cycles():
    """Test execution plan with cyclic graph."""
    graph_def = GraphDefinition(
        id="cyclic",
        version="1.0",
        nodes=[
            NodeDefinition(id="a", type=NodeType.TRANSFORM, config={}),
            NodeDefinition(id="b", type=NodeType.TRANSFORM, config={}),
        ],
        edges=[
            EdgeDefinition(from_node="a", to_node="b"),
            EdgeDefinition(from_node="b", to_node="a"),  # Cycle
        ],
        entry_points=["a"],
        exit_points=["b"],
        allow_cycles=True,
    )

    graph = Graph(graph_def)
    for node_def in graph_def.nodes:
        graph.add_node(TransformNode(node_def))

    optimizer = GraphOptimizer()
    plan = optimizer.compile_execution_plan(graph)

    # Should handle cycles gracefully
    assert plan.graph_id == "cyclic"
    assert len(plan.nodes_in_order) > 0
    assert plan.metadata["has_cycles"] is True


def test_empty_graph():
    """Test optimizing an empty graph."""
    graph_def = GraphDefinition(
        id="empty",
        version="1.0",
        nodes=[],
        edges=[],
        entry_points=[],
        exit_points=[],
    )

    graph = Graph(graph_def)
    optimizer = GraphOptimizer()

    result = optimizer.optimize(graph)
    assert result.original_nodes == 0
    assert result.optimized_nodes == 0


def test_single_node_graph():
    """Test optimizing a graph with a single node."""
    graph_def = GraphDefinition(
        id="single",
        version="1.0",
        nodes=[NodeDefinition(id="only", type=NodeType.TRANSFORM, config={})],
        edges=[],
        entry_points=["only"],
        exit_points=["only"],
    )

    graph = Graph(graph_def)
    graph.add_node(TransformNode(graph_def.nodes[0]))

    optimizer = GraphOptimizer()
    result = optimizer.optimize(graph)

    assert result.original_nodes == 1
    assert result.optimized_nodes == 1


def test_protected_nodes_not_removed(graph_with_unreachable):
    """Test that protected nodes are not removed."""
    # Mark unreachable1 as protected
    graph_with_unreachable._protected.add("unreachable1")

    optimizer = GraphOptimizer()
    result = optimizer.optimize(graph_with_unreachable)

    # Should only remove unreachable2, not unreachable1
    dead_code_pass = next(
        (p for p in result.passes if p.pass_name == "dead_code_elimination"), None
    )
    assert dead_code_pass is not None
    assert dead_code_pass.nodes_removed == 1  # Only unreachable2
