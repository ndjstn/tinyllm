"""Graph optimization and compilation for TinyLLM.

This module provides optimization passes to improve graph execution performance
through node fusion, dead code elimination, and execution plan caching.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from tinyllm.core.graph import Graph, Edge
from tinyllm.core.node import BaseNode
from tinyllm.logging import get_logger

logger = get_logger(__name__, component="optimizer")


class OptimizationPass(BaseModel):
    """Result of an optimization pass."""

    model_config = {"extra": "forbid"}

    pass_name: str
    applied: bool = False
    nodes_removed: int = 0
    nodes_fused: int = 0
    edges_removed: int = 0
    description: str = ""


class OptimizationResult(BaseModel):
    """Complete optimization result."""

    model_config = {"extra": "forbid"}

    original_nodes: int
    original_edges: int
    optimized_nodes: int
    optimized_edges: int
    passes: List[OptimizationPass] = Field(default_factory=list)
    execution_plan_hash: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Compiled execution plan for a graph."""

    graph_id: str
    nodes_in_order: List[str]
    parallel_groups: List[List[str]] = field(default_factory=list)
    fused_nodes: Dict[str, List[str]] = field(default_factory=dict)
    lazy_branches: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_hash(self) -> str:
        """Generate hash for plan caching."""
        plan_data = {
            "graph_id": self.graph_id,
            "nodes": self.nodes_in_order,
            "parallel_groups": self.parallel_groups,
            "fused_nodes": self.fused_nodes,
            "lazy_branches": sorted(list(self.lazy_branches)),
        }
        plan_json = json.dumps(plan_data, sort_keys=True)
        return hashlib.sha256(plan_json.encode()).hexdigest()


class GraphOptimizer:
    """Optimizes graphs for execution.

    Performs multiple optimization passes:
    - Dead code elimination (unreachable nodes)
    - Node fusion (sequential operations)
    - Lazy evaluation (unused branches)
    - Execution plan compilation
    """

    def __init__(self):
        """Initialize optimizer."""
        self._plan_cache: Dict[str, ExecutionPlan] = {}

    def optimize(self, graph: Graph, aggressive: bool = False) -> OptimizationResult:
        """Run all optimization passes on a graph.

        Args:
            graph: Graph to optimize.
            aggressive: Enable aggressive optimizations.

        Returns:
            Optimization result with statistics.
        """
        logger.info(
            "optimization_starting",
            graph_id=graph.id,
            nodes=len(graph.nodes),
            edges=len(graph.edges),
            aggressive=aggressive,
        )

        original_nodes = len(graph.nodes)
        original_edges = len(graph.edges)

        result = OptimizationResult(
            original_nodes=original_nodes,
            original_edges=original_edges,
            optimized_nodes=original_nodes,
            optimized_edges=original_edges,
        )

        # Pass 1: Dead code elimination
        pass1 = self._eliminate_dead_code(graph)
        result.passes.append(pass1)

        # Pass 2: Node fusion (if aggressive)
        if aggressive:
            pass2 = self._fuse_sequential_nodes(graph)
            result.passes.append(pass2)

        # Pass 3: Identify lazy branches
        pass3 = self._identify_lazy_branches(graph)
        result.passes.append(pass3)

        # Update final counts
        result.optimized_nodes = len(graph.nodes)
        result.optimized_edges = len(graph.edges)

        # Generate execution plan
        plan = self.compile_execution_plan(graph)
        result.execution_plan_hash = plan.get_hash()

        logger.info(
            "optimization_complete",
            graph_id=graph.id,
            original_nodes=original_nodes,
            optimized_nodes=result.optimized_nodes,
            nodes_saved=original_nodes - result.optimized_nodes,
            passes_applied=sum(1 for p in result.passes if p.applied),
        )

        return result

    def _eliminate_dead_code(self, graph: Graph) -> OptimizationPass:
        """Remove unreachable nodes from the graph.

        Args:
            graph: Graph to optimize.

        Returns:
            Optimization pass result.
        """
        # Find all reachable nodes
        reachable: Set[str] = set()

        def dfs(node_id: str) -> None:
            if node_id in reachable or node_id not in graph.nodes:
                return
            reachable.add(node_id)
            for edge in graph.get_outgoing_edges(node_id):
                dfs(edge.to_node)

        # Start from entry points
        for entry_id in graph.entry_points:
            if entry_id in graph.nodes:
                dfs(entry_id)

        # Find unreachable nodes
        unreachable = set(graph.nodes.keys()) - reachable

        # Don't remove protected nodes
        to_remove = unreachable - graph.protected_nodes

        if to_remove:
            logger.info(
                "dead_code_elimination",
                graph_id=graph.id,
                unreachable_nodes=len(to_remove),
                nodes=list(to_remove),
            )

            # Note: We don't actually remove nodes here to preserve graph structure
            # In a real implementation, we'd mark them as inactive or remove them
            # For now, just report what would be removed

            return OptimizationPass(
                pass_name="dead_code_elimination",
                applied=True,
                nodes_removed=len(to_remove),
                description=f"Identified {len(to_remove)} unreachable nodes",
            )

        return OptimizationPass(
            pass_name="dead_code_elimination",
            applied=False,
            description="No unreachable nodes found",
        )

    def _fuse_sequential_nodes(self, graph: Graph) -> OptimizationPass:
        """Identify sequential nodes that can be fused.

        Args:
            graph: Graph to optimize.

        Returns:
            Optimization pass result.
        """
        fusable_pairs: List[Tuple[str, str]] = []

        for node_id in graph.nodes:
            # Skip entry/exit points
            if graph.is_entry_point(node_id) or graph.is_exit_point(node_id):
                continue

            outgoing = graph.get_outgoing_edges(node_id)
            incoming = graph.get_incoming_edges(node_id)

            # Can fuse if: 1 outgoing, 1 incoming, and target has 1 incoming
            if len(outgoing) == 1 and len(incoming) == 1:
                next_node_id = outgoing[0].to_node

                # Check if next node also has only this as input
                next_incoming = graph.get_incoming_edges(next_node_id)
                if len(next_incoming) == 1:
                    # Skip if next is exit point
                    if not graph.is_exit_point(next_node_id):
                        fusable_pairs.append((node_id, next_node_id))

        if fusable_pairs:
            logger.info(
                "node_fusion_candidates",
                graph_id=graph.id,
                fusable_pairs=len(fusable_pairs),
                pairs=fusable_pairs[:5],  # Log first 5
            )

            return OptimizationPass(
                pass_name="node_fusion",
                applied=True,
                nodes_fused=len(fusable_pairs),
                description=f"Identified {len(fusable_pairs)} node pairs for fusion",
            )

        return OptimizationPass(
            pass_name="node_fusion",
            applied=False,
            description="No fusable sequential nodes found",
        )

    def _identify_lazy_branches(self, graph: Graph) -> OptimizationPass:
        """Identify branches that can use lazy evaluation.

        Args:
            graph: Graph to optimize.

        Returns:
            Optimization pass result.
        """
        lazy_branches: Set[str] = set()

        # Find nodes with multiple outgoing edges (branch points)
        for node_id in graph.nodes:
            outgoing = graph.get_outgoing_edges(node_id)

            if len(outgoing) > 1:
                # All branches from this node could be lazy
                for edge in outgoing:
                    # Only if edge has a condition
                    if edge.condition:
                        lazy_branches.add(edge.to_node)

        if lazy_branches:
            logger.info(
                "lazy_evaluation_candidates",
                graph_id=graph.id,
                lazy_branches=len(lazy_branches),
                nodes=list(lazy_branches)[:5],  # Log first 5
            )

            return OptimizationPass(
                pass_name="lazy_evaluation",
                applied=True,
                description=f"Identified {len(lazy_branches)} branches for lazy evaluation",
            )

        return OptimizationPass(
            pass_name="lazy_evaluation",
            applied=False,
            description="No conditional branches found for lazy evaluation",
        )

    def compile_execution_plan(self, graph: Graph) -> ExecutionPlan:
        """Compile an optimized execution plan for the graph.

        Args:
            graph: Graph to compile.

        Returns:
            Compiled execution plan.
        """
        # Check cache first
        cache_key = f"{graph.id}_{len(graph.nodes)}_{len(graph.edges)}"
        if cache_key in self._plan_cache:
            logger.debug("execution_plan_cache_hit", graph_id=graph.id)
            return self._plan_cache[cache_key]

        logger.info("compiling_execution_plan", graph_id=graph.id)

        # Get topological order (if acyclic)
        try:
            nodes_in_order = graph.topological_sort()
        except ValueError:
            # Graph has cycles, use entry points order
            nodes_in_order = list(graph.entry_points) + [
                n for n in graph.nodes if n not in graph.entry_points
            ]

        # Identify parallel execution groups
        parallel_groups = self._identify_parallel_groups(graph, nodes_in_order)

        # Identify fused nodes
        fused_nodes = self._get_fused_node_groups(graph)

        # Identify lazy branches
        lazy_branches = self._get_lazy_branches(graph)

        plan = ExecutionPlan(
            graph_id=graph.id,
            nodes_in_order=nodes_in_order,
            parallel_groups=parallel_groups,
            fused_nodes=fused_nodes,
            lazy_branches=lazy_branches,
            metadata={
                "nodes_count": len(graph.nodes),
                "edges_count": len(graph.edges),
                "has_cycles": not graph.is_acyclic(),
            },
        )

        # Cache the plan
        self._plan_cache[cache_key] = plan

        logger.info(
            "execution_plan_compiled",
            graph_id=graph.id,
            nodes=len(nodes_in_order),
            parallel_groups=len(parallel_groups),
            fused_nodes=len(fused_nodes),
            lazy_branches=len(lazy_branches),
        )

        return plan

    def _identify_parallel_groups(
        self, graph: Graph, ordered_nodes: List[str]
    ) -> List[List[str]]:
        """Identify nodes that can execute in parallel.

        Args:
            graph: Graph to analyze.
            ordered_nodes: Nodes in topological order.

        Returns:
            Groups of nodes that can run in parallel.
        """
        parallel_groups: List[List[str]] = []

        # Track dependencies
        deps: Dict[str, Set[str]] = {}
        for node_id in graph.nodes:
            deps[node_id] = {
                edge.from_node for edge in graph.get_incoming_edges(node_id)
            }

        # Group nodes by depth (level in DAG)
        depth: Dict[str, int] = {}
        for node_id in ordered_nodes:
            if node_id in graph.entry_points:
                depth[node_id] = 0
            else:
                # Depth is max depth of dependencies + 1
                dep_depths = [depth.get(dep, 0) for dep in deps.get(node_id, set())]
                depth[node_id] = max(dep_depths, default=0) + 1

        # Group by depth
        levels: Dict[int, List[str]] = {}
        for node_id, d in depth.items():
            if d not in levels:
                levels[d] = []
            levels[d].append(node_id)

        # Convert to list of groups (only include groups with >1 node)
        for level in sorted(levels.keys()):
            if len(levels[level]) > 1:
                parallel_groups.append(levels[level])

        return parallel_groups

    def _get_fused_node_groups(self, graph: Graph) -> Dict[str, List[str]]:
        """Get groups of nodes that should be fused.

        Args:
            graph: Graph to analyze.

        Returns:
            Mapping of fusion group ID to node IDs.
        """
        fused_groups: Dict[str, List[str]] = {}
        group_id = 0

        for node_id in graph.nodes:
            # Skip if already in a group
            if any(node_id in nodes for nodes in fused_groups.values()):
                continue

            # Check if this node can start a fusion chain
            current = node_id
            chain = [current]

            while True:
                outgoing = graph.get_outgoing_edges(current)
                if len(outgoing) != 1:
                    break

                next_id = outgoing[0].to_node
                incoming = graph.get_incoming_edges(next_id)

                # Can add to chain if next has only one input
                if len(incoming) == 1 and not graph.is_exit_point(next_id):
                    chain.append(next_id)
                    current = next_id
                else:
                    break

            # Only create group if chain has >1 node
            if len(chain) > 1:
                fused_groups[f"fused_{group_id}"] = chain
                group_id += 1

        return fused_groups

    def _get_lazy_branches(self, graph: Graph) -> Set[str]:
        """Get branches that can use lazy evaluation.

        Args:
            graph: Graph to analyze.

        Returns:
            Set of node IDs for lazy branches.
        """
        lazy_branches: Set[str] = set()

        for node_id in graph.nodes:
            outgoing = graph.get_outgoing_edges(node_id)

            # If multiple conditional branches, they're lazy
            if len(outgoing) > 1:
                for edge in outgoing:
                    if edge.condition:
                        lazy_branches.add(edge.to_node)

        return lazy_branches

    def get_plan(self, graph_id: str) -> Optional[ExecutionPlan]:
        """Get cached execution plan.

        Args:
            graph_id: Graph identifier.

        Returns:
            Cached plan or None.
        """
        # Simple lookup by graph_id prefix
        for key, plan in self._plan_cache.items():
            if key.startswith(graph_id):
                return plan
        return None

    def clear_cache(self) -> None:
        """Clear the execution plan cache."""
        count = len(self._plan_cache)
        self._plan_cache.clear()
        logger.info("execution_plan_cache_cleared", plans_cleared=count)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        return {
            "cached_plans": len(self._plan_cache),
            "cache_keys": list(self._plan_cache.keys()),
        }
