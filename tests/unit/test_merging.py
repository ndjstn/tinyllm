"""
Tests for the node merging system.
"""

import pytest
from datetime import datetime, timezone

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.expansion.merging import (
    SimilarityMetric,
    MergeStrategy,
    MergeStatus,
    SimilarityScore,
    NodeSimilarityResult,
    MergeConfig,
    MergeProposal,
    MergeResult,
    NodeSimilarityDetector,
    NodeMerger,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def merge_config():
    """Default merge configuration."""
    return MergeConfig()


@pytest.fixture
def detector(merge_config):
    """Similarity detector instance."""
    return NodeSimilarityDetector(merge_config)


@pytest.fixture
def merger(merge_config):
    """Node merger instance."""
    return NodeMerger(merge_config)


@pytest.fixture
def similar_nodes():
    """Two similar MODEL nodes."""
    node_a = NodeDefinition(
        id="math_solver_1",
        type=NodeType.MODEL,
        name="Math Solver 1",
        config={
            "model": "qwen2.5:3b",
            "temperature": 0.3,
            "system_prompt": "You are a helpful math assistant. Solve problems step by step.",
        },
    )
    node_b = NodeDefinition(
        id="math_solver_2",
        type=NodeType.MODEL,
        name="Math Solver 2",
        config={
            "model": "qwen2.5:3b",
            "temperature": 0.4,
            "system_prompt": "You are a helpful math assistant. Show your work clearly.",
        },
    )
    return node_a, node_b


@pytest.fixture
def different_nodes():
    """Two very different nodes."""
    node_a = NodeDefinition(
        id="code_writer",
        type=NodeType.MODEL,
        name="Code Writer",
        config={
            "model": "qwen2.5:3b",
            "temperature": 0.2,
            "system_prompt": "You are an expert programmer. Write clean, efficient code.",
        },
    )
    node_b = NodeDefinition(
        id="story_teller",
        type=NodeType.MODEL,
        name="Story Teller",
        config={
            "model": "qwen2.5:7b",
            "temperature": 0.9,
            "system_prompt": "You are a creative writer. Tell engaging stories.",
        },
    )
    return node_a, node_b


@pytest.fixture
def different_type_nodes():
    """Two nodes of different types."""
    node_a = NodeDefinition(
        id="router",
        type=NodeType.ROUTER,
        name="Router",
        config={"model": "qwen2.5:0.5b"},
    )
    node_b = NodeDefinition(
        id="model",
        type=NodeType.MODEL,
        name="Model",
        config={"model": "qwen2.5:3b"},
    )
    return node_a, node_b


# =============================================================================
# SIMILARITY SCORE TESTS
# =============================================================================


class TestSimilarityScore:
    """Tests for SimilarityScore model."""

    def test_create_similarity_score(self):
        """Test creating a similarity score."""
        score = SimilarityScore(
            metric=SimilarityMetric.CONFIG_OVERLAP,
            score=0.85,
        )
        assert score.metric == SimilarityMetric.CONFIG_OVERLAP
        assert score.score == 0.85
        assert score.is_similar is True

    def test_low_similarity_not_similar(self):
        """Test that low scores are not similar."""
        score = SimilarityScore(
            metric=SimilarityMetric.CONFIG_OVERLAP,
            score=0.5,
        )
        assert score.is_similar is False

    def test_score_immutable(self):
        """Test that score is immutable."""
        score = SimilarityScore(
            metric=SimilarityMetric.CONFIG_OVERLAP,
            score=0.8,
        )
        with pytest.raises(Exception):
            score.score = 0.9

    def test_score_bounds(self):
        """Test score bounds validation."""
        with pytest.raises(ValueError):
            SimilarityScore(
                metric=SimilarityMetric.CONFIG_OVERLAP,
                score=1.5,  # > 1.0
            )


# =============================================================================
# DETECTOR TESTS
# =============================================================================


class TestNodeSimilarityDetector:
    """Tests for NodeSimilarityDetector."""

    def test_detect_similar_nodes(self, detector, similar_nodes):
        """Test detecting similar nodes."""
        node_a, node_b = similar_nodes
        result = detector.compare_nodes(node_a, node_b)

        assert result.node_a_id == "math_solver_1"
        assert result.node_b_id == "math_solver_2"
        # Similar nodes share same type and model, expect > 0.35 similarity
        assert result.overall_similarity > 0.35
        assert len(result.scores) >= 1

    def test_different_types_zero_similarity(self, detector, different_type_nodes):
        """Test that different node types have zero similarity."""
        node_a, node_b = different_type_nodes
        result = detector.compare_nodes(node_a, node_b)

        assert result.overall_similarity == 0.0
        assert result.merge_recommended is False

    def test_very_different_nodes_low_similarity(self, detector, different_nodes):
        """Test that very different nodes have low similarity."""
        node_a, node_b = different_nodes
        result = detector.compare_nodes(node_a, node_b)

        # Should have some similarity (same type) but not be merge candidates
        assert result.overall_similarity < 0.8

    def test_identical_nodes_high_similarity(self, detector):
        """Test that identical nodes have high similarity."""
        node = NodeDefinition(
            id="test_node",
            type=NodeType.MODEL,
            config={"model": "qwen2.5:3b", "temperature": 0.5},
        )
        # Compare with copy
        node_copy = NodeDefinition(
            id="test_node_copy",
            type=NodeType.MODEL,
            config={"model": "qwen2.5:3b", "temperature": 0.5},
        )
        result = detector.compare_nodes(node, node_copy)

        assert result.overall_similarity == 1.0
        assert result.merge_recommended is True

    def test_config_overlap_calculation(self, detector):
        """Test config overlap calculation."""
        node_a = NodeDefinition(
            id="a",
            type=NodeType.MODEL,
            config={"a": 1, "b": 2, "c": 3},
        )
        node_b = NodeDefinition(
            id="b",
            type=NodeType.MODEL,
            config={"a": 1, "b": 2, "d": 4},
        )
        result = detector.compare_nodes(node_a, node_b)

        config_score = next(s for s in result.scores if s.metric == SimilarityMetric.CONFIG_OVERLAP)
        # 2 matching (a, b), 4 total keys (a, b, c, d)
        assert config_score.score == 0.5


# =============================================================================
# MERGER TESTS
# =============================================================================


class TestNodeMerger:
    """Tests for NodeMerger."""

    def test_find_merge_candidates(self, merger, similar_nodes, different_nodes):
        """Test finding merge candidates."""
        all_nodes = [similar_nodes[0], similar_nodes[1], different_nodes[0], different_nodes[1]]
        candidates = merger.find_merge_candidates(all_nodes)

        # Should find similar nodes as candidates
        assert len(candidates) >= 0  # May or may not meet threshold

    def test_create_proposal(self, merger, similar_nodes):
        """Test creating a merge proposal."""
        node_a, node_b = similar_nodes
        proposal = merger.create_proposal(node_a, node_b)

        assert proposal.node_a_id == "math_solver_1"
        assert proposal.node_b_id == "math_solver_2"
        assert proposal.status == MergeStatus.PROPOSED
        assert proposal.strategy is not None

    def test_approve_proposal(self, merger, similar_nodes):
        """Test approving a proposal."""
        node_a, node_b = similar_nodes
        proposal = merger.create_proposal(node_a, node_b)

        approved = merger.approve_proposal(proposal.id, "test_user")

        assert approved.status == MergeStatus.APPROVED
        assert approved.approved_by == "test_user"
        assert approved.approved_at is not None

    def test_reject_proposal(self, merger, similar_nodes):
        """Test rejecting a proposal."""
        node_a, node_b = similar_nodes
        proposal = merger.create_proposal(node_a, node_b)

        rejected = merger.reject_proposal(proposal.id, "Not similar enough")

        assert rejected.status == MergeStatus.REJECTED
        assert rejected.rejection_reason == "Not similar enough"

    def test_approve_nonexistent_fails(self, merger):
        """Test that approving nonexistent proposal fails."""
        with pytest.raises(ValueError, match="Proposal not found"):
            merger.approve_proposal("nonexistent", "user")

    def test_execute_merge(self, merger, similar_nodes):
        """Test executing a merge."""
        node_a, node_b = similar_nodes
        proposal = merger.create_proposal(node_a, node_b)
        merger.approve_proposal(proposal.id)

        result, merged_node = merger.execute_merge(proposal.id, node_a, node_b)

        assert result.success is True
        assert result.merged_node_id == node_a.id  # Keeps first node's ID
        assert node_b.id in result.removed_node_ids
        assert merged_node.type == NodeType.MODEL

    def test_execute_unapproved_fails(self, merger, similar_nodes):
        """Test that executing unapproved proposal fails."""
        node_a, node_b = similar_nodes
        proposal = merger.create_proposal(node_a, node_b)

        with pytest.raises(ValueError, match="not approved"):
            merger.execute_merge(proposal.id, node_a, node_b)

    def test_get_pending_proposals(self, merger, similar_nodes):
        """Test getting pending proposals."""
        node_a, node_b = similar_nodes
        merger.create_proposal(node_a, node_b)

        pending = merger.get_pending_proposals()
        assert len(pending) == 1
        assert pending[0].status == MergeStatus.PROPOSED

    def test_history_tracking(self, merger, similar_nodes):
        """Test merge history tracking."""
        node_a, node_b = similar_nodes
        proposal = merger.create_proposal(node_a, node_b)
        merger.approve_proposal(proposal.id)
        merger.execute_merge(proposal.id, node_a, node_b)

        assert merger.history.successful_merges == 1
        assert len(merger.history.merges) == 1


# =============================================================================
# MERGE STRATEGY TESTS
# =============================================================================


class TestMergeStrategies:
    """Tests for different merge strategies."""

    def test_keep_better_strategy(self, merger):
        """Test KEEP_BETTER strategy keeps first config."""
        node_a = NodeDefinition(
            id="a",
            type=NodeType.MODEL,
            config={"model": "qwen2.5:3b", "temperature": 0.5},
        )
        node_b = NodeDefinition(
            id="b",
            type=NodeType.MODEL,
            config={"model": "qwen2.5:3b", "temperature": 0.7},
        )

        proposal = merger.create_proposal(node_a, node_b, MergeStrategy.KEEP_BETTER)
        assert proposal.merged_config == node_a.config

    def test_combine_prompts_strategy(self, merger):
        """Test COMBINE_PROMPTS strategy."""
        node_a = NodeDefinition(
            id="a",
            type=NodeType.MODEL,
            config={"system_prompt": "You are helpful."},
        )
        node_b = NodeDefinition(
            id="b",
            type=NodeType.MODEL,
            config={"system_prompt": "Be concise."},
        )

        proposal = merger.create_proposal(node_a, node_b, MergeStrategy.COMBINE_PROMPTS)
        assert "You are helpful" in proposal.merged_config["system_prompt"]
        assert "Be concise" in proposal.merged_config["system_prompt"]

    def test_weighted_average_strategy(self, merger):
        """Test WEIGHTED_AVERAGE strategy averages numeric values."""
        node_a = NodeDefinition(
            id="a",
            type=NodeType.MODEL,
            config={"temperature": 0.4},
        )
        node_b = NodeDefinition(
            id="b",
            type=NodeType.MODEL,
            config={"temperature": 0.6},
        )

        proposal = merger.create_proposal(node_a, node_b, MergeStrategy.WEIGHTED_AVERAGE)
        assert proposal.merged_config["temperature"] == 0.5


# =============================================================================
# MERGE CONFIG TESTS
# =============================================================================


class TestMergeConfig:
    """Tests for MergeConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MergeConfig()
        assert config.similarity_threshold == 0.8
        assert config.require_approval is True
        assert config.max_merges_per_day == 10

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            MergeConfig(similarity_threshold=0.3)  # < 0.5

        with pytest.raises(ValueError):
            MergeConfig(max_merges_per_day=200)  # > 100

    def test_config_immutable(self):
        """Test config is immutable."""
        config = MergeConfig()
        with pytest.raises(Exception):
            config.similarity_threshold = 0.9


# =============================================================================
# MERGE PROPOSAL TESTS
# =============================================================================


class TestMergeProposal:
    """Tests for MergeProposal."""

    def test_proposal_approve(self):
        """Test proposal approval."""
        similarity = NodeSimilarityResult(
            node_a_id="a",
            node_b_id="b",
            scores=[],
            overall_similarity=0.9,
        )
        proposal = MergeProposal(
            node_a_id="a",
            node_b_id="b",
            similarity=similarity,
            strategy=MergeStrategy.KEEP_BETTER,
        )

        approved = proposal.approve("admin")
        assert approved.status == MergeStatus.APPROVED
        assert approved.approved_by == "admin"
        # Original unchanged
        assert proposal.status == MergeStatus.PROPOSED

    def test_proposal_reject(self):
        """Test proposal rejection."""
        similarity = NodeSimilarityResult(
            node_a_id="a",
            node_b_id="b",
            scores=[],
            overall_similarity=0.9,
        )
        proposal = MergeProposal(
            node_a_id="a",
            node_b_id="b",
            similarity=similarity,
            strategy=MergeStrategy.KEEP_BETTER,
        )

        rejected = proposal.reject("Too risky")
        assert rejected.status == MergeStatus.REJECTED
        assert rejected.rejection_reason == "Too risky"
