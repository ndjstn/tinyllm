"""Self-improvement and expansion system.

Provides automatic graph expansion based on node performance
monitoring and failure pattern analysis.
"""

from tinyllm.expansion.models import (
    EdgeCreationSpec,
    ExpansionBenefit,
    ExpansionConfig,
    ExpansionCost,
    ExpansionProposal,
    ExpansionResult,
    ExpansionStrategy,
    FailureCategory,
    FailurePattern,
    NodeCreationSpec,
    StrategyType,
)
from tinyllm.expansion.analyzer import PatternAnalyzer, PatternAnalyzerConfig
from tinyllm.expansion.strategies import StrategyGenerator, StrategyGeneratorConfig
from tinyllm.expansion.engine import (
    AutoExpansionMonitor,
    ExpansionEngine,
    ExpansionTrigger,
)
from tinyllm.expansion.versioning import (
    GraphVersion,
    GraphVersionManager,
    VersionHistory,
)
from tinyllm.expansion.spawning import (
    NodeFactory,
    NodeSpawner,
    SpawnConfig,
    SpawnMetrics,
    SpawnRecord,
    SpawnTrigger,
)
from tinyllm.expansion.pruning import (
    NodeHealth,
    NodeHealthAnalyzer,
    NodeHealthReport,
    NodePruner,
    PruneConfig,
    PruneHistory,
    PruneProposal,
    PruneReason,
    PruneResult,
    PruneStatus,
)
from tinyllm.expansion.merging import (
    MergeConfig,
    MergeHistory,
    MergeProposal,
    MergeResult,
    MergeStatus,
    MergeStrategy,
    NodeMerger,
    NodeSimilarityDetector,
    NodeSimilarityResult,
    SimilarityMetric,
    SimilarityScore,
)

__all__ = [
    # Models
    "EdgeCreationSpec",
    "ExpansionBenefit",
    "ExpansionConfig",
    "ExpansionCost",
    "ExpansionProposal",
    "ExpansionResult",
    "ExpansionStrategy",
    "FailureCategory",
    "FailurePattern",
    "NodeCreationSpec",
    "StrategyType",
    # Analyzer
    "PatternAnalyzer",
    "PatternAnalyzerConfig",
    # Strategies
    "StrategyGenerator",
    "StrategyGeneratorConfig",
    # Engine
    "AutoExpansionMonitor",
    "ExpansionEngine",
    "ExpansionTrigger",
    # Versioning
    "GraphVersion",
    "GraphVersionManager",
    "VersionHistory",
    # Spawning
    "NodeFactory",
    "NodeSpawner",
    "SpawnConfig",
    "SpawnMetrics",
    "SpawnRecord",
    "SpawnTrigger",
    # Pruning
    "NodeHealth",
    "NodeHealthAnalyzer",
    "NodeHealthReport",
    "NodePruner",
    "PruneConfig",
    "PruneHistory",
    "PruneProposal",
    "PruneReason",
    "PruneResult",
    "PruneStatus",
    # Merging
    "MergeConfig",
    "MergeHistory",
    "MergeProposal",
    "MergeResult",
    "MergeStatus",
    "MergeStrategy",
    "NodeMerger",
    "NodeSimilarityDetector",
    "NodeSimilarityResult",
    "SimilarityMetric",
    "SimilarityScore",
]
