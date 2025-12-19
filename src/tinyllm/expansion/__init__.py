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
]
