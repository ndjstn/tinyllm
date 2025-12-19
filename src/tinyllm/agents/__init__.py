"""Agent framework for TinyLLM.

This module provides agent implementations including the ReAct and Plan-and-Execute patterns.
"""

from tinyllm.agents.react import (
    ActionResult,
    ReActAgent,
    ReActConfig,
    ReActStep,
    ReActStepType,
    ToolExecutionError,
)
from tinyllm.agents.plan_execute import (
    ExecutionResult,
    PlanExecuteAgent,
    PlanExecuteConfig,
    PlanExecuteError,
    PlanExecuteStatistics,
    PlanStep,
    StepStatus,
)

__all__ = [
    # ReAct agent
    "ReActAgent",
    "ReActConfig",
    "ReActStep",
    "ReActStepType",
    "ActionResult",
    "ToolExecutionError",
    # Plan-and-Execute agent
    "PlanExecuteAgent",
    "PlanExecuteConfig",
    "PlanStep",
    "ExecutionResult",
    "StepStatus",
    "PlanExecuteStatistics",
    "PlanExecuteError",
]
