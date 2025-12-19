"""LLM-as-Judge implementation.

Uses a larger LLM to evaluate the quality of responses
from smaller specialist models.
"""

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.grading.models import (
    DimensionScore,
    Grade,
    GradeLevel,
    GradingContext,
    GradingCriteria,
    GradingResult,
    QualityDimension,
)
from tinyllm.models.client import OllamaClient

if TYPE_CHECKING:
    pass


class JudgeConfig(BaseModel):
    """Configuration for the LLM judge."""

    model: str = Field(
        default="qwen3:14b",
        description="Model to use for judging (T3 tier recommended)",
    )
    fallback_model: str = Field(
        default="qwen2.5:3b",
        description="Fallback model if primary unavailable",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Low temperature for consistent judging",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Retries on parse failure",
    )
    timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=120000,
        description="Timeout for judge response",
    )


class Judge:
    """LLM-based judge for evaluating response quality.

    Uses a larger model (T3 tier) to evaluate responses from
    smaller specialist models, providing grades and feedback.
    """

    def __init__(self, config: Optional[JudgeConfig] = None):
        """Initialize the judge.

        Args:
            config: Judge configuration. Uses defaults if not provided.
        """
        self.config = config or JudgeConfig()
        self._client: Optional[OllamaClient] = None

    def _get_client(self) -> OllamaClient:
        """Get or create Ollama client."""
        if self._client is None:
            self._client = OllamaClient()
        return self._client

    async def grade(
        self,
        context: GradingContext,
        criteria: Optional[GradingCriteria] = None,
    ) -> GradingResult:
        """Grade a response using LLM-as-judge.

        Args:
            context: The grading context with task and response.
            criteria: Grading criteria. Uses defaults if not provided.

        Returns:
            GradingResult with grade and feedback.
        """
        import time

        start_time = time.time()
        criteria = criteria or GradingCriteria()

        # Build the judging prompt
        prompt = self._build_prompt(context, criteria)
        system = self._get_system_prompt(criteria)

        client = self._get_client()

        # Try to get judge response with retries
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await client.generate(
                    model=self.config.model,
                    prompt=prompt,
                    system=system,
                    options={"temperature": self.config.temperature},
                )

                # Parse the structured response
                grade = self._parse_response(response.response, criteria)

                latency_ms = (time.time() - start_time) * 1000

                return GradingResult(
                    context=context,
                    grade=grade,
                    criteria=criteria,
                    judge_model=self.config.model,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                if attempt == self.config.max_retries:
                    # Create a failure grade
                    latency_ms = (time.time() - start_time) * 1000
                    return self._create_failure_result(
                        context, criteria, str(e), latency_ms
                    )

    async def grade_batch(
        self,
        contexts: List[GradingContext],
        criteria: Optional[GradingCriteria] = None,
    ) -> List[GradingResult]:
        """Grade multiple responses.

        Args:
            contexts: List of grading contexts.
            criteria: Shared grading criteria.

        Returns:
            List of grading results.
        """
        import asyncio

        tasks = [self.grade(ctx, criteria) for ctx in contexts]
        return await asyncio.gather(*tasks)

    def _build_prompt(
        self, context: GradingContext, criteria: GradingCriteria
    ) -> str:
        """Build the grading prompt."""
        dimensions_list = "\n".join(
            f"- {d.value}: {self._dimension_description(d)}"
            for d in criteria.dimensions
        )

        prompt = f"""Please evaluate the following response to a task.

## Task
{context.task}

## Response to Evaluate
{context.response}
"""

        if context.expected:
            prompt += f"""
## Expected/Reference Answer
{context.expected}
"""

        prompt += f"""
## Evaluation Dimensions
{dimensions_list}

## Instructions
For each dimension, provide:
1. A score from 0.0 to 1.0
2. Brief reasoning for the score
3. Specific evidence from the response

Then provide:
- Overall feedback summary
- Specific suggestions for improvement

Respond in JSON format:
```json
{{
    "dimensions": [
        {{"dimension": "correctness", "score": 0.8, "reasoning": "...", "evidence": "..."}},
        ...
    ],
    "feedback": "Overall feedback...",
    "suggestions": ["suggestion 1", "suggestion 2"]
}}
```"""
        return prompt

    def _get_system_prompt(self, criteria: GradingCriteria) -> str:
        """Get the system prompt for judging."""
        return """You are an expert evaluator assessing the quality of AI-generated responses.

Your role is to:
1. Objectively evaluate responses across multiple dimensions
2. Provide clear, actionable feedback
3. Be consistent and fair in your assessments
4. Focus on substantive issues, not minor stylistic preferences

Guidelines:
- 0.9-1.0: Exceptional, no meaningful improvements needed
- 0.75-0.89: Good quality, minor issues
- 0.6-0.74: Acceptable, some noticeable issues
- 0.4-0.59: Poor, significant problems
- 0.0-0.39: Failing, fundamental issues

Always respond with valid JSON."""

    def _dimension_description(self, dimension: QualityDimension) -> str:
        """Get description for a dimension."""
        descriptions = {
            QualityDimension.CORRECTNESS: "Is the information factually accurate?",
            QualityDimension.COMPLETENESS: "Does it fully address the task?",
            QualityDimension.RELEVANCE: "Is it relevant to what was asked?",
            QualityDimension.CLARITY: "Is it clear and well-structured?",
            QualityDimension.CONCISENESS: "Is it appropriately concise?",
            QualityDimension.CODE_QUALITY: "Is the code well-written and correct?",
            QualityDimension.SAFETY: "Does it avoid harmful content?",
        }
        return descriptions.get(dimension, "Evaluate this dimension")

    def _parse_response(self, response: str, criteria: GradingCriteria) -> Grade:
        """Parse the judge's response into a Grade."""
        # Try to extract JSON from the response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in judge response")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in judge response: {e}")

        # Parse dimension scores
        dimension_scores = []
        for dim_data in data.get("dimensions", []):
            try:
                dimension = QualityDimension(dim_data["dimension"])
                if dimension in criteria.dimensions:
                    dimension_scores.append(
                        DimensionScore(
                            dimension=dimension,
                            score=float(dim_data["score"]),
                            reasoning=dim_data.get("reasoning", ""),
                            evidence=dim_data.get("evidence"),
                        )
                    )
            except (KeyError, ValueError):
                continue

        # Ensure we have scores for all required dimensions
        scored_dims = {ds.dimension for ds in dimension_scores}
        for dim in criteria.dimensions:
            if dim not in scored_dims:
                # Default score for missing dimensions
                dimension_scores.append(
                    DimensionScore(
                        dimension=dim,
                        score=0.5,
                        reasoning="Dimension not evaluated by judge",
                    )
                )

        feedback = data.get("feedback", "No feedback provided")
        suggestions = data.get("suggestions", [])

        return Grade.create(
            dimension_scores=dimension_scores,
            criteria=criteria,
            feedback=feedback,
            suggestions=suggestions,
        )

    def _create_failure_result(
        self,
        context: GradingContext,
        criteria: GradingCriteria,
        error: str,
        latency_ms: float,
    ) -> GradingResult:
        """Create a failure result when grading fails."""
        # Create neutral scores for all dimensions
        dimension_scores = [
            DimensionScore(
                dimension=dim,
                score=0.5,
                reasoning=f"Grading failed: {error}",
            )
            for dim in criteria.dimensions
        ]

        grade = Grade.create(
            dimension_scores=dimension_scores,
            criteria=criteria,
            feedback=f"Grading failed: {error}",
            suggestions=["Re-run grading with different judge model"],
        )

        return GradingResult(
            context=context,
            grade=grade,
            criteria=criteria,
            judge_model=self.config.model,
            latency_ms=latency_ms,
        )


class RuleBasedJudge:
    """Fast rule-based judge for simple quality checks.

    Used as a first-pass filter before LLM judging, or for
    cases where LLM judging is too slow/expensive.
    """

    def __init__(self):
        """Initialize the rule-based judge."""
        pass

    def quick_check(self, context: GradingContext) -> Optional[Grade]:
        """Perform quick rule-based checks.

        Returns a failing grade if obvious issues are found,
        None if LLM judging should be used.
        """
        issues = []
        score_penalties = 0.0

        # Check for empty response
        if not context.response or not context.response.strip():
            return self._create_failing_grade("Response is empty")

        # Check for very short response (likely incomplete)
        if len(context.response.strip()) < 10:
            issues.append("Response is very short")
            score_penalties += 0.3

        # Check for error indicators
        error_patterns = [
            "I cannot",
            "I'm unable to",
            "I don't have access",
            "Error:",
            "Exception:",
        ]
        for pattern in error_patterns:
            if pattern.lower() in context.response.lower():
                issues.append(f"Response contains error indicator: {pattern}")
                score_penalties += 0.2
                break

        # Check for refusal patterns
        refusal_patterns = [
            "I cannot help with",
            "I'm not able to",
            "I won't",
        ]
        for pattern in refusal_patterns:
            if pattern.lower() in context.response.lower():
                # This might be appropriate, don't penalize heavily
                issues.append("Response indicates refusal")
                score_penalties += 0.1
                break

        # If significant penalties, return a preliminary grade
        if score_penalties >= 0.5:
            base_score = max(0.0, 1.0 - score_penalties)
            return self._create_partial_grade(base_score, issues)

        # Return None to indicate LLM judging should be used
        return None

    def _create_failing_grade(self, reason: str) -> Grade:
        """Create a failing grade for obvious failures."""
        return Grade(
            level=GradeLevel.FAILING,
            overall_score=0.0,
            dimension_scores=[],
            feedback=reason,
            suggestions=["Provide a non-empty, meaningful response"],
            is_passing=False,
        )

    def _create_partial_grade(self, score: float, issues: List[str]) -> Grade:
        """Create a partial grade based on rule checks."""
        return Grade(
            level=GradeLevel.from_score(score),
            overall_score=score,
            dimension_scores=[],
            feedback=f"Rule-based issues found: {'; '.join(issues)}",
            suggestions=["Address the identified issues"],
            is_passing=score >= 0.6,
        )
