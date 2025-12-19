"""Transform node implementation.

Applies data transformations to messages for preprocessing and postprocessing.
"""

import json
import re
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import BaseNode, NodeConfig, NodeResult
from tinyllm.core.registry import NodeRegistry
from tinyllm.logging import get_logger

if TYPE_CHECKING:
    from tinyllm.core.context import ExecutionContext

logger = get_logger(__name__, component="transform_node")


class TransformType(str, Enum):
    """Types of transformations available."""

    # Text transformations
    UPPERCASE = "uppercase"
    LOWERCASE = "lowercase"
    STRIP = "strip"
    TRUNCATE = "truncate"

    # JSON operations
    JSON_EXTRACT = "json_extract"
    JSON_WRAP = "json_wrap"
    JSON_PARSE = "json_parse"
    JSON_STRINGIFY = "json_stringify"

    # Text extraction
    REGEX_EXTRACT = "regex_extract"
    REGEX_REPLACE = "regex_replace"

    # Structural
    TEMPLATE = "template"
    SPLIT = "split"
    JOIN = "join"

    # Custom
    CUSTOM = "custom"


class TransformSpec(BaseModel):
    """Specification for a single transformation."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    transform_type: TransformType = Field(description="Type of transformation")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the transformation"
    )

    @model_validator(mode="after")
    def validate_params(self) -> "TransformSpec":
        """Validate params match the transform type."""
        required_params = {
            TransformType.TRUNCATE: ["max_length"],
            TransformType.JSON_EXTRACT: ["path"],
            TransformType.REGEX_EXTRACT: ["pattern"],
            TransformType.REGEX_REPLACE: ["pattern", "replacement"],
            TransformType.TEMPLATE: ["template"],
            TransformType.SPLIT: ["separator"],
            TransformType.JOIN: ["separator"],
        }

        required = required_params.get(self.transform_type, [])
        for param in required:
            if param not in self.params:
                raise ValueError(
                    f"Transform {self.transform_type} requires param '{param}'"
                )
        return self


class TransformPipeline(BaseModel):
    """Pipeline of transformations to apply in sequence."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    transforms: List[TransformSpec] = Field(
        min_length=1, description="Ordered list of transformations"
    )
    stop_on_error: bool = Field(
        default=True, description="Stop pipeline on first error"
    )


class TransformResult(BaseModel):
    """Result of a transformation."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    success: bool
    input_value: str
    output_value: Optional[str] = None
    error: Optional[str] = None
    transform_type: TransformType


class TransformNodeConfig(NodeConfig):
    """Configuration for transform nodes."""

    model_config = {"extra": "forbid"}

    transforms: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of transform specifications"
    )
    stop_on_error: bool = Field(
        default=True, description="Stop on first transform error"
    )


@NodeRegistry.register(NodeType.TRANSFORM)
class TransformNode(BaseNode):
    """Applies data transformations to message content.

    The TransformNode can apply a pipeline of transformations including
    text manipulation, JSON operations, regex extraction, and templating.
    """

    def __init__(self, definition: NodeDefinition):
        """Initialize transform node."""
        super().__init__(definition)
        self._transform_config = TransformNodeConfig(**definition.config)
        self._transforms = self._parse_transforms()
        logger.info(
            "transform_node_initialized",
            node_id=self.id,
            transforms_count=len(self._transforms),
            stop_on_error=self._transform_config.stop_on_error,
        )

    def _parse_transforms(self) -> List[TransformSpec]:
        """Parse transform specifications from config."""
        specs = []
        for transform_dict in self._transform_config.transforms:
            spec = TransformSpec(
                transform_type=TransformType(transform_dict.get("type", "strip")),
                params=transform_dict.get("params", {}),
            )
            specs.append(spec)
        return specs

    async def execute(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Execute the transformation pipeline.

        Args:
            message: Input message to transform.
            context: Execution context.

        Returns:
            NodeResult with transformed message.
        """
        content = message.payload.content or message.payload.task or ""
        results: List[TransformResult] = []

        logger.debug(
            "transform_pipeline_started",
            node_id=self.id,
            input_length=len(content),
            transforms_count=len(self._transforms),
        )

        current_value = content
        for spec in self._transforms:
            result = self._apply_transform(current_value, spec)
            results.append(result)

            logger.debug(
                "transform_applied",
                node_id=self.id,
                transform_type=spec.transform_type.value,
                success=result.success,
                error=result.error,
            )

            if not result.success:
                if self._transform_config.stop_on_error:
                    return NodeResult.failure_result(
                        error=f"Transform {spec.transform_type} failed: {result.error}",
                        metadata={"transform_results": [r.model_dump() for r in results]},
                    )
            else:
                current_value = result.output_value or ""

        # Create output message with transformed content
        output_payload = MessagePayload(
            task=message.payload.task,
            content=current_value,
            metadata={
                **message.payload.metadata,
                "transformed": True,
                "transforms_applied": len(self._transforms),
            },
        )

        output_message = message.create_child(
            source_node=self.id,
            payload=output_payload,
        )

        logger.debug(
            "transform_pipeline_completed",
            node_id=self.id,
            transforms_applied=len(results),
            original_length=len(content),
            output_length=len(current_value),
        )

        return NodeResult.success_result(
            output_messages=[output_message],
            next_nodes=[],
            metadata={
                "transforms_applied": len(results),
                "original_length": len(content),
                "output_length": len(current_value),
            },
        )

    def _apply_transform(self, value: str, spec: TransformSpec) -> TransformResult:
        """Apply a single transformation."""
        try:
            output = self._execute_transform(value, spec)
            return TransformResult(
                success=True,
                input_value=value[:100],  # Truncate for logging
                output_value=output,
                transform_type=spec.transform_type,
            )
        except Exception as e:
            return TransformResult(
                success=False,
                input_value=value[:100],
                error=str(e),
                transform_type=spec.transform_type,
            )

    def _execute_transform(self, value: str, spec: TransformSpec) -> str:
        """Execute the actual transformation logic."""
        params = spec.params

        match spec.transform_type:
            # Text transformations
            case TransformType.UPPERCASE:
                return value.upper()
            case TransformType.LOWERCASE:
                return value.lower()
            case TransformType.STRIP:
                return value.strip()
            case TransformType.TRUNCATE:
                max_length = int(params["max_length"])
                suffix = params.get("suffix", "...")
                if len(value) <= max_length:
                    return value
                return value[: max_length - len(suffix)] + suffix

            # JSON operations
            case TransformType.JSON_EXTRACT:
                data = json.loads(value)
                path = params["path"]
                return self._json_path_extract(data, path)
            case TransformType.JSON_WRAP:
                key = params.get("key", "content")
                return json.dumps({key: value})
            case TransformType.JSON_PARSE:
                # Validate it's valid JSON and return it
                json.loads(value)
                return value
            case TransformType.JSON_STRINGIFY:
                # Pretty print if it's JSON, otherwise wrap as string
                try:
                    data = json.loads(value)
                    return json.dumps(data, indent=2)
                except json.JSONDecodeError:
                    return json.dumps(value)

            # Regex operations
            case TransformType.REGEX_EXTRACT:
                pattern = params["pattern"]
                group = params.get("group", 0)
                match = re.search(pattern, value)
                if not match:
                    raise ValueError(f"Pattern '{pattern}' not found")
                return match.group(group)
            case TransformType.REGEX_REPLACE:
                pattern = params["pattern"]
                replacement = params["replacement"]
                return re.sub(pattern, replacement, value)

            # Structural
            case TransformType.TEMPLATE:
                template = params["template"]
                return template.replace("{content}", value)
            case TransformType.SPLIT:
                separator = params["separator"]
                index = params.get("index")
                parts = value.split(separator)
                if index is not None:
                    return parts[int(index)]
                return json.dumps(parts)
            case TransformType.JOIN:
                separator = params["separator"]
                # Assume value is JSON array
                parts = json.loads(value)
                return separator.join(str(p) for p in parts)

            # Custom (not implemented - would need safe eval)
            case TransformType.CUSTOM:
                raise NotImplementedError("Custom transforms not yet supported")

            case _:
                raise ValueError(f"Unknown transform type: {spec.transform_type}")

    def _json_path_extract(self, data: Any, path: str) -> str:
        """Extract value from JSON using dot notation path."""
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                current = current[int(part)]
            else:
                raise ValueError(f"Cannot traverse path '{path}' at '{part}'")
            if current is None:
                raise ValueError(f"Path '{path}' not found")
        if isinstance(current, (dict, list)):
            return json.dumps(current)
        return str(current)
