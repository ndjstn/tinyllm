"""JSON processing tools for parsing, querying, validating, and transforming JSON data."""

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata

# Try to import jsonpath_ng for advanced queries, fallback to basic implementation
try:
    from jsonpath_ng import parse as jsonpath_parse
    from jsonpath_ng.exceptions import JsonPathParserError

    HAS_JSONPATH = True
except ImportError:
    HAS_JSONPATH = False
    JsonPathParserError = Exception  # type: ignore


class JSONOperation(str, Enum):
    """Supported JSON operations."""

    PARSE = "parse"
    QUERY = "query"
    VALIDATE = "validate"
    TRANSFORM = "transform"
    MERGE = "merge"
    DIFF = "diff"
    FLATTEN = "flatten"
    UNFLATTEN = "unflatten"


class JSONConfig(ToolConfig):
    """Configuration for JSON tools."""

    max_size_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)  # 10MB default
    allow_comments: bool = Field(default=False)
    strict_mode: bool = Field(default=True)
    max_depth: int = Field(default=100, ge=1, le=1000)


# --- Parse Tool ---


class JSONParseInput(BaseModel):
    """Input for JSON parsing."""

    data: str = Field(
        description="JSON string to parse",
        max_length=10 * 1024 * 1024,
    )
    strict: bool = Field(
        default=True,
        description="If True, raise errors on invalid JSON. If False, return None for invalid JSON.",
    )


class JSONParseOutput(BaseModel):
    """Output from JSON parsing."""

    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    data_type: Optional[str] = None  # object, array, string, number, boolean, null


class JSONParseTool(BaseTool[JSONParseInput, JSONParseOutput]):
    """Parse JSON strings into Python objects."""

    metadata = ToolMetadata(
        id="json_parse",
        name="JSON Parse",
        description="Parse a JSON string into a Python object. Returns the parsed data with type information.",
        category="data",
        sandbox_required=False,
    )
    input_type = JSONParseInput
    output_type = JSONParseOutput

    async def execute(self, input: JSONParseInput) -> JSONParseOutput:
        """Parse JSON string."""
        try:
            result = json.loads(input.data)
            data_type = self._get_json_type(result)
            return JSONParseOutput(
                success=True,
                result=result,
                data_type=data_type,
            )
        except json.JSONDecodeError as e:
            if input.strict:
                return JSONParseOutput(
                    success=False,
                    error=f"Invalid JSON: {e.msg} at line {e.lineno}, column {e.colno}",
                )
            return JSONParseOutput(
                success=True,
                result=None,
                data_type="null",
            )
        except Exception as e:
            return JSONParseOutput(success=False, error=str(e))

    def _get_json_type(self, value: Any) -> str:
        """Get JSON type of a value."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "unknown"


# --- Query Tool ---


class JSONQueryInput(BaseModel):
    """Input for JSON querying."""

    data: Union[str, Dict[str, Any], List[Any]] = Field(
        description="JSON data to query (string or parsed object)",
    )
    jsonpath: str = Field(
        description="JSONPath expression to query (e.g., '$.users[*].name', '$.store.book[0].title')",
        examples=["$.users[*].name", "$.store.book[?@.price<10]", "$..author"],
    )


class JSONQueryOutput(BaseModel):
    """Output from JSON querying."""

    success: bool
    results: List[Any] = Field(default_factory=list)
    count: int = 0
    paths: List[str] = Field(default_factory=list)  # JSONPath to each match
    error: Optional[str] = None


class JSONQueryTool(BaseTool[JSONQueryInput, JSONQueryOutput]):
    """Query JSON data using JSONPath expressions."""

    metadata = ToolMetadata(
        id="json_query",
        name="JSON Query",
        description="Query JSON data using JSONPath expressions. "
        "Supports wildcards (*), recursive descent (..), array slicing, and filters.",
        category="data",
        sandbox_required=False,
    )
    input_type = JSONQueryInput
    output_type = JSONQueryOutput

    async def execute(self, input: JSONQueryInput) -> JSONQueryOutput:
        """Execute JSONPath query."""
        try:
            # Parse data if string
            if isinstance(input.data, str):
                data = json.loads(input.data)
            else:
                data = input.data

            if HAS_JSONPATH:
                return self._query_with_jsonpath_ng(data, input.jsonpath)
            else:
                return self._query_basic(data, input.jsonpath)

        except json.JSONDecodeError as e:
            return JSONQueryOutput(
                success=False,
                error=f"Invalid JSON: {e.msg}",
            )
        except JsonPathParserError as e:
            return JSONQueryOutput(
                success=False,
                error=f"Invalid JSONPath: {e}",
            )
        except Exception as e:
            return JSONQueryOutput(success=False, error=str(e))

    def _query_with_jsonpath_ng(self, data: Any, jsonpath: str) -> JSONQueryOutput:
        """Query using jsonpath_ng library."""
        expr = jsonpath_parse(jsonpath)
        matches = expr.find(data)

        results = [match.value for match in matches]
        paths = [str(match.full_path) for match in matches]

        return JSONQueryOutput(
            success=True,
            results=results,
            count=len(results),
            paths=paths,
        )

    def _query_basic(self, data: Any, jsonpath: str) -> JSONQueryOutput:
        """Basic JSONPath query without external library.

        Supports: $, ., [], [*], [n], .key
        Does NOT support: .., filters, slices
        """
        if not jsonpath.startswith("$"):
            return JSONQueryOutput(
                success=False,
                error="JSONPath must start with '$'",
            )

        # Simple parser for basic paths
        path = jsonpath[1:]  # Remove $
        current = [data]
        current_paths = ["$"]

        i = 0
        while i < len(path):
            if path[i] == ".":
                # Check for .. (recursive descent - not supported in basic)
                if i + 1 < len(path) and path[i + 1] == ".":
                    return JSONQueryOutput(
                        success=False,
                        error="Recursive descent (..) not supported. Install jsonpath-ng for full support.",
                    )
                # Property access
                i += 1
                if i >= len(path):
                    break

                # Find property name
                prop_start = i
                while i < len(path) and path[i] not in ".[":
                    i += 1
                prop = path[prop_start:i]

                if prop == "*":
                    # Wildcard - get all values
                    new_current = []
                    new_paths = []
                    for idx, item in enumerate(current):
                        if isinstance(item, dict):
                            for k, v in item.items():
                                new_current.append(v)
                                new_paths.append(f"{current_paths[idx]}.{k}")
                        elif isinstance(item, list):
                            for j, v in enumerate(item):
                                new_current.append(v)
                                new_paths.append(f"{current_paths[idx]}[{j}]")
                    current = new_current
                    current_paths = new_paths
                else:
                    # Specific property
                    new_current = []
                    new_paths = []
                    for idx, item in enumerate(current):
                        if isinstance(item, dict) and prop in item:
                            new_current.append(item[prop])
                            new_paths.append(f"{current_paths[idx]}.{prop}")
                    current = new_current
                    current_paths = new_paths

            elif path[i] == "[":
                i += 1
                # Find closing bracket
                bracket_start = i
                while i < len(path) and path[i] != "]":
                    i += 1
                if i >= len(path):
                    return JSONQueryOutput(
                        success=False,
                        error="Unclosed bracket in JSONPath",
                    )
                bracket_content = path[bracket_start:i]
                i += 1  # Skip ]

                if bracket_content == "*":
                    # Wildcard array access
                    new_current = []
                    new_paths = []
                    for idx, item in enumerate(current):
                        if isinstance(item, list):
                            for j, v in enumerate(item):
                                new_current.append(v)
                                new_paths.append(f"{current_paths[idx]}[{j}]")
                    current = new_current
                    current_paths = new_paths
                elif bracket_content.startswith("?"):
                    return JSONQueryOutput(
                        success=False,
                        error="Filters not supported. Install jsonpath-ng for full support.",
                    )
                elif ":" in bracket_content:
                    return JSONQueryOutput(
                        success=False,
                        error="Slices not supported. Install jsonpath-ng for full support.",
                    )
                else:
                    # Index or property access
                    try:
                        index = int(bracket_content)
                        new_current = []
                        new_paths = []
                        for idx, item in enumerate(current):
                            if isinstance(item, list) and -len(item) <= index < len(item):
                                new_current.append(item[index])
                                new_paths.append(f"{current_paths[idx]}[{index}]")
                        current = new_current
                        current_paths = new_paths
                    except ValueError:
                        # Property name in brackets (e.g., ['key'])
                        prop = bracket_content.strip("'\"")
                        new_current = []
                        new_paths = []
                        for idx, item in enumerate(current):
                            if isinstance(item, dict) and prop in item:
                                new_current.append(item[prop])
                                new_paths.append(f"{current_paths[idx]}['{prop}']")
                        current = new_current
                        current_paths = new_paths
            else:
                i += 1

        return JSONQueryOutput(
            success=True,
            results=current,
            count=len(current),
            paths=current_paths,
        )


# --- Validate Tool ---


class JSONValidateInput(BaseModel):
    """Input for JSON validation."""

    model_config = {"populate_by_name": True}

    data: Union[str, Dict[str, Any], List[Any]] = Field(
        description="JSON data to validate",
    )
    json_schema: Dict[str, Any] = Field(
        description="JSON Schema to validate against",
        alias="schema",
    )


class JSONValidateOutput(BaseModel):
    """Output from JSON validation."""

    success: bool
    valid: bool = False
    errors: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class JSONValidateTool(BaseTool[JSONValidateInput, JSONValidateOutput]):
    """Validate JSON data against a JSON Schema."""

    metadata = ToolMetadata(
        id="json_validate",
        name="JSON Validate",
        description="Validate JSON data against a JSON Schema. Returns validation errors if any.",
        category="data",
        sandbox_required=False,
    )
    input_type = JSONValidateInput
    output_type = JSONValidateOutput

    async def execute(self, input: JSONValidateInput) -> JSONValidateOutput:
        """Validate JSON against schema."""
        try:
            # Try to use jsonschema if available
            try:
                import jsonschema
                from jsonschema import ValidationError, validate

                HAS_JSONSCHEMA = True
            except ImportError:
                HAS_JSONSCHEMA = False

            # Parse data if string
            if isinstance(input.data, str):
                data = json.loads(input.data)
            else:
                data = input.data

            if HAS_JSONSCHEMA:
                try:
                    validate(instance=data, schema=input.json_schema)
                    return JSONValidateOutput(success=True, valid=True)
                except ValidationError as e:
                    return JSONValidateOutput(
                        success=True,
                        valid=False,
                        errors=[e.message],
                    )
            else:
                # Basic validation without jsonschema library
                return self._basic_validate(data, input.json_schema)

        except json.JSONDecodeError as e:
            return JSONValidateOutput(
                success=False,
                error=f"Invalid JSON data: {e.msg}",
            )
        except Exception as e:
            return JSONValidateOutput(success=False, error=str(e))

    def _basic_validate(
        self, data: Any, schema: Dict[str, Any]
    ) -> JSONValidateOutput:
        """Basic schema validation without jsonschema library."""
        errors = []

        # Check type
        if "type" in schema:
            expected_type = schema["type"]
            actual_type = self._get_json_type(data)

            type_mapping = {
                "object": "object",
                "array": "array",
                "string": "string",
                "number": "number",
                "integer": "number",
                "boolean": "boolean",
                "null": "null",
            }

            if type_mapping.get(expected_type) != actual_type:
                if expected_type == "integer" and actual_type == "number":
                    if not isinstance(data, int) or isinstance(data, bool):
                        errors.append(f"Expected integer, got {type(data).__name__}")
                else:
                    errors.append(f"Expected {expected_type}, got {actual_type}")

        # Check required properties for objects
        if isinstance(data, dict) and "required" in schema:
            for prop in schema["required"]:
                if prop not in data:
                    errors.append(f"Missing required property: {prop}")

        # Check properties
        if isinstance(data, dict) and "properties" in schema:
            for prop, prop_schema in schema["properties"].items():
                if prop in data:
                    sub_result = self._basic_validate(data[prop], prop_schema)
                    if not sub_result.valid:
                        for error in sub_result.errors:
                            errors.append(f"{prop}: {error}")

        # Check array items
        if isinstance(data, list) and "items" in schema:
            for i, item in enumerate(data):
                sub_result = self._basic_validate(item, schema["items"])
                if not sub_result.valid:
                    for error in sub_result.errors:
                        errors.append(f"[{i}]: {error}")

        return JSONValidateOutput(
            success=True,
            valid=len(errors) == 0,
            errors=errors,
        )

    def _get_json_type(self, value: Any) -> str:
        """Get JSON type of a value."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "unknown"


# --- Transform Tool ---


class JSONTransformInput(BaseModel):
    """Input for JSON transformation."""

    data: Union[str, Dict[str, Any], List[Any]] = Field(
        description="JSON data to transform",
    )
    operations: List[Dict[str, Any]] = Field(
        description="List of transformation operations",
        examples=[
            [{"op": "set", "path": "$.name", "value": "New Name"}],
            [{"op": "delete", "path": "$.password"}],
            [{"op": "rename", "from": "old_key", "to": "new_key"}],
        ],
    )


class JSONTransformOutput(BaseModel):
    """Output from JSON transformation."""

    success: bool
    result: Optional[Any] = None
    operations_applied: int = 0
    error: Optional[str] = None


class JSONTransformTool(BaseTool[JSONTransformInput, JSONTransformOutput]):
    """Transform JSON data using a series of operations."""

    metadata = ToolMetadata(
        id="json_transform",
        name="JSON Transform",
        description="Transform JSON data using operations like set, delete, rename, and more.",
        category="data",
        sandbox_required=False,
    )
    input_type = JSONTransformInput
    output_type = JSONTransformOutput

    async def execute(self, input: JSONTransformInput) -> JSONTransformOutput:
        """Apply transformations to JSON data."""
        try:
            # Parse data if string
            if isinstance(input.data, str):
                data = json.loads(input.data)
            else:
                # Deep copy to avoid modifying original
                data = json.loads(json.dumps(input.data))

            ops_applied = 0
            for op in input.operations:
                op_type = op.get("op", "").lower()

                if op_type == "set":
                    self._set_value(data, op.get("path", ""), op.get("value"))
                    ops_applied += 1
                elif op_type == "delete":
                    self._delete_path(data, op.get("path", ""))
                    ops_applied += 1
                elif op_type == "rename":
                    self._rename_key(data, op.get("from", ""), op.get("to", ""))
                    ops_applied += 1
                elif op_type == "copy":
                    self._copy_value(data, op.get("from", ""), op.get("to", ""))
                    ops_applied += 1
                elif op_type == "move":
                    self._move_value(data, op.get("from", ""), op.get("to", ""))
                    ops_applied += 1
                else:
                    return JSONTransformOutput(
                        success=False,
                        error=f"Unknown operation: {op_type}",
                    )

            return JSONTransformOutput(
                success=True,
                result=data,
                operations_applied=ops_applied,
            )

        except json.JSONDecodeError as e:
            return JSONTransformOutput(
                success=False,
                error=f"Invalid JSON: {e.msg}",
            )
        except Exception as e:
            return JSONTransformOutput(success=False, error=str(e))

    def _get_parent_and_key(
        self, data: Any, path: str
    ) -> tuple[Any, Union[str, int], bool]:
        """Navigate to parent of a path and return (parent, key, found)."""
        if not path.startswith("$"):
            raise ValueError("Path must start with $")

        parts = self._parse_path(path[1:])
        if not parts:
            return None, "", False

        current = data
        for part in parts[:-1]:
            if isinstance(part, int):
                if not isinstance(current, list) or part >= len(current):
                    return None, "", False
                current = current[part]
            else:
                if not isinstance(current, dict) or part not in current:
                    return None, "", False
                current = current[part]

        return current, parts[-1], True

    def _parse_path(self, path: str) -> List[Union[str, int]]:
        """Parse a simple path like .foo.bar[0].baz into parts."""
        parts: List[Union[str, int]] = []
        i = 0
        while i < len(path):
            if path[i] == ".":
                i += 1
                prop_start = i
                while i < len(path) and path[i] not in ".[":
                    i += 1
                if i > prop_start:
                    parts.append(path[prop_start:i])
            elif path[i] == "[":
                i += 1
                bracket_start = i
                while i < len(path) and path[i] != "]":
                    i += 1
                content = path[bracket_start:i]
                i += 1
                try:
                    parts.append(int(content))
                except ValueError:
                    parts.append(content.strip("'\""))
            else:
                i += 1
        return parts

    def _set_value(self, data: Any, path: str, value: Any) -> None:
        """Set a value at a path."""
        parent, key, found = self._get_parent_and_key(data, path)
        if parent is not None:
            if isinstance(parent, dict):
                parent[key] = value
            elif isinstance(parent, list) and isinstance(key, int):
                if key < len(parent):
                    parent[key] = value

    def _delete_path(self, data: Any, path: str) -> None:
        """Delete a value at a path."""
        parent, key, found = self._get_parent_and_key(data, path)
        if parent is not None:
            if isinstance(parent, dict) and key in parent:
                del parent[key]
            elif isinstance(parent, list) and isinstance(key, int) and key < len(parent):
                del parent[key]

    def _rename_key(self, data: Any, from_key: str, to_key: str) -> None:
        """Rename a key at the root level or nested."""
        if isinstance(data, dict):
            if from_key in data:
                data[to_key] = data.pop(from_key)

    def _copy_value(self, data: Any, from_path: str, to_path: str) -> None:
        """Copy a value from one path to another."""
        # Get value from source
        parent, key, found = self._get_parent_and_key(data, from_path)
        if parent is not None:
            if isinstance(parent, dict) and key in parent:
                value = parent[key]
            elif isinstance(parent, list) and isinstance(key, int) and key < len(parent):
                value = parent[key]
            else:
                return
            # Set at destination
            self._set_value(data, to_path, value)

    def _move_value(self, data: Any, from_path: str, to_path: str) -> None:
        """Move a value from one path to another."""
        self._copy_value(data, from_path, to_path)
        self._delete_path(data, from_path)


# --- Merge Tool ---


class JSONMergeInput(BaseModel):
    """Input for JSON merging."""

    target: Union[str, Dict[str, Any]] = Field(
        description="Target JSON object to merge into",
    )
    source: Union[str, Dict[str, Any]] = Field(
        description="Source JSON object to merge from",
    )
    deep: bool = Field(
        default=True,
        description="If True, perform deep merge. If False, shallow merge.",
    )


class JSONMergeOutput(BaseModel):
    """Output from JSON merging."""

    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JSONMergeTool(BaseTool[JSONMergeInput, JSONMergeOutput]):
    """Merge two JSON objects."""

    metadata = ToolMetadata(
        id="json_merge",
        name="JSON Merge",
        description="Merge two JSON objects. Supports shallow and deep merge strategies.",
        category="data",
        sandbox_required=False,
    )
    input_type = JSONMergeInput
    output_type = JSONMergeOutput

    async def execute(self, input: JSONMergeInput) -> JSONMergeOutput:
        """Merge two JSON objects."""
        try:
            # Parse if strings
            target = json.loads(input.target) if isinstance(input.target, str) else input.target
            source = json.loads(input.source) if isinstance(input.source, str) else input.source

            if not isinstance(target, dict) or not isinstance(source, dict):
                return JSONMergeOutput(
                    success=False,
                    error="Both target and source must be JSON objects",
                )

            if input.deep:
                result = self._deep_merge(target, source)
            else:
                result = {**target, **source}

            return JSONMergeOutput(success=True, result=result)

        except json.JSONDecodeError as e:
            return JSONMergeOutput(success=False, error=f"Invalid JSON: {e.msg}")
        except Exception as e:
            return JSONMergeOutput(success=False, error=str(e))

    def _deep_merge(self, target: Dict, source: Dict) -> Dict:
        """Deep merge source into target."""
        result = target.copy()
        for key, value in source.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


# --- Flatten Tool ---


class JSONFlattenInput(BaseModel):
    """Input for JSON flattening."""

    data: Union[str, Dict[str, Any]] = Field(
        description="JSON object to flatten",
    )
    separator: str = Field(
        default=".",
        description="Separator for flattened keys",
    )
    max_depth: Optional[int] = Field(
        default=None,
        description="Maximum depth to flatten (None for unlimited)",
    )


class JSONFlattenOutput(BaseModel):
    """Output from JSON flattening."""

    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JSONFlattenTool(BaseTool[JSONFlattenInput, JSONFlattenOutput]):
    """Flatten a nested JSON object."""

    metadata = ToolMetadata(
        id="json_flatten",
        name="JSON Flatten",
        description="Flatten a nested JSON object into a single-level object with dot-notation keys.",
        category="data",
        sandbox_required=False,
    )
    input_type = JSONFlattenInput
    output_type = JSONFlattenOutput

    async def execute(self, input: JSONFlattenInput) -> JSONFlattenOutput:
        """Flatten a JSON object."""
        try:
            data = json.loads(input.data) if isinstance(input.data, str) else input.data

            if not isinstance(data, dict):
                return JSONFlattenOutput(
                    success=False,
                    error="Input must be a JSON object",
                )

            result = self._flatten(data, "", input.separator, 0, input.max_depth)
            return JSONFlattenOutput(success=True, result=result)

        except json.JSONDecodeError as e:
            return JSONFlattenOutput(success=False, error=f"Invalid JSON: {e.msg}")
        except Exception as e:
            return JSONFlattenOutput(success=False, error=str(e))

    def _flatten(
        self,
        obj: Any,
        prefix: str,
        sep: str,
        depth: int,
        max_depth: Optional[int],
    ) -> Dict[str, Any]:
        """Recursively flatten an object."""
        result: Dict[str, Any] = {}

        if max_depth is not None and depth >= max_depth:
            if prefix:
                result[prefix] = obj
            return result

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}{sep}{key}" if prefix else key
                result.update(self._flatten(value, new_key, sep, depth + 1, max_depth))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_key = f"{prefix}[{i}]"
                result.update(self._flatten(item, new_key, sep, depth + 1, max_depth))
        else:
            if prefix:
                result[prefix] = obj

        return result


# --- Factory Function ---


def create_json_tools(config: JSONConfig | None = None) -> List[BaseTool]:
    """Create all JSON tools with optional configuration."""
    return [
        JSONParseTool(config),
        JSONQueryTool(config),
        JSONValidateTool(config),
        JSONTransformTool(config),
        JSONMergeTool(config),
        JSONFlattenTool(config),
    ]
