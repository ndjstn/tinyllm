"""CSV processing tools for reading, writing, filtering, and transforming CSV data."""

import csv
import io
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata

# Try to import pandas for advanced operations
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class CSVOperation(str, Enum):
    """Supported CSV operations."""

    READ = "read"
    WRITE = "write"
    FILTER = "filter"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"
    MERGE = "merge"


class AggregateFunction(str, Enum):
    """Supported aggregate functions."""

    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    FIRST = "first"
    LAST = "last"


class CSVConfig(ToolConfig):
    """Configuration for CSV tools."""

    max_rows: int = Field(default=100000, ge=1, le=10000000)
    max_file_size: int = Field(default=100 * 1024 * 1024, ge=1024)  # 100MB
    allowed_directories: Optional[List[str]] = Field(default=None)
    default_encoding: str = Field(default="utf-8")


# --- Read Tool ---


class ReadCSVInput(BaseModel):
    """Input for reading CSV files."""

    path: Optional[str] = Field(
        default=None,
        description="Path to CSV file to read",
    )
    content: Optional[str] = Field(
        default=None,
        description="CSV content as string (alternative to path)",
    )
    delimiter: str = Field(
        default=",",
        description="Column delimiter",
        max_length=5,
    )
    has_header: bool = Field(
        default=True,
        description="Whether the first row is a header",
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="Select specific columns by name",
    )
    skip_rows: int = Field(
        default=0,
        description="Number of rows to skip from the start",
        ge=0,
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of rows to return",
        ge=1,
    )
    encoding: str = Field(
        default="utf-8",
        description="File encoding",
    )


class ReadCSVOutput(BaseModel):
    """Output from reading CSV."""

    success: bool
    data: List[Dict[str, Any]] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    row_count: int = 0
    total_rows: int = 0  # Total rows in file (before limit)
    error: Optional[str] = None


class ReadCSVTool(BaseTool[ReadCSVInput, ReadCSVOutput]):
    """Read CSV files or content into structured data."""

    metadata = ToolMetadata(
        id="csv_read",
        name="CSV Read",
        description="Read CSV files or content into a list of dictionaries. "
        "Supports selecting columns, limiting rows, and skipping rows.",
        category="data",
        sandbox_required=False,
    )
    input_type = ReadCSVInput
    output_type = ReadCSVOutput

    def __init__(self, config: CSVConfig | None = None):
        super().__init__(config or CSVConfig())

    async def execute(self, input: ReadCSVInput) -> ReadCSVOutput:
        """Read CSV data."""
        try:
            # Get content from file or string
            if input.path:
                path = Path(input.path)
                if not path.exists():
                    return ReadCSVOutput(
                        success=False, error=f"File not found: {input.path}"
                    )
                if not path.is_file():
                    return ReadCSVOutput(
                        success=False, error=f"Not a file: {input.path}"
                    )

                # Check file size
                config = self.config
                if isinstance(config, CSVConfig):
                    if path.stat().st_size > config.max_file_size:
                        return ReadCSVOutput(
                            success=False,
                            error=f"File too large: {path.stat().st_size} bytes",
                        )

                with open(path, "r", encoding=input.encoding) as f:
                    content = f.read()
            elif input.content:
                content = input.content
            else:
                return ReadCSVOutput(
                    success=False, error="Either path or content must be provided"
                )

            # Parse CSV
            reader = csv.reader(io.StringIO(content), delimiter=input.delimiter)
            rows = list(reader)

            if not rows:
                return ReadCSVOutput(success=True, data=[], columns=[], row_count=0)

            # Handle headers
            if input.has_header:
                headers = rows[0]
                data_rows = rows[1:]
            else:
                headers = [f"col_{i}" for i in range(len(rows[0]))]
                data_rows = rows

            # Skip rows
            if input.skip_rows > 0:
                data_rows = data_rows[input.skip_rows:]

            total_rows = len(data_rows)

            # Limit rows
            if input.limit:
                data_rows = data_rows[: input.limit]

            # Select columns
            if input.columns:
                col_indices = []
                for col in input.columns:
                    if col in headers:
                        col_indices.append(headers.index(col))
                    else:
                        return ReadCSVOutput(
                            success=False, error=f"Column not found: {col}"
                        )
                headers = input.columns
            else:
                col_indices = list(range(len(headers)))

            # Convert to dictionaries
            data = []
            for row in data_rows:
                record = {}
                for i, col_idx in enumerate(col_indices):
                    if col_idx < len(row):
                        record[headers[i]] = row[col_idx]
                    else:
                        record[headers[i]] = None
                data.append(record)

            return ReadCSVOutput(
                success=True,
                data=data,
                columns=headers,
                row_count=len(data),
                total_rows=total_rows,
            )

        except UnicodeDecodeError as e:
            return ReadCSVOutput(success=False, error=f"Encoding error: {e}")
        except csv.Error as e:
            return ReadCSVOutput(success=False, error=f"CSV parsing error: {e}")
        except Exception as e:
            return ReadCSVOutput(success=False, error=str(e))


# --- Write Tool ---


class WriteCSVInput(BaseModel):
    """Input for writing CSV files."""

    path: str = Field(
        description="Path to write CSV file",
    )
    data: List[Dict[str, Any]] = Field(
        description="Data to write as list of dictionaries",
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="Column order (uses all keys if None)",
    )
    delimiter: str = Field(
        default=",",
        description="Column delimiter",
        max_length=5,
    )
    include_header: bool = Field(
        default=True,
        description="Whether to write header row",
    )
    encoding: str = Field(
        default="utf-8",
        description="File encoding",
    )
    append: bool = Field(
        default=False,
        description="Append to existing file instead of overwriting",
    )


class WriteCSVOutput(BaseModel):
    """Output from writing CSV."""

    success: bool
    path: Optional[str] = None
    rows_written: int = 0
    error: Optional[str] = None


class WriteCSVTool(BaseTool[WriteCSVInput, WriteCSVOutput]):
    """Write data to CSV files."""

    metadata = ToolMetadata(
        id="csv_write",
        name="CSV Write",
        description="Write data to CSV files. "
        "Supports custom delimiters, column ordering, and append mode.",
        category="data",
        sandbox_required=False,
    )
    input_type = WriteCSVInput
    output_type = WriteCSVOutput

    async def execute(self, input: WriteCSVInput) -> WriteCSVOutput:
        """Write CSV data to file."""
        try:
            if not input.data:
                return WriteCSVOutput(
                    success=False, error="No data to write"
                )

            # Determine columns
            if input.columns:
                columns = input.columns
            else:
                # Collect all keys from data
                all_keys = set()
                for row in input.data:
                    all_keys.update(row.keys())
                columns = sorted(all_keys)

            # Open file
            mode = "a" if input.append else "w"
            path = Path(input.path)

            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Check if header needed for append mode
            write_header = input.include_header
            if input.append and path.exists() and path.stat().st_size > 0:
                write_header = False

            with open(path, mode, encoding=input.encoding, newline="") as f:
                writer = csv.writer(f, delimiter=input.delimiter)

                if write_header:
                    writer.writerow(columns)

                for row in input.data:
                    writer.writerow([row.get(col, "") for col in columns])

            return WriteCSVOutput(
                success=True,
                path=str(path.absolute()),
                rows_written=len(input.data),
            )

        except PermissionError:
            return WriteCSVOutput(
                success=False, error=f"Permission denied: {input.path}"
            )
        except Exception as e:
            return WriteCSVOutput(success=False, error=str(e))


# --- Filter Tool ---


class FilterOperator(str, Enum):
    """Filter operators."""

    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    GT = "gt"  # Greater than
    GE = "ge"  # Greater than or equal
    LT = "lt"  # Less than
    LE = "le"  # Less than or equal
    CONTAINS = "contains"  # String contains
    STARTSWITH = "startswith"  # String starts with
    ENDSWITH = "endswith"  # String ends with
    IN = "in"  # Value in list
    REGEX = "regex"  # Regex match
    ISNULL = "isnull"  # Is null/empty
    NOTNULL = "notnull"  # Is not null/empty


class FilterCondition(BaseModel):
    """A filter condition."""

    column: str = Field(description="Column to filter on")
    operator: FilterOperator = Field(description="Filter operator")
    value: Optional[Any] = Field(default=None, description="Value to compare against")


class FilterCSVInput(BaseModel):
    """Input for filtering CSV data."""

    data: List[Dict[str, Any]] = Field(
        description="Data to filter",
    )
    conditions: List[FilterCondition] = Field(
        description="Filter conditions (ANDed together)",
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum rows to return",
        ge=1,
    )


class FilterCSVOutput(BaseModel):
    """Output from filtering CSV."""

    success: bool
    data: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    filtered_count: int = 0  # Rows that didn't match
    error: Optional[str] = None


class FilterCSVTool(BaseTool[FilterCSVInput, FilterCSVOutput]):
    """Filter CSV data based on conditions."""

    metadata = ToolMetadata(
        id="csv_filter",
        name="CSV Filter",
        description="Filter CSV data using conditions like equals, contains, greater than, etc. "
        "Multiple conditions are ANDed together.",
        category="data",
        sandbox_required=False,
    )
    input_type = FilterCSVInput
    output_type = FilterCSVOutput

    async def execute(self, input: FilterCSVInput) -> FilterCSVOutput:
        """Filter CSV data."""
        import re

        try:
            result = []
            filtered_count = 0

            for row in input.data:
                if self._matches_all_conditions(row, input.conditions):
                    result.append(row)
                    if input.limit and len(result) >= input.limit:
                        break
                else:
                    filtered_count += 1

            return FilterCSVOutput(
                success=True,
                data=result,
                row_count=len(result),
                filtered_count=filtered_count,
            )

        except Exception as e:
            return FilterCSVOutput(success=False, error=str(e))

    def _matches_all_conditions(
        self, row: Dict[str, Any], conditions: List[FilterCondition]
    ) -> bool:
        """Check if row matches all conditions."""
        import re

        for cond in conditions:
            value = row.get(cond.column)

            if cond.operator == FilterOperator.ISNULL:
                if value is not None and value != "":
                    return False
            elif cond.operator == FilterOperator.NOTNULL:
                if value is None or value == "":
                    return False
            elif cond.operator == FilterOperator.EQ:
                if str(value) != str(cond.value):
                    return False
            elif cond.operator == FilterOperator.NE:
                if str(value) == str(cond.value):
                    return False
            elif cond.operator == FilterOperator.CONTAINS:
                if cond.value is None or str(cond.value) not in str(value or ""):
                    return False
            elif cond.operator == FilterOperator.STARTSWITH:
                if not str(value or "").startswith(str(cond.value or "")):
                    return False
            elif cond.operator == FilterOperator.ENDSWITH:
                if not str(value or "").endswith(str(cond.value or "")):
                    return False
            elif cond.operator == FilterOperator.IN:
                if cond.value is None or value not in cond.value:
                    return False
            elif cond.operator == FilterOperator.REGEX:
                try:
                    if not re.search(str(cond.value or ""), str(value or "")):
                        return False
                except re.error:
                    return False
            elif cond.operator in (
                FilterOperator.GT,
                FilterOperator.GE,
                FilterOperator.LT,
                FilterOperator.LE,
            ):
                try:
                    num_value = float(value) if value else 0
                    num_compare = float(cond.value) if cond.value else 0

                    if cond.operator == FilterOperator.GT and not (num_value > num_compare):
                        return False
                    if cond.operator == FilterOperator.GE and not (num_value >= num_compare):
                        return False
                    if cond.operator == FilterOperator.LT and not (num_value < num_compare):
                        return False
                    if cond.operator == FilterOperator.LE and not (num_value <= num_compare):
                        return False
                except (ValueError, TypeError):
                    return False

        return True


# --- Aggregate Tool ---


class AggregateSpec(BaseModel):
    """Specification for an aggregation."""

    column: str = Field(description="Column to aggregate")
    function: AggregateFunction = Field(description="Aggregation function")
    alias: Optional[str] = Field(default=None, description="Output column name")


class AggregateCSVInput(BaseModel):
    """Input for aggregating CSV data."""

    data: List[Dict[str, Any]] = Field(
        description="Data to aggregate",
    )
    group_by: Optional[List[str]] = Field(
        default=None,
        description="Columns to group by",
    )
    aggregations: List[AggregateSpec] = Field(
        description="Aggregation specifications",
    )


class AggregateCSVOutput(BaseModel):
    """Output from aggregating CSV."""

    success: bool
    data: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None


class AggregateCSVTool(BaseTool[AggregateCSVInput, AggregateCSVOutput]):
    """Aggregate CSV data with grouping."""

    metadata = ToolMetadata(
        id="csv_aggregate",
        name="CSV Aggregate",
        description="Aggregate CSV data with functions like sum, count, avg, min, max. "
        "Supports grouping by columns.",
        category="data",
        sandbox_required=False,
    )
    input_type = AggregateCSVInput
    output_type = AggregateCSVOutput

    async def execute(self, input: AggregateCSVInput) -> AggregateCSVOutput:
        """Aggregate CSV data."""
        try:
            if not input.data:
                return AggregateCSVOutput(success=True, data=[], row_count=0)

            # Group data
            groups: Dict[tuple, List[Dict[str, Any]]] = {}
            for row in input.data:
                if input.group_by:
                    key = tuple(row.get(col, None) for col in input.group_by)
                else:
                    key = ()

                if key not in groups:
                    groups[key] = []
                groups[key].append(row)

            # Aggregate each group
            result = []
            for group_key, rows in groups.items():
                record: Dict[str, Any] = {}

                # Add group by columns
                if input.group_by:
                    for i, col in enumerate(input.group_by):
                        record[col] = group_key[i]

                # Apply aggregations
                for agg in input.aggregations:
                    values = [row.get(agg.column) for row in rows]
                    agg_result = self._aggregate(values, agg.function)
                    alias = agg.alias or f"{agg.function.value}_{agg.column}"
                    record[alias] = agg_result

                result.append(record)

            return AggregateCSVOutput(
                success=True,
                data=result,
                row_count=len(result),
            )

        except Exception as e:
            return AggregateCSVOutput(success=False, error=str(e))

    def _aggregate(self, values: List[Any], func: AggregateFunction) -> Any:
        """Apply aggregation function to values."""
        # Filter None values for numeric operations
        if func in (
            AggregateFunction.SUM,
            AggregateFunction.AVG,
            AggregateFunction.MIN,
            AggregateFunction.MAX,
        ):
            numeric_values = []
            for v in values:
                if v is not None and v != "":
                    try:
                        numeric_values.append(float(v))
                    except (ValueError, TypeError):
                        pass
            values = numeric_values

        if func == AggregateFunction.COUNT:
            return len(values)
        elif func == AggregateFunction.SUM:
            return sum(values) if values else 0
        elif func == AggregateFunction.AVG:
            return sum(values) / len(values) if values else 0
        elif func == AggregateFunction.MIN:
            return min(values) if values else None
        elif func == AggregateFunction.MAX:
            return max(values) if values else None
        elif func == AggregateFunction.FIRST:
            return values[0] if values else None
        elif func == AggregateFunction.LAST:
            return values[-1] if values else None
        else:
            return None


# --- Transform Tool ---


class TransformType(str, Enum):
    """Transform types."""

    RENAME = "rename"
    DROP = "drop"
    SELECT = "select"
    ADD = "add"
    MAP = "map"
    SORT = "sort"


class TransformSpec(BaseModel):
    """Specification for a transformation."""

    type: TransformType = Field(description="Type of transformation")
    column: Optional[str] = Field(default=None, description="Column to transform")
    columns: Optional[List[str]] = Field(default=None, description="Columns (for multi-column ops)")
    new_name: Optional[str] = Field(default=None, description="New column name (for rename)")
    value: Optional[Any] = Field(default=None, description="Value (for add)")
    expression: Optional[str] = Field(default=None, description="Expression (for map)")
    ascending: bool = Field(default=True, description="Sort order (for sort)")


class TransformCSVInput(BaseModel):
    """Input for transforming CSV data."""

    data: List[Dict[str, Any]] = Field(
        description="Data to transform",
    )
    transforms: List[TransformSpec] = Field(
        description="Transformations to apply in order",
    )


class TransformCSVOutput(BaseModel):
    """Output from transforming CSV."""

    success: bool
    data: List[Dict[str, Any]] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None


class TransformCSVTool(BaseTool[TransformCSVInput, TransformCSVOutput]):
    """Transform CSV data with various operations."""

    metadata = ToolMetadata(
        id="csv_transform",
        name="CSV Transform",
        description="Transform CSV data with operations like rename, drop, select, add columns, and sort.",
        category="data",
        sandbox_required=False,
    )
    input_type = TransformCSVInput
    output_type = TransformCSVOutput

    async def execute(self, input: TransformCSVInput) -> TransformCSVOutput:
        """Transform CSV data."""
        try:
            data = [dict(row) for row in input.data]  # Deep copy

            for transform in input.transforms:
                if transform.type == TransformType.RENAME:
                    if transform.column and transform.new_name:
                        for row in data:
                            if transform.column in row:
                                row[transform.new_name] = row.pop(transform.column)

                elif transform.type == TransformType.DROP:
                    cols_to_drop = transform.columns or ([transform.column] if transform.column else [])
                    for row in data:
                        for col in cols_to_drop:
                            row.pop(col, None)

                elif transform.type == TransformType.SELECT:
                    if transform.columns:
                        data = [
                            {k: v for k, v in row.items() if k in transform.columns}
                            for row in data
                        ]

                elif transform.type == TransformType.ADD:
                    if transform.column is not None:
                        for row in data:
                            row[transform.column] = transform.value

                elif transform.type == TransformType.SORT:
                    if transform.column:
                        try:
                            data.sort(
                                key=lambda x: (
                                    float(x.get(transform.column, 0))
                                    if x.get(transform.column) not in (None, "")
                                    else 0
                                ),
                                reverse=not transform.ascending,
                            )
                        except (ValueError, TypeError):
                            # Fall back to string sorting
                            data.sort(
                                key=lambda x: str(x.get(transform.column, "")),
                                reverse=not transform.ascending,
                            )

                elif transform.type == TransformType.MAP:
                    # Simple expression evaluation for safety
                    if transform.column and transform.expression:
                        for row in data:
                            try:
                                # Very basic expression support: column references like {col_name}
                                expr = transform.expression
                                for key, val in row.items():
                                    expr = expr.replace(f"{{{key}}}", str(val) if val else "")
                                row[transform.column] = expr
                            except Exception:
                                pass

            # Collect columns
            columns = set()
            for row in data:
                columns.update(row.keys())

            return TransformCSVOutput(
                success=True,
                data=data,
                columns=sorted(columns),
                row_count=len(data),
            )

        except Exception as e:
            return TransformCSVOutput(success=False, error=str(e))


# --- Factory Function ---


def create_csv_tools(config: CSVConfig | None = None) -> List[BaseTool]:
    """Create all CSV tools with optional configuration."""
    cfg = config or CSVConfig()
    return [
        ReadCSVTool(cfg),
        WriteCSVTool(cfg),
        FilterCSVTool(cfg),
        AggregateCSVTool(cfg),
        TransformCSVTool(cfg),
    ]
