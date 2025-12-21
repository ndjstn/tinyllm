"""Tests for CSV processing tools."""

import os
import tempfile
from pathlib import Path

import pytest

from tinyllm.tools.csv_tool import (
    ReadCSVTool,
    ReadCSVInput,
    WriteCSVTool,
    WriteCSVInput,
    FilterCSVTool,
    FilterCSVInput,
    FilterCondition,
    FilterOperator,
    AggregateCSVTool,
    AggregateCSVInput,
    AggregateSpec,
    AggregateFunction,
    TransformCSVTool,
    TransformCSVInput,
    TransformSpec,
    TransformType,
    create_csv_tools,
)


class TestReadCSVTool:
    """Tests for ReadCSVTool."""

    @pytest.fixture
    def tool(self):
        return ReadCSVTool()

    @pytest.fixture
    def sample_csv(self):
        return "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago"

    @pytest.mark.asyncio
    async def test_read_from_content(self, tool, sample_csv):
        """Test reading CSV from content string."""
        result = await tool.execute(ReadCSVInput(content=sample_csv))
        assert result.success is True
        assert result.row_count == 3
        assert result.columns == ["name", "age", "city"]
        assert result.data[0] == {"name": "Alice", "age": "30", "city": "NYC"}

    @pytest.mark.asyncio
    async def test_read_from_file(self, tool, sample_csv, tmp_path):
        """Test reading CSV from file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(sample_csv)

        result = await tool.execute(ReadCSVInput(path=str(csv_file)))
        assert result.success is True
        assert result.row_count == 3

    @pytest.mark.asyncio
    async def test_read_select_columns(self, tool, sample_csv):
        """Test reading with column selection."""
        result = await tool.execute(
            ReadCSVInput(content=sample_csv, columns=["name", "city"])
        )
        assert result.success is True
        assert result.columns == ["name", "city"]
        assert "age" not in result.data[0]

    @pytest.mark.asyncio
    async def test_read_with_limit(self, tool, sample_csv):
        """Test reading with row limit."""
        result = await tool.execute(ReadCSVInput(content=sample_csv, limit=2))
        assert result.success is True
        assert result.row_count == 2
        assert result.total_rows == 3

    @pytest.mark.asyncio
    async def test_read_skip_rows(self, tool, sample_csv):
        """Test reading with row skipping."""
        result = await tool.execute(ReadCSVInput(content=sample_csv, skip_rows=1))
        assert result.success is True
        assert result.row_count == 2
        assert result.data[0]["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_read_no_header(self, tool):
        """Test reading CSV without header."""
        csv = "Alice,30,NYC\nBob,25,LA"
        result = await tool.execute(
            ReadCSVInput(content=csv, has_header=False)
        )
        assert result.success is True
        assert result.columns == ["col_0", "col_1", "col_2"]

    @pytest.mark.asyncio
    async def test_read_custom_delimiter(self, tool):
        """Test reading with custom delimiter."""
        csv = "name;age;city\nAlice;30;NYC"
        result = await tool.execute(
            ReadCSVInput(content=csv, delimiter=";")
        )
        assert result.success is True
        assert result.data[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_read_missing_column(self, tool, sample_csv):
        """Test reading with non-existent column."""
        result = await tool.execute(
            ReadCSVInput(content=sample_csv, columns=["name", "nonexistent"])
        )
        assert result.success is False
        assert "Column not found" in result.error

    @pytest.mark.asyncio
    async def test_read_missing_file(self, tool):
        """Test reading non-existent file."""
        result = await tool.execute(ReadCSVInput(path="/nonexistent/file.csv"))
        assert result.success is False
        assert "File not found" in result.error

    @pytest.mark.asyncio
    async def test_read_empty_csv(self, tool):
        """Test reading empty CSV (header only)."""
        result = await tool.execute(ReadCSVInput(content="name,age,city"))
        assert result.success is True
        assert result.row_count == 0
        assert result.columns == ["name", "age", "city"]


class TestWriteCSVTool:
    """Tests for WriteCSVTool."""

    @pytest.fixture
    def tool(self):
        return WriteCSVTool()

    @pytest.fixture
    def sample_data(self):
        return [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
        ]

    @pytest.mark.asyncio
    async def test_write_basic(self, tool, sample_data, tmp_path):
        """Test basic CSV writing."""
        csv_file = tmp_path / "output.csv"
        result = await tool.execute(
            WriteCSVInput(path=str(csv_file), data=sample_data)
        )
        assert result.success is True
        assert result.rows_written == 2
        assert csv_file.exists()

        content = csv_file.read_text()
        assert "name" in content
        assert "Alice" in content

    @pytest.mark.asyncio
    async def test_write_custom_columns(self, tool, sample_data, tmp_path):
        """Test writing with custom column order."""
        csv_file = tmp_path / "output.csv"
        result = await tool.execute(
            WriteCSVInput(
                path=str(csv_file),
                data=sample_data,
                columns=["city", "name"],
            )
        )
        assert result.success is True

        content = csv_file.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "city,name"

    @pytest.mark.asyncio
    async def test_write_no_header(self, tool, sample_data, tmp_path):
        """Test writing without header."""
        csv_file = tmp_path / "output.csv"
        result = await tool.execute(
            WriteCSVInput(
                path=str(csv_file),
                data=sample_data,
                include_header=False,
            )
        )
        assert result.success is True

        content = csv_file.read_text()
        assert "name" not in content.split("\n")[0] or "Alice" in content.split("\n")[0]

    @pytest.mark.asyncio
    async def test_write_append(self, tool, sample_data, tmp_path):
        """Test appending to existing file."""
        csv_file = tmp_path / "output.csv"

        # Write initial data
        await tool.execute(
            WriteCSVInput(path=str(csv_file), data=sample_data[:1])
        )

        # Append more data
        result = await tool.execute(
            WriteCSVInput(
                path=str(csv_file),
                data=sample_data[1:],
                append=True,
            )
        )
        assert result.success is True

        content = csv_file.read_text()
        assert "Alice" in content
        assert "Bob" in content
        # Header should only appear once
        assert content.count("name") == 1

    @pytest.mark.asyncio
    async def test_write_empty_data(self, tool, tmp_path):
        """Test writing empty data."""
        csv_file = tmp_path / "output.csv"
        result = await tool.execute(
            WriteCSVInput(path=str(csv_file), data=[])
        )
        assert result.success is False
        assert "No data" in result.error


class TestFilterCSVTool:
    """Tests for FilterCSVTool."""

    @pytest.fixture
    def tool(self):
        return FilterCSVTool()

    @pytest.fixture
    def sample_data(self):
        return [
            {"name": "Alice", "age": "30", "city": "NYC"},
            {"name": "Bob", "age": "25", "city": "LA"},
            {"name": "Charlie", "age": "35", "city": "NYC"},
            {"name": "Diana", "age": "28", "city": "Chicago"},
        ]

    @pytest.mark.asyncio
    async def test_filter_eq(self, tool, sample_data):
        """Test equality filter."""
        result = await tool.execute(
            FilterCSVInput(
                data=sample_data,
                conditions=[FilterCondition(column="city", operator=FilterOperator.EQ, value="NYC")],
            )
        )
        assert result.success is True
        assert result.row_count == 2
        assert all(row["city"] == "NYC" for row in result.data)

    @pytest.mark.asyncio
    async def test_filter_gt(self, tool, sample_data):
        """Test greater than filter."""
        result = await tool.execute(
            FilterCSVInput(
                data=sample_data,
                conditions=[FilterCondition(column="age", operator=FilterOperator.GT, value="28")],
            )
        )
        assert result.success is True
        assert result.row_count == 2  # Alice (30) and Charlie (35)

    @pytest.mark.asyncio
    async def test_filter_contains(self, tool, sample_data):
        """Test contains filter."""
        result = await tool.execute(
            FilterCSVInput(
                data=sample_data,
                conditions=[FilterCondition(column="name", operator=FilterOperator.CONTAINS, value="li")],
            )
        )
        assert result.success is True
        assert result.row_count == 2  # Alice and Charlie

    @pytest.mark.asyncio
    async def test_filter_multiple_conditions(self, tool, sample_data):
        """Test multiple filter conditions (AND)."""
        result = await tool.execute(
            FilterCSVInput(
                data=sample_data,
                conditions=[
                    FilterCondition(column="city", operator=FilterOperator.EQ, value="NYC"),
                    FilterCondition(column="age", operator=FilterOperator.GT, value="25"),
                ],
            )
        )
        assert result.success is True
        assert result.row_count == 2

    @pytest.mark.asyncio
    async def test_filter_with_limit(self, tool, sample_data):
        """Test filter with limit."""
        result = await tool.execute(
            FilterCSVInput(
                data=sample_data,
                conditions=[FilterCondition(column="age", operator=FilterOperator.GT, value="20")],
                limit=2,
            )
        )
        assert result.success is True
        assert result.row_count == 2

    @pytest.mark.asyncio
    async def test_filter_in_operator(self, tool, sample_data):
        """Test IN operator."""
        result = await tool.execute(
            FilterCSVInput(
                data=sample_data,
                conditions=[
                    FilterCondition(
                        column="city",
                        operator=FilterOperator.IN,
                        value=["NYC", "LA"],
                    )
                ],
            )
        )
        assert result.success is True
        assert result.row_count == 3

    @pytest.mark.asyncio
    async def test_filter_regex(self, tool, sample_data):
        """Test regex filter."""
        result = await tool.execute(
            FilterCSVInput(
                data=sample_data,
                conditions=[
                    FilterCondition(column="name", operator=FilterOperator.REGEX, value="^[A-C]")
                ],
            )
        )
        assert result.success is True
        assert result.row_count == 3  # Alice, Bob, Charlie


class TestAggregateCSVTool:
    """Tests for AggregateCSVTool."""

    @pytest.fixture
    def tool(self):
        return AggregateCSVTool()

    @pytest.fixture
    def sample_data(self):
        return [
            {"city": "NYC", "sales": "100", "quantity": "5"},
            {"city": "NYC", "sales": "200", "quantity": "10"},
            {"city": "LA", "sales": "150", "quantity": "7"},
            {"city": "LA", "sales": "50", "quantity": "3"},
        ]

    @pytest.mark.asyncio
    async def test_aggregate_sum(self, tool, sample_data):
        """Test sum aggregation."""
        result = await tool.execute(
            AggregateCSVInput(
                data=sample_data,
                aggregations=[
                    AggregateSpec(column="sales", function=AggregateFunction.SUM)
                ],
            )
        )
        assert result.success is True
        assert result.row_count == 1
        assert result.data[0]["sum_sales"] == 500

    @pytest.mark.asyncio
    async def test_aggregate_with_group(self, tool, sample_data):
        """Test aggregation with grouping."""
        result = await tool.execute(
            AggregateCSVInput(
                data=sample_data,
                group_by=["city"],
                aggregations=[
                    AggregateSpec(column="sales", function=AggregateFunction.SUM)
                ],
            )
        )
        assert result.success is True
        assert result.row_count == 2

        nyc_data = next(r for r in result.data if r["city"] == "NYC")
        assert nyc_data["sum_sales"] == 300

    @pytest.mark.asyncio
    async def test_aggregate_multiple_functions(self, tool, sample_data):
        """Test multiple aggregation functions."""
        result = await tool.execute(
            AggregateCSVInput(
                data=sample_data,
                aggregations=[
                    AggregateSpec(column="sales", function=AggregateFunction.SUM, alias="total"),
                    AggregateSpec(column="sales", function=AggregateFunction.AVG, alias="average"),
                    AggregateSpec(column="sales", function=AggregateFunction.COUNT, alias="count"),
                ],
            )
        )
        assert result.success is True
        assert result.data[0]["total"] == 500
        assert result.data[0]["average"] == 125
        assert result.data[0]["count"] == 4

    @pytest.mark.asyncio
    async def test_aggregate_min_max(self, tool, sample_data):
        """Test min/max aggregation."""
        result = await tool.execute(
            AggregateCSVInput(
                data=sample_data,
                aggregations=[
                    AggregateSpec(column="sales", function=AggregateFunction.MIN),
                    AggregateSpec(column="sales", function=AggregateFunction.MAX),
                ],
            )
        )
        assert result.success is True
        assert result.data[0]["min_sales"] == 50
        assert result.data[0]["max_sales"] == 200

    @pytest.mark.asyncio
    async def test_aggregate_empty_data(self, tool):
        """Test aggregation on empty data."""
        result = await tool.execute(
            AggregateCSVInput(
                data=[],
                aggregations=[
                    AggregateSpec(column="sales", function=AggregateFunction.SUM)
                ],
            )
        )
        assert result.success is True
        assert result.row_count == 0


class TestTransformCSVTool:
    """Tests for TransformCSVTool."""

    @pytest.fixture
    def tool(self):
        return TransformCSVTool()

    @pytest.fixture
    def sample_data(self):
        return [
            {"name": "Alice", "age": "30", "city": "NYC"},
            {"name": "Bob", "age": "25", "city": "LA"},
        ]

    @pytest.mark.asyncio
    async def test_transform_rename(self, tool, sample_data):
        """Test column renaming."""
        result = await tool.execute(
            TransformCSVInput(
                data=sample_data,
                transforms=[
                    TransformSpec(type=TransformType.RENAME, column="name", new_name="full_name")
                ],
            )
        )
        assert result.success is True
        assert "full_name" in result.data[0]
        assert "name" not in result.data[0]

    @pytest.mark.asyncio
    async def test_transform_drop(self, tool, sample_data):
        """Test column dropping."""
        result = await tool.execute(
            TransformCSVInput(
                data=sample_data,
                transforms=[TransformSpec(type=TransformType.DROP, column="age")],
            )
        )
        assert result.success is True
        assert "age" not in result.data[0]

    @pytest.mark.asyncio
    async def test_transform_select(self, tool, sample_data):
        """Test column selection."""
        result = await tool.execute(
            TransformCSVInput(
                data=sample_data,
                transforms=[TransformSpec(type=TransformType.SELECT, columns=["name", "city"])],
            )
        )
        assert result.success is True
        assert set(result.data[0].keys()) == {"name", "city"}

    @pytest.mark.asyncio
    async def test_transform_add(self, tool, sample_data):
        """Test adding column."""
        result = await tool.execute(
            TransformCSVInput(
                data=sample_data,
                transforms=[TransformSpec(type=TransformType.ADD, column="country", value="USA")],
            )
        )
        assert result.success is True
        assert all(row["country"] == "USA" for row in result.data)

    @pytest.mark.asyncio
    async def test_transform_sort(self, tool, sample_data):
        """Test sorting."""
        result = await tool.execute(
            TransformCSVInput(
                data=sample_data,
                transforms=[TransformSpec(type=TransformType.SORT, column="age", ascending=True)],
            )
        )
        assert result.success is True
        assert result.data[0]["name"] == "Bob"  # age 25 first

    @pytest.mark.asyncio
    async def test_transform_multiple(self, tool, sample_data):
        """Test multiple transformations."""
        result = await tool.execute(
            TransformCSVInput(
                data=sample_data,
                transforms=[
                    TransformSpec(type=TransformType.DROP, column="city"),
                    TransformSpec(type=TransformType.RENAME, column="name", new_name="person"),
                    TransformSpec(type=TransformType.ADD, column="status", value="active"),
                ],
            )
        )
        assert result.success is True
        assert "city" not in result.data[0]
        assert "person" in result.data[0]
        assert result.data[0]["status"] == "active"


class TestCreateCSVTools:
    """Tests for create_csv_tools factory."""

    def test_creates_all_tools(self):
        """Test that factory creates all tools."""
        tools = create_csv_tools()
        assert len(tools) == 5

        tool_ids = {t.metadata.id for t in tools}
        expected = {"csv_read", "csv_write", "csv_filter", "csv_aggregate", "csv_transform"}
        assert tool_ids == expected

    def test_all_tools_have_data_category(self):
        """Test all tools are in data category."""
        tools = create_csv_tools()
        for tool in tools:
            assert tool.metadata.category == "data"
