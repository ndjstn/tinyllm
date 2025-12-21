"""Tests for JSON processing tools."""

import pytest

from tinyllm.tools.json_tool import (
    JSONParseTool,
    JSONParseInput,
    JSONQueryTool,
    JSONQueryInput,
    JSONValidateTool,
    JSONValidateInput,
    JSONTransformTool,
    JSONTransformInput,
    JSONMergeTool,
    JSONMergeInput,
    JSONFlattenTool,
    JSONFlattenInput,
    create_json_tools,
)


class TestJSONParseTool:
    """Tests for JSONParseTool."""

    @pytest.fixture
    def tool(self):
        return JSONParseTool()

    @pytest.mark.asyncio
    async def test_parse_object(self, tool):
        """Test parsing a JSON object."""
        result = await tool.execute(
            JSONParseInput(data='{"name": "Alice", "age": 30}')
        )
        assert result.success is True
        assert result.result == {"name": "Alice", "age": 30}
        assert result.data_type == "object"

    @pytest.mark.asyncio
    async def test_parse_array(self, tool):
        """Test parsing a JSON array."""
        result = await tool.execute(JSONParseInput(data="[1, 2, 3]"))
        assert result.success is True
        assert result.result == [1, 2, 3]
        assert result.data_type == "array"

    @pytest.mark.asyncio
    async def test_parse_string(self, tool):
        """Test parsing a JSON string."""
        result = await tool.execute(JSONParseInput(data='"hello"'))
        assert result.success is True
        assert result.result == "hello"
        assert result.data_type == "string"

    @pytest.mark.asyncio
    async def test_parse_number(self, tool):
        """Test parsing a JSON number."""
        result = await tool.execute(JSONParseInput(data="42.5"))
        assert result.success is True
        assert result.result == 42.5
        assert result.data_type == "number"

    @pytest.mark.asyncio
    async def test_parse_boolean(self, tool):
        """Test parsing a JSON boolean."""
        result = await tool.execute(JSONParseInput(data="true"))
        assert result.success is True
        assert result.result is True
        assert result.data_type == "boolean"

    @pytest.mark.asyncio
    async def test_parse_null(self, tool):
        """Test parsing JSON null."""
        result = await tool.execute(JSONParseInput(data="null"))
        assert result.success is True
        assert result.result is None
        assert result.data_type == "null"

    @pytest.mark.asyncio
    async def test_parse_invalid_json_strict(self, tool):
        """Test parsing invalid JSON in strict mode."""
        result = await tool.execute(
            JSONParseInput(data="{invalid}", strict=True)
        )
        assert result.success is False
        assert result.error is not None
        assert "Invalid JSON" in result.error

    @pytest.mark.asyncio
    async def test_parse_invalid_json_non_strict(self, tool):
        """Test parsing invalid JSON in non-strict mode."""
        result = await tool.execute(
            JSONParseInput(data="{invalid}", strict=False)
        )
        assert result.success is True
        assert result.result is None


class TestJSONQueryTool:
    """Tests for JSONQueryTool."""

    @pytest.fixture
    def tool(self):
        return JSONQueryTool()

    @pytest.fixture
    def sample_data(self):
        return {
            "store": {
                "book": [
                    {"title": "Book 1", "price": 10},
                    {"title": "Book 2", "price": 20},
                    {"title": "Book 3", "price": 15},
                ],
                "name": "My Store",
            },
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ],
        }

    @pytest.mark.asyncio
    async def test_query_simple_property(self, tool, sample_data):
        """Test querying a simple property."""
        result = await tool.execute(
            JSONQueryInput(data=sample_data, jsonpath="$.store.name")
        )
        assert result.success is True
        assert result.results == ["My Store"]
        assert result.count == 1

    @pytest.mark.asyncio
    async def test_query_array_index(self, tool, sample_data):
        """Test querying an array index."""
        result = await tool.execute(
            JSONQueryInput(data=sample_data, jsonpath="$.store.book[0].title")
        )
        assert result.success is True
        assert result.results == ["Book 1"]

    @pytest.mark.asyncio
    async def test_query_array_wildcard(self, tool, sample_data):
        """Test querying with array wildcard."""
        result = await tool.execute(
            JSONQueryInput(data=sample_data, jsonpath="$.users[*].name")
        )
        assert result.success is True
        assert result.results == ["Alice", "Bob"]
        assert result.count == 2

    @pytest.mark.asyncio
    async def test_query_from_string(self, tool):
        """Test querying from JSON string input."""
        result = await tool.execute(
            JSONQueryInput(
                data='{"items": [1, 2, 3]}',
                jsonpath="$.items[*]",
            )
        )
        assert result.success is True
        assert result.results == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_query_no_match(self, tool, sample_data):
        """Test query with no matches."""
        result = await tool.execute(
            JSONQueryInput(data=sample_data, jsonpath="$.nonexistent")
        )
        assert result.success is True
        assert result.results == []
        assert result.count == 0

    @pytest.mark.asyncio
    async def test_query_invalid_jsonpath(self, tool, sample_data):
        """Test query with invalid JSONPath."""
        result = await tool.execute(
            JSONQueryInput(data=sample_data, jsonpath="invalid")
        )
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_query_invalid_json(self, tool):
        """Test query with invalid JSON string."""
        result = await tool.execute(
            JSONQueryInput(data="{invalid}", jsonpath="$.key")
        )
        assert result.success is False
        assert "Invalid JSON" in result.error


class TestJSONValidateTool:
    """Tests for JSONValidateTool."""

    @pytest.fixture
    def tool(self):
        return JSONValidateTool()

    @pytest.mark.asyncio
    async def test_validate_valid_object(self, tool):
        """Test validating a valid object."""
        json_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
            "required": ["name"],
        }
        result = await tool.execute(
            JSONValidateInput(
                data={"name": "Alice", "age": 30},
                json_schema=json_schema,
            )
        )
        assert result.success is True
        assert result.valid is True
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_validate_missing_required(self, tool):
        """Test validation with missing required field."""
        json_schema = {
            "type": "object",
            "required": ["name", "email"],
        }
        result = await tool.execute(
            JSONValidateInput(
                data={"name": "Alice"},
                json_schema=json_schema,
            )
        )
        assert result.success is True
        assert result.valid is False
        assert any("email" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_validate_wrong_type(self, tool):
        """Test validation with wrong type."""
        json_schema = {"type": "string"}
        result = await tool.execute(
            JSONValidateInput(data="123", json_schema=json_schema)  # Pass as JSON string
        )
        assert result.success is True
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_validate_array(self, tool):
        """Test validating an array."""
        json_schema = {
            "type": "array",
            "items": {"type": "number"},
        }
        result = await tool.execute(
            JSONValidateInput(data=[1, 2, 3], json_schema=json_schema)
        )
        assert result.success is True
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_validate_from_string(self, tool):
        """Test validating from JSON string."""
        json_schema = {"type": "object"}
        result = await tool.execute(
            JSONValidateInput(data='{"key": "value"}', json_schema=json_schema)
        )
        assert result.success is True
        assert result.valid is True


class TestJSONTransformTool:
    """Tests for JSONTransformTool."""

    @pytest.fixture
    def tool(self):
        return JSONTransformTool()

    @pytest.mark.asyncio
    async def test_set_operation(self, tool):
        """Test set operation."""
        result = await tool.execute(
            JSONTransformInput(
                data={"name": "Alice"},
                operations=[{"op": "set", "path": "$.age", "value": 30}],
            )
        )
        assert result.success is True
        assert result.result == {"name": "Alice", "age": 30}
        assert result.operations_applied == 1

    @pytest.mark.asyncio
    async def test_delete_operation(self, tool):
        """Test delete operation."""
        result = await tool.execute(
            JSONTransformInput(
                data={"name": "Alice", "password": "secret"},
                operations=[{"op": "delete", "path": "$.password"}],
            )
        )
        assert result.success is True
        assert result.result == {"name": "Alice"}

    @pytest.mark.asyncio
    async def test_rename_operation(self, tool):
        """Test rename operation."""
        result = await tool.execute(
            JSONTransformInput(
                data={"old_name": "value"},
                operations=[{"op": "rename", "from": "old_name", "to": "new_name"}],
            )
        )
        assert result.success is True
        assert result.result == {"new_name": "value"}

    @pytest.mark.asyncio
    async def test_copy_operation(self, tool):
        """Test copy operation."""
        result = await tool.execute(
            JSONTransformInput(
                data={"source": "value"},
                operations=[{"op": "copy", "from": "$.source", "to": "$.target"}],
            )
        )
        assert result.success is True
        assert result.result["source"] == "value"
        assert result.result["target"] == "value"

    @pytest.mark.asyncio
    async def test_move_operation(self, tool):
        """Test move operation."""
        result = await tool.execute(
            JSONTransformInput(
                data={"source": "value"},
                operations=[{"op": "move", "from": "$.source", "to": "$.target"}],
            )
        )
        assert result.success is True
        assert "source" not in result.result
        assert result.result["target"] == "value"

    @pytest.mark.asyncio
    async def test_multiple_operations(self, tool):
        """Test multiple operations."""
        result = await tool.execute(
            JSONTransformInput(
                data={"a": 1, "b": 2},
                operations=[
                    {"op": "set", "path": "$.c", "value": 3},
                    {"op": "delete", "path": "$.a"},
                ],
            )
        )
        assert result.success is True
        assert result.result == {"b": 2, "c": 3}
        assert result.operations_applied == 2

    @pytest.mark.asyncio
    async def test_unknown_operation(self, tool):
        """Test unknown operation."""
        result = await tool.execute(
            JSONTransformInput(
                data={"a": 1},
                operations=[{"op": "unknown"}],
            )
        )
        assert result.success is False
        assert "Unknown operation" in result.error


class TestJSONMergeTool:
    """Tests for JSONMergeTool."""

    @pytest.fixture
    def tool(self):
        return JSONMergeTool()

    @pytest.mark.asyncio
    async def test_shallow_merge(self, tool):
        """Test shallow merge."""
        result = await tool.execute(
            JSONMergeInput(
                target={"a": 1, "b": {"x": 1}},
                source={"b": {"y": 2}, "c": 3},
                deep=False,
            )
        )
        assert result.success is True
        assert result.result == {"a": 1, "b": {"y": 2}, "c": 3}

    @pytest.mark.asyncio
    async def test_deep_merge(self, tool):
        """Test deep merge."""
        result = await tool.execute(
            JSONMergeInput(
                target={"a": 1, "b": {"x": 1}},
                source={"b": {"y": 2}, "c": 3},
                deep=True,
            )
        )
        assert result.success is True
        assert result.result == {"a": 1, "b": {"x": 1, "y": 2}, "c": 3}

    @pytest.mark.asyncio
    async def test_merge_from_strings(self, tool):
        """Test merging from JSON strings."""
        result = await tool.execute(
            JSONMergeInput(
                target='{"a": 1}',
                source='{"b": 2}',
                deep=True,
            )
        )
        assert result.success is True
        assert result.result == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_merge_non_objects(self, tool):
        """Test merging non-objects fails."""
        result = await tool.execute(
            JSONMergeInput(
                target="[1, 2, 3]",  # Pass as JSON string
                source='{"a": 1}',
                deep=True,
            )
        )
        assert result.success is False
        assert "must be JSON objects" in result.error


class TestJSONFlattenTool:
    """Tests for JSONFlattenTool."""

    @pytest.fixture
    def tool(self):
        return JSONFlattenTool()

    @pytest.mark.asyncio
    async def test_flatten_nested_object(self, tool):
        """Test flattening a nested object."""
        result = await tool.execute(
            JSONFlattenInput(
                data={"a": {"b": {"c": 1}}},
            )
        )
        assert result.success is True
        assert result.result == {"a.b.c": 1}

    @pytest.mark.asyncio
    async def test_flatten_with_array(self, tool):
        """Test flattening with arrays."""
        result = await tool.execute(
            JSONFlattenInput(
                data={"items": [1, 2, 3]},
            )
        )
        assert result.success is True
        assert result.result == {
            "items[0]": 1,
            "items[1]": 2,
            "items[2]": 3,
        }

    @pytest.mark.asyncio
    async def test_flatten_custom_separator(self, tool):
        """Test flattening with custom separator."""
        result = await tool.execute(
            JSONFlattenInput(
                data={"a": {"b": 1}},
                separator="_",
            )
        )
        assert result.success is True
        assert result.result == {"a_b": 1}

    @pytest.mark.asyncio
    async def test_flatten_max_depth(self, tool):
        """Test flattening with max depth."""
        result = await tool.execute(
            JSONFlattenInput(
                data={"a": {"b": {"c": 1}}},
                max_depth=1,
            )
        )
        assert result.success is True
        assert result.result == {"a": {"b": {"c": 1}}}

    @pytest.mark.asyncio
    async def test_flatten_from_string(self, tool):
        """Test flattening from JSON string."""
        result = await tool.execute(
            JSONFlattenInput(data='{"a": {"b": 1}}')
        )
        assert result.success is True
        assert result.result == {"a.b": 1}

    @pytest.mark.asyncio
    async def test_flatten_non_object(self, tool):
        """Test flattening non-object fails."""
        result = await tool.execute(
            JSONFlattenInput(data="[1, 2, 3]")  # Pass as JSON string
        )
        assert result.success is False
        assert "must be a JSON object" in result.error


class TestCreateJSONTools:
    """Tests for create_json_tools factory."""

    def test_creates_all_tools(self):
        """Test that factory creates all tools."""
        tools = create_json_tools()
        assert len(tools) == 6

        tool_ids = {t.metadata.id for t in tools}
        expected = {
            "json_parse",
            "json_query",
            "json_validate",
            "json_transform",
            "json_merge",
            "json_flatten",
        }
        assert tool_ids == expected

    def test_all_tools_have_data_category(self):
        """Test all tools are in data category."""
        tools = create_json_tools()
        for tool in tools:
            assert tool.metadata.category == "data"
