"""Tests for text processing tools."""

import pytest

from tinyllm.tools.text_processor import (
    RegexMatchTool,
    RegexMatchInput,
    RegexReplaceTool,
    RegexReplaceInput,
    TextSplitTool,
    TextSplitInput,
    TextJoinTool,
    TextJoinInput,
    TextCleanTool,
    TextCleanInput,
    CleanOperation,
    TextCaseTool,
    TextCaseInput,
    CaseOperation,
    TextExtractTool,
    TextExtractInput,
    ExtractType,
    TextTruncateTool,
    TextTruncateInput,
    TruncatePosition,
    create_text_tools,
)


class TestRegexMatchTool:
    """Tests for RegexMatchTool."""

    @pytest.fixture
    def tool(self):
        return RegexMatchTool()

    @pytest.mark.asyncio
    async def test_match_simple(self, tool):
        """Test simple regex matching."""
        result = await tool.execute(
            RegexMatchInput(text="hello world", pattern=r"\w+")
        )
        assert result.success is True
        assert result.count == 2
        assert result.matches[0].match == "hello"
        assert result.matches[1].match == "world"

    @pytest.mark.asyncio
    async def test_match_with_groups(self, tool):
        """Test matching with capture groups."""
        result = await tool.execute(
            RegexMatchInput(
                text="John: 25, Jane: 30",
                pattern=r"(\w+): (\d+)",
            )
        )
        assert result.success is True
        assert result.count == 2
        assert result.matches[0].groups == ["John", "25"]

    @pytest.mark.asyncio
    async def test_match_named_groups(self, tool):
        """Test matching with named groups."""
        result = await tool.execute(
            RegexMatchInput(
                text="email: test@example.com",
                pattern=r"(?P<label>\w+): (?P<value>\S+)",
            )
        )
        assert result.success is True
        assert result.matches[0].group_dict["label"] == "email"
        assert result.matches[0].group_dict["value"] == "test@example.com"

    @pytest.mark.asyncio
    async def test_match_case_insensitive(self, tool):
        """Test case insensitive matching."""
        result = await tool.execute(
            RegexMatchInput(
                text="Hello WORLD",
                pattern=r"hello",
                flags=["i"],
            )
        )
        assert result.success is True
        assert result.count == 1

    @pytest.mark.asyncio
    async def test_match_first_only(self, tool):
        """Test finding first match only."""
        result = await tool.execute(
            RegexMatchInput(
                text="one two three",
                pattern=r"\w+",
                find_all=False,
            )
        )
        assert result.success is True
        assert result.count == 1
        assert result.matches[0].match == "one"

    @pytest.mark.asyncio
    async def test_match_no_results(self, tool):
        """Test no matches."""
        result = await tool.execute(
            RegexMatchInput(text="hello world", pattern=r"\d+")
        )
        assert result.success is True
        assert result.count == 0

    @pytest.mark.asyncio
    async def test_match_invalid_regex(self, tool):
        """Test invalid regex pattern."""
        result = await tool.execute(
            RegexMatchInput(text="hello", pattern=r"[invalid")
        )
        assert result.success is False
        assert "Invalid regex" in result.error


class TestRegexReplaceTool:
    """Tests for RegexReplaceTool."""

    @pytest.fixture
    def tool(self):
        return RegexReplaceTool()

    @pytest.mark.asyncio
    async def test_replace_simple(self, tool):
        """Test simple replacement."""
        result = await tool.execute(
            RegexReplaceInput(
                text="hello world",
                pattern=r"world",
                replacement="universe",
            )
        )
        assert result.success is True
        assert result.result == "hello universe"
        assert result.replacements_made == 1

    @pytest.mark.asyncio
    async def test_replace_all(self, tool):
        """Test replacing all matches."""
        result = await tool.execute(
            RegexReplaceInput(
                text="cat dog cat dog",
                pattern=r"cat",
                replacement="bird",
            )
        )
        assert result.success is True
        assert result.result == "bird dog bird dog"
        assert result.replacements_made == 2

    @pytest.mark.asyncio
    async def test_replace_with_backreference(self, tool):
        """Test replacement with backreference."""
        result = await tool.execute(
            RegexReplaceInput(
                text="John Smith",
                pattern=r"(\w+) (\w+)",
                replacement=r"\2, \1",
            )
        )
        assert result.success is True
        assert result.result == "Smith, John"

    @pytest.mark.asyncio
    async def test_replace_limited(self, tool):
        """Test limited replacements."""
        result = await tool.execute(
            RegexReplaceInput(
                text="a a a a a",
                pattern=r"a",
                replacement="b",
                count=2,
            )
        )
        assert result.success is True
        assert result.result == "b b a a a"
        assert result.replacements_made == 2


class TestTextSplitTool:
    """Tests for TextSplitTool."""

    @pytest.fixture
    def tool(self):
        return TextSplitTool()

    @pytest.mark.asyncio
    async def test_split_on_whitespace(self, tool):
        """Test splitting on whitespace."""
        result = await tool.execute(
            TextSplitInput(text="hello  world  test")
        )
        assert result.success is True
        assert result.parts == ["hello", "world", "test"]

    @pytest.mark.asyncio
    async def test_split_on_separator(self, tool):
        """Test splitting on separator."""
        result = await tool.execute(
            TextSplitInput(text="a,b,c", separator=",")
        )
        assert result.success is True
        assert result.parts == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_split_on_regex(self, tool):
        """Test splitting on regex."""
        result = await tool.execute(
            TextSplitInput(
                text="a1b2c3d",
                separator=r"\d",
                regex=True,
            )
        )
        assert result.success is True
        assert result.parts == ["a", "b", "c", "d"]

    @pytest.mark.asyncio
    async def test_split_with_limit(self, tool):
        """Test splitting with limit."""
        result = await tool.execute(
            TextSplitInput(
                text="a,b,c,d",
                separator=",",
                max_splits=2,
            )
        )
        assert result.success is True
        assert result.parts == ["a", "b", "c,d"]


class TestTextJoinTool:
    """Tests for TextJoinTool."""

    @pytest.fixture
    def tool(self):
        return TextJoinTool()

    @pytest.mark.asyncio
    async def test_join_simple(self, tool):
        """Test simple join."""
        result = await tool.execute(
            TextJoinInput(parts=["a", "b", "c"], separator=",")
        )
        assert result.success is True
        assert result.result == "a,b,c"

    @pytest.mark.asyncio
    async def test_join_with_prefix_suffix(self, tool):
        """Test join with prefix and suffix."""
        result = await tool.execute(
            TextJoinInput(
                parts=["1", "2", "3"],
                separator=", ",
                prefix="[",
                suffix="]",
            )
        )
        assert result.success is True
        assert result.result == "[1, 2, 3]"


class TestTextCleanTool:
    """Tests for TextCleanTool."""

    @pytest.fixture
    def tool(self):
        return TextCleanTool()

    @pytest.mark.asyncio
    async def test_clean_whitespace(self, tool):
        """Test whitespace normalization."""
        result = await tool.execute(
            TextCleanInput(
                text="  hello   world  \n\t test  ",
                operations=[CleanOperation.WHITESPACE],
            )
        )
        assert result.success is True
        assert result.result == "hello world test"

    @pytest.mark.asyncio
    async def test_clean_html(self, tool):
        """Test HTML stripping."""
        result = await tool.execute(
            TextCleanInput(
                text="<p>Hello <b>World</b></p>",
                operations=[CleanOperation.HTML],
            )
        )
        assert result.success is True
        assert result.result == "Hello World"

    @pytest.mark.asyncio
    async def test_clean_punctuation(self, tool):
        """Test punctuation removal."""
        result = await tool.execute(
            TextCleanInput(
                text="Hello, World! How are you?",
                operations=[CleanOperation.PUNCTUATION],
            )
        )
        assert result.success is True
        assert result.result == "Hello World How are you"

    @pytest.mark.asyncio
    async def test_clean_multiple(self, tool):
        """Test multiple cleaning operations."""
        result = await tool.execute(
            TextCleanInput(
                text="  <p>Hello123</p>  ",
                operations=[
                    CleanOperation.HTML,
                    CleanOperation.DIGITS,
                    CleanOperation.WHITESPACE,
                ],
            )
        )
        assert result.success is True
        assert result.result == "Hello"


class TestTextCaseTool:
    """Tests for TextCaseTool."""

    @pytest.fixture
    def tool(self):
        return TextCaseTool()

    @pytest.mark.asyncio
    async def test_case_lower(self, tool):
        """Test lowercase conversion."""
        result = await tool.execute(
            TextCaseInput(text="Hello World", operation=CaseOperation.LOWER)
        )
        assert result.success is True
        assert result.result == "hello world"

    @pytest.mark.asyncio
    async def test_case_upper(self, tool):
        """Test uppercase conversion."""
        result = await tool.execute(
            TextCaseInput(text="Hello World", operation=CaseOperation.UPPER)
        )
        assert result.success is True
        assert result.result == "HELLO WORLD"

    @pytest.mark.asyncio
    async def test_case_title(self, tool):
        """Test title case conversion."""
        result = await tool.execute(
            TextCaseInput(text="hello world", operation=CaseOperation.TITLE)
        )
        assert result.success is True
        assert result.result == "Hello World"

    @pytest.mark.asyncio
    async def test_case_camel(self, tool):
        """Test camelCase conversion."""
        result = await tool.execute(
            TextCaseInput(text="hello world test", operation=CaseOperation.CAMEL)
        )
        assert result.success is True
        assert result.result == "helloWorldTest"

    @pytest.mark.asyncio
    async def test_case_snake(self, tool):
        """Test snake_case conversion."""
        result = await tool.execute(
            TextCaseInput(text="HelloWorld", operation=CaseOperation.SNAKE)
        )
        assert result.success is True
        assert result.result == "hello_world"

    @pytest.mark.asyncio
    async def test_case_kebab(self, tool):
        """Test kebab-case conversion."""
        result = await tool.execute(
            TextCaseInput(text="HelloWorld", operation=CaseOperation.KEBAB)
        )
        assert result.success is True
        assert result.result == "hello-world"


class TestTextExtractTool:
    """Tests for TextExtractTool."""

    @pytest.fixture
    def tool(self):
        return TextExtractTool()

    @pytest.mark.asyncio
    async def test_extract_emails(self, tool):
        """Test email extraction."""
        result = await tool.execute(
            TextExtractInput(
                text="Contact us at info@example.com or support@test.org",
                extract_type=ExtractType.EMAILS,
            )
        )
        assert result.success is True
        assert result.count == 2
        assert "info@example.com" in result.items

    @pytest.mark.asyncio
    async def test_extract_urls(self, tool):
        """Test URL extraction."""
        result = await tool.execute(
            TextExtractInput(
                text="Visit https://example.com or www.test.org",
                extract_type=ExtractType.URLS,
            )
        )
        assert result.success is True
        assert result.count == 2

    @pytest.mark.asyncio
    async def test_extract_numbers(self, tool):
        """Test number extraction."""
        result = await tool.execute(
            TextExtractInput(
                text="The price is 19.99 and quantity is 5",
                extract_type=ExtractType.NUMBERS,
            )
        )
        assert result.success is True
        assert "19.99" in result.items
        assert "5" in result.items

    @pytest.mark.asyncio
    async def test_extract_hashtags(self, tool):
        """Test hashtag extraction."""
        result = await tool.execute(
            TextExtractInput(
                text="Check out #python and #coding",
                extract_type=ExtractType.HASHTAGS,
            )
        )
        assert result.success is True
        assert result.count == 2
        assert "#python" in result.items

    @pytest.mark.asyncio
    async def test_extract_words(self, tool):
        """Test word extraction."""
        result = await tool.execute(
            TextExtractInput(
                text="Hello, world! How are you?",
                extract_type=ExtractType.WORDS,
            )
        )
        assert result.success is True
        assert "Hello" in result.items
        assert "world" in result.items

    @pytest.mark.asyncio
    async def test_extract_unique(self, tool):
        """Test unique extraction."""
        result = await tool.execute(
            TextExtractInput(
                text="cat dog cat bird dog",
                extract_type=ExtractType.WORDS,
                unique=True,
            )
        )
        assert result.success is True
        assert result.count == 3

    @pytest.mark.asyncio
    async def test_extract_with_limit(self, tool):
        """Test extraction with limit."""
        result = await tool.execute(
            TextExtractInput(
                text="one two three four five",
                extract_type=ExtractType.WORDS,
                limit=3,
            )
        )
        assert result.success is True
        assert result.count == 3


class TestTextTruncateTool:
    """Tests for TextTruncateTool."""

    @pytest.fixture
    def tool(self):
        return TextTruncateTool()

    @pytest.mark.asyncio
    async def test_truncate_no_change(self, tool):
        """Test truncation when text is short enough."""
        result = await tool.execute(
            TextTruncateInput(text="hello", max_length=10)
        )
        assert result.success is True
        assert result.result == "hello"
        assert result.truncated is False

    @pytest.mark.asyncio
    async def test_truncate_end(self, tool):
        """Test truncation at end."""
        result = await tool.execute(
            TextTruncateInput(
                text="hello world test",
                max_length=10,
                position=TruncatePosition.END,
            )
        )
        assert result.success is True
        assert result.result == "hello w..."
        assert result.truncated is True
        assert len(result.result) == 10

    @pytest.mark.asyncio
    async def test_truncate_start(self, tool):
        """Test truncation at start."""
        result = await tool.execute(
            TextTruncateInput(
                text="hello world test",
                max_length=10,
                position=TruncatePosition.START,
            )
        )
        assert result.success is True
        assert result.result.startswith("...")
        assert result.truncated is True

    @pytest.mark.asyncio
    async def test_truncate_middle(self, tool):
        """Test truncation in middle."""
        result = await tool.execute(
            TextTruncateInput(
                text="hello world test",
                max_length=12,
                position=TruncatePosition.MIDDLE,
            )
        )
        assert result.success is True
        assert "..." in result.result
        assert result.truncated is True

    @pytest.mark.asyncio
    async def test_truncate_custom_ellipsis(self, tool):
        """Test truncation with custom ellipsis."""
        result = await tool.execute(
            TextTruncateInput(
                text="hello world",
                max_length=8,
                ellipsis="->",
            )
        )
        assert result.success is True
        assert result.result.endswith("->")


class TestCreateTextTools:
    """Tests for create_text_tools factory."""

    def test_creates_all_tools(self):
        """Test that factory creates all tools."""
        tools = create_text_tools()
        assert len(tools) == 8

        tool_ids = {t.metadata.id for t in tools}
        expected = {
            "text_regex_match",
            "text_regex_replace",
            "text_split",
            "text_join",
            "text_clean",
            "text_case",
            "text_extract",
            "text_truncate",
        }
        assert tool_ids == expected

    def test_all_tools_have_data_category(self):
        """Test all tools are in data category."""
        tools = create_text_tools()
        for tool in tools:
            assert tool.metadata.category == "data"
