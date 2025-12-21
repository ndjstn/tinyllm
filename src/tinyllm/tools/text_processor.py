"""Text processing tools for regex operations, string manipulation, and text extraction."""

import re
import unicodedata
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata


class TextOperation(str, Enum):
    """Supported text operations."""

    REGEX_MATCH = "regex_match"
    REGEX_REPLACE = "regex_replace"
    REGEX_SPLIT = "regex_split"
    EXTRACT = "extract"
    SPLIT = "split"
    JOIN = "join"
    CLEAN = "clean"
    CASE = "case"
    TRIM = "trim"
    PAD = "pad"
    TRUNCATE = "truncate"


class CleanOperation(str, Enum):
    """Text cleaning operations."""

    WHITESPACE = "whitespace"  # Normalize whitespace
    UNICODE = "unicode"  # Normalize Unicode
    HTML = "html"  # Strip HTML tags
    PUNCTUATION = "punctuation"  # Remove punctuation
    DIGITS = "digits"  # Remove digits
    NON_ASCII = "non_ascii"  # Remove non-ASCII
    LOWERCASE = "lowercase"  # Convert to lowercase


class CaseOperation(str, Enum):
    """Case transformation operations."""

    LOWER = "lower"
    UPPER = "upper"
    TITLE = "title"
    CAPITALIZE = "capitalize"
    SWAP = "swap"
    CAMEL = "camel"
    SNAKE = "snake"
    KEBAB = "kebab"


class TextConfig(ToolConfig):
    """Configuration for text processing tools."""

    max_input_length: int = Field(default=10 * 1024 * 1024, ge=1024)  # 10MB
    regex_timeout_ms: int = Field(default=5000, ge=100, le=60000)
    max_matches: int = Field(default=10000, ge=1, le=1000000)


# --- Regex Match Tool ---


class RegexMatchInput(BaseModel):
    """Input for regex matching."""

    text: str = Field(
        description="Text to search in",
        max_length=10 * 1024 * 1024,
    )
    pattern: str = Field(
        description="Regular expression pattern",
        max_length=10000,
    )
    flags: List[str] = Field(
        default_factory=list,
        description="Regex flags: i (ignorecase), m (multiline), s (dotall), x (verbose)",
    )
    find_all: bool = Field(
        default=True,
        description="Find all matches (True) or just first (False)",
    )
    include_groups: bool = Field(
        default=True,
        description="Include capture groups in results",
    )


class RegexMatch(BaseModel):
    """A single regex match."""

    match: str = Field(description="The matched text")
    start: int = Field(description="Start position")
    end: int = Field(description="End position")
    groups: List[str] = Field(default_factory=list, description="Capture groups")
    group_dict: Dict[str, str] = Field(default_factory=dict, description="Named groups")


class RegexMatchOutput(BaseModel):
    """Output from regex matching."""

    success: bool
    matches: List[RegexMatch] = Field(default_factory=list)
    count: int = 0
    error: Optional[str] = None


class RegexMatchTool(BaseTool[RegexMatchInput, RegexMatchOutput]):
    """Match text using regular expressions."""

    metadata = ToolMetadata(
        id="text_regex_match",
        name="Regex Match",
        description="Find matches in text using regular expressions. "
        "Returns match positions and capture groups.",
        category="data",
        sandbox_required=False,
    )
    input_type = RegexMatchInput
    output_type = RegexMatchOutput

    async def execute(self, input: RegexMatchInput) -> RegexMatchOutput:
        """Execute regex match."""
        try:
            flags = self._parse_flags(input.flags)
            pattern = re.compile(input.pattern, flags)

            matches = []
            if input.find_all:
                for m in pattern.finditer(input.text):
                    match_obj = RegexMatch(
                        match=m.group(0),
                        start=m.start(),
                        end=m.end(),
                        groups=list(m.groups()) if input.include_groups else [],
                        group_dict=m.groupdict() if input.include_groups else {},
                    )
                    matches.append(match_obj)
            else:
                m = pattern.search(input.text)
                if m:
                    matches.append(
                        RegexMatch(
                            match=m.group(0),
                            start=m.start(),
                            end=m.end(),
                            groups=list(m.groups()) if input.include_groups else [],
                            group_dict=m.groupdict() if input.include_groups else {},
                        )
                    )

            return RegexMatchOutput(
                success=True,
                matches=matches,
                count=len(matches),
            )

        except re.error as e:
            return RegexMatchOutput(success=False, error=f"Invalid regex: {e}")
        except Exception as e:
            return RegexMatchOutput(success=False, error=str(e))

    def _parse_flags(self, flags: List[str]) -> int:
        """Parse regex flags."""
        result = 0
        flag_map = {
            "i": re.IGNORECASE,
            "m": re.MULTILINE,
            "s": re.DOTALL,
            "x": re.VERBOSE,
        }
        for f in flags:
            if f.lower() in flag_map:
                result |= flag_map[f.lower()]
        return result


# --- Regex Replace Tool ---


class RegexReplaceInput(BaseModel):
    """Input for regex replacement."""

    text: str = Field(
        description="Text to search in",
        max_length=10 * 1024 * 1024,
    )
    pattern: str = Field(
        description="Regular expression pattern",
        max_length=10000,
    )
    replacement: str = Field(
        description="Replacement string (can use \\1, \\g<name> for groups)",
        max_length=100000,
    )
    flags: List[str] = Field(
        default_factory=list,
        description="Regex flags",
    )
    count: int = Field(
        default=0,
        description="Max replacements (0 for all)",
        ge=0,
    )


class RegexReplaceOutput(BaseModel):
    """Output from regex replacement."""

    success: bool
    result: Optional[str] = None
    replacements_made: int = 0
    error: Optional[str] = None


class RegexReplaceTool(BaseTool[RegexReplaceInput, RegexReplaceOutput]):
    """Replace text using regular expressions."""

    metadata = ToolMetadata(
        id="text_regex_replace",
        name="Regex Replace",
        description="Replace text using regular expressions. "
        "Supports backreferences and named groups.",
        category="data",
        sandbox_required=False,
    )
    input_type = RegexReplaceInput
    output_type = RegexReplaceOutput

    async def execute(self, input: RegexReplaceInput) -> RegexReplaceOutput:
        """Execute regex replacement."""
        try:
            flags = self._parse_flags(input.flags)
            pattern = re.compile(input.pattern, flags)

            # Count replacements
            count = 0

            def replacement_counter(m):
                nonlocal count
                count += 1
                return m.expand(input.replacement)

            if input.count > 0:
                result = pattern.sub(replacement_counter, input.text, count=input.count)
            else:
                result = pattern.sub(replacement_counter, input.text)

            return RegexReplaceOutput(
                success=True,
                result=result,
                replacements_made=count,
            )

        except re.error as e:
            return RegexReplaceOutput(success=False, error=f"Invalid regex: {e}")
        except Exception as e:
            return RegexReplaceOutput(success=False, error=str(e))

    def _parse_flags(self, flags: List[str]) -> int:
        """Parse regex flags."""
        result = 0
        flag_map = {
            "i": re.IGNORECASE,
            "m": re.MULTILINE,
            "s": re.DOTALL,
            "x": re.VERBOSE,
        }
        for f in flags:
            if f.lower() in flag_map:
                result |= flag_map[f.lower()]
        return result


# --- Text Split Tool ---


class TextSplitInput(BaseModel):
    """Input for text splitting."""

    text: str = Field(
        description="Text to split",
        max_length=10 * 1024 * 1024,
    )
    separator: Optional[str] = Field(
        default=None,
        description="Separator string (None for whitespace)",
    )
    regex: bool = Field(
        default=False,
        description="Treat separator as regex",
    )
    max_splits: int = Field(
        default=0,
        description="Maximum number of splits (0 for unlimited)",
        ge=0,
    )
    keep_empty: bool = Field(
        default=False,
        description="Keep empty strings in result",
    )


class TextSplitOutput(BaseModel):
    """Output from text splitting."""

    success: bool
    parts: List[str] = Field(default_factory=list)
    count: int = 0
    error: Optional[str] = None


class TextSplitTool(BaseTool[TextSplitInput, TextSplitOutput]):
    """Split text by separator or regex."""

    metadata = ToolMetadata(
        id="text_split",
        name="Text Split",
        description="Split text by a separator string or regex pattern.",
        category="data",
        sandbox_required=False,
    )
    input_type = TextSplitInput
    output_type = TextSplitOutput

    async def execute(self, input: TextSplitInput) -> TextSplitOutput:
        """Execute text splitting."""
        try:
            if input.regex and input.separator:
                pattern = re.compile(input.separator)
                if input.max_splits > 0:
                    parts = pattern.split(input.text, maxsplit=input.max_splits)
                else:
                    parts = pattern.split(input.text)
            elif input.separator:
                if input.max_splits > 0:
                    parts = input.text.split(input.separator, maxsplit=input.max_splits)
                else:
                    parts = input.text.split(input.separator)
            else:
                # Split on whitespace
                if input.max_splits > 0:
                    parts = input.text.split(maxsplit=input.max_splits)
                else:
                    parts = input.text.split()

            if not input.keep_empty:
                parts = [p for p in parts if p]

            return TextSplitOutput(
                success=True,
                parts=parts,
                count=len(parts),
            )

        except re.error as e:
            return TextSplitOutput(success=False, error=f"Invalid regex: {e}")
        except Exception as e:
            return TextSplitOutput(success=False, error=str(e))


# --- Text Join Tool ---


class TextJoinInput(BaseModel):
    """Input for text joining."""

    parts: List[str] = Field(
        description="Text parts to join",
    )
    separator: str = Field(
        default="",
        description="Separator to use between parts",
    )
    prefix: str = Field(
        default="",
        description="Prefix to add before result",
    )
    suffix: str = Field(
        default="",
        description="Suffix to add after result",
    )


class TextJoinOutput(BaseModel):
    """Output from text joining."""

    success: bool
    result: Optional[str] = None
    length: int = 0
    error: Optional[str] = None


class TextJoinTool(BaseTool[TextJoinInput, TextJoinOutput]):
    """Join text parts with a separator."""

    metadata = ToolMetadata(
        id="text_join",
        name="Text Join",
        description="Join text parts with a separator, optionally adding prefix and suffix.",
        category="data",
        sandbox_required=False,
    )
    input_type = TextJoinInput
    output_type = TextJoinOutput

    async def execute(self, input: TextJoinInput) -> TextJoinOutput:
        """Execute text joining."""
        try:
            result = input.prefix + input.separator.join(input.parts) + input.suffix
            return TextJoinOutput(
                success=True,
                result=result,
                length=len(result),
            )
        except Exception as e:
            return TextJoinOutput(success=False, error=str(e))


# --- Text Clean Tool ---


class TextCleanInput(BaseModel):
    """Input for text cleaning."""

    text: str = Field(
        description="Text to clean",
        max_length=10 * 1024 * 1024,
    )
    operations: List[CleanOperation] = Field(
        description="Cleaning operations to apply in order",
    )


class TextCleanOutput(BaseModel):
    """Output from text cleaning."""

    success: bool
    result: Optional[str] = None
    original_length: int = 0
    cleaned_length: int = 0
    error: Optional[str] = None


class TextCleanTool(BaseTool[TextCleanInput, TextCleanOutput]):
    """Clean and normalize text."""

    metadata = ToolMetadata(
        id="text_clean",
        name="Text Clean",
        description="Clean and normalize text with various operations: "
        "whitespace normalization, Unicode normalization, HTML stripping, etc.",
        category="data",
        sandbox_required=False,
    )
    input_type = TextCleanInput
    output_type = TextCleanOutput

    async def execute(self, input: TextCleanInput) -> TextCleanOutput:
        """Execute text cleaning."""
        try:
            result = input.text
            original_length = len(result)

            for op in input.operations:
                if op == CleanOperation.WHITESPACE:
                    # Normalize whitespace
                    result = " ".join(result.split())
                elif op == CleanOperation.UNICODE:
                    # Normalize Unicode to NFC
                    result = unicodedata.normalize("NFC", result)
                elif op == CleanOperation.HTML:
                    # Strip HTML tags
                    result = re.sub(r"<[^>]+>", "", result)
                elif op == CleanOperation.PUNCTUATION:
                    # Remove punctuation
                    result = re.sub(r"[^\w\s]", "", result)
                elif op == CleanOperation.DIGITS:
                    # Remove digits
                    result = re.sub(r"\d", "", result)
                elif op == CleanOperation.NON_ASCII:
                    # Remove non-ASCII
                    result = result.encode("ascii", "ignore").decode("ascii")
                elif op == CleanOperation.LOWERCASE:
                    result = result.lower()

            return TextCleanOutput(
                success=True,
                result=result,
                original_length=original_length,
                cleaned_length=len(result),
            )

        except Exception as e:
            return TextCleanOutput(success=False, error=str(e))


# --- Text Case Tool ---


class TextCaseInput(BaseModel):
    """Input for case transformation."""

    text: str = Field(
        description="Text to transform",
        max_length=10 * 1024 * 1024,
    )
    operation: CaseOperation = Field(
        description="Case transformation to apply",
    )


class TextCaseOutput(BaseModel):
    """Output from case transformation."""

    success: bool
    result: Optional[str] = None
    error: Optional[str] = None


class TextCaseTool(BaseTool[TextCaseInput, TextCaseOutput]):
    """Transform text case."""

    metadata = ToolMetadata(
        id="text_case",
        name="Text Case",
        description="Transform text case: lower, upper, title, camelCase, snake_case, kebab-case.",
        category="data",
        sandbox_required=False,
    )
    input_type = TextCaseInput
    output_type = TextCaseOutput

    async def execute(self, input: TextCaseInput) -> TextCaseOutput:
        """Execute case transformation."""
        try:
            text = input.text
            op = input.operation

            if op == CaseOperation.LOWER:
                result = text.lower()
            elif op == CaseOperation.UPPER:
                result = text.upper()
            elif op == CaseOperation.TITLE:
                result = text.title()
            elif op == CaseOperation.CAPITALIZE:
                result = text.capitalize()
            elif op == CaseOperation.SWAP:
                result = text.swapcase()
            elif op == CaseOperation.CAMEL:
                # Convert to camelCase
                words = re.split(r"[\s_\-]+", text)
                result = words[0].lower() + "".join(w.capitalize() for w in words[1:])
            elif op == CaseOperation.SNAKE:
                # Convert to snake_case
                result = re.sub(r"(?<!^)(?=[A-Z])", "_", text)
                result = re.sub(r"[\s\-]+", "_", result).lower()
            elif op == CaseOperation.KEBAB:
                # Convert to kebab-case
                result = re.sub(r"(?<!^)(?=[A-Z])", "-", text)
                result = re.sub(r"[\s_]+", "-", result).lower()
            else:
                result = text

            return TextCaseOutput(success=True, result=result)

        except Exception as e:
            return TextCaseOutput(success=False, error=str(e))


# --- Text Extract Tool ---


class ExtractType(str, Enum):
    """Types of data to extract."""

    EMAILS = "emails"
    URLS = "urls"
    PHONES = "phones"
    NUMBERS = "numbers"
    DATES = "dates"
    IPS = "ips"
    HASHTAGS = "hashtags"
    MENTIONS = "mentions"
    WORDS = "words"
    SENTENCES = "sentences"
    LINES = "lines"


class TextExtractInput(BaseModel):
    """Input for text extraction."""

    text: str = Field(
        description="Text to extract from",
        max_length=10 * 1024 * 1024,
    )
    extract_type: ExtractType = Field(
        description="Type of data to extract",
    )
    unique: bool = Field(
        default=False,
        description="Return only unique values",
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of items to return",
        ge=1,
    )


class TextExtractOutput(BaseModel):
    """Output from text extraction."""

    success: bool
    items: List[str] = Field(default_factory=list)
    count: int = 0
    error: Optional[str] = None


class TextExtractTool(BaseTool[TextExtractInput, TextExtractOutput]):
    """Extract structured data from text."""

    metadata = ToolMetadata(
        id="text_extract",
        name="Text Extract",
        description="Extract structured data from text: emails, URLs, phone numbers, "
        "dates, IP addresses, hashtags, mentions, words, sentences, lines.",
        category="data",
        sandbox_required=False,
    )
    input_type = TextExtractInput
    output_type = TextExtractOutput

    # Regex patterns for extraction
    PATTERNS = {
        ExtractType.EMAILS: r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        ExtractType.URLS: r"https?://[^\s<>\"]+|www\.[^\s<>\"]+",
        ExtractType.PHONES: r"[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}",
        ExtractType.NUMBERS: r"-?\d+\.?\d*",
        ExtractType.DATES: r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
        ExtractType.IPS: r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        ExtractType.HASHTAGS: r"#[a-zA-Z0-9_]+",
        ExtractType.MENTIONS: r"@[a-zA-Z0-9_]+",
    }

    async def execute(self, input: TextExtractInput) -> TextExtractOutput:
        """Execute text extraction."""
        try:
            items = []

            if input.extract_type == ExtractType.WORDS:
                items = re.findall(r"\b\w+\b", input.text)
            elif input.extract_type == ExtractType.SENTENCES:
                items = re.split(r"(?<=[.!?])\s+", input.text)
                items = [s.strip() for s in items if s.strip()]
            elif input.extract_type == ExtractType.LINES:
                items = input.text.splitlines()
                items = [line for line in items if line.strip()]
            else:
                pattern = self.PATTERNS.get(input.extract_type)
                if pattern:
                    items = re.findall(pattern, input.text)

            # Remove duplicates if requested
            if input.unique:
                seen = set()
                unique_items = []
                for item in items:
                    if item not in seen:
                        seen.add(item)
                        unique_items.append(item)
                items = unique_items

            # Apply limit
            if input.limit:
                items = items[: input.limit]

            return TextExtractOutput(
                success=True,
                items=items,
                count=len(items),
            )

        except Exception as e:
            return TextExtractOutput(success=False, error=str(e))


# --- Text Truncate Tool ---


class TruncatePosition(str, Enum):
    """Where to truncate."""

    END = "end"
    START = "start"
    MIDDLE = "middle"


class TextTruncateInput(BaseModel):
    """Input for text truncation."""

    text: str = Field(
        description="Text to truncate",
        max_length=10 * 1024 * 1024,
    )
    max_length: int = Field(
        description="Maximum length",
        ge=1,
    )
    position: TruncatePosition = Field(
        default=TruncatePosition.END,
        description="Where to truncate",
    )
    ellipsis: str = Field(
        default="...",
        description="String to use as ellipsis",
    )
    word_boundary: bool = Field(
        default=False,
        description="Truncate at word boundaries",
    )


class TextTruncateOutput(BaseModel):
    """Output from text truncation."""

    success: bool
    result: Optional[str] = None
    truncated: bool = False
    original_length: int = 0
    error: Optional[str] = None


class TextTruncateTool(BaseTool[TextTruncateInput, TextTruncateOutput]):
    """Truncate text to a maximum length."""

    metadata = ToolMetadata(
        id="text_truncate",
        name="Text Truncate",
        description="Truncate text to a maximum length with customizable ellipsis and position.",
        category="data",
        sandbox_required=False,
    )
    input_type = TextTruncateInput
    output_type = TextTruncateOutput

    async def execute(self, input: TextTruncateInput) -> TextTruncateOutput:
        """Execute text truncation."""
        try:
            text = input.text
            original_length = len(text)

            if len(text) <= input.max_length:
                return TextTruncateOutput(
                    success=True,
                    result=text,
                    truncated=False,
                    original_length=original_length,
                )

            ellipsis_len = len(input.ellipsis)
            target_len = input.max_length - ellipsis_len

            if target_len <= 0:
                result = input.ellipsis[: input.max_length]
            elif input.position == TruncatePosition.END:
                result = text[:target_len]
                if input.word_boundary:
                    # Find last word boundary
                    last_space = result.rfind(" ")
                    if last_space > target_len // 2:
                        result = result[:last_space]
                result = result + input.ellipsis
            elif input.position == TruncatePosition.START:
                result = text[-target_len:]
                if input.word_boundary:
                    first_space = result.find(" ")
                    if first_space > 0 and first_space < target_len // 2:
                        result = result[first_space + 1 :]
                result = input.ellipsis + result
            else:  # MIDDLE
                half = target_len // 2
                start = text[:half]
                end = text[-(target_len - half) :]
                if input.word_boundary:
                    last_space = start.rfind(" ")
                    if last_space > half // 2:
                        start = start[:last_space]
                    first_space = end.find(" ")
                    if first_space > 0 and first_space < (target_len - half) // 2:
                        end = end[first_space + 1 :]
                result = start + input.ellipsis + end

            return TextTruncateOutput(
                success=True,
                result=result,
                truncated=True,
                original_length=original_length,
            )

        except Exception as e:
            return TextTruncateOutput(success=False, error=str(e))


# --- Factory Function ---


def create_text_tools(config: TextConfig | None = None) -> List[BaseTool]:
    """Create all text processing tools with optional configuration."""
    cfg = config or TextConfig()
    return [
        RegexMatchTool(cfg),
        RegexReplaceTool(cfg),
        TextSplitTool(cfg),
        TextJoinTool(cfg),
        TextCleanTool(cfg),
        TextCaseTool(cfg),
        TextExtractTool(cfg),
        TextTruncateTool(cfg),
    ]
