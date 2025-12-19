"""Tests for calculator tool."""

import pytest
from tinyllm.tools.calculator import CalculatorTool, CalculatorInput


class TestCalculator:
    """Tests for CalculatorTool."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return CalculatorTool()

    @pytest.mark.asyncio
    async def test_basic_addition(self, calculator):
        """Test basic addition."""
        result = await calculator.execute(CalculatorInput(expression="2 + 2"))
        assert result.success is True
        assert result.value == 4
        assert result.formatted == "4"

    @pytest.mark.asyncio
    async def test_basic_subtraction(self, calculator):
        """Test basic subtraction."""
        result = await calculator.execute(CalculatorInput(expression="10 - 3"))
        assert result.success is True
        assert result.value == 7

    @pytest.mark.asyncio
    async def test_basic_multiplication(self, calculator):
        """Test basic multiplication."""
        result = await calculator.execute(CalculatorInput(expression="6 * 7"))
        assert result.success is True
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_basic_division(self, calculator):
        """Test basic division."""
        result = await calculator.execute(CalculatorInput(expression="15 / 3"))
        assert result.success is True
        assert result.value == 5

    @pytest.mark.asyncio
    async def test_order_of_operations(self, calculator):
        """Test order of operations."""
        result = await calculator.execute(CalculatorInput(expression="2 + 3 * 4"))
        assert result.success is True
        assert result.value == 14  # 3*4=12, then +2

    @pytest.mark.asyncio
    async def test_parentheses(self, calculator):
        """Test parentheses."""
        result = await calculator.execute(CalculatorInput(expression="(2 + 3) * 4"))
        assert result.success is True
        assert result.value == 20

    @pytest.mark.asyncio
    async def test_power(self, calculator):
        """Test exponentiation."""
        result = await calculator.execute(CalculatorInput(expression="2 ** 10"))
        assert result.success is True
        assert result.value == 1024

    @pytest.mark.asyncio
    async def test_sqrt(self, calculator):
        """Test square root."""
        result = await calculator.execute(CalculatorInput(expression="sqrt(16)"))
        assert result.success is True
        assert result.value == 4

    @pytest.mark.asyncio
    async def test_pi_constant(self, calculator):
        """Test pi constant."""
        result = await calculator.execute(CalculatorInput(expression="pi"))
        assert result.success is True
        assert abs(result.value - 3.14159) < 0.001

    @pytest.mark.asyncio
    async def test_sin(self, calculator):
        """Test sine function."""
        result = await calculator.execute(CalculatorInput(expression="sin(0)"))
        assert result.success is True
        assert result.value == 0

    @pytest.mark.asyncio
    async def test_cos(self, calculator):
        """Test cosine function."""
        result = await calculator.execute(CalculatorInput(expression="cos(0)"))
        assert result.success is True
        assert result.value == 1

    @pytest.mark.asyncio
    async def test_log(self, calculator):
        """Test natural log."""
        result = await calculator.execute(CalculatorInput(expression="log(e)"))
        assert result.success is True
        assert abs(result.value - 1) < 0.0001

    @pytest.mark.asyncio
    async def test_complex_expression(self, calculator):
        """Test complex expression."""
        result = await calculator.execute(
            CalculatorInput(expression="sqrt(16) + 2 ** 3 - 10 / 2")
        )
        assert result.success is True
        assert result.value == 7  # 4 + 8 - 5 = 7

    @pytest.mark.asyncio
    async def test_division_by_zero(self, calculator):
        """Test division by zero."""
        result = await calculator.execute(CalculatorInput(expression="1 / 0"))
        assert result.success is False
        assert "zero" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_syntax(self, calculator):
        """Test invalid syntax."""
        result = await calculator.execute(CalculatorInput(expression="2 +"))
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_unknown_function(self, calculator):
        """Test unknown function."""
        result = await calculator.execute(CalculatorInput(expression="unknown(5)"))
        assert result.success is False
        assert "unknown" in result.error.lower()

    @pytest.mark.asyncio
    async def test_security_import_blocked(self, calculator):
        """Test that import is blocked."""
        result = await calculator.execute(CalculatorInput(expression="__import__('os')"))
        assert result.success is False

    @pytest.mark.asyncio
    async def test_negative_numbers(self, calculator):
        """Test negative numbers."""
        result = await calculator.execute(CalculatorInput(expression="-5 + 3"))
        assert result.success is True
        assert result.value == -2

    @pytest.mark.asyncio
    async def test_float_result(self, calculator):
        """Test floating point result."""
        result = await calculator.execute(CalculatorInput(expression="7 / 2"))
        assert result.success is True
        assert result.value == 3.5
        assert result.formatted == "3.5"

    @pytest.mark.asyncio
    async def test_large_number(self, calculator):
        """Test large number."""
        result = await calculator.execute(CalculatorInput(expression="2 ** 100"))
        assert result.success is True
        assert result.value == 2**100


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("1 + 1", 2),
        ("10 - 5", 5),
        ("3 * 4", 12),
        ("20 / 4", 5),
        ("2 ** 8", 256),
        ("sqrt(25)", 5),
        ("abs(-10)", 10),
        ("floor(3.7)", 3),
        ("ceil(3.2)", 4),
        ("round(3.5)", 4),
    ],
)
@pytest.mark.asyncio
async def test_parametrized_expressions(expr, expected):
    """Test various expressions."""
    calculator = CalculatorTool()
    result = await calculator.execute(CalculatorInput(expression=expr))
    assert result.success is True
    assert result.value == expected
