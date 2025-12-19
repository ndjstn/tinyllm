"""Calculator tool for safe math expression evaluation."""

import ast
import math
import operator
from typing import Optional

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata


class CalculatorInput(BaseModel):
    """Input for calculator tool."""

    expression: str = Field(
        description="Mathematical expression to evaluate",
        examples=["2 + 2", "sqrt(16)", "sin(pi/2)", "2**10"],
        max_length=1000,
    )


class CalculatorOutput(BaseModel):
    """Output from calculator tool."""

    success: bool
    value: Optional[float] = None
    formatted: Optional[str] = None  # Human-readable result
    error: Optional[str] = None


class CalculatorTool(BaseTool[CalculatorInput, CalculatorOutput]):
    """Safely evaluates mathematical expressions."""

    metadata = ToolMetadata(
        id="calculator",
        name="Calculator",
        description="Evaluates mathematical expressions safely. Supports basic arithmetic, "
        "trigonometry (sin, cos, tan), logarithms (log, log10), and common math functions.",
        category="computation",
        sandbox_required=False,
    )
    input_type = CalculatorInput
    output_type = CalculatorOutput

    # Safe operations whitelist
    SAFE_NAMES: dict[str, any] = {
        # Basic operators (accessed via AST)
        # Math functions
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "abs": abs,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
        "pow": pow,
        # Constants
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
    }

    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    async def execute(self, input: CalculatorInput) -> CalculatorOutput:
        """Evaluate mathematical expression safely."""
        try:
            result = self._safe_eval(input.expression)
            return CalculatorOutput(
                success=True,
                value=float(result),
                formatted=self._format_result(result),
            )
        except ZeroDivisionError:
            return CalculatorOutput(success=False, error="Division by zero")
        except ValueError as e:
            return CalculatorOutput(success=False, error=f"Math error: {e}")
        except SyntaxError as e:
            return CalculatorOutput(success=False, error=f"Invalid syntax: {e}")
        except Exception as e:
            return CalculatorOutput(success=False, error=f"Evaluation error: {e}")

    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate expression using AST parsing."""
        # Parse expression to AST
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise SyntaxError(f"Invalid expression: {e}")

        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> float:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Invalid constant: {node.value}")

        elif isinstance(node, ast.Name):
            if node.id in self.SAFE_NAMES:
                value = self.SAFE_NAMES[node.id]
                if callable(value):
                    raise ValueError(f"{node.id} is a function, needs arguments")
                return float(value)
            raise ValueError(f"Unknown name: {node.id}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.SAFE_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.SAFE_OPERATORS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.SAFE_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return self.SAFE_OPERATORS[op_type](operand)

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            func_name = node.func.id
            if func_name not in self.SAFE_NAMES:
                raise ValueError(f"Unknown function: {func_name}")
            func = self.SAFE_NAMES[func_name]
            if not callable(func):
                raise ValueError(f"{func_name} is not a function")
            args = [self._eval_node(arg) for arg in node.args]
            return func(*args)

        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    def _format_result(self, result: float) -> str:
        """Format result for human readability."""
        if result == float("inf"):
            return "∞"
        if result == float("-inf"):
            return "-∞"
        if math.isnan(result):
            return "NaN"
        if result == int(result) and abs(result) < 1e15:
            return str(int(result))
        return f"{result:.10g}"
