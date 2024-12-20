from uuid import UUID
from typing import Dict, Any
import sympy as sp
from .base import BaseObject

class LogicExpression(BaseObject):
    """Represents a logical expression in the knowledge graph.

    Handles logical expressions with multiple format representations,
    specifically supporting Lean4 theorem prover syntax and SymPy
    symbolic mathematics library format.

    Attributes:
        expression_id (UUID): Unique identifier for the expression
        expression_lean4 (str): Expression in Lean4 theorem prover syntax
        expression_sympy (str): Expression in SymPy symbolic format

    Example:
        ```python
        logic_expr = LogicExpression(
            expression_id=UUID('123e4567-e89b-12d3-a456-426614174000'),
            expression_lean4='∀ x y, x ∧ y → y ∧ x',
            expression_sympy='And(x, y) == And(y, x)'
        )
        mysql_data = logic_expr.to_mysql()
        ```

    Note:
        The expressions are stored as strings to maintain compatibility
        across different systems and databases while preserving the
        exact syntax of each format.
    """
    expression_id: UUID
    expression_lean4: str  # Using string representation for Lean4 expressions
    expression_sympy: str  # Using string representation for sympy expressions

    def to_mysql(self) -> Dict[str, Any]:
        """Convert logical expression to MySQL compatible format.

        Serializes the logical expression for storage in MySQL database,
        ensuring proper string representation of UUIDs.

        Returns:
            Dict[str, Any]: MySQL-compatible dictionary containing
                expression ID and both format representations

        Example:
            ```python
            data = logic_expr.to_mysql()
            # Returns: {
            #     'expression_id': '123e4567-e89b-12d3-a456-426614174000',
            #     'expression_lean4': '∀ x y, x ∧ y → y ∧ x',
            #     'expression_sympy': 'And(x, y) == And(y, x)'
            # }
            ```
        """
        return {
            'expression_id': str(self.expression_id),
            'expression_lean4': self.expression_lean4,
            'expression_sympy': self.expression_sympy
        }

class MathExpression(BaseObject):
    """Represents a mathematical expression in the knowledge graph.

    Handles mathematical expressions with multiple format representations,
    supporting LaTeX, SymPy symbolic mathematics library format, and
    Wolfram Language syntax.

    Attributes:
        expression_id (UUID): Unique identifier for the expression
        expression_latex (str): Expression in LaTeX format
        expression_sympy (str): Expression in SymPy symbolic format
        expression_wolfram (str): Expression in Wolfram Language syntax

    Example:
        ```python
        math_expr = MathExpression(
            expression_id=UUID('123e4567-e89b-12d3-a456-426614174000'),
            expression_latex='\\int_{0}^{\\infty} e^{-x^2} dx',
            expression_sympy='Integral(exp(-x**2), (x, 0, oo))',
            expression_wolfram='Integrate[Exp[-x^2], {x, 0, Infinity}]'
        )
        mysql_data = math_expr.to_mysql()
        ```

    Note:
        The expressions are stored as strings to maintain compatibility
        across different systems and databases while preserving the
        exact syntax of each format.
    """
    expression_id: UUID
    expression_latex: str
    expression_sympy: str  # Using string representation for sympy expressions
    expression_wolfram: str

    def to_mysql(self) -> Dict[str, Any]:
        """Convert mathematical expression to MySQL compatible format.

        Serializes the mathematical expression for storage in MySQL database,
        ensuring proper string representation of UUIDs.

        Returns:
            Dict[str, Any]: MySQL-compatible dictionary containing
                expression ID and all format representations

        Example:
            ```python
            data = math_expr.to_mysql()
            # Returns: {
            #     'expression_id': '123e4567-e89b-12d3-a456-426614174000',
            #     'expression_latex': '\\int_{0}^{\\infty} e^{-x^2} dx',
            #     'expression_sympy': 'Integral(exp(-x**2), (x, 0, oo))',
            #     'expression_wolfram': 'Integrate[Exp[-x^2], {x, 0, Infinity}]'
            # }
            ```
        """
        return {
            'expression_id': str(self.expression_id),
            'expression_latex': self.expression_latex,
            'expression_sympy': self.expression_sympy,
            'expression_wolfram': self.expression_wolfram
        }
