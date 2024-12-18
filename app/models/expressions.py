from uuid import UUID
from typing import Dict, Any
import sympy as sp
from .base import BaseObject

class LogicExpression(BaseObject):
    """Represents a logical expression."""
    expression_id: UUID
    expression_lean4: str  # Using string representation for Lean4 expressions
    expression_sympy: str  # Using string representation for sympy expressions

    def to_mysql(self) -> Dict[str, Any]:
        """Special handling for MySQL database."""
        return {
            'expression_id': str(self.expression_id),
            'expression_lean4': self.expression_lean4,
            'expression_sympy': self.expression_sympy
        }

class MathExpression(BaseObject):
    """Represents a mathematical expression."""
    expression_id: UUID
    expression_latex: str
    expression_sympy: str  # Using string representation for sympy expressions
    expression_wolfram: str

    def to_mysql(self) -> Dict[str, Any]:
        """Special handling for MySQL database."""
        return {
            'expression_id': str(self.expression_id),
            'expression_latex': self.expression_latex,
            'expression_sympy': self.expression_sympy,
            'expression_wolfram': self.expression_wolfram
        }
