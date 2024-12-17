from uuid import UUID
from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np
import sympy as sp
from pydantic import BaseModel
from .base import BaseObject

if TYPE_CHECKING:
    from .structured import StructuredData

class EntitySemantic(BaseObject):
    """Represents a semantic entity in the knowledge graph."""
    semantic_id: UUID
    name: str
    vector_representation: List[float]  # Using List[float] instead of np.ndarray for serialization

    def to_chroma(self) -> Dict[str, Any]:
        """Special handling for Chroma vector database."""
        data = self.model_dump()
        data['vector'] = self.vector_representation
        return data

class EntitySymbol(BaseObject):
    """Represents a symbolic entity in the knowledge graph."""
    symbol_id: UUID
    name: str
    descriptions: List[str]
    semantics: List[EntitySemantic]
    propertys: List['StructuredData']
    label: List['StructuredData']

    def to_neo4j(self) -> Dict[str, Any]:
        """Special handling for Neo4j graph database."""
        return {
            'symbol_id': str(self.symbol_id),
            'name': self.name,
            'descriptions': self.descriptions,
            'labels': [label.data_value for label in self.label]
        }
