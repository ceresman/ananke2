from uuid import UUID
from typing import List, Dict, Any, TYPE_CHECKING, ForwardRef
import numpy as np
from pydantic import BaseModel, ConfigDict
from .base import BaseObject

if TYPE_CHECKING:
    from .structured import StructuredData
else:
    StructuredData = ForwardRef('StructuredData')

class EntitySemantic(BaseObject):
    """Represents a semantic entity in the knowledge graph."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol_id: UUID
    name: str
    entity_type: str
    descriptions: List[str]
    semantics: List[EntitySemantic]
    properties: List[StructuredData]  # Fixed pluralization
    labels: List[StructuredData]  # Renamed for consistency

    def to_neo4j(self) -> Dict[str, Any]:
        """Special handling for Neo4j graph database."""
        return {
            'symbol_id': str(self.symbol_id),
            'name': self.name,
            'descriptions': self.descriptions,
            'labels': [label.data_value for label in self.labels]
        }
