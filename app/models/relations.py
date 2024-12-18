from uuid import UUID
from typing import List, Dict, Any
import numpy as np
from .base import BaseObject

class RelationSemantic(BaseObject):
    """Represents a semantic relation in the knowledge graph."""
    relation_id: UUID
    name: str
    semantic: List[float]  # Using List[float] instead of np.ndarray for serialization
    hit_count: int

    def to_chroma(self) -> Dict[str, Any]:
        """Special handling for Chroma vector database."""
        data = self.model_dump()
        data['vector'] = self.semantic
        return data

class RelationSymbol(BaseObject):
    """Represents a symbol representing a relation in the knowledge graph."""
    relation_id: UUID
    name: str
    description: str
    semantics: RelationSemantic
    hit_count: int
    relationship_strength: int

    def to_neo4j(self) -> Dict[str, Any]:
        """Special handling for Neo4j graph database."""
        return {
            'relation_id': str(self.relation_id),
            'name': self.name,
            'description': self.description,
            'hit_count': self.hit_count
        }
