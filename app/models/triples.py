from uuid import UUID
from typing import Dict, Any
from .base import BaseObject
from .entities import EntitySymbol, EntitySemantic
from .relations import RelationSymbol, RelationSemantic

class TripleSymbol(BaseObject):
    """Represents a triple in the knowledge graph."""
    triple_id: UUID
    subject: EntitySymbol
    predicate: RelationSymbol
    obj: EntitySymbol

    def to_neo4j(self) -> Dict[str, Any]:
        """Special handling for Neo4j graph database."""
        return {
            'triple_id': str(self.triple_id),
            'subject': self.subject.to_neo4j(),
            'predicate': self.predicate.to_neo4j(),
            'object': self.obj.to_neo4j()
        }

class TripleSemantic(BaseObject):
    """Represents a semantic triple in the knowledge graph."""
    triple_id: UUID
    subject: EntitySemantic
    predicate: RelationSemantic
    obj: EntitySemantic

    def to_chroma(self) -> Dict[str, Any]:
        """Special handling for Chroma vector database."""
        return {
            'triple_id': str(self.triple_id),
            'subject_vector': self.subject.vector_representation,
            'predicate_vector': self.predicate.semantic,
            'object_vector': self.obj.vector_representation
        }
