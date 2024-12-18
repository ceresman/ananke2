from uuid import UUID
from typing import Dict, Any
from pydantic import Field

from .types import SemanticBase, SymbolBase

class TripleSymbol(SymbolBase):
    """Represents a triple in the knowledge graph."""
    subject_id: UUID
    predicate_id: UUID
    object_id: UUID

    def to_neo4j(self) -> Dict[str, Any]:
        """Special handling for Neo4j graph database."""
        return {
            'triple_id': str(self.symbol_id),
            'subject_id': str(self.subject_id),
            'predicate_id': str(self.predicate_id),
            'object_id': str(self.object_id),
            'properties': self.properties,
            'labels': self.labels
        }

class TripleSemantic(SemanticBase):
    """Represents a semantic triple in the knowledge graph."""
    subject_id: UUID
    predicate_id: UUID
    object_id: UUID

    def to_chroma(self) -> Dict[str, Any]:
        """Special handling for Chroma vector database."""
        return {
            'triple_id': str(self.semantic_id),
            'subject_id': str(self.subject_id),
            'predicate_id': str(self.predicate_id),
            'object_id': str(self.object_id),
            'semantic_type': self.semantic_type,
            'semantic_value': self.semantic_value,
            'confidence': self.confidence
        }
