from uuid import UUID
from typing import List, Dict, Any, Optional
from pydantic import Field
from .types import SemanticBase, SymbolBase

class EntitySemantic(SemanticBase):
    """Semantic information for an entity."""
    pass

class EntitySymbol(SymbolBase):
    """Symbol representing an entity in the knowledge graph."""
    entity_type: str
    semantics: List[EntitySemantic] = Field(default_factory=list)

    def to_neo4j(self) -> Dict[str, Any]:
        """Convert to Neo4j node properties."""
        return {
            'symbol_id': str(self.symbol_id),
            'name': self.name,
            'entity_type': self.entity_type,
            'descriptions': self.descriptions,
            'properties': self.properties,
            'labels': self.labels,
            'document_id': str(self.document_id) if self.document_id else None
        }
