from uuid import UUID
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from .types import SemanticBase, SymbolBase, StructuredDataBase

class Entity(BaseModel):
    name: str
    type: str
    description: str

    def dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description
        }

class Relationship(BaseModel):
    source: str
    target: str
    relationship: str
    relationship_strength: int

    def dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
            "relationship_strength": self.relationship_strength
        }

class EntitySemantic(SemanticBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = ""
    vector_representation: List[float] = Field(default_factory=list)
    semantic_type: str = Field(default="DEFINITION")
    semantic_value: str = Field(default="")

    def to_dict(self) -> Dict[str, Any]:
        """Convert semantic entity to dictionary."""
        return {
            "id": str(self.semantic_id),
            "name": self.name,
            "semantic_type": self.semantic_type,
            "semantic_value": self.semantic_value,
            "vector_representation": self.vector_representation
        }

class EntitySymbol(SymbolBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_type: str
    semantics: List[EntitySemantic] = Field(default_factory=list)
    properties: List[StructuredDataBase] = Field(default_factory=list)
    labels: List[StructuredDataBase] = Field(default_factory=list)

    def to_neo4j(self) -> Dict[str, Any]:
        return {
            'symbol_id': str(self.symbol_id),
            'name': self.name,
            'entity_type': self.entity_type,
            'descriptions': self.descriptions,
            'properties': [p.model_dump() for p in self.properties],
            'labels': [l.model_dump() for l in self.labels],
            'document_id': str(self.document_id) if self.document_id else None
        }
