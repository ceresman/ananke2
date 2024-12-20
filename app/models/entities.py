from uuid import UUID
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from .types import SemanticBase, SymbolBase, StructuredDataBase

class Entity(BaseModel):
    """Base entity model for knowledge graph nodes.

    Represents a fundamental entity in the knowledge graph with basic
    attributes like name, type, and description. Used primarily for
    initial entity extraction from documents.

    Attributes:
        name (str): The name or identifier of the entity
        type (str): The classification/category of the entity
        description (str): Detailed description of the entity

    Example:
        ```python
        entity = Entity(
            name="Neural Network",
            type="CONCEPT",
            description="A computing system inspired by biological neural networks"
        )
        data = entity.dict()
        ```
    """
    name: str
    type: str
    description: str

    def dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary containing entity attributes
        """
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description
        }

class Relationship(BaseModel):
    """Represents a relationship between two entities in the knowledge graph.

    Defines directed relationships between entities, including the type of
    relationship and its strength. Used for building the knowledge graph's
    edge structure.

    Attributes:
        source (str): Name of the source entity
        target (str): Name of the target entity
        relationship (str): Type or description of the relationship
        relationship_strength (int): Strength of relationship (1-10)

    Example:
        ```python
        rel = Relationship(
            source="Neural Network",
            target="Deep Learning",
            relationship="IS_COMPONENT_OF",
            relationship_strength=8
        )
        ```
    """
    source: str
    target: str
    relationship: str
    relationship_strength: int

    def dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary containing relationship attributes
        """
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
            "relationship_strength": self.relationship_strength
        }

class EntitySemantic(SemanticBase):
    """Semantic representation of an entity in the knowledge graph.

    Extends SemanticBase to provide semantic vector representations and
    additional semantic metadata for entities. Used for semantic similarity
    search and embedding-based operations.

    Attributes:
        name (str): Entity name for semantic context
        vector_representation (List[float]): Vector embedding of the entity
        semantic_type (str): Type of semantic representation (default: "DEFINITION")
        semantic_value (str): Actual semantic content or value

    Example:
        ```python
        semantic = EntitySemantic(
            name="Neural Network",
            vector_representation=[0.1, 0.2, ...],
            semantic_type="DEFINITION",
            semantic_value="A computational model inspired by biological networks"
        )
        ```
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = ""
    vector_representation: List[float] = Field(default_factory=list)
    semantic_type: str = Field(default="DEFINITION")
    semantic_value: str = Field(default="")

    def to_dict(self) -> Dict[str, Any]:
        """Convert semantic entity to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary containing semantic entity attributes
                including ID, name, type, value, and vector representation
        """
        return {
            "id": str(self.semantic_id),
            "name": self.name,
            "semantic_type": self.semantic_type,
            "semantic_value": self.semantic_value,
            "vector_representation": self.vector_representation
        }

class EntitySymbol(SymbolBase):
    """Symbolic representation of an entity with associated metadata.

    Extends SymbolBase to provide a comprehensive entity representation
    including semantic information, properties, and labels. Used as the
    primary entity model in the knowledge graph.

    Attributes:
        entity_type (str): Classification of the entity
        semantics (List[EntitySemantic]): Associated semantic representations
        properties (List[StructuredDataBase]): Structured properties
        labels (List[StructuredDataBase]): Classification labels

    Example:
        ```python
        symbol = EntitySymbol(
            name="Neural Network",
            entity_type="ALGORITHM",
            semantics=[EntitySemantic(...)],
            properties=[StructuredDataBase(...)],
            labels=[StructuredDataBase(...)]
        )
        ```
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_type: str
    semantics: List[EntitySemantic] = Field(default_factory=list)
    properties: List[StructuredDataBase] = Field(default_factory=list)
    labels: List[StructuredDataBase] = Field(default_factory=list)

    def to_neo4j(self) -> Dict[str, Any]:
        """Convert entity symbol to Neo4j compatible format.

        Serializes the entity symbol and its associated data for
        storage in Neo4j graph database.

        Returns:
            Dict[str, Any]: Neo4j-compatible dictionary containing all
                entity attributes and relationships

        Raises:
            ValueError: If required attributes are missing or invalid
        """
        return {
            'symbol_id': str(self.symbol_id),
            'name': self.name,
            'entity_type': self.entity_type,
            'descriptions': self.descriptions,
            'properties': [p.model_dump() for p in self.properties],
            'labels': [l.model_dump() for l in self.labels],
            'document_id': str(self.document_id) if self.document_id else None
        }
