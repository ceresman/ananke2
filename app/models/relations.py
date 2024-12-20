from uuid import UUID
from typing import List, Dict, Any
import numpy as np
from .base import BaseObject

class RelationSemantic(BaseObject):
    """Represents a semantic relation in the knowledge graph.

    Handles semantic vector representations of relationships between entities,
    storing both the relationship name and its vector embedding. Tracks hit
    count for relationship frequency analysis.

    Attributes:
        relation_id (UUID): Unique identifier for the relation
        name (str): Name of the relationship
        semantic (List[float]): Vector embedding representation of the relation.
            Stored as List[float] instead of np.ndarray for serialization.
        hit_count (int): Number of times this relation has been observed

    Example:
        ```python
        relation = RelationSemantic(
            relation_id=UUID('123e4567-e89b-12d3-a456-426614174000'),
            name='AUTHORED_BY',
            semantic=[0.1, 0.2, 0.3, 0.4],  # Vector embedding
            hit_count=1
        )
        chroma_data = relation.to_chroma()
        ```

    Note:
        The semantic vector is stored as List[float] rather than np.ndarray
        to ensure proper serialization when storing in databases.
    """
    relation_id: UUID
    name: str
    semantic: List[float]  # Using List[float] instead of np.ndarray for serialization
    hit_count: int

    def to_chroma(self) -> Dict[str, Any]:
        """Convert semantic relation to Chroma vector database format.

        Prepares the relation data for storage in Chroma by including
        the semantic vector as the embedding vector.

        Returns:
            Dict[str, Any]: Chroma-compatible dictionary containing
                relation metadata and vector embedding

        Example:
            ```python
            data = relation.to_chroma()
            # Returns: {
            #     'relation_id': UUID('123e4567-e89b-12d3-a456-426614174000'),
            #     'name': 'AUTHORED_BY',
            #     'vector': [0.1, 0.2, 0.3, 0.4],
            #     'hit_count': 1
            # }
            ```
        """
        data = self.model_dump()
        data['vector'] = self.semantic
        return data

class RelationSymbol(BaseObject):
    """Represents a symbolic relation in the knowledge graph.

    Handles symbolic representations of relationships between entities,
    combining semantic information with descriptive metadata and strength
    metrics. Used for graph database storage and relationship analysis.

    Attributes:
        relation_id (UUID): Unique identifier for the relation
        name (str): Name of the relationship
        description (str): Detailed description of the relationship
        semantics (RelationSemantic): Associated semantic representation
        hit_count (int): Number of times this relation has been observed
        relationship_strength (int): Strength of the relationship (1-10)

    Example:
        ```python
        symbol = RelationSymbol(
            relation_id=UUID('123e4567-e89b-12d3-a456-426614174000'),
            name='AUTHORED_BY',
            description='Indicates authorship relationship between person and document',
            semantics=RelationSemantic(...),
            hit_count=1,
            relationship_strength=8
        )
        neo4j_data = symbol.to_neo4j()
        ```

    Note:
        The relationship_strength attribute provides a normalized measure
        of relationship significance on a scale of 1-10.
    """
    relation_id: UUID
    name: str
    description: str
    semantics: RelationSemantic
    hit_count: int
    relationship_strength: int

    def to_neo4j(self) -> Dict[str, Any]:
        """Convert symbolic relation to Neo4j graph database format.

        Prepares the relation data for storage in Neo4j, including
        metadata but excluding the semantic vector representation.

        Returns:
            Dict[str, Any]: Neo4j-compatible dictionary containing
                relation metadata without vector embedding

        Example:
            ```python
            data = symbol.to_neo4j()
            # Returns: {
            #     'relation_id': '123e4567-e89b-12d3-a456-426614174000',
            #     'name': 'AUTHORED_BY',
            #     'description': 'Indicates authorship relationship...',
            #     'hit_count': 1
            # }
            ```
        """
        return {
            'relation_id': str(self.relation_id),
            'name': self.name,
            'description': self.description,
            'hit_count': self.hit_count
        }
