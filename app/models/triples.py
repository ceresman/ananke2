"""Triple models for knowledge graph relationships.

This module defines the core data structures for representing triples
in both semantic and symbolic forms. Triples are fundamental building
blocks of the knowledge graph, representing subject-predicate-object
relationships.
"""
from uuid import UUID
from typing import Dict, Any
from pydantic import Field

from .types import SemanticBase, SymbolBase

class TripleSymbol(SymbolBase):
    """Represents a symbolic triple in the knowledge graph.

    A triple represents a relationship between two entities (subject and object)
    through a predicate. This symbolic representation includes metadata and
    properties inherited from SymbolBase.

    Attributes:
        subject_id (UUID): ID of the subject entity
        predicate_id (UUID): ID of the predicate relation
        object_id (UUID): ID of the object entity
        symbol_id (UUID): Inherited from SymbolBase, unique identifier
        properties (Dict): Inherited from SymbolBase, additional properties
        labels (List[str]): Inherited from SymbolBase, classification labels

    Example:
        ```python
        triple = TripleSymbol(
            subject_id=UUID('...'),
            predicate_id=UUID('...'),
            object_id=UUID('...'),
            properties={'confidence': 0.95},
            labels=['ACADEMIC', 'CITATION']
        )
        neo4j_data = triple.to_neo4j()
        ```
    """
    subject_id: UUID
    predicate_id: UUID
    object_id: UUID

    def to_neo4j(self) -> Dict[str, Any]:
        """Convert triple to Neo4j graph database format.

        Prepares the triple data for storage in Neo4j, including
        entity references and inherited properties.

        Returns:
            Dict[str, Any]: Neo4j-compatible dictionary containing
                triple data and metadata

        Example:
            ```python
            data = triple.to_neo4j()
            # Returns: {
            #     'triple_id': 'uuid-string',
            #     'subject_id': 'uuid-string',
            #     'predicate_id': 'uuid-string',
            #     'object_id': 'uuid-string',
            #     'properties': {'confidence': 0.95},
            #     'labels': ['ACADEMIC', 'CITATION']
            # }
            ```
        """
        return {
            'triple_id': str(self.symbol_id),
            'subject_id': str(self.subject_id),
            'predicate_id': str(self.predicate_id),
            'object_id': str(self.object_id),
            'properties': self.properties,
            'labels': self.labels
        }

class TripleSemantic(SemanticBase):
    """Represents a semantic triple in the knowledge graph.

    A semantic triple captures the vector representation and semantic
    meaning of a subject-predicate-object relationship, inheriting
    semantic properties from SemanticBase.

    Attributes:
        subject_id (UUID): ID of the subject entity
        predicate_id (UUID): ID of the predicate relation
        object_id (UUID): ID of the object entity
        semantic_id (UUID): Inherited from SemanticBase, unique identifier
        semantic_type (str): Inherited from SemanticBase, semantic category
        semantic_value (List[float]): Inherited from SemanticBase, vector representation
        confidence (float): Inherited from SemanticBase, confidence score

    Example:
        ```python
        triple = TripleSemantic(
            subject_id=UUID('...'),
            predicate_id=UUID('...'),
            object_id=UUID('...'),
            semantic_type='RELATIONSHIP',
            semantic_value=[0.1, 0.2, 0.3],
            confidence=0.95
        )
        chroma_data = triple.to_chroma()
        ```
    """
    subject_id: UUID
    predicate_id: UUID
    object_id: UUID

    def to_chroma(self) -> Dict[str, Any]:
        """Convert triple to Chroma vector database format.

        Prepares the triple data for storage in Chroma, including
        entity references and semantic properties.

        Returns:
            Dict[str, Any]: Chroma-compatible dictionary containing
                triple data and semantic information

        Example:
            ```python
            data = triple.to_chroma()
            # Returns: {
            #     'triple_id': 'uuid-string',
            #     'subject_id': 'uuid-string',
            #     'predicate_id': 'uuid-string',
            #     'object_id': 'uuid-string',
            #     'semantic_type': 'RELATIONSHIP',
            #     'semantic_value': [0.1, 0.2, 0.3],
            #     'confidence': 0.95
            # }
            ```
        """
        return {
            'triple_id': str(self.semantic_id),
            'subject_id': str(self.subject_id),
            'predicate_id': str(self.predicate_id),
            'object_id': str(self.object_id),
            'semantic_type': self.semantic_type,
            'semantic_value': self.semantic_value,
            'confidence': self.confidence
        }
