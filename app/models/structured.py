"""Structured data models.

This module defines the core data structures for handling structured content
in the knowledge graph system, including documents, chunks, and sentences.
The models support various data formats and database-specific serialization.
"""
from uuid import UUID
from typing import List, Dict, Any, TYPE_CHECKING
from pydantic import Field, ConfigDict
from .base import BaseObject
from .expressions import LogicExpression, MathExpression
from .types import StructuredDataBase

if TYPE_CHECKING:
    from .entities import EntitySymbol
    from .relations import RelationSymbol
    from .triples import TripleSymbol
else:
    # Runtime imports for Pydantic
    from .entities import EntitySymbol
    from .relations import RelationSymbol
    from .triples import TripleSymbol

class StructuredData(StructuredDataBase):
    """Represents structured metadata for documents.

    Extends StructuredDataBase to provide a standardized format for
    document metadata storage and retrieval.

    Note:
        This class inherits all attributes and methods from StructuredDataBase.
        See StructuredDataBase documentation for detailed information.
    """
    pass

class StructuredSentence(BaseObject):
    """Represents a structured sentence within a document chunk.

    Handles the storage and organization of sentence-level information,
    including entity relationships, expressions, and vector representations.

    Attributes:
        entity_relations (List[TripleSymbol]): Entity relationship triples
        logic_expressions (List[LogicExpression]): Logical expressions
        math_expressions (List[MathExpression]): Mathematical expressions
        sentence_vectorization (List[float]): Vector representation
        parent_chunk_id (UUID): ID of the containing chunk
        document_id (UUID): ID of the parent document

    Example:
        ```python
        sentence = StructuredSentence(
            entity_relations=[TripleSymbol(...)],
            logic_expressions=[LogicExpression(...)],
            math_expressions=[MathExpression(...)],
            sentence_vectorization=[0.1, 0.2, 0.3],
            parent_chunk_id=UUID('...'),
            document_id=UUID('...')
        )
        chroma_data = sentence.to_chroma()
        ```
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_relations: List[TripleSymbol] = Field(default_factory=list)
    logic_expressions: List[LogicExpression] = Field(default_factory=list)
    math_expressions: List[MathExpression] = Field(default_factory=list)
    sentence_vectorization: List[float] = Field(default_factory=list)
    parent_chunk_id: UUID
    document_id: UUID

    def to_chroma(self) -> Dict[str, Any]:
        """Convert sentence to Chroma vector database format.

        Prepares the sentence data for storage in Chroma, including
        vector representation and document/chunk identifiers.

        Returns:
            Dict[str, Any]: Chroma-compatible dictionary containing
                vector embedding and metadata

        Example:
            ```python
            data = sentence.to_chroma()
            # Returns: {
            #     'vector': [0.1, 0.2, 0.3],
            #     'metadata': {
            #         'document_id': 'uuid-string',
            #         'chunk_id': 'uuid-string'
            #     }
            # }
            ```
        """
        return {
            'vector': self.sentence_vectorization,
            'metadata': {
                'document_id': str(self.document_id),
                'chunk_id': str(self.parent_chunk_id)
            }
        }

class StructuredChunk(BaseObject):
    """Represents a structured chunk within a document.

    Handles the organization and storage of document chunks, including
    raw content, summaries, and extracted information like entities,
    relations, and expressions.

    Attributes:
        chunk_id (UUID): Unique identifier for the chunk
        chunk_raw_content (str): Original unprocessed content
        chunk_summary_content (str): Summarized content
        modality_identifier (str): Type/format of the chunk content
        document_id (UUID): ID of the parent document
        extraction_entity_results (List[EntitySymbol]): Extracted entities
        extraction_relation_results (List[RelationSymbol]): Extracted relations
        extraction_triple_results (List[TripleSymbol]): Extracted triples
        logic_expression_extraction_results (List[LogicExpression]): Logical expressions
        math_expression_extraction_results (List[MathExpression]): Mathematical expressions

    Example:
        ```python
        chunk = StructuredChunk(
            chunk_id=UUID('...'),
            chunk_raw_content='Original text...',
            chunk_summary_content='Summary...',
            modality_identifier='text/plain',
            document_id=UUID('...'),
            extraction_entity_results=[EntitySymbol(...)],
            extraction_relation_results=[RelationSymbol(...)],
            extraction_triple_results=[TripleSymbol(...)]
        )
        ```
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_id: UUID
    chunk_raw_content: str
    chunk_summary_content: str
    modality_identifier: str
    document_id: UUID
    extraction_entity_results: List[EntitySymbol] = Field(default_factory=list)
    extraction_relation_results: List[RelationSymbol] = Field(default_factory=list)
    extraction_triple_results: List[TripleSymbol] = Field(default_factory=list)
    logic_expression_extraction_results: List[LogicExpression] = Field(default_factory=list)
    math_expression_extraction_results: List[MathExpression] = Field(default_factory=list)

class Document(BaseObject):
    """Represents a complete document or file in the knowledge graph.

    Handles the storage and organization of document content, metadata,
    and structured chunks. Supports vector embeddings for document-level
    search and retrieval.

    Attributes:
        id (UUID): Unique identifier for the document
        meta (StructuredData): Document metadata
        meta_embedding (List[float]): Vector embedding of metadata
        raw_content (str): Original document content
        structured_chunks (List[StructuredChunk]): Processed document chunks

    Example:
        ```python
        document = Document(
            id=UUID('...'),
            meta=StructuredData(...),
            meta_embedding=[0.1, 0.2, 0.3],
            raw_content='Document text...',
            structured_chunks=[StructuredChunk(...)]
        )
        chroma_data = document.to_chroma()
        ```
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID
    meta: StructuredData
    meta_embedding: List[float] = Field(default_factory=list)
    raw_content: str
    structured_chunks: List[StructuredChunk] = Field(default_factory=list)

    def to_chroma(self) -> Dict[str, Any]:
        """Convert document to Chroma vector database format.

        Prepares the document data for storage in Chroma, including
        metadata embedding and document information.

        Returns:
            Dict[str, Any]: Chroma-compatible dictionary containing
                vector embedding and metadata

        Example:
            ```python
            data = document.to_chroma()
            # Returns: {
            #     'vector': [0.1, 0.2, 0.3],
            #     'metadata': {
            #         'document_id': 'uuid-string',
            #         'meta': {...}
            #     }
            # }
            ```
        """
        return {
            'vector': self.meta_embedding,
            'metadata': {
                'document_id': str(self.id),
                'meta': self.meta.model_dump()
            }
        }

# Update forward references after all models are defined
Document.model_rebuild()
StructuredChunk.model_rebuild()
StructuredSentence.model_rebuild()
