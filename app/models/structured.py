from uuid import UUID
from typing import List, Dict, Any, TYPE_CHECKING
from pydantic import Field, ConfigDict
from .base import BaseObject
from .expressions import LogicExpression, MathExpression
from .triples import TripleSymbol
from .entities import EntitySymbol
from .relations import RelationSymbol

class StructuredData(BaseObject):
    """Represents structured data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_id: UUID
    data_type: str
    data_value: Dict[str, Any]

    def to_mysql(self) -> Dict[str, Any]:
        """Special handling for MySQL database."""
        return {
            'data_id': str(self.data_id),
            'data_type': self.data_type,
            'data_value': self.data_value
        }

class StructuredSentence(BaseObject):
    """Represents a structured sentence within a chunk."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_relations: List[TripleSymbol] = Field(default_factory=list)
    logic_expressions: List[LogicExpression] = Field(default_factory=list)
    math_expressions: List[MathExpression] = Field(default_factory=list)
    sentence_vectorization: List[float] = Field(default_factory=list)
    parent_chunk_id: UUID
    document_id: UUID

    def to_chroma(self) -> Dict[str, Any]:
        """Special handling for Chroma vector database."""
        return {
            'vector': self.sentence_vectorization,
            'metadata': {
                'document_id': str(self.document_id),
                'chunk_id': str(self.parent_chunk_id)
            }
        }

class StructuredChunk(BaseObject):
    """Represents a structured chunk within a document."""
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
    """Represents a document or a file."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID
    meta: StructuredData
    meta_embedding: List[float] = Field(default_factory=list)
    raw_content: str
    StructuredChunks: List[StructuredChunk] = Field(default_factory=list)

    def to_chroma(self) -> Dict[str, Any]:
        """Special handling for Chroma vector database."""
        return {
            'vector': self.meta_embedding,
            'metadata': {
                'document_id': str(self.id),
                'meta': self.meta.model_dump()
            }
        }

# Update forward references
Document.model_rebuild()
StructuredChunk.model_rebuild()
StructuredSentence.model_rebuild()
