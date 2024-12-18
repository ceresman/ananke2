from uuid import UUID
from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np
from .base import BaseObject
from .expressions import LogicExpression, MathExpression
from .triples import TripleSymbol

if TYPE_CHECKING:
    from .entities import EntitySymbol
    from .relations import RelationSymbol

class StructuredData(BaseObject):
    """Represents structured data."""
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
    entity_relations: List[TripleSymbol]
    logic_expressions: List[LogicExpression]
    math_expressions: List[MathExpression]
    sentence_vectorization: List[float]  # Using List[float] instead of np.ndarray
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
    chunk_id: UUID
    chunk_raw_content: str
    chunk_summary_content: str
    modality_identifier: str
    document_id: UUID
    extraction_entity_results: List['EntitySymbol']
    extraction_relation_results: List['RelationSymbol']
    extraction_triple_results: List[TripleSymbol]
    logic_expression_extraction_results: List[LogicExpression]
    math_expression_extraction_results: List[MathExpression]

class Document(BaseObject):
    """Represents a document or a file."""
    id: UUID
    meta: StructuredData
    meta_embedding: List[float]  # Using List[float] instead of np.ndarray
    raw_content: str
    StructuredChunks: List[StructuredChunk]

    def to_chroma(self) -> Dict[str, Any]:
        """Special handling for Chroma vector database."""
        return {
            'vector': self.meta_embedding,
            'metadata': {
                'document_id': str(self.id),
                'meta': self.meta.model_dump()
            }
        }
