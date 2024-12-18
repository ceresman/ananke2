from .base import BaseObject
from .types import SemanticBase, SymbolBase
from .expressions import LogicExpression, MathExpression
from .structured import (
    StructuredData,
    StructuredSentence,
    StructuredChunk,
    Document
)
from .entities import EntitySemantic, EntitySymbol
from .relations import RelationSemantic, RelationSymbol
from .triples import TripleSymbol, TripleSemantic

__all__ = [
    'BaseObject',
    'SemanticBase',
    'SymbolBase',
    'EntitySemantic',
    'EntitySymbol',
    'RelationSemantic',
    'RelationSymbol',
    'TripleSymbol',
    'TripleSemantic',
    'LogicExpression',
    'MathExpression',
    'StructuredData',
    'StructuredSentence',
    'StructuredChunk',
    'Document'
]

# Rebuild base models first
SemanticBase.model_rebuild()
SymbolBase.model_rebuild()

# Then rebuild derived models
EntitySemantic.model_rebuild()
EntitySymbol.model_rebuild()
RelationSemantic.model_rebuild()
RelationSymbol.model_rebuild()
TripleSymbol.model_rebuild()
TripleSemantic.model_rebuild()

# Finally rebuild structured models
StructuredData.model_rebuild()
StructuredSentence.model_rebuild()
StructuredChunk.model_rebuild()
Document.model_rebuild()
