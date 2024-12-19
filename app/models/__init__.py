from .base import BaseObject
from .types import SemanticBase, SymbolBase, StructuredDataBase
from .expressions import LogicExpression, MathExpression
from .entities import EntitySemantic, EntitySymbol
from .relations import RelationSemantic, RelationSymbol
from .triples import TripleSymbol, TripleSemantic
from .structured import (
    StructuredData,
    StructuredSentence,
    StructuredChunk,
    Document
)

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
BaseObject.model_rebuild()
SemanticBase.model_rebuild()
SymbolBase.model_rebuild()
StructuredDataBase.model_rebuild()

# Then rebuild semantic models
EntitySemantic.model_rebuild()
RelationSemantic.model_rebuild()
TripleSemantic.model_rebuild()

# Then rebuild symbol models
EntitySymbol.model_rebuild()
RelationSymbol.model_rebuild()
TripleSymbol.model_rebuild()

# Finally rebuild structured models
StructuredData.model_rebuild()
StructuredSentence.model_rebuild()
StructuredChunk.model_rebuild()
Document.model_rebuild()
