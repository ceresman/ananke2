"""Common type definitions for models."""
from uuid import UUID
from typing import List, Dict, Any, Optional
from pydantic import Field, ConfigDict
from .base import BaseObject

class StructuredDataBase(BaseObject):
    """Base class for structured data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_id: UUID
    data_type: str
    data_value: Dict[str, Any]

class SemanticBase(BaseObject):
    """Base class for semantic information."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    semantic_id: UUID
    semantic_type: str
    semantic_value: str
    confidence: float = Field(default=1.0)

class SymbolBase(BaseObject):
    """Base class for symbolic entities."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol_id: UUID
    name: str
    descriptions: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    labels: List[str] = Field(default_factory=list)
    document_id: Optional[UUID] = None

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        data = super().model_dump(**kwargs)
        data['symbol_id'] = str(data['symbol_id'])
        if data.get('document_id'):
            data['document_id'] = str(data['document_id'])
        return data
