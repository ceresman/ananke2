"""Common type definitions for models."""
from uuid import UUID
from typing import List, Dict, Any, Optional
from pydantic import Field, ConfigDict
from .base import BaseObject

class StructuredDataBase(BaseObject):
    """Base class for structured data storage and manipulation.

    Provides a standardized format for storing structured data with
    type information and arbitrary value storage. Used as the base
    for document metadata and other structured content.

    Attributes:
        data_id (UUID): Unique identifier for the data instance
        data_type (str): Type identifier for the data
        data_value (Dict[str, Any]): Arbitrary structured data content

    Example:
        ```python
        data = StructuredDataBase(
            data_id=UUID('...'),
            data_type='document_metadata',
            data_value={'title': 'Example', 'author': 'John Doe'}
        )
        ```
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_id: UUID
    data_type: str
    data_value: Dict[str, Any]

class SemanticBase(BaseObject):
    """Base class for semantic information representation.

    Handles vector-based semantic representations with confidence
    scores. Used for embedding-based search and semantic similarity
    comparisons.

    Attributes:
        semantic_id (UUID): Unique identifier for the semantic entity
        semantic_type (str): Type of semantic information
        semantic_value (str): Vector representation or semantic content
        confidence (float): Confidence score for the semantic information

    Example:
        ```python
        semantic = SemanticBase(
            semantic_id=UUID('...'),
            semantic_type='entity_embedding',
            semantic_value='[0.1, 0.2, 0.3]',
            confidence=0.95
        )
        ```
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    semantic_id: UUID
    semantic_type: str
    semantic_value: str
    confidence: float = Field(default=1.0)

class SymbolBase(BaseObject):
    """Base class for symbolic entity representation.

    Provides a foundation for representing named entities with
    descriptions, properties, and labels. Used for graph database
    storage and symbolic reasoning.

    Attributes:
        symbol_id (UUID): Unique identifier for the symbol
        name (str): Name or identifier of the symbol
        descriptions (List[str]): Descriptive texts about the symbol
        properties (Dict[str, Any]): Additional properties
        labels (List[str]): Classification labels
        document_id (Optional[UUID]): Source document identifier

    Example:
        ```python
        symbol = SymbolBase(
            symbol_id=UUID('...'),
            name='Example Entity',
            descriptions=['An example entity'],
            properties={'type': 'concept'},
            labels=['ACADEMIC'],
            document_id=UUID('...')
        )
        dumped = symbol.model_dump()
        ```
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol_id: UUID
    name: str
    descriptions: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    labels: List[str] = Field(default_factory=list)
    document_id: Optional[UUID] = None

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert symbol to dictionary format with string UUIDs.

        Extends the base model_dump method to ensure UUIDs are
        properly serialized as strings.

        Args:
            **kwargs: Additional arguments passed to model_dump

        Returns:
            Dict[str, Any]: Dictionary representation with string UUIDs

        Example:
            ```python
            data = symbol.model_dump()
            # Returns: {
            #     'symbol_id': 'uuid-string',
            #     'name': 'Example Entity',
            #     'document_id': 'uuid-string'  # if present
            # }
            ```
        """
        data = super().model_dump(**kwargs)
        data['symbol_id'] = str(data['symbol_id'])
        if data.get('document_id'):
            data['document_id'] = str(data['document_id'])
        return data
