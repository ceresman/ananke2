"""Unit tests for database interfaces using mocks."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from uuid import UUID
import numpy as np
from typing import List

from app.models.entities import EntitySymbol, EntitySemantic
from app.models.structured import StructuredData
from app.database.graph import Neo4jInterface
from app.database.vector import ChromaInterface
from app.database.relational import MySQLInterface

@pytest.fixture
def entity_symbol():
    """Create a test EntitySymbol."""
    return EntitySymbol(
        symbol_id=UUID('12345678-1234-5678-1234-567812345678'),
        name="test_entity",
        descriptions=["Test description"],
        semantics=[
            EntitySemantic(
                semantic_id=UUID('87654321-4321-8765-4321-876543210987'),
                name="test_semantic",
                vector_representation=[0.1, 0.2, 0.3]
            )
        ],
        propertys=[
            StructuredData(
                data_id=UUID('11111111-1111-1111-1111-111111111111'),
                data_type="property",
                data_value={"key": "value"}
            )
        ],
        label=[
            StructuredData(
                data_id=UUID('22222222-2222-2222-2222-222222222222'),
                data_type="label",
                data_value={"category": "test"}
            )
        ]
    )

@pytest.mark.asyncio
async def test_neo4j_interface():
    """Test Neo4j interface with mocked connection."""
    with patch('neo4j.AsyncGraphDatabase') as mock_neo4j:
        # Setup mock
        mock_session = AsyncMock()
        mock_neo4j.driver.return_value.async_session.return_value = mock_session
        mock_session.__aenter__.return_value.run = AsyncMock()

        # Initialize interface
        interface = Neo4jInterface(uri="bolt://localhost:7687", username="neo4j", password="password")
        await interface.connect()

        # Test entity
        entity = entity_symbol()

        # Test create
        created_id = await interface.create(entity)
        assert created_id == entity.symbol_id
        mock_session.__aenter__.return_value.run.assert_called()

        # Test read
        mock_session.__aenter__.return_value.run.return_value.single.return_value = {
            "n": {
                "symbol_id": str(entity.symbol_id),
                "name": entity.name,
                "descriptions": entity.descriptions
            }
        }
        retrieved = await interface.read(entity.symbol_id)
        assert retrieved is not None
        assert retrieved.name == entity.name

        await interface.disconnect()

@pytest.mark.asyncio
async def test_chroma_interface():
    """Test Chroma interface with mocked client."""
    with patch('chromadb.HttpClient') as mock_chroma:
        # Setup mock
        mock_client = AsyncMock()
        mock_chroma.return_value = mock_client
        mock_client.get_collection = AsyncMock()

        # Initialize interface
        interface = ChromaInterface(host="localhost", port=8000, collection_name="test")
        await interface.connect()

        # Test semantic entity
        semantic = EntitySemantic(
            semantic_id=UUID('12345678-1234-5678-1234-567812345678'),
            name="test_semantic",
            vector_representation=[0.1, 0.2, 0.3]
        )

        # Test create
        created_id = await interface.create(semantic)
        assert created_id == semantic.semantic_id
        mock_client.get_collection.assert_called()

        await interface.disconnect()

@pytest.mark.asyncio
async def test_mysql_interface():
    """Test MySQL interface with mocked connection."""
    with patch('sqlalchemy.ext.asyncio.create_async_engine'):
        # Initialize interface
        interface = MySQLInterface(
            host="localhost",
            port=3306,
            user="root",
            password="password",
            database="test"
        )
        await interface.connect()

        # Test structured data
        data = StructuredData(
            data_id=UUID('12345678-1234-5678-1234-567812345678'),
            data_type="test_type",
            data_value={"key": "value"}
        )

        # Test create
        created_id = await interface.create(data)
        assert created_id == data.data_id

        # Test serialization
        serialized = interface._serialize_model(data)
        assert isinstance(serialized, dict)
        assert "data_id" in serialized
        assert "data_type" in serialized
        assert "data_value" in serialized

        await interface.disconnect()

def test_model_serialization():
    """Test that all data models can be properly serialized."""
    # Test EntitySymbol serialization
    entity = entity_symbol()
    serialized = entity.model_dump()
    assert isinstance(serialized, dict)
    assert "symbol_id" in serialized
    assert "name" in serialized
    assert "descriptions" in serialized
    assert "semantics" in serialized
    assert isinstance(serialized["semantics"], list)

    # Test EntitySemantic serialization
    semantic = entity.semantics[0]
    semantic_serialized = semantic.model_dump()
    assert isinstance(semantic_serialized, dict)
    assert "semantic_id" in semantic_serialized
    assert "name" in semantic_serialized
    assert "vector_representation" in semantic_serialized

    # Test StructuredData serialization
    structured = entity.propertys[0]
    structured_serialized = structured.model_dump()
    assert isinstance(structured_serialized, dict)
    assert "data_id" in structured_serialized
    assert "data_type" in structured_serialized
    assert "data_value" in structured_serialized
