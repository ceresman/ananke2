"""Unit tests for database interfaces using mocks."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
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
async def test_neo4j_interface(entity_symbol):
    """Test Neo4j interface with mocked connection."""
    with patch('neo4j.AsyncGraphDatabase') as mock_neo4j:
        # Setup mock
        mock_session = AsyncMock()
        mock_driver = AsyncMock()
        mock_neo4j.driver.return_value = mock_driver
        mock_driver.async_session.return_value = mock_session
        mock_session.__aenter__.return_value.run = AsyncMock()

        # Initialize interface
        interface = Neo4jInterface(uri="bolt://localhost:7687", username="neo4j", password="password")
        await interface.connect()

        # Test create
        created_id = await interface.create(entity_symbol)
        assert created_id == entity_symbol.symbol_id
        mock_session.__aenter__.return_value.run.assert_called()

        # Test read
        mock_session.__aenter__.return_value.run.return_value.single.return_value = {
            "n": {
                "symbol_id": str(entity_symbol.symbol_id),
                "name": entity_symbol.name,
                "descriptions": entity_symbol.descriptions
            }
        }
        retrieved = await interface.read(entity_symbol.symbol_id)
        assert retrieved is not None
        assert retrieved.name == entity_symbol.name

        await interface.disconnect()

@pytest.mark.asyncio
async def test_chroma_interface():
    """Test Chroma interface with mocked client."""
    with patch('chromadb.Client') as mock_chroma:
        # Setup mock
        mock_client = MagicMock()
        mock_chroma.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        # Initialize interface
        interface = ChromaInterface(host="localhost", port=8000, collection_name="test")
        await interface.connect()

        # Test semantic entity
        semantic = EntitySemantic(
            semantic_id=UUID('12345678-1234-5678-1234-567812345678'),
            name="test_semantic",
            vector_representation=np.array([0.1, 0.2, 0.3])
        )

        # Test create
        created_id = await interface.create(semantic)
        assert created_id == semantic.semantic_id
        mock_client.get_or_create_collection.assert_called_once()

        await interface.disconnect()

@pytest.mark.asyncio
async def test_mysql_interface():
    """Test MySQL interface with mocked connection."""
    with patch('sqlalchemy.ext.asyncio.create_async_engine') as mock_engine:
        # Setup mock
        mock_conn = AsyncMock()
        mock_engine.return_value = AsyncMock()
        mock_engine.return_value.begin.return_value.__aenter__.return_value = mock_conn

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

        await interface.disconnect()

@pytest.mark.asyncio
async def test_model_serialization(entity_symbol):
    """Test that all data models can be properly serialized."""
    # Test EntitySymbol serialization
    serialized = entity_symbol.model_dump()
    assert isinstance(serialized, dict)
    assert "symbol_id" in serialized
    assert "name" in serialized
    assert "descriptions" in serialized
    assert "semantics" in serialized
    assert isinstance(serialized["semantics"], list)

    # Test EntitySemantic serialization
    semantic = entity_symbol.semantics[0]
    semantic_serialized = semantic.model_dump()
    assert isinstance(semantic_serialized, dict)
    assert "semantic_id" in semantic_serialized
    assert "name" in semantic_serialized
    assert "vector_representation" in semantic_serialized

    # Test StructuredData serialization
    structured = entity_symbol.propertys[0]
    structured_serialized = structured.model_dump()
    assert isinstance(structured_serialized, dict)
    assert "data_id" in structured_serialized
    assert "data_type" in structured_serialized
    assert "data_value" in structured_serialized
