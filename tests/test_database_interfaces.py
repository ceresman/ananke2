"""Unit tests for database interfaces using mocks."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import UUID
import neo4j
from neo4j.exceptions import AuthError
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.graph import Neo4jInterface
from app.database.vector import ChromaInterface
from app.database.relational import AsyncRelationalDatabase, MySQLInterface
from app.models.entities import Entity, EntitySymbol, EntitySemantic
from app.models.structured import StructuredData
from contextlib import ExitStack
from sqlalchemy.engine.cursor import CursorResult
from sqlalchemy.engine.result import ResultMetaData
import os

@pytest.fixture
def entity_symbol():
    """Create a test EntitySymbol."""
    return EntitySymbol(
        symbol_id=UUID('12345678-1234-5678-1234-567812345678'),
        name="test_entity",
        descriptions=["Test description"],
        entity_type="TEST_ENTITY",  # Add required entity_type
        semantics=[
            EntitySemantic(
                semantic_id=UUID('87654321-4321-8765-4321-876543210987'),
                name="test_semantic",
                vector_representation=[0.1, 0.2, 0.3]
            )
        ],
        properties=[
            StructuredData(
                data_id=UUID('11111111-1111-1111-1111-111111111111'),
                data_type="property",
                data_value={"key": "value"}
            )
        ],
        labels=[
            StructuredData(
                data_id=UUID('22222222-2222-2222-2222-222222222222'),
                data_type="label",
                data_value={"category": "test"}
            )
        ]
    )

# Mock data for Neo4j test
mock_read_data = {
    'e': {
        'id': str(UUID('12345678-1234-5678-1234-567812345678')),
        'name': "Test Entity",
        'descriptions': ["Test Description"],
        'entity_type': "TEST_TYPE"  # Add entity_type to mock data
    }
}

@pytest.mark.asyncio
async def test_neo4j_interface():
    """Test Neo4j interface operations."""
    class MockResult:
        def __init__(self, records=None):
            self._records = records or [{"e": mock_read_data}]

        async def single(self):
            return self._records[0] if self._records else None

        async def all(self):
            return self._records

    class MockSession:
        def __init__(self):
            self._closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.close()

        async def run(self, query, **kwargs):
            if "CREATE" in query:
                return MockResult([{"e": {"id": str(UUID("123e4567-e89b-12d3-a456-426614174000"))}}])
            elif "MATCH" in query:
                return MockResult()
            return MockResult([{"count": 1}])

        async def close(self):
            self._closed = True

    class MockDriver:
        def __init__(self):
            self._closed = False

        async def verify_connectivity(self):
            return True

        def session(self):
            return MockSession()

        async def close(self):
            self._closed = True

    # Create interface with mock driver
    interface = Neo4jInterface(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    interface._driver = MockDriver()

    # Test connection
    await interface.connect()
    assert interface._driver is not None

    # Test entity creation
    entity = EntitySymbol(
        symbol_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
        name="test",
        descriptions=["Test description"],
        entity_type="TEST",
        semantics=[],
        properties=[],
        labels=[]
    )
    result = await interface.create(entity)
    assert result == entity.symbol_id

    # Test disconnection
    await interface.disconnect()
    assert interface._driver is None

@pytest.mark.unit
async def test_chroma_interface():
    """Test Chroma interface with mocked connection."""
    # Create mock client
    mock_client = MagicMock()
    mock_collection = MagicMock()

    # Setup mock collection methods
    mock_collection.add = MagicMock()
    mock_collection.get = MagicMock(return_value={
        'ids': ['12345678-1234-5678-1234-567812345678'],
        'embeddings': [[0.1, 0.2, 0.3]],
        'metadatas': [{'name': 'test_entity'}]
    })

    mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)

    # Patch the ChromaDB client constructor
    with patch('chromadb.Client', return_value=mock_client):
        # Initialize interface with default settings
        interface = ChromaInterface()
        await interface.connect()

        # Test semantic entity
        semantic = EntitySemantic(
            semantic_id=UUID('12345678-1234-5678-1234-567812345678'),
            name='test_entity',
            vector_representation=np.array([0.1, 0.2, 0.3])
        )

        # Test create and read operations
        created_id = await interface.create(semantic)
        assert created_id == semantic.semantic_id

        read_data = await interface.read(semantic.semantic_id)
        assert read_data is not None
        assert isinstance(read_data, EntitySemantic)
        assert read_data.semantic_id == semantic.semantic_id
        assert read_data.name == semantic.name
        assert np.array_equal(read_data.vector_representation, semantic.vector_representation)

        await interface.disconnect()

@pytest.mark.asyncio
async def test_mysql_interface():
    """Test MySQL interface."""
    # Create mock session maker
    class MockAsyncSession:
        def __init__(self):
            self._closed = False
            self._committed = False
            self._rolled_back = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.close()

        async def commit(self):
            self._committed = True

        async def rollback(self):
            self._rolled_back = True

        async def close(self):
            self._closed = True

        async def execute(self, query, params=None):
            class Result:
                def __init__(self):
                    self._data = [{"id": "123e4567-e89b-12d3-a456-426614174000"}]

                def scalars(self):
                    return self

                def first(self):
                    return self._data[0] if self._data else None

                def all(self):
                    return self._data

                def one(self):
                    if not self._data:
                        raise Exception("No results")
                    return self._data[0]

            return Result()

    class AsyncSessionMaker:
        def __call__(self):
            return MockAsyncSession()

    # Create interface with mock session
    interface = AsyncRelationalDatabase(
        host="localhost",
        port=3306,
        user="test",
        password="test",
        database="test"
    )
    interface._session_factory = AsyncSessionMaker()

    # Test entity creation
    entity = Entity(
        id=UUID("123e4567-e89b-12d3-a456-426614174000"),
        name="Test Entity",
        type="TEST",
        description="Test Description",
        document_id=UUID("123e4567-e89b-12d3-a456-426614174001"),
        properties={"test": "value"}
    )

    result = await interface.create_entity(entity)
    assert result["id"] == str(entity.id)

@pytest.mark.asyncio
async def test_model_serialization():
    """Test model serialization and deserialization."""
    entity = EntitySymbol(
        symbol_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
        name="test",
        descriptions=["Test description"],
        entity_type="TEST",
        semantics=[],
        properties=[],
        labels=[]
    )

    # Test serialization
    serialized = entity.model_dump()
    assert serialized["name"] == "test"
    assert serialized["descriptions"] == ["Test description"]
    assert serialized["entity_type"] == "TEST"

    # Test deserialization
    deserialized = EntitySymbol(**serialized)
    assert deserialized.name == entity.name
    assert deserialized.descriptions == entity.descriptions
    assert deserialized.entity_type == entity.entity_type
