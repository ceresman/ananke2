"""Unit tests for database interfaces using mocks."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import UUID, uuid4
import numpy as np
from contextlib import ExitStack

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

@pytest.mark.asyncio
async def test_neo4j_interface(entity_symbol):
    """Test Neo4j interface with mocked connection."""
    # Create mock driver and session
    mock_driver = AsyncMock()
    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_read_result = AsyncMock()

    # Setup mock result for create
    mock_result.single = AsyncMock(return_value={'e.id': str(entity_symbol.symbol_id)})

    # Setup mock result for read
    mock_read_result.single = AsyncMock(return_value={
        'e': {
            'id': str(entity_symbol.symbol_id),
            'name': entity_symbol.name,
            'descriptions': entity_symbol.descriptions
        }
    })

    # Setup mock session with proper async context management
    mock_session.__aenter__.return_value = mock_session

    # Configure run method to return different results based on the query
    async def mock_run(query, **kwargs):
        if 'CREATE' in query:
            return mock_result
        elif 'MATCH' in query:
            return mock_read_result
        return AsyncMock()

    mock_session.run = AsyncMock(side_effect=mock_run)

    # Setup mock driver with proper session creation
    mock_driver.session.return_value = mock_session

    # Patch Neo4j driver creation
    with patch('neo4j.AsyncGraphDatabase.driver', return_value=mock_driver):
        # Initialize interface
        interface = Neo4jInterface(uri="bolt://localhost:7687", username="neo4j", password="password")
        await interface.connect()

        # Test create
        created_id = await interface.create(entity_symbol)
        assert created_id == entity_symbol.symbol_id
        assert mock_session.run.called

        # Test read
        read_data = await interface.read(entity_symbol.symbol_id)
        assert read_data is not None
        assert isinstance(read_data, EntitySymbol)
        assert read_data.symbol_id == entity_symbol.symbol_id

        # Test error handling
        mock_session.run = AsyncMock(side_effect=Exception("Database error"))
        with pytest.raises(Exception):
            await interface.create(entity_symbol)

        await interface.disconnect()
        assert mock_driver.close.called

@pytest.mark.asyncio
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
    with patch('chromadb.Client', return_value=mock_client), \
         patch('chromadb.Settings') as mock_settings:

        # Initialize interface with default settings
        interface = ChromaInterface()
        await interface.connect()

        # Test semantic entity
        semantic = EntitySemantic(
            semantic_id=UUID('12345678-1234-5678-1234-567812345678'),
            name='test_entity',
            vector_representation=np.array([0.1, 0.2, 0.3])
        )

        # Test create
        created_id = await interface.create(semantic)
        assert created_id == semantic.semantic_id
        assert mock_collection.add.called

        # Test read
        read_data = await interface.read(semantic.semantic_id)
        assert read_data is not None
        assert isinstance(read_data, EntitySemantic)
        assert read_data.semantic_id == semantic.semantic_id
        assert read_data.name == semantic.name
        assert np.array_equal(read_data.vector_representation, semantic.vector_representation)

        # Test error handling
        mock_collection.add = MagicMock(side_effect=Exception("Vector store error"))
        with pytest.raises(Exception):
            await interface.create(semantic)

        await interface.disconnect()

@pytest.mark.asyncio
async def test_mysql_interface():
    """Test MySQL interface with mocked connection."""
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine, AsyncConnection
    from sqlalchemy.engine.url import URL
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import Pool, _ConnectionFairy
    import aiomysql
    import asyncio

    # Create mock session and result
    mock_session = AsyncMock(spec=AsyncSession)
    mock_result = AsyncMock()
    mock_result.scalar.return_value = None
    mock_session.execute.return_value = mock_result
    mock_session.commit = AsyncMock()

    # Create mock connection fairy with async context manager support
    class AsyncConnectionFairy(MagicMock):
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        def __await__(self):
            async def _await():
                return self
            return _await().__await__()

    mock_fairy = AsyncConnectionFairy(spec=_ConnectionFairy)
    mock_fairy._connection = AsyncMock()

    # Create mock pool with async support
    mock_pool = MagicMock(spec=Pool)
    async def async_connect():
        return mock_fairy
    mock_pool.connect = async_connect

    # Create mock engine with proper async behavior
    mock_engine = AsyncMock(spec=AsyncEngine)
    mock_engine.pool = mock_pool
    mock_engine.raw_connection = async_connect
    mock_engine.begin = AsyncMock()
    mock_engine.begin.return_value = AsyncMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_fairy)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)

    # Create mock session factory
    class MockSessionFactory:
        def __init__(self):
            self.session = mock_session

        async def __call__(self):
            return self.session

        async def __aenter__(self):
            return self.session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.session.close()

    # Initialize MySQL interface with mocked components
    with patch('sqlalchemy.ext.asyncio.create_async_engine', return_value=mock_engine), \
         patch('sqlalchemy.orm.sessionmaker', return_value=MockSessionFactory()):

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
        assert mock_session.execute.called

        # Test read
        read_data = await interface.read(data.data_id)
        assert read_data is not None
        assert isinstance(read_data, StructuredData)
        assert read_data.data_id == data.data_id
        assert read_data.data_type == data.data_type
        assert read_data.data_value == data.data_value

        # Test error handling
        mock_session.execute = AsyncMock(side_effect=Exception("Database error"))
        with pytest.raises(Exception):
            await interface.create(data)

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
    assert isinstance(serialized["semantics"], list)
    assert "properties" in serialized
    assert "labels" in serialized

    # Test EntitySemantic serialization
    if entity_symbol.semantics:
        semantic = entity_symbol.semantics[0]
        semantic_serialized = semantic.model_dump()
        assert isinstance(semantic_serialized, dict)
        assert "semantic_id" in semantic_serialized
        assert "name" in semantic_serialized
        assert "vector_representation" in semantic_serialized

    # Test StructuredData serialization
    property_data = entity_symbol.properties[0]
    property_serialized = property_data.model_dump()
    assert isinstance(property_serialized, dict)
    assert "data_id" in property_serialized
    assert "data_type" in property_serialized
    assert "data_value" in property_serialized
