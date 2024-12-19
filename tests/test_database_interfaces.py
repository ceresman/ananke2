"""Unit tests for database interfaces using mocks."""

import pytest
from uuid import UUID
from app.database.graph import Neo4jInterface
from app.database.vector import ChromaInterface
from app.database.relational import MySQLInterface
from app.models.entities import EntitySymbol, EntitySemantic
from app.models.structured import StructuredData
import os

# Set up test environment
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TORCH_CPU_ONLY"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["TF_CPU_ONLY"] = "1"

@pytest.fixture
def test_uuid():
    """Generate a test UUID."""
    return UUID('123e4567-e89b-12d3-a456-426614174000')

@pytest.fixture
def test_structured_data(test_uuid):
    """Create test structured data."""
    class TestStructuredData(StructuredDataBase):
        pass

    return TestStructuredData(
        data_id=test_uuid,
        data_type="TEST",
        data_value={"test": "value"}
    )

@pytest.mark.asyncio
async def test_entity_symbol():
    """Test EntitySymbol model."""
    entity = EntitySymbol(
        symbol_id=UUID(int=1),
        name="TEST_ENTITY",
        entity_type="TEST",
        descriptions=["Test entity description"],
        properties=[
            StructuredData(
                data_id=UUID(int=2),
                data_type="test_property",
                data_value={"key": "value", "number": 42}
            )
        ],
        labels=[
            StructuredData(
                data_id=UUID(int=3),
                data_type="test_label",
                data_value={"label": "test"}
            )
        ]
    )
    assert entity.name == "TEST_ENTITY"
    assert entity.entity_type == "TEST"
    assert len(entity.descriptions) == 1
    assert entity.descriptions[0] == "Test entity description"
    assert len(entity.properties) == 1
    assert entity.properties[0].data_type == "test_property"
    assert len(entity.labels) == 1
    assert entity.labels[0].data_type == "test_label"

@pytest.fixture
def test_entity_semantic(test_uuid):
    """Create test entity semantic."""
    return EntitySemantic(
        semantic_id=test_uuid,
        name="Test Semantic",
        semantic_type="TEST",
        semantic_value="Test Value",
        vector_representation=[0.1, 0.2, 0.3]
    )

# Mock data for Neo4j test
mock_read_data = {
    'e': {
        'id': UUID('12345678-1234-5678-1234-567812345678').bytes,  # Store as bytes
        'name': "Test Entity",
        'descriptions': ["Test Description"],
        'entity_type': "TEST_TYPE"
    }
}

@pytest.mark.asyncio
async def test_neo4j_interface():
    """Test Neo4j interface with mocked driver."""
    class MockRecord:
        def __init__(self, data):
            self._data = data

        def get(self, key):
            return self._data.get(key)

        def data(self):
            return self._data

    class MockResult:
        def __init__(self, records):
            self._records = records if records else []

        async def single(self):
            return self._records[0] if self._records else None

        async def all(self):
            return self._records

    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def run(self, query, **kwargs):
            if "CREATE" in query:
                record = MockRecord({
                    "e.id": test_entity_symbol.symbol_id.bytes  # Return bytes
                })
                return MockResult([record])
            elif "MATCH" in query:
                if kwargs.get("id") == test_entity_symbol.symbol_id.bytes:  # Compare bytes
                    record = MockRecord({
                        "e": {
                            "id": test_entity_symbol.symbol_id.bytes,
                            "name": test_entity_symbol.name,
                            "descriptions": test_entity_symbol.descriptions,
                            "entity_type": test_entity_symbol.entity_type
                        }
                    })
                    return MockResult([record])
            return MockResult([])

        async def close(self):
            pass

    class MockDriver:
        def __init__(self):
            self.closed = False
            self._auth_verified = True  # Start authenticated
            self._conn_verified = True  # Start connected
            self._session = MockSession()

        async def verify_authentication(self):
            """Mock authentication verification that always succeeds."""
            return True

        async def verify_connectivity(self):
            """Mock connectivity verification that always succeeds."""
            return True

        def session(self):  # Not async - Neo4j driver's session() is not async
            """Return mock session."""
            return self._session

        async def close(self):
            self.closed = True

    # Test Neo4j interface with proper credentials
    interface = Neo4jInterface("bolt://localhost:7687", "neo4j", "test123")
    interface._driver = MockDriver()

    # Test connection
    await interface.connect()
    assert interface._driver is not None
    assert interface._driver._auth_verified
    assert interface._driver._conn_verified

    # Test create
    created_id = await interface.create(test_entity_symbol)
    assert isinstance(created_id, UUID)
    assert str(created_id) == str(test_entity_symbol.symbol_id)

    # Test read
    read_entity = await interface.read(created_id)
    assert read_entity is not None
    assert isinstance(read_entity, EntitySymbol)
    assert str(read_entity.symbol_id) == str(test_entity_symbol.symbol_id)

    # Test disconnect
    await interface.disconnect()
    assert interface._driver.closed

@pytest.mark.asyncio
async def test_chroma_interface(test_entity_semantic):
    """Test Chroma interface implementation."""
    class MockCollection:
        def __init__(self):
            self.data = {}

        async def add(self, ids, embeddings, metadatas):
            """Mock add method that returns immediately."""
            for id_, embedding, metadata in zip(ids, embeddings, metadatas):
                # Ensure all metadata values are primitive types
                safe_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        safe_metadata[k] = v
                    elif isinstance(v, (list, tuple)):
                        # Convert first list item to string if exists
                        safe_metadata[k] = str(v[0]) if v else ""
                    elif isinstance(v, dict):
                        # Convert dict to string representation
                        safe_metadata[k] = str(v)
                    elif v is None:
                        safe_metadata[k] = ""
                    else:
                        safe_metadata[k] = str(v)
                self.data[id_] = {
                    "embedding": embedding,
                    "metadata": safe_metadata
                }
            return True

        async def get(self, ids=None, include=None, limit=None, offset=None):
            if ids:
                return {
                    "ids": ids,
                    "embeddings": [self.data[id_]["embedding"] for id_ in ids if id_ in self.data],
                    "metadatas": [self.data[id_]["metadata"] for id_ in ids if id_ in self.data]
                }
            return {"ids": [], "embeddings": [], "metadatas": []}

    class MockClient:
        def get_or_create_collection(self, name):
            return MockCollection()

    # Test Chroma interface
    interface = ChromaInterface()
    interface._client = MockClient()
    interface._collection = None  # Should be set during connect

    # Test connection
    await interface.connect()
    assert interface._collection is not None

    # Test create with proper UUID handling
    semantic_id = test_entity_semantic.semantic_id
    created_id = await interface.create(test_entity_semantic)
    assert isinstance(created_id, UUID)
    assert created_id == semantic_id

    # Test read with proper UUID handling
    read_semantic = await interface.read(created_id)
    assert read_semantic is not None
    assert isinstance(read_semantic, EntitySemantic)
    assert read_semantic.semantic_id == semantic_id
    assert read_semantic.name == test_entity_semantic.name
    assert read_semantic.semantic_type == test_entity_semantic.semantic_type
    assert read_semantic.semantic_value == test_entity_semantic.semantic_value

    # Test disconnect
    await interface.disconnect()
    assert interface._client is None
    assert interface._collection is None

@pytest.mark.asyncio
async def test_mysql_interface(test_structured_data):
    """Test MySQL interface implementation."""
    class MockResult:
        def __init__(self, data=None):
            self._data = data if data else []

        def first(self):
            return self._data[0] if self._data else None

        def all(self):
            return self._data

        def scalar(self):
            return self._data[0] if self._data else None

        def mappings(self):
            return [dict(item) for item in self._data] if self._data else []

    class MockTransaction:
        def __init__(self):
            self._committed = False
            self._rolled_back = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                await self.rollback()
            else:
                await self.commit()

        async def commit(self):
            self._committed = True

        async def rollback(self):
            self._rolled_back = True

    class MockSession:
        def __init__(self):
            self._closed = False
            self._transaction = None
            self._auth_checked = True  # Always authenticated
            self._connected = True  # Always connected

        async def __aenter__(self):
            if not self._connected:
                raise Exception("Not connected")
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.close()

        async def execute(self, statement, params=None):
            """Mock execute that simulates authentication."""
            if hasattr(self, '_auth_checked') and not self._auth_checked:
                raise Exception("Authentication required")

            if "INSERT" in str(statement):
                return MockResult([{"id": test_structured_data.data_id.bytes}])
            elif "SELECT" in str(statement):
                if params and params.get("id") == test_structured_data.data_id.bytes:
                    return MockResult([{
                        "id": test_structured_data.data_id.bytes,
                        "data_type": test_structured_data.data_type,
                        "data_value": test_structured_data.data_value
                    }])
            return MockResult()

        async def begin(self):
            self._transaction = MockTransaction()
            return self._transaction

        async def close(self):
            self._closed = True

    class MockEngine:
        def __init__(self):
            self._session = MockSession()
            self._pool = None

        async def begin(self):
            """Simulate SQLAlchemy engine begin."""
            class AsyncConnection:
                def __init__(self, session):
                    self.session = session
                async def __aenter__(self):
                    return self.session
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    await self.session.close()
                async def close(self):
                    await self.session.close()
                async def execute(self, statement):
                    return await self.session.execute(statement)
                async def run_sync(self, func):
                    """Mock run_sync for table creation."""
                    return True
            return AsyncConnection(self._session)

        async def dispose(self):
            """Simulate engine disposal."""
            await self._session.close()

        def raw_connection(self):
            """Return raw connection for SQLAlchemy."""
            return self._session

    # Test MySQL interface with mocked engine
    interface = MySQLInterface(
        host="localhost", port=3306, user="test", password="test", database="test"
    )
    interface._session_factory = lambda: MockSession()
    interface._engine = MockEngine()  # Use proper mock engine

    # Test connection
    await interface.connect()
    assert interface._session_factory is not None

    # Test create
    created_id = await interface.create(test_structured_data)
    assert isinstance(created_id, UUID)
    assert str(created_id) == str(test_structured_data.data_id)

    # Test read
    read_data = await interface.read(created_id)
    assert read_data is not None
    assert isinstance(read_data, StructuredDataBase)
    assert str(read_data.data_id) == str(test_structured_data.data_id)

    # Test disconnect
    await interface.disconnect()
    assert interface._engine is None

@pytest.mark.asyncio
async def test_model_serialization(test_entity_symbol):
    """Test model serialization and deserialization."""
    # Test serialization
    serialized = test_entity_symbol.model_dump()
    assert isinstance(serialized, dict)
    assert isinstance(serialized['symbol_id'], str)
    assert serialized['name'] == "Test Entity"
    assert serialized['entity_type'] == "TEST"
    assert len(serialized['properties']) > 0
    assert len(serialized['labels']) > 0

    # Test Neo4j format
    neo4j_format = test_entity_symbol.to_neo4j()
    assert isinstance(neo4j_format, dict)
    assert 'symbol_id' in neo4j_format
    assert 'name' in neo4j_format
    assert 'entity_type' in neo4j_format

    # Test MySQL format
    mysql_format = test_entity_symbol.to_mysql()
    assert isinstance(mysql_format, dict)
    assert 'symbol_id' in mysql_format
    assert 'name' in mysql_format
    assert 'entity_type' in mysql_format
