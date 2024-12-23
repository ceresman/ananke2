"""Unit tests for database interfaces using mocks."""

import pytest
from uuid import UUID
import neo4j
from app.models.types import StructuredDataBase  # Add import for test_structured_data
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
async def test_neo4j_interface(test_structured_data):
    """Test Neo4j interface implementation."""
    class MockRecord:
        def __init__(self, data=None):
            self._data = data if data else {}

        def __getitem__(self, key):
            return self._data.get(key)

        async def get(self, key):
            return self._data.get(key)

        async def data(self):
            return self._data

    class MockResult:
        def __init__(self, records=None):
            self._records = records if records else []

        async def single(self):
            if not self._records:
                return None
            return self._records[0]

        async def all(self):
            return self._records

    class MockSession:
        def __init__(self):
            self._closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.close()

        async def run(self, query, **params):
            """Mock query execution."""
            if "CREATE" in query:
                record = MockRecord({
                    "e": {"id": params["id"]},
                    "e.id": params["id"]
                })
                return MockResult([record])
            elif "MATCH" in query:
                if params.get("id") == test_structured_data.data_id.bytes:
                    record = MockRecord({
                        "e": {
                            "id": params["id"],
                            "name": "Test Entity",
                            "descriptions": ["Test description"],
                            "entity_type": "TEST",
                            "semantics": [],
                            "properties": [],
                            "labels": []
                        },
                        "e.id": params["id"],
                        "e.name": "Test Entity",
                        "e.descriptions": ["Test description"],
                        "e.entity_type": "TEST"
                    })
                    return MockResult([record])
            return MockResult()

        async def close(self):
            self._closed = True

    class MockDriver:
        def __init__(self):
            self._session = MockSession()
            self.closed = False

        async def session(self):
            return self._session

        async def verify_authentication(self):
            return True

        async def verify_connectivity(self):
            return True

        async def close(self):
            self.closed = True

    # Create interface with mock driver
    interface = Neo4jInterface("bolt://localhost:7687", "neo4j", "test123")
    interface.test_mode = True  # Enable test mode to skip real connection
    interface._driver = MockDriver()

    # Test operations
    entity = EntitySymbol(
        symbol_id=test_structured_data.data_id,
        name="Test Entity",
        descriptions=["Test description"],
        entity_type="TEST",
        semantics=[],
        properties=[],
        labels=[]
    )
    created_id = await interface.create(entity)
    assert created_id == test_structured_data.data_id

    read_data = await interface.read(test_structured_data.data_id)
    assert read_data is not None
    assert read_data.entity_type == entity.entity_type

    # Test cleanup
    driver = interface._driver  # Store reference before disconnect
    await interface.disconnect()
    assert driver.closed  # Check closed status on stored reference

    # Test disconnect
    await interface.disconnect()  # Should handle None gracefully

@pytest.mark.asyncio
async def test_chroma_interface(test_structured_data):
    """Test Chroma interface implementation."""
    class MockCollection:
        def __init__(self):
            self.data = {}

        async def add(self, ids, embeddings, metadatas):
            """Mock add with proper metadata validation."""
            for id_, embedding, metadata in zip(ids, embeddings, metadatas):
                # Convert complex types to strings, keep primitives as-is
                safe_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        safe_metadata[k] = v
                    elif isinstance(v, (list, dict, tuple)):
                        safe_metadata[k] = str(v)
                    else:
                        safe_metadata[k] = str(v)
                self.data[id_] = {
                    "embedding": embedding,
                    "metadata": safe_metadata
                }
            return True

        async def get(self, ids=None, where=None):
            """Mock get with proper return format."""
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
    interface._collection = interface._client.get_or_create_collection("test")

    # Test data operations
    created_id = await interface.create(test_structured_data)
    assert created_id == test_structured_data.data_id

    # Test read operation
    read_data = await interface.read(test_structured_data.data_id)
    assert read_data is not None
    assert isinstance(read_data, EntitySemantic)

    # Test read with non-existent ID
    non_existent_data = await interface.read(UUID('00000000-0000-0000-0000-000000000000'))
    assert non_existent_data is None

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

        def scalar_one_or_none(self):
            if not self._data:
                return None
            data = self._data[0]
            return type('StructuredDataTable', (), {
                'id': data['id'],
                'data_type': data['data_type'],
                'data_value': data['data_value']
            })()

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
            self._auth_checked = True
            self._connected = True
            self._items = []

        async def __aenter__(self):
            if not self._connected:
                raise Exception("Not connected")
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.close()

        async def execute(self, statement, params=None):
            """Mock execute that simulates database operations."""
            if "INSERT" in str(statement):
                return MockResult([{
                    "id": test_structured_data.data_id.bytes,
                    "data_type": test_structured_data.data_type,
                    "data_value": test_structured_data.data_value
                }])
            elif "SELECT" in str(statement):
                if params and "id" in params and params["id"] == test_structured_data.data_id.bytes:
                    return MockResult([{
                        "id": test_structured_data.data_id.bytes,
                        "data_type": test_structured_data.data_type,
                        "data_value": test_structured_data.data_value
                    }])
                elif "id" in str(statement):
                    return MockResult([{
                        "id": test_structured_data.data_id.bytes,
                        "data_type": test_structured_data.data_type,
                        "data_value": test_structured_data.data_value
                    }])
            return MockResult()

        def begin(self):
            """Return self as transaction."""
            return self

        async def commit(self):
            """Mock commit."""
            pass

        async def rollback(self):
            """Mock rollback."""
            pass

        def add(self, item):
            """Mock SQLAlchemy add method."""
            self._items.append(item)

        async def flush(self):
            """Mock SQLAlchemy flush method."""
            pass

        async def close(self):
            self._closed = True

    class MockEngine:
        def __init__(self):
            self._session = MockSession()

        async def begin(self):
            """Simulate SQLAlchemy engine begin."""
            class AsyncConnection:
                def __init__(self, session):
                    self.session = session
                    self.connection = None

                async def __aenter__(self):
                    self.connection = self.session
                    return self.connection

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self.connection:
                        await self.connection.close()

                async def close(self):
                    if self.connection:
                        await self.connection.close()

                async def execute(self, statement, params=None):
                    return await self.session.execute(statement, params)

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

    # Set up mock engine and session
    mock_engine = MockEngine()
    interface._engine = mock_engine
    interface._session_factory = lambda: mock_engine._session

    # Test basic operations without real connection
    assert interface._engine is not None
    assert isinstance(interface._engine, MockEngine)

    # Test data operations
    data = await interface.create(test_structured_data)
    assert data is not None

    read_data = await interface.read(test_structured_data.data_id)
    assert read_data is not None
    assert read_data.data_type == test_structured_data.data_type

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
async def test_model_serialization():
    """Test model serialization with proper type handling."""
    entity = EntitySymbol(
        symbol_id=UUID('12345678-1234-5678-1234-567812345678'),
        name="Test Entity",
        entity_type="TEST",
        descriptions=["Test Description"]
    )

    # Test Neo4j serialization
    neo4j_data = entity.to_neo4j()
    assert isinstance(neo4j_data['symbol_id'], str)

    # Test structured data serialization
    data = StructuredData(
        data_id=UUID('12345678-1234-5678-1234-567812345678'),
        data_type="test",
        data_value={"key": "value"}
    )
    assert isinstance(data.data_id, UUID)
    assert isinstance(data.model_dump()['data_id'], str)
