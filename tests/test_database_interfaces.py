"""Unit tests for database interfaces using mocks."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import UUID, uuid4
import numpy as np
from contextlib import ExitStack
from sqlalchemy.engine.cursor import CursorResult
from sqlalchemy.engine.result import ResultMetaData

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

@pytest.mark.unit
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
            'descriptions': entity_symbol.descriptions,
            'entity_type': entity_symbol.entity_type
        }
    })

    # Setup mock session
    mock_session.__aenter__.return_value = mock_session
    mock_session.run = AsyncMock(side_effect=lambda query, **kwargs: mock_result if 'CREATE' in query else mock_read_result)
    mock_driver.session.return_value = mock_session

    # Initialize interface with mock driver
    with patch('neo4j.AsyncGraphDatabase.driver', return_value=mock_driver):
        interface = Neo4jInterface(uri="bolt://localhost:7687", username="neo4j", password="password")
        await interface.connect()

        # Test create
        created_id = await interface.create(entity_symbol)
        assert created_id == entity_symbol.symbol_id

        # Test read
        read_data = await interface.read(entity_symbol.symbol_id)
        assert read_data is not None
        assert isinstance(read_data, EntitySymbol)
        assert read_data.symbol_id == entity_symbol.symbol_id

        await interface.disconnect()

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

@pytest.mark.unit
async def test_mysql_interface():
    """Test MySQL interface operations."""
    import asyncio
    from unittest.mock import AsyncMock, patch
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.pool import Pool

    async def async_func(f):
        return await f()

    # Create mock session factory
    class MockSessionFactory:
        """Mock SQLAlchemy session factory."""
        def __init__(self, *args, **kwargs):
            self.mock_session = AsyncMock(spec=AsyncSession)
            self.mock_session.execute = AsyncMock()
            self.mock_session.commit = AsyncMock()
            self.mock_session.close = AsyncMock()

        async def __call__(self):
            """Return mock session."""
            return self.mock_session

        async def __aenter__(self):
            """Enter context."""
            return await self()

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            """Exit context."""
            await self.mock_session.close()

    # Create mock cursor
    class MockCursor:
        """Mock cursor that matches aiomysql's Cursor interface."""
        async def __init__(self, connection=None, echo=False):
            self._connection = connection
            self._echo = echo
            self._description = None
            self._rows = []
            self._rowcount = 0
            self._lastrowid = None
            self._executed = None
            self._closed = False
            self.loop = asyncio.get_event_loop()

            # Mock result metadata
            class MockResultMetaData(ResultMetaData):
                """Mock result metadata."""
                def __init__(self):
                    self._keys = []  # Changed to private attribute
                    self._keymap = {}
                    self._processors = {}
                    self._key_to_index = {}
                    self._indexes = {}
                    self._keymap_by_result_key = {}
                    self._effective_processors = None
                    # Initialize with default column mapping
                    self._initialize_columns()

                @property
                def keys(self):
                    """Get the keys."""
                    return self._keys

                @keys.setter
                def keys(self, value):
                    """Set the keys."""
                    self._keys = value

                def _initialize_columns(self):
                    """Initialize column mapping."""
                    columns = [
                        ('id', 0),
                        ('data_type', 1),
                        ('data_value', 2)
                    ]
                    for key, idx in columns:
                        self._keys.append(key)
                        self._keymap[key] = idx
                        self._key_to_index[key] = idx
                        self._indexes[idx] = key

                def _indexes_for_keys(self, keys):
                    """Get indexes for the given keys."""
                    if isinstance(keys, (list, tuple)):
                        return [self._keymap[key.name if hasattr(key, 'name') else key]
                              for key in keys]
                    return [self._keymap[keys.name if hasattr(keys, 'name') else keys]]

                def _key_fallback(self, key, err=None):
                    """Fallback for key lookup."""
                    if isinstance(key, str):
                        return self._keymap.get(key)
                    elif hasattr(key, 'name'):
                        return self._keymap.get(key.name)
                    return None

                def _has_key(self, key):
                    """Check if key exists."""
                    key = self._key_fallback(key)
                    return key in self._keymap or key in self.keys

                def _contains_key(self, key):
                    """Check if key exists (SQLAlchemy interface method)."""
                    return self._has_key(key)

                def _key_to_index(self, key):
                    """Convert key to index."""
                    key = self._key_fallback(key)
                    return self._keymap.get(key, None)

            self._metadata = MockResultMetaData()
            self._result_metadata = self._metadata
            self._result = type('Result', (), {
                'affected_rows': 0,
                'description': None,
                'insert_id': None,
                'rows': [],
                'warning_count': 0,
                '_metadata': None  # Add metadata field
            })()
            self._result._metadata = self._metadata  # Set metadata on result
            self.echo = echo

        def __await__(self):
            """Make cursor awaitable."""
            async def _await():
                return self
            return _await().__await__()

        def _get_db(self):
            """Get the database connection."""
            return self._connection._connection

        async def __aenter__(self):
            """Async context manager entry."""
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            """Async context manager exit."""
            await self.close()

        def __iter__(self):
            """Make cursor iterable."""
            return iter(self._rows)

        def __aiter__(self):
            """Make cursor async iterable."""
            return self

        async def __anext__(self):
            """Get next row in async iteration."""
            row = await self.fetchone()
            if row is None:
                raise StopAsyncIteration
            return row

        @property
        def description(self):
            """Get the cursor's column description."""
            return self._description

        @property
        def rowcount(self):
            """Get the cursor rowcount."""
            return self._rowcount

        async def fetchall(self):
            """Fetch all rows."""
            return self._rows

        async def fetchone(self):
            """Fetch one row."""
            if self._row_index < len(self._rows):
                row = self._rows[self._row_index]
                self._row_index += 1
                return row
            return None

        async def execute(self, operation, parameters=None):
            """Execute a query."""
            return await self._execute_async(operation, parameters)

        def _get_result(self):
            """Get the result object."""
            if not self._result:
                class Result:
                    """Mock result object."""
                    def __init__(self, cursor):
                        self.cursor = cursor
                        self._metadata = cursor._metadata
                        self._rows = cursor._rows
                        self._rowcount = cursor._rowcount
                        self._description = cursor._description
                        self._row_index = 0
                        self._closed = False

                    async def _fetchone_impl(self):
                        """Fetch one row."""
                        if self._row_index < len(self._rows):
                            row = self._rows[self._row_index]
                            self._row_index += 1
                            return row
                        return None

                    async def _fetchall_impl(self):
                        """Fetch all rows."""
                        return self._rows[self._row_index:]

                    async def _soft_close(self):
                        """Close the result."""
                        self._closed = True

                self._result = Result(self)
            return self._result

        async def _execute_async(self, operation, parameters=None):
            """Execute a query asynchronously."""
            # Mock query execution
            self._rows = []
            self._rowcount = 0
            self._description = None

            # Parse operation to determine what kind of query it is
            operation = operation.strip().upper()

            if operation.startswith('SELECT'):
                # Mock SELECT query
                self._rows = [('test_id', 'test_name', 'test_value')]
                self._description = [
                    ('id', 'INTEGER', None, None, None, None, None),
                    ('name', 'VARCHAR', None, None, None, None, None),
                    ('value', 'VARCHAR', None, None, None, None, None)
                ]
                self._metadata._initialize_columns(['id', 'name', 'value'])
            elif operation.startswith('INSERT'):
                # Mock INSERT query
                self._rowcount = 1
                self._lastrowid = 1
            elif operation.startswith('UPDATE'):
                # Mock UPDATE query
                self._rowcount = 1
            elif operation.startswith('DELETE'):
                # Mock DELETE query
                self._rowcount = 1

            # Return self for method chaining
            return self

        async def close(self):
            """Close the cursor."""
            self._closed = True
            self._connection = None

    # Create mock DBAPI that matches aiomysql's interface
    class MockDBAPI:
        def __init__(self):
            self.paramstyle = 'format'
            self.threadsafety = 1
            self.apilevel = '2.0'

            # Create async cursor class
            class AsyncCursor:
                def __init__(self, *args, **kwargs):
                    self._cursor = None
                    self._args = args
                    self._kwargs = kwargs

                async def __aenter__(self):
                    if not self._cursor:
                        self._cursor = MockCursor(*self._args, **self._kwargs)
                        await self._cursor.__aenter__()
                    return self._cursor

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if self._cursor:
                        await self._cursor.__aexit__(exc_type, exc_val, exc_tb)
                        self._cursor = None

                def __await__(self):
                    async def _await():
                        return await self.__aenter__()
                    return _await().__await__()

            self.Cursor = AsyncCursor
            self.SSCursor = AsyncCursor

            # Standard DBAPI error hierarchy
            self.Error = type('Error', (Exception,), {})
            self.InterfaceError = type('InterfaceError', (self.Error,), {})
            self.DatabaseError = type('DatabaseError', (self.Error,), {})
            self.InternalError = type('InternalError', (self.DatabaseError,), {})
            self.OperationalError = type('OperationalError', (self.DatabaseError,), {})
            self.ProgrammingError = type('ProgrammingError', (self.DatabaseError,), {})
            self.IntegrityError = type('IntegrityError', (self.DatabaseError,), {})
            self.DataError = type('DataError', (self.DatabaseError,), {})
            self.NotSupportedError = type('NotSupportedError', (self.DatabaseError,), {})

            # Add SQLAlchemy required attributes
            self.STRING = 'STRING'
            self.NUMBER = 'NUMBER'
            self.DATETIME = 'DATETIME'
            self.BINARY = 'BINARY'
            self.TIMESTAMP = 'TIMESTAMP'
            self.Binary = lambda x: x
            self._init_dbapi_attributes()

        def _init_dbapi_attributes(self):
            """Initialize DBAPI attributes."""
            self.STRING = self.BINARY = self.NUMBER = self.DATETIME = self.ROWID = None
            self.Binary = lambda x: x

    # Create mock connection that properly handles cursor creation
    class MockConnection:
        """Mock connection that matches aiomysql's Connection interface."""
        def __init__(self, *args, **kwargs):
            self._cursor = None
            self.dbapi = MockDBAPI()
            self.loop = asyncio.get_event_loop()  # Changed from _loop to loop
            self.charset = 'utf8mb4'
            self.cursors = type('cursors', (), {'Cursor': MockCursor})
            self._result = type('Result', (), {
                'affected_rows': 0,
                'description': None,
                'insert_id': None,
                'rows': [],
                'warning_count': 0
            })()
            self.closed = False
            self.host = 'localhost'
            self.port = 3306
            self.user = 'root'
            self.db = 'test'
            self.echo = False
            self._autocommit = True

            async def await_(obj):
                """Handle awaiting coroutines and regular objects."""
                if hasattr(obj, '__await__'):
                    return await obj
                return obj

            self.await_ = await_

            # SQLAlchemy required attributes
            self.isolation_level = None
            self.default_isolation_level = "REPEATABLE READ"
            self._isolation_level = None
            self.in_transaction = False
            self.server_version_info = (8, 0, 0)
            self._server_version = "8.0.0"

            # Create inner connection that matches MySQL-Python's Connection interface
            class InnerConnection:
                def __init__(self, outer):
                    self.outer = outer
                    self.charset = outer.charset
                    self.encoding = 'utf8mb4'
                    self.server_version = '8.0.0'
                    self.loop = outer.loop  # Added loop attribute to inner connection
                    self._autocommit = True
                    self.isolation_level = None
                    self.in_transaction = False

                def character_set_name(self):
                    return self.charset

                def get_server_info(self):
                    return self.server_version

                def get_server_version_info(self):
                    return self.outer.server_version_info

                def get_server_version(self):
                    return self.outer._server_version

                async def ping(self, reconnect=True):
                    async def _async_return():
                        return None
                    return await _async_return()

                def set_character_set(self, charset):
                    self.charset = charset
                    self.outer.charset = charset

                async def autocommit(self, value):
                    self._autocommit = value
                    self.outer._autocommit = value
                    async def _async_return():
                        return self
                    return await _async_return()

                async def set_isolation_level(self, level):
                    self.isolation_level = level
                    self.outer.isolation_level = level
                    return None

                async def query(self, sql, *args):
                    cursor = self.outer.cursor()
                    await cursor._execute_async(sql, args if args else None)
                    return cursor._result

                async def rollback(self):
                    """Rollback the current transaction."""
                    self.in_transaction = False
                    self.outer.in_transaction = False
                    async def _async_return():
                        return self
                    return await _async_return()

                async def commit(self):
                    """Commit the current transaction."""
                    async def _async_return():
                        return None
                    return await _async_return()

            self._connection = InnerConnection(self)
            self.encoding = 'utf8mb4'
            self.server_version = '8.0.0'
            self._adapted_connection = None
            self._pool = None

        async def autocommit(self, value):
            """Set autocommit mode."""
            await self._connection.autocommit(value)
            self._autocommit = value
            async def _async_return():
                return self
            return await _async_return()

        async def query(self, sql, *args):
            """Execute a query directly on the connection."""
            if not hasattr(self, '_cursor') or self._cursor is None:
                self._cursor = await self.dbapi.Cursor(connection=self, echo=self.echo)
            return await self._cursor._execute_async(sql, args if args else None)

        async def cursor(self, cursor_class=None):
            """Create a new cursor object."""
            if cursor_class is None:
                cursor_class = self.dbapi.Cursor

            # Create our base cursor
            cursor = MockCursor(connection=self, echo=self.echo)
            await cursor.__aenter__()

            # If SQLAlchemy provides its own cursor class, wrap our mock cursor
            if hasattr(cursor_class, '_adapt_connection'):
                # Create an adapter that inherits from SQLAlchemy's cursor class
                adapted_cursor = type('AdaptedCursor', (cursor_class,), {
                    '_wrapped_cursor': cursor,
                    '_connection': self,
                    'await_': lambda self: self._wrapped_cursor,
                    'execute': lambda self, operation, parameters=None: self._wrapped_cursor._execute_async(operation, parameters),
                    '_execute_async': lambda self, operation, parameters=None: self._wrapped_cursor._execute_async(operation, parameters),
                    'fetchall': lambda self: self._wrapped_cursor.fetchall(),
                    'fetchone': lambda self: self._wrapped_cursor.fetchone(),
                    'close': lambda self: self._wrapped_cursor.close(),
                    'description': property(lambda self: self._wrapped_cursor._description),
                    'rowcount': property(lambda self: len(self._wrapped_cursor._rows)),
                    'arraysize': property(lambda self: 1),
                    'lastrowid': property(lambda self: None),
                    '__aiter__': lambda self: self._wrapped_cursor,
                    '__anext__': lambda self: self._wrapped_cursor.__anext__(),
                    '__await__': lambda self: self._wrapped_cursor.__await__(),
                    '__aenter__': lambda self: self._wrapped_cursor.__aenter__(),
                    '__aexit__': lambda self, *args: self._wrapped_cursor.__aexit__(*args)
                })()
                return adapted_cursor

            return cursor

        async def ensure_closed(self):
            """Ensure the connection is closed."""
            if self._cursor:
                await self._cursor.close()
            await self.close()
            self.closed = True

        async def _execute_async(self, operation, parameters=None):
            """Execute a query."""
            cursor = await self.cursor()
            return await cursor._execute_async(operation, parameters)

        async def run_sync(self, fn, *args, **kwargs):
            """Run a synchronous function."""
            return fn(*args, **kwargs)

        async def close(self):
            """Close the connection."""
            self.closed = True

        def character_set_name(self):
            """Return the character set name."""
            return self._connection.character_set_name()

        def get_server_info(self):
            """Return server version info."""
            return self._connection.get_server_info()

        def __await__(self):
            """Make connection awaitable."""
            async def _await():
                return self
            return _await().__await__()

        async def __aenter__(self):
            """Async context manager entry."""
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            """Async context manager exit."""
            if self._cursor:
                await self._cursor.close()
            await self.close()
            self.closed = True

        async def ping(self, reconnect=True):
            """Ping the server."""
            async def _async_return():
                return await self._connection.ping(reconnect)
            return await _async_return()

        def _set_autocommit(self, autocommit):
            """Set autocommit mode."""
            self.autocommit = autocommit

        @property
        def connection(self):
            """Get the underlying connection."""
            return self._connection

        def detach(self):
            """Detach from the current connection."""
            return self._connection

        def begin(self):
            """Begin a transaction."""
            return self

        async def commit(self):
            """Commit the current transaction."""
            return await self._connection.commit()

        async def rollback(self):
            """Rollback the current transaction."""
            return await self._connection.rollback()

        def set_character_set(self, charset):
            """Set the connection character set."""
            self._connection.set_character_set(charset)

    # Create mock engine
    class MockEngine:
        """Mock SQLAlchemy async engine."""
        def __init__(self, *args, **kwargs):
            self.url = "mysql+aiomysql://user:pass@localhost/test"
            self._connection = None
            self.dialect = type('Dialect', (), {
                'name': 'mysql',
                'dbapi': MockDBAPI()
            })()
            self._in_transaction = False
            self.pool = None
            # Create sync engine
            self.sync_engine = type('SyncEngine', (), {
                'url': self.url,
                'dialect': self.dialect,
                'pool': None,
                'connect': lambda: self.pool.connect() if self.pool else None
            })()

        async def setup(self):
            """Initialize the engine's pool."""
            self.pool = await self._create_pool()
            self.sync_engine.pool = self.pool

        async def _create_pool(self):
            """Create a mock connection pool."""
            pool = AsyncMock(spec=Pool)
            pool.connect = AsyncMock(return_value=await self._create_connection_fairy())
            pool._invoke_creator = AsyncMock(return_value=MockConnection())
            return pool

        async def _create_connection_fairy(self):
            """Create a mock connection fairy."""
            fairy = AsyncMock()
            fairy._connection = MockConnection()
            # Add required methods to the fairy
            fairy.__aenter__.return_value = fairy
            fairy.__aexit__.return_value = None
            return fairy

        async def connect(self):
            """Get a connection from the pool."""
            if not self._connection:
                self._connection = await self.pool.connect()
            return self._connection

        async def raw_connection(self):
            """Get a raw connection."""
            if not self._connection:
                self._connection = await self.pool.connect()
            return self._connection._connection

        async def dispose(self):
            """Dispose of the engine."""
            if self._connection:
                await self._connection._connection.ensure_closed()

        def begin(self):
            """Begin a transaction."""
            return self.BeginTransaction(self)

        class BeginTransaction:
            def __init__(self, engine):
                self.engine = engine
                self._connection = None

            async def __aenter__(self):
                self.engine._in_transaction = True
                self._connection = await self.engine.connect()
                return self._connection

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.engine._in_transaction = False
                if self._connection:
                    await self._connection._connection.ensure_closed()

    # Create mock connection
    mock_connection = MockConnection()

    # Create mock fairy using AsyncMock
    mock_fairy = AsyncMock()
    mock_fairy._connection = mock_connection
    mock_fairy.__aenter__ = AsyncMock(return_value=mock_fairy)
    mock_fairy.__aexit__ = AsyncMock()

    # Create mock session and result
    mock_session = AsyncMock(spec=AsyncSession)
    mock_result = AsyncMock()
    mock_result.scalar.return_value = None
    mock_session.execute.return_value = mock_result
    mock_session.commit = AsyncMock()
    mock_session.close = AsyncMock()

    # Create mock pool
    mock_pool = AsyncMock(spec=Pool)
    mock_pool.connect = AsyncMock(return_value=mock_fairy)
    mock_pool._invoke_creator = AsyncMock(return_value=mock_connection)

    # Create mock engine
    mock_engine = MockEngine()

    # Initialize MySQL interface with mocked components
    with patch('sqlalchemy.ext.asyncio.create_async_engine', return_value=mock_engine), \
         patch('sqlalchemy.orm.sessionmaker', return_value=MockSessionFactory), \
         patch('aiomysql.connect', new_callable=AsyncMock) as mock_aiomysql_connect:
        # Configure aiomysql mock to return our mock connection
        mock_aiomysql_connect.return_value = mock_connection

        interface = MySQLInterface(
            host="localhost",
            port=3306,
            user="root",
            password="password",
            database="test"
        )

        # Setup the mock engine
        await mock_engine.setup()

        # Mock the connection pool's connect method to return a working connection
        mock_engine.pool.connect = AsyncMock(return_value=mock_fairy)
        mock_engine.pool._invoke_creator = AsyncMock(return_value=mock_connection)

        # Test connection
        await interface.connect()

        # Test create operation
        test_data = StructuredData(
            data_id=UUID("12345678-1234-5678-1234-567812345678"),
            data_type="test_type",
            data_value={"name": "test", "value": 42}
        )
        created_id = await interface.create(test_data)

        # Verify create operation
        mock_result = AsyncMock()
        mock_result.fetchone.return_value = test_data
        mock_session.execute.return_value = mock_result
        result = await interface.read(UUID("12345678-1234-5678-1234-567812345678"))
        assert result == test_data

        # Test update operation
        updated_data = {"id": "123", "name": "updated", "value": 43}
        await interface.update("test_table", "123", updated_data)

        # Verify update operation
        mock_result = AsyncMock()
        mock_result.fetchone.return_value = updated_data
        mock_session.execute.return_value = mock_result
        result = await interface.read("test_table", "123")
        assert result == updated_data

        # Test delete operation
        await interface.delete("test_table", "123")

        # Verify delete operation
        mock_result = AsyncMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result
        result = await interface.read("test_table", "123")
        assert result is None

        # Test list operation
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = [test_data, updated_data]
        mock_session.execute.return_value = mock_result
        results = await interface.list("test_table")
        assert len(results) == 2
        assert test_data in results
        assert updated_data in results

        # Test cleanup
        await interface.dispose()
        await mock_engine.dispose()

    # Mock aiomysql.connect to return our mock connection
    async def mock_connect(*args, **kwargs):
        return mock_connection

    # Initialize MySQL interface with mocked components
    with patch('sqlalchemy.ext.asyncio.create_async_engine', return_value=mock_engine), \
         patch('sqlalchemy.orm.sessionmaker', return_value=MockSessionFactory), \
         patch('aiomysql.connect', mock_connect):

        interface = MySQLInterface(
            host="localhost",
            port=3306,
            user="root",
            password="password",
            database="test"
        )

        await interface.connect()

        # Test creating an entity
        entity = EntitySymbol(
            symbol_id=uuid4(),
            name="test_entity",
            descriptions=["test description"],
            semantics=[],
            properties=[
                StructuredData(
                    data_id=uuid4(),
                    data_type="property",
                    data_value={"key": "value"}
                )
            ],
            labels=[
                StructuredData(
                    data_id=uuid4(),
                    data_type="label",
                    data_value={"category": "test"}
                )
            ]
        )

        # Test CRUD operations
        await interface.create(entity)
        mock_session.execute.assert_called()
        mock_session.commit.assert_called()

        # Reset mocks
        mock_session.execute.reset_mock()
        mock_session.commit.reset_mock()

        # Test read operation
        mock_result.scalar.return_value = entity
        result = await interface.read(entity.symbol_id, EntitySymbol)
        assert result == entity
        mock_session.execute.assert_called()

        await interface.disconnect()
        mock_engine.dispose.assert_called_once()

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
