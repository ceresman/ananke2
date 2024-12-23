"""MySQL database interface implementation for Ananke2 knowledge framework.

This module provides MySQL implementations for structured data storage using SQLAlchemy ORM:
- MySQLInterface: Implements DatabaseInterface for structured data
- AsyncRelationalDatabase: Specialized interface for documents and entities
- SQLAlchemy models for data mapping

Features:
- Async/await support using aiomysql
- Connection pooling and transaction management
- Test mode for development without real database
- UUID handling with byte storage
- JSON column support for flexible data storage

Example:
    >>> interface = MySQLInterface(
    ...     host="localhost",
    ...     user="ananke",
    ...     password="secret",
    ...     database="knowledge"
    ... )
    >>> await interface.connect()
    >>> data = StructuredData(data_type="document", data_value={"text": "content"})
    >>> data_id = await interface.create(data)
    >>> await interface.disconnect()
"""

from typing import Any, Dict, List, Optional
from uuid import UUID
import aiomysql
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, JSON, select
from sqlalchemy.dialects.mysql import BINARY
import asyncio

from .base import DatabaseInterface
from ..models.structured import StructuredData
from ..models.entities import Entity

Base = declarative_base()

class StructuredDataTable(Base):
    """SQLAlchemy model for structured data storage.

    Maps structured data to MySQL table with UUID primary key and JSON value column.
    Used by both MySQLInterface and AsyncRelationalDatabase for data storage.

    Attributes:
        id (BINARY): UUID primary key stored as 16 bytes
        data_type (str): Type identifier for the stored data
        data_value (JSON): Flexible JSON storage for structured data
    """
    __tablename__ = "structured_data"

    id = Column(BINARY(16), primary_key=True)
    data_type = Column(String(255), nullable=False)
    data_value = Column(JSON, nullable=False)

class EntityTable(Base):
    """SQLAlchemy model for entity storage.

    Maps knowledge graph entities to MySQL table with properties and relationships.
    Primarily used by AsyncRelationalDatabase for entity management.

    Attributes:
        id (BINARY): UUID primary key stored as 16 bytes
        name (str): Entity name
        type (str): Entity type classification
        description (str): Optional entity description
        document_id (BINARY): Optional reference to source document
        properties (JSON): Flexible storage for entity properties
    """
    __tablename__ = "entities"

    id = Column(BINARY(16), primary_key=True)
    name = Column(String(255), nullable=False)
    type = Column(String(255), nullable=False)
    description = Column(String(1024))
    document_id = Column(BINARY(16))
    properties = Column(JSON)

class MySQLInterface(DatabaseInterface[StructuredData]):
    """MySQL implementation of DatabaseInterface for structured data storage.

    Provides CRUD operations for structured data using SQLAlchemy ORM with
    connection pooling, transaction management, and test mode support.

    Args:
        host (str, optional): MySQL server hostname. Defaults to "localhost".
        port (int, optional): MySQL server port. Defaults to 3306.
        user (str, optional): Database username. Defaults to "root".
        password (str, optional): Database password. Defaults to "".
        database (str, optional): Database name. Defaults to "ananke".
        test_mode (bool, optional): Enable test mode. Defaults to False.

    Attributes:
        host (str): MySQL server hostname
        port (int): MySQL server port
        user (str): Database username
        password (str): Database password
        database (str): Database name
        test_mode (bool): Whether test mode is enabled
        _engine: SQLAlchemy engine instance
        _session_factory: SQLAlchemy session factory
    """

    def __init__(self, host: str = "localhost", port: int = 3306,
                 user: str = "root", password: str = "", database: str = "ananke",
                 test_mode: bool = False):
        """Initialize MySQL interface with connection parameters."""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.test_mode = test_mode
        self._engine = None
        self._session_factory = None

    async def connect(self) -> None:
        """Establish connection to MySQL with connection pooling.

        Sets up SQLAlchemy engine with optimized pool settings and creates
        session factory for transaction management. In test mode, skips
        actual connection.

        Raises:
            SQLAlchemyError: If database connection fails
            Exception: For other connection errors
        """
        if self.test_mode:
            return

        url = f"mysql+aiomysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

        self._engine = create_async_engine(
            url,
            echo=True,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=20,
            max_overflow=0,
            pool_timeout=30,
            future=True,
            isolation_level="READ COMMITTED"
        )

        self._session_factory = sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            future=True,
            autoflush=False
        )

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def disconnect(self) -> None:
        """Disconnect from MySQL database and cleanup resources.

        Properly closes the database connection and cleans up the driver
        instance. Safe to call multiple times.
        """
        if self._engine:
            await self._engine.dispose()
            self._engine = None

    async def create(self, item: StructuredData) -> UUID:
        """Create new structured data entry with transaction.

        Args:
            item (StructuredData): Data to store in database

        Returns:
            UUID: Unique identifier of created data

        Raises:
            SQLAlchemyError: If database operation fails
            Exception: For other creation errors
        """
        if self.test_mode:
            return item.data_id

        async with self._session_factory() as session:
            async with session.begin():
                try:
                    db_item = StructuredDataTable(
                        id=item.data_id.bytes,
                        data_type=item.data_type,
                        data_value=item.data_value
                    )
                    session.add(db_item)
                    await session.commit()
                    return item.data_id
                except Exception as e:
                    print(f"Error creating structured data in MySQL: {str(e)}")
                    await session.rollback()
                    raise

    async def read(self, id: UUID) -> Optional[StructuredData]:
        """Read structured data by ID with error handling.

        Args:
            id (UUID): Unique identifier of data to read

        Returns:
            Optional[StructuredData]: Found data or None if not found

        Raises:
            SQLAlchemyError: If database operation fails
            Exception: For other read errors
        """
        if self.test_mode:
            return StructuredData(
                data_id=id,
                data_type="test_type",
                data_value={"test": "value"}
            )

        async with self._session_factory() as session:
            async with session.begin():
                stmt = select(StructuredDataTable).where(StructuredDataTable.id == id.bytes)
                result = await session.execute(stmt)
                db_item = result.scalar_one_or_none()

                if not db_item:
                    return None

                return StructuredData(
                    data_id=UUID(bytes=db_item.id),
                    data_type=db_item.data_type,
                    data_value=db_item.data_value
                )

    async def update(self, id: UUID, item: StructuredData) -> bool:
        """Update existing structured data with transaction.

        Args:
            id (UUID): Unique identifier of data to update
            item (StructuredData): New data to store

        Returns:
            bool: True if data was found and updated

        Raises:
            SQLAlchemyError: If database operation fails
            Exception: For other update errors
        """
        if self.test_mode:
            return True

        async with self._session_factory() as session:
            async with session.begin():
                db_item = await session.get(StructuredDataTable, id.bytes)
                if not db_item:
                    return False

                db_item.data_type = item.data_type
                db_item.data_value = item.data_value
                return True

    async def delete(self, id: UUID) -> bool:
        """Delete structured data by ID with transaction.

        Args:
            id (UUID): Unique identifier of data to delete

        Returns:
            bool: True if data was found and deleted

        Raises:
            SQLAlchemyError: If database operation fails
            Exception: For other deletion errors
        """
        if self.test_mode:
            return True

        async with self._session_factory() as session:
            async with session.begin():
                db_item = await session.get(StructuredDataTable, id.bytes)
                if not db_item:
                    return False

                await session.delete(db_item)
                return True

    async def list(self, skip: int = 0, limit: int = 100) -> List[StructuredData]:
        """List structured data with pagination support.

        Args:
            skip (int, optional): Number of records to skip. Defaults to 0.
            limit (int, optional): Maximum records to return. Defaults to 100.

        Returns:
            List[StructuredData]: List of found records

        Raises:
            SQLAlchemyError: If database query fails
            ValueError: If skip/limit are invalid
        """
        if self.test_mode:
            return [
                StructuredData(
                    data_id=UUID(int=1),
                    data_type="test_type",
                    data_value={"test": "value"}
                )
            ]

        async with self._session_factory() as session:
            async with session.begin():
                stmt = select(StructuredDataTable).offset(skip).limit(limit)
                result = await session.execute(stmt)
                rows = result.scalars().all()

                return [
                    StructuredData(
                        data_id=UUID(bytes=row.id),
                        data_type=row.data_type,
                        data_value=row.data_value
                    )
                    for row in rows
                ]

    async def search(self, query: Dict[str, Any]) -> List[StructuredData]:
        """Search structured data using property matching.

        Args:
            query (Dict[str, Any]): Search criteria as property key-value pairs

        Returns:
            List[StructuredData]: List of matching records

        Raises:
            SQLAlchemyError: If database query fails
            ValueError: If query format is invalid
        """
        if self.test_mode:
            if "data_type" in query and query["data_type"] == "test_type":
                return [
                    StructuredData(
                        data_id=UUID(int=1),
                        data_type="test_type",
                        data_value={"test": "value"}
                    )
                ]
            return []

        conditions = []

        if "data_type" in query:
            conditions.append(StructuredDataTable.data_type == query["data_type"])

        async with self._session_factory() as session:
            async with session.begin():
                stmt = select(StructuredDataTable)
                for condition in conditions:
                    stmt = stmt.where(condition)

                result = await session.execute(stmt)
                rows = result.scalars().all()

                return [
                    StructuredData(
                        data_id=UUID(bytes=row.id),
                        data_type=row.data_type,
                        data_value=row.data_value
                    )
                    for row in rows
                ]

class AsyncRelationalDatabase:
    """Specialized async MySQL interface for document and entity storage.

    Provides optimized operations for document processing workflow:
    - Document storage and retrieval
    - Entity creation and management
    - Relationship tracking
    - Transaction management

    Args:
        host (str, optional): MySQL server hostname. Defaults to "localhost".
        port (int, optional): MySQL server port. Defaults to 3306.
        user (str, optional): Database username. Defaults to "root".
        password (str, optional): Database password. Defaults to "".
        database (str, optional): Database name. Defaults to "ananke".

    Attributes:
        host (str): MySQL server hostname
        port (int): MySQL server port
        user (str): Database username
        password (str): Database password
        database (str): Database name
        _engine: SQLAlchemy engine instance
        _session_factory: SQLAlchemy session factory
    """

    def __init__(self, host: str = "localhost", port: int = 3306,
                 user: str = "root", password: str = "", database: str = "ananke"):
        """Initialize MySQL interface with connection parameters."""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._engine = None
        self._session_factory = None

    async def connect(self) -> None:
        """Establish connection to MySQL with connection pooling.

        Sets up SQLAlchemy engine and session factory for transaction
        management with optimized pool settings.

        Raises:
            SQLAlchemyError: If database connection fails
            Exception: For other connection errors
        """
        url = f"mysql+aiomysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        self._engine = create_async_engine(url, echo=True)
        self._session_factory = sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def disconnect(self) -> None:
        """Close the MySQL database connection."""
        if self._engine:
            await self._engine.dispose()

    async def create(self, item: StructuredData) -> UUID:
        """Create a new structured data entry."""
        async with self._session_factory() as session:
            async with session.begin():
                db_item = StructuredDataTable(
                    id=item.data_id.bytes,
                    data_type=item.data_type,
                    data_value=item.data_value
                )
                session.add(db_item)
                await session.commit()
                return item.data_id

    async def store_document(self, doc_data: Dict[str, Any]) -> str:
        """Store document data with automatic transaction handling.

        Args:
            doc_data (Dict[str, Any]): Document data including:
                - data_id: UUID of document
                - data_type: Type of document
                - data_value: Document content and metadata

        Returns:
            str: String representation of document UUID

        Raises:
            SQLAlchemyError: If database operation fails
            KeyError: If required fields are missing
        """
        async with self._session_factory() as session:
            async with session.begin():
                db_item = StructuredDataTable(
                    id=doc_data["data_id"].bytes,
                    data_type=doc_data["data_type"],
                    data_value=doc_data["data_value"]
                )
                session.add(db_item)
                await session.commit()
                return str(doc_data["data_id"])

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID with error handling.

        Args:
            doc_id (str): UUID string of document to retrieve

        Returns:
            Optional[Dict[str, Any]]: Document data or None if not found

        Raises:
            SQLAlchemyError: If database operation fails
            ValueError: If doc_id format is invalid
        """
        async with self._session_factory() as session:
            async with session.begin():
                stmt = select(StructuredDataTable).where(StructuredDataTable.id == UUID(doc_id).bytes)
                result = await session.execute(stmt)
                db_item = result.scalar_one_or_none()
                if not db_item:
                    return None
                return {
                    "id": str(UUID(bytes=db_item.id)),
                    "type": db_item.data_type,
                    "value": db_item.data_value
                }

    async def update_document(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing document with partial updates.

        Args:
            doc_id (str): UUID string of document to update
            updates (Dict[str, Any]): Fields to update in document

        Returns:
            bool: True if document was found and updated

        Raises:
            SQLAlchemyError: If database operation fails
            ValueError: If doc_id format is invalid
        """
        async with self._session_factory() as session:
            async with session.begin():
                stmt = select(StructuredDataTable).where(StructuredDataTable.id == UUID(doc_id).bytes)
                result = await session.execute(stmt)
                db_item = result.scalar_one_or_none()
                if not db_item:
                    return False
                db_item.data_value.update(updates)
                await session.commit()
                return True

    async def create_entity(self, entity: Entity) -> Dict[str, Any]:
        """Create new entity with relationship tracking.

        Args:
            entity (Entity): Entity to create with properties and relationships

        Returns:
            Dict[str, Any]: Created entity data with ID

        Raises:
            SQLAlchemyError: If database operation fails
            ValueError: If entity data is invalid
        """
        async with self._session_factory() as session:
            async with session.begin():
                db_entity = EntityTable(
                    id=entity.id.bytes,
                    name=entity.name,
                    type=entity.type,
                    description=entity.description,
                    document_id=entity.document_id.bytes if entity.document_id else None,
                    properties=entity.properties
                )
                session.add(db_entity)
                await session.commit()
                return {"id": str(entity.id)}
