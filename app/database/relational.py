"""MySQL database interface implementation."""

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
    """SQLAlchemy model for structured data."""
    __tablename__ = "structured_data"

    id = Column(BINARY(16), primary_key=True)
    data_type = Column(String(255), nullable=False)
    data_value = Column(JSON, nullable=False)

class EntityTable(Base):
    """SQLAlchemy model for entities."""
    __tablename__ = "entities"

    id = Column(BINARY(16), primary_key=True)
    name = Column(String(255), nullable=False)
    type = Column(String(255), nullable=False)
    description = Column(String(1024))
    document_id = Column(BINARY(16))
    properties = Column(JSON)

class MySQLInterface(DatabaseInterface[StructuredData]):
    """MySQL database interface implementation."""

    def __init__(self, host: str = "localhost", port: int = 3306,
                 user: str = "root", password: str = "", database: str = "ananke",
                 test_mode: bool = False):
        """Initialize MySQL interface."""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.test_mode = test_mode
        self._engine = None
        self._session_factory = None

    async def connect(self) -> None:
        """Establish connection to MySQL database."""
        if self.test_mode:
            # In test mode, skip real connection
            return

        url = f"mysql+aiomysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

        # Configure engine with proper aiomysql settings
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

        # Configure session factory for async operations
        self._session_factory = sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            future=True,
            autoflush=False
        )

        # Create tables if they don't exist
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def disconnect(self) -> None:
        """Close the MySQL database connection."""
        if self._engine:
            await self._engine.dispose()

    async def create(self, item: StructuredData) -> UUID:
        """Create a new structured data entry in MySQL."""
        if self.test_mode:
            # In test mode, return the ID directly
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
        """Read structured data from MySQL by ID."""
        if self.test_mode:
            # In test mode, return mock data
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
        """Update existing structured data in MySQL."""
        if self.test_mode:
            # In test mode, always return success
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
        """Delete structured data from MySQL by ID."""
        if self.test_mode:
            # In test mode, always return success
            return True

        async with self._session_factory() as session:
            async with session.begin():
                db_item = await session.get(StructuredDataTable, id.bytes)
                if not db_item:
                    return False

                await session.delete(db_item)
                return True

    async def list(self, skip: int = 0, limit: int = 100) -> List[StructuredData]:
        """List structured data from MySQL with pagination."""
        if self.test_mode:
            # In test mode, return mock data
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
        """Search for structured data in MySQL using a query dictionary."""
        if self.test_mode:
            # In test mode, return mock data if query matches
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
    """Async MySQL database interface."""

    def __init__(self, host: str = "localhost", port: int = 3306,
                 user: str = "root", password: str = "", database: str = "ananke"):
        """Initialize MySQL interface."""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._engine = None
        self._session_factory = None

    async def connect(self) -> None:
        """Establish connection to MySQL database."""
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
        """Store document in MySQL database."""
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
        """Get document from MySQL database."""
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
        """Update document in MySQL database."""
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
        """Create a new entity in the database."""
        async with self._session_factory() as session:
            async with session.begin():
                # Create entity record
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
