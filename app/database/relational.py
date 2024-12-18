"""MySQL database interface implementation."""

from typing import Any, Dict, List, Optional
from uuid import UUID
import aiomysql
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, JSON
from sqlalchemy.dialects.mysql import BINARY
from sqlalchemy.ext.declarative import declarative_base

from .base import DatabaseInterface
from ..models.structured import StructuredData

Base = declarative_base()

class StructuredDataTable(Base):
    """SQLAlchemy model for structured data."""
    __tablename__ = "structured_data"

    id = Column(BINARY(16), primary_key=True)
    data_type = Column(String(255), nullable=False)
    data_value = Column(JSON, nullable=False)

class MySQLInterface(DatabaseInterface[StructuredData]):
    """MySQL database interface implementation."""

    def __init__(self, host: str, port: int, user: str, password: str, database: str):
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
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

        # Create tables in a transaction
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def disconnect(self) -> None:
        """Close the MySQL database connection."""
        if self._engine:
            await self._engine.dispose()

    async def create(self, item: StructuredData) -> UUID:
        """Create a new structured data entry in MySQL."""
        async with self._session_factory() as session:
            db_item = StructuredDataTable(
                id=item.data_id.bytes,
                data_type=item.data_type,
                data_value=item.data_value
            )
            session.add(db_item)
            await session.commit()
            return item.data_id

    async def read(self, id: UUID) -> Optional[StructuredData]:
        """Read structured data from MySQL by ID."""
        async with self._session_factory() as session:
            result = await session.get(StructuredDataTable, id.bytes)
            if not result:
                return None

            return StructuredData(
                data_id=UUID(bytes=result.id),
                data_type=result.data_type,
                data_value=result.data_value
            )

    async def update(self, id: UUID, item: StructuredData) -> bool:
        """Update existing structured data in MySQL."""
        async with self._session_factory() as session:
            db_item = await session.get(StructuredDataTable, id.bytes)
            if not db_item:
                return False

            db_item.data_type = item.data_type
            db_item.data_value = item.data_value
            await session.commit()
            return True

    async def delete(self, id: UUID) -> bool:
        """Delete structured data from MySQL by ID."""
        async with self._session_factory() as session:
            db_item = await session.get(StructuredDataTable, id.bytes)
            if not db_item:
                return False

            await session.delete(db_item)
            await session.commit()
            return True

    async def list(self, skip: int = 0, limit: int = 100) -> List[StructuredData]:
        """List structured data from MySQL with pagination."""
        async with self._session_factory() as session:
            result = await session.execute(
                StructuredDataTable.select()
                .offset(skip)
                .limit(limit)
            )
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
        conditions = []
        values = {}

        if "data_type" in query:
            conditions.append(StructuredDataTable.data_type == query["data_type"])

        async with self._session_factory() as session:
            stmt = StructuredDataTable.select()
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
