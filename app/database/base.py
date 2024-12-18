"""Base database interface for Ananke2."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from uuid import UUID

T = TypeVar('T')

class DatabaseInterface(ABC, Generic[T]):
    """Abstract base class for database interfaces."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the database."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the database connection."""
        pass

    @abstractmethod
    async def create(self, item: T) -> UUID:
        """Create a new item in the database."""
        pass

    @abstractmethod
    async def read(self, id: UUID) -> Optional[T]:
        """Read an item from the database by ID."""
        pass

    @abstractmethod
    async def update(self, id: UUID, item: T) -> bool:
        """Update an existing item in the database."""
        pass

    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """Delete an item from the database by ID."""
        pass

    @abstractmethod
    async def list(self, skip: int = 0, limit: int = 100) -> List[T]:
        """List items from the database with pagination."""
        pass

    @abstractmethod
    async def search(self, query: Dict[str, Any]) -> List[T]:
        """Search for items in the database using a query dictionary."""
        pass
