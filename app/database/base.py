"""Base database interface for Ananke2 knowledge framework.

This module defines the abstract base interface for all database operations
in the Ananke2 framework. It provides a consistent API across different
storage backends (relational, graph, vector) through generic typing.

Example:
    >>> class Neo4jInterface(DatabaseInterface[Entity]):
    ...     async def connect(self) -> None:
    ...         # Implementation for Neo4j connection
    ...         pass
    ...     async def create(self, item: Entity) -> UUID:
    ...         # Implementation for Neo4j entity creation
    ...         return item.id

Note:
    All concrete implementations must handle their own connection lifecycle
    and implement proper error handling for their specific database type.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from uuid import UUID

T = TypeVar('T')

class DatabaseInterface(ABC, Generic[T]):
    """Abstract base class for database interfaces.

    This interface defines the standard CRUD operations and search capabilities
    that all database implementations must provide. The generic type T allows
    for type-safe implementations with different data models.

    Type Parameters:
        T: The type of items stored in the database (e.g., Entity, Document)
           Must be a valid Pydantic model or dataclass.

    Note:
        All async methods should implement proper error handling and connection
        management. Implementations should document their specific error cases.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the database.

        This method should handle all connection setup including:
        - Authentication
        - Connection pooling
        - Initial database/collection creation

        Raises:
            ConnectionError: If database connection fails
            ConfigError: If connection configuration is invalid
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the database connection.

        Ensures proper cleanup of database resources including:
        - Closing all active connections
        - Shutting down connection pools
        - Releasing any acquired locks

        Raises:
            ConnectionError: If disconnection fails
        """
        pass

    @abstractmethod
    async def create(self, item: T) -> UUID:
        """Create a new item in the database.

        Args:
            item (T): The item to create. Must be a valid instance of type T.

        Returns:
            UUID: The unique identifier of the created item.

        Raises:
            ValueError: If item is invalid
            DuplicateError: If item with same ID already exists
            ConnectionError: If database operation fails
        """
        pass

    @abstractmethod
    async def read(self, id: UUID) -> Optional[T]:
        """Read an item from the database by ID.

        Args:
            id (UUID): Unique identifier of the item to read

        Returns:
            Optional[T]: The found item or None if not found

        Raises:
            ValueError: If ID format is invalid
            ConnectionError: If database operation fails
        """
        pass

    @abstractmethod
    async def update(self, id: UUID, item: T) -> bool:
        """Update an existing item in the database.

        Args:
            id (UUID): Unique identifier of item to update
            item (T): New item data to apply

        Returns:
            bool: True if update successful, False if item not found

        Raises:
            ValueError: If ID format or item data is invalid
            ConnectionError: If database operation fails
        """
        pass

    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """Delete an item from the database by ID.

        Args:
            id (UUID): Unique identifier of item to delete

        Returns:
            bool: True if deletion successful, False if item not found

        Raises:
            ValueError: If ID format is invalid
            ConnectionError: If database operation fails
        """
        pass

    @abstractmethod
    async def list(self, skip: int = 0, limit: int = 100) -> List[T]:
        """List items from the database with pagination.

        Args:
            skip (int, optional): Number of items to skip. Defaults to 0.
            limit (int, optional): Maximum items to return. Defaults to 100.

        Returns:
            List[T]: List of items, may be empty if no items found

        Raises:
            ValueError: If skip or limit are negative
            ConnectionError: If database operation fails
        """
        pass

    @abstractmethod
    async def search(self, query: Dict[str, Any]) -> List[T]:
        """Search for items in the database using a query dictionary.

        Args:
            query (Dict[str, Any]): Search criteria as key-value pairs.
                Format depends on specific implementation.

        Returns:
            List[T]: List of matching items, may be empty if none found

        Raises:
            ValueError: If query format is invalid
            ConnectionError: If database operation fails
        """
        pass
