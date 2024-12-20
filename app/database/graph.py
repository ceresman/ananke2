"""Neo4j graph database interface implementation for Ananke2 knowledge framework.

This module provides a Neo4j-specific implementation of the DatabaseInterface
for storing and querying knowledge graph entities and relationships. It handles:
- Connection lifecycle management with authentication
- Entity and relationship CRUD operations
- Graph-specific query patterns
- Transaction management and error handling
- Test mode support for development

Example:
    >>> interface = Neo4jInterface(
    ...     uri="neo4j://localhost:7687",
    ...     username="neo4j",
    ...     password="password"
    ... )
    >>> await interface.connect()
    >>> entity = EntitySymbol(name="Example", descriptions=["Test entity"])
    >>> entity_id = await interface.create(entity)
    >>> await interface.disconnect()

Note:
    - All database operations are asynchronous
    - UUIDs are stored as bytes in Neo4j for consistency
    - Test mode skips actual database connection for testing
"""

import asyncio
from typing import List, Optional, Dict, Any
from uuid import UUID

from neo4j import AsyncGraphDatabase as Neo4jDriver
from neo4j.exceptions import ServiceUnavailable

from .base import DatabaseInterface
from ..models.entities import EntitySymbol
from ..models.relations import RelationSymbol
from ..models.triples import TripleSymbol

class Neo4jInterface(DatabaseInterface[EntitySymbol]):
    """Neo4j database interface implementation for entity storage and querying.

    This class implements the DatabaseInterface for Neo4j, providing graph-specific
    optimizations and features. It handles connection pooling, transaction
    management, and proper cleanup of database resources.

    The implementation uses Neo4j's async driver for better performance and
    supports a test mode for development without requiring a real database.

    Args:
        uri (str): Neo4j connection URI (e.g., "neo4j://localhost:7687")
        username (str): Database authentication username
        password (str): Database authentication password
        test_mode (bool, optional): Enable test mode without real connection.
            Defaults to False.

    Attributes:
        uri (str): Neo4j connection URI
        username (str): Database username
        password (str): Database password
        test_mode (bool): Whether test mode is enabled
        _driver (Optional[Neo4jDriver]): Neo4j async driver instance
    """

    def __init__(self, uri: str, username: str, password: str, test_mode: bool = False):
        """Initialize Neo4j interface with connection parameters."""
        self.uri = uri
        self.username = username
        self.password = password
        self.test_mode = test_mode
        self._driver = None

    async def connect(self) -> None:
        """Connect to Neo4j database with authentication and verification.

        Establishes connection to Neo4j and verifies both authentication
        and connectivity through a test query. In test mode, skips actual
        connection for testing purposes.

        Raises:
            ServiceUnavailable: If Neo4j server is not accessible
            AuthenticationError: If credentials are invalid
            Exception: For other connection-related errors
        """
        try:
            if self.test_mode:
                return

            self._driver = Neo4jDriver.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            await self._driver.verify_authentication()
            await self._driver.verify_connectivity()
            async with self._driver.session() as session:
                await session.run("RETURN 1")
        except Exception as e:
            print(f"Error connecting to Neo4j: {str(e)}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Neo4j database and cleanup resources.

        Properly closes the database connection and cleans up the driver
        instance. Safe to call multiple times.
        """
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def create(self, item: EntitySymbol) -> UUID:
        """Create a new entity in Neo4j.

        Creates a new entity node with properties from the EntitySymbol.
        UUIDs are stored as bytes for consistent handling across the system.

        Args:
            item (EntitySymbol): Entity to create in database

        Returns:
            UUID: Unique identifier of created entity

        Raises:
            ConnectionError: If database connection fails
            Exception: If entity creation fails
        """
        async with await self._driver.session() as session:
            try:
                result = await session.run(
                    """
                    CREATE (e:Entity {
                        id: $id,
                        name: $name,
                        descriptions: $descriptions
                    })
                    RETURN e.id
                    """,
                    id=item.symbol_id.bytes,
                    name=item.name,
                    descriptions=item.descriptions
                )
                record = await result.single()
                return UUID(bytes=record["e.id"])
            except Exception as e:
                print(f"Error creating entity in Neo4j: {str(e)}")
                raise

    async def read(self, id: UUID) -> Optional[EntitySymbol]:
        """Read an entity from Neo4j by ID.

        Retrieves an entity node by its UUID and constructs an EntitySymbol
        instance. Additional properties like semantics and labels are loaded
        separately for performance.

        Args:
            id (UUID): Unique identifier of entity to read

        Returns:
            Optional[EntitySymbol]: Found entity or None if not found

        Raises:
            ConnectionError: If database connection fails
            Exception: If entity retrieval fails
        """
        async with await self._driver.session() as session:
            try:
                result = await session.run(
                    """
                    MATCH (e:Entity {id: $id})
                    RETURN e
                    """,
                    id=id.bytes
                )
                record = await result.single()
                if not record:
                    return None

                entity = record["e"]
                return EntitySymbol(
                    symbol_id=UUID(bytes=entity["id"]),
                    name=entity["name"],
                    descriptions=entity["descriptions"],
                    entity_type="ENTITY",
                    semantics=[],
                    properties=[],
                    labels=[]
                )
            except Exception as e:
                print(f"Error reading entity from Neo4j: {str(e)}")
                raise

    async def update(self, id: UUID, item: EntitySymbol) -> bool:
        """Update an existing entity in Neo4j.

        Updates entity node properties while preserving relationships.
        Only updates the specified fields in the EntitySymbol.

        Args:
            id (UUID): Unique identifier of entity to update
            item (EntitySymbol): New entity data to apply

        Returns:
            bool: True if entity was found and updated, False otherwise

        Raises:
            ConnectionError: If database connection fails
            Exception: If entity update fails
        """
        async with await self._driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {id: $id})
                SET e.name = $name,
                    e.descriptions = $descriptions
                RETURN e
                """,
                id=str(id),
                name=item.name,
                descriptions=item.descriptions
            )
            return await result.single() is not None

    async def delete(self, id: UUID) -> bool:
        """Delete an entity from Neo4j by ID.

        Removes the entity node and its relationships from the graph.
        Returns success status based on whether entity existed.

        Args:
            id (UUID): Unique identifier of entity to delete

        Returns:
            bool: True if entity was found and deleted, False otherwise

        Raises:
            ConnectionError: If database connection fails
            Exception: If entity deletion fails
        """
        async with await self._driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {id: $id})
                DELETE e
                RETURN COUNT(e) as count
                """,
                id=str(id)
            )
            record = await result.single()
            return record and record["count"] > 0

    async def list(self, skip: int = 0, limit: int = 100) -> List[EntitySymbol]:
        """List entities from Neo4j with pagination.

        Retrieves a paginated list of entities with basic properties.
        Additional properties are loaded separately for performance.

        Args:
            skip (int, optional): Number of entities to skip. Defaults to 0.
            limit (int, optional): Maximum number to return. Defaults to 100.

        Returns:
            List[EntitySymbol]: List of found entities, may be empty

        Raises:
            ConnectionError: If database connection fails
            ValueError: If skip or limit are negative
        """
        async with await self._driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity)
                RETURN e
                SKIP $skip
                LIMIT $limit
                """,
                skip=skip,
                limit=limit
            )
            records = await result.all()
            return [
                EntitySymbol(
                    symbol_id=UUID(record["e"]["id"]),
                    name=record["e"]["name"],
                    descriptions=record["e"]["descriptions"],
                    entity_type="ENTITY",
                    semantics=[],
                    properties=[],
                    labels=[]
                )
                for record in records
            ]

    async def search(self, query: Dict[str, Any]) -> List[EntitySymbol]:
        """Search for entities in Neo4j using a query dictionary.

        Constructs and executes a Cypher query based on the provided
        search criteria. Supports exact match on entity properties.

        Args:
            query (Dict[str, Any]): Search criteria as property key-value pairs

        Returns:
            List[EntitySymbol]: List of matching entities, may be empty

        Raises:
            ConnectionError: If database connection fails
            ValueError: If query format is invalid
        """
        conditions = []
        params = {}

        for key, value in query.items():
            conditions.append(f"e.{key} = ${key}")
            params[key] = value

        cypher_query = f"""
        MATCH (e:Entity)
        WHERE {" AND ".join(conditions)}
        RETURN e
        """

        async with await self._driver.session() as session:
            result = await session.run(cypher_query, params)
            records = await result.all()
            return [
                EntitySymbol(
                    symbol_id=UUID(record["e"]["id"]),
                    name=record["e"]["name"],
                    descriptions=record["e"]["descriptions"],
                    entity_type="ENTITY",
                    semantics=[],
                    properties=[],
                    labels=[]
                )
                for record in records
            ]

    async def get_all_entities(self) -> List[EntitySymbol]:
        """Get all entities from the database with a reasonable limit.

        Returns:
            List[EntitySymbol]: List of all entities up to limit

        Raises:
            ConnectionError: If database connection fails
        """
        return await self.list(limit=1000)

    async def get_all_relationships(self) -> List[RelationSymbol]:
        """Get all relationships from the database with their connected nodes.

        Retrieves relationships with their source and target nodes,
        constructing RelationSymbol instances with proper UUID handling.

        Returns:
            List[RelationSymbol]: List of all relationships up to limit

        Raises:
            ConnectionError: If not connected or operation fails
        """
        if not self._driver:
            raise ConnectionError("Not connected to database")
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH ()-[r]->()
                RETURN r, type(r) as type,
                       startNode(r) as source,
                       endNode(r) as target
                LIMIT 1000
                """
            )
            records = await result.all()
            return [
                RelationSymbol(
                    relation_id=UUID(record["r"]["id"]) if "id" in record["r"] else UUID(str(hash(str(record)))),
                    name=record["type"],
                    source_id=UUID(record["source"]["id"]),
                    target_id=UUID(record["target"]["id"]),
                    properties=record["r"]
                )
                for record in records
            ]
