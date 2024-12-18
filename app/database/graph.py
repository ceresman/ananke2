"""Neo4j database interface implementation."""

from typing import Any, Dict, List, Optional
from uuid import UUID
import neo4j
from neo4j import AsyncGraphDatabase

from .base import DatabaseInterface
from ..models.entities import EntitySymbol
from ..models.relations import RelationSymbol
from ..models.triples import TripleSymbol

class AsyncGraphDatabase(DatabaseInterface[EntitySymbol]):
    """Neo4j database interface implementation."""

    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j interface."""
        self.uri = uri
        self.username = username
        self.password = password
        self._driver = None

    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        self._driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )

    async def disconnect(self) -> None:
        """Close the Neo4j database connection."""
        if self._driver:
            await self._driver.close()

    async def create(self, item: EntitySymbol) -> UUID:
        """Create a new entity in Neo4j."""
        session = await self._driver.session()
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
                id=str(item.symbol_id),
                name=item.name,
                descriptions=item.descriptions
            )
            record = await result.single()
            return UUID(record["e.id"])
        finally:
            await session.close()

    async def read(self, id: UUID) -> Optional[EntitySymbol]:
        """Read an entity from Neo4j by ID."""
        session = await self._driver.session()
        try:
            result = await session.run(
                """
                MATCH (e:Entity {id: $id})
                RETURN e
                """,
                id=str(id)
            )
            record = await result.single()
            if not record:
                return None

            entity = record["e"]
            return EntitySymbol(
                symbol_id=UUID(entity["id"]),
                name=entity["name"],
                descriptions=entity["descriptions"],
                entity_type="ENTITY",  # Add default entity type
                semantics=[],  # Load semantics separately
                properties=[],  # Updated from propertys
                labels=[]  # Updated from label
            )
        finally:
            await session.close()

    async def update(self, id: UUID, item: EntitySymbol) -> bool:
        """Update an existing entity in Neo4j."""
        session = await self._driver.session()
        try:
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
        finally:
            await session.close()

    async def delete(self, id: UUID) -> bool:
        """Delete an entity from Neo4j by ID."""
        session = await self._driver.session()
        try:
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
        finally:
            await session.close()

    async def list(self, skip: int = 0, limit: int = 100) -> List[EntitySymbol]:
        """List entities from Neo4j with pagination."""
        session = await self._driver.session()
        try:
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
                    entity_type="ENTITY",  # Add default entity type
                    semantics=[],
                    properties=[],
                    labels=[]
                )
                for record in records
            ]
        finally:
            await session.close()

    async def search(self, query: Dict[str, Any]) -> List[EntitySymbol]:
        """Search for entities in Neo4j using a query dictionary."""
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

        session = await self._driver.session()
        try:
            result = await session.run(cypher_query, params)
            records = await result.all()
            return [
                EntitySymbol(
                    symbol_id=UUID(record["e"]["id"]),
                    name=record["e"]["name"],
                    descriptions=record["e"]["descriptions"],
                    entity_type="ENTITY",  # Add default entity type
                    semantics=[],
                    propertys=[],
                    label=[]
                )
                for record in records
            ]
        finally:
            await session.close()
