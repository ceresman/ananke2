"""Synchronous wrappers for async database interfaces."""

import asyncio
from typing import List, Optional, Dict, Any
from ..models.entities import Entity, Relationship

def run_async(coro):
    """Run an async coroutine in a sync context."""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, create a new loop
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
                asyncio.set_event_loop(loop)
        else:
            # If we're not in an async context, use the current loop
            return loop.run_until_complete(coro)
    except RuntimeError:
        # If there's no event loop, create one
        return asyncio.run(coro)

class GraphDatabase:
    """Synchronous wrapper for Neo4j interface."""

    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j interface."""
        from .graph import AsyncGraphDatabase
        # Replace localhost with neo4j in URI if needed
        if "localhost" in uri:
            uri = uri.replace("localhost", "neo4j")
        self._async_db = AsyncGraphDatabase(
            uri=uri,
            username=username,
            password=password
        )
        run_async(self._async_db.connect())

    def store_entity(self, entity: Entity) -> None:
        """Store an entity in the graph database."""
        from ..models.entities import EntitySymbol
        entity_symbol = EntitySymbol(
            name=entity.name,
            descriptions=[entity.description],
            entity_type=entity.type,
            semantics=[],
            properties=[],
            labels=[]
        )
        run_async(self._async_db.create(entity_symbol))

    def store_relationship(self, rel: Relationship) -> None:
        """Store a relationship in the graph database."""
        from ..models.relations import RelationSymbol
        rel_symbol = RelationSymbol(
            source=rel.source,
            target=rel.target,
            relation_type="RELATED_TO",
            descriptions=[rel.relationship],
            strength=rel.relationship_strength
        )
        run_async(self._async_db.create(rel_symbol))

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get an entity by name."""
        entities = run_async(self._async_db.search({"name": name}))
        if not entities:
            return None
        entity = entities[0]
        return Entity(
            name=entity.name,
            type=entity.entity_type,
            description=entity.descriptions[0] if entity.descriptions else ""
        )

    def get_relationship(self, source: str, target: str) -> Optional[Relationship]:
        """Get a relationship between two entities."""
        rels = run_async(self._async_db.search({
            "source": source,
            "target": target
        }))
        if not rels:
            return None
        rel = rels[0]
        return Relationship(
            source=rel.source,
            target=rel.target,
            relationship=rel.descriptions[0] if rel.descriptions else "",
            relationship_strength=rel.strength
        )

    def list_entities(self) -> List[Entity]:
        """List all entities in the database."""
        entities = run_async(self._async_db.list())
        return [
            Entity(
                name=e.name,
                type=e.entity_type,
                description=e.descriptions[0] if e.descriptions else ""
            )
            for e in entities
        ]

    def list_relationships(self) -> List[Relationship]:
        """List all relationships in the database."""
        rels = run_async(self._async_db.list())
        return [
            Relationship(
                source=r.source,
                target=r.target,
                relationship=r.descriptions[0] if r.descriptions else "",
                relationship_strength=r.strength
            )
            for r in rels
        ]

class VectorDatabase:
    """Synchronous wrapper for ChromaDB interface."""

    def __init__(self, host: str, port: int, collection_name: str = "ananke2"):
        """Initialize ChromaDB interface."""
        from .vector import AsyncVectorDatabase
        self._async_db = AsyncVectorDatabase(
            host=host,
            port=port,
            collection_name=collection_name
        )
        run_async(self._async_db.connect())

    def store_embedding(self, id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Store an embedding with metadata."""
        from ..models.entities import EntitySemantic
        semantic = EntitySemantic(
            name=id,
            vector_representation=embedding
        )
        run_async(self._async_db.create(semantic))

    def get_embedding(self, id: str) -> Optional[Dict[str, Any]]:
        """Get an embedding by ID."""
        result = run_async(self._async_db.read(id))
        if not result:
            return None
        return {
            "id": str(result.semantic_id),
            "embedding": result.vector_representation.tolist(),
            "name": result.name
        }

    def list_embeddings(self) -> List[Dict[str, Any]]:
        """List all embeddings in the database."""
        results = run_async(self._async_db.list())
        return [
            {
                "id": str(r.semantic_id),
                "embedding": r.vector_representation.tolist(),
                "name": r.name
            }
            for r in results
        ]

class RelationalDatabase:
    """Synchronous wrapper for MySQL interface."""

    def __init__(self, uri: str = None, username: str = None, password: str = None,
                 host: str = None, port: int = None, database: str = None):
        """Initialize MySQL interface."""
        from .relational import AsyncRelationalDatabase
        from ..config import settings

        # Support both URI-style and individual parameter initialization
        if uri:
            host, port = uri.split(":")
            port = int(port)

        self._async_db = AsyncRelationalDatabase(
            host=host,
            port=port,
            user=username,
            password=password,
            database=database or settings.MYSQL_DATABASE
        )
        run_async(self._async_db.connect())

    def store_document(self, data: Dict[str, Any]) -> str:
        """Store a document and return its ID."""
        from ..models.structured import StructuredData
        doc = StructuredData(
            data_type="document",
            data_value=data
        )
        doc_id = run_async(self._async_db.create(doc))
        return str(doc_id)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        result = run_async(self._async_db.read(doc_id))
        if not result:
            return None
        return result.data_value

    def update_document(self, doc_id: str, data: Dict[str, Any]) -> None:
        """Update a document's data."""
        doc = run_async(self._async_db.read(doc_id))
        if doc:
            doc.data_value.update(data)
            run_async(self._async_db.update(doc_id, doc))

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database."""
        results = run_async(self._async_db.list())
        return [r.data_value for r in results]
