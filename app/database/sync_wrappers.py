"""Synchronous wrappers for async database interfaces."""

import asyncio
from typing import List, Optional, Dict, Any
from ..models.entities import Entity, Relationship

def run_async(coro):
    """Run an async coroutine in a sync context."""
    try:
        # Get current event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # If we're in an async context (loop is running)
        if loop.is_running():
            # Use asyncio.run() in a new thread to avoid loop conflicts
            import threading
            result = None
            exception = None

            def run_coro():
                nonlocal result, exception
                try:
                    result = asyncio.run(coro)
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=run_coro)
            thread.start()
            thread.join()

            if exception:
                raise exception
            return result
        else:
            # Use existing loop
            return loop.run_until_complete(coro)
    except Exception as e:
        print(f"Error in run_async: {str(e)}")
        raise

class GraphDatabase:
    """Synchronous wrapper for Neo4j interface."""

    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j interface."""
        from .graph import Neo4jInterface
        # Replace localhost with neo4j in URI if needed
        if "localhost" in uri:
            uri = uri.replace("localhost", "neo4j")
        self._async_db = Neo4jInterface(
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
        from .vector import ChromaInterface
        self._async_db = ChromaInterface(
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

    def __init__(self, host: str = None, port: int = None, user: str = None,
                 password: str = None, database: str = None):
        """Initialize MySQL interface."""
        from .relational import AsyncRelationalDatabase
        from ..config import settings

        self._async_db = AsyncRelationalDatabase(
            host=host or settings.MYSQL_HOST,
            port=port or settings.MYSQL_PORT,
            user=user or settings.MYSQL_USER,
            password=password or settings.MYSQL_PASSWORD,
            database=database or settings.MYSQL_DATABASE
        )
        run_async(self._async_db.connect())

    def store_document(self, data: Dict[str, Any]) -> str:
        """Store a document and return its ID."""
        from ..models.structured import StructuredData
        doc = StructuredData(
            data_id=data["data_id"],
            data_type=data["data_type"],
            data_value=data["data_value"]
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


def get_sync_relational_db():
    """Get synchronous relational database interface."""
    from ..config import settings
    db = RelationalDatabase(
        host=settings.MYSQL_HOST,
        port=settings.MYSQL_PORT,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        database=settings.MYSQL_DATABASE
    )
    return db

def get_sync_vector_db():
    """Get synchronous vector database interface."""
    from ..config import settings
    db = VectorDatabase(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        collection_name=settings.CHROMA_COLLECTION
    )
    return db

def get_sync_graph_db():
    """Get synchronous graph database interface."""
    from ..config import settings
    db = GraphDatabase(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )
    return db
