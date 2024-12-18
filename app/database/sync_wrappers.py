"""Synchronous wrappers for async database interfaces."""

import asyncio
from typing import List, Optional, Dict, Any
from ..models.entities import Entity, Relationship

class GraphDatabase:
    """Synchronous wrapper for Neo4j interface."""

    def __init__(self):
        """Initialize Neo4j interface."""
        from .graph import AsyncGraphDatabase
        from ..config import settings
        self._async_db = AsyncGraphDatabase(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD
        )
        asyncio.run(self._async_db.connect())

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
        asyncio.run(self._async_db.create(entity_symbol))

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
        asyncio.run(self._async_db.create(rel_symbol))

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get an entity by name."""
        entities = asyncio.run(self._async_db.search({"name": name}))
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
        rels = asyncio.run(self._async_db.search({
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
        entities = asyncio.run(self._async_db.list())
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
        rels = asyncio.run(self._async_db.list())
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

    def __init__(self):
        """Initialize ChromaDB interface."""
        from .vector import AsyncVectorDatabase
        from ..config import settings
        self._async_db = AsyncVectorDatabase(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT
        )
        asyncio.run(self._async_db.connect())

    def store_embedding(self, id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Store an embedding with metadata."""
        from ..models.entities import EntitySemantic
        semantic = EntitySemantic(
            name=id,
            vector_representation=embedding
        )
        asyncio.run(self._async_db.create(semantic))

    def get_embedding(self, id: str) -> Optional[Dict[str, Any]]:
        """Get an embedding by ID."""
        result = asyncio.run(self._async_db.read(id))
        if not result:
            return None
        return {
            "id": str(result.semantic_id),
            "embedding": result.vector_representation.tolist(),
            "name": result.name
        }

    def list_embeddings(self) -> List[Dict[str, Any]]:
        """List all embeddings in the database."""
        results = asyncio.run(self._async_db.list())
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

    def __init__(self):
        """Initialize MySQL interface."""
        from .relational import AsyncRelationalDatabase
        from ..config import settings
        self._async_db = AsyncRelationalDatabase(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE
        )
        asyncio.run(self._async_db.connect())

    def store_document(self, data: Dict[str, Any]) -> str:
        """Store a document and return its ID."""
        from ..models.structured import StructuredData
        doc = StructuredData(
            data_type="document",
            data_value=data
        )
        doc_id = asyncio.run(self._async_db.create(doc))
        return str(doc_id)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        result = asyncio.run(self._async_db.read(doc_id))
        if not result:
            return None
        return result.data_value

    def update_document(self, doc_id: str, data: Dict[str, Any]) -> None:
        """Update a document's data."""
        doc = asyncio.run(self._async_db.read(doc_id))
        if doc:
            doc.data_value.update(data)
            asyncio.run(self._async_db.update(doc_id, doc))

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database."""
        results = asyncio.run(self._async_db.list())
        return [r.data_value for r in results]
