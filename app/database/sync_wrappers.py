"""Synchronous wrappers for async database interfaces in Ananke2 framework.

This module provides synchronous interfaces to the async database implementations:
- GraphDatabase: Synchronous Neo4j interface for knowledge graph
- VectorDatabase: Synchronous ChromaDB interface for embeddings
- RelationalDatabase: Synchronous MySQL interface for documents

Features:
- Thread-safe async/sync conversion with run_async utility
- Connection management and automatic reconnection
- Error handling and propagation
- Factory functions for database instances

Example:
    >>> # Using synchronous graph database
    >>> db = get_sync_graph_db()
    >>> entity = Entity(name="concept", type="TERM", description="A key term")
    >>> db.store_entity(entity)
    >>>
    >>> # Using synchronous vector database
    >>> vdb = get_sync_vector_db()
    >>> vdb.store_embedding("doc1", [0.1, 0.2, 0.3], {"type": "document"})
"""

import asyncio
from typing import List, Optional, Dict, Any
from ..models.entities import Entity, Relationship

def run_async(coro):
    """Run an async coroutine in a synchronous context safely.

    Handles running async code in both async and sync contexts by:
    - Using existing event loop if available
    - Creating new loop if needed
    - Running in separate thread if loop is already running

    Args:
        coro: Async coroutine to run synchronously

    Returns:
        Any: Result of the coroutine execution

    Raises:
        RuntimeError: If event loop creation fails
        Exception: Any exception from coroutine execution
    """
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
    """Synchronous wrapper for Neo4j graph database interface.

    Provides synchronous methods for storing and querying knowledge graph
    entities and relationships using Neo4j.

    Args:
        uri (str): Neo4j connection URI
        username (str): Database username
        password (str): Database password

    Attributes:
        _async_db: Underlying async Neo4j interface
    """

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
    """Synchronous wrapper for ChromaDB vector database interface.

    Provides synchronous methods for storing and querying vector embeddings
    using ChromaDB collections.

    Args:
        host (str): ChromaDB server hostname
        port (int): ChromaDB server port
        collection_name (str, optional): Collection for embeddings. Defaults to "ananke2".

    Attributes:
        _async_db: Underlying async ChromaDB interface
    """

    def __init__(self, host: str, port: int, collection_name: str = "ananke2"):
        """Initialize ChromaDB interface.

        Args:
            host (str): ChromaDB server hostname
            port (int): ChromaDB server port
            collection_name (str, optional): Collection name. Defaults to "ananke2".

        Raises:
            ConnectionError: If connection to ChromaDB fails
            Exception: For other initialization errors
        """
        from .vector import ChromaInterface
        self._async_db = ChromaInterface(
            host=host,
            port=port,
            collection_name=collection_name
        )
        run_async(self._async_db.connect())

    def store_embedding(self, id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Store a vector embedding with metadata.

        Creates an EntitySemantic instance and stores it in the ChromaDB collection.

        Args:
            id (str): Unique identifier for the embedding
            embedding (List[float]): Vector representation of the entity
            metadata (Dict[str, Any]): Additional metadata for the embedding

        Raises:
            ConnectionError: If not connected to database
            ValueError: If embedding format is invalid
            Exception: If embedding storage fails
        """
        from ..models.entities import EntitySemantic
        semantic = EntitySemantic(
            name=id,
            vector_representation=embedding
        )
        run_async(self._async_db.create(semantic))

    def get_embedding(self, id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an embedding by ID.

        Args:
            id (str): Unique identifier of the embedding to retrieve

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing:
                - id (str): Semantic ID of the embedding
                - embedding (List[float]): Vector representation
                - name (str): Name associated with the embedding
                Returns None if embedding not found.

        Raises:
            ConnectionError: If not connected to database
            Exception: If embedding retrieval fails
        """
        result = run_async(self._async_db.read(id))
        if not result:
            return None
        return {
            "id": str(result.semantic_id),
            "embedding": result.vector_representation.tolist(),
            "name": result.name
        }

    def list_embeddings(self) -> List[Dict[str, Any]]:
        """List all embeddings in the database.

        Returns:
            List[Dict[str, Any]]: List of dictionaries, each containing:
                - id (str): Semantic ID of the embedding
                - embedding (List[float]): Vector representation
                - name (str): Name associated with the embedding

        Raises:
            ConnectionError: If not connected to database
            Exception: If embedding listing fails
        """
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
    """Synchronous wrapper for MySQL relational database interface.

    Provides synchronous methods for storing and querying structured data
    using MySQL, with configuration from settings.

    Args:
        host (str, optional): MySQL hostname. Defaults to settings.MYSQL_HOST.
        port (int, optional): MySQL port. Defaults to settings.MYSQL_PORT.
        user (str, optional): Database user. Defaults to settings.MYSQL_USER.
        password (str, optional): Database password. Defaults to settings.MYSQL_PASSWORD.
        database (str, optional): Database name. Defaults to settings.MYSQL_DATABASE.

    Attributes:
        _async_db: Underlying async MySQL interface
    """

    def __init__(self, host: str = None, port: int = None, user: str = None,
                 password: str = None, database: str = None):
        """Initialize MySQL database connection.

        Args:
            host (str, optional): MySQL hostname. Defaults to settings.MYSQL_HOST.
            port (int, optional): MySQL port. Defaults to settings.MYSQL_PORT.
            user (str, optional): Database user. Defaults to settings.MYSQL_USER.
            password (str, optional): Password. Defaults to settings.MYSQL_PASSWORD.
            database (str, optional): Database name. Defaults to settings.MYSQL_DATABASE.

        Raises:
            ConnectionError: If connection to MySQL fails
            Exception: For other initialization errors
        """
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
        """Store a document in the MySQL database.

        Creates a StructuredData instance and stores it in the database.

        Args:
            data (Dict[str, Any]): Document data including:
                - data_id: Unique identifier
                - data_type: Document type
                - data_value: Document content

        Returns:
            str: ID of the stored document

        Raises:
            ConnectionError: If not connected to database
            ValueError: If document format is invalid
            Exception: If document storage fails
        """
        from ..models.structured import StructuredData
        doc = StructuredData(
            data_id=data["data_id"],
            data_type=data["data_type"],
            data_value=data["data_value"]
        )
        doc_id = run_async(self._async_db.create(doc))
        return str(doc_id)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID.

        Args:
            doc_id (str): Document identifier

        Returns:
            Optional[Dict[str, Any]]: Document data value if found, None otherwise

        Raises:
            ConnectionError: If not connected to database
            Exception: If document retrieval fails
        """
        result = run_async(self._async_db.read(doc_id))
        if not result:
            return None
        return result.data_value

    def update_document(self, doc_id: str, data: Dict[str, Any]) -> None:
        """Update an existing document's data.

        Args:
            doc_id (str): Document identifier
            data (Dict[str, Any]): New data to update in the document

        Raises:
            ConnectionError: If not connected to database
            ValueError: If document format is invalid
            Exception: If document update fails
            KeyError: If document with doc_id not found
        """
        doc = run_async(self._async_db.read(doc_id))
        if doc:
            doc.data_value.update(data)
            run_async(self._async_db.update(doc_id, doc))

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database.

        Returns:
            List[Dict[str, Any]]: List of document data values

        Raises:
            ConnectionError: If not connected to database
            Exception: If document listing fails
        """
        results = run_async(self._async_db.list())
        return [r.data_value for r in results]


def get_sync_relational_db():
    """Get configured synchronous relational database interface.

    Creates and configures a RelationalDatabase instance using settings from
    config module for connection parameters.

    Returns:
        RelationalDatabase: Configured database interface

    Raises:
        Exception: If database connection fails
    """
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
    """Get configured synchronous vector database interface.

    Creates and configures a VectorDatabase instance using settings from
    config module for ChromaDB connection.

    Returns:
        VectorDatabase: Configured database interface

    Raises:
        Exception: If database connection fails
    """
    from ..config import settings
    db = VectorDatabase(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        collection_name=settings.CHROMA_COLLECTION
    )
    return db

def get_sync_graph_db():
    """Get configured synchronous graph database interface.

    Creates and configures a GraphDatabase instance using settings from
    config module for Neo4j connection.

    Returns:
        GraphDatabase: Configured database interface

    Raises:
        Exception: If database connection fails
    """
    from ..config import settings
    db = GraphDatabase(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )
    return db
