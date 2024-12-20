"""ChromaDB vector database interface implementation for Ananke2 knowledge framework.

This module provides a ChromaDB implementation for vector embedding storage and similarity search:
- ChromaInterface: Implements DatabaseInterface for semantic entity embeddings
- Async wrapper patterns for synchronous ChromaDB client
- Vector similarity search functionality

Features:
- Async/await support using asyncio.to_thread
- Connection management with automatic reconnection
- Batch operations for embeddings
- Metadata storage alongside vectors
- Configurable similarity search

Example:
    >>> interface = ChromaInterface(
    ...     host="localhost",
    ...     port=8000,
    ...     collection_name="entities"
    ... )
    >>> await interface.connect()
    >>> entity = EntitySemantic(name="concept", vector=[0.1, 0.2, 0.3])
    >>> entity_id = await interface.create(entity)
    >>> similar = await interface.search_similar([0.1, 0.2, 0.3])
"""

from typing import Any, Dict, List, Optional
from uuid import UUID
import asyncio
import chromadb
from chromadb.config import Settings
import numpy as np

from .base import DatabaseInterface
from ..models.entities import EntitySemantic

class ChromaInterface(DatabaseInterface[EntitySemantic]):
    """ChromaDB interface for vector embedding storage and similarity search.

    Provides vector database operations for storing and querying semantic
    entity embeddings using ChromaDB's collection-based architecture.

    Args:
        host (str, optional): ChromaDB server hostname. Defaults to "localhost".
        port (int, optional): ChromaDB server port. Defaults to 8000.
        collection_name (str, optional): Collection for embeddings. Defaults to "default".

    Attributes:
        host (str): ChromaDB server hostname
        port (int): ChromaDB server port
        collection_name (str): Name of ChromaDB collection
        _client: ChromaDB client instance
        _collection: ChromaDB collection instance
    """

    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = "default"):
        """Initialize Chroma interface."""
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    async def connect(self) -> None:
        """Establish connection to ChromaDB server.

        Creates ChromaDB client and collection with automatic reconnection
        support. Uses asyncio.to_thread to wrap synchronous client operations.

        Raises:
            ConnectionError: If ChromaDB server is not accessible
            Exception: For other connection errors
        """
        try:
            def _connect():
                settings = Settings(
                    chroma_server_host=self.host,
                    chroma_server_http_port=self.port
                )
                self._client = chromadb.Client(settings)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name
                )
            await asyncio.to_thread(_connect)
        except Exception as e:
            print(f"Error connecting to Chroma: {str(e)}")
            self._client = None
            self._collection = None
            raise

    async def disconnect(self) -> None:
        """Close ChromaDB connection and cleanup resources.

        Properly closes the database connection and cleans up client
        and collection instances. Safe to call multiple times.
        """
        if self._client:
            self._client = None
            self._collection = None

    async def create(self, item: EntitySemantic) -> UUID:
        """Create new semantic entity with vector embedding.

        Args:
            item (EntitySemantic): Entity with vector representation to store

        Returns:
            UUID: Unique identifier of created entity

        Raises:
            ConnectionError: If not connected to ChromaDB
            ValueError: If vector representation is invalid
            Exception: For other creation errors
        """
        try:
            await asyncio.to_thread(
                self._collection.add,
                ids=[str(item.semantic_id)],
                embeddings=[item.vector_representation],
                metadatas=[item.to_dict()]
            )
            return item.semantic_id
        except Exception as e:
            print(f"Error creating semantic entity in Chroma: {str(e)}")
            raise

    async def read(self, id: UUID) -> Optional[EntitySemantic]:
        """Read semantic entity by ID with vector embedding.

        Args:
            id (UUID): Unique identifier of entity to read

        Returns:
            Optional[EntitySemantic]: Found entity or None if not found

        Raises:
            ConnectionError: If not connected to ChromaDB
            Exception: For other read errors
        """
        async def _read():
            result = self._collection.get(
                ids=[str(id)],
                include=["embeddings", "metadatas"]
            )
            if not result["ids"]:
                return None
            return EntitySemantic(
                semantic_id=UUID(result["ids"][0]),
                name=result["metadatas"][0]["name"],
                vector_representation=np.array(result["embeddings"][0])
            )
        return await asyncio.to_thread(_read)

    async def update(self, id: UUID, item: EntitySemantic) -> bool:
        """Update existing semantic entity and its embedding.

        Args:
            id (UUID): Unique identifier of entity to update
            item (EntitySemantic): New entity data with vector

        Returns:
            bool: True if entity was found and updated

        Raises:
            ConnectionError: If not connected to ChromaDB
            ValueError: If vector representation is invalid
            Exception: For other update errors
        """
        async def _update():
            try:
                vector = np.array(item.vector_representation).tolist()
                self._collection.update(
                    ids=[str(id)],
                    embeddings=[vector],
                    metadatas=[{"name": item.name}]
                )
                return True
            except Exception:
                return False
        return await asyncio.to_thread(_update)

    async def delete(self, id: UUID) -> bool:
        """Delete semantic entity and its embedding by ID.

        Args:
            id (UUID): Unique identifier of entity to delete

        Returns:
            bool: True if entity was found and deleted

        Raises:
            ConnectionError: If not connected to ChromaDB
            Exception: For other deletion errors
        """
        async def _delete():
            try:
                self._collection.delete(ids=[str(id)])
                return True
            except Exception:
                return False
        return await asyncio.to_thread(_delete)

    async def list(self, skip: int = 0, limit: int = 100) -> List[EntitySemantic]:
        """List semantic entities with pagination support.

        Args:
            skip (int, optional): Number of records to skip. Defaults to 0.
            limit (int, optional): Maximum records to return. Defaults to 100.

        Returns:
            List[EntitySemantic]: List of found entities with embeddings

        Raises:
            ConnectionError: If not connected to ChromaDB
            ValueError: If skip/limit are invalid
            Exception: For other listing errors
        """
        async def _list():
            result = self._collection.get(
                limit=limit,
                offset=skip,
                include=["embeddings", "metadatas"]
            )
            return [
                EntitySemantic(
                    semantic_id=UUID(id),
                    name=metadata["name"],
                    vector_representation=np.array(embedding)
                )
                for id, metadata, embedding in zip(
                    result["ids"],
                    result["metadatas"],
                    result["embeddings"]
                )
            ]
        return await asyncio.to_thread(_list)

    async def search(self, query: Dict[str, Any]) -> List[EntitySemantic]:
        """Search semantic entities by vector similarity.

        Args:
            query (Dict[str, Any]): Search parameters including:
                - vector: Query vector for similarity search
                - limit: Maximum number of results (default: 10)

        Returns:
            List[EntitySemantic]: List of similar entities by vector distance

        Raises:
            ConnectionError: If not connected to ChromaDB
            ValueError: If query vector is invalid
            Exception: For other search errors
        """
        async def _search():
            if "vector" not in query:
                return []
            vector = np.array(query["vector"]).tolist()
            result = self._collection.query(
                query_embeddings=[vector],
                n_results=query.get("limit", 10),
                include=["embeddings", "metadatas"]
            )
            return [
                EntitySemantic(
                    semantic_id=UUID(id),
                    name=metadata["name"],
                    vector_representation=np.array(embedding)
                )
                for id, metadata, embedding in zip(
                    result["ids"],
                    result["metadatas"],
                    result["embeddings"]
                )
            ]
        return await asyncio.to_thread(_search)

    async def store_embedding(self, id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Store vector embedding with metadata.

        Args:
            id (str): Unique identifier for embedding
            embedding (List[float]): Vector embedding to store
            metadata (Dict[str, Any]): Additional metadata for embedding

        Raises:
            ConnectionError: If not connected to ChromaDB
            ValueError: If embedding format is invalid
            Exception: For other storage errors
        """
        async def _store():
            if not self._collection:
                await self.connect()
            self._collection.add(
                ids=[id],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        await asyncio.to_thread(_store)

    async def search_similar(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings by vector distance.

        Args:
            embedding (List[float]): Query vector for similarity search
            limit (int, optional): Maximum results to return. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: Similar embeddings with scores and metadata

        Raises:
            ConnectionError: If not connected to ChromaDB
            ValueError: If embedding format is invalid
            Exception: For other search errors
        """
        async def _search_similar():
            if not self._collection:
                await self.connect()
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=limit
            )
            return [{"id": id, "score": 1.0, "metadata": metadata}
                    for id, metadata in zip(results["ids"][0], results["metadatas"][0])]
        return await asyncio.to_thread(_search_similar)

    async def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Retrieve all stored embeddings with metadata.

        Returns:
            List[Dict[str, Any]]: All embeddings with IDs and metadata

        Raises:
            ConnectionError: If not connected to ChromaDB
            Exception: For other retrieval errors
        """
        async def _get_all():
            if not self._collection:
                raise ConnectionError("Not connected to database")
            results = self._collection.get()
            return [{"id": id, "embedding": embedding, "metadata": metadata}
                    for id, embedding, metadata in zip(results["ids"], results["embeddings"], results["metadatas"])]
        return await asyncio.to_thread(_get_all)
