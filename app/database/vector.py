"""ChromaDB vector database interface implementation."""

from typing import Any, Dict, List, Optional
from uuid import UUID
import asyncio
import chromadb
from chromadb.config import Settings
import numpy as np

from .base import DatabaseInterface
from ..models.entities import EntitySemantic

class ChromaInterface(DatabaseInterface[EntitySemantic]):
    """Chroma vector database interface implementation."""

    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = "default"):
        """Initialize Chroma interface."""
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    async def connect(self) -> None:
        """Establish connection to Chroma database."""
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
        """Close the Chroma database connection."""
        if self._client:
            self._client = None
            self._collection = None

    async def create(self, item: EntitySemantic) -> UUID:
        """Create a new semantic entity in Chroma."""
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
        """Read a semantic entity from Chroma by ID."""
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
        """Update an existing semantic entity in Chroma."""
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
        """Delete a semantic entity from Chroma by ID."""
        async def _delete():
            try:
                self._collection.delete(ids=[str(id)])
                return True
            except Exception:
                return False
        return await asyncio.to_thread(_delete)

    async def list(self, skip: int = 0, limit: int = 100) -> List[EntitySemantic]:
        """List semantic entities from Chroma with pagination."""
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
        """Search for semantic entities in Chroma using a query dictionary."""
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
        """Store an embedding in the database."""
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
        """Search for similar embeddings."""
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
        """Get all embeddings from the database."""
        async def _get_all():
            if not self._collection:
                raise ConnectionError("Not connected to database")
            results = self._collection.get()
            return [{"id": id, "embedding": embedding, "metadata": metadata}
                    for id, embedding, metadata in zip(results["ids"], results["embeddings"], results["metadatas"])]
        return await asyncio.to_thread(_get_all)
