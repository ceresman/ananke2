"""ChromaDB vector database interface implementation."""

from typing import Any, Dict, List, Optional
from uuid import UUID
import chromadb
from chromadb.config import Settings
import numpy as np

from .base import DatabaseInterface
from ..models.entities import EntitySemantic

class AsyncVectorDatabase(DatabaseInterface[EntitySemantic]):
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
        settings = Settings(
            chroma_server_host=self.host,
            chroma_server_http_port=self.port
        )
        self._client = chromadb.Client(settings)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name
        )

    async def disconnect(self) -> None:
        """Close the Chroma database connection."""
        if self._client:
            self._client = None
            self._collection = None

    async def create(self, item: EntitySemantic) -> UUID:
        """Create a new semantic entity in Chroma."""
        vector = np.array(item.vector_representation).tolist()
        self._collection.add(
            ids=[str(item.semantic_id)],
            embeddings=[vector],
            metadatas=[{"name": item.name}]
        )
        return item.semantic_id

    async def read(self, id: UUID) -> Optional[EntitySemantic]:
        """Read a semantic entity from Chroma by ID."""
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

    async def update(self, id: UUID, item: EntitySemantic) -> bool:
        """Update an existing semantic entity in Chroma."""
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

    async def delete(self, id: UUID) -> bool:
        """Delete a semantic entity from Chroma by ID."""
        try:
            self._collection.delete(ids=[str(id)])
            return True
        except Exception:
            return False

    async def list(self, skip: int = 0, limit: int = 100) -> List[EntitySemantic]:
        """List semantic entities from Chroma with pagination."""
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

    async def search(self, query: Dict[str, Any]) -> List[EntitySemantic]:
        """Search for semantic entities in Chroma using a query dictionary."""
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

    async def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Get all embeddings from the database."""
        if not self._collection:
            raise ConnectionError("Not connected to database")
        results = self._collection.get()
        return [{"id": id, "embedding": embedding, "metadata": metadata}
                for id, embedding, metadata in zip(results["ids"], results["embeddings"], results["metadatas"])]
