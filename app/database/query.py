"""Cross-database query interface for Ananke2."""

from typing import List, Dict, Any, Optional
from uuid import UUID

from ..models.structured import Document
from ..models.entities import EntitySymbol
from ..utils.qwen import QwenClient
from ..config import settings
from .vector import ChromaInterface
from .graph import Neo4jInterface
from .relational import MySQLInterface

class CrossDatabaseQuery:
    """Cross-database query interface supporting semantic, graph, and structured queries."""

    def __init__(
        self,
        vector_db: Optional[ChromaInterface] = None,
        graph_db: Optional[Neo4jInterface] = None,
        mysql_db: Optional[MySQLInterface] = None,
        qwen_client: Optional[QwenClient] = None
    ):
        """Initialize database interfaces."""
        self.vector_db = vector_db or ChromaInterface()
        self.graph_db = graph_db or Neo4jInterface(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD
        )
        self.mysql_db = mysql_db or MySQLInterface(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE
        )
        self.qwen_client = qwen_client or QwenClient()

    async def search_by_embedding(
        self,
        query_text: str,
        modality: str = "text",
        limit: int = 10
    ) -> List[Document]:
        """Search documents by semantic similarity.

        Args:
            query_text: Text to search for
            modality: Content modality (text, math, code)
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        # Generate query embedding
        embedding = await self.qwen_client.generate_embeddings(query_text, modality)

        # Search vector database
        results = await self.vector_db.search({
            "vector": embedding,
            "limit": limit
        })

        # Fetch full documents from MySQL
        documents = []
        for result in results:
            doc = await self.mysql_db.get(result.document_id)
            if doc:
                documents.append(doc)

        return documents

    async def search_by_graph(
        self,
        entity_type: Optional[str] = None,
        entity_name: Optional[str] = None,
        relationship_type: Optional[str] = None,
        min_strength: Optional[int] = None,
        limit: int = 10
    ) -> List[EntitySymbol]:
        """Search entities in knowledge graph.

        Args:
            entity_type: Filter by entity type
            entity_name: Filter by entity name
            relationship_type: Filter by relationship type
            min_strength: Minimum relationship strength
            limit: Maximum number of results

        Returns:
            List of matching entities
        """
        query = {}
        if entity_type:
            query["type"] = entity_type
        if entity_name:
            query["name"] = entity_name
        if relationship_type:
            query["relationship"] = relationship_type
        if min_strength:
            query["min_strength"] = min_strength
        query["limit"] = limit

        return await self.graph_db.search(query)

    async def search_structured(
        self,
        filters: Dict[str, Any],
        limit: int = 10
    ) -> List[Document]:
        """Search structured data in MySQL.

        Args:
            filters: Dictionary of field-value pairs to filter on
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        filters["limit"] = limit
        return await self.mysql_db.search(filters)

    async def combined_search(
        self,
        query_text: str,
        entity_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Document]:
        """Search across all databases.

        Args:
            query_text: Semantic search query
            entity_type: Knowledge graph entity type filter
            filters: Structured data filters
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        # Search each database in parallel
        semantic_results = await self.search_by_embedding(query_text, limit=limit)
        graph_results = await self.search_by_graph(entity_type=entity_type, limit=limit)
        structured_results = await self.search_structured(filters or {}, limit=limit)

        # Combine and deduplicate results
        seen_ids = set()
        combined = []

        for doc in semantic_results + structured_results:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                combined.append(doc)

        # Add documents referenced by graph entities
        for entity in graph_results:
            if entity.document_id and entity.document_id not in seen_ids:
                doc = await self.mysql_db.get(entity.document_id)
                if doc:
                    seen_ids.add(doc.id)
                    combined.append(doc)

        return combined[:limit]
