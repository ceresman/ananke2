"""Cross-database query interface for Ananke2.

This module provides a unified interface for querying across multiple database
types (vector, graph, and relational) in the Ananke2 knowledge framework.
It supports semantic similarity search, graph relationship queries, and
structured data filtering with the ability to combine results.
"""

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
    """Cross-database query interface supporting semantic, graph, and structured queries.

    This class provides a unified interface for searching across multiple database
    types in the Ananke2 knowledge framework. It handles:
    - Semantic similarity search using vector embeddings
    - Knowledge graph traversal and relationship queries
    - Structured data filtering and retrieval
    - Combined multi-database search with result deduplication

    Attributes:
        vector_db (ChromaInterface): Vector database for semantic search
        graph_db (Neo4jInterface): Graph database for relationship queries
        mysql_db (MySQLInterface): Relational database for structured data
        qwen_client (QwenClient): Client for generating embeddings

    Example:
        ```python
        query = CrossDatabaseQuery()

        # Search by semantic similarity
        docs = await query.search_by_embedding("quantum computing")

        # Search knowledge graph
        entities = await query.search_by_graph(entity_type="TECHNOLOGY")

        # Combined search across all databases
        results = await query.combined_search(
            query_text="quantum computing",
            entity_type="TECHNOLOGY",
            filters={"year": 2023}
        )
        ```
    """

    def __init__(
        self,
        vector_db: Optional[ChromaInterface] = None,
        graph_db: Optional[Neo4jInterface] = None,
        mysql_db: Optional[MySQLInterface] = None,
        qwen_client: Optional[QwenClient] = None
    ):
        """Initialize database interfaces.

        Args:
            vector_db (Optional[ChromaInterface]): Vector database interface
            graph_db (Optional[Neo4jInterface]): Graph database interface
            mysql_db (Optional[MySQLInterface]): Relational database interface
            qwen_client (Optional[QwenClient]): Embedding generation client

        If interfaces are not provided, they will be initialized with
        default configuration from settings.
        """
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

        Uses the Qwen API to generate embeddings for the query text
        and searches the vector database for similar documents.

        Args:
            query_text (str): Text to search for
            modality (str): Content modality (text, math, code)
            limit (int): Maximum number of results

        Returns:
            List[Document]: List of matching documents ordered by similarity

        Example:
            ```python
            # Search for similar mathematical content
            docs = await query.search_by_embedding(
                "Maxwell's equations",
                modality="math",
                limit=5
            )
            ```
        """
        embedding = await self.qwen_client.generate_embeddings(query_text, modality)
        results = await self.vector_db.search({
            "vector": embedding,
            "limit": limit
        })
        documents = []
        for result in results:
            doc_id = result.get('metadata', {}).get('document_id')
            if doc_id:
                doc = await self.mysql_db.get(UUID(doc_id))
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

        Performs a graph database query to find entities and relationships
        matching the specified criteria.

        Args:
            entity_type (Optional[str]): Filter by entity type
            entity_name (Optional[str]): Filter by entity name
            relationship_type (Optional[str]): Filter by relationship type
            min_strength (Optional[int]): Minimum relationship strength
            limit (int): Maximum number of results

        Returns:
            List[EntitySymbol]: List of matching entities

        Example:
            ```python
            # Find strongly related technology concepts
            entities = await query.search_by_graph(
                entity_type="TECHNOLOGY",
                relationship_type="RELATED_TO",
                min_strength=8,
                limit=5
            )
            ```
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

        Performs a filtered search on structured document metadata
        in the relational database.

        Args:
            filters (Dict[str, Any]): Dictionary of field-value pairs to filter on
            limit (int): Maximum number of results

        Returns:
            List[Document]: List of matching documents

        Example:
            ```python
            # Find recent papers by a specific author
            docs = await query.search_structured({
                "year": 2023,
                "author": "Jane Doe",
                "type": "research_paper"
            })
            ```
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

        Performs parallel searches across semantic, graph, and structured
        databases, combining and deduplicating results.

        Args:
            query_text (str): Semantic search query
            entity_type (Optional[str]): Knowledge graph entity type filter
            filters (Optional[Dict[str, Any]]): Structured data filters
            limit (int): Maximum number of results

        Returns:
            List[Document]: Combined and deduplicated list of matching documents

        Example:
            ```python
            # Search for quantum computing papers from 2023
            results = await query.combined_search(
                query_text="quantum computing applications",
                entity_type="TECHNOLOGY",
                filters={"year": 2023, "type": "research_paper"}
            )
            ```
        """
        semantic_results = await self.search_by_embedding(query_text, limit=limit)
        graph_results = await self.search_by_graph(entity_type=entity_type, limit=limit)
        structured_results = await self.search_structured(filters or {}, limit=limit)
        seen_ids = set()
        combined = []
        for doc in semantic_results + structured_results:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                combined.append(doc)
        for entity in graph_results:
            if entity.document_id and entity.document_id not in seen_ids:
                doc = await self.mysql_db.get(entity.document_id)
                if doc:
                    seen_ids.add(doc.id)
                    combined.append(doc)
        return combined[:limit]
