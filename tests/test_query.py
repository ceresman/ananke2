"""Tests for cross-database query interface."""

import pytest
from unittest.mock import Mock, AsyncMock
from uuid import UUID, uuid4
from app.database.query import CrossDatabaseQuery
from app.models.structured import Document, StructuredData
from app.models.entities import EntitySymbol, EntitySemantic
from app.database.vector import ChromaInterface
from app.database.graph import Neo4jInterface
from app.database.relational import MySQLInterface

Document.model_rebuild()

@pytest.fixture
def mock_databases():
    """Create mock database interfaces."""
    chroma = Mock(spec=ChromaInterface)
    neo4j = Mock(spec=Neo4jInterface)
    mysql = Mock(spec=MySQLInterface)
    return chroma, neo4j, mysql

@pytest.fixture
def mock_entity():
    """Create a mock entity for testing."""
    return EntitySymbol(
        symbol_id=uuid4(),
        name="TEST_ENTITY",
        entity_type="TECHNOLOGY",
        descriptions=["Test entity description"],
        semantics=[
            EntitySemantic(
                semantic_id=uuid4(),
                semantic_type="DEFINITION",
                semantic_value="Test semantic value"
            )
        ],
        properties={
            "test_prop": "test_value"
        },
        labels=["TEST"]
    )

@pytest.fixture
def query(mock_databases):
    """Create a CrossDatabaseQuery instance with mock databases."""
    chroma, neo4j, mysql = mock_databases

    async def mock_init():
        return chroma, neo4j, mysql

    # Create mock search results
    mock_doc = Document(
        id=uuid4(),
        meta=StructuredData(
            data_id=uuid4(),
            data_type="test",
            data_value={"test": "value"}
        ),
        meta_embedding=[0.1, 0.2, 0.3],
        raw_content="Test content"
    )

    chroma.search_by_embedding = AsyncMock(return_value=[mock_doc])
    neo4j.search_by_entity_type = AsyncMock(return_value=[mock_entity()])
    mysql.search_structured = AsyncMock(return_value=[mock_doc])

    query = CrossDatabaseQuery()
    query._init_databases = mock_init
    return query

@pytest.mark.asyncio
async def test_search_by_embedding(query):
    """Test searching by embedding vector."""
    results = await query.search_by_embedding("test query")
    assert len(results) > 0
    assert isinstance(results[0], Document)

@pytest.mark.asyncio
async def test_search_by_graph(query):
    """Test searching by graph entity type."""
    results = await query.search_by_graph("TECHNOLOGY")
    assert len(results) > 0
    assert isinstance(results[0], EntitySymbol)

@pytest.mark.asyncio
async def test_search_structured(query):
    """Test searching structured data."""
    results = await query.search_structured({"test": "value"})
    assert len(results) > 0
    assert isinstance(results[0], Document)

@pytest.mark.asyncio
async def test_combined_search(query):
    """Test combined search across all databases."""
    results = await query.combined_search(
        query_text="test",
        entity_type="TECHNOLOGY",
        structured_filter={"test": "value"}
    )
    assert len(results) > 0

@pytest.mark.asyncio
async def test_search_with_modality(query):
    """Test searching with specific modality."""
    results = await query.search_by_embedding(
        "test query",
        modality="text"
    )
    assert len(results) > 0
    assert isinstance(results[0], Document)

@pytest.mark.asyncio
async def test_error_handling(query):
    """Test error handling in search operations."""
    # Mock an error in the database
    query._chroma.search_by_embedding = AsyncMock(
        side_effect=Exception("Test error")
    )

    # Should return empty list on error
    results = await query.search_by_embedding("test query")
    assert len(results) == 0
