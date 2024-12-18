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
from app.models.types import StructuredDataBase

@pytest.fixture
def mock_databases():
    """Create mock database interfaces."""
    vector_db = AsyncMock(spec=ChromaInterface)
    graph_db = AsyncMock(spec=Neo4jInterface)
    mysql_db = AsyncMock(spec=MySQLInterface)
    return vector_db, graph_db, mysql_db

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
        properties=[
            StructuredDataBase(
                data_id=uuid4(),
                data_type="property",
                data_value={"test_prop": "test_value"}
            )
        ],
        labels=[
            StructuredDataBase(
                data_id=uuid4(),
                data_type="label",
                data_value={"label": "TEST"}
            )
        ]
    )

@pytest.fixture
def mock_doc():
    """Create a mock document for testing."""
    return Document(
        id=uuid4(),
        meta=StructuredData(
            data_id=uuid4(),
            data_type="test",
            data_value={"test": "value"}
        ),
        meta_embedding=[0.1, 0.2, 0.3],
        raw_content="Test content"
    )

@pytest.fixture
def query(mock_databases, mock_entity, mock_doc):
    """Create a CrossDatabaseQuery instance with mock databases."""
    vector_db, graph_db, mysql_db = mock_databases

    # Set up mock returns
    vector_db.search = AsyncMock(return_value=[{
        'id': str(mock_doc.id),
        'metadata': {'document_id': str(mock_doc.id)}
    }])
    graph_db.search = AsyncMock(return_value=[mock_entity])
    mysql_db.search = AsyncMock(return_value=[mock_doc])
    mysql_db.get = AsyncMock(return_value=mock_doc)

    # Create mock Qwen client
    qwen_client = AsyncMock()
    qwen_client.generate_embeddings = AsyncMock(return_value=[0.1, 0.2, 0.3])

    # Create query instance with mocked dependencies
    query = CrossDatabaseQuery(
        vector_db=vector_db,
        graph_db=graph_db,
        mysql_db=mysql_db,
        qwen_client=qwen_client
    )
    return query

@pytest.mark.asyncio
async def test_search_by_embedding(query, mock_doc):
    """Test searching by embedding vector."""
    results = await query.search_by_embedding("test query")
    assert len(results) > 0
    assert isinstance(results[0], Document)

@pytest.mark.asyncio
async def test_search_by_graph(query, mock_entity):
    """Test searching by graph entity type."""
    results = await query.search_by_graph(entity_type="TECHNOLOGY")
    assert len(results) > 0
    assert isinstance(results[0], EntitySymbol)

@pytest.mark.asyncio
async def test_search_structured(query, mock_doc):
    """Test searching structured data."""
    results = await query.search_structured({"test": "value"})
    assert len(results) > 0
    assert isinstance(results[0], Document)

@pytest.mark.asyncio
async def test_combined_search(query, mock_doc):
    """Test combined search across all databases."""
    results = await query.combined_search(
        query_text="test",
        entity_type="TECHNOLOGY",
        filters={"test": "value"}
    )
    assert len(results) > 0
    assert isinstance(results[0], Document)

@pytest.mark.asyncio
async def test_search_with_modality(query, mock_doc):
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
    query.vector_db.search = AsyncMock(side_effect=Exception("Test error"))
    with pytest.raises(Exception):
        await query.search_by_embedding("test query")
