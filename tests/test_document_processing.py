import os
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from uuid import UUID
from app.models.entities import Entity, Relationship
from app.database.relational import AsyncRelationalDatabase
from app.database.graph import Neo4jInterface
from app.database.vector import ChromaInterface
from app.tasks.document import process_document, process_pdf

@pytest.fixture
def sample_pdf():
    """Create a sample PDF file for testing."""
    import tempfile
    import os

    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
        temp_pdf.write(b"%PDF-1.4\nTest PDF content")
        temp_pdf.flush()
        return temp_pdf.name

@pytest.fixture
def mock_databases():
    """Create mock database interfaces."""
    class MockAsyncRelationalDB:
        async def connect(self):
            pass

        async def create(self, item):
            return item.data_id

        async def store_document(self, doc_data):
            return str(doc_data["data_id"])

        async def update_document(self, doc_id, updates):
            return True

        async def create_entity(self, entity):
            return {"id": str(entity.id)}

    class MockNeo4jInterface:
        async def connect(self):
            pass

        async def create(self, entity):
            return entity.symbol_id

        async def create_entity(self, entity):
            return {"id": "test-entity-id"}

        async def create_relationship(self, rel):
            return {"id": "test-rel-id"}

    class MockChromaInterface:
        async def connect(self):
            pass

        async def create(self, semantic):
            return semantic.semantic_id

        async def store_embedding(self, id, embedding, metadata):
            return "test-embedding-id"

    return {
        "relational": MockAsyncRelationalDB(),
        "graph": MockNeo4jInterface(),
        "vector": MockChromaInterface()
    }

@pytest.mark.asyncio
async def test_document_processing():
    """Test document processing functionality."""
    # Set up environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TORCH_CPU_ONLY"] = "1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
    os.environ["TF_CPU_ONLY"] = "1"

    class MockQwenClient:
        async def extract_entities(self, text):
            return [
                {
                    "name": "TEST_ENTITY",
                    "type": "TEST",
                    "description": "Test entity description"
                }
            ]

        async def extract_relationships(self, text, entities):
            return [
                {
                    "source": "TEST_ENTITY",
                    "target": "TEST_ENTITY",
                    "relationship": "test relationship",
                    "relationship_strength": 5
                }
            ]

        async def generate_embedding(self, text):
            return [0.1] * 768

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    # Mock PDF processing and database interfaces
    with patch("app.tasks.document.partition_pdf", return_value=["Test content"]), \
         patch("app.tasks.document.QwenClient", return_value=MockQwenClient()), \
         patch("app.tasks.document.AsyncRelationalDatabase") as mock_rel_db, \
         patch("app.tasks.document.Neo4jInterface") as mock_neo4j, \
         patch("app.tasks.document.ChromaInterface") as mock_chroma:

        # Configure mock databases
        mock_rel_db.return_value = AsyncMock(spec=AsyncRelationalDatabase)
        mock_rel_db.return_value.connect = AsyncMock()
        mock_rel_db.return_value.create = AsyncMock(return_value=UUID("123e4567-e89b-12d3-a456-426614174000"))
        mock_rel_db.return_value.store_document = AsyncMock(return_value="test-doc-id")
        mock_rel_db.return_value.update_document = AsyncMock(return_value=True)

        mock_neo4j.return_value = AsyncMock(spec=Neo4jInterface)
        mock_neo4j.return_value.connect = AsyncMock()
        mock_neo4j.return_value.create = AsyncMock(return_value=UUID("123e4567-e89b-12d3-a456-426614174001"))
        mock_neo4j.return_value.create_entity = AsyncMock(return_value={"id": "test-entity-id"})
        mock_neo4j.return_value.create_relationship = AsyncMock(return_value={"id": "test-rel-id"})

        mock_chroma.return_value = AsyncMock(spec=ChromaInterface)
        mock_chroma.return_value.connect = AsyncMock()
        mock_chroma.return_value.create = AsyncMock(return_value=UUID("123e4567-e89b-12d3-a456-426614174002"))
        mock_chroma.return_value.store_embedding = AsyncMock(return_value="test-embedding-id")

        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as pdf_file:
            pdf_file.write(b"%PDF-1.4\nTest PDF content")
            pdf_file.flush()

            # Process document
            result = await process_document(pdf_file.name)

            # Verify result structure
            assert isinstance(result, dict)
            assert "document_id" in result
            assert "text" in result
            assert isinstance(result["document_id"], str)
            assert isinstance(result["text"], str)
            assert len(result["text"]) > 0

def test_entity_format():
    """Test entity format validation."""
    sample_entity = {
        "name": "CENTRAL INSTITUTION",
        "type": "ORGANIZATION",  # Changed from entity_type to type
        "description": "The Central Institution is the Federal Reserve"
    }
    entity = Entity(
        name=sample_entity["name"],
        type=sample_entity["type"],  # Changed from entity_type to type
        description=sample_entity["description"]
    )
    assert entity.name == "CENTRAL INSTITUTION"
    assert entity.type == "ORGANIZATION"  # Changed from entity_type to type

def test_relationship_format():
    """Test relationship format validation."""
    sample_relationship = {
        "source": "MARTIN SMITH",
        "target": "CENTRAL INSTITUTION",
        "relationship": "is Chair of",
        "relationship_strength": 9
    }
    relationship = Relationship(
        source=sample_relationship["source"],
        target=sample_relationship["target"],
        relationship=sample_relationship["relationship"],
        relationship_strength=sample_relationship["relationship_strength"],
        relationship_id=UUID("123e4567-e89b-12d3-a456-426614174001"),
        document_id=UUID("123e4567-e89b-12d3-a456-426614174002"),
        properties={},
        labels=[]
    )
    assert relationship.source == "MARTIN SMITH"
    assert relationship.target == "CENTRAL INSTITUTION"
    assert relationship.relationship_strength == 9
