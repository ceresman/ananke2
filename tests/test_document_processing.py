import pytest
from unittest.mock import patch, MagicMock
import os
import tempfile
from uuid import UUID
from app.tasks.document import process_document
from app.utils.qwen import QwenClient
from app.models.entities import Entity, Relationship

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
    class MockRelationalDB:
        def __init__(self):
            self.documents = {}

        def get_document(self, doc_id):
            return self.documents.get(doc_id)

        def update_document(self, doc_id, status, metadata=None):
            if doc_id not in self.documents:
                self.documents[doc_id] = {}
            self.documents[doc_id].update({
                "status": status,
                **(metadata or {})
            })
            return True

    class MockGraphDB:
        def create_entity(self, entity):
            return {"id": "test-entity-id"}

        def create_relationship(self, relationship):
            return {"id": "test-relationship-id"}

    class MockVectorDB:
        def store_embedding(self, doc_id, embedding):
            return True

        def search_similar(self, embedding, limit=10):
            return [{"id": "test-doc", "score": 0.9}]

    return {
        "relational": MockRelationalDB(),
        "graph": MockGraphDB(),
        "vector": MockVectorDB()
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
                    "target": "TEST_ENTITY_2",
                    "relationship": "test relationship",
                    "relationship_strength": 5
                }
            ]

        async def generate_embedding(self, text):
            return [0.1] * 768

    # Create temporary PDF file
    with tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_file:
        pdf_file.write(b"%PDF-1.4\nTest PDF content")
        pdf_file.flush()

        # Mock database interfaces
        with patch("app.tasks.document.QwenClient", return_value=MockQwenClient()), \
             patch("app.tasks.document.AsyncRelationalDatabase"), \
             patch("app.tasks.document.Neo4jInterface"), \
             patch("app.tasks.document.ChromaInterface"), \
             patch("app.tasks.document.process_pdf", return_value="Test content"):

            # Process document
            result = await process_document(pdf_file.name)

            # Verify result structure
            assert "document_id" in result
            assert "entities" in result
            assert "relationships" in result
            assert len(result["entities"]) > 0
            assert len(result["relationships"]) > 0

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
