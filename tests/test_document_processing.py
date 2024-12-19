import pytest
import asyncio
from app.tasks.document import process_document, extract_knowledge_graph
from app.models.entities import Entity, Relationship
from app.database.sync_wrappers import GraphDatabase, VectorDatabase, RelationalDatabase
import os

@pytest.fixture
def sample_pdf():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_paper.pdf")

@pytest.fixture
def databases():
    """Initialize test databases with mock connections."""
    return {
        "graph": GraphDatabase(uri="bolt://localhost:7687", username="neo4j", password="test"),
        "vector": VectorDatabase(host="localhost", port=6379),
        "relational": RelationalDatabase(host="localhost", port=3306, database="test")
    }

def test_document_processing(sample_pdf, databases):
    """Test end-to-end document processing pipeline."""
    # Process document
    doc_id = process_document(sample_pdf)
    assert doc_id is not None

    # Verify document stored in relational DB
    doc = databases["relational"].get_document(doc_id)
    assert doc is not None
    assert doc["path"] == sample_pdf
    assert doc["status"] == "processed"

    # Extract knowledge graph
    result = extract_knowledge_graph(doc_id)
    assert result is not None

    # Verify entity format and storage
    entities = result.get("entities", [])
    assert len(entities) > 0
    for entity in entities:
        assert "name" in entity
        assert "type" in entity
        assert "description" in entity
        assert isinstance(entity["name"], str)
        assert len(entity["name"]) > 0

        # Verify entity in graph DB
        stored_entity = databases["graph"].get_entity(entity["name"])
        assert stored_entity is not None
        assert stored_entity.type == entity["type"]

        # Verify embedding in vector DB
        embedding = databases["vector"].get_embedding(entity["name"])
        assert embedding is not None

    # Verify relationship format and storage
    relationships = result.get("relationships", [])
    assert len(relationships) > 0
    for rel in relationships:
        assert "source" in rel
        assert "target" in rel
        assert "relationship" in rel
        assert "relationship_strength" in rel
        assert isinstance(rel["relationship_strength"], int)
        assert 1 <= rel["relationship_strength"] <= 10

        # Verify relationship in graph DB
        stored_rel = databases["graph"].get_relationship(rel["source"], rel["target"])
        assert stored_rel is not None
        assert stored_rel.relationship_strength == rel["relationship_strength"]

def test_entity_format():
    """Test entity format matches requirements."""
    sample_entity = {
        "name": "CENTRAL INSTITUTION",
        "type": "ORGANIZATION",
        "description": "The Central Institution is the Federal Reserve of Verdantis"
    }
    entity = Entity(**sample_entity)
    assert entity.dict() == sample_entity

def test_relationship_format():
    """Test relationship format matches requirements."""
    sample_relationship = {
        "source": "MARTIN SMITH",
        "target": "CENTRAL INSTITUTION",
        "relationship": "Martin Smith is the Chair",
        "relationship_strength": 9
    }
    relationship = Relationship(**sample_relationship)
    assert relationship.dict() == sample_relationship
