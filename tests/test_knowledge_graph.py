"""Tests for knowledge graph extraction functionality."""

import pytest
from app.utils.qwen import QwenClient
from app.tasks import document
from app.tasks.workflow import process_document_workflow

EXAMPLE_TEXT = """The Verdantis's Central Institution is scheduled to meet on Monday and Thursday,
with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT,
followed by a press conference where Central Institution Chair Martin Smith will take questions.
Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%."""

def test_qwen_client_initialization():
    """Test QwenClient initialization."""
    client = QwenClient()
    assert client is not None
    assert client.max_retries == 3
    assert client.retry_delay == 1

def test_entity_extraction():
    """Test entity extraction from example text."""
    client = QwenClient()
    entities = client.extract_entities(EXAMPLE_TEXT)

    # Verify we got some entities
    assert len(entities) > 0
    assert all(isinstance(e, dict) for e in entities)

    # Check entity format
    for entity in entities:
        if "type" in entity:  # Entity object
            assert "name" in entity
            assert "type" in entity
            assert "description" in entity
            assert entity["name"].isupper()  # Name should be capitalized
            assert entity["type"] in ["PERSON", "ORGANIZATION", "GEO", "EVENT", "CONCEPT"]
            assert isinstance(entity["description"], str)
        else:  # Relationship object
            assert "source" in entity
            assert "target" in entity
            assert "relationship" in entity
            assert "relationship_strength" in entity
            assert isinstance(entity["relationship_strength"], int)
            assert 1 <= entity["relationship_strength"] <= 10

def test_knowledge_graph_task():
    """Test knowledge graph extraction task."""
    result = document.extract_knowledge_graph({
        "document_id": "test-doc",
        "content": EXAMPLE_TEXT,
        "status": "completed"
    })

    assert result["status"] == "completed"
    assert result["document_id"] == "test-doc"
    assert "entities" in result
    assert isinstance(result["entities"], list)
    assert len(result["entities"]) > 0

@pytest.mark.asyncio
async def test_document_workflow():
    """Test complete document processing workflow with KG extraction."""
    workflow = process_document_workflow("test-doc")
    assert workflow is not None

    # Verify workflow includes knowledge graph extraction
    task_names = [task.name for task in workflow.tasks]
    assert "app.tasks.document.extract_knowledge_graph" in task_names
    assert task_names.index("app.tasks.document.extract_content") < \
           task_names.index("app.tasks.document.extract_knowledge_graph")

def test_arxiv_paper_extraction():
    """Test knowledge graph extraction on a real arXiv paper."""
    import PyPDF2
    import os

    # Read PDF content
    pdf_path = os.path.join(os.path.dirname(__file__), "data", "sample_paper.pdf")
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        content = ""
        for page in pdf_reader.pages[:2]:  # Process first two pages for test
            content += page.extract_text()

    # Process through knowledge graph extraction
    result = document.extract_knowledge_graph({
        "document_id": "arxiv-1706.03762",
        "content": content,
        "status": "completed"
    })

    # Verify result structure
    assert result["status"] == "completed"
    assert result["document_id"] == "arxiv-1706.03762"
    assert "entities" in result
    entities = result["entities"]

    # Verify entities and relationships format
    for item in entities:
        if "type" in item:  # Entity
            assert set(item.keys()) == {"name", "type", "description"}
            assert item["name"].isupper()
            assert item["type"] in ["PERSON", "ORGANIZATION", "GEO", "EVENT", "CONCEPT"]
            assert len(item["description"]) > 0
        else:  # Relationship
            assert set(item.keys()) == {"source", "target", "relationship", "relationship_strength"}
            assert isinstance(item["relationship_strength"], int)
            assert 1 <= item["relationship_strength"] <= 10
            assert len(item["relationship"]) > 0

    # Verify we found some specific expected entities (from the paper)
    entity_names = {e["name"] for e in entities if "type" in e}
    expected_names = {"TRANSFORMER", "ATTENTION", "NEURAL NETWORK"}
    assert any(name in entity_names for name in expected_names), "Should find some key technical concepts"

    # Verify relationships between entities
    relationships = [e for e in entities if "source" in e]
    assert len(relationships) > 0, "Should find relationships between entities"
