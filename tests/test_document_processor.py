"""Test document processor implementation."""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from app.processors.document import DocumentProcessor

def test_document_processor_init():
    """Test document processor initialization."""
    processor = DocumentProcessor()
    assert '.pdf' in processor.supported_types
    assert '.txt' in processor.supported_types
    assert '.docx' in processor.supported_types
    assert processor.qwen_api_key == 'sk-46e78b90eb8e4d6ebef79f265891f238'

@pytest.mark.asyncio
async def test_process_document(tmp_path):
    """Test document processing with sample text file."""
    # Create a sample text file
    test_file = tmp_path / "test.txt"
    test_content = "This is a test document.\nIt contains multiple lines.\n"
    test_file.write_text(test_content)

    processor = DocumentProcessor()
    elements = processor.process_document(str(test_file))

    assert len(elements) > 0
    assert "test document" in " ".join(elements).lower()

@pytest.mark.asyncio
async def test_extract_knowledge_graph_with_mock_api():
    """Test knowledge graph extraction with mocked Qwen API."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.output.text = json.dumps([
        {
            "name": "CENTRAL INSTITUTION",
            "type": "ORGANIZATION",
            "description": "The Central Institution is setting interest rates"
        },
        {
            "source": "MARTIN SMITH",
            "target": "CENTRAL INSTITUTION",
            "relationship": "Chair of the institution",
            "relationship_strength": 9
        }
    ])

    processor = DocumentProcessor()
    text_elements = [
        "The Central Institution is scheduled to meet on Monday.",
        "Martin Smith, the chair, will take questions at the press conference."
    ]

    with patch('dashscope.Generation.call', return_value=mock_response):
        graph_data = processor.extract_knowledge_graph(text_elements)

    assert "entities" in graph_data
    assert "relationships" in graph_data
    assert len(graph_data["entities"]) > 0

    # Verify entity format
    entity = graph_data["entities"][0]
    assert "name" in entity
    assert "type" in entity
    assert "description" in entity
    assert entity["name"] == "CENTRAL INSTITUTION"
    assert entity["type"] == "ORGANIZATION"

    # Verify relationship format
    rel = graph_data["relationships"][0]
    assert "source" in rel
    assert "target" in rel
    assert "relationship" in rel
    assert "relationship_strength" in rel
    assert rel["source"] == "MARTIN SMITH"
    assert rel["target"] == "CENTRAL INSTITUTION"
    assert isinstance(rel["relationship_strength"], int)
    assert 1 <= rel["relationship_strength"] <= 10

@pytest.mark.asyncio
async def test_save_knowledge_graph(tmp_path):
    """Test saving knowledge graph to file."""
    processor = DocumentProcessor()
    graph_data = {
        "entities": [{
            "name": "CENTRAL INSTITUTION",
            "type": "ORGANIZATION",
            "description": "The Central Institution is setting interest rates"
        }],
        "relationships": [{
            "source": "MARTIN SMITH",
            "target": "CENTRAL INSTITUTION",
            "relationship": "Chair of the institution",
            "relationship_strength": 9
        }]
    }

    output_file = tmp_path / "test_graph.json"
    processor.save_knowledge_graph(graph_data, str(output_file))

    assert output_file.exists()
    with open(output_file, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    assert loaded_data == graph_data
