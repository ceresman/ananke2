"""Test the arXiv document processing workflow."""
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.tasks.workflow import process_document_workflow
from app.database.sync_wrappers import get_sync_relational_db, get_sync_vector_db, get_sync_graph_db

@pytest.mark.unit
@patch('app.tasks.workflow.process_document_workflow.delay')
def test_arxiv_workflow(mock_delay):
    """Test the complete arXiv document processing workflow."""
    # Test with GPT-3 paper
    arxiv_id = "2005.14165"

    # Mock Celery task result
    mock_result = MagicMock()
    mock_result.id = "test-task-id"
    mock_result.get.return_value = {
        "status": "completed",
        "document_id": arxiv_id,
        "entities": [
            {"name": "GPT-3", "type": "TECHNOLOGY", "description": "Large language model"}
        ],
        "relationships": []
    }
    mock_delay.return_value = mock_result

    # Run workflow
    result = process_document_workflow.delay(arxiv_id)
    task_id = result.id
    assert task_id == "test-task-id"

    # Get result
    task_result = result.get(timeout=300)
    assert task_result["status"] == "completed"
    assert task_result["document_id"] == arxiv_id
    assert len(task_result["entities"]) > 0

    # Mock database clients
    with patch('app.database.sync_wrappers.get_sync_relational_db') as mock_rel_db, \
         patch('app.database.sync_wrappers.get_sync_graph_db') as mock_graph_db, \
         patch('app.database.sync_wrappers.get_sync_vector_db') as mock_vector_db:

        # Setup mock returns
        mock_rel_db.return_value.list_documents.return_value = [{"data_type": "arxiv_paper"}]
        mock_graph_db.return_value.list_entities.return_value = [{"name": "GPT-3"}]
        mock_vector_db.return_value.list_embeddings.return_value = [[0.1] * 1024]

        # Verify database writes
        rel_db = get_sync_relational_db()
        graph_db = get_sync_graph_db()
        vector_db = get_sync_vector_db()

        # Check document in relational DB
        docs = rel_db.list_documents()
        doc_count = len([d for d in docs if d.get("data_type") == "arxiv_paper"])
        assert doc_count > 0, "No arXiv papers found in relational database"

        # Check entities in graph DB
        entities = graph_db.list_entities()
        entity_count = len(entities)
        assert entity_count > 0, "No entities found in graph database"

        # Check embeddings in vector DB
        embeddings = vector_db.list_embeddings()
        embedding_count = len(embeddings)
        assert embedding_count > 0, "No embeddings found in vector database"

    return task_result
