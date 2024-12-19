"""Tests for task queue functionality."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.tasks import workflow, celery_app, document
from celery.result import AsyncResult
import json
from http import HTTPStatus

@pytest.fixture
async def mock_databases():
    """Mock database connections."""
    with patch('app.tasks.document.get_sync_graph_db') as mock_graph_db, \
         patch('app.tasks.document.get_sync_vector_db') as mock_vector_db, \
         patch('app.tasks.document.get_sync_relational_db') as mock_relational_db:

        mock_graph_db.return_value = MagicMock()
        mock_vector_db.return_value = MagicMock()
        mock_relational_db.return_value = MagicMock()

        yield {
            'graph': mock_graph_db.return_value,
            'vector': mock_vector_db.return_value,
            'relational': mock_relational_db.return_value
        }

@pytest.fixture
async def mock_celery_result():
    """Create a mock Celery result factory."""
    def create_mock(doc_id=None):
        async def mock_get():
            return {
                "status": "completed",
                "document_id": doc_id or "test-doc-id",
                "task_id": "test-task-id",
                "entities": [
                    {"name": "TEST_ENTITY", "type": "CONCEPT", "description": "Test entity"}
                ],
                "relationships": []
            }

        mock_result = AsyncMock(spec=AsyncResult)
        mock_result.get = mock_get
        return mock_result
    return create_mock

@pytest.mark.asyncio
async def test_document_processing(mock_databases, mock_celery_result):
    """Test document processing workflow."""
    document_id = "test-doc-id"

    with patch('app.tasks.document.QwenClient') as MockQwenClient, \
         patch('app.tasks.document.process_document') as mock_process:

        # Setup QwenClient mock
        mock_qwen_instance = AsyncMock()
        mock_qwen_instance.extract_entities.return_value = [
            {"name": "TEST_ENTITY", "type": "CONCEPT", "description": "Test entity"}
        ]
        mock_qwen_instance.extract_relationships.return_value = []
        MockQwenClient.return_value = mock_qwen_instance

        # Setup process mock
        mock_process.apply_async.return_value = mock_celery_result(document_id)

        # Test
        result = await workflow.process_document_workflow(document_id)

        assert isinstance(result, dict)
        assert result["status"] == "completed"
        assert result["document_id"] == document_id
        assert "entities" in result
        assert len(result["entities"]) > 0

        # Verify database interactions
        mock_databases['graph'].store_entity.assert_called()
        mock_databases['vector'].store_embedding.assert_called()

@pytest.mark.asyncio
async def test_batch_document_processing(mock_databases, mock_celery_result):
    """Test batch document processing workflow."""
    document_ids = ["test-doc-1", "test-doc-2"]

    with patch('app.tasks.document.QwenClient') as MockQwenClient, \
         patch('app.tasks.document.process_document') as mock_process:

        # Setup QwenClient mock
        mock_qwen_instance = AsyncMock()
        mock_qwen_instance.extract_entities_batch.return_value = [
            [{"name": "TEST_ENTITY", "type": "CONCEPT", "description": "Test entity"}]
        ] * len(document_ids)
        mock_qwen_instance.extract_relationships_batch.return_value = [[]] * len(document_ids)
        MockQwenClient.return_value = mock_qwen_instance

        # Setup mocks for each document ID
        mock_process.apply_async.side_effect = [
            mock_celery_result(doc_id) for doc_id in document_ids
        ]

        # Test
        results = await workflow.process_documents_batch(document_ids)

        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)
        assert all(r["status"] == "completed" for r in results)
        assert all(r["document_id"] in document_ids for r in results)
        assert all("entities" in r for r in results)
        assert all(len(r["entities"]) > 0 for r in results)

@pytest.mark.asyncio
async def test_task_progress_tracking(mock_databases, mock_celery_result):
    """Test task progress tracking functionality."""
    document_id = "test-doc-progress"

    with patch('app.tasks.document.QwenClient') as MockQwenClient, \
         patch('app.tasks.document.process_document') as mock_process:

        # Setup QwenClient mock
        mock_qwen_instance = AsyncMock()
        mock_qwen_instance.extract_entities.return_value = [
            {"name": "TEST_ENTITY", "type": "CONCEPT", "description": "Test entity"}
        ]
        mock_qwen_instance.extract_relationships.return_value = []
        MockQwenClient.return_value = mock_qwen_instance

        # Setup mock with specific document ID
        mock_process.apply_async.return_value = mock_celery_result(document_id)

        # Test
        result = await workflow.process_document_workflow(document_id)

        assert isinstance(result, dict)
        assert result["status"] == "completed"
        assert result["document_id"] == document_id
        assert "entities" in result
        assert len(result["entities"]) > 0
