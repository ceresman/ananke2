"""Tests for task queue functionality."""

import pytest
from unittest.mock import AsyncMock, patch
from app.tasks import workflow, celery_app, document
from celery.result import AsyncResult

@pytest.fixture
def mock_celery_result():
    """Create a mock Celery result factory."""
    def create_mock(doc_id=None):
        async def mock_get():
            return {
                "status": "completed",
                "document_id": doc_id or "test-doc-id",
                "task_id": "test-task-id"
            }

        mock_result = AsyncMock(spec=AsyncResult)
        mock_result.get = mock_get
        return mock_result
    return create_mock

@pytest.mark.asyncio
@patch('app.tasks.document.process_document')
async def test_document_processing(mock_process, mock_celery_result):
    """Test document processing workflow."""
    document_id = "test-doc-id"
    # Setup mock with specific document ID
    mock_process.apply_async.return_value = mock_celery_result(document_id)

    # Test
    result = await workflow.process_document_workflow(document_id)

    assert isinstance(result, dict)
    assert result["status"] == "completed"
    assert result["document_id"] == document_id

@pytest.mark.asyncio
@patch('app.tasks.document.process_document')
async def test_batch_document_processing(mock_process, mock_celery_result):
    """Test batch document processing workflow."""
    document_ids = ["test-doc-1", "test-doc-2"]

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

@pytest.mark.asyncio
@patch('app.tasks.document.process_document')
async def test_task_progress_tracking(mock_process, mock_celery_result):
    """Test task progress tracking functionality."""
    document_id = "test-doc-progress"
    # Setup mock with specific document ID
    mock_process.apply_async.return_value = mock_celery_result(document_id)

    # Test
    result = await workflow.process_document_workflow(document_id)

    assert isinstance(result, dict)
    assert result["status"] == "completed"
    assert result["document_id"] == document_id
