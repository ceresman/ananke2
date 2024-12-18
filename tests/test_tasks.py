"""Tests for task queue functionality."""

import pytest
from app.tasks import workflow, celery_app
from celery.result import AsyncResult

@pytest.mark.asyncio
async def test_document_processing():
    """Test document processing workflow."""
    document_id = "test-doc-id"
    task = workflow.process_document_workflow(document_id)
    result = task.apply_async()
    assert isinstance(result, AsyncResult)

    # Wait for result with timeout
    task_result = result.get(timeout=10)
    assert task_result is not None
    assert task_result.get("status") == "completed"
    assert task_result.get("document_id") == document_id

@pytest.mark.asyncio
async def test_batch_document_processing():
    """Test batch document processing workflow."""
    document_ids = ["test-doc-1", "test-doc-2"]
    tasks = workflow.process_documents_batch(document_ids)

    # Start all tasks
    results = [task.apply_async() for task in tasks]
    assert all(isinstance(r, AsyncResult) for r in results)

    # Wait for all results
    task_results = [r.get(timeout=10) for r in results]
    assert all(r is not None for r in task_results)
    assert all(r.get("status") == "completed" for r in task_results)
    assert all(r.get("document_id") in document_ids for r in task_results)

@pytest.mark.asyncio
async def test_task_progress_tracking():
    """Test task progress tracking functionality."""
    document_id = "test-doc-progress"
    task = workflow.process_document_workflow(document_id)
    result = task.apply_async()

    # Get task status
    status = celery_app.AsyncResult(result.id)
    assert status.state in ['PENDING', 'STARTED', 'PROCESSING', 'SUCCESS', 'FAILURE']

    # Wait for completion
    task_result = result.get(timeout=10)
    assert task_result.get("status") == "completed"
