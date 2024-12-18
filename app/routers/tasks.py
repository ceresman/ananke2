"""Task management router for Ananke2."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from ..tasks import workflow, celery_app

router = APIRouter(tags=["tasks"])

@router.post("/tasks/process-document/{document_id}")
async def start_document_processing(document_id: str) -> Dict[str, str]:
    """Start document processing workflow.

    Args:
        document_id: The ID of the document to process

    Returns:
        Dict containing the task ID
    """
    try:
        task = workflow.process_document_workflow(document_id)
        result = task.apply_async()
        return {"task_id": result.id}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start document processing: {str(e)}"
        )

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get task status.

    Args:
        task_id: The ID of the task to check

    Returns:
        Dict containing task status information
    """
    try:
        result = celery_app.AsyncResult(task_id)
        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result,
            "progress": result.info.get('progress', 0) if result.info else 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}"
        )

@router.get("/tasks/{task_id}/progress")
async def get_task_progress(task_id: str) -> Dict[str, Any]:
    """Get detailed task progress information.

    Args:
        task_id: The ID of the task to check

    Returns:
        Dict containing detailed progress information
    """
    try:
        result = celery_app.AsyncResult(task_id)
        return {
            "task_id": task_id,
            "status": result.status,
            "progress": result.info.get('progress', 0) if result.info else 0,
            "current_operation": result.info.get('current_operation', None) if result.info else None,
            "errors": result.info.get('errors', []) if result.info else []
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task progress: {str(e)}"
        )
