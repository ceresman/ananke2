"""Task management router for Ananke2."""

from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import Dict, Any
from pydantic import BaseModel
from ..tasks import workflow, celery_app
import uuid

router = APIRouter()  # Remove prefix since it's set in main.py

class DocumentRequest(BaseModel):
    """Request model for document processing."""
    document_path: str

class TaskResponse(BaseModel):
    task_id: str
    status: str

@router.post("/process-document", response_model=TaskResponse)
async def start_document_processing(request: DocumentRequest) -> Dict[str, Any]:
    """Start document processing workflow.

    Args:
        request: The document processing request containing the document path

    Returns:
        Dict containing the task ID
    """
    try:
        # Start document processing task
        task = workflow.process_document_workflow.delay(
            document_path=request.document_path
        )

        return {
            "task_id": task.id,
            "status": "PENDING"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start document processing: {str(e)}"
        )

@router.post("/upload-document")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, str]:
    """Upload and process a document.

    Args:
        file: The uploaded file

    Returns:
        Dict containing the task ID
    """
    try:
        # Save uploaded file
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Start processing task
        task = workflow.process_document_workflow.delay(
            document_path=file_path
        )
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process uploaded document: {str(e)}"
        )

@router.get("/{task_id}")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get task status.

    Args:
        task_id: The ID of the task to check

    Returns:
        Dict containing task status information
    """
    try:
        result = celery_app.AsyncResult(task_id)
        response = {
            "task_id": task_id,
            "status": result.status,
        }

        # Handle different result states
        if result.ready():
            if result.successful():
                response["result"] = result.get()
            else:
                response["error"] = str(result.result)
        else:
            response["info"] = result.info if isinstance(result.info, dict) else str(result.info)

        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}"
        )

@router.get("/{task_id}/progress")
async def get_task_progress(task_id: str) -> Dict[str, Any]:
    """Get detailed task progress information.

    Args:
        task_id: The ID of the task to check

    Returns:
        Dict containing detailed progress information
    """
    try:
        result = celery_app.AsyncResult(task_id)
        response = {
            "task_id": task_id,
            "status": result.status,
        }

        # Add progress info if available
        if isinstance(result.info, dict):
            response.update({
                "progress": result.info.get('progress', 0),
                "current_operation": result.info.get('current_operation'),
                "errors": result.info.get('errors', [])
            })

        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task progress: {str(e)}"
        )
