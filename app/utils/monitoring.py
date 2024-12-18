"""Task monitoring utilities for Ananke2."""

from typing import Dict, Any, Optional
from ..tasks import celery_app

def get_active_tasks() -> Dict[str, Any]:
    """Get currently active tasks from all workers.

    Returns:
        Dict mapping worker names to their active tasks
    """
    inspector = celery_app.control.inspect()
    return inspector.active() or {}

def get_task_progress(task_id: str) -> Dict[str, Any]:
    """Get detailed task progress information.

    Args:
        task_id: The ID of the task to check

    Returns:
        Dict containing task status, info, and progress
    """
    result = celery_app.AsyncResult(task_id)
    info = result.info or {}

    return {
        'status': result.status,
        'info': info,
        'progress': getattr(info, 'progress', 0) if info else 0,
        'current_operation': info.get('current_operation', None) if isinstance(info, dict) else None,
        'errors': info.get('errors', []) if isinstance(info, dict) else []
    }

def get_worker_stats() -> Dict[str, Any]:
    """Get statistics about all Celery workers.

    Returns:
        Dict containing worker statistics
    """
    inspector = celery_app.control.inspect()
    return {
        'active': inspector.active() or {},
        'scheduled': inspector.scheduled() or {},
        'reserved': inspector.reserved() or {},
        'revoked': inspector.revoked() or {},
        'registered': inspector.registered() or {}
    }
