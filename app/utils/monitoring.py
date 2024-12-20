"""Task monitoring utilities for Ananke2.

This module provides utilities for monitoring Celery tasks and workers in the
Ananke2 knowledge framework. It enables:
- Real-time task progress tracking
- Worker status monitoring
- Task state inspection
- Error tracking and reporting

The monitoring utilities are essential for:
- Debugging task execution issues
- Load balancing across workers
- Identifying bottlenecks in document processing
- Tracking long-running knowledge extraction tasks
"""

from typing import Dict, Any, Optional
from ..tasks import celery_app

def get_active_tasks() -> Dict[str, Any]:
    """Get currently active tasks from all Celery workers.

    Retrieves a snapshot of all tasks currently being executed across
    all registered Celery workers. This is useful for monitoring system
    load and identifying potential bottlenecks.

    Returns:
        Dict[str, Any]: Dictionary mapping worker hostnames to their active tasks.
            Each task entry contains:
            - id (str): Task ID
            - name (str): Task name
            - args (List): Task arguments
            - kwargs (Dict): Task keyword arguments
            - started (float): Timestamp when task started

    Example:
        ```python
        # Get active tasks and print count per worker
        active = get_active_tasks()
        for worker, tasks in active.items():
            print(f"Worker {worker} has {len(tasks)} active tasks")
        ```

    Note:
        Returns empty dict if no workers are available or if inspector
        fails to communicate with workers.
    """
    inspector = celery_app.control.inspect()
    return inspector.active() or {}

def get_task_progress(task_id: str) -> Dict[str, Any]:
    """Get detailed progress information for a specific task.

    Retrieves comprehensive status information about a task, including
    its current state, progress percentage, and any errors encountered.
    Particularly useful for monitoring long-running document processing
    and knowledge extraction tasks.

    Args:
        task_id (str): Unique identifier of the task to check

    Returns:
        Dict[str, Any]: Task progress information containing:
            - status (str): Current task state (PENDING, STARTED, SUCCESS, etc.)
            - info (Dict): Task-specific information
            - progress (float): Progress percentage (0-100)
            - current_operation (str, optional): Description of current operation
            - errors (List[str]): List of encountered errors

    Example:
        ```python
        # Monitor task progress with error handling
        progress = get_task_progress("task-123")
        if progress['status'] == 'FAILURE':
            print(f"Task failed with errors: {progress['errors']}")
        else:
            print(f"Task is {progress['progress']}% complete")
            if progress['current_operation']:
                print(f"Currently: {progress['current_operation']}")
        ```
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
    """Get comprehensive statistics about all Celery workers.

    Collects detailed statistics about registered Celery workers, including
    their active, scheduled, and reserved tasks. This information is crucial
    for monitoring system health and load distribution.

    Returns:
        Dict[str, Any]: Worker statistics containing:
            - active (Dict): Currently executing tasks per worker
            - scheduled (Dict): Tasks scheduled for future execution
            - reserved (Dict): Tasks reserved by workers but not yet started
            - revoked (Dict): Tasks that were revoked
            - registered (Dict): All task types registered with each worker

    Example:
        ```python
        # Monitor worker load and task distribution
        stats = get_worker_stats()

        # Check active tasks per worker
        for worker, tasks in stats['active'].items():
            print(f"{worker}: {len(tasks)} active tasks")

        # Check scheduled tasks
        total_scheduled = sum(len(tasks) for tasks in stats['scheduled'].values())
        print(f"Total scheduled tasks: {total_scheduled}")
        ```

    Note:
        All dictionary values default to empty dict if the corresponding
        information cannot be retrieved from workers.
    """
    inspector = celery_app.control.inspect()
    return {
        'active': inspector.active() or {},
        'scheduled': inspector.scheduled() or {},
        'reserved': inspector.reserved() or {},
        'revoked': inspector.revoked() or {},
        'registered': inspector.registered() or {}
    }
