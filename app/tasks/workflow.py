"""Task workflow manager for Ananke2."""

from typing import Dict, Any
from celery import chain
from . import celery_app, document

@celery_app.task(name='app.tasks.workflow.process_document_workflow', bind=True)
def process_document_workflow(self, document_path: str) -> Dict[str, Any]:
    """Process document workflow.

    Args:
        document_path: Path to the document file

    Returns:
        Dict containing processing results
    """
    try:
        self.update_state(state='PROCESSING',
                         meta={'progress': 0, 'current_operation': 'Starting document processing'})

        # Chain document processing tasks using proper Celery chain syntax
        # First task gets argument directly, subsequent tasks receive previous task's output
        workflow = chain(
            document.process_document.s(str(document_path)),  # First task needs argument
            document.extract_knowledge_graph.s(),  # Gets doc_id from process_document
            document.extract_content.s()  # Gets doc_id from extract_knowledge_graph
        )
        result = workflow.apply_async()

        self.update_state(state='COMPLETED',
                         meta={'progress': 100, 'current_operation': 'Document processing completed'})

        return {
            'status': 'COMPLETED',
            'task_id': result.id
        }

    except Exception as e:
        self.update_state(state='FAILED',
                         meta={'progress': 0, 'current_operation': 'Failed', 'error': str(e)})
        raise
