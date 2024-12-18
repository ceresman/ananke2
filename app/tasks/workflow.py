"""Task workflow manager for Ananke2."""

from typing import Dict, Any
from celery import chain
from . import celery_app, document

@celery_app.task(name='app.tasks.workflow.process_arxiv_workflow', bind=True)
def process_arxiv_workflow(self, arxiv_id: str) -> Dict[str, Any]:
    """Process arXiv paper workflow.

    Args:
        arxiv_id: ArXiv paper ID (e.g., '2404.16130')

    Returns:
        Dict containing processing results
    """
    try:
        self.update_state(state='PROCESSING',
                         meta={'progress': 0, 'current_operation': 'Starting arXiv processing'})

        # Chain arXiv processing tasks
        workflow = chain(
            document.download_arxiv.s(arxiv_id=arxiv_id),
            document.process_document.s(),
            document.extract_knowledge_graph.s(),
            document.extract_content.s()
        )
        result = workflow.apply_async()

        self.update_state(state='COMPLETED',
                         meta={'progress': 100, 'current_operation': 'ArXiv processing completed'})

        return {
            'status': 'COMPLETED',
            'task_id': result.id,
            'arxiv_id': arxiv_id
        }

    except Exception as e:
        self.update_state(state='FAILED',
                         meta={'progress': 0, 'current_operation': 'Failed', 'error': str(e)})
        raise
