"""Task workflow manager for Ananke2."""

from typing import Dict, Any, List
from celery import shared_task
from . import celery_app, document

@celery_app.task(name='workflow.process_document_workflow')
def process_document_workflow(document_id: str) -> Dict[str, Any]:
    """Process arXiv paper workflow."""
    try:
        # Chain arXiv processing tasks
        result = document.download_arxiv.delay(arxiv_id=document_id)
        download_result = result.get()

        process_result = document.process_document.delay(download_result["pdf_path"]).get()
        kg_result = document.extract_knowledge_graph.delay(process_result).get()
        content_result = document.extract_content.delay(process_result).get()

        return {
            'status': 'completed',
            'task_id': result.id,
            'document_id': document_id,
            'entities': kg_result.get('entities', []),
            'relationships': kg_result.get('relationships', [])
        }

    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e),
            'document_id': document_id
        }

@celery_app.task(name='workflow.process_documents_batch')
def process_documents_batch(document_ids: List[str]) -> List[Dict[str, Any]]:
    """Process multiple documents in batch."""
    results = []
    for doc_id in document_ids:
        try:
            result = process_document_workflow.delay(doc_id).get()
            results.append(result)
        except Exception as e:
            results.append({
                'status': 'failed',
                'error': str(e),
                'document_id': doc_id
            })
    return results
