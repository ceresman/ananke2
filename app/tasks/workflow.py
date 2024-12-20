"""Task workflow manager for Ananke2.

This module implements Celery tasks for managing document processing workflows
in the Ananke2 knowledge framework. It orchestrates:
- Single document processing pipelines
- Batch document processing
- Task chaining and error handling
- Result aggregation and status tracking
"""

from typing import Dict, Any, List
from celery import shared_task
from . import celery_app, document

@celery_app.task(name='workflow.process_document_workflow')
def process_document_workflow(document_id: str) -> Dict[str, Any]:
    """Execute complete document processing workflow for a single arXiv paper.

    Orchestrates the full document processing pipeline by chaining multiple tasks:
    1. Download arXiv paper and store metadata
    2. Process PDF and extract text content
    3. Extract knowledge graph (entities and relationships)
    4. Generate and store content embeddings

    Args:
        document_id (str): arXiv paper identifier (e.g., "2101.00123")

    Returns:
        Dict[str, Any]: Contains:
            - status (str): "completed" or "failed"
            - task_id (str): Celery task ID
            - document_id (str): Input document ID
            - entities (List[dict]): Extracted entities if successful
            - relationships (List[dict]): Extracted relationships if successful
            - error (str, optional): Error message if failed

    Example:
        ```python
        # Process single arXiv paper
        result = process_document_workflow.delay("2101.00123").get()

        if result['status'] == 'completed':
            print(f"Extracted {len(result['entities'])} entities")
        else:
            print(f"Processing failed: {result.get('error')}")
        ```
    """
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
    """Process multiple arXiv papers in batch.

    Executes the complete document processing workflow for multiple papers
    in parallel using Celery's task distribution. Each paper is processed
    independently, and failures in one document don't affect others.

    Args:
        document_ids (List[str]): List of arXiv paper identifiers

    Returns:
        List[Dict[str, Any]]: List of results, one per document, each containing:
            - status (str): "completed" or "failed"
            - document_id (str): Input document ID
            - entities (List[dict], optional): Extracted entities if successful
            - relationships (List[dict], optional): Extracted relationships if successful
            - error (str, optional): Error message if failed

    Example:
        ```python
        # Process multiple papers in batch
        papers = ["2101.00123", "2101.00124", "2101.00125"]
        results = process_documents_batch.delay(papers).get()

        # Analyze results
        success_count = sum(1 for r in results if r['status'] == 'completed')
        print(f"Successfully processed {success_count}/{len(papers)} papers")
        ```
    """
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
