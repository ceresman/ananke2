"""Task workflow manager for Ananke2."""

from celery import chain
from . import document

def process_document_workflow(document_id: str):
    """Create document processing workflow.

    Args:
        document_id: The ID of the document to process

    Returns:
        A Celery chain of tasks to process the document
    """
    return chain(
        document.process_document.s(document_id),
        document.extract_content.s(),
        document.extract_knowledge_graph.s(),
        document.process_math_expressions.s()
    )

def process_documents_batch(document_ids: list[str]):
    """Create parallel document processing workflows for a batch of documents.

    Args:
        document_ids: List of document IDs to process

    Returns:
        List of Celery chains for parallel processing
    """
    return [
        process_document_workflow(doc_id)
        for doc_id in document_ids
    ]
