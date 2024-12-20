"""Document processing tasks for Ananke2.

This module implements Celery tasks for document processing in the Ananke2
knowledge framework. It handles:
- PDF document text extraction using unstructured.io
- arXiv paper downloading and metadata storage
- Knowledge graph extraction using Qwen API
- Content embedding generation and storage

The tasks form a processing pipeline that can be chained together:
1. download_arxiv -> process_document
2. process_document -> extract_knowledge_graph
3. process_document -> extract_content
"""

import os
from typing import Dict, Any, Optional
from unstructured.partition.pdf import partition_pdf
import dashscope
from http import HTTPStatus
from celery import shared_task
import arxiv
import json
from uuid import uuid4
from ..utils.qwen import QwenClient
from ..database.sync_wrappers import get_sync_relational_db, get_sync_vector_db, get_sync_graph_db
from ..database.relational import AsyncRelationalDatabase
from ..database.graph import Neo4jInterface
from ..database.vector import ChromaInterface
from ..models.entities import Entity, Relationship
from ..config import settings

# Initialize Qwen client
qwen_client = QwenClient(api_key=settings.QWEN_API_KEY)

async def process_pdf(pdf_path: str) -> str:
    """Process PDF file and extract text content.

    Uses unstructured.io's partition_pdf function to extract text
    content from PDF files while preserving document structure.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text content from the PDF

    Raises:
        FileNotFoundError: If PDF file does not exist
        Exception: If PDF processing fails

    Example:
        ```python
        text = await process_pdf("/path/to/paper.pdf")
        print(f"Extracted {len(text)} characters")
        ```
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        # Process PDF using unstructured library
        elements = partition_pdf(filename=pdf_path)
        text = "\n".join([str(element) for element in elements])
        return text
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise


@shared_task(name='document.download_arxiv')
def download_arxiv(arxiv_id: str) -> dict:
    """Download arXiv paper and store metadata.

    Downloads a paper from arXiv using its ID, stores the PDF locally,
    and saves metadata to the relational database. This task is typically
    the first step in processing arXiv papers.

    Args:
        arxiv_id (str): arXiv paper identifier (e.g., "2101.00123")

    Returns:
        dict: Contains:
            - doc_id (str): UUID of stored document
            - pdf_path (str): Path to downloaded PDF

    Raises:
        Exception: If paper download or metadata storage fails

    Example:
        ```python
        from celery import chain
        from .tasks import download_arxiv, process_document

        # Chain download and processing
        workflow = chain(
            download_arxiv.s("2101.00123"),
            process_document.s()
        )
        result = workflow.apply_async()
        ```
    """
    print(f"Downloading arXiv paper {arxiv_id}")

    try:
        # Create downloads directory
        download_dir = os.path.join(settings.DATA_DIR, 'arxiv')
        os.makedirs(download_dir, exist_ok=True)

        # Download paper using arxiv API
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())

        # Download PDF
        pdf_path = os.path.join(download_dir, f"{arxiv_id}.pdf")
        paper.download_pdf(filename=pdf_path)
        print(f"Downloaded PDF to {pdf_path}")

        # Store metadata
        rel_db = get_sync_relational_db()
        metadata = {
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "authors": [str(author) for author in paper.authors],
            "abstract": paper.summary,
            "categories": paper.categories,
            "pdf_path": pdf_path,
            "status": "downloaded"
        }
        doc_id = rel_db.store_document({
            "data_id": arxiv_id,
            "data_type": "arxiv_paper",
            "data_value": metadata
        })

        return {"doc_id": doc_id, "pdf_path": pdf_path}

    except Exception as e:
        print(f"Error downloading arXiv paper: {str(e)}")
        raise

@shared_task(name='document.process_document')
async def process_document(document_path: str) -> dict:
    """Process PDF document and extract text content.

    Extracts text content from a PDF document and stores it in the
    relational database with metadata. This task can be chained with
    either extract_knowledge_graph or extract_content for further processing.

    Args:
        document_path (str): Path to the PDF document or result dict from download_arxiv

    Returns:
        dict: Contains:
            - doc_id (str): UUID of processed document
            - text (str): Extracted text content

    Raises:
        FileNotFoundError: If document does not exist
        Exception: If processing fails

    Example:
        ```python
        # Process a local PDF
        result = await process_document("/path/to/paper.pdf")

        # Chain with knowledge graph extraction
        workflow = chain(
            process_document.s("/path/to/paper.pdf"),
            extract_knowledge_graph.s()
        )
        ```
    """
    print(f"Starting document processing for {document_path}")
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"PDF file not found: {document_path}")

    try:
        print("Processing PDF document...")
        # Process PDF using process_pdf function
        text = await process_pdf(document_path)
        print(f"Extracted {len(text)} characters of text")

        # Store document metadata
        print("Storing document in relational database...")
        rel_db = get_sync_relational_db()
        doc_id = uuid4()
        doc_data = {
            "data_id": doc_id,
            "data_type": "document",
            "data_value": {
                "path": document_path,
                "content": text,
                "status": "processed",
                "type": "document"
            }
        }
        rel_db.store_document(doc_data)
        print(f"Document stored with ID: {doc_id}")

        return {"doc_id": str(doc_id), "text": text}
    except Exception as e:
        print(f"Error in process_document: {str(e)}")
        raise

@shared_task(name='document.extract_knowledge_graph')
async def extract_knowledge_graph(result_dict: dict) -> dict:
    """Extract knowledge graph from document text.

    Uses Qwen API to extract entities and relationships from document text
    and stores them in the Neo4j graph database. This task is typically
    chained after process_document.

    Args:
        result_dict (dict): Contains:
            - doc_id (str): Document UUID
            - text (str): Document text content

    Returns:
        dict: Contains:
            - status (str): "completed" or "failed"
            - doc_id (str): Document UUID
            - entities (List[dict]): Extracted entities
            - relationships (List[dict]): Extracted relationships
            - error (str, optional): Error message if failed

    Example:
        ```python
        # Extract knowledge graph from processed document
        result = await extract_knowledge_graph({
            "doc_id": "123e4567-e89b-12d3-a456-426614174000",
            "text": "Einstein developed the theory of relativity..."
        })

        # Access extracted entities
        for entity in result['entities']:
            print(f"Found entity: {entity['name']} ({entity['type']})")
        ```
    """
    doc_id = result_dict.get("doc_id")
    text = result_dict.get("text")

    print(f"Starting knowledge graph extraction for document {doc_id}")
    try:
        # Initialize QwenClient
        client = QwenClient(api_key=settings.QWEN_API_KEY)

        # Extract entities and relationships using async calls
        entities = await client.extract_entities(text)
        relationships = await client.extract_relationships(text)

        # Store in graph database
        graph_db = get_sync_graph_db()
        for entity in entities:
            graph_db.store_entity(Entity(**entity))
        for rel in relationships:
            graph_db.store_relationship(Relationship(**rel))

        return {
            "status": "completed",
            "doc_id": doc_id,
            "entities": entities,
            "relationships": relationships
        }

    except Exception as e:
        print(f"Error extracting knowledge graph: {str(e)}")
        return {
            "status": "failed",
            "doc_id": doc_id,
            "error": str(e)
        }

@shared_task(name='document.extract_content')
def extract_content(result_dict: dict) -> dict:
    """Extract and store content in vector and relational databases.

    Generates embeddings for document content using DashScope's text-embedding-v3
    model and stores them in the Chroma vector database. Updates document status
    in MySQL database. This task is typically chained after process_document.

    Args:
        result_dict (dict): Contains:
            - doc_id (str): Document UUID
            - text (str): Document text content

    Returns:
        dict: Contains:
            - status (str): "success"
            - doc_id (str): Document UUID
            - embedding_id (str): ID of stored embedding

    Raises:
        ValueError: If document not found in database
        Exception: If embedding generation or storage fails

    Example:
        ```python
        # Generate and store embeddings for processed document
        result = await extract_content({
            "doc_id": "123e4567-e89b-12d3-a456-426614174000",
            "text": "Document content..."
        })

        # Chain processing and embedding generation
        workflow = chain(
            process_document.s("/path/to/paper.pdf"),
            extract_content.s()
        )
        ```
    """
    doc_id = result_dict.get("doc_id")
    text = result_dict.get("text")

    print(f"Starting content extraction for document {doc_id}")
    try:
        # Get document from relational DB
        print("Retrieving document from relational database...")
        rel_db = get_sync_relational_db()
        doc = rel_db.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document not found: {doc_id}")
        print(f"Retrieved document with {len(doc['content'])} characters")

        # Generate embeddings using dashscope
        print("Generating embeddings for document content...")
        resp = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v3,
            input=text,
            dimension=1024,
            output_type="dense"
        )
        if resp.status_code != HTTPStatus.OK:
            raise Exception("Failed to generate embeddings")
        content_embedding = resp.output["embeddings"][0]["embedding"]
        print("Generated document embedding")

        # Store in vector database
        print("Storing document embedding in vector database...")
        vector_db = get_sync_vector_db()
        vector_db.store_embedding(
            f"doc_{doc_id}",
            content_embedding,
            {"type": "document", "path": doc["path"]}
        )
        print("Document embedding stored successfully")

        # Update document status
        print("Updating document status...")
        rel_db.update_document(doc_id, {"status": "content_extracted"})
        print("Document status updated")

        return {
            "status": "success",
            "doc_id": doc_id,
            "embedding_id": f"doc_{doc_id}"
        }

    except Exception as e:
        print(f"Error in extract_content: {str(e)}")
        raise Exception(f"Error extracting content: {str(e)}")
