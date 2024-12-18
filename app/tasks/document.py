"""Document processing tasks for Ananke2."""

import os
from typing import Dict, Any, Optional
import asyncio
import fitz  # PyMuPDF
from celery import shared_task
from ..utils.qwen import QwenClient
from ..database.sync_wrappers import GraphDatabase, VectorDatabase, RelationalDatabase
from ..models.entities import Entity, Relationship
from ..config import settings

qwen_client = QwenClient(api_key=settings.QWEN_API_KEY)
graph_db = GraphDatabase(uri=settings.NEO4J_URI, username=settings.NEO4J_USER, password=settings.NEO4J_PASSWORD)
vector_db = VectorDatabase(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT, collection_name=settings.CHROMA_COLLECTION)
rel_db = RelationalDatabase(uri=f"{settings.MYSQL_HOST}:{settings.MYSQL_PORT}", username=settings.MYSQL_USER, password=settings.MYSQL_PASSWORD)

@shared_task(name='document.process_document')
def process_document(pdf_path: str) -> str:
    """Process PDF document and extract text content."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        # Store document metadata
        doc_id = rel_db.store_document({
            "path": pdf_path,
            "content": text,
            "status": "processed"
        })

        return str(doc_id)
    except Exception as e:
        raise Exception(f"Error processing document: {str(e)}")

@shared_task(name='document.extract_knowledge_graph')
def extract_knowledge_graph(doc_id: str) -> dict:
    """Extract knowledge graph from document text."""
    try:
        # Get document text from relational DB
        doc = rel_db.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document not found: {doc_id}")

        # Extract entities and relationships
        loop = asyncio.get_event_loop()
        entities = loop.run_until_complete(qwen_client.extract_entities(doc["content"]))
        relationships = loop.run_until_complete(qwen_client.extract_relationships(doc["content"]))

        # Store in graph database
        for entity in entities:
            graph_db.store_entity(Entity(**entity))

        for rel in relationships:
            graph_db.store_relationship(Relationship(**rel))

        # Generate and store embeddings
        embeddings = loop.run_until_complete(qwen_client.generate_embeddings_batch(
            [entity["description"] for entity in entities]
        ))

        # Store embeddings in vector database
        for entity, embedding in zip(entities, embeddings):
            vector_db.store_embedding(
                entity["name"],
                embedding,
                {"type": entity["type"], "description": entity["description"]}
            )

        return {
            "entities": entities,
            "relationships": relationships,
            "doc_id": doc_id
        }

    except Exception as e:
        raise Exception(f"Error extracting knowledge graph: {str(e)}")

@shared_task(name='document.extract_content')
def extract_content(doc_id: str) -> dict:
    """Extract and store content in vector and relational databases."""
    try:
        # Get document from relational DB
        doc = rel_db.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document not found: {doc_id}")

        # Generate embeddings for document content
        loop = asyncio.get_event_loop()
        content_embedding = loop.run_until_complete(qwen_client.generate_embeddings(doc["content"]))

        # Store in vector database
        vector_db.store_embedding(
            f"doc_{doc_id}",
            content_embedding,
            {"type": "document", "path": doc["path"]}
        )

        # Update document status
        rel_db.update_document(doc_id, {"status": "content_extracted"})

        return {
            "status": "success",
            "doc_id": doc_id,
            "embedding_id": f"doc_{doc_id}"
        }

    except Exception as e:
        raise Exception(f"Error extracting content: {str(e)}")
