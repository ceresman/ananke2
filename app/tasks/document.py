"""Document processing tasks for Ananke2."""

import os
from typing import Dict, Any, Optional
import asyncio
import fitz  # PyMuPDF
from celery import shared_task
from ..utils.qwen import QwenClient
from ..database.sync_wrappers import GraphDatabase, VectorDatabase, RelationalDatabase
from ..models.entities import Entity, Relationship
from ..config import Settings
from functools import lru_cache

# Initialize settings with Docker networking enabled for worker processes
settings = Settings(DOCKER_NETWORK=True)

# Initialize Qwen client
qwen_client = QwenClient(api_key=settings.QWEN_API_KEY)

@lru_cache()
def get_graph_db():
    """Get or create graph database connection."""
    db = GraphDatabase(
        uri=settings.get_neo4j_uri(),
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )
    return db

@lru_cache()
def get_vector_db():
    """Get or create vector database connection."""
    db = VectorDatabase(
        host=settings.get_chroma_host(),
        port=settings.CHROMA_PORT,
        collection_name=settings.CHROMA_COLLECTION
    )
    return db

@lru_cache()
def get_relational_db():
    """Get or create relational database connection."""
    db = RelationalDatabase(
        host=settings.get_mysql_host(),
        port=settings.MYSQL_PORT,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        database=settings.MYSQL_DATABASE
    )
    return db

@shared_task(name='document.process_document', bind=True)
def process_document(self, document_path: str) -> str:
    """Process PDF document and extract text content."""
    print(f"Starting document processing for {document_path}")
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"PDF file not found: {document_path}")

    try:
        print("Opening PDF document...")
        doc = fitz.open(document_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        print(f"Extracted {len(text)} characters of text")

        # Store document metadata
        print("Storing document in relational database...")
        rel_db = get_relational_db()
        doc_id = rel_db.store_document({
            "path": document_path,
            "content": text,
            "status": "processed"
        })
        print(f"Document stored with ID: {doc_id}")

        return {"doc_id": str(doc_id)}
    except Exception as e:
        print(f"Error in process_document: {str(e)}")
        raise Exception(f"Error processing document: {str(e)}")

@shared_task(name='document.extract_knowledge_graph', bind=True)
def extract_knowledge_graph(self, doc_id: str) -> dict:
    """Extract knowledge graph from document text."""
    print(f"Starting knowledge graph extraction for document {doc_id}")
    try:
        # Get document text from relational DB
        print("Retrieving document from relational database...")
        rel_db = get_relational_db()
        doc = rel_db.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document not found: {doc_id}")
        print(f"Retrieved document with {len(doc['content'])} characters")

        # Extract entities and relationships
        print("Extracting entities using Qwen API...")
        loop = asyncio.get_event_loop()
        entities = loop.run_until_complete(qwen_client.extract_entities(doc["content"]))
        print(f"Extracted {len(entities)} entities")

        print("Extracting relationships using Qwen API...")
        relationships = loop.run_until_complete(qwen_client.extract_relationships(doc["content"]))
        print(f"Extracted {len(relationships)} relationships")

        # Store in graph database
        print("Storing entities in graph database...")
        graph_db = get_graph_db()
        for entity in entities:
            graph_db.store_entity(Entity(**entity))

        print("Storing relationships in graph database...")
        for rel in relationships:
            graph_db.store_relationship(Relationship(**rel))

        # Generate and store embeddings
        print("Generating embeddings for entities...")
        embeddings = loop.run_until_complete(qwen_client.generate_embeddings_batch(
            [entity["description"] for entity in entities]
        ))
        print(f"Generated {len(embeddings)} embeddings")

        # Store embeddings in vector database
        print("Storing embeddings in vector database...")
        vector_db = get_vector_db()
        for entity, embedding in zip(entities, embeddings):
            vector_db.store_embedding(
                entity["name"],
                embedding,
                {"type": entity["type"], "description": entity["description"]}
            )
        print("Embeddings stored successfully")

        return {
            "entities": entities,
            "relationships": relationships,
            "doc_id": doc_id
        }

    except Exception as e:
        print(f"Error in extract_knowledge_graph: {str(e)}")
        raise Exception(f"Error extracting knowledge graph: {str(e)}")

@shared_task(name='document.extract_content', bind=True)
def extract_content(self, result: dict) -> dict:
    """Extract and store content in vector and relational databases."""
    print(f"Starting content extraction for document {result['doc_id']}")
    try:
        doc_id = result['doc_id']
        # Get document from relational DB
        print("Retrieving document from relational database...")
        rel_db = get_relational_db()
        doc = rel_db.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document not found: {doc_id}")
        print(f"Retrieved document with {len(doc['content'])} characters")

        # Generate embeddings for document content
        print("Generating embeddings for document content...")
        loop = asyncio.get_event_loop()
        content_embedding = loop.run_until_complete(qwen_client.generate_embeddings(doc["content"]))
        print("Generated document embedding")

        # Store in vector database
        print("Storing document embedding in vector database...")
        vector_db = get_vector_db()
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
