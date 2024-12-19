"""Document processing tasks for Ananke2."""

import os
from typing import Dict, Any, Optional
from unstructured.partition.pdf import partition_pdf
import dashscope
from http import HTTPStatus
from celery import shared_task
import arxiv
import json
from ..utils.qwen import QwenClient
from ..database.sync_wrappers import get_sync_relational_db, get_sync_vector_db, get_sync_graph_db
from ..models.entities import Entity, Relationship
from ..config import settings

# Initialize Qwen client
qwen_client = QwenClient(api_key=settings.QWEN_API_KEY)

@shared_task(name='document.download_arxiv')
def download_arxiv(arxiv_id: str) -> dict:
    """Download arXiv paper and store metadata."""
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
def process_document(document_path: str) -> dict:
    """Process PDF document and extract text content."""
    print(f"Starting document processing for {document_path}")
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"PDF file not found: {document_path}")

    try:
        print("Opening PDF document...")
        # Process PDF using unstructured library directly
        elements = partition_pdf(filename=document_path)
        text = "\n".join([str(element) for element in elements])
        print(f"Extracted {len(text)} characters of text")

        # Store document metadata
        print("Storing document in relational database...")
        rel_db = get_sync_relational_db()
        from uuid import uuid4
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
def extract_knowledge_graph(result_dict: dict) -> dict:
    """Extract knowledge graph from document text."""
    doc_id = result_dict.get("doc_id")
    text = result_dict.get("text")

    print(f"Starting knowledge graph extraction for document {doc_id}")
    try:
        # Extract entities and relationships using Qwen format
        prompt = """Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

For each identified entity, extract:
- entity_name: Name of the entity, capitalized
- entity_type: One of [PERSON, ORGANIZATION, GEO, EVENT, CONCEPT]
- entity_description: Comprehensive description of the entity's attributes and activities

For each relationship between entities, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: explanation of the relationship
- relationship_strength: integer score 1-10 indicating strength

Format as JSON:
{"entities": [{"name": <name>, "type": <type>, "description": <description>}],
 "relationships": [{"source": <source>, "target": <target>, "relationship": <description>, "relationship_strength": <strength>}]}

Text:
{text}

Just return the JSON output, nothing else."""

        # Call Qwen API
        resp = dashscope.Generation.call(
            model="qwen-max",
            prompt=prompt.format(text=text),
            result_format='message'
        )

        if resp.status_code != HTTPStatus.OK:
            raise Exception(f"Failed to extract knowledge graph: {resp.message}")

        try:
            # Parse results
            result = json.loads(resp.output.choices[0].message.content)
            if not isinstance(result, dict) or "entities" not in result:
                raise ValueError("Invalid response format from Qwen API")

            entities = result.get("entities", [])
            relationships = result.get("relationships", [])
        except json.JSONDecodeError:
            raise Exception("Failed to parse Qwen API response")
        except KeyError as e:
            raise Exception(f"Missing required field in response: {str(e)}")

        # Store in graph database
        graph_db = get_sync_graph_db()
        for entity in entities:
            graph_db.store_entity(Entity(**entity))
        for rel in relationships:
            graph_db.store_relationship(Relationship(**rel))

        return {
            "doc_id": doc_id,
            "entities": entities,
            "relationships": relationships
        }

    except Exception as e:
        print(f"Error extracting knowledge graph: {str(e)}")
        raise

@shared_task(name='document.extract_content')
def extract_content(result_dict: dict) -> dict:
    """Extract and store content in vector and relational databases."""
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
