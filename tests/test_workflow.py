"""Test the arXiv document processing workflow."""
import os
import pytest
from app.tasks.workflow import process_document_workflow
from app.database.sync_wrappers import get_sync_relational_db, get_sync_vector_db, get_sync_graph_db

def test_arxiv_workflow():
    """Test the complete arXiv document processing workflow."""
    # Test with GPT-3 paper
    arxiv_id = "2005.14165"
    print(f"Processing arXiv paper {arxiv_id}...")

    # Run workflow
    result = process_document_workflow.delay(arxiv_id)
    task_id = result.id
    print(f"Task ID: {task_id}")

    # Wait for result
    print("Waiting for result...")
    task_result = result.get(timeout=300)  # Wait up to 5 minutes for completion
    print("Processing completed successfully!")
    print(f"Result: {task_result}")
    print()

    # Verify database writes
    print("Verifying database writes...")
    try:
        # Get database clients using sync wrappers
        rel_db = get_sync_relational_db()
        graph_db = get_sync_graph_db()
        vector_db = get_sync_vector_db()

        # Check document in relational DB
        docs = rel_db.list_documents()
        doc_count = len([d for d in docs if d.get("data_type") == "arxiv_paper"])
        assert doc_count > 0, "No arXiv papers found in relational database"

        # Check entities in graph DB
        entities = graph_db.list_entities()
        entity_count = len(entities)
        assert entity_count > 0, "No entities found in graph database"

        # Check embeddings in vector DB
        embeddings = vector_db.list_embeddings()
        embedding_count = len(embeddings)
        assert embedding_count > 0, "No embeddings found in vector database"

        print("Database verification successful!")
        print(f"Found {doc_count} papers, {entity_count} entities, and {embedding_count} embeddings")

    except Exception as e:
        print(f"Error processing arXiv paper: {str(e)}")
        raise

    return task_result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
