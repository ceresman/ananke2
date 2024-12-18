"""Test document processing workflow."""

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tasks.workflow import process_document_workflow
from app.database.graph import AsyncGraphDatabase
from app.database.vector import AsyncVectorDatabase
from app.database.relational import AsyncRelationalDatabase
from app.config import Settings

settings = Settings()

async def verify_database_writes():
    """Verify that data was written to all databases."""
    print("\nVerifying database writes...")

    # Check Neo4j
    graph_db = AsyncGraphDatabase(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )
    await graph_db.connect()
    entities = await graph_db.get_all_entities()
    relationships = await graph_db.get_all_relationships()
    print(f"Neo4j - Entities: {len(entities)}, Relationships: {len(relationships)}")
    await graph_db.close()

    # Check ChromaDB
    vector_db = AsyncVectorDatabase(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        collection_name=settings.CHROMA_COLLECTION
    )
    await vector_db.connect()
    embeddings = await vector_db.get_all_embeddings()
    print(f"ChromaDB - Embeddings: {len(embeddings)}")
    await vector_db.close()

    # Check MySQL
    relational_db = AsyncRelationalDatabase(
        host=settings.MYSQL_HOST,
        port=settings.MYSQL_PORT,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        database=settings.MYSQL_DATABASE
    )
    await relational_db.connect()
    structured_data = await relational_db.get_all_structured_data()
    print(f"MySQL - Structured records: {len(structured_data)}")
    await relational_db.close()

    return len(entities) > 0 and len(embeddings) > 0 and len(structured_data) > 0

async def test_document_processing():
    """Test document processing workflow with sample PDF."""
    # Download test paper if not exists
    test_paper_path = Path("/home/ubuntu/repos/ananke2/test_paper.pdf")
    if not test_paper_path.exists():
        print("Downloading test paper...")
        os.system(f"curl -o {test_paper_path} https://arxiv.org/pdf/2404.16130")

    print("Starting document processing test...")
    result = process_document_workflow.delay(str(test_paper_path))
    print(f"Task ID: {result.id}")

    print("Waiting for result...")
    try:
        result_data = result.get(timeout=300)
        print("Processing completed successfully!")
        print("Result:", result_data)

        # Verify database writes
        success = await verify_database_writes()
        if success:
            print("\nTest passed! Document processed and data written to all databases.")
            return True
        else:
            print("\nTest failed! Data not written to all databases.")
            return False

    except Exception as e:
        print(f"Error processing document: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_document_processing())
