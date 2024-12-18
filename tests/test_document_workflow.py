"""Test document processing workflow."""

import asyncio
import sys
import os
import time
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tasks.workflow import process_document_workflow
from app.database.graph import AsyncGraphDatabase
from app.database.vector import AsyncVectorDatabase
from app.database.relational import AsyncRelationalDatabase
from app.config import Settings

settings = Settings()

def wait_for_neo4j():
    """Wait for Neo4j to be ready."""
    print("Waiting for Neo4j...")
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://neo4j:7474")
            if response.status_code == 200:
                print("Neo4j is ready!")
                return True
        except Exception as e:
            print(f"Neo4j not ready yet: {e}")
        time.sleep(2)
        print(f"Waiting for Neo4j... attempt {attempt + 1}/{max_attempts}")
    return False

def wait_for_mysql():
    """Wait for MySQL to be ready."""
    print("Waiting for MySQL...")
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            import pymysql
            conn = pymysql.connect(
                host="mysql",
                port=settings.MYSQL_PORT,
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD,
                database=settings.MYSQL_DATABASE
            )
            conn.close()
            print("MySQL is ready!")
            return True
        except Exception as e:
            print(f"MySQL not ready yet: {e}")
        time.sleep(2)
        print(f"Waiting for MySQL... attempt {attempt + 1}/{max_attempts}")
    return False

async def verify_database_writes():
    """Verify that data was written to all databases."""
    print("\nVerifying database writes...")

    # Check Neo4j
    graph_db = AsyncGraphDatabase(
        uri="bolt://neo4j:7687",
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )
    await graph_db.connect()
    entities = await graph_db.get_all_entities()
    relationships = await graph_db.get_all_relationships()
    print(f"Neo4j - Entities: {len(entities)}, Relationships: {len(relationships)}")
    await graph_db.disconnect()

    # Check ChromaDB
    vector_db = AsyncVectorDatabase(
        host="chromadb",
        port=settings.CHROMA_PORT,
        collection_name=settings.CHROMA_COLLECTION
    )
    await vector_db.connect()
    embeddings = await vector_db.get_all_embeddings()
    print(f"ChromaDB - Embeddings: {len(embeddings)}")
    await vector_db.disconnect()

    # Check MySQL
    relational_db = AsyncRelationalDatabase(
        host="mysql",
        port=settings.MYSQL_PORT,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        database=settings.MYSQL_DATABASE
    )
    await relational_db.connect()
    structured_data = await relational_db.get_all_structured_data()
    print(f"MySQL - Structured records: {len(structured_data)}")
    await relational_db.disconnect()

    return len(entities) > 0 and len(embeddings) > 0 and len(structured_data) > 0

async def test_document_processing():
    """Test document processing workflow with sample PDF."""
    # Wait for databases to be ready
    if not wait_for_neo4j() or not wait_for_mysql():
        print("Failed to connect to databases")
        return False

    # Check for test paper
    test_paper_path = Path("tests/data/sample.pdf")
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
