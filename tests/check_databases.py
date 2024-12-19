"""Script to check database contents after document processing."""
import asyncio
import json
from neo4j import AsyncGraphDatabase
from app.database.graph import Neo4jInterface
from app.database.vector import ChromaInterface
from app.database.relational import MySQLInterface
from app.config import settings

async def get_relationships(graph_db: Neo4jInterface, limit: int = 5):
    """Get relationships from Neo4j database."""
    session = AsyncGraphDatabase.driver(
        graph_db.uri,
        auth=(graph_db.username, graph_db.password)
    ).session()

    try:
        result = await session.run("""
            MATCH (source:Entity)-[r]->(target:Entity)
            RETURN {
                source: source.name,
                target: target.name,
                relationship: r.description,
                relationship_strength: r.strength
            } as rel
            LIMIT $limit
        """, limit=limit)
        records = await result.all()
        return [record["rel"] for record in records]
    finally:
        await session.close()

async def check_databases():
    """Check contents of all databases."""
    print("Checking Neo4j database...")
    graph_db = Neo4jInterface(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )
    await graph_db.connect()

    # Get entities
    entities = await graph_db.list(limit=5)
    print("\nNeo4j Entities:")
    print(json.dumps([{
        "name": e.name,
        "type": e.entity_type,
        "description": e.descriptions[0] if e.descriptions else ""
    } for e in entities], indent=2))

    # Get relationships
    relationships = await get_relationships(graph_db)
    print("\nExtracted Relationships:")
    print(json.dumps(relationships, indent=2))

    print("\nChecking ChromaDB vectors...")
    vector_db = ChromaInterface(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT
    )
    await vector_db.connect()
    vectors = await vector_db.list(limit=5)
    print("\nChromaDB Vectors:")
    print(json.dumps([{
        "id": str(v.semantic_id),
        "name": v.name,
        "vector_size": len(v.vector_representation)
    } for v in vectors], indent=2))

    print("\nChecking MySQL metadata...")
    relational_db = MySQLInterface(
        host=settings.MYSQL_HOST,
        port=settings.MYSQL_PORT,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        database=settings.MYSQL_DATABASE
    )
    await relational_db.connect()
    metadata = await relational_db.list(limit=5)
    print("\nMySQL Metadata:")
    print(json.dumps([{
        "id": str(m.data_id),
        "type": m.data_type,
        "value": m.data_value
    } for m in metadata], indent=2))

if __name__ == "__main__":
    asyncio.run(check_databases())
