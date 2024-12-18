"""Script to verify database installations and connections."""
import sys
import asyncio
import chromadb
from neo4j import GraphDatabase
import mysql.connector
from redis import Redis

def verify_redis():
    """Verify Redis connection."""
    try:
        redis_client = Redis(host='localhost', port=6379, db=0)
        if redis_client.ping():
            print("Redis connection successful")
            return True
    except Exception as e:
        print(f"Redis verification failed: {e}")
        return False

def verify_chromadb():
    """Verify ChromaDB installation."""
    try:
        print(f"ChromaDB version: {chromadb.__version__}")
        client = chromadb.Client()
        client.heartbeat()
        print("ChromaDB connection successful")
        return True
    except Exception as e:
        print(f"ChromaDB verification failed: {e}")
        return False

def verify_neo4j():
    """Verify Neo4j connection."""
    try:
        driver = GraphDatabase.driver("bolt://localhost:7687",
                                    auth=("neo4j", "ananke2024secure"))
        with driver.session() as session:
            result = session.run("RETURN 1")
            result.single()[0]
        print("Neo4j connection successful")
        driver.close()
        return True
    except Exception as e:
        print(f"Neo4j verification failed: {e}")
        return False

def verify_mysql():
    """Verify MySQL connection."""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="ananke",
            database="ananke"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        print(f"MySQL version: {version[0]}")
        cursor.close()
        conn.close()
        print("MySQL connection successful")
        return True
    except Exception as e:
        print(f"MySQL verification failed: {e}")
        return False

def main():
    """Run all verifications."""
    success = True
    print("Verifying database installations...\n")

    if not verify_redis():
        success = False
    print()

    if not verify_chromadb():
        success = False
    print()

    if not verify_neo4j():
        success = False
    print()

    if not verify_mysql():
        success = False

    if not success:
        print("\nSome verifications failed!")
        sys.exit(1)
    else:
        print("\nAll database connections verified successfully!")

if __name__ == "__main__":
    main()
