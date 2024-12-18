"""Verify unstructured library and arXiv processing workflow."""
import os
import arxiv
from unstructured.partition.pdf import partition_pdf
from app.tasks.workflow import process_arxiv_workflow
from app.database.sync_wrappers import get_sync_relational_db, get_sync_vector_db, get_sync_graph_db

def verify_arxiv_download():
    """Test arXiv paper download."""
    arxiv_id = "2005.14165"  # GPT-3 paper
    print(f"\nTesting arXiv download for paper {arxiv_id}...")

    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        download_dir = "/tmp/arxiv"
        os.makedirs(download_dir, exist_ok=True)
        pdf_path = os.path.join(download_dir, f"{arxiv_id}.pdf")
        paper.download_pdf(filename=pdf_path)
        print(f"Successfully downloaded: {paper.title}")
        print(f"PDF saved to: {pdf_path}")
        assert os.path.exists(pdf_path), "PDF file not found"
        return pdf_path
    except Exception as e:
        print(f"Error downloading paper: {str(e)}")
        raise

def verify_pdf_processing(pdf_path):
    """Test PDF processing with unstructured."""
    print(f"\nTesting PDF processing for {pdf_path}...")

    try:
        elements = partition_pdf(filename=pdf_path)
        print(f"Successfully extracted {len(elements)} elements")
        print("\nSample elements:")
        for i, element in enumerate(elements[:3]):
            print(f"{i+1}. {str(element)[:100]}...")
        return elements
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise

def verify_workflow():
    """Test complete workflow including database writes."""
    print("\nTesting complete workflow...")
    arxiv_id = "2005.14165"

    try:
        # Run workflow
        result = process_arxiv_workflow.delay(arxiv_id)
        print(f"Started workflow task: {result.id}")

        # Wait for completion
        task_result = result.get(timeout=300)
        print("Workflow completed successfully")
        print(f"Result: {task_result}")

        # Verify database entries
        rel_db = get_sync_relational_db()
        graph_db = get_sync_graph_db()
        vector_db = get_sync_vector_db()

        docs = rel_db.list_documents()
        doc_count = len([d for d in docs if d.get("data_type") == "arxiv_paper"])
        entities = graph_db.list_entities()
        embeddings = vector_db.list_embeddings()

        print("\nDatabase verification:")
        print(f"- Found {doc_count} arXiv papers")
        print(f"- Found {len(entities)} entities")
        print(f"- Found {len(embeddings)} embeddings")

        assert doc_count > 0, "No arXiv papers found"
        assert len(entities) > 0, "No entities found"
        assert len(embeddings) > 0, "No embeddings found"

        return True
    except Exception as e:
        print(f"Error in workflow verification: {str(e)}")
        raise

def main():
    """Run all verification tests."""
    try:
        print("Starting verification tests...")
        pdf_path = verify_arxiv_download()
        verify_pdf_processing(pdf_path)
        verify_workflow()
        print("\nAll verification tests passed successfully!")
    except Exception as e:
        print(f"\nVerification failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
