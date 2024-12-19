"""Script to verify document processing output format."""
from app.processors.document import DocumentProcessor
import json
from pathlib import Path

def main():
    # Create test document
    test_dir = Path("test_docs")
    test_dir.mkdir(exist_ok=True)

    test_file = test_dir / "sample.txt"
    test_content = """LayoutParser: A Unified Toolkit for Deep Learning Based Document Image Analysis

Zejiang Shen, Ruochen Zhang, Melissa Dell, Benjamin Charles Germain Lee, Jacob Carlson, and Weining Li

Abstract: Recent advances in document image analysis (DIA) have been primarily driven by the application of neural networks. This paper introduces LayoutParser, an open-source library for streamlining the usage of DL in DIA research and applications.

The core LayoutParser library comes with interfaces for layout detection, character recognition, and document processing tasks. The library incorporates a community platform for sharing pre-trained models and document digitization pipelines."""

    test_file.write_text(test_content)

    # Process document
    processor = DocumentProcessor()
    print("\nProcessing document...")
    elements = processor.process_document(str(test_file))
    print("\nExtracted text elements:")
    print("\n".join(elements))

    print("\nExtracting knowledge graph...")
    graph = processor.extract_knowledge_graph(elements)
    print("\nExtracted Knowledge Graph:")
    print(json.dumps(graph, indent=2))

    # Save graph
    output_file = test_dir / "output_graph.json"
    processor.save_knowledge_graph(graph, str(output_file))
    print(f"\nSaved knowledge graph to {output_file}")

if __name__ == "__main__":
    main()
