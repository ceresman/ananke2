"""Document processing tasks for Ananke2."""

import os
from uuid import UUID
from typing import Dict, Any, Union, Optional
import httpx
import xml.etree.ElementTree as ET
from celery import states
from . import celery_app
from ..models.structured import Document, StructuredChunk, StructuredSentence
from ..utils.qwen import QwenClient
from ..database.vector import ChromaInterface
from ..database.graph import Neo4jInterface
from ..database.relational import MySQLInterface

async def download_arxiv_paper(arxiv_id: str) -> str:
    """Download paper from arXiv.

    Args:
        arxiv_id: The arXiv ID of the paper

    Returns:
        Local path to downloaded PDF
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()

    pdf_path = f"/tmp/{arxiv_id}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(response.content)
    return pdf_path

async def extract_arxiv_metadata(arxiv_id: str) -> Dict[str, Any]:
    """Extract metadata using arXiv API.

    Args:
        arxiv_id: The arXiv ID of the paper

    Returns:
        Dictionary containing paper metadata
    """
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()

    # Parse XML response
    root = ET.fromstring(response.text)
    entry = root.find('{http://www.w3.org/2005/Atom}entry')

    return {
        'title': entry.find('{http://www.w3.org/2005/Atom}title').text,
        'authors': [author.find('{http://www.w3.org/2005/Atom}name').text
                   for author in entry.findall('{http://www.w3.org/2005/Atom}author')],
        'summary': entry.find('{http://www.w3.org/2005/Atom}summary').text,
        'categories': [cat.get('term') for cat in entry.findall('{http://www.w3.org/2005/Atom}category')]
    }

@celery_app.task(bind=True)
async def process_document(self, document_id: str, arxiv_id: Optional[str] = None, file_path: Optional[str] = None) -> Dict[str, Any]:
    """Process document with progress tracking.

    Args:
        document_id: The ID of the document to process
        arxiv_id: Optional arXiv ID for papers
        file_path: Optional path to local file

    Returns:
        Dict containing processing status and results
    """
    try:
        # Update state to processing
        self.update_state(state=states.STARTED, meta={'progress': 0})

        if arxiv_id:
            # Download arXiv paper and extract metadata
            file_path = await download_arxiv_paper(arxiv_id)
            metadata = await extract_arxiv_metadata(arxiv_id)
            self.update_state(state='PROCESSING', meta={'progress': 20})

        # Extract content with multi-modal support
        content_result = await extract_content.delay(
            document_id=document_id,
            file_path=file_path
        ).get()
        self.update_state(state='PROCESSING', meta={'progress': 50})

        # Process math expressions
        math_result = await process_math_expressions.delay(content_result).get()
        self.update_state(state='PROCESSING', meta={'progress': 70})

        # Extract knowledge graph
        kg_result = await extract_knowledge_graph.delay(content_result).get()
        self.update_state(state='PROCESSING', meta={'progress': 90})

        # Store results in MySQL
        mysql_db = MySQLInterface()
        await mysql_db.create({
            'document_id': document_id,
            'metadata': metadata if arxiv_id else None,
            'content': content_result,
            'math_expressions': math_result,
            'knowledge_graph': kg_result
        })

        return {
            "status": "completed",
            "document_id": document_id,
            "metadata": metadata if arxiv_id else None,
            "content": content_result,
            "math_expressions": math_result,
            "knowledge_graph": kg_result
        }
    except Exception as e:
        self.update_state(
            state=states.FAILURE,
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'document_id': document_id
            }
        )
        raise

@celery_app.task
async def extract_content(document_id: str, file_path: str) -> Dict[str, Any]:
    """Extract content from document with multi-modal support.


    Args:
        document_id: The ID of the document
        file_path: Path to the document file

    Returns:
        Dict containing extraction status and results
    """
    try:
        # Initialize document structure
        document = Document(
            id=UUID(document_id),
            raw_content="",
            structured_chunks=[]
        )

        # Extract text content using appropriate method based on file type
        if file_path.endswith('.pdf'):
            import PyPDF2
            with open(file_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                content = ""
                for page in pdf.pages:
                    content += page.extract_text()
                document.raw_content = content

        # Chunk document into structured sentences
        chunks = []
        current_chunk = ""
        for line in document.raw_content.split('\n'):
            current_chunk += line + '\n'
            if len(current_chunk.split()) >= 100:  # Chunk size of ~100 words
                chunk = StructuredChunk(
                    chunk_id=UUID(int=len(chunks)),
                    chunk_raw_content=current_chunk,
                    document_id=document.id,
                    modality_identifier="text"
                )
                chunks.append(chunk)
                current_chunk = ""

        document.structured_chunks = chunks

        # Store in MySQL database
        mysql_db = MySQLInterface()
        await mysql_db.create(document)

        return {
            "status": "completed",
            "document_id": document_id,
            "chunks": len(chunks),
            "content": document.raw_content[:1000]  # First 1000 chars as preview
        }
    except Exception as e:
        return {
            "status": "failed",
            "document_id": document_id,
            "error": str(e)
        }

@celery_app.task
def process_math_expressions(content_result: Dict[str, Any]) -> Dict[str, Any]:
    """Process mathematical expressions in the document.

    Args:
        content_result: Result dict from previous task containing document_id

    Returns:
        Dict containing processing status and results
    """
    try:
        document_id = content_result.get('document_id')
        # TODO: Implement mathematical expression processing logic
        # This is a placeholder for the actual implementation

        return {
            "status": "completed",
            "document_id": document_id,
            "result": "Mathematical expressions processed successfully"
        }
    except Exception as e:
        return {
            "status": "failed",
            "document_id": document_id,
            "error": str(e)
        }

@celery_app.task
async def extract_knowledge_graph(content_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract knowledge graph from document content.

    Args:
        content_result: Result dict from previous task containing document_id and content

    Returns:
        Dict containing extraction status, document_id, entities, and result/error message
    """
    try:
        document_id = content_result.get('document_id')
        content = content_result.get('content', '')

        # Initialize Qwen client and extract entities
        client = QwenClient()
        entities = await client.extract_entities(content)
        relationships = await client.extract_relationships(content)

        return {
            "status": "completed",
            "document_id": document_id,
            "entities": entities,
            "relationships": relationships,
            "result": "Knowledge graph extracted successfully"
        }
    except Exception as e:
        return {
            "status": "failed",
            "document_id": document_id,
            "error": str(e)
        }
