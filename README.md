# Ananke2 Knowledge Framework

## Overview
Ananke2 is a comprehensive multi-modal knowledge extraction framework designed for processing and analyzing academic papers and technical documents. The framework supports:

- Multi-modal information processing (text, math expressions, logic expressions, code snippets)
- Knowledge graph extraction with semantic triple tracking
- Vector embeddings for semantic search
- Multi-language support (English, Chinese, French, German, Japanese)
- Document chunking with structured sentence parsing
- Asynchronous task processing with Redis queue

## Architecture

### Database Layer
- **Graph Database (Neo4j)**
  - Knowledge graph storage
  - Entity and relationship management
  - Semantic triple storage with hit count tracking
  - Graph-based querying capabilities

- **Vector Database (ChromaDB)**
  - Embedding storage for semantic search
  - Multi-modal content vectorization
  - Similarity search capabilities
  - Document chunk embeddings

- **Structured Database (MySQL)**
  - Traditional data storage
  - Metadata management
  - Document structure information
  - Processing task status tracking

### Task Processing
- **Redis Task Queue**
  - Asynchronous task management
  - Document processing workflow
  - Retry mechanism for failed tasks
  - Task status monitoring

### Document Processing
- **Multi-modal Content Extraction**
  - PDF document parsing
  - LaTeX content processing
  - Mathematical expression extraction
  - Code snippet identification
  - Logic expression parsing

- **Document Chunking**
  - Structured sentence parsing
  - Context-aware chunking
  - Cross-reference preservation
  - Multi-language chunk handling

- **Knowledge Graph Extraction**
  - Entity identification
  - Relationship extraction
  - Semantic triple generation
  - Hit count tracking
  - Multi-language entity linking

## Setup

### Prerequisites
- Python 3.12+
- Docker and Docker Compose
- Python venv module

### Installation
1. Clone the repository:
```bash
git clone https://github.com/ceresman/ananke2.git
cd ananke2
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Configure pip to use BFSU mirror:
```bash
pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
```

4. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

5. Configure environment:
```bash
cp .env.example .env
# Edit .env with your credentials and API keys
```

6. Start services:
```bash
docker-compose up -d
```

### Troubleshooting
- If you encounter SSL errors with the BFSU mirror, try:
  ```bash
  pip install --trusted-host mirrors.bfsu.edu.cn -r requirements.txt
  ```
- For permission errors on Unix/Linux:
  ```bash
  python -m venv .venv --system-site-packages
  ```

### Configuration
Key environment variables in `.env`:
```
# Database Configuration
NEO4J_URI=bolt://0.0.0.0:7687
CHROMA_PERSIST_DIRECTORY=/path/to/chroma
MYSQL_HOST=0.0.0.0

# Redis Configuration
REDIS_HOST=0.0.0.0
REDIS_PORT=6379

# API Keys
QWEN_API_KEY=your-api-key

# Network Configuration
HOST=0.0.0.0
PORT=8000
```

## Usage

### Document Processing
```python
from ananke2.tasks.document import process_document

# Process an arXiv paper
document_id = await process_document(arxiv_id="2301.00001")

# Process a local PDF
document_id = await process_document(file_path="/path/to/paper.pdf")
```

### Knowledge Graph Queries
```python
from ananke2.database.graph import Neo4jInterface

# Query entities by type
entities = await graph_db.search({
    "type": "TECHNOLOGY",
    "limit": 10
})

# Get related entities
related = await graph_db.get_relations(entity_id)
```

### Semantic Search
```python
from ananke2.database.vector import ChromaInterface

# Search similar documents
results = await vector_db.search({
    "query": "quantum computing applications",
    "limit": 5
})
```

### Multi-language Support
```python
from ananke2.tasks.document import process_document

# Process documents in different languages
doc_zh = await process_document(file_path="paper_zh.pdf", language="zh")
doc_fr = await process_document(file_path="paper_fr.pdf", language="fr")
```

## API Documentation

### REST Endpoints
- `POST /api/v1/documents`: Upload and process new documents
- `GET /api/v1/documents/{id}`: Retrieve document information
- `GET /api/v1/entities`: Query knowledge graph entities
- `GET /api/v1/search`: Semantic search across documents

### WebSocket Events
- `document_status`: Real-time document processing status
- `extraction_progress`: Knowledge extraction progress updates

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
```bash
black .
isort .
flake8
```

## License
Apache License 2.0

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.
