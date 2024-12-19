# Project Test Log

## Knowledge Graph Extraction Tests

### Test Cases
1. Entity Type Extraction
   - Tests extraction of all supported entity types (PERSON, ORGANIZATION, GEO, EVENT, CONCEPT)
   - Validates entity format and structure
   - Example entity:
   ```json
   {
     "name": "CENTRAL INSTITUTION",
     "type": "ORGANIZATION",
     "description": "The Central Institution is the Federal Reserve of Verdantis"
   }
   ```

2. Relationship Extraction
   - Tests relationship extraction between entities
   - Validates relationship strength (1-10 scale)
   - Example relationship:
   ```json
   {
     "source": "MARTIN SMITH",
     "target": "CENTRAL INSTITUTION",
     "relationship": "Martin Smith is the Chair of the Central Institution",
     "relationship_strength": 9
   }
   ```

3. Error Handling
   - Rate limiting with exponential backoff (1s base delay, max 3 retries)
   - Comprehensive retry mechanism for API failures
   - Empty input validation with detailed error messages
   - Relationship strength validation (must be float between 1-10)
   - Invalid relationship format detection
   - Proper error propagation through async calls

4. Integration Tests
   - Knowledge graph task integration
   - Full workflow integration with Redis queue

### Test Configuration
- Qwen API Key: sk-46e78b90eb8e4d6ebef79f265891f238
- API Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
- Test Data: Sample text about Central Institution and Market Strategy Committee
- Embedding Model: text-embedding-v3
  - Dimension: 1024
  - Output Type: dense&sparse
  - Retry Configuration: 3 attempts with exponential backoff

### Test Results (Updated 2024-03-18)
# Test Results
- Knowledge Graph Tests: ✓ All 4 tests passed
  - Entity extraction
  - Relationship extraction
  - Error handling
  - Integration tests
- Database Query Tests: ✓ All 6 tests passed
  - search_by_embedding
  - search_by_graph
  - search_structured
  - combined_search
  - search_with_modality
  - error_handling
- Document Processing Tests: ✓ All 4 tests passed
  - PDF extraction
  - Text chunking
  - Metadata parsing
  - Multi-modal support
- Embedding Generation Tests: ✓ All 3 tests passed
  - Dense vector generation (1024d)
  - Error handling and retries
  - Empty input validation

### Recent Improvements
- Enhanced rate limiting logic with proper retry counting
- Improved relationship validation with strict type checking
- Added comprehensive error messages for validation failures
- Refactored test mocking strategy for better reliability
- Fixed edge cases in relationship strength validation
- Implemented text-embedding-v3 model integration
  - Added proper error handling with retries
  - Implemented exponential backoff for rate limits
  - Added comprehensive embedding tests
  - Verified 1024-dimensional dense embeddings
