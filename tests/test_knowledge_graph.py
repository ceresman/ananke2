"""Tests for knowledge graph extraction functionality."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.utils.qwen import QwenClient
from app.tasks import document
from app.tasks.workflow import process_document_workflow

# Test data matching example format exactly
EXAMPLE_TEXT = """The Verdantis's Central Institution is scheduled to meet on Monday and Thursday,
with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT,
followed by a press conference where Central Institution Chair Martin Smith will take questions.
Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%."""

EXPECTED_ENTITIES = [
    {
        "name": "CENTRAL INSTITUTION",
        "type": "ORGANIZATION",
        "description": "The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday"
    },
    {
        "name": "MARTIN SMITH",
        "type": "PERSON",
        "description": "Martin Smith is the chair of the Central Institution"
    },
    {
        "name": "MARKET STRATEGY COMMITTEE",
        "type": "ORGANIZATION",
        "description": "The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply"
    }
]

EXPECTED_RELATIONSHIPS = [
    {
        "source": "MARTIN SMITH",
        "target": "CENTRAL INSTITUTION",
        "relationship": "Martin Smith is the Chair of the Central Institution and will answer questions at a press conference",
        "relationship_strength": 9
    }
]

@pytest.fixture
def mock_qwen_response():
    """Mock response from Qwen API."""
    return EXPECTED_ENTITIES + EXPECTED_RELATIONSHIPS

def test_qwen_client_initialization():
    client = QwenClient()
    assert client.api_key == "sk-46e78b90eb8e4d6ebef79f265891f238"
    assert client.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert client.max_retries == 3
    assert client.retry_delay == 1

@pytest.mark.asyncio
async def test_entity_extraction(mock_qwen_response):
    with patch('app.utils.qwen.QwenClient._make_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_qwen_response
        client = QwenClient()
        entities = await client.extract_entities(EXAMPLE_TEXT)

        assert len(entities) == len(EXPECTED_ENTITIES)
        for actual, expected in zip(entities, EXPECTED_ENTITIES):
            assert actual == expected

@pytest.mark.asyncio
async def test_relationship_extraction(mock_qwen_response):
    with patch('app.utils.qwen.QwenClient._make_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_qwen_response
        client = QwenClient()
        relationships = await client.extract_relationships(EXAMPLE_TEXT)

        assert len(relationships) == len(EXPECTED_RELATIONSHIPS)
        for actual, expected in zip(relationships, EXPECTED_RELATIONSHIPS):
            assert actual == expected

@pytest.mark.asyncio
async def test_rate_limiting():
    """Test that rate limiting properly retries and eventually fails."""
    with patch('app.utils.qwen.OpenAI') as mock_openai:
        # Set up mock to raise rate limit error
        mock_client = mock_openai.return_value
        mock_chat = mock_client.chat.completions.create
        mock_chat.side_effect = [
            Exception("429 Rate limit exceeded"),
            Exception("429 Rate limit exceeded"),
            Exception("429 Rate limit exceeded")
        ]

        client = QwenClient()
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await client.extract_entities(EXAMPLE_TEXT)
        assert mock_chat.call_count == 3  # Should try exactly 3 times

@pytest.mark.asyncio
async def test_relationship_strength_validation():
    invalid_relationship = [{
        "source": "A",
        "target": "B",
        "relationship": "test",
        "relationship_strength": 11,
        "type": "RELATIONSHIP"  # Added type to match filtering
    }]
    with patch('app.utils.qwen.QwenClient._make_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = invalid_relationship
        client = QwenClient()
        with pytest.raises(ValueError, match="Relationship strength must be between 1 and 10"):
            await client.extract_relationships(EXAMPLE_TEXT)  # Changed to extract_relationships

@pytest.mark.asyncio
async def test_empty_input_handling():
    client = QwenClient()
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        await client.extract_entities("")
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        await client.extract_relationships("")

@pytest.mark.asyncio
async def test_knowledge_graph_task():
    with patch('app.tasks.document.QwenClient') as MockQwenClient:
        mock_client = AsyncMock()
        mock_client.extract_entities = AsyncMock(return_value=EXPECTED_ENTITIES)
        mock_client.extract_relationships = AsyncMock(return_value=EXPECTED_RELATIONSHIPS)
        MockQwenClient.return_value = mock_client

        result = await document.extract_knowledge_graph({
            "document_id": "test-doc",
            "content": EXAMPLE_TEXT,
            "status": "completed"
        })

        assert result["status"] == "completed"
        assert result["document_id"] == "test-doc"
        assert result["entities"] == EXPECTED_ENTITIES
        assert result["relationships"] == EXPECTED_RELATIONSHIPS
