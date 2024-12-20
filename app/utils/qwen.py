"""Qwen API client for knowledge graph extraction and embeddings.

This module provides a client for interacting with the Qwen API to:
- Extract entities and relationships from text
- Generate text embeddings using text-embedding-v3 model
- Handle API rate limiting and retries
- Validate and process API responses

The client supports both synchronous and asynchronous operations with:
- Automatic retry mechanisms for API failures
- Rate limit handling with exponential backoff
- Response validation and error handling
- Batch processing capabilities
"""

from typing import Dict, List, Any, Optional
import json
import asyncio
import dashscope
from http import HTTPStatus
from ..config import settings

class QwenClient:
    """Client for interacting with Qwen API for knowledge extraction and embeddings.

    This client provides methods for extracting entities and relationships from text,
    as well as generating embeddings using the text-embedding-v3 model. It handles:
    - API authentication and configuration
    - Automatic retries with exponential backoff
    - Response validation and error handling
    - Batch processing of multiple texts

    Attributes:
        api_key (str): Qwen API authentication key
        base_url (str): Base URL for API requests
        max_retries (int): Maximum number of retry attempts (default: 3)
        retry_delay (int): Base delay between retries in seconds (default: 1)

    Example:
        ```python
        # Initialize client
        client = QwenClient(api_key="your-api-key")

        # Extract entities from text
        text = "Apple Inc. CEO Tim Cook announced new products."
        entities = await client.extract_entities(text)

        # Generate embeddings
        embeddings = await client.generate_embeddings(text)
        ```
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Qwen client with API key and configuration.

        Args:
            api_key (Optional[str]): Qwen API key. If not provided, reads from settings.

        Raises:
            ValueError: If no API key is available in settings
        """
        self.api_key = api_key
        if not self.api_key:
            self.api_key = settings.QWEN_API_KEY
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.max_retries = 3
        self.retry_delay = 1
        dashscope.api_key = self.api_key

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using Qwen API.

        Identifies and extracts entities of various types (PERSON, ORGANIZATION,
        GEO, EVENT, CONCEPT) from the input text.

        Args:
            text (str): Input text to extract entities from

        Returns:
            List[Dict[str, Any]]: List of extracted entities, each containing:
                - name (str): Entity name (capitalized)
                - type (str): Entity type
                - description (str): Entity description

        Raises:
            ValueError: If input text is empty
            Exception: If API request fails after max retries

        Example:
            ```python
            text = "Microsoft CEO Satya Nadella spoke at the conference."
            entities = await client.extract_entities(text)
            # Returns: [
            #     {"name": "MICROSOFT", "type": "ORGANIZATION",
            #      "description": "Technology company..."},
            #     {"name": "SATYA NADELLA", "type": "PERSON",
            #      "description": "CEO of Microsoft..."}
            # ]
            ```
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        result = await self._make_request(text, request_type="entities")
        if not result:
            return []
        return result

    async def _make_request(self, text: str, request_type: str = "entities") -> List[Dict[str, Any]]:
        """Make request to Qwen API with retry mechanism.

        Internal method that handles API communication with retry logic and
        error handling.

        Args:
            text (str): Input text for API request
            request_type (str): Type of request ("entities" or "relationships")

        Returns:
            List[Dict[str, Any]]: Parsed API response

        Raises:
            Exception: If API request fails after max retries or returns invalid response
        """
        prompt = self._get_prompt(text, request_type)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = dashscope.Generation.call(
                    model="qwen-max",
                    prompt=prompt,
                    result_format='message'
                )

                if resp.status_code == HTTPStatus.OK:
                    try:
                        content = resp.output.choices[0].message.content
                        result = json.loads(content)
                        if isinstance(result, list):
                            return result
                        return []
                    except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
                        return []

                if resp.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                    last_error = Exception("API error: Rate limit exceeded")
                else:
                    last_error = Exception(f"API error: {resp.message}")

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                continue

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                continue

        raise last_error or Exception("Failed to extract data after max retries")

    def _get_prompt(self, text: str, request_type: str) -> str:
        """Get appropriate prompt template based on request type.

        Internal method that returns the prompt template for entity or
        relationship extraction.

        Args:
            text (str): Input text to process
            request_type (str): Type of request ("entities" or "relationships")

        Returns:
            str: Formatted prompt for API request
        """
        if request_type == "entities":
            return f"""Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text.

For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [PERSON, ORGANIZATION, GEO, EVENT, CONCEPT]
- entity_description: Comprehensive description of the entity's attributes and activities

Format each entity output as a JSON entry with the following format:
{{"name": "<entity name>", "type": "<type>", "description": "<entity description>"}}

Text:
{text}

Just return output as a list of JSON entities, nothing else."""
        else:
            return f"""Given a text document, identify all relationships between entities in the text.

For each relationship, extract:
- source_entity: Name of the source entity (capitalized)
- target_entity: Name of the target entity (capitalized)
- relationship_description: Explanation of how they are related
- relationship_strength: Integer score 1-10 indicating strength

Format as JSON:
{{"source": "<source>", "target": "<target>", "relationship": "<description>", "relationship_strength": <strength>}}

Text:
{text}

Just return output as a list of JSON relationships, nothing else."""

    async def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities in text using Qwen API.

        Identifies and extracts relationships between entities, including
        relationship descriptions and strength scores.

        Args:
            text (str): Input text to extract relationships from

        Returns:
            List[Dict[str, Any]]: List of relationships, each containing:
                - source (str): Source entity name
                - target (str): Target entity name
                - relationship (str): Description of relationship
                - relationship_strength (int): Strength score (1-10)

        Raises:
            ValueError: If input text is empty or relationship strength invalid
            Exception: If API request fails after max retries

        Example:
            ```python
            text = "Tim Cook leads Apple Inc. as CEO since 2011."
            relationships = await client.extract_relationships(text)
            # Returns: [
            #     {"source": "TIM COOK", "target": "APPLE INC.",
            #      "relationship": "CEO of company",
            #      "relationship_strength": 9}
            # ]
            ```
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        result = await self._make_request(text, request_type="relationships")
        if not result:
            return []
        self._validate_relationships(result)
        return result

    def _validate_relationships(self, relationships: List[Dict[str, Any]]) -> None:
        """Validate relationship data structure and strength values.

        Args:
            relationships (List[Dict[str, Any]]): List of extracted relationships

        Raises:
            ValueError: If relationship strength is not an integer between 1 and 10
        """
        for rel in relationships:
            strength = rel.get("relationship_strength")
            if not isinstance(strength, int) or strength < 1 or strength > 10:
                raise ValueError("Relationship strength must be between 1 and 10")

    async def generate_embeddings(self, text: str, modality: str = "text") -> List[float]:
        """Generate embeddings using Qwen text-embedding-v3 model.

        Generates dense vector embeddings for input text using the
        text-embedding-v3 model with 1024 dimensions.

        Args:
            text (str): Input text to generate embeddings for
            modality (str): Input modality (currently only "text" supported)

        Returns:
            List[float]: 1024-dimensional dense embedding vector

        Raises:
            ValueError: If input text is empty
            Exception: If API request fails or returns invalid response

        Example:
            ```python
            text = "Example text for embedding generation"
            embedding = await client.generate_embeddings(text)
            print(f"Generated {len(embedding)}-dimensional embedding")
            ```
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = dashscope.TextEmbedding.call(
                    model=dashscope.TextEmbedding.Models.text_embedding_v3,
                    input=text,
                    dimension=1024,
                    output_type="dense&sparse"
                )

                if response.status_code == HTTPStatus.OK:
                    try:
                        return response.output["embeddings"][0]["dense"]
                    except (KeyError, IndexError):
                        raise Exception("Invalid embedding response format")

                if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                    last_error = Exception("API error: Rate limit exceeded")
                else:
                    last_error = Exception(f"API error: {response.message}")

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                continue

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                continue

        raise last_error or Exception("Failed to generate embeddings after max retries")

    async def generate_embeddings_batch(self, texts: List[str], modality: str = "text") -> List[List[float]]:
        """Generate embeddings for multiple texts in batch.

        Processes multiple texts in sequence, returning embeddings for each.
        Failed generations return empty lists without stopping the batch.

        Args:
            texts (List[str]): List of input texts
            modality (str): Input modality (currently only "text" supported)

        Returns:
            List[List[float]]: List of embedding vectors, empty lists for failed generations

        Example:
            ```python
            texts = ["First example", "Second example", "Third example"]
            embeddings = await client.generate_embeddings_batch(texts)
            print(f"Generated embeddings for {len(embeddings)} texts")
            ```
        """
        results = []
        for text in texts:
            try:
                embedding = await self.generate_embeddings(text, modality)
                results.append(embedding)
            except Exception:
                results.append([])  # Empty list for failed generations
        return results
