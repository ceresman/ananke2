"""Qwen API client for knowledge graph extraction and embeddings."""

from typing import Dict, List, Any, Optional
import time
from openai import OpenAI
from ..config import settings

ENTITY_EXTRACTION_PROMPT = """Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [PERSON, ORGANIZATION, GEO, EVENT, CONCEPT]
- entity_description: Comprehensive description of the entity's attributes and activities

For each pair of related entities, extract the following information:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: explanation of how the entities are related
- relationship_strength: integer score between 1 to 10 indicating relationship strength

Format output as a list of JSON objects with two types:
1. Entity objects: {"name": <entity name>, "type": <type>, "description": <entity description>}
2. Relationship objects: {"source": <source>, "target": <target>, "relationship": <description>, "relationship_strength": <strength>}

Return only the JSON list, no other text."""

class QwenClient:
    """Client for interacting with Qwen API."""

    def __init__(self):
        """Initialize Qwen client with API configuration."""
        self.api_key = settings.qwen_api_key
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.max_retries = 3
        self.retry_delay = 1  # Initial delay in seconds

    def _handle_rate_limit(self, attempt: int) -> bool:
        """Handle rate limiting with exponential backoff. Returns True if should retry."""
        if attempt >= self.max_retries - 1:
            return False
        delay = self.retry_delay * (2 ** attempt)
        time.sleep(delay)
        return True

    def _validate_relationships(self, data: List[Dict[str, Any]]) -> None:
        """Validate relationship strength is within bounds."""
        if not isinstance(data, list):
            raise ValueError("Expected list of relationships")

        for item in data:
            if "source" in item and "target" in item:  # Only validate relationship items
                if "relationship_strength" not in item:
                    raise ValueError("Relationship strength must be between 1 and 10")
                try:
                    strength = float(item["relationship_strength"])
                    if not (1 <= strength <= 10):
                        raise ValueError("Relationship strength must be between 1 and 10")
                except (ValueError, TypeError):
                    raise ValueError("Relationship strength must be between 1 and 10")

    async def _make_request(self, text: str) -> List[Dict[str, Any]]:
        """Make request to Qwen API with retry logic."""
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
                        {"role": "user", "content": text}
                    ]
                )
                content = completion.choices[0].message.content
                try:
                    result = eval(content)
                    return result
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Invalid response format: {str(e)}")

            except Exception as e:
                last_error = e
                if "429" in str(e):
                    if attempt < self.max_retries - 1:  # Only retry if not last attempt
                        self._handle_rate_limit(attempt)
                        continue
                    raise Exception("Rate limit exceeded")
                raise e

        raise last_error or Exception("Failed to extract entities after max retries")

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using Qwen API."""
        result = await self._make_request(text)
        return [e for e in result if "type" in e]

    async def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships from text using Qwen API."""
        result = await self._make_request(text)
        relationships = [e for e in result if "source" in e and "target" in e]
        self._validate_relationships(relationships)  # Validate after filtering
        return relationships

    async def extract_entities_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract entities and relationships from multiple texts."""
        results = []
        for text in texts:
            try:
                result = await self.extract_entities(text)
                results.append(result)
            except Exception as e:
                results.append([])  # Empty list for failed extractions
        return results

    async def generate_embeddings(self, text: str, modality: str = "text") -> List[float]:
        """Generate embeddings using Qwen model."""
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model="qwen"
                )
                return response.data[0].embedding

            except Exception as e:
                last_error = e
                if "429" in str(e):
                    if attempt < self.max_retries - 1:
                        self._handle_rate_limit(attempt)
                        continue
                    raise Exception("Rate limit exceeded")
                raise e

        raise last_error or Exception("Failed to generate embeddings after max retries")

    async def generate_embeddings_batch(self, texts: List[str], modality: str = "text") -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        results = []
        for text in texts:
            try:
                embedding = await self.generate_embeddings(text, modality)
                results.append(embedding)
            except Exception as e:
                results.append([])  # Empty list for failed generations
        return results
