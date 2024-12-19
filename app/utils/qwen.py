"""Qwen API client for knowledge graph extraction and embeddings."""

from typing import Dict, List, Any
import json
import asyncio
import dashscope
from http import HTTPStatus
from ..config import settings

class QwenClient:
    """Client for interacting with Qwen API."""

    def __init__(self, api_key: str = None):
        """Initialize Qwen client."""
        self.api_key = api_key or settings.QWEN_API_KEY
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.max_retries = 3
        self.retry_delay = 1
        dashscope.api_key = self.api_key

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using Qwen API."""
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        result = await self._make_request(text)
        entities = [e for e in result if "name" in e and "type" in e]
        return entities

    async def _make_request(self, text: str) -> List[Dict[str, Any]]:
        """Make request to Qwen API with retries."""
        prompt = """Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text.

For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [PERSON, ORGANIZATION, GEO, EVENT, CONCEPT]
- entity_description: Comprehensive description of the entity's attributes and activities

Format each entity output as a JSON entry with the following format:
{"name": <entity name>, "type": <type>, "description": <entity description>}

Text:
{text}

Just return output as a list of JSON entities, nothing else."""

        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = dashscope.Generation.call(
                    model="qwen-max",
                    prompt=prompt.format(text=text),
                    result_format='message'
                )

                if resp.status_code == HTTPStatus.OK:
                    try:
                        return json.loads(resp.output.choices[0].message.content)
                    except json.JSONDecodeError:
                        return []

                last_error = Exception(f"API error: {resp.message}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                continue

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                continue

        raise last_error or Exception("Failed to extract entities after max retries")

    async def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships from text using Qwen API."""
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        result = await self._make_request(text)
        relationships = [e for e in result if "source" in e and "target" in e]
        self._validate_relationships(relationships)
        return relationships

    def _validate_relationships(self, relationships: List[Dict[str, Any]]) -> None:
        """Validate relationship strength is between 1 and 10."""
        for rel in relationships:
            strength = rel.get("relationship_strength")
            if not isinstance(strength, int) or strength < 1 or strength > 10:
                raise ValueError("Relationship strength must be between 1 and 10")

    async def generate_embeddings(self, text: str, modality: str = "text") -> List[float]:
        """Generate embeddings using Qwen text-embedding-v3 model."""
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        last_error = None
        for attempt in range(3):  # Max 3 retries
            try:
                response = dashscope.TextEmbedding.call(
                    model=dashscope.TextEmbedding.Models.text_embedding_v3,
                    input=text,
                    dimension=1024,
                    output_type="dense&sparse"
                )

                if response.status_code == HTTPStatus.OK:
                    return response.output["embeddings"][0]["dense"]

                last_error = Exception(f"API error: {response.message}")
                if attempt < 2:  # Only sleep if we're going to retry
                    await asyncio.sleep(1 * (attempt + 1))
                continue

            except Exception as e:
                last_error = e
                if attempt < 2:
                    await asyncio.sleep(1 * (attempt + 1))
                continue

        raise last_error or Exception("Failed to generate embeddings after max retries")

    async def generate_embeddings_batch(self, texts: List[str], modality: str = "text") -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        results = []
        for text in texts:
            try:
                embedding = await self.generate_embeddings(text, modality)
                results.append(embedding)
            except Exception:
                results.append([])  # Empty list for failed generations
        return results
