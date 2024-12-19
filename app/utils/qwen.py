"""Qwen API client for knowledge graph extraction and embeddings."""

from typing import Dict, List, Any
import json
import asyncio
import dashscope
from http import HTTPStatus
from ..config import settings

class QwenClient:
    """Client for interacting with Qwen API."""

    def __init__(self, api_key: str):
        """Initialize Qwen client."""
        self.api_key = api_key
        dashscope.api_key = api_key

    def extract_entities_sync(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using Qwen API."""
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

        resp = dashscope.Generation.call(
            model="qwen-max",
            prompt=prompt.format(text=text),
            result_format='message'
        )

        if resp.status_code != HTTPStatus.OK:
            raise Exception(f"Failed to extract entities: {resp.message}")

        try:
            return json.loads(resp.output.choices[0].message.content)
        except json.JSONDecodeError:
            return []

    def extract_relationships_sync(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships from text using Qwen API."""
        prompt = """From the entities identified in the text, identify all pairs of (source_entity, target_entity) that are clearly related to each other.

For each pair of related entities, extract the following information:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: an integer score between 1 to 10, indicating strength of the relationship

Format each relationship as a JSON entry with the following format:
{"source": <source_entity>, "target": <target_entity>, "relationship": <relationship_description>, "relationship_strength": <relationship_strength>}

Text:
{text}

Just return output as a list of JSON relationships, nothing else."""

        resp = dashscope.Generation.call(
            model="qwen-max",
            prompt=prompt.format(text=text),
            result_format='message'
        )

        if resp.status_code != HTTPStatus.OK:
            raise Exception(f"Failed to extract relationships: {resp.message}")

        try:
            return json.loads(resp.output.choices[0].message.content)
        except json.JSONDecodeError:
            return []

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
                    # Return dense embeddings
                    return response.output["embeddings"][0]["dense"]

                last_error = Exception(f"API error: {response.message}")
                if attempt < 2:  # Only sleep if we're going to retry
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                continue

            except Exception as e:
                last_error = e
                if attempt < 2:  # Only sleep if we're going to retry
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
            except Exception as e:
                results.append([])  # Empty list for failed generations
        return results
