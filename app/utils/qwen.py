"""Qwen API client for knowledge graph extraction."""

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
        self.client = OpenAI(
            api_key=settings.qwen_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.max_retries = 3
        self.retry_delay = 1  # Initial delay in seconds

    def _handle_rate_limit(self, attempt: int) -> None:
        """Handle rate limiting with exponential backoff.

        Args:
            attempt: Current retry attempt number
        """
        if attempt < self.max_retries:
            delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
            time.sleep(delay)
        else:
            raise Exception("Max retries exceeded for rate limiting")

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities and relationships from text using Qwen API.

        Args:
            text: Input text to extract entities and relationships from

        Returns:
            List of dictionaries containing entities and relationships

        Raises:
            Exception: If API call fails after max retries
        """
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
                        {"role": "user", "content": text}
                    ]
                )
                # Parse response into list of dictionaries
                result = eval(completion.choices[0].message.content)
                return result

            except Exception as e:
                if "429" in str(e):
                    self._handle_rate_limit(attempt)
                    continue
                raise

        raise Exception("Failed to extract entities after max retries")

    def extract_entities_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract entities and relationships from multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of lists containing entities and relationships for each text
        """
        results = []
        for text in texts:
            try:
                result = self.extract_entities(text)
                results.append(result)
            except Exception as e:
                results.append([])  # Empty list for failed extractions
        return results
