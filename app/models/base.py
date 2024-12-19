from uuid import UUID
from pydantic import BaseModel
from typing import Dict, Any

class BaseObject(BaseModel):
    """Base class for all objects in the knowledge graph."""

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        data = super().model_dump(**kwargs)
        for key, value in data.items():
            if isinstance(value, UUID):
                data[key] = str(value)
        return data

    def to_neo4j(self) -> Dict[str, Any]:
        """Convert the object to a Neo4j compatible format."""
        return self.model_dump()

    def to_mysql(self) -> Dict[str, Any]:
        """Convert the object to a MySQL compatible format."""
        return self.model_dump()

    def to_chroma(self) -> Dict[str, Any]:
        """Convert the object to a Chroma compatible format."""
        return self.model_dump()
