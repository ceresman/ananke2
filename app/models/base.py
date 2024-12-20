from uuid import UUID
from pydantic import BaseModel
from typing import Dict, Any

class BaseObject(BaseModel):
    """Base class for all objects in the knowledge graph.

    Provides common functionality for data serialization and database-specific
    format conversion. Inherits from Pydantic BaseModel for validation and
    serialization capabilities.

    The class handles:
    - UUID field serialization to strings
    - Conversion to database-specific formats (Neo4j, MySQL, ChromaDB)
    - Common data validation through Pydantic

    Example:
        ```python
        class Entity(BaseObject):
            name: str
            type: str
            description: str

        entity = Entity(name="Example", type="CONCEPT", description="An example entity")
        neo4j_data = entity.to_neo4j()
        mysql_data = entity.to_mysql()
        chroma_data = entity.to_chroma()
        ```
    """

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert model to dictionary with UUID string conversion.

        Extends Pydantic's model_dump to ensure UUID fields are properly
        serialized as strings for database compatibility.

        Args:
            **kwargs: Additional arguments passed to Pydantic's model_dump

        Returns:
            Dict[str, Any]: Dictionary representation of the model with
                UUID fields converted to strings

        Example:
            ```python
            data = obj.model_dump()
            assert isinstance(data["id"], str)  # UUID converted to string
            ```
        """
        data = super().model_dump(**kwargs)
        for key, value in data.items():
            if isinstance(value, UUID):
                data[key] = str(value)
        return data

    def to_neo4j(self) -> Dict[str, Any]:
        """Convert the object to a Neo4j compatible format.

        Serializes the object for Neo4j graph database storage.
        Currently uses the base model_dump implementation, but can be
        overridden by subclasses for custom Neo4j formatting.

        Returns:
            Dict[str, Any]: Neo4j-compatible dictionary representation

        Raises:
            ValueError: If object contains data types incompatible with Neo4j
        """
        return self.model_dump()

    def to_mysql(self) -> Dict[str, Any]:
        """Convert the object to a MySQL compatible format.

        Serializes the object for MySQL relational database storage.
        Currently uses the base model_dump implementation, but can be
        overridden by subclasses for custom MySQL formatting.

        Returns:
            Dict[str, Any]: MySQL-compatible dictionary representation

        Raises:
            ValueError: If object contains data types incompatible with MySQL
        """
        return self.model_dump()

    def to_chroma(self) -> Dict[str, Any]:
        """Convert the object to a ChromaDB compatible format.

        Serializes the object for ChromaDB vector database storage.
        Currently uses the base model_dump implementation, but can be
        overridden by subclasses for custom ChromaDB formatting.

        Returns:
            Dict[str, Any]: ChromaDB-compatible dictionary representation

        Raises:
            ValueError: If object contains data types incompatible with ChromaDB
        """
        return self.model_dump()
