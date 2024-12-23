"""Test model functionality and serialization."""
import pytest
from uuid import UUID
from app.models.types import StructuredDataBase
from app.models.entities import EntitySymbol, EntitySemantic
from app.models.structured import StructuredData

@pytest.mark.asyncio
async def test_entity_symbol():
    """Test EntitySymbol creation and serialization."""
    entity = EntitySymbol(
        symbol_id=UUID('12345678-1234-5678-1234-567812345678'),
        name="Test Entity",
        entity_type="TEST",
        descriptions=["Test Description"]
    )

    # Test basic attributes
    assert entity.name == "Test Entity"
    assert entity.entity_type == "TEST"
    assert len(entity.descriptions) == 1

    # Test serialization
    serialized = entity.model_dump()
    assert isinstance(serialized['symbol_id'], str)
    assert serialized['name'] == "Test Entity"

    # Test Neo4j format
    neo4j_data = entity.to_neo4j()
    assert isinstance(neo4j_data['symbol_id'], str)
    assert neo4j_data['name'] == "Test Entity"

@pytest.mark.asyncio
async def test_structured_data():
    """Test StructuredData creation and serialization."""
    data = StructuredData(
        data_id=UUID('12345678-1234-5678-1234-567812345678'),
        data_type="test",
        data_value={"key": "value"}
    )

    # Test basic attributes
    assert isinstance(data.data_id, UUID)
    assert data.data_type == "test"
    assert data.data_value["key"] == "value"

    # Test serialization
    serialized = data.model_dump()
    assert isinstance(serialized['data_id'], str)
    assert serialized['data_type'] == "test"
    assert isinstance(serialized['data_value'], dict)
