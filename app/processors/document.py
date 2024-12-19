"""Document processor implementation using unstructured.io."""
from typing import List, Dict, Any
import json
from pathlib import Path
import os
from unstructured.partition.auto import partition
import dashscope
from dashscope import Generation

# Configure dashscope with API key
os.environ['DASHSCOPE_API_KEY'] = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

class DocumentProcessor:
    """Process documents using unstructured.io CPU-only implementation."""

    def __init__(self):
        """Initialize document processor."""
        self.supported_types = ['.pdf', '.txt', '.docx']
        self.qwen_api_key = os.getenv('QWEN_API_KEY', 'sk-46e78b90eb8e4d6ebef79f265891f238')
        dashscope.api_key = self.qwen_api_key

    def process_document(self, file_path: str) -> List[str]:
        """Process document and extract text elements."""
        try:
            elements = partition(filename=file_path)
            return [str(el) for el in elements]
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

    def extract_knowledge_graph(self, text_elements: List[str]) -> Dict[str, Any]:
        """Extract knowledge graph from text elements using Qwen API."""
        try:
            text = " ".join(text_elements)
            prompt = f"""Extract entities and relationships from this text as JSON:
            Text: {text}
            Return a list containing entities and relationships in this exact format:
            [
                {{"name": "LAYOUTPARSER", "type": "SOFTWARE", "description": "A document analysis toolkit"}},
                {{"name": "ZEJIANG SHEN", "type": "PERSON", "description": "Creator of LayoutParser"}},
                {{"source": "ZEJIANG SHEN", "target": "LAYOUTPARSER", "relationship": "Created the software", "relationship_strength": 9}}
            ]
            Important: Entity names must be capitalized. Relationship strength must be between 1 and 10."""

            print("\nSending request to Qwen API...")
            print(f"Using API key: {dashscope.api_key[:5]}...")

            response = Generation.call(
                model='qwen-max',
                prompt=prompt,
                api_key=dashscope.api_key,
                result_format='text',
                max_tokens=2000,
                temperature=0.1,
                top_p=0.1,
            )

            print(f"\nAPI Response Status: {getattr(response, 'status_code', 'unknown')}")
            print(f"Response Output Type: {type(response.output) if hasattr(response, 'output') else 'No output'}")

            # Default empty response
            entities = []
            relationships = []

            if response and hasattr(response, 'status_code') and response.status_code == 200:
                try:
                    # Parse the outer JSON structure
                    outer_json = json.loads(response.output) if isinstance(response.output, str) else response.output

                    # Get the text field containing the markdown JSON
                    response_text = outer_json.get('text', '')
                    print(f"\nRaw Response Text: {response_text}")

                    # Extract JSON from markdown code block
                    json_start = response_text.find('```json\n')
                    json_end = response_text.find('\n```', json_start)

                    if json_start >= 0 and json_end > json_start:
                        clean_text = response_text[json_start + 8:json_end].strip()
                    else:
                        # Try to find just the JSON array if no markdown block
                        json_start = response_text.find('[')
                        json_end = response_text.rfind(']') + 1
                        if json_start >= 0 and json_end > json_start:
                            clean_text = response_text[json_start:json_end].strip()
                        else:
                            clean_text = response_text.strip()

                    print(f"\nCleaned JSON Text: {clean_text}")

                    # Parse the actual JSON content
                    result = json.loads(clean_text)
                    print(f"\nParsed JSON: {json.dumps(result, indent=2)}")

                    if isinstance(result, list):
                        entities = [r for r in result if "type" in r]
                        relationships = [r for r in result if "relationship_strength" in r]
                    elif isinstance(result, dict):
                        entities = result.get("entities", [])
                        relationships = result.get("relationships", [])

                    # Validate entity names are capitalized
                    for entity in entities:
                        if "name" in entity:
                            entity["name"] = entity["name"].upper()

                    print(f"\nExtracted Entities: {json.dumps(entities, indent=2)}")
                    print(f"Extracted Relationships: {json.dumps(relationships, indent=2)}")

                except json.JSONDecodeError as e:
                    print(f"\nJSON Decode Error: {str(e)}")
                    print(f"Failed to parse response text: {response_text}")
            else:
                status = getattr(response, 'status_code', 'unknown')
                print(f"\nWarning: API request failed with status: {status}")
                if hasattr(response, 'code'):
                    print(f"Error Code: {response.code}")
                if hasattr(response, 'message'):
                    print(f"Error Message: {response.message}")

            return {
                "entities": entities,
                "relationships": relationships
            }
        except Exception as e:
            print(f"\nError in knowledge graph extraction: {str(e)}")
            if hasattr(e, '__dict__'):
                print(f"Error details: {e.__dict__}")
            return {"entities": [], "relationships": []}

    def save_knowledge_graph(self, graph_data: Dict[str, Any], output_path: str):
        """Save extracted knowledge graph to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
