"""
JSON document parser module for the document pointer system.

This module parses JSON documents into structured elements.
"""

import hashlib
import json
import logging
import uuid
from typing import Dict, Any, List, Optional

from .base import DocumentParser

logger = logging.getLogger(__name__)


class JSONParser(DocumentParser):
    """Parser for JSON documents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the JSON parser."""
        super().__init__(config)
        self.config = config or {}
        self.max_preview_length = self.config.get("max_preview_length", 100)
        self.include_field_names = self.config.get("include_field_names", True)
        self.flatten_arrays = self.config.get("flatten_arrays", False)
        self.max_depth = self.config.get("max_depth", 10)  # Prevent infinite recursion

    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a JSON document into structured elements."""
        # Extract metadata from doc_content
        source_id = doc_content["id"]
        content = doc_content["content"]
        metadata = doc_content.get("metadata", {}).copy()

        # Generate document ID if not present
        doc_id = metadata.get("doc_id", self._generate_id("doc_"))

        # Try to parse the JSON content
        try:
            if isinstance(content, str):
                json_data = json.loads(content)
            elif isinstance(content, dict) or isinstance(content, list):
                json_data = content
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON content: {str(e)}")
            raise

        # Create document record with metadata
        document = {
            "doc_id": doc_id,
            "doc_type": "json",
            "source": source_id,
            "metadata": metadata,
            "content_hash": doc_content.get("content_hash", "")
        }

        # Create root element
        elements = [self._create_root_element(doc_id, source_id)]
        root_id = elements[0]["element_id"]

        # Parse JSON structure recursively
        self._parse_json_element(json_data, doc_id, root_id, source_id, elements, "$", 0)

        # Extract links from the document
        links = self._extract_links_from_json(json_data, elements)

        # Return the parsed document
        return {
            "document": document,
            "elements": elements,
            "links": links,
            "relationships": []
        }

    def _parse_json_element(self, data: Any, doc_id: str, parent_id: str, source_id: str,
                            elements: List[Dict[str, Any]], json_path: str, depth: int) -> None:
        """
        Recursively parse a JSON element and its children.

        Args:
            data: The JSON data to parse
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier
            elements: List to add elements to
            json_path: The JSON path to this element
            depth: Current recursion depth
        """
        # Prevent infinite recursion
        if depth > self.max_depth:
            logger.warning(f"Max recursion depth reached at {json_path}")
            return

        if isinstance(data, dict):
            # Create object element
            object_id = self._generate_id("obj_")
            object_preview = self._get_preview(data)

            object_element = {
                "element_id": object_id,
                "doc_id": doc_id,
                "element_type": "json_object",
                "parent_id": parent_id,
                "content_preview": object_preview,
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "json_object",
                    "path": json_path
                }),
                "content_hash": self._generate_hash(json.dumps(data, sort_keys=True)),
                "metadata": {
                    "fields": list(data.keys()),
                    "item_count": len(data),
                    "json_path": json_path
                }
            }

            elements.append(object_element)

            # Process each field
            for key, value in data.items():
                field_path = f"{json_path}.{key}"

                # Create field element
                field_id = self._generate_id("field_")
                field_preview = self._get_preview(value)

                field_element = {
                    "element_id": field_id,
                    "doc_id": doc_id,
                    "element_type": "json_field",
                    "parent_id": object_id,
                    "content_preview": f"{key}: {field_preview}" if self.include_field_names else field_preview,
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "json_field",
                        "path": field_path
                    }),
                    "content_hash": self._generate_hash(json.dumps(value, sort_keys=True) + key),
                    "metadata": {
                        "field_name": key,
                        "field_type": self._get_type(value),
                        "json_path": field_path
                    }
                }

                elements.append(field_element)

                # Recursively process child elements
                if isinstance(value, (dict, list)) and not (isinstance(value, list) and self.flatten_arrays):
                    self._parse_json_element(value, doc_id, field_id, source_id, elements, field_path, depth + 1)

        elif isinstance(data, list):
            # If flattening arrays, add items directly to parent
            if self.flatten_arrays:
                for i, item in enumerate(data):
                    item_path = f"{json_path}[{i}]"
                    item_id = self._generate_id("item_")
                    item_preview = self._get_preview(item)

                    item_element = {
                        "element_id": item_id,
                        "doc_id": doc_id,
                        "element_type": "json_item",
                        "parent_id": parent_id,
                        "content_preview": item_preview,
                        "content_location": json.dumps({
                            "source": source_id,
                            "type": "json_item",
                            "path": item_path
                        }),
                        "content_hash": self._generate_hash(json.dumps(item, sort_keys=True)),
                        "metadata": {
                            "index": i,
                            "item_type": self._get_type(item),
                            "json_path": item_path
                        }
                    }

                    elements.append(item_element)

                    # Recursively process child elements
                    if isinstance(item, (dict, list)):
                        self._parse_json_element(item, doc_id, item_id, source_id, elements, item_path, depth + 1)
            else:
                # Create array element
                array_id = self._generate_id("arr_")
                array_preview = self._get_preview(data)

                array_element = {
                    "element_id": array_id,
                    "doc_id": doc_id,
                    "element_type": "json_array",
                    "parent_id": parent_id,
                    "content_preview": array_preview,
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "json_array",
                        "path": json_path
                    }),
                    "content_hash": self._generate_hash(json.dumps(data, sort_keys=True)),
                    "metadata": {
                        "item_count": len(data),
                        "json_path": json_path
                    }
                }

                elements.append(array_element)

                # Process each item
                for i, item in enumerate(data):
                    item_path = f"{json_path}[{i}]"
                    item_id = self._generate_id("item_")
                    item_preview = self._get_preview(item)

                    item_element = {
                        "element_id": item_id,
                        "doc_id": doc_id,
                        "element_type": "json_item",
                        "parent_id": array_id,
                        "content_preview": item_preview,
                        "content_location": json.dumps({
                            "source": source_id,
                            "type": "json_item",
                            "path": item_path
                        }),
                        "content_hash": self._generate_hash(json.dumps(item, sort_keys=True)),
                        "metadata": {
                            "index": i,
                            "item_type": self._get_type(item),
                            "json_path": item_path
                        }
                    }

                    elements.append(item_element)

                    # Recursively process child elements
                    if isinstance(item, (dict, list)):
                        self._parse_json_element(item, doc_id, item_id, source_id, elements, item_path, depth + 1)

    def _get_preview(self, data: Any) -> str:
        """Generate a preview of JSON data."""
        if isinstance(data, dict):
            preview = "{" + ", ".join(f"{key}: ..." for key in list(data.keys())[:3])
            if len(data) > 3:
                preview += ", ..."
            preview += "}"
            return preview
        elif isinstance(data, list):
            preview = "[" + ", ".join("..." for _ in range(min(3, len(data))))
            if len(data) > 3:
                preview += ", ..."
            preview += "]"
            return preview
        elif isinstance(data, str):
            if len(data) > self.max_preview_length:
                return data[:self.max_preview_length] + "..."
            return data
        else:
            return str(data)

    @staticmethod
    def _get_type(data: Any) -> str:
        """Get the type of a JSON value."""
        if isinstance(data, dict):
            return "object"
        elif isinstance(data, list):
            return "array"
        elif isinstance(data, str):
            return "string"
        elif isinstance(data, int):
            return "integer"
        elif isinstance(data, float):
            return "number"
        elif isinstance(data, bool):
            return "boolean"
        elif data is None:
            return "null"
        else:
            return str(type(data).__name__)

    @staticmethod
    def _extract_links_from_json(_data: Any, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract links from JSON content.

        This looks for URLs in string values and creates link entries.

        Args:
            _data: The JSON data
            elements: List of parsed elements

        Returns:
            List of extracted links
        """
        import re

        links = []
        url_pattern = r'https?://[^\s()<>]+(?:\([\w\d]+\)|(?:[^,.;:`!()\[\]{}<>"\'\s]|/))'

        # Find element IDs by type
        element_ids_by_type = {}
        for element in elements:
            element_type = element.get("element_type", "")
            if element_type not in element_ids_by_type:
                element_ids_by_type[element_type] = []
            element_ids_by_type[element_type].append(element["element_id"])

        # Function to extract URLs from a value
        def extract_urls(value, element_id):
            if isinstance(value, str):
                for url in re.findall(url_pattern, value):
                    links.append({
                        "source_id": element_id,
                        "link_text": url,
                        "link_target": url,
                        "link_type": "url"
                    })
            elif isinstance(value, dict):
                for k, v in value.items():
                    extract_urls(v, element_id)
            elif isinstance(value, list):
                for item in value:
                    extract_urls(item, element_id)

        # Extract from each element
        for element in elements:
            element_id = element["element_id"]
            content_preview = element.get("content_preview", "")

            # Extract URLs from preview text
            extract_urls(content_preview, element_id)

        return links

    @staticmethod
    def _generate_id(prefix: str = "") -> str:
        """Generate a unique ID with optional prefix."""
        return f"{prefix}{uuid.uuid4()}"

    @staticmethod
    def _generate_hash(content: str) -> str:
        """Generate a hash of content for change detection."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
