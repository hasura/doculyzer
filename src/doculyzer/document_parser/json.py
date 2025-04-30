"""
JSON document parser module for the document pointer system.

This module parses JSON documents into structured elements.
"""

import hashlib
import json
import logging
import os
import re
import uuid
from typing import Dict, Any, List, Optional, Union

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
        self.temp_dir = self.config.get("temp_dir", os.path.join(os.path.dirname(__file__), 'temp'))

    def _resolve_element_content(self, location_data: Dict[str, Any],
                                 source_content: Optional[Union[str, bytes]] = None) -> str:
        """
        Resolve content for specific JSON element types.

        Args:
            location_data: Content location data
            source_content: Optional pre-loaded source content

        Returns:
            Resolved content string
        """
        source = location_data.get("source", "")
        element_type = location_data.get("type", "")
        json_path = location_data.get("path", "$")

        # Load the content if not provided
        json_data = None
        if source_content is None:
            # Check if source is a file path
            if os.path.exists(source):
                try:
                    with open(source, 'r', encoding='utf-8') as f:
                        content = f.read()
                    json_data = json.loads(content)
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    raise ValueError(f"Error loading JSON from file: {str(e)}")
            else:
                raise ValueError(f"Source file not found: {source}")
        else:
            # Parse JSON from provided content
            if isinstance(source_content, bytes):
                try:
                    content = source_content.decode('utf-8')
                except UnicodeDecodeError:
                    raise ValueError("Cannot decode binary content as JSON")
            else:
                content = source_content

            try:
                json_data = json.loads(content)
            except json.JSONDecodeError as e:
                # If content is already a Python dict or list, use it directly
                if isinstance(content, (dict, list)):
                    json_data = content
                else:
                    raise ValueError(f"Error parsing JSON content: {str(e)}")

        # Resolve the JSON path to get the specific element
        if json_path == "$":
            # Return the entire document if root path
            if element_type == "root":
                return json.dumps(json_data, indent=2)
            else:
                target_data = json_data
        else:
            # Parse the JSON path to navigate to the specific element
            target_data = self._resolve_json_path(json_data, json_path)

            if target_data is None:
                return f"Element not found at path: {json_path}"

        # Handle specific element types
        if element_type == "json_object" and isinstance(target_data, dict):
            return json.dumps(target_data, indent=2)

        elif element_type == "json_array" and isinstance(target_data, list):
            return json.dumps(target_data, indent=2)

        elif element_type == "json_field":
            # For fields, we need to get the parent object and extract the field
            parent_path, field_name = self._split_field_path(json_path)
            parent_data = self._resolve_json_path(json_data, parent_path)

            if isinstance(parent_data, dict) and field_name in parent_data:
                field_value = parent_data[field_name]
                if isinstance(field_value, (dict, list)):
                    return json.dumps(field_value, indent=2)
                else:
                    return str(field_value)
            else:
                return f"Field '{field_name}' not found in parent object at path: {parent_path}"

        elif element_type == "json_item" and json_path.endswith("]"):
            # For array items, we need to parse the array index
            match = re.search(r'\[(\d+)\]$', json_path)
            if match:
                try:
                    index = int(match.group(1))
                    # Already resolved with json_path, so just format the result
                    if isinstance(target_data, (dict, list)):
                        return json.dumps(target_data, indent=2)
                    else:
                        return str(target_data)
                except (ValueError, IndexError):
                    return f"Invalid array index in path: {json_path}"
            else:
                return f"Invalid array item path: {json_path}"

        # Default: return formatted JSON for the element
        if isinstance(target_data, (dict, list)):
            return json.dumps(target_data, indent=2)
        else:
            return str(target_data)

    def supports_location(self, content_location: str) -> bool:
        """
        Check if this parser supports resolving the given location.

        Args:
            content_location: Content location pointer

        Returns:
            True if supported, False otherwise
        """
        try:
            location_data = json.loads(content_location)
            source = location_data.get("source", "")
            element_type = location_data.get("type", "")

            # If source is a file, check if it exists and is a JSON file
            if os.path.exists(source) and os.path.isfile(source):
                _, ext = os.path.splitext(source.lower())
                return ext == '.json'

            # For non-file sources, check if we have a JSON element type
            return element_type in ["root", "json_object", "json_array", "json_field", "json_item"]

        except (json.JSONDecodeError, TypeError):
            return False

    @staticmethod
    def _resolve_json_path(data: Any, path: str) -> Any:
        """
        Resolve a JSON path to find the targeted element.

        Args:
            data: The JSON data
            path: JSON path (e.g., "$.users[0].name")

        Returns:
            The resolved data element or None if not found
        """
        if path == "$":
            return data

        # Remove root symbol if present
        if path.startswith("$"):
            path = path[1:]

        parts = []
        # Parse path components
        in_brackets = False
        current_part = ""

        for char in path:
            if char == '.' and not in_brackets:
                if current_part:
                    parts.append(current_part)
                    current_part = ""
            elif char == '[':
                if current_part:
                    parts.append(current_part)
                    current_part = ""
                in_brackets = True
                current_part = '['
            elif char == ']' and in_brackets:
                current_part += ']'
                parts.append(current_part)
                current_part = ""
                in_brackets = False
            else:
                current_part += char

        if current_part:
            parts.append(current_part)

        # Navigate through the parts
        current = data
        for part in parts:
            if part.startswith('[') and part.endswith(']'):
                # Array index
                try:
                    index = int(part[1:-1])
                    if isinstance(current, list) and 0 <= index < len(current):
                        current = current[index]
                    else:
                        return None
                except ValueError:
                    return None
            else:
                # Object field
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None

        return current

    @staticmethod
    def _split_field_path(path: str) -> tuple:
        """
        Split a JSON path into parent path and field name.

        Args:
            path: JSON path (e.g., "$.users.name")

        Returns:
            Tuple of (parent_path, field_name)
        """
        if '.' not in path:
            return "$", path.replace('$', '')

        last_dot = path.rindex('.')
        parent_path = path[:last_dot] if last_dot > 0 else "$"
        field_name = path[last_dot + 1:]

        return parent_path, field_name

    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a JSON document into structured elements.

        Args:
            doc_content: Document content and metadata

        Returns:
            Dictionary with document metadata, elements, relationships, and extracted links
        """
        # Extract metadata from doc_content
        source_id = doc_content["id"]
        metadata = doc_content.get("metadata", {}).copy()

        # Get content from binary_path or direct content
        content = None

        if "binary_path" in doc_content and os.path.exists(doc_content["binary_path"]):
            try:
                with open(doc_content["binary_path"], 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try to read as binary if text fails
                with open(doc_content["binary_path"], 'rb') as f:
                    binary_content = f.read()
                    try:
                        content = binary_content.decode('utf-8')
                    except UnicodeDecodeError:
                        raise ValueError(f"Cannot decode content as text: {doc_content['binary_path']}")
        elif "content" in doc_content:
            content = doc_content["content"]
            if isinstance(content, bytes):
                try:
                    content = content.decode('utf-8')
                except UnicodeDecodeError:
                    raise ValueError("Cannot decode binary content as text")

        if content is None:
            raise ValueError("No content provided for JSON parsing")

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

        # Generate document ID if not present
        doc_id = metadata.get("doc_id", self._generate_id("doc_"))

        # Create document record with metadata
        document = {
            "doc_id": doc_id,
            "doc_type": "json",
            "source": source_id,
            "metadata": metadata,
            "content_hash": doc_content.get("content_hash", self._generate_hash(json.dumps(json_data, sort_keys=True)))
        }

        # Create root element
        elements = [self._create_root_element(doc_id, source_id)]
        root_id = elements[0]["element_id"]

        # Parse JSON structure recursively
        self._parse_json_element(json_data, doc_id, root_id, source_id, elements, "$", 0)

        # Extract links from the document
        links = self._extract_links(json.dumps(json_data), root_id)

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

    def _extract_links(self, content: str, element_id: str) -> List[Dict[str, Any]]:
        """
        Extract links from JSON content.

        Args:
            content: JSON content as a string
            element_id: Source element ID

        Returns:
            List of extracted links
        """
        links = []

        # Extract URLs from string content
        url_pattern = r'(https?://[^\s<>"\'\(\)]+(?:\([\w\d]+\)|(?:[^,.;:`!()\[\]{}<>"\'\s]|/)))'
        urls = re.findall(url_pattern, content)

        # Create link entries
        for url in urls:
            links.append({
                "source_id": element_id,
                "link_text": url,
                "link_target": url,
                "link_type": "url"
            })

        # Extract email addresses
        email_pattern = r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b'
        emails = re.findall(email_pattern, content)

        for email in emails:
            links.append({
                "source_id": element_id,
                "link_text": email,
                "link_target": f"mailto:{email}",
                "link_type": "email"
            })

        return links

    @staticmethod
    def _generate_id(prefix: str = "") -> str:
        """Generate a unique ID with optional prefix."""
        return f"{prefix}{uuid.uuid4()}"

    @staticmethod
    def _generate_hash(content: str) -> str:
        """Generate a hash of content for change detection."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
