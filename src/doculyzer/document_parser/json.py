"""
JSON document parser module with caching strategies for the document pointer system.

This module parses JSON documents into structured elements and provides
efficient caching strategies for improved performance.
"""

import functools
import hashlib
import json
import logging
import os
import re
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple

import time

from .base import DocumentParser
from .lru_cache import LRUCache, ttl_cache
from .temporal_semantics import detect_temporal_type, TemporalType, create_semantic_temporal_expression
from ..relationships import RelationshipType
from ..storage import ElementType

logger = logging.getLogger(__name__)


class JSONParser(DocumentParser):
    """Parser for JSON documents with caching for improved performance."""

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
            return element_type in [
                ElementType.ROOT.value,
                ElementType.JSON_OBJECT.value,
                ElementType.JSON_ARRAY.value,
                ElementType.JSON_FIELD.value,
                ElementType.JSON_ITEM.value,
            ]

        except (json.JSONDecodeError, TypeError):
            return False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the JSON parser with caching capabilities."""
        super().__init__(config)

        # Configuration options
        self.config = config or {}
        self.max_preview_length = self.config.get("max_preview_length", 100)
        self.include_field_names = self.config.get("include_field_names", True)
        self.flatten_arrays = self.config.get("flatten_arrays", False)
        self.max_depth = self.config.get("max_depth", 10)  # Prevent infinite recursion
        self.temp_dir = self.config.get("temp_dir", os.path.join(os.path.dirname(__file__), 'temp'))

        # Cache configurations
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # Default 1 hour TTL
        self.max_cache_size = self.config.get("max_cache_size", 128)  # Default max cache size
        self.enable_caching = self.config.get("enable_caching", True)

        # Performance monitoring
        self.enable_performance_monitoring = self.config.get("enable_performance_monitoring", False)
        self.performance_stats = {
            "parse_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_parse_time": 0.0,
            "total_path_generation_time": 0.0,
            "total_element_processing_time": 0.0,
            "total_link_extraction_time": 0.0,
            "method_times": {}
        }

        # Initialize caches
        self.document_cache = LRUCache(max_size=self.max_cache_size, ttl=self.cache_ttl)
        self.json_cache = LRUCache(max_size=min(50, self.max_cache_size), ttl=self.cache_ttl)  # For parsed JSON objects
        self.text_cache = LRUCache(max_size=self.max_cache_size * 2, ttl=self.cache_ttl)

    @staticmethod
    def _load_source_content(source_path: str) -> Tuple[Union[str, dict, list, None], Optional[str]]:
        """
        Load content from a source file with proper error handling.

        Args:
            source_path: Path to the source file

        Returns:
            Tuple of (content, error_message)
            - content: The file content as string, dict, or list
            - error_message: Error message if loading failed, None otherwise
        """
        if not os.path.exists(source_path):
            error_msg = f"Error: Source file not found: {source_path}"
            logger.error(error_msg)
            return None, error_msg

        try:
            # Read as text
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Try to parse as JSON
            try:
                json_data = json.loads(content)
                return json_data, None
            except json.JSONDecodeError as e:
                error_msg = f"Error: Invalid JSON content in {source_path}: {str(e)}"
                logger.error(error_msg)
                return None, error_msg

        except Exception as e:
            error_msg = f"Error: Cannot read content from {source_path}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def _get_or_create_json_object(self, content: Union[str, bytes, dict, list]) -> Any:
        """
        Get a cached JSON object or create one if not cached.

        Args:
            content: JSON content as string, bytes, or Python object

        Returns:
            Parsed JSON object
        """
        # If already a Python object, no need to parse
        if isinstance(content, (dict, list)):
            return content

        # Ensure content is string for JSON parsing
        if isinstance(content, bytes):
            try:
                content_str = content.decode('utf-8')
            except UnicodeDecodeError:
                logger.error("Cannot decode binary content as UTF-8")
                return None
        else:
            content_str = content

        # Generate a key for the parsed JSON cache
        json_cache_key = hashlib.md5(content_str.encode('utf-8')).hexdigest()

        # Try to get cached parsed JSON
        json_obj = None
        if self.enable_caching and json_cache_key:
            json_obj = self.json_cache.get(json_cache_key)
            if json_obj is not None:
                if self.enable_performance_monitoring:
                    self.performance_stats["cache_hits"] += 1
                logger.debug("JSON cache hit")
                return json_obj

        if self.enable_performance_monitoring:
            self.performance_stats["cache_misses"] += 1

        # Parse if not cached
        start_time = time.time()

        try:
            json_obj = json.loads(content_str)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            return None

        parse_time = time.time() - start_time
        logger.debug(f"JSON parsing time: {parse_time:.4f} seconds")

        # Cache the parsed JSON
        if self.enable_caching and json_cache_key:
            self.json_cache.set(json_cache_key, json_obj)

        return json_obj

    def _is_identity_field(self, field_name: str) -> bool:
        """
        Determines if a field likely represents an identity or entity.

        Args:
            field_name: The name of the JSON field

        Returns:
            True if it appears to be an identity field, False otherwise
        """
        # Common entity/identity fields
        common_identities = [
            # General identifiers
            "id", "identifier", "uuid", "guid", "key", "name", "title", "label",
            # People and organizations
            "person", "author", "publisher", "company", "organization", "user", "owner",
            "creator", "editor", "manager", "developer", "provider", "customer",
            # Places
            "country", "city", "state", "province", "location", "address", "region",
            # Descriptors
            "type", "category", "class", "genre", "style", "format", "model", "brand", "version"
        ]

        # Simple text matching approach
        field_lower = field_name.lower()

        # Check if it's in our list of common entities
        if field_lower in common_identities:
            return True
        # Check for possessive forms that suggest identity
        elif field_lower.endswith("'s"):
            base_word = field_lower[:-2]
            return base_word in common_identities
        # Check for compound words containing identity terms (e.g., userId, productName)
        else:
            return any(identity in field_lower and identity != field_lower for identity in common_identities)

    def _resolve_element_text(self, location_data: Dict[str, Any], source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve the plain text representation of a JSON element with caching.

        Args:
            location_data: Content location data
            source_content: Optional preloaded source content

        Returns:
            Plain text representation of the element
        """
        source = location_data.get("source", "")
        element_type = location_data.get("type", "")
        json_path = location_data.get("path", "$")

        # Generate cache key
        cache_key = f"text_{source}_{element_type}_{json_path}"
        if self.enable_caching:
            cached_text = self.text_cache.get(cache_key)
            if cached_text is not None:
                logger.debug(f"Cache hit for element text: {cache_key}")
                return cached_text

        logger.debug(f"Cache miss for element text: {cache_key}")

        # Load and parse the JSON data
        json_data = None
        if source_content is None:
            json_data, error = self._load_source_content(source)
            if error:
                return error
        else:
            # Get or create JSON from provided content
            json_data = self._get_or_create_json_object(source_content)
            if json_data is None:
                return "Error parsing JSON content"

        # Resolve the JSON path to get the specific element
        if json_path == "$":
            target_data = json_data
        else:
            target_data = self._resolve_json_path(json_data, json_path)
            if target_data is None:
                result = f"Element not found at path: {json_path}"
                # Don't cache error results
                return result

        # Get the element name (last part of the path)
        element_name = json_path.split('.')[-1] if '.' in json_path else json_path
        if '[' in element_name:
            # For array items, extract index
            element_name = element_name.split('[')[0]

        # Handle different element types with appropriate text representation
        if element_type == ElementType.JSON_OBJECT.value and isinstance(target_data, dict):
            if element_name == "$":
                result = "Root object"
            else:
                result = element_name  # Just return the object name
        elif element_type == ElementType.JSON_ARRAY.value and isinstance(target_data, list):
            if element_name == "$":
                result = f"Array with {len(target_data)} items"
            else:
                result = f"{element_name} (array with {len(target_data)} items)"
        elif element_type == ElementType.JSON_FIELD.value:
            # For fields, we need to get the parent object and extract the field
            parent_path, field_name = self._split_field_path(json_path)

            # Check for temporal data
            if isinstance(target_data, str):
                temporal_type = detect_temporal_type(target_data)
                if temporal_type is not TemporalType.NONE:
                    semantic_value = create_semantic_temporal_expression(target_data)

                    # Format based on field semantics
                    is_identity = self._is_identity_field(field_name)

                    if is_identity:
                        result = f"{field_name} is {semantic_value}"
                    else:
                        result = f"{field_name}: {semantic_value}"

                    # Cache the result
                    if self.enable_caching:
                        self.text_cache.set(cache_key, result)
                    return result

            # For primitive values, return "name: value"
            if not isinstance(target_data, (dict, list)):
                # Check if this is an identity field
                if self._is_identity_field(field_name):
                    result = f"{field_name} is \"{target_data}\""
                else:
                    result = f"{field_name}: {target_data}"
            else:
                # For container values, just return the name
                result = field_name
        elif element_type == ElementType.JSON_ITEM.value and json_path.endswith("]"):
            # For array items, check if it might be a temporal value
            if isinstance(target_data, str):
                temporal_type = detect_temporal_type(target_data)
                if temporal_type is not TemporalType.NONE:
                    semantic_value = create_semantic_temporal_expression(target_data)

                    # Get index from path
                    index = json_path.split('[')[-1].rstrip(']')
                    result = f"Item {index}: {semantic_value}"

                    # Cache the result
                    if self.enable_caching:
                        self.text_cache.set(cache_key, result)
                    return result

            # Regular processing for non-temporal values
            if isinstance(target_data, (dict, list)):
                item_type = "object" if isinstance(target_data, dict) else "array"
                result = f"Item {json_path.split('[')[-1].rstrip(']')} ({item_type})"
            else:
                # For primitive values in an array, return the value
                result = str(target_data)
        else:
            # Default case: try to provide a meaningful representation
            if isinstance(target_data, dict):
                result = element_name if element_name != "$" else "Root object"
            elif isinstance(target_data, list):
                result = f"{element_name} (array)" if element_name != "$" else "Array"
            else:
                # Check for temporal values
                if isinstance(target_data, str):
                    temporal_type = detect_temporal_type(target_data)
                    if temporal_type is not TemporalType.NONE:
                        semantic_value = create_semantic_temporal_expression(target_data)
                        if element_name != "$":
                            result = f"{element_name}: {semantic_value}"
                        else:
                            result = semantic_value

                        # Cache the result
                        if self.enable_caching:
                            self.text_cache.set(cache_key, result)
                        return result

                # For primitive values, include the name if available
                if element_name != "$":
                    result = f"{element_name}: {target_data}"
                else:
                    result = str(target_data)

        # Cache the result
        if self.enable_caching:
            self.text_cache.set(cache_key, result)

        return result

    def _resolve_element_content(self, location_data: Dict[str, Any],
                                 source_content: Optional[Union[str, bytes]] = None) -> str:
        """
        Resolve content for specific JSON element types with caching.

        Args:
            location_data: Content location data
            source_content: Optional preloaded source content

        Returns:
            Resolved content as properly formatted JSON
        """
        source = location_data.get("source", "")
        element_type = location_data.get("type", "")
        json_path = location_data.get("path", "$")

        # Generate cache key
        if self.enable_caching:
            cache_key = f"content_{source}_{element_type}_{json_path}"
            cached_content = self.text_cache.get(cache_key)
            if cached_content is not None:
                logger.debug(f"Content cache hit for {cache_key}")
                return cached_content

        logger.debug(f"Content cache miss for {json_path}")

        # Load the content if not provided
        json_data = None
        if source_content is None:
            json_data, error = self._load_source_content(source)
            if error:
                error_result = json.dumps({"error": error})
                return error_result
        else:
            # Get or create JSON from provided content
            json_data = self._get_or_create_json_object(source_content)
            if json_data is None:
                error_result = json.dumps({"error": "Error parsing JSON content"})
                return error_result

        # Extract the element name from the path
        element_name = json_path.split('.')[-1] if '.' in json_path else json_path
        if '[' in element_name:
            # For array items, use the array name
            element_name = element_name.split('[')[0]
        if element_name == "$":
            element_name = "root"

        # Resolve the JSON path to get the specific element
        try:
            if json_path == "$":
                # Root element, return the entire JSON
                target_data = json_data
                result = json.dumps(target_data, indent=2)
            else:
                # Parse the JSON path to navigate to the specific element
                target_data = self._resolve_json_path(json_data, json_path)

                if target_data is None:
                    error_result = json.dumps({"error": f"Element not found at path: {json_path}"})
                    return error_result

                # Check for temporal values in string fields
                if isinstance(target_data, str):
                    temporal_type = detect_temporal_type(target_data)

                    if temporal_type is not TemporalType.NONE:
                        # Create an enriched representation with semantic information
                        semantic_value = create_semantic_temporal_expression(target_data)

                        # Add semantic information to output
                        enriched_data = {
                            "value": target_data,
                            "semantics": {
                                "temporal_type": temporal_type.name,
                                "semantic_expression": semantic_value
                            }
                        }

                        # Return the enriched content
                        result = json.dumps({element_name: enriched_data}, indent=2)

                        # Cache the result
                        if self.enable_caching:
                            self.text_cache.set(cache_key, result)

                        return result

                # Handle specific element types
                if element_type == ElementType.JSON_OBJECT.value and isinstance(target_data, dict):
                    # Scan for temporal values in the object
                    for key, value in target_data.items():
                        if isinstance(value, str):
                            temporal_type = detect_temporal_type(value)
                            if temporal_type is not TemporalType.NONE:
                                # Enrich the value with semantic information
                                target_data[key] = {
                                    "value": value,
                                    "semantics": {
                                        "temporal_type": temporal_type.name,
                                        "semantic_expression": create_semantic_temporal_expression(value)
                                    }
                                }

                    # Return object with its name as key
                    result = json.dumps({element_name: target_data}, indent=2)

                elif element_type == ElementType.JSON_ARRAY.value and isinstance(target_data, list):
                    # Process array items for temporal values
                    for i, item in enumerate(target_data):
                        if isinstance(item, str):
                            temporal_type = detect_temporal_type(item)
                            if temporal_type is not TemporalType.NONE:
                                # Enrich the value with semantic information
                                target_data[i] = {
                                    "value": item,
                                    "semantics": {
                                        "temporal_type": temporal_type.name,
                                        "semantic_expression": create_semantic_temporal_expression(item)
                                    }
                                }

                    # Return array with its name as key
                    result = json.dumps({element_name: target_data}, indent=2)

                elif element_type == ElementType.JSON_FIELD.value:
                    # For fields, extract the field name and return as a named object
                    parent_path, field_name = self._split_field_path(json_path)

                    # Check for temporal data
                    if isinstance(target_data, str):
                        temporal_type = detect_temporal_type(target_data)
                        if temporal_type is not TemporalType.NONE:
                            # Enrich with semantic information
                            enriched_data = {
                                "value": target_data,
                                "semantics": {
                                    "temporal_type": temporal_type.name,
                                    "semantic_expression": create_semantic_temporal_expression(target_data)
                                }
                            }
                            result = json.dumps({field_name: enriched_data}, indent=2)
                        else:
                            result = json.dumps({field_name: target_data}, indent=2)
                    else:
                        result = json.dumps({field_name: target_data}, indent=2)

                elif element_type == ElementType.JSON_ITEM.value and json_path.endswith("]"):
                    # For array items, extract the index
                    match = re.search(r'\[(\d+)\]$', json_path)
                    if match:
                        try:
                            index = int(match.group(1))

                            # Check for temporal values
                            if isinstance(target_data, str):
                                temporal_type = detect_temporal_type(target_data)
                                if temporal_type is not TemporalType.NONE:
                                    # Enrich with semantic information
                                    enriched_data = {
                                        "value": target_data,
                                        "semantics": {
                                            "temporal_type": temporal_type.name,
                                            "semantic_expression": create_semantic_temporal_expression(target_data)
                                        }
                                    }
                                    # Return as a named object with index
                                    result = json.dumps({f"{element_name}[{index}]": enriched_data}, indent=2)
                                else:
                                    # Return as a named object with index
                                    result = json.dumps({f"{element_name}[{index}]": target_data}, indent=2)
                            else:
                                # Return as a named object with index
                                result = json.dumps({f"{element_name}[{index}]": target_data}, indent=2)

                        except (ValueError, IndexError):
                            error_result = json.dumps({"error": f"Invalid array index in path: {json_path}"})
                            return error_result
                    else:
                        error_result = json.dumps({"error": f"Invalid array item path: {json_path}"})
                        return error_result

                else:
                    # Default: return as a named object
                    result = json.dumps({element_name: target_data}, indent=2)
        except Exception as e:
            error_result = json.dumps({"error": f"Error processing JSON: {str(e)}"})
            logger.error(f"Error resolving element content: {str(e)}")
            return error_result

        # Cache the result
        if self.enable_caching:
            self.text_cache.set(cache_key, result)

        return result

    @ttl_cache(maxsize=256, ttl=3600)  # Cache resolution for 1 hour
    def _resolve_json_path(self, data: Any, path: str) -> Any:
        """
        Resolve a JSON path to find the targeted element.
        Uses caching for improved performance.

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

    @ttl_cache(256, 3600)
    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a JSON document into structured elements with caching for performance.

        Args:
            doc_content: Document content and metadata

        Returns:
            Dictionary with document metadata, elements, relationships, and extracted links
        """
        start_time = time.time()

        # Extract metadata from doc_content
        source_id = doc_content["id"]
        metadata = doc_content.get("metadata", {}).copy()

        # Generate document cache key
        if self.enable_caching:
            if "content" in doc_content:
                if isinstance(doc_content["content"], (str, bytes)):
                    content_hash = self._generate_hash(doc_content["content"])
                else:
                    # For dict/list content, convert to JSON string first
                    content_hash = self._generate_hash(json.dumps(doc_content["content"], sort_keys=True))
            elif "binary_path" in doc_content:
                content_hash = self._generate_hash(
                    doc_content["binary_path"] + str(os.path.getmtime(doc_content["binary_path"])))
            else:
                content_hash = self._generate_hash(str(doc_content))

            doc_cache_key = f"{source_id}_{content_hash}"

            # Check document cache
            cached_doc = self.document_cache.get(doc_cache_key)
            if cached_doc is not None:
                logger.info(f"Document cache hit for {source_id}")
                if self.enable_performance_monitoring:
                    self.performance_stats["cache_hits"] += 1
                    self.performance_stats["parse_count"] += 1
                return cached_doc

        logger.info(f"Document cache miss for {source_id}")
        if self.enable_performance_monitoring:
            self.performance_stats["cache_misses"] += 1

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
            elif isinstance(content, (dict, list)):
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

        # Initialize relationships list
        relationships = []

        # Parse JSON structure recursively with relationships
        self._parse_json_element(json_data, doc_id, root_id, source_id, elements, relationships, "$", 0)

        # Extract links from the document
        extract_links_start = time.time()
        links = self._extract_links(json.dumps(json_data), root_id)
        if self.enable_performance_monitoring:
            self.performance_stats["total_link_extraction_time"] += time.time() - extract_links_start

        # Create result
        result = {
            "document": document,
            "elements": elements,
            "links": links,
            "relationships": relationships
        }

        # Add performance metrics if enabled
        total_time = time.time() - start_time
        if self.enable_performance_monitoring:
            self.performance_stats["parse_count"] += 1
            self.performance_stats["total_parse_time"] += total_time
            result["performance"] = self.get_performance_stats()

        # Cache the document
        if self.enable_caching:
            self.document_cache.set(doc_cache_key, result)

        return result

    @ttl_cache(maxsize=256, ttl=3600)
    def _parse_json_element(self, data: Any, doc_id: str, parent_id: str, source_id: str,
                            elements: List[Dict[str, Any]], relationships: List[Dict[str, Any]],
                            json_path: str, depth: int) -> None:
        """
        Recursively parse a JSON element and its children, creating relationship records.
        Uses performance monitoring for optimization analysis.

        Args:
            data: The JSON data to parse
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier
            elements: List to add elements to
            relationships: List to add relationships to
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
                "element_type": ElementType.JSON_OBJECT.value,
                "parent_id": parent_id,
                "content_preview": object_preview,
                "content_location": json.dumps({
                    "source": source_id,
                    "type": ElementType.JSON_OBJECT.value,
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

            # Create relationship from parent to object (CONTAINS)
            contains_relationship = {
                "relationship_id": self._generate_id("rel_"),
                "source_id": parent_id,
                "target_id": object_id,
                "relationship_type": RelationshipType.CONTAINS.value,
                "metadata": {
                    "confidence": 1.0
                }
            }
            relationships.append(contains_relationship)

            # Create relationship from object to parent (CONTAINED_BY)
            contained_by_relationship = {
                "relationship_id": self._generate_id("rel_"),
                "source_id": object_id,
                "target_id": parent_id,
                "relationship_type": RelationshipType.CONTAINED_BY.value,
                "metadata": {
                    "confidence": 1.0
                }
            }
            relationships.append(contained_by_relationship)

            # Process each field
            for key, value in data.items():
                field_path = f"{json_path}.{key}"

                # Create field element
                field_id = self._generate_id("field_")
                field_preview = self._get_preview(value)

                # Check for temporal data if this is a string value
                temporal_metadata = {}
                if isinstance(value, str):
                    temporal_type = detect_temporal_type(value)
                    if temporal_type is not TemporalType.NONE:
                        temporal_metadata = {
                            "temporal_type": temporal_type.name,
                            "semantic_value": create_semantic_temporal_expression(value)
                        }

                field_element = {
                    "element_id": field_id,
                    "doc_id": doc_id,
                    "element_type": ElementType.JSON_FIELD.value,
                    "parent_id": object_id,
                    "content_preview": f"{key}: {field_preview}" if self.include_field_names else field_preview,
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": ElementType.JSON_FIELD.value,
                        "path": field_path
                    }),
                    "content_hash": self._generate_hash(json.dumps(value, sort_keys=True) + key),
                    "metadata": {
                        "field_name": key,
                        "field_type": self._get_type(value),
                        "json_path": field_path,
                        "is_identity_field": self._is_identity_field(key),
                        **temporal_metadata
                    }
                }

                elements.append(field_element)

                # Create relationship from object to field (CONTAINS)
                contains_field_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": object_id,
                    "target_id": field_id,
                    "relationship_type": RelationshipType.CONTAINS.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contains_field_relationship)

                # Create relationship from field to object (CONTAINED_BY)
                field_contained_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": field_id,
                    "target_id": object_id,
                    "relationship_type": RelationshipType.CONTAINED_BY.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(field_contained_relationship)

                # Recursively process child elements
                if isinstance(value, (dict, list)) and not (isinstance(value, list) and self.flatten_arrays):
                    self._parse_json_element(value, doc_id, field_id, source_id, elements, relationships, field_path,
                                             depth + 1)

        elif isinstance(data, list):
            # If flattening arrays, add items directly to parent
            if self.flatten_arrays:
                for i, item in enumerate(data):
                    item_path = f"{json_path}[{i}]"
                    item_id = self._generate_id("item_")
                    item_preview = self._get_preview(item)

                    # Check for temporal data if this is a string value
                    temporal_metadata = {}
                    if isinstance(item, str):
                        temporal_type = detect_temporal_type(item)
                        if temporal_type is not TemporalType.NONE:
                            temporal_metadata = {
                                "temporal_type": temporal_type.name,
                                "semantic_value": create_semantic_temporal_expression(item)
                            }

                    item_element = {
                        "element_id": item_id,
                        "doc_id": doc_id,
                        "element_type": ElementType.JSON_ITEM.value,
                        "parent_id": parent_id,
                        "content_preview": item_preview,
                        "content_location": json.dumps({
                            "source": source_id,
                            "type": ElementType.JSON_ITEM.value,
                            "path": item_path
                        }),
                        "content_hash": self._generate_hash(json.dumps(item, sort_keys=True)),
                        "metadata": {
                            "index": i,
                            "item_type": self._get_type(item),
                            "json_path": item_path,
                            **temporal_metadata
                        }
                    }

                    elements.append(item_element)

                    # Create relationship from parent to item (CONTAINS_ARRAY_ITEM)
                    contains_item_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": parent_id,
                        "target_id": item_id,
                        "relationship_type": RelationshipType.CONTAINS_ARRAY_ITEM.value,
                        "metadata": {
                            "confidence": 1.0,
                            "index": i
                        }
                    }
                    relationships.append(contains_item_relationship)

                    # Create relationship from item to parent (CONTAINED_BY)
                    item_contained_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": item_id,
                        "target_id": parent_id,
                        "relationship_type": RelationshipType.CONTAINED_BY.value,
                        "metadata": {
                            "confidence": 1.0
                        }
                    }
                    relationships.append(item_contained_relationship)

                    # Recursively process child elements
                    if isinstance(item, (dict, list)):
                        self._parse_json_element(item, doc_id, item_id, source_id, elements, relationships, item_path,
                                                 depth + 1)
            else:
                # Create array element
                array_id = self._generate_id("arr_")
                array_preview = self._get_preview(data)

                array_element = {
                    "element_id": array_id,
                    "doc_id": doc_id,
                    "element_type": ElementType.JSON_ARRAY.value,
                    "parent_id": parent_id,
                    "content_preview": array_preview,
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": ElementType.JSON_ARRAY.value,
                        "path": json_path
                    }),
                    "content_hash": self._generate_hash(json.dumps(data, sort_keys=True)),
                    "metadata": {
                        "item_count": len(data),
                        "json_path": json_path
                    }
                }

                elements.append(array_element)

                # Create relationship from parent to array (CONTAINS)
                contains_array_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": parent_id,
                    "target_id": array_id,
                    "relationship_type": RelationshipType.CONTAINS.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contains_array_relationship)

                # Create relationship from array to parent (CONTAINED_BY)
                array_contained_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": array_id,
                    "target_id": parent_id,
                    "relationship_type": RelationshipType.CONTAINED_BY.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(array_contained_relationship)

                # Process each item
                for i, item in enumerate(data):
                    item_path = f"{json_path}[{i}]"
                    item_id = self._generate_id("item_")
                    item_preview = self._get_preview(item)

                    # Check for temporal data if this is a string value
                    temporal_metadata = {}
                    if isinstance(item, str):
                        temporal_type = detect_temporal_type(item)
                        if temporal_type is not TemporalType.NONE:
                            temporal_metadata = {
                                "temporal_type": temporal_type.name,
                                "semantic_value": create_semantic_temporal_expression(item)
                            }

                    item_element = {
                        "element_id": item_id,
                        "doc_id": doc_id,
                        "element_type": ElementType.JSON_ITEM.value,
                        "parent_id": array_id,
                        "content_preview": item_preview,
                        "content_location": json.dumps({
                            "source": source_id,
                            "type": ElementType.JSON_ITEM.value,
                            "path": item_path
                        }),
                        "content_hash": self._generate_hash(json.dumps(item, sort_keys=True)),
                        "metadata": {
                            "index": i,
                            "item_type": self._get_type(item),
                            "json_path": item_path,
                            **temporal_metadata
                        }
                    }

                    elements.append(item_element)

                    # Create relationship from array to item (CONTAINS_ARRAY_ITEM)
                    contains_item_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": array_id,
                        "target_id": item_id,
                        "relationship_type": RelationshipType.CONTAINS_ARRAY_ITEM.value,
                        "metadata": {
                            "confidence": 1.0,
                            "index": i
                        }
                    }
                    relationships.append(contains_item_relationship)

                    # Create relationship from item to array (CONTAINED_BY)
                    item_contained_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": item_id,
                        "target_id": array_id,
                        "relationship_type": RelationshipType.CONTAINED_BY.value,
                        "metadata": {
                            "confidence": 1.0
                        }
                    }
                    relationships.append(item_contained_relationship)

                    # Recursively process child elements
                    if isinstance(item, (dict, list)):
                        self._parse_json_element(item, doc_id, item_id, source_id, elements, relationships, item_path,
                                                 depth + 1)

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
            # Check for temporal data
            temporal_type = detect_temporal_type(data)
            if temporal_type is not TemporalType.NONE:
                # Add a marker to indicate this is temporal data
                if len(data) > self.max_preview_length:
                    return f"[TIME] {data[:self.max_preview_length]}..."
                return f"[TIME] {data}"

            # Standard string preview
            if len(data) > self.max_preview_length:
                return data[:self.max_preview_length] + "..."
            return data
        else:
            return str(data)

    @staticmethod
    def _get_type(data: Any) -> str:
        """Get the type of JSON value."""
        if isinstance(data, dict):
            return "object"
        elif isinstance(data, list):
            return "array"
        elif isinstance(data, str):
            # Check if this is a temporal value
            temporal_type = detect_temporal_type(data)
            if temporal_type is not TemporalType.NONE:
                return f"temporal_{temporal_type.name.lower()}"
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

    def _create_root_element(self, doc_id: str, source_id: str) -> Dict[str, Any]:
        """
        Create a document root element.

        Args:
            doc_id: Document ID
            source_id: Source identifier

        Returns:
            Document root element dictionary
        """
        root_id = self._generate_id("doc_root_")
        root_element = {
            "element_id": root_id,
            "doc_id": doc_id,
            "element_type": ElementType.ROOT.value,
            "parent_id": None,
            "content_preview": f"Document: {source_id}",
            "content_location": json.dumps({
                "source": source_id,
                "type": ElementType.ROOT.value
            }),
            "content_hash": self._generate_hash(source_id),
            "metadata": {
                "source_id": source_id,
                "path": "$"
            }
        }
        return root_element

    @staticmethod
    def _generate_id(prefix: str = "") -> str:
        """Generate a unique ID with optional prefix."""
        return f"{prefix}{uuid.uuid4()}"

    @staticmethod
    def _generate_hash(content: Union[str, bytes]) -> str:
        """Generate a hash of content for change detection."""
        if isinstance(content, str):
            return hashlib.md5(content.encode('utf-8')).hexdigest()
        elif isinstance(content, bytes):
            return hashlib.md5(content).hexdigest()
        else:
            # Convert any other type to string first
            return hashlib.md5(str(content).encode('utf-8')).hexdigest()

    def performance_monitor(self, method_name):
        """
        Decorator for monitoring method performance.

        Args:
            method_name: Name of the method to monitor
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_performance_monitoring:
                    return func(*args, **kwargs)

                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time

                # Update statistics
                if method_name not in self.performance_stats["method_times"]:
                    self.performance_stats["method_times"][method_name] = {
                        "calls": 0,
                        "total_time": 0,
                        "min_time": float('inf'),
                        "max_time": 0
                    }

                stats = self.performance_stats["method_times"][method_name]
                stats["calls"] += 1
                stats["total_time"] += elapsed_time
                stats["min_time"] = min(stats["min_time"], elapsed_time)
                stats["max_time"] = max(stats["max_time"], elapsed_time)

                # Log if this is a particularly slow operation
                if elapsed_time > 1.0:  # Log operations taking more than 1 second
                    logger.warning(f"Slow operation: {method_name} took {elapsed_time:.4f} seconds")

                return result

            return wrapper

        return decorator

    def clear_caches(self):
        """Clear all caches."""
        self.document_cache.clear()
        self.json_cache.clear()
        self.text_cache.clear()
        logger.info("All caches cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the caches.

        Returns:
            Dictionary with cache statistics
        """
        if not self.enable_caching:
            return {"caching_enabled": False}

        return {
            "caching_enabled": True,
            "document_cache": {
                "size": len(self.document_cache.cache),
                "max_size": self.document_cache.max_size,
                "ttl": self.document_cache.ttl
            },
            "json_cache": {
                "size": len(self.json_cache.cache),
                "max_size": self.json_cache.max_size,
                "ttl": self.json_cache.ttl
            },
            "text_cache": {
                "size": len(self.text_cache.cache),
                "max_size": self.text_cache.max_size,
                "ttl": self.text_cache.ttl
            }
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance statistics
        """
        if not self.enable_performance_monitoring:
            return {"performance_monitoring_enabled": False}

        stats = self.performance_stats.copy()

        # Add derived metrics
        if stats["parse_count"] > 0:
            stats["avg_parse_time"] = stats["total_parse_time"] / stats["parse_count"]

        # Add cache efficiency
        total_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_requests > 0:
            stats["cache_hit_ratio"] = stats["cache_hits"] / total_requests

        return {
            "performance_monitoring_enabled": True,
            **stats
        }

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            "parse_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_parse_time": 0.0,
            "total_path_generation_time": 0.0,
            "total_element_processing_time": 0.0,
            "total_link_extraction_time": 0.0,
            "method_times": {}
        }
        logger.info("Performance statistics reset")
