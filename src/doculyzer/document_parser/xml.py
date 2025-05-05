"""
XML document parser module with caching strategies for the document pointer system.

This module parses XML documents into structured elements and provides
semantic textual representations of the data with improved performance.
"""

import functools
import hashlib
import json
import logging
import os
import uuid
from typing import Dict, Any, Optional, Union, Tuple

import time
from lxml import etree

from .base import DocumentParser
from .lru_cache import LRUCache, ttl_cache
from .temporal_semantics import detect_temporal_type, TemporalType, create_semantic_temporal_expression
from ..relationships import RelationshipType
from ..storage import ElementType

logger = logging.getLogger(__name__)


class XmlParser(DocumentParser):
    """Parser for XML documents with caching for improved performance using lxml."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the XML parser with caching capabilities."""
        super().__init__(config)
        # Configuration options
        self.config = config or {}
        self.max_content_preview = self.config.get("max_content_preview", 100)
        self.extract_attributes = self.config.get("extract_attributes", True)
        self.flatten_namespaces = self.config.get("flatten_namespaces", True)
        self.treat_namespaces_as_elements = self.config.get("treat_namespaces_as_elements", False)
        self.extract_namespace_declarations = self.config.get("extract_namespace_declarations", True)
        self.parser_features = self.config.get("parser_features", None)  # No BeautifulSoup parser features needed

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
        self.tree_cache = LRUCache(max_size=min(50, self.max_cache_size), ttl=self.cache_ttl)  # For etree objects
        self.text_cache = LRUCache(max_size=self.max_cache_size * 2, ttl=self.cache_ttl)

    @staticmethod
    def _load_source_content(source_path: str) -> Tuple[Union[str, bytes], Optional[str]]:
        """
        Load content from a source file with proper error handling.

        Args:
            source_path: Path to the source file

        Returns:
            Tuple of (content, error_message)
            - content: The file content as string or bytes
            - error_message: Error message if loading failed, None otherwise
        """
        if not os.path.exists(source_path):
            error_msg = f"Error: Source file not found: {source_path}"
            logger.error(error_msg)
            return None, error_msg

        try:
            # First try to read as text
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return content, None
        except UnicodeDecodeError:
            # If that fails, try to read as binary
            try:
                with open(source_path, 'rb') as f:
                    content = f.read()
                    return content, None
            except Exception as e:
                error_msg = f"Error: Cannot read content from {source_path}: {str(e)}"
                logger.error(error_msg)
                return None, error_msg
        except Exception as e:
            error_msg = f"Error: Cannot read content from {source_path}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def _get_or_create_lxml_root(self, content: Union[str, bytes]) -> Any:
        """
        Get a cached lxml root element or create one if not cached.

        Args:
            content: XML content as string or bytes

        Returns:
            lxml root element
        """
        # Ensure content is bytes for lxml
        if isinstance(content, str):
            xml_bytes = content.encode('utf-8')
        else:
            xml_bytes = content

        # Generate a key for the parsed tree cache
        tree_cache_key = hashlib.md5(xml_bytes).hexdigest()

        # Try to get cached parsed tree
        root = None
        if self.enable_caching and tree_cache_key:
            root = self.tree_cache.get(tree_cache_key)
            if root is not None:
                if self.enable_performance_monitoring:
                    self.performance_stats["cache_hits"] += 1
                logger.debug("Tree cache hit")
                return root

        if self.enable_performance_monitoring:
            self.performance_stats["cache_misses"] += 1

        # Parse if not cached
        start_time = time.time()

        # Use a parser with error recovery
        parser = etree.XMLParser(recover=True, remove_blank_text=True)
        root = etree.fromstring(xml_bytes, parser)

        parse_time = time.time() - start_time
        logger.debug(f"XML parsing time: {parse_time:.4f} seconds")

        # Cache the parsed tree
        if self.enable_caching and tree_cache_key:
            self.tree_cache.set(tree_cache_key, root)

        return root

    @staticmethod
    def _prepare_namespace_dict(namespaces: Optional[Dict[str, str]]) -> Dict[str, str]:
        """
        Prepare namespace dictionary for XPath queries.

        Args:
            namespaces: Optional namespace dictionary

        Returns:
            Prepared namespace dictionary for lxml
        """
        ns_dict = {}
        if namespaces:
            # Add namespaces from location_data
            if "default" in namespaces:
                ns_dict["ns"] = namespaces["default"]
                etree.register_namespace("ns", namespaces["default"])

            # Add other namespaces
            for prefix, uri in namespaces.items():
                if prefix != "default":
                    ns_dict[prefix] = uri
                    etree.register_namespace(prefix, uri)

        return ns_dict

    def _get_element_by_xpath(self, root, xpath: str, namespaces: Optional[Dict[str, str]] = None) -> Optional[Any]:
        """
        Get an element directly using XPath expression.

        Args:
            root: lxml root element
            xpath: XPath expression
            namespaces: Optional namespace dictionary

        Returns:
            Found element or None
        """
        try:
            ns_dict = self._prepare_namespace_dict(namespaces)
            results = root.xpath(xpath, namespaces=ns_dict)
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Error in XPath query '{xpath}': {str(e)}")
            return None

    @staticmethod
    def _get_normalized_tag_name(tag: str) -> str:
        """
        Get normalized tag name by removing namespace prefix.

        Args:
            tag: Element tag name, possibly with namespace

        Returns:
            Normalized tag name without namespace
        """
        if tag is None:
            return "unknown"

        # Handle namespace with Clark notation {namespace}localname
        if '}' in tag:
            return tag.split('}')[1]
        return tag

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
            "element_type": "document_root",
            "parent_id": None,
            "content_preview": f"Document: {source_id}",
            "content_location": json.dumps({
                "source": source_id,
                "type": "root"
            }),
            "content_hash": self._generate_hash(source_id),
            "metadata": {
                "source_id": source_id,
                "path": "/"
            }
        }
        return root_element

    def _generate_hash(self, content):
        """Generate a hash for content, always returning a string."""
        try:
            if isinstance(content, str):
                return hashlib.md5(content.encode('utf-8')).hexdigest()
            elif isinstance(content, bytes):
                return hashlib.md5(content).hexdigest()
            else:
                # Convert any other type to string first
                return hashlib.md5(str(content).encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f"Error generating hash: {str(e)}")
            # Return a UUID as string for fallback
            return str(uuid.uuid4())

    def _generate_id(self, prefix: str = "") -> str:
        """
        Generate a unique ID with an optional prefix.

        Args:
            prefix: Optional ID prefix

        Returns:
            Unique ID string
        """
        return f"{prefix}{uuid.uuid4()}"

    def _resolve_element_text(self, location_data: Dict[str, Any], source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve the plain text representation of an XML element using lxml's native XPath.
        Includes natural language representation of attributes.

        Args:
            location_data: Content location data
            source_content: Optional preloaded source content

        Returns:
            Plain text representation of the element with attributes
        """
        source = location_data.get("source", "")
        element_type = location_data.get("type", "")
        path = location_data.get("path", "")
        namespaces = location_data.get("namespaces", None)

        # Generate cache key
        cache_key = f"text_{source}_{element_type}_{path}"
        if self.enable_caching:
            cached_text = self.text_cache.get(cache_key)
            if cached_text is not None:
                logger.debug(f"Cache hit for element text: {cache_key}")
                return cached_text

        logger.debug(f"Cache miss for element text: {cache_key}")

        # Load content if not provided
        content = source_content
        if content is None:
            content, error = self._load_source_content(source)
            if error:
                return error

        # Process with lxml
        try:
            # Get or create lxml root
            root = self._get_or_create_lxml_root(content)

            # Execute XPath query
            ns_dict = self._prepare_namespace_dict(namespaces)
            elements = root.xpath(path, namespaces=ns_dict)

            if not elements:
                error_msg = f"Element not found at path: {path}"
                logger.warning(error_msg)
                # Don't cache errors to allow retry
                return error_msg

            # Get element and extract text
            element = elements[0]

            # Extract text content based on element type
            if element_type == "xml_text" or "text()" in path:
                # For text nodes or explicit text() XPath
                if isinstance(element, str):
                    # If XPath returns the text node directly
                    text_content = element.strip()
                else:
                    # Try to get text content from an element
                    text_content = element.text.strip() if element.text else ""
            elif element_type == "xml_list":
                # For list containers, get a summary of child elements
                if hasattr(element, 'itertext'):
                    text_content = ''.join(element.itertext()).strip()
                    if len(text_content) > 100:
                        # Summarize if too long
                        text_content = f"List with {len(element)} items"
                else:
                    text_content = f"List with {len(element)} items"
            elif element_type == "xml_object":
                # For object containers, summarize properties
                properties = []
                for child in element:
                    child_name = self._get_normalized_tag_name(child.tag)
                    properties.append(child_name)

                if properties:
                    text_content = f"Object with properties: {', '.join(properties[:5])}" + (
                        "..." if len(properties) > 5 else "")
                else:
                    text_content = "Empty object"
            else:
                # For regular elements, get all contained text
                if hasattr(element, 'itertext'):
                    text_content = ''.join(element.itertext()).strip()
                else:
                    # Fallback for non-element objects
                    text_content = str(element).strip()

            logger.debug(f"Extracted text content: '{text_content}'")

            # Get element name
            if hasattr(element, 'tag'):
                # Handle namespaces in tag names
                element_name = self._get_normalized_tag_name(element.tag)
            else:
                # Use path component as name for text nodes
                path_parts = path.split('/')
                last_part = path_parts[-1].split('[')[0]

                if last_part == "text()":
                    # Get parent element name for text() nodes
                    if len(path_parts) > 1:
                        parent_part = path_parts[-2].split('[')[0]
                        # Remove namespace prefix if present
                        if ':' in parent_part:
                            element_name = parent_part.split(':')[1]
                        else:
                            element_name = parent_part
                    else:
                        element_name = "text"
                else:
                    # Remove namespace prefix if present
                    if ':' in last_part:
                        element_name = last_part.split(':')[1]
                    else:
                        element_name = last_part

            logger.debug(f"Element name: {element_name}")

            # Check if this might be a date/time value
            temporal_type = detect_temporal_type(text_content)

            # Extract attributes (new code)
            attributes = {}
            if hasattr(element, 'attrib') and element.attrib:
                attributes = dict(element.attrib)

                # Check if any attribute values contain temporal data
                for attr_name, attr_value in attributes.items():
                    attr_temporal_type = detect_temporal_type(attr_value)
                    if attr_temporal_type is not TemporalType.NONE:
                        # Convert to semantic expression
                        attributes[attr_name] = (attr_value, create_semantic_temporal_expression(attr_value))

            # Format semantic representation based on node type
            if attributes and not text_content:
                # Attribute-only node
                result = self._format_attribute_only_node(element_name, attributes)
            elif attributes and text_content:
                # Node with both text and attributes
                result = self._format_node_with_text_and_attributes(element_name, text_content, attributes,
                                                                    temporal_type)
            else:
                # Text-only node (existing code)
                if temporal_type is not TemporalType.NONE:
                    result = f"{element_name} is {create_semantic_temporal_expression(text_content)}"
                else:
                    # Format as appropriate for the element type
                    is_container, container_type = self._analyze_container_type(element_name)
                    is_identity_element = self._is_identity_element(element_name)

                    if is_identity_element:
                        result = f"{element_name} is \"{text_content}\""
                    elif is_container:
                        result = f"{element_name} contains \"{text_content}\""
                    else:
                        result = text_content

            logger.debug(f"Formatted text result: {result}")

            # Cache the result
            if self.enable_caching:
                self.text_cache.set(cache_key, result)

            return result

        except Exception as e:
            error_msg = f"Error resolving element text: {str(e)}"
            logger.error(error_msg)
            # Don't cache errors to allow retry
            return error_msg

    @staticmethod
    def _format_attribute_only_node(element_name: str, attributes: Dict[str, Any]) -> str:
        """
        Format an attribute-only node into a natural language representation.

        Args:
            element_name: Name of the element
            attributes: Dictionary of attributes

        Returns:
            Natural language representation
        """
        # Select different patterns based on attribute count
        if len(attributes) == 1:
            # Single attribute
            attr_name, attr_value = next(iter(attributes.items()))

            # Check if attribute value is temporal (already processed)
            if isinstance(attr_value, tuple):
                raw_value, semantic_value = attr_value
                return f"{element_name} has {attr_name} of {semantic_value}"

            # Special handling for common attributes
            if attr_name in ['id', 'name', 'title', 'type', 'class']:
                return f"{element_name} with {attr_name} \"{attr_value}\""
            elif attr_name in ['href', 'src', 'link', 'url']:
                return f"{element_name} links to \"{attr_value}\""
            elif attr_name in ['date', 'time', 'datetime']:
                # Check if this is a time range
                temporal_type = detect_temporal_type(attr_value)
                if temporal_type is not TemporalType.NONE:
                    semantic_value = create_semantic_temporal_expression(attr_value)
                    return f"{element_name} has {attr_name} of {semantic_value}"
                else:
                    return f"{element_name} has {attr_name} of \"{attr_value}\""
            else:
                return f"{element_name} has {attr_name} of \"{attr_value}\""

        else:
            # Multiple attributes
            attr_phrases = []

            # Process each attribute
            for attr_name, attr_value in attributes.items():
                # Handle pre-processed temporal values
                if isinstance(attr_value, tuple):
                    raw_value, semantic_value = attr_value
                    attr_phrases.append(f"has {attr_name} of {semantic_value}")
                    continue

                # Special handling for common attributes
                if attr_name in ['id', 'name']:
                    attr_phrases.append(f"identified as \"{attr_value}\"")
                elif attr_name in ['title', 'label']:
                    attr_phrases.append(f"titled \"{attr_value}\"")
                elif attr_name in ['href', 'src', 'link', 'url']:
                    attr_phrases.append(f"linking to \"{attr_value}\"")
                elif attr_name == 'class':
                    attr_phrases.append(f"of class \"{attr_value}\"")
                elif attr_name == 'type':
                    attr_phrases.append(f"of type \"{attr_value}\"")
                elif attr_name in ['date', 'time', 'datetime']:
                    # Check if this is a time range
                    temporal_type = detect_temporal_type(attr_value)
                    if temporal_type is not TemporalType.NONE:
                        semantic_value = create_semantic_temporal_expression(attr_value)
                        attr_phrases.append(f"with {attr_name} of {semantic_value}")
                    else:
                        attr_phrases.append(f"with {attr_name} of \"{attr_value}\"")
                else:
                    attr_phrases.append(f"with {attr_name} of \"{attr_value}\"")

            # Join attributes with commas and 'and'
            if len(attr_phrases) == 2:
                attr_text = f"{attr_phrases[0]} and {attr_phrases[1]}"
            else:
                attr_text = ", ".join(attr_phrases[:-1]) + f", and {attr_phrases[-1]}"

            return f"{element_name} {attr_text}"

    def _format_node_with_text_and_attributes(self, element_name: str, text_content: str,
                                              attributes: Dict[str, Any], temporal_type: TemporalType) -> str:
        """
        Format a node that has both text content and attributes.

        Args:
            element_name: Name of the element
            text_content: Text content of the element
            attributes: Dictionary of attributes
            temporal_type: Type of temporal data if applicable

        Returns:
            Natural language representation
        """
        # First check for time range in attributes that should be prioritized
        time_range_attr = None
        for attr_name, attr_value in attributes.items():
            if attr_name in ['time', 'period', 'duration', 'range']:
                # Check if the value is already processed as temporal
                if isinstance(attr_value, tuple):
                    raw_value, semantic_value = attr_value
                    time_range_attr = (attr_name, semantic_value)
                    break

                # Otherwise check and process
                if detect_temporal_type(attr_value) is TemporalType.TIME_RANGE:
                    semantic_value = create_semantic_temporal_expression(attr_value)
                    time_range_attr = (attr_name, semantic_value)
                    break

        # If this is a time element with range attribute, prioritize the range
        if time_range_attr and element_name in ['time', 'meeting', 'appointment', 'schedule', 'event']:
            attr_name, semantic_value = time_range_attr
            return f"{element_name} for \"{text_content}\" is {semantic_value}"

        # Select key attributes for compact representation
        key_attrs = {}

        # Prioritize common identifying attributes
        for attr_name in ['id', 'name', 'title', 'type', 'class']:
            if attr_name in attributes:
                # Handle pre-processed temporal values
                if isinstance(attributes[attr_name], tuple):
                    raw_value, semantic_value = attributes[attr_name]
                    key_attrs[attr_name] = semantic_value
                else:
                    key_attrs[attr_name] = attributes[attr_name]

        # Include other attributes if we don't have many key ones
        if len(key_attrs) < 2:
            for attr_name, attr_value in attributes.items():
                if attr_name not in key_attrs and len(key_attrs) < 3:
                    # Handle pre-processed temporal values
                    if isinstance(attr_value, tuple):
                        raw_value, semantic_value = attr_value
                        key_attrs[attr_name] = semantic_value
                    else:
                        key_attrs[attr_name] = attr_value

        # Format attributes
        if len(key_attrs) == 0:
            attr_text = ""
        elif len(key_attrs) == 1:
            attr_name, attr_value = next(iter(key_attrs.items()))
            if attr_name in ['id', 'name']:
                attr_text = f" identified as \"{attr_value}\""
            elif attr_name == 'title':
                attr_text = f" titled \"{attr_value}\""
            elif attr_name in ['date', 'time', 'datetime']:
                attr_text = f" with {attr_name} {attr_value}"
            else:
                attr_text = f" with {attr_name} \"{attr_value}\""
        else:
            attr_phrases = []
            for attr_name, attr_value in key_attrs.items():
                # Handle special case for temporal values
                if isinstance(attr_value, str) and not attr_value.startswith('"'):
                    attr_phrases.append(f"{attr_name}={attr_value}")
                else:
                    attr_phrases.append(f"{attr_name}=\"{attr_value}\"")
            attr_text = f" ({', '.join(attr_phrases)})"

        # Combine with text content
        if temporal_type is not TemporalType.NONE:
            text_repr = create_semantic_temporal_expression(text_content)
            return f"{element_name}{attr_text} is {text_repr}"
        else:
            is_container, container_type = self._analyze_container_type(element_name)
            is_identity_element = self._is_identity_element(element_name)

            if is_identity_element:
                return f"{element_name}{attr_text} is \"{text_content}\""
            elif is_container:
                return f"{element_name}{attr_text} contains \"{text_content}\""
            else:
                # For regular content, put the attributes in parentheses before the content
                if attr_text:
                    return f"{element_name}{attr_text}: {text_content}"
                else:
                    return text_content

    def _resolve_element_content(self, location_data: Dict[str, Any],
                                 source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve content for specific XML element types with lxml's XPath support.

        Args:
            location_data: Content location data
            source_content: Optional preloaded source content

        Returns:
            Resolved content as properly formatted XML
        """
        source = location_data.get("source", "")
        element_type = location_data.get("type", "")
        path = location_data.get("path", "")
        namespaces = location_data.get("namespaces", None)

        # Generate cache key
        if self.enable_caching:
            cache_key = f"content_{source}_{element_type}_{path}"
            cached_content = self.text_cache.get(cache_key)
            if cached_content is not None:
                logger.debug(f"Content cache hit for {cache_key}")
                return cached_content

        logger.debug(f"Content cache miss for {path}")

        # Load content if not provided
        content = source_content
        if content is None:
            content, error = self._load_source_content(source)
            if error:
                error_result = f'<error source="{source}">{error}</error>'
                return error_result

        # Process with lxml
        try:
            # Get or create lxml root
            root = self._get_or_create_lxml_root(content)

            # Special case for root or document
            if not path or path == "/":
                result = etree.tostring(root, encoding='unicode', pretty_print=True)
                if self.enable_caching:
                    self.text_cache.set(cache_key, result)
                return result

            # Execute XPath query
            ns_dict = self._prepare_namespace_dict(namespaces)
            elements = root.xpath(path, namespaces=ns_dict)

            if not elements:
                error_result = f'<error path="{path}">Element not found</error>'
                logger.warning(f"No elements found for path '{path}'")
                return error_result

            # Get the element
            element = elements[0]

            # Process based on type
            if element_type == "xml_text" or "text()" in path:
                # For text nodes
                if isinstance(element, str):
                    # If XPath returns the text node directly
                    text = element.strip()
                else:
                    # Try to get text from element
                    text = element.text.strip() if element.text else ""

                result = f"<text>{text}</text>"
            else:
                # For XML elements
                if hasattr(element, 'tag'):
                    # Use lxml's tostring to serialize the element
                    result = etree.tostring(element, encoding='unicode', pretty_print=True)
                else:
                    # Fallback for non-element objects
                    result = f"<value>{str(element)}</value>"

            # Cache the result
            if self.enable_caching:
                self.text_cache.set(cache_key, result)

            return result

        except Exception as e:
            error_result = f'<error path="{path}">Error processing XML: {str(e)}</error>'
            logger.error(f"Error resolving element content: {str(e)}")
            return error_result

    @ttl_cache(maxsize=256, ttl=3600)
    def _analyze_container_type(self, element_name: str, element=None) -> Tuple[bool, str]:
        """
        Analyzes if an element represents a container and determines its type.
        Uses both semantic naming patterns and structural analysis (if element provided).

        Args:
            element_name: The name of the XML element
            element: Optional XML element for structural analysis

        Returns:
            Tuple of (is_container, container_type), where container_type is one of:
            - "array" for homogeneous list-like containers
            - "object" for heterogeneous structure-like containers
            - "element" for non-containers
        """
        element_lower = element_name.lower()
        is_container = False
        container_type = "element"  # Default type

        # Step 1: Check naming patterns to determine if this is potentially a container

        # Check for plural endings (most common array signal)
        if (element_lower.endswith('s') and not element_lower.endswith('ss') and
                not element_lower.endswith('us') and not element_lower.endswith('is')):
            is_container = True
            container_type = "array"  # Plurals suggest arrays by default

        # Check for common container words
        container_words = {
            # Array-like containers
            "array": "array",
            "list": "array",
            "collection": "array",
            "items": "array",
            "records": "array",
            "entries": "array",

            # Object-like containers
            "map": "object",
            "dictionary": "object",
            "object": "object",
            "container": "object",
            "wrapper": "object",
            "package": "object",

            # Ambiguous containers (need structural analysis)
            "set": "array",  # Default, but may be overridden
            "group": "object",  # Default, but may be overridden
            "data": "object",  # Default, but may be overridden
            "bundle": "object",
            "batch": "array",
            "series": "array",
            "catalog": "object",
            "index": "object",
            "directory": "object",
            "table": "object",
            "results": "array"
        }

        if element_lower in container_words:
            is_container = True
            container_type = container_words[element_lower]

        # Check for compound words with container terms
        if not is_container:
            for word, word_type in container_words.items():
                if word in element_lower and word != element_lower:
                    is_container = True
                    container_type = word_type
                    break

        # Check for container-implying prefixes
        if not is_container:
            collection_prefixes = ["all", "each", "every", "many", "multi"]
            for prefix in collection_prefixes:
                if (element_lower.startswith(prefix) and
                        len(element_lower) > len(prefix) and
                        len(element_name) > len(prefix) and
                        element_name[len(prefix)].isupper()):
                    is_container = True
                    # "all" and "many" suggest arrays, "each" suggests objects
                    container_type = "array" if prefix in ["all", "many", "multi"] else "object"
                    break

        # Step 2: If we have the actual element, use structural analysis to confirm
        # or override the type determined by naming conventions
        if is_container and element is not None:
            children = list(element)

            # If no children or only one child, this might still be a container
            # but we can't determine the type from structure
            if len(children) > 1:
                # Get all child tag names
                child_tags = [self._get_normalized_tag_name(child.tag) for child in children]

                # Criteria 1: All children have the same tag name - definitely an array
                if len(set(child_tags)) == 1:
                    container_type = "array"
                    return True, container_type

                # Criteria 2: Most children (>80%) have the same tag name - likely an array
                tag_counts = {}
                for tag in child_tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

                most_common_tag_count = max(tag_counts.values()) if tag_counts else 0
                if most_common_tag_count / len(children) > 0.8:
                    container_type = "array"
                    return True, container_type

                # Criteria 3: Many different child tag types - likely an object
                if len(set(child_tags)) > len(children) * 0.5:
                    container_type = "object"
                    return True, container_type

        return is_container, container_type

    def _extract_xml_links_direct(self, lxml_root, elements, namespaces):
        """
        Extract links directly using lxml's element attributes.

        Args:
            lxml_root: lxml root element
            elements: Processed elements
            namespaces: XML namespaces

        Returns:
            List of extracted links
        """
        links = []

        # Create a map of element paths to element IDs
        element_map = {
            json.loads(element.get("content_location", "{}")).get("path", ""): element.get("element_id")
            for element in elements
            if element.get("element_type") in [ElementType.XML_ELEMENT.value, ElementType.XML_TEXT.value,
                                               ElementType.XML_LIST.value, ElementType.XML_OBJECT.value]
               and element.get("content_location")
        }

        # Look for various link-like attributes
        link_attrs = [
            "href", "src", "xlink:href", "uri", "url", "link",
            "reference", "ref", "target", "to", "from"
        ]

        # Process each element
        tree = etree.ElementTree(lxml_root)
        for element in lxml_root.iter():
            # Get element path
            element_path = tree.getpath(element)
            element_id = element_map.get(element_path)

            if not element_id:
                continue

            # Check attributes for links
            for attr_name, attr_value in element.attrib.items():
                # Normalize namespace prefixed attributes
                attr_local_name = attr_name.split('}')[-1] if '}' in attr_name else attr_name

                # Check if this is a link attribute
                if attr_local_name in link_attrs or any(attr_local_name.endswith(f":{name}") for name in link_attrs):
                    if attr_value and isinstance(attr_value, str):
                        # Determine link type
                        if "href" in attr_local_name:
                            link_type = RelationshipType.LINK.value
                        elif "src" in attr_local_name:
                            link_type = RelationshipType.LINK.value
                        else:
                            link_type = RelationshipType.REFERENCED_BY.value

                        links.append({
                            "source_id": element_id,
                            "link_text": f"{attr_local_name}='{attr_value}'",
                            "link_target": attr_value,
                            "link_type": link_type
                        })

        return links

    def _process_document_structure(self, lxml_root, doc_id, parent_id, source_id, namespaces):
        """
        Process XML document structure using direct XPath and lxml.

        Args:
            lxml_root: lxml root element (for XPath queries)
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier
            namespaces: XML namespaces

        Returns:
            Tuple of (elements, relationships)
        """
        elements = []
        relationships = []

        # Create a map to track element paths to IDs
        element_path_to_id = {}
        # Map to store lxml elements by path for container type detection
        element_path_to_lxml = {}

        # Create the document root element
        root_tag = lxml_root.tag
        root_name = self._get_normalized_tag_name(root_tag)

        # Check if root is a container and determine type
        is_container, container_type = self._analyze_container_type(root_name, lxml_root)

        # Set the appropriate element_type based on container analysis
        if container_type == "array":
            element_type = "xml_list"  # For array-like containers
        elif container_type == "object":
            element_type = "xml_object"  # For object-like containers
        else:
            element_type = "xml_element"  # For non-containers

        # Create XML root element
        xml_root_id = self._generate_id("xml_root_")
        xml_root_element = {
            "element_id": xml_root_id,
            "doc_id": doc_id,
            "element_type": element_type,  # Use determined element type
            "parent_id": parent_id,
            "content_preview": f"<{root_name}>",
            "content_location": json.dumps({
                "source": source_id,
                "type": element_type,  # Also store it in the content location
                "path": "/",
                "namespaces": namespaces
            }),
            "content_hash": self._generate_hash(etree.tostring(lxml_root)),
            "metadata": {
                "element_name": root_name,
                "has_attributes": bool(lxml_root.attrib),
                "attributes": dict(lxml_root.attrib) if self.extract_attributes else {},
                "path": "/",
                "text": lxml_root.text.strip() if lxml_root.text else "",
                "is_container": is_container,
                "container_type": container_type,
                "child_count": len(lxml_root) if is_container else 0
            }
        }
        elements.append(xml_root_element)
        element_path_to_id["/"] = xml_root_id
        element_path_to_lxml["/"] = lxml_root

        # Create relationship between document root and XML root
        relationship = {
            "relationship_id": self._generate_id("rel_"),
            "source_id": parent_id,
            "target_id": xml_root_id,
            "relationship_type": RelationshipType.CONTAINS.value,
            "metadata": {
                "confidence": 1.0
            }
        }
        relationships.append(relationship)
        relationship = {
            "relationship_id": self._generate_id("rel_"),
            "source_id": xml_root_id,
            "target_id": parent_id,
            "relationship_type": RelationshipType.CONTAINED_BY.value,
            "metadata": {
                "confidence": 1.0
            }
        }
        relationships.append(relationship)

        # Process all elements recursively
        # Use lxml's native iter for efficiency
        tree = etree.ElementTree(lxml_root)
        for element in lxml_root.iter():
            # Skip the root element since we already processed it
            if element == lxml_root:
                continue

            # Get element path
            element_path = tree.getpath(element)

            # Generate element ID
            element_name = self._get_normalized_tag_name(element.tag)
            element_id = self._generate_id(f"xml_elem_{element_name}_")

            # Get parent path and ID
            parent_path = '/'.join(element_path.split('/')[:-1]) or '/'
            parent_id = element_path_to_id.get(parent_path, xml_root_id)

            # Check if element is a container and determine type
            is_container, container_type = self._analyze_container_type(element_name, element)

            # Set the appropriate element_type based on container analysis
            if container_type == "array":
                element_type = "xml_list"  # For array-like containers
            elif container_type == "object":
                element_type = "xml_object"  # For object-like containers
            else:
                element_type = "xml_element"  # For non-containers

            # Create element metadata
            element_data = {
                "element_id": element_id,
                "doc_id": doc_id,
                "element_type": element_type,  # Use determined element type
                "parent_id": parent_id,
                "content_preview": f"<{element_name}>",
                "content_location": json.dumps({
                    "source": source_id,
                    "type": element_type,  # Also store it in the content location
                    "path": element_path,
                    "namespaces": namespaces
                }),
                "content_hash": self._generate_hash(etree.tostring(element)),
                "metadata": {
                    "element_name": element_name,
                    "has_attributes": bool(element.attrib),
                    "attributes": dict(element.attrib) if self.extract_attributes else {},
                    "path": element_path,
                    "text": element.text.strip() if element.text else "",
                    "is_container": is_container,
                    "container_type": container_type,
                    "child_count": len(element) if is_container else 0
                }
            }
            elements.append(element_data)
            element_path_to_id[element_path] = element_id
            element_path_to_lxml[element_path] = element

            # Determine the relationship type
            # If the parent is an array container, use a more specific relationship type
            parent_element = element_path_to_lxml.get(parent_path)
            relationship_type = RelationshipType.CONTAINS.value

            if parent_element:
                parent_name = self._get_normalized_tag_name(parent_element.tag)
                parent_is_container, parent_container_type = self._analyze_container_type(parent_name, parent_element)

                if parent_is_container and parent_container_type == "array":
                    relationship_type = RelationshipType.CONTAINS_ARRAY_ITEM.value

            relationship = {
                "relationship_id": self._generate_id("rel_"),
                "source_id": parent_id,
                "target_id": element_id,
                "relationship_type": relationship_type,
                "metadata": {
                    "confidence": 1.0
                }
            }
            relationships.append(relationship)
            relationship = {
                "relationship_id": self._generate_id("rel_"),
                "source_id": element_id,
                "target_id": parent_id,
                "relationship_type": RelationshipType.CONTAINED_BY.value,
                "metadata": {
                    "confidence": 1.0
                }
            }
            relationships.append(relationship)

            # Process text nodes if they have content
            if element.text and element.text.strip():
                text_id = self._generate_id("text_")
                text_content = element.text.strip()
                text_preview = text_content[:self.max_content_preview] + (
                    "..." if len(text_content) > self.max_content_preview else "")

                # Create text path
                text_path = f"{element_path}/text()[1]"

                # Check for temporal data
                temporal_type = detect_temporal_type(text_content)
                temporal_metadata = {"temporal_type": temporal_type.name}

                # Add semantic representation if it's a temporal value
                if temporal_type is not TemporalType.NONE:
                    temporal_metadata["semantic_value"] = create_semantic_temporal_expression(text_content)

                text_element = {
                    "element_id": text_id,
                    "doc_id": doc_id,
                    "element_type": "xml_text",
                    "parent_id": element_id,
                    "content_preview": text_preview,
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "xml_text",
                        "path": text_path,
                        "namespaces": namespaces
                    }),
                    "content_hash": self._generate_hash(text_content),
                    "metadata": {
                        "parent_element": element_name,
                        "path": text_path,
                        "text": text_content,
                        **temporal_metadata
                    }
                }
                elements.append(text_element)
                element_path_to_id[text_path] = text_id

                # Create relationship for text
                relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": element_id,
                    "target_id": text_id,
                    "relationship_type": RelationshipType.CONTAINS_TEXT.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(relationship)
                # Create relationship for text
                relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": text_id,
                    "target_id": element_id,
                    "relationship_type": RelationshipType.CONTAINED_BY.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(relationship)

        return elements, relationships

    def _extract_document_metadata(self, content: Union[str, bytes], base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from XML document with lxml.

        Args:
            content: XML content
            base_metadata: Base metadata from content source

        Returns:
            Dictionary of document metadata
        """
        # Generate cache key
        if self.enable_caching:
            content_hash = self._generate_hash(content)
            cache_key = f"metadata_{content_hash}"

            # Check metadata cache
            cached_metadata = self.text_cache.get(cache_key)
            if cached_metadata is not None:
                logger.debug(f"Metadata cache hit for {cache_key}")
                return cached_metadata

        logger.debug(f"Metadata cache miss")

        metadata = base_metadata.copy()

        try:
            # Use lxml for processing
            lxml_root = self._get_or_create_lxml_root(content)

            # Extract root element name
            root_tag = lxml_root.tag
            if '}' in root_tag:
                root_name = root_tag.split('}')[1]
            else:
                root_name = root_tag

            metadata["root_element"] = root_name

            # Extract namespace declarations
            if self.extract_namespace_declarations:
                namespaces = {}
                # Get namespaces from lxml
                for prefix, uri in lxml_root.nsmap.items():
                    if prefix is None:
                        namespaces["default"] = uri
                    else:
                        namespaces[prefix] = uri

                if namespaces:
                    metadata["namespaces"] = namespaces

            # Basic document statistics
            metadata["element_count"] = len([e for e in lxml_root.iter()])

            # Try to detect schema or DTD information
            schema_locations = []
            for attr_name, attr_value in lxml_root.attrib.items():
                if attr_name.endswith("}schemaLocation") or attr_name == "schemaLocation":
                    schema_locations.append(attr_value)

            if schema_locations:
                metadata["schema_locations"] = schema_locations

        except Exception as e:
            logger.warning(f"Error extracting document metadata: {str(e)}")

        # Cache the metadata
        if self.enable_caching:
            self.text_cache.set(cache_key, metadata)

        return metadata

    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse an XML document into structured elements with direct lxml support."""
        content = doc_content["content"]
        source_id = doc_content["id"]  # Should already be a fully qualified path
        metadata = doc_content.get("metadata", {}).copy()  # Make a copy to avoid modifying original

        # Generate document cache key
        if self.enable_caching:
            content_hash = self._generate_hash(content)
            doc_cache_key = f"{source_id}_{content_hash}"

            # Check document cache
            cached_doc = self.document_cache.get(doc_cache_key)
            if cached_doc is not None:
                logger.info(f"Document cache hit for {source_id}")
                return cached_doc

        logger.info(f"Document cache miss for {source_id}")

        # Generate document ID if not present
        doc_id = metadata.get("doc_id", self._generate_id("doc_"))

        # Create document record
        document = {
            "doc_id": doc_id,
            "doc_type": "xml",
            "source": source_id,
            "metadata": self._extract_document_metadata(content, metadata),
            "content_hash": doc_content.get("content_hash", self._generate_hash(content))
        }

        # Use the helper method to get lxml root
        lxml_root = self._get_or_create_lxml_root(content)

        # Create root element
        elements = [self._create_root_element(doc_id, source_id)]
        root_id = elements[0]["element_id"]

        # Extract namespace information from root
        namespaces = {}
        for prefix, uri in lxml_root.nsmap.items():
            if prefix is None:
                namespaces["default"] = uri
            else:
                namespaces[prefix] = uri

        # Process document structure using the revised approach
        parsed_elements, relationships = self._process_document_structure(
            lxml_root, doc_id, root_id, source_id, namespaces
        )
        elements.extend(parsed_elements)

        # Extract links from the document
        links = self._extract_xml_links_direct(lxml_root, elements, namespaces)

        # Create result
        result = {
            "document": document,
            "elements": elements,
            "links": links,
            "relationships": relationships
        }

        # Add performance metrics if enabled
        if self.enable_performance_monitoring:
            self.performance_stats["parse_count"] += 1
            result["performance"] = self.get_performance_stats()

        # Cache the document
        if self.enable_caching:
            self.document_cache.set(doc_cache_key, result)

        return result

    def clear_caches(self):
        """Clear all caches."""
        self.document_cache.clear()
        self.tree_cache.clear()
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
            "tree_cache": {
                "size": len(self.tree_cache.cache),
                "max_size": self.tree_cache.max_size,
                "ttl": self.tree_cache.ttl
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

            # Check if source exists and is a file
            if not os.path.exists(source) or not os.path.isfile(source):
                return False

            # Check if element type is one we handle
            if element_type not in ["root", "xml_element", "xml_text", "xml_list", "xml_object"]:
                return False

            # Check file extension for XML
            _, ext = os.path.splitext(source.lower())
            return ext in ['.xml', '.xsd', '.rdf', '.rss', '.svg', '.wsdl', '.xslt']

        except (json.JSONDecodeError, TypeError):
            return False

    @ttl_cache(maxsize=256, ttl=3600)
    def _is_identity_element(self, element_name: str) -> bool:
        """
        Determines if an element likely represents an identity or entity.
        Uses caching for improved performance.

        Args:
            element_name: The name of the XML element

        Returns:
            True if it appears to be an identity element, False otherwise
        """
        # Use natural language processing principles to identify likely entity elements
        # Check if it's a common entity/identity type
        common_entities = [
            # Places
            "country", "city", "state", "province", "location", "address", "region",
            # People and organizations
            "person", "author", "publisher", "company", "organization", "corporation", "vendor",
            "owner", "creator", "editor", "manager", "developer", "provider", "customer",
            # Identifiers
            "name", "title", "label", "id", "identifier", "category", "type", "class",
            # Descriptors
            "genre", "style", "format", "model", "brand", "version"
        ]

        # Simple text matching approach
        element_lower = element_name.lower()

        # Check if it's in our list of common entities
        if element_lower in common_entities:
            return True
        # Check for possessive forms that suggest identity (e.g., author's, company's)
        elif element_lower.endswith("'s"):
            base_word = element_lower[:-2]
            return base_word in common_entities
        # Advanced: Check for compound words containing entity terms
        # E.g., "productName", "bookAuthor", "companyTitle"
        else:
            return any(entity in element_lower and entity != element_lower for entity in common_entities)
