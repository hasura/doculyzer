"""
XML document parser module for the document pointer system.

This module parses XML documents into structured elements and provides
semantic textual representations of the data.
"""

import json
import logging
import os
import re
import datetime
import enum
from typing import Dict, Any, Optional, List, Union, Tuple

from bs4 import BeautifulSoup

from .base import DocumentParser

logger = logging.getLogger(__name__)


class TemporalType(enum.Enum):
    """Enumeration for different types of temporal data."""
    NONE = 0       # Not a temporal string
    DATE = 1       # Date only (no time component)
    TIME = 2       # Time only (no date component)
    DATETIME = 3   # Combined date and time


def detect_temporal_type(input_string: str) -> TemporalType:
    """
    Detect if a string represents a date, time, datetime, or none of these.

    Args:
        input_string: String to analyze

    Returns:
        TemporalType enum indicating the type of temporal data
    """
    try:
        # Import dateutil parser
        from dateutil import parser

        # Check if it's an obviously non-temporal string
        if not input_string or not isinstance(input_string, str):
            return TemporalType.NONE

        # If the string is very long or has many words, it's probably not a date/time
        if len(input_string) > 50 or len(input_string.split()) > 8:
            return TemporalType.NONE

        # Check if it's a time-only string (no date component)
        time_patterns = [
            r'^\d{1,2}:\d{2}(:\d{2})?(\s*[ap]\.?m\.?)?$',  # 3:45pm, 15:30, 3:45:30 PM
            r'^\d{1,2}\s*[ap]\.?m\.?$',                    # 3pm, 11 a.m.
            r'^([01]?\d|2[0-3])([.:][0-5]\d)?([.:][0-5]\d)?$',  # 0500, 13.45, 22:30:15
            r'^(noon|midnight)$'                            # noon, midnight
        ]

        for pattern in time_patterns:
            if re.match(pattern, input_string.lower().strip()):
                return TemporalType.TIME

        # Try to parse with dateutil
        try:
            result = parser.parse(input_string, fuzzy=False)

            # Check if it has a non-default time component
            # Default time is usually 00:00:00
            has_time = (result.hour != 0 or result.minute != 0 or result.second != 0 or
                       'am' in input_string.lower() or 'pm' in input_string.lower() or
                       ':' in input_string)

            # If the input string contains typical time separators (:) or indicators (am/pm)
            # even if the parsed time is 00:00:00, consider it a datetime
            if has_time:
                return TemporalType.DATETIME
            else:
                return TemporalType.DATE

        except (parser.ParserError, ValueError):
            # If dateutil couldn't parse it, it's likely not a date/time
            return TemporalType.NONE

    except Exception as e:
        logger.warning(f"Error in detect_temporal_type: {str(e)}")
        return TemporalType.NONE


def create_semantic_time_expression(time_obj):
    """
    Convert a time object into a rich semantic natural language expression.

    Args:
        time_obj: A datetime object containing time information

    Returns:
        A natural language representation of the time with rich semantic context
    """
    try:
        # Get hour, minute, second
        hour = time_obj.hour
        minute = time_obj.minute
        second = time_obj.second
        microsecond = time_obj.microsecond

        # Determine AM/PM
        am_pm = "AM" if hour < 12 else "PM"

        # Convert to 12-hour format
        hour_12 = hour % 12
        if hour_12 == 0:
            hour_12 = 12

        # Time of day label
        if 4 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        # Determine quarter of hour
        quarter_labels = {0: "o'clock", 15: "quarter past", 30: "half past", 45: "quarter to"}

        # Default time description
        time_desc = f"{hour_12}:{minute:02d} {am_pm}"

        # Explicit minute description
        minute_desc = f", at minute {minute}" if minute != 0 else ""

        # Check for special times
        if minute in quarter_labels and second == 0:
            if minute == 45:
                next_hour = (hour_12 % 12) + 1
                if next_hour == 0:
                    next_hour = 12
                time_desc = f"{quarter_labels[minute]} {next_hour} {am_pm}"
            else:
                time_desc = f"{quarter_labels[minute]} {hour_12} {am_pm}"

        # Create full semantic expression
        result = f"at {time_desc} in the {time_of_day}{minute_desc}"

        # Add seconds if non-zero
        if second > 0 or microsecond > 0:
            if microsecond > 0:
                result += f", at second {second}.{microsecond//1000:03d}"
            else:
                result += f", at second {second}"

        return result

    except Exception as e:
        logger.warning(f"Error converting time to semantic expression: {str(e)}")
        return str(time_obj)  # Return string representation on error


def create_semantic_date_expression(date_str: str) -> str:
    """
    Convert a date string into a rich semantic natural language expression.

    Args:
        date_str: A string representing a date in various possible formats

    Returns:
        A natural language representation of the date with rich semantic context
    """
    try:
        # Import dateutil parser
        from dateutil import parser

        # Parse the date string using dateutil's flexible parser
        parsed_date = parser.parse(date_str)

        # Check if this is a datetime with significant time component
        has_time = False
        if hasattr(parsed_date, 'hour') and hasattr(parsed_date, 'minute'):
            if parsed_date.hour != 0 or parsed_date.minute != 0 or parsed_date.second != 0:
                has_time = True

        # If this has a significant time component, use the datetime formatter
        if has_time:
            return create_semantic_date_time_expression(date_str)

        # Get month name, day, and year
        month_name = parsed_date.strftime("%B")
        day = parsed_date.day
        year = parsed_date.year

        # Calculate week of month (approximate)
        week_of_month = (day - 1) // 7 + 1

        # Convert week number to ordinal word
        week_ordinals = ["first", "second", "third", "fourth", "fifth"]
        if 1 <= week_of_month <= 5:
            week_ordinal = week_ordinals[week_of_month - 1]
        else:
            week_ordinal = f"{week_of_month}th"  # Fallback if calculation is off

        # Calculate day of week
        day_of_week = parsed_date.strftime("%A")

        # Calculate quarter and convert to ordinal word
        quarter_num = (parsed_date.month - 1) // 3 + 1
        quarter_ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth"}
        quarter_ordinal = quarter_ordinals.get(quarter_num, f"{quarter_num}th")

        # Calculate decade as ordinal within century
        decade_in_century = (year % 100) // 10 + 1

        # Convert decade to ordinal word
        decade_ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth",
                          5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
                          9: "ninth", 10: "tenth"}
        decade_ordinal = decade_ordinals.get(decade_in_century, f"{decade_in_century}th")

        # Calculate century
        century = (year // 100) + 1

        # Format century as ordinal
        century_ordinals = {1: "1st", 2: "2nd", 3: "3rd"}
        century_ordinal = century_ordinals.get(century, f"{century}th")

        # Format as more descriptive natural language
        return f"the month of {month_name} ({quarter_ordinal} quarter), in the {week_ordinal} week, on {day_of_week} day {day}, in the year {year}, during the {decade_ordinal} decade of the {century_ordinal} century"

    except Exception as e:
        logger.warning(f"Error converting date to semantic expression: {str(e)}")
        return date_str  # Return original on any error


def create_semantic_date_time_expression(dt_str):
    """
    Convert a datetime string into a rich semantic natural language expression
    that includes both date and time information.

    Args:
        dt_str: A string representing a datetime

    Returns:
        A natural language representation with rich semantic context
    """
    try:
        # Import dateutil parser
        from dateutil import parser

        # Parse the datetime string
        parsed_dt = parser.parse(dt_str)

        # Generate date part
        date_part = create_semantic_date_expression(dt_str)

        # Generate time part
        time_part = create_semantic_time_expression(parsed_dt)

        # Combine them
        return f"{date_part}, {time_part}"

    except Exception as e:
        logger.warning(f"Error converting datetime to semantic expression: {str(e)}")
        return dt_str  # Return original on error


class XmlParser(DocumentParser):
    """Parser for XML documents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the XML parser."""
        super().__init__(config)
        # Configuration options
        self.config = config or {}
        self.max_content_preview = self.config.get("max_content_preview", 100)
        self.extract_attributes = self.config.get("extract_attributes", True)
        self.flatten_namespaces = self.config.get("flatten_namespaces", True)
        self.treat_namespaces_as_elements = self.config.get("treat_namespaces_as_elements", False)
        self.extract_namespace_declarations = self.config.get("extract_namespace_declarations", True)
        self.parser_features = self.config.get("parser_features", "xml")  # Use "xml" as the BeautifulSoup parser
        self.xpath_support = self.config.get("xpath_support", False)

        # Initialize lxml for XPath if requested and available
        self.lxml_available = False
        if self.xpath_support:
            try:
                # noinspection PyPackageRequirements
                import lxml.etree
                self.lxml_available = True
                logger.info("lxml is available - XPath support enabled")
            except ImportError:
                logger.warning("lxml not available. Install with 'pip install lxml' to enable XPath support")

    def _is_identity_element(self, element_name: str) -> bool:
        """
        Determines if an element likely represents an identity or entity.

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
        if element_lower.endswith("'s"):
            base_word = element_lower[:-2]
            if base_word in common_entities:
                return True

        # Advanced: Check for compound words containing entity terms
        # E.g., "productName", "bookAuthor", "companyTitle"
        for entity in common_entities:
            if entity in element_lower and entity != element_lower:
                return True

        # Default to False for unknown elements
        return False

    def _is_container_element(self, element_name: str) -> bool:
        """
        Determines if an element likely represents a container or collection.

        Args:
            element_name: The name of the XML element

        Returns:
            True if it appears to be a container element, False otherwise
        """
        # Check for plural endings (most common signal)
        element_lower = element_name.lower()

        # Common plural endings in English
        if (element_lower.endswith('s') and not element_lower.endswith('ss') and
            not element_lower.endswith('us') and not element_lower.endswith('is')):
            return True

        # Check for common container words
        container_words = [
            "list", "array", "collection", "set", "group", "series", "catalog",
            "index", "directory", "table", "map", "dictionary", "container",
            "package", "bundle", "batch", "items", "entries", "records",
            "results", "data"
        ]

        if element_lower in container_words:
            return True

        # Check for compound words with container terms
        for container in container_words:
            # Look for patterns like "userList", "productArray", etc.
            if container in element_lower and container != element_lower:
                return True

        # Check for container-implying prefixes
        collection_prefixes = ["all", "each", "every", "many", "multi"]
        for prefix in collection_prefixes:
            if element_lower.startswith(prefix) and len(element_lower) > len(prefix):
                # Check if the character after the prefix is uppercase (camelCase)
                if len(element_name) > len(prefix) and element_name[len(prefix)].isupper():
                    return True

        return False

    def _get_path(self, element):
        """Get the path for an element from our path map."""
        if hasattr(self, '_path_map'):
            return self._path_map.get(id(element))
        return None

    def _resolve_element_text(self, location_data: Dict[str, Any], source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve the plain text representation of an XML element.

        Args:
            location_data: Content location data
            source_content: Optional preloaded source content

        Returns:
            Plain text representation of the element
        """
        source = location_data.get("source", "")
        element_type = location_data.get("type", "")
        path = location_data.get("path", "")

        logger.debug(f"RESOLVING ELEMENT TEXT: source={source}, type={element_type}, path={path}")

        # Load content if not provided
        content = source_content
        if content is None:
            if os.path.exists(source):
                try:
                    with open(source, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # Try different encodings
                    try:
                        with open(source, 'rb') as f:
                            content = f.read()
                            content = content.decode('latin1')
                    except UnicodeDecodeError:
                        logger.error(f"Failed to decode content from {source}")
                        return "Binary content (cannot be displayed as text)"
            else:
                logger.error(f"Source file not found: {source}")
                return f"Source file not found: {source}"

        # Ensure content is string (not bytes)
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    content = content.decode('latin1')
                except UnicodeDecodeError:
                    logger.error(f"Failed to decode content bytes from {source}")
                    return "Binary content (cannot be displayed as text)"

        # Parse XML
        soup = BeautifulSoup(content, self.parser_features)
        logger.debug(f"SOUP CREATED: parser={self.parser_features}, root tag={soup.find().name if soup.find() else 'None'}")

        # Add paths to elements if they don't exist
        self._add_paths(soup)
        logger.debug("Added XPath-like paths to elements")

        # Check that paths were correctly added to a sample of elements
        for tag in list(soup.find_all())[:5]:  # Check first 5 elements
            path = self._get_path(tag)
            logger.debug(f"Element {tag.name} path: {path}")

        # For XPath-style paths, try to directly search and find the element
        element = None
        text_content = ""

        if path.startswith('//'):
            # Handle XPath path by direct searching
            parts = path.split('/')
            parts = [p for p in parts if p]  # Remove empty parts

            # For a path like //books/book[6]/copies, extract these components:
            if len(parts) >= 1:
                # Get the target element (the last part before any text() node)
                target_name = parts[-1]
                if "text()" in target_name:
                    # This is a text node reference - get the parent instead
                    if len(parts) >= 2:
                        target_name = parts[-2]
                    else:
                        target_name = None

                # Check if it has an index like book[6]
                target_idx = None
                if target_name and '[' in target_name and ']' in target_name:
                    idx_str = target_name.split('[')[1].split(']')[0]
                    if idx_str.isdigit():
                        target_idx = int(idx_str) - 1  # Convert to 0-based index
                    target_name = target_name.split('[')[0]

                logger.debug(f"Target element: {target_name}, index: {target_idx}")

                # Find all matching target elements
                if target_name:
                    targets = soup.find_all(target_name)
                    logger.debug(f"Found {len(targets)} {target_name} elements")

                    # If we have a specific index and it's valid, use that element
                    if target_idx is not None and 0 <= target_idx < len(targets):
                        element = targets[target_idx]
                        logger.debug(f"Selected {target_name} at index {target_idx+1}")
                    elif targets:
                        # Otherwise use the first element (default behavior)
                        element = targets[0]
                        logger.debug(f"Selected first {target_name} element")

                # If this is a text node reference, extract the text
                if element and "text()" in parts[-1]:
                    # Just get the text content of the element
                    text_content = element.get_text().strip()
                    logger.debug(f"Extracted text from element: '{text_content}'")

            # If no element or text found yet, try to find by tag name as a fallback
            if not element and not text_content:
                # Try direct tag search - use the last part of the path
                last_part = parts[-1]
                element_name = last_part.split('[')[0] if '[' in last_part else last_part

                # If it's a text node reference, use the parent
                if element_name == "text()":
                    element_name = parts[-2].split('[')[0] if len(parts) > 1 else None

                if element_name:
                    targets = soup.find_all(element_name)
                    if targets:
                        element = targets[0]
                        logger.debug(f"Fallback: found element by name {element_name}")
        else:
            # Standard path lookup using our path map
            for tag in soup.find_all():
                if self._get_path(tag) == path:
                    element = tag
                    logger.debug(f"Found element with exact path match: {path}")
                    break

        # Handle the case where we have plain text content already
        if text_content:
            logger.debug(f"Using previously extracted text content: '{text_content}'")

            # Determine element name from path
            element_name = ""
            for part in reversed(path.split("/")):
                if part and not part.startswith("text("):
                    # Get the base name without any index
                    element_name = part.split('[')[0] if '[' in part else part
                    break

            logger.debug(f"Determined element name from path: {element_name}")

            # Format as appropriate for the element type
            is_identity_element = self._is_identity_element(element_name)
            is_container_element = self._is_container_element(element_name)

            if is_identity_element:
                return f"{element_name} is \"{text_content}\""
            elif is_container_element:
                return f"{element_name} contains \"{text_content}\""
            else:
                return f"{element_name} is \"{text_content}\""

        # Get element name from path - simplified for non-text cases
        element_name = path.split('/')[-1] if '/' in path else path
        if '[' in element_name:
            element_name = element_name.split('[')[0]
        if element_name.startswith("text()"):
            parts = path.split('/')
            element_name = parts[-2].split('[')[0] if len(parts) > 1 else "text"

        logger.debug(f"Element name from path: {element_name}")

        # Handle element types - very simplified to the core functionality
        if element_type == "xml_text":
            # For text nodes, just return the text
            if element:
                text = element.get_text().strip()
                logger.debug(f"Returning text from text node: '{text}'")
                return text
            return ""

        elif element_type == "xml_element":
            if element:
                # Extract text content using get_text
                text_content = element.get_text().strip()
                logger.debug(f"Text content: '{text_content}'")

                # Apply semantic formatting based on element name
                is_identity_element = self._is_identity_element(element_name)
                is_container_element = self._is_container_element(element_name)

                if text_content:
                    if is_identity_element:
                        result = f"{element_name} is \"{text_content}\""
                        logger.debug(f"Formatted as identity: {result}")
                        return result
                    elif is_container_element:
                        result = f"{element_name} contains \"{text_content}\""
                        logger.debug(f"Formatted as container: {result}")
                        return result
                    else:
                        result = f"{element_name} is \"{text_content}\""
                        logger.debug(f"Formatted as generic: {result}")
                        return result
                else:
                    return element_name
            else:
                return element_name

        # Default fallback just returns element name from path
        return element_name

    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse an XML document into structured elements."""
        content = doc_content["content"]
        source_id = doc_content["id"]  # Should already be a fully qualified path
        metadata = doc_content.get("metadata", {}).copy()  # Make a copy to avoid modifying original

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

        # Create root element
        elements = [self._create_root_element(doc_id, source_id)]
        root_id = elements[0]["element_id"]

        # Parse XML
        soup = BeautifulSoup(content, self.parser_features)

        # Add XPath-like paths to elements for better location tracking
        self._add_paths(soup)

        # Parse XML elements
        parsed_elements, relationships = self._parse_document(soup, doc_id, root_id, source_id)
        elements.extend(parsed_elements)

        # Extract links from XML (e.g., xlink:href attributes, etc.)
        links = self._extract_xml_links(soup, elements)

        # Return the parsed document with extracted links and relationships
        return {
            "document": document,
            "elements": elements,
            "links": links,
            "relationships": relationships
        }

    def _extract_document_metadata(self, content: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from XML document.

        Args:
            content: XML content
            base_metadata: Base metadata from content source

        Returns:
            Dictionary of document metadata
        """
        metadata = base_metadata.copy()

        try:
            # Parse XML
            soup = BeautifulSoup(content, self.parser_features)

            # Extract root element name
            root = soup.find()
            if root:
                metadata["root_element"] = root.name

                # Extract namespace declarations
                if self.extract_namespace_declarations:
                    namespaces = {}
                    for attr_name, attr_value in root.attrs.items():
                        if attr_name.startswith("xmlns:"):
                            prefix = attr_name[6:]  # Remove "xmlns:" prefix
                            namespaces[prefix] = attr_value
                        elif attr_name == "xmlns":
                            namespaces["default"] = attr_value

                    if namespaces:
                        metadata["namespaces"] = namespaces

            # Extract document type declaration if available
            if hasattr(soup, 'doctype') and soup.doctype:
                metadata["doctype"] = {
                    "name": soup.doctype.name,
                    "public_id": soup.doctype.publicId,
                    "system_id": soup.doctype.systemId
                }

            # Basic document statistics
            metadata["element_count"] = len(soup.find_all())

            # Extract XML processing instructions
            if soup.find_all(
                    text=lambda text: isinstance(text, str) and text.strip().startswith("<?") and text.strip().endswith(
                        "?>")):
                proc_instructions = [
                    pi.strip() for pi in soup.find_all(
                        text=lambda text: isinstance(text, str) and text.strip().startswith(
                            "<?") and text.strip().endswith("?>"))
                ]
                if proc_instructions:
                    metadata["processing_instructions"] = proc_instructions

            # Try to detect schema or DTD information
            schema_locations = []
            if root and hasattr(root, 'attrs'):
                for attr_name, attr_value in root.attrs.items():
                    if attr_name.endswith(":schemaLocation") or attr_name == "schemaLocation":
                        schema_locations.append(attr_value)

                if schema_locations:
                    metadata["schema_locations"] = schema_locations

        except Exception as e:
            logger.warning(f"Error extracting document metadata: {str(e)}")

        return metadata

    def _parse_document(self, soup, doc_id: str, parent_id: str, source_id: str) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Parse the entire XML document.

        Args:
            soup: BeautifulSoup object
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier

        Returns:
            Tuple of (elements, relationships)
        """
        elements = []
        relationships = []

        # Create a map to track element IDs
        element_id_map = {}

        # Start with the root element
        root_tag = soup.find()
        if root_tag:
            # Create the document root element
            xml_root_id = self._generate_id("xml_root_")
            xml_root_element = {
                "element_id": xml_root_id,
                "doc_id": doc_id,
                "element_type": "xml_element",
                "parent_id": parent_id,
                "content_preview": f"<{root_tag.name}>",
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "xml_element",
                    "path": "/"
                }),
                "content_hash": self._generate_hash(str(root_tag)),
                "metadata": {
                    "element_name": root_tag.name,
                    "has_attributes": bool(root_tag.attrs),
                    "attributes": root_tag.attrs if self.extract_attributes else {},
                    "path": "/",
                    "text": root_tag.string.strip() if root_tag.string else ""
                }
            }
            elements.append(xml_root_element)
            element_id_map[root_tag] = xml_root_id

            # Create relationship between document root and XML root
            relationship = {
                "source_id": parent_id,
                "target_id": xml_root_id,
                "relationship_type": "contains"
            }
            relationships.append(relationship)

            # Process child elements recursively
            self._process_element_children(root_tag, doc_id, xml_root_id, source_id, elements, relationships,
                                           element_id_map)

        return elements, relationships

    def _process_element_children(self, parent_tag, doc_id, parent_id, source_id, elements, relationships,
                                  element_id_map, parent_path="/"):
        """
        Process all children of an XML element.

        Args:
            parent_tag: Parent XML element
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier
            elements: List to add elements to
            relationships: List to add relationships to
            element_id_map: Map of BeautifulSoup tags to element IDs
            parent_path: XPath-like path to parent element
        """
        # Track text node positions for each parent
        text_node_position = 0

        for i, child in enumerate(parent_tag.children):
            # Skip all non-element nodes unless they are text nodes with content
            if not hasattr(child, 'name') or child.name is None:
                # Only process text nodes with actual content
                if isinstance(child, str) and child.strip():
                    # This is a text node with content
                    text_node_position += 1
                    text_id = self._generate_id("text_")
                    text_content = child.strip()
                    text_preview = text_content[:self.max_content_preview] + (
                        "..." if len(text_content) > self.max_content_preview else "")

                    text_element = {
                        "element_id": text_id,
                        "doc_id": doc_id,
                        "element_type": "xml_text",
                        "parent_id": parent_id,
                        "content_preview": text_preview,
                        "content_location": json.dumps({
                            "source": source_id,
                            "type": "xml_text",
                            "path": f"{parent_path}/text()[{text_node_position}]"
                        }),
                        "content_hash": self._generate_hash(text_content),
                        "metadata": {
                            "parent_element": parent_tag.name,
                            "path": f"{parent_path}/text()[{text_node_position}]",
                            "index": i,
                            "text": text_content
                        }
                    }
                    elements.append(text_element)

                    # Create relationship
                    relationship = {
                        "source_id": parent_id,
                        "target_id": text_id,
                        "relationship_type": "contains_text"
                    }
                    relationships.append(relationship)
                # Skip all non-element nodes (including whitespace)
                continue

            # Skip comment nodes
            if child.name == 'comment':
                continue

            # Skip processing instructions for now
            if isinstance(child, str) and child.strip().startswith("<?") and child.strip().endswith("?>"):
                continue

            # Process element
            element_path = f"{parent_path}/{child.name}"

            # If there are multiple siblings with the same name, add index
            siblings = [s for s in parent_tag.find_all(child.name, recursive=False)]
            if len(siblings) > 1:
                # Find position of this child among siblings with same name
                # This is safe now since we've verified child has a name attribute
                pos = siblings.index(child) + 1
                element_path = f"{parent_path}/{child.name}[{pos}]"

            # Generate element ID
            element_id = self._generate_id(f"xml_elem_{child.name}_")

            # Get content preview
            if child.string and child.string.strip():
                text_content = child.string.strip()
                content_preview = f"<{child.name}>{text_content[:self.max_content_preview]}" + (
                    "..." if len(text_content) > self.max_content_preview else "") + f"</{child.name}>"
            else:
                content_preview = f"<{child.name}>" + ("..." if len(child.contents) > 0 else "") + f"</{child.name}>"

            # Create element with attributes
            element = {
                "element_id": element_id,
                "doc_id": doc_id,
                "element_type": "xml_element",
                "parent_id": parent_id,
                "content_preview": content_preview,
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "xml_element",
                    "path": element_path
                }),
                "content_hash": self._generate_hash(str(child)),
                "metadata": {
                    "element_name": child.name,
                    "has_attributes": bool(child.attrs),
                    "attributes": child.attrs if self.extract_attributes else {},
                    "path": element_path,
                    "text": child.string.strip() if child.string else ""
                }
            }

            elements.append(element)
            element_id_map[child] = element_id

            # Create relationship
            relationship = {
                "source_id": parent_id,
                "target_id": element_id,
                "relationship_type": "contains"
            }
            relationships.append(relationship)

            # Process child's children recursively
            self._process_element_children(child, doc_id, element_id, source_id, elements, relationships,
                                           element_id_map, element_path)

    def _add_paths(self, soup, current_path="/", parent=None):
        """
        Add XPath-like paths to XML elements.

        Args:
            soup: BeautifulSoup object
            current_path: Current XPath
            parent: Parent element
        """
        # Create a dictionary to store paths, using id(element) as key
        if not hasattr(self, '_path_map'):
            self._path_map = {}
            logger.debug(f"Creating new path map dictionary")

        logger.debug(f"Adding paths starting with path={current_path}, parent={parent.name if parent else 'None'}")

        if not parent:
            # Start with root element
            root = soup.find()
            if root:
                # Store path in our map using element's id as key
                self._path_map[id(root)] = "/"
                logger.debug(f"Set root path to / for element {root.name}")
                self._add_paths(soup, "/", root)
            return

        # Process child elements
        tag_counts = {}
        for child in parent.children:
            if hasattr(child, 'name') and child.name:
                # Count siblings with same name
                tag_counts[child.name] = tag_counts.get(child.name, 0) + 1
                logger.debug(f"Processing child: {child.name}, count: {tag_counts[child.name]}")

                # Create path
                if tag_counts[child.name] > 1:
                    child_path = f"{current_path}/{child.name}[{tag_counts[child.name]}]"
                else:
                    count_same_tags = len(parent.find_all(child.name, recursive=False))
                    if count_same_tags > 1:
                        child_path = f"{current_path}/{child.name}[1]"
                    else:
                        child_path = f"{current_path}/{child.name}"

                # Store path in our map
                self._path_map[id(child)] = child_path
                logger.debug(f"Set path to {child_path} for element {child.name}")

                # Recurse to child's children
                self._add_paths(soup, child_path, child)

    @staticmethod
    def _extract_xml_links(soup, elements) -> List[Dict[str, Any]]:
        """
        Extract links from XML document.

        Args:
            soup: BeautifulSoup object
            elements: List of parsed elements

        Returns:
            List of extracted links
        """
        links = []

        # Create a map of element paths to element IDs
        element_map = {}
        for element in elements:
            if element.get("element_type") in ["xml_element", "xml_text"]:
                content_location = element.get("content_location", "{}")
                try:
                    location = json.loads(content_location)
                    path = location.get("path", "")
                    element_map[path] = element.get("element_id")
                except (json.JSONDecodeError, TypeError):
                    pass

        # Look for various link-like attributes
        link_attrs = [
            "href", "xlink:href", "src", "uri", "url", "link",
            "reference", "ref", "target", "to", "from"
        ]

        # Find all elements with link attributes
        for tag in soup.find_all():
            if not hasattr(tag, '_path'):
                continue

            element_path = tag._path
            element_id = element_map.get(element_path)

            if not element_id:
                continue

            # Check for link attributes
            for attr in link_attrs:
                if attr in tag.attrs:
                    target = tag.attrs[attr]
                    if target and isinstance(target, str):
                        # Determine link type
                        link_type = "xml_attribute"
                        if attr == "xlink:href":
                            link_type = "xlink"
                        elif attr in ["href", "src"]:
                            link_type = attr

                        links.append({
                            "source_id": element_id,
                            "link_text": f"{attr}='{target}'",
                            "link_target": target,
                            "link_type": link_type
                        })

        return links

    def _resolve_element_content(self, location_data: Dict[str, Any],
                                 source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve content for specific XML element types, returning valid XML.

        Args:
            location_data: Content location data
            source_content: Optional preloaded source content

        Returns:
            Resolved content as properly formatted XML
        """
        source = location_data.get("source", "")
        element_type = location_data.get("type", "")
        path = location_data.get("path", "")

        logger.debug(f"RESOLVING ELEMENT CONTENT: source={source}, type={element_type}, path={path}")

        # Load content if not provided
        content = source_content
        if content is None:
            if os.path.exists(source):
                try:
                    with open(source, 'r', encoding='utf-8') as f:
                        content = f.read()
                        logger.debug(f"Successfully read content from file with utf-8 encoding")
                except UnicodeDecodeError:
                    # Try binary mode if text mode fails
                    try:
                        with open(source, 'rb') as f:
                            content = f.read()
                            try:
                                content = content.decode('utf-8')
                                logger.debug(f"Successfully decoded binary content with utf-8")
                            except UnicodeDecodeError:
                                try:
                                    content = content.decode('latin1')
                                    logger.debug(f"Successfully decoded binary content with latin1")
                                except UnicodeDecodeError:
                                    logger.error(f"Failed to decode binary content from {source}")
                                    raise ValueError("Cannot decode binary content as XML")
                    except Exception as e:
                        logger.error(f"Error reading binary content: {str(e)}")
                        raise ValueError(f"Error reading content: {str(e)}")
            else:
                logger.error(f"Source file not found: {source}")
                raise ValueError(f"Source file not found: {source}")

        # Ensure content is string (not bytes)
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8')
                logger.debug(f"Decoded bytes content with utf-8")
            except UnicodeDecodeError:
                try:
                    content = content.decode('latin1')
                    logger.debug(f"Decoded bytes content with latin1")
                except UnicodeDecodeError:
                    logger.error(f"Failed to decode content bytes")
                    raise ValueError("Cannot decode binary content as XML")

        # Parse XML
        soup = BeautifulSoup(content, self.parser_features)
        logger.debug(f"Created BeautifulSoup object using {self.parser_features} parser")

        # Add paths to elements
        self._add_paths(soup)
        logger.debug(f"Added XPath-like paths to elements")

        # If root path or no path, return full document
        if not path or path == "/":
            logger.debug(f"Root path requested")
            if element_type == "root":
                logger.debug(f"Returning full document")
                return str(soup)
            else:
                # Get the root element
                root = soup.find()
                if root:
                    logger.debug(f"Returning root element: {root.name}")
                    return str(root)
                else:
                    logger.warning(f"No root element found")
                    return "<empty/>"

        # Check if path uses // which needs special handling
        found_element = None
        if path.startswith('//'):
            logger.debug(f"Path starts with '//', using more flexible path matching")
            # For paths that use '//' we need a more flexible approach
            # Extract the element name from the path
            if '[' in path:
                # Handle paths with indexes like "//books/book[15]/summary"
                parts = path.split('/')
                element_name = parts[-1]
                if '[' in element_name:
                    element_name = element_name.split('[')[0]
                    logger.debug(f"Extracted element name '{element_name}' from path with index")
            else:
                # Simple path like "//books/summary"
                element_name = path.split('/')[-1]
                logger.debug(f"Extracted element name '{element_name}' from simple path")

            # Try to find all elements with this name
            candidates = soup.find_all(element_name)
            logger.debug(f"Found {len(candidates)} candidate elements with name '{element_name}'")

            # See if any match our path - handling both exact matches and partial paths
            for tag in candidates:
                if hasattr(tag, '_path'):
                    logger.debug(f"Checking candidate with path: {tag._path}")

                    # Check for exact match or if the path ends with what we're looking for
                    if tag._path == path or tag._path.endswith(path[1:]) or path.endswith(tag._path):
                        found_element = tag
                        logger.debug(f"FOUND MATCH: {tag._path}")
                        break
        else:
            # Regular path search
            logger.debug(f"Searching for exact path match: {path}")
            for tag in soup.find_all():
                if hasattr(tag, '_path') and tag._path == path:
                    found_element = tag
                    logger.debug(f"Found exact path match")
                    break

        # Process the found element based on type
        if found_element:
            logger.debug(f"Processing found element of type: {element_type}")
            if element_type == "xml_text":
                # For text nodes, wrap in a simple container to maintain XML validity
                text = found_element.string.strip() if found_element.string else ""
                logger.debug(f"Returning text node content: '{text}'")
                return f"<text>{text}</text>"
            elif element_type == "xml_element":
                # Return the element as XML
                result = str(found_element)
                logger.debug(f"Returning element content: '{result[:100]}...' (truncated)")
                return result
            else:
                # Default to returning the element
                result = str(found_element)
                logger.debug(f"Returning default element content: '{result[:100]}...' (truncated)")
                return result
        else:
            # If no element found through direct path search, try using element name
            logger.warning(f"Element not found with path: {path}")

            # Extract element name from path
            element_name = path.split('/')[-1]
            if '[' in element_name:
                element_name = element_name.split('[')[0]

            logger.debug(f"Trying to find by element name: {element_name}")

            # Look for elements with this name
            elements = soup.find_all(element_name)
            if elements:
                logger.debug(f"Found {len(elements)} elements with name '{element_name}'")
                # Just use the first one as a fallback
                result = str(elements[0])
                logger.warning(f"Using first matching element as fallback")
                return result

            # If element not found at all, return an error indicator as valid XML
            logger.error(f"No elements found with name '{element_name}' for path '{path}'")
            return f'<error path="{path}">Element not found</error>'

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
            if element_type not in ["root", "xml_element", "xml_text"]:
                return False

            # Check file extension for XML
            _, ext = os.path.splitext(source.lower())
            return ext in ['.xml', '.xsd', '.rdf', '.rss', '.svg', '.wsdl', '.xslt']

        except (json.JSONDecodeError, TypeError):
            return False

    def _extract_links(self, content: str, element_id: str) -> List[Dict[str, Any]]:
        """
        Extract links from XML content.

        Args:
            content: XML content
            element_id: ID of the element containing the links

        Returns:
            List of extracted links
        """
        links = []

        # Parse XML with BeautifulSoup
        soup = BeautifulSoup(content, self.parser_features)

        # Look for various link-like attributes
        link_attrs = [
            "href", "xlink:href", "src", "uri", "url", "link",
            "reference", "ref", "target", "to", "from"
        ]

        # Find all elements with link attributes
        for tag in soup.find_all():
            for attr in link_attrs:
                if attr in tag.attrs:
                    target = tag.attrs[attr]
                    if target and isinstance(target, str):
                        # Determine link type
                        link_type = "xml_attribute"
                        if attr == "xlink:href":
                            link_type = "xlink"
                        elif attr in ["href", "src"]:
                            link_type = attr

                        links.append({
                            "source_id": element_id,
                            "link_text": f"{attr}='{target}'",
                            "link_target": target,
                            "link_type": link_type
                        })

        return links
