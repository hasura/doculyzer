"""
XML document parser module for the document pointer system.

This module parses XML documents into structured elements.
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List, Union, Tuple

from bs4 import BeautifulSoup

from .base import DocumentParser

logger = logging.getLogger(__name__)


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
                        return "Binary content (cannot be displayed as text)"
            else:
                return f"Source file not found: {source}"

        # Ensure content is string (not bytes)
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    content = content.decode('latin1')
                except UnicodeDecodeError:
                    return "Binary content (cannot be displayed as text)"

        # Parse XML
        soup = BeautifulSoup(content, self.parser_features)

        # Add paths to elements if they don't exist
        if not hasattr(soup.find(), '_path'):
            self._add_paths(soup)

        # Find element by path
        element = None
        if path:
            for tag in soup.find_all():
                if hasattr(tag, '_path') and tag['_path'] == path:
                    element = tag
                    break

        # Get element name from path
        element_name = path.split('/')[-1] if '/' in path else path
        # Remove index if present in path (e.g., "item[1]" -> "item")
        if '[' in element_name:
            element_name = element_name.split('[')[0]

        # Handle element types
        if element_type == "xml_text":
            # For text nodes, just return the text
            if element and element.string:
                return element.string.strip()
            return ""

        elif element_type == "xml_element":
            if element:
                # For elements with just text content, return "element_name: text_content"
                if element.string and element.string.strip():
                    return f"{element_name}: {element.string.strip()}"

                # For elements with attributes, include them in a readable format
                elif element.attrs and self.extract_attributes:
                    attrs_text = ", ".join(f"{k}='{v}'" for k, v in element.attrs.items()
                                           if not k.startswith('_'))
                    return f"{element_name} ({attrs_text})"

                # For elements with children, just return the name
                else:
                    return element_name
            else:
                # Element not found
                return element_name if element_name else path

        # Default case
        if element:
            # Try to represent the element meaningfully
            if element.string and element.string.strip():
                return f"{element_name}: {element.string.strip()}"
            elif element.attrs and self.extract_attributes:
                attrs_text = ", ".join(f"{k}='{v}'" for k, v in element.attrs.items()
                                       if not k.startswith('_'))
                return f"{element_name} ({attrs_text})"
            else:
                return element_name

        # Fallback if element not found
        return element_name if element_name else path

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
        for i, child in enumerate(parent_tag.children):
            # Skip NavigableString unless it contains non-whitespace
            if not hasattr(child, 'name'):
                if isinstance(child, str) and child.strip():
                    # This is a text node with content
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
                            "path": f"{parent_path}/text()[{i + 1}]"
                        }),
                        "content_hash": self._generate_hash(text_content),
                        "metadata": {
                            "parent_element": parent_tag.name,
                            "path": f"{parent_path}/text()[{i + 1}]",
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
        if not parent:
            # Start with root element
            root = soup.find()
            if root:
                root['_path'] = "/"
                self._add_paths(soup, "/", root)
            return

        # Process child elements
        tag_counts = {}
        for child in parent.children:
            if hasattr(child, 'name') and child.name:
                # Count siblings with same name
                tag_counts[child.name] = tag_counts.get(child.name, 0) + 1

                # Create path
                if tag_counts[child.name] > 1:
                    child_path = f"{current_path}/{child.name}[{tag_counts[child.name]}]"
                else:
                    count_same_tags = len(parent.find_all(child.name, recursive=False))
                    if count_same_tags > 1:
                        child_path = f"{current_path}/{child.name}[1]"
                    else:
                        child_path = f"{current_path}/{child.name}"

                child['_path'] = child_path

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

            element_path = tag['_path']
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

        # Load content if not provided
        content = source_content
        if content is None:
            if os.path.exists(source):
                try:
                    with open(source, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # Try binary mode if text mode fails
                    with open(source, 'rb') as f:
                        content = f.read()
                        try:
                            content = content.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                content = content.decode('latin1')
                            except UnicodeDecodeError:
                                raise ValueError("Cannot decode binary content as XML")
            else:
                raise ValueError(f"Source file not found: {source}")

        # Ensure content is string (not bytes)
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    content = content.decode('latin1')
                except UnicodeDecodeError:
                    raise ValueError("Cannot decode binary content as XML")

        # Parse XML
        soup = BeautifulSoup(content, self.parser_features)

        # Add paths to elements
        self._add_paths(soup)

        # If root path or no path, return full document
        if not path or path == "/":
            if element_type == "root":
                return str(soup)
            else:
                # Get the root element
                root = soup.find()
                if root:
                    return str(root)
                else:
                    return "<empty/>"

        # Find element by path
        for tag in soup.find_all():
            if hasattr(tag, '_path') and tag['_path'] == path:
                if element_type == "xml_text":
                    # For text nodes, wrap in a simple container to maintain XML validity
                    text = tag.string.strip() if tag.string else ""
                    return f"<text>{text}</text>"
                elif element_type == "xml_element":
                    # Return the element as XML
                    return str(tag)
                else:
                    # Default to returning the element
                    return str(tag)

        # If element not found, return an error indicator as valid XML
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
