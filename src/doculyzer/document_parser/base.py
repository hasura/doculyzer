"""
Base document parser module for the document pointer system.

This module defines the abstract base class for all document parsers.
"""

import hashlib
import json
import os
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union


class DocumentParser(ABC):
    """Abstract base class for document parsers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document parser.

        Args:
            config: Parser configuration
        """
        self.config = config or {}

    @abstractmethod
    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a document into structured elements.

        Args:
            doc_content: Document content and metadata

        Returns:
            Dictionary with document metadata, elements, relationships, and extracted links
        """
        pass

    def resolve_content(self, content_location: str, source_content: Optional[Union[str, bytes]] = None) -> str:
        """
        Resolve content based on location data.

        Args:
            content_location: JSON-formatted content location pointer
            source_content: Optional preloaded source content

        Returns:
            Resolved content as string

        Raises:
            ValueError: If content cannot be resolved
        """
        try:
            # Parse location data
            location_data = json.loads(content_location)

            # Get source and element type
            source = location_data.get("source", "")
            element_type = location_data.get("type", "")

            # Handle root type specially
            if element_type == "root":
                if source_content is not None:
                    # Return source content directly if provided
                    if isinstance(source_content, bytes):
                        # Try to decode bytes to string
                        try:
                            return source_content.decode('utf-8')
                        except UnicodeDecodeError:
                            return f"(Binary content of {len(source_content)} bytes)"
                    return str(source_content)

                # Otherwise, load content from source
                if os.path.exists(source):
                    try:
                        with open(source, 'r', encoding='utf-8') as f:
                            return f.read()
                    except UnicodeDecodeError:
                        # Try to read as binary if text fails
                        with open(source, 'rb') as f:
                            binary_content = f.read()
                            return f"(Binary content of {len(binary_content)} bytes)"

            # For non-root types, delegate to specific parser implementations
            return self._resolve_element_content(location_data, source_content)

        except Exception as e:
            raise ValueError(f"Error resolving content: {str(e)}")

    @abstractmethod
    def _resolve_element_content(self, location_data: Dict[str, Any],
                                 source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve content for specific element types.

        Args:
            location_data: Content location data
            source_content: Optional preloaded source content

        Returns:
            Resolved content string
        """
        pass

    @abstractmethod
    def _resolve_element_text(self, location_data: Dict[str, Any],
                              source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve content for specific element types.

        Args:
            location_data: Content location data
            source_content: Optional preloaded source content

        Returns:
            Resolved content string
        """
        pass

    @abstractmethod
    def supports_location(self, content_location: str) -> bool:
        """
        Check if this parser supports resolving the given location.

        Args:
            content_location: Content location pointer

        Returns:
            True if supported, False otherwise
        """
        pass

    @staticmethod
    def get_document_binary(content_location: str) -> bytes:
        """
        Get the containing document as a binary blob.

        Args:
            content_location: Content location pointer

        Returns:
            Document binary content

        Raises:
            ValueError: If document cannot be retrieved
        """
        try:
            location_data = json.loads(content_location)
            source = location_data.get("source", "")

            # Check if source is a file path
            if os.path.exists(source):
                with open(source, 'rb') as f:
                    return f.read()

            raise ValueError(f"Cannot retrieve binary content for {source}")

        except Exception as e:
            raise ValueError(f"Error retrieving binary content: {str(e)}")

    @staticmethod
    def _generate_id(prefix: str = "") -> str:
        """
        Generate a unique ID for a document or element.

        Args:
            prefix: Optional prefix for the ID

        Returns:
            Unique ID string
        """
        return f"{prefix}{uuid.uuid4()}"

    @staticmethod
    def _generate_hash(content: str) -> str:
        """
        Generate a hash of content for change detection.

        Args:
            content: Text content

        Returns:
            MD5 hash of content
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _create_root_element(self, doc_id: str, source_id: str) -> Dict[str, Any]:
        """
        Create a root element for the document.

        Args:
            doc_id: Document ID
            source_id: Source identifier (now a fully qualified path)

        Returns:
            Root element dictionary
        """
        element_id = self._generate_id("root_")

        # Ensure source_id is an absolute path if it's a file path
        if os.path.exists(source_id):
            source_id = os.path.abspath(source_id)

        return {
            "element_id": element_id,
            "doc_id": doc_id,
            "element_type": "root",
            "parent_id": None,
            "content_preview": "",
            "content_location": json.dumps({
                "source": source_id,  # Now using fully qualified path
                "type": "root"
            }),
            "content_hash": self._generate_hash(""),
            "metadata": {
                "full_path": source_id  # Store the fully qualified path in metadata too
            }
        }

    def _extract_links(self, content: str, element_id: str) -> List[Dict[str, Any]]:
        """
        Base method for extracting links from content.
        Should be overridden by specific parsers.

        Args:
            content: Text content
            element_id: ID of the element containing the links

        Returns:
            List of extracted link dictionaries
        """
        return []

    @staticmethod
    def _ensure_absolute_path(path: str, base_path: Optional[str] = None) -> str:
        """
        Ensure a path is absolute.

        Args:
            path: Path to convert
            base_path: Optional base path to use if path is relative

        Returns:
            Absolute path
        """
        if os.path.isabs(path):
            return path

        if base_path:
            # Use the base path to resolve relative paths
            return os.path.abspath(os.path.join(base_path, path))

        # If no base path and path is relative, use current directory
        return os.path.abspath(path)
