"""
Base content source adapter for the document pointer system.

This module defines the abstract base class for all content source adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ContentSourceAdapter(ABC):
    """Abstract base class for content source adapters."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the content source adapter.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    def get_content(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get content from the source.

        Args:
            location_data: Location data specifying what to retrieve

        Returns:
            Dictionary with:
                - content: The actual content (string or bytes)
                - content_type: Type of content (markdown, html, docx, etc.)
                - metadata: Additional metadata about the content

        Raises:
            ValueError: If content cannot be retrieved
        """
        pass

    @abstractmethod
    def supports_location(self, location_data: Dict[str, Any]) -> bool:
        """
        Check if this adapter supports the location.

        Args:
            location_data: Location data

        Returns:
            True if supported, False otherwise
        """
        pass

    @abstractmethod
    def get_binary_content(self, location_data: Dict[str, Any]) -> bytes:
        """
        Get the content as binary data.

        Args:
            location_data: Location data

        Returns:
            Binary content

        Raises:
            ValueError: If content cannot be retrieved as binary
        """
        pass

    def get_metadata(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get metadata about the content without retrieving the full content.

        Args:
            location_data: Location data

        Returns:
            Dictionary with metadata

        Raises:
            ValueError: If metadata cannot be retrieved
        """
        # Default implementation: Get content and return just the metadata
        content_info = self.get_content(location_data)
        return content_info.get('metadata', {})

    def resolve_uri(self, uri: str) -> Dict[str, Any]:
        """
        Parse a URI into location data.

        Args:
            uri: Content URI string

        Returns:
            Dictionary with parsed location data

        Raises:
            ValueError: If URI cannot be parsed
        """
        # Default implementation: Return URI as source
        return {"source": uri}

    def cleanup(self):
        """
        Clean up resources used by this adapter.

        This method should be called when the adapter is no longer needed
        to release any held resources like database connections, file handles, etc.
        """
        pass

    @staticmethod
    def validate_location(location_data: Dict[str, Any]) -> bool:
        """
        Validate location data.

        Args:
            location_data: Location data to validate

        Returns:
            True if valid, False otherwise
        """
        # Basic validation: Check if source is present
        return 'source' in location_data

    @staticmethod
    def apply_path_mapping(location_data: Dict[str, Any], mappings: Dict[str, str]) -> Dict[str, Any]:
        """
        Apply path mappings to location data.

        Args:
            location_data: Original location data
            mappings: Dictionary of path mappings (original -> new)

        Returns:
            Updated location data with mappings applied
        """
        # Create a copy to avoid modifying the original
        result = location_data.copy()

        # Apply mappings to source
        source = result.get('source', '')
        for original, new in mappings.items():
            if source.startswith(original):
                result['source'] = source.replace(original, new, 1)
                break

        return result


"""
Base content resolver module for the document pointer system.

This module defines the abstract base class for all content resolvers.
"""


class ContentResolver(ABC):
    """Abstract base class for content resolvers."""

    @abstractmethod
    def resolve_content(self, content_location: str, text: bool) -> str:
        """
        Resolve a content pointer to actual content.

        Args:
            content_location: Content location pointer in JSON format
            text: if True returns the semantic text representation, else the native content representation

        Returns:
            Resolved content as string
        """
        pass

    @abstractmethod
    def supports_location(self, content_location: str) -> bool:
        """
        Check if this resolver supports a content location.

        Args:
            content_location: Content location pointer

        Returns:
            True if supported, False otherwise
        """
        pass

    @abstractmethod
    def get_document_binary(self, content_location: str) -> bytes:
        """
        Get the containing document as a binary blob.

        Args:
            content_location: Content location pointer

        Returns:
            Document binary content

        Raises:
            ValueError: If document cannot be retrieved
        """
        pass

    def add_path_mapping(self, original_prefix: str, new_prefix: str) -> None:
        """
        Add a mapping to remap paths in content locations.

        Args:
            original_prefix: Original path prefix to be replaced
            new_prefix: New path prefix to use
        """
        pass

    def clear_cache(self) -> None:
        """
        Clear any cached content.
        """
        pass

    def cleanup(self) -> None:
        """
        Clean up resources used by this resolver.

        This should be called when the resolver is no longer needed.
        """
        pass

    def get_metadata(self, content_location: str) -> Dict[str, Any]:
        """
        Get metadata about content without retrieving full content.

        Args:
            content_location: Content location pointer

        Returns:
            Dictionary with metadata

        Raises:
            ValueError: If metadata cannot be retrieved
        """
        raise NotImplementedError("get_metadata not implemented")
