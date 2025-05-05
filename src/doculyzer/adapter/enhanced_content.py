"""
Enhanced content resolver module for the document pointer system.

This module provides an enhanced content resolver that integrates content source adapters
with document parsers to support a wide variety of content sources and formats.
"""

import json
import logging
import os
from typing import Dict, Any, Optional

from .base import ContentResolver
from .base import ContentSourceAdapter
from ..document_parser.base import DocumentParser
from ..document_parser.document_type_detector import DocumentTypeDetector

logger = logging.getLogger(__name__)


class EnhancedContentResolver(ContentResolver):
    """
    Enhanced content resolver that integrates content source adapters with document parsers.
    """

    def __init__(self, adapters: Optional[Dict[str, ContentSourceAdapter]] = None,
                 parsers: Optional[Dict[str, DocumentParser]] = None,
                 path_mappings: Optional[Dict[str, str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced content resolver.

        Args:
            adapters: Dictionary of content source adapters by type
            parsers: Dictionary of document parsers by type
            path_mappings: Optional path mappings for source remapping
            config: Optional configuration dictionary
        """
        self.adapters = adapters or {}
        self.parsers = parsers or {}
        self.path_mappings = path_mappings or {}
        self.config = config or {}
        self.cache = {}  # Cache for resolved content

        # Default cache settings
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.max_cache_size = self.config.get("max_cache_size", 1000)
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour in seconds

    def resolve_content(self, content_location: str, text: bool = True) -> str:
        """
        Resolve content using appropriate adapter and parser.

        Args:
            content_location: JSON-formatted content location pointer
            text: Returns the semantic text representation if True else the native content representation

        Returns:
            Resolved content as string
        """
        # Skip if empty
        if not content_location:
            return ""

        # Check cache if enabled
        if self.cache_enabled:
            cache_key = self._get_cache_key(content_location, text)
            if cache_key in self.cache:
                return self.cache[cache_key]

        try:
            # Parse location
            location_data = json.loads(content_location)

            # Apply path remappings
            location_data = self._apply_path_mappings(location_data)

            # Determine source type from URI
            source_type = self._get_source_type(location_data.get("source", ""))

            # Get appropriate adapter
            adapter = self.adapters.get(source_type)
            if not adapter:
                logger.warning(f"No adapter available for source type: {source_type}")
                return f"No adapter available for source type: {source_type}"

            # Check if this adapter supports this location
            if not adapter.supports_location(location_data):
                logger.warning(f"Adapter {source_type} does not support location: {location_data.get('source', '')}")
                return f"Adapter {source_type} does not support this location"

            # Handle root element type specially for efficiency
            element_type = location_data.get("type", "")
            if element_type == "root":
                # Get content directly from adapter
                content_info = adapter.get_content(location_data)
                content = content_info.get("content", "")

                # Convert to string if binary
                if isinstance(content, bytes):
                    try:
                        content = content.decode('utf-8')
                    except UnicodeDecodeError:
                        content = f"(Binary content of {len(content)} bytes)"

                # Cache if enabled
                if self.cache_enabled:
                    self._update_cache(cache_key, content)

                return content

            # For specific element types, get content and pass to appropriate parser
            content_info = adapter.get_content(location_data)
            content = content_info.get("content", "")
            metadata = content_info.get("metadata", {})
            content_type = DocumentTypeDetector.detect_from_content(content, metadata)

            # Get appropriate parser
            parser = self.parsers.get(content_type)
            if not parser:
                logger.warning(f"No parser available for content type: {content_type}")
                fallback_parser = self.parsers.get('text')
                if fallback_parser:
                    parser = fallback_parser
                else:
                    return f"No parser available for content type: {content_type}"

            # Use parser to resolve specific element
            if text:
                resolved_content = parser._resolve_element_text(location_data, content)
            else:
                resolved_content = parser._resolve_element_content(location_data, content)

            # Cache result if enabled
            if self.cache_enabled:
                self._update_cache(cache_key, resolved_content)

            return resolved_content

        except Exception as e:
            logger.error(f"Error resolving content: {str(e)}")
            return f"Error resolving content: {str(e)}"

    def supports_location(self, content_location: str) -> bool:
        """
        Check if any adapter and parser support the location.

        Args:
            content_location: Content location pointer

        Returns:
            True if supported, False otherwise
        """
        try:
            # Parse location
            location_data = json.loads(content_location)

            # Apply path remappings
            location_data = self._apply_path_mappings(location_data)

            # Determine source type
            source_type = self._get_source_type(location_data.get("source", ""))

            # Check if adapter supports this location
            adapter = self.adapters.get(source_type)
            if not adapter or not adapter.supports_location(location_data):
                return False

            # For non-root elements, check if parser support is needed
            element_type = location_data.get("type", "")
            if element_type != "root":
                # Get content type
                content_info = adapter.get_content(location_data)
                content_type = content_info.get("content_type", "")

                # If no content type provided, detect it
                if not content_type:
                    metadata = content_info.get("metadata", {})
                    content = content_info.get("content", "")
                    content_type = DocumentTypeDetector.detect_from_content(content, metadata)

                # Check if parser supports this content type
                parser = self.parsers.get(content_type)
                if not parser:
                    fallback_parser = self.parsers.get('text')
                    if not fallback_parser:
                        return False

            return True

        except Exception as e:
            logger.debug(f"Error checking support for location: {str(e)}")
            return False

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
        try:
            # Parse location
            location_data = json.loads(content_location)

            # Apply path remappings
            location_data = self._apply_path_mappings(location_data)

            # Determine source type
            source_type = self._get_source_type(location_data.get("source", ""))

            # Get appropriate adapter
            adapter = self.adapters.get(source_type)
            if not adapter:
                raise ValueError(f"No adapter available for source type: {source_type}")

            # Get binary content directly from adapter
            return adapter.get_binary_content(location_data)

        except Exception as e:
            raise ValueError(f"Error getting document binary: {str(e)}")

    def get_metadata(self, content_location: str) -> Dict[str, Any]:
        """
        Get metadata about the content without retrieving the full content.

        Args:
            content_location: Content location pointer

        Returns:
            Dictionary with metadata

        Raises:
            ValueError: If metadata cannot be retrieved
        """
        try:
            # Parse location
            location_data = json.loads(content_location)

            # Apply path remappings
            location_data = self._apply_path_mappings(location_data)

            # Determine source type
            source_type = self._get_source_type(location_data.get("source", ""))

            # Get appropriate adapter
            adapter = self.adapters.get(source_type)
            if not adapter:
                raise ValueError(f"No adapter available for source type: {source_type}")

            # Get metadata from adapter
            return adapter.get_metadata(location_data)

        except Exception as e:
            raise ValueError(f"Error getting metadata: {str(e)}")

    def add_path_mapping(self, original_prefix: str, new_prefix: str) -> None:
        """
        Add a mapping to remap file paths.

        Args:
            original_prefix: Original path prefix to be replaced
            new_prefix: New path prefix to use
        """
        self.path_mappings[original_prefix] = new_prefix

        # Clear cache when mappings change
        if self.cache_enabled:
            self.clear_cache()

    def clear_cache(self) -> None:
        """Clear the content cache."""
        self.cache = {}

    def cleanup(self) -> None:
        """
        Clean up resources used by adapters.

        This should be called when the resolver is no longer needed.
        """
        for adapter in self.adapters.values():
            try:
                adapter.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up adapter: {str(e)}")

    def _apply_path_mappings(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply path mappings to location data.

        Args:
            location_data: Original location data

        Returns:
            Updated location data with mappings applied
        """
        # Create a copy to avoid modifying the original
        result = location_data.copy()

        # Apply mappings to source
        source = result.get('source', '')
        for original, new in self.path_mappings.items():
            if source.startswith(original):
                result['source'] = source.replace(original, new, 1)
                break

        return result

    @staticmethod
    def _get_source_type(source: str) -> str:
        """
        Determine source type from source URI.

        Args:
            source: Source URI

        Returns:
            Source type string
        """
        if os.path.exists(source):
            return 'file'
        elif source.startswith('db://'):
            return 'database'
        elif source.startswith('s3://'):
            return 's3'
        elif source.startswith(('http://', 'https://')):
            return 'web'
        elif source.startswith('confluence://'):
            return 'confluence'
        elif source.startswith('jira://'):
            return 'jira'
        elif source.startswith('mongodb://'):
            return 'mongodb'
        elif source.startswith('servicenow://'):
            return 'servicenow'
        else:
            # Default to file for unknown sources
            return 'file'

    def _update_cache(self, key: str, value: str) -> None:
        """
        Update the cache with new value.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Add to cache
        self.cache[key] = value

        # Trim cache if it exceeds max size
        if len(self.cache) > self.max_cache_size:
            # Simple LRU implementation: remove oldest entries
            excess = len(self.cache) - self.max_cache_size
            keys_to_remove = list(self.cache.keys())[:excess]
            for k in keys_to_remove:
                del self.cache[k]

    @staticmethod
    def _get_cache_key(content_location: str, text: bool) -> str:
        """
        Generate cache key for content location.

        Args:
            content_location: Content location string

        Returns:
            Cache key
        """
        return content_location + ("-text" if text else "-native")
