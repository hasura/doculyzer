"""
Factory for creating content resolvers.
"""
import json
import logging
from typing import Dict, Any

from .base import ContentResolver
from .confluence import ConfluenceContentResolver
from .database import DatabaseContentResolver
from .file import FileContentResolver
from .jira import JiraContentResolver
from .s3 import S3ContentResolver
from .web import WebContentResolver
from .mongodb import MongoDBContentResolver
from ..config import Config

logger = logging.getLogger(__name__)


class GenericContentResolver(ContentResolver):
    """Generic content resolver that delegates to specific resolvers."""

    def __init__(self, content_sources: Dict[str, Any] = None):
        """
        Initialize the generic content resolver.

        Args:
            content_sources: Dictionary of content sources keyed by type
        """
        self.resolvers = []
        self.cache = {}  # Simple cache for resolved content
        self.path_mappings = {}  # Path mappings dictionary

        # Create specific resolvers
        self._register_resolvers(content_sources or {})

    def add_path_mapping(self, original_prefix: str, new_prefix: str) -> None:
        """
        Add a mapping to remap file paths.

        Args:
            original_prefix: Original path prefix to be replaced
            new_prefix: New path prefix to use
        """
        self.path_mappings[original_prefix] = new_prefix

    def resolve_content(self, content_location: str) -> str:
        """Resolve content using appropriate resolver."""
        # Skip if empty
        if not content_location:
            return ""

        # Check cache
        cache_key = self._get_cache_key(content_location)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Parse location
        try:
            location_data = json.loads(content_location)

            # Apply path remappings if source is a path
            source = location_data.get("source", "")
            for original, new in self.path_mappings.items():
                if source.startswith(original):
                    location_data["source"] = source.replace(original, new, 1)
                    # Update content_location with remapped path
                    content_location = json.dumps(location_data)
                    break
        except json.JSONDecodeError:
            logger.error(f"Invalid content location format: {content_location}")
            return ""

        # Find appropriate resolver
        for resolver in self.resolvers:
            if resolver.supports_location(content_location):
                try:
                    content = resolver.resolve_content(content_location)

                    # Cache result
                    self.cache[cache_key] = content

                    return content
                except Exception as e:
                    logger.error(f"Error resolving content: {str(e)}")
                    return f"Error resolving content: {str(e)}"

        logger.warning(f"No resolver found for content location: {content_location}")
        return ""

    def supports_location(self, content_location: str) -> bool:
        """Check if any resolver supports the location."""
        try:
            # Attempt to parse location
            location_data = json.loads(content_location)

            # Apply path remappings if source is a path
            source = location_data.get("source", "")
            for original, new in self.path_mappings.items():
                if source.startswith(original):
                    location_data["source"] = source.replace(original, new, 1)
                    # Update content_location with remapped path
                    content_location = json.dumps(location_data)
                    break

            # Check if any resolver supports it
            return any(resolver.supports_location(content_location) for resolver in self.resolvers)

        except json.JSONDecodeError:
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
            location_data = json.loads(content_location)

            # Apply path remappings if source is a path
            source = location_data.get("source", "")
            for original, new in self.path_mappings.items():
                if source.startswith(original):
                    location_data["source"] = source.replace(original, new, 1)
                    # Update content_location with remapped path
                    content_location = json.dumps(location_data)
                    break
        except json.JSONDecodeError:
            raise ValueError(f"Invalid content location format: {content_location}")

        # Find appropriate resolver
        for resolver in self.resolvers:
            if resolver.supports_location(content_location):
                try:
                    return resolver.get_document_binary(content_location)
                except Exception as e:
                    logger.error(f"Error getting document binary: {str(e)}")
                    raise ValueError(f"Error getting document binary: {str(e)}")

        logger.warning(f"No resolver found for content location: {content_location}")
        raise ValueError(f"No resolver found for content location: {content_location}")

    def add_resolver(self, resolver: ContentResolver) -> None:
        """
        Add a resolver to the resolver list.

        Args:
            resolver: ContentResolver instance
        """
        self.resolvers.append(resolver)

    def clear_cache(self) -> None:
        """Clear the content cache."""
        self.cache = {}

    def _register_resolvers(self, _content_sources: Dict[str, Any]) -> None:
        """
        Register specific resolvers based on available content sources.

        Args:
            _content_sources: Dictionary of content sources keyed by type
        """
        # File resolver is always available
        self.add_resolver(FileContentResolver())

    @staticmethod
    def _get_cache_key(content_location):
        return content_location


def create_content_resolver(config: Config) -> ContentResolver:
    """
    Create content resolver from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        ContentResolver instance
    """
    # Extract content sources from config
    content_sources = config.config.get('content_sources', [])

    # Process content sources into dictionary if it's a list
    sources_dict = {}
    if isinstance(content_sources, list):
        for source_config in content_sources:
            source_type = source_config.get("type")
            if source_type:
                sources_dict[source_type] = source_config
    else:
        sources_dict = content_sources

    # Create generic resolver
    resolver = GenericContentResolver(sources_dict)

    # Add path mappings if configured
    path_mappings = config.config.get("path_mappings", {})
    for original, new in path_mappings.items():
        resolver.add_path_mapping(original, new)

    # Add specific resolvers
    resolver.add_resolver(FileContentResolver())
    resolver.add_resolver(DatabaseContentResolver())
    resolver.add_resolver(WebContentResolver(sources_dict.get("web", {})))

    # Add Confluence resolver if needed
    if "confluence" in sources_dict or any(source.get("type") == "confluence" for source in sources_dict.values()):
        resolver.add_resolver(ConfluenceContentResolver())

    # Add JIRA resolver if needed
    if "jira" in sources_dict or any(source.get("type") == "jira" for source in sources_dict.values()):
        resolver.add_resolver(JiraContentResolver())

    # Add S3 resolver if needed
    if "s3" in sources_dict or any(source.get("type") == "s3" for source in sources_dict.values()):
        resolver.add_resolver(S3ContentResolver())

    # Add MongoDB resolver if needed
    if "mongodb" in sources_dict or any(source.get("type") == "mongodb" for source in sources_dict.values()):
        resolver.add_resolver(MongoDBContentResolver())

    return resolver
