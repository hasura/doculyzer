"""
MongoDB Content Resolver implementation for the document pointer system.

This module resolves MongoDB content pointers to actual content.
"""
import json
import logging
from typing import Dict, Any

from .base import ContentResolver

logger = logging.getLogger(__name__)

# Try to import pymongo, but don't fail if not available
try:
    import pymongo
    from pymongo import MongoClient
    from bson import ObjectId, json_util

    PYMONGO_AVAILABLE = True
except ImportError:
    MongoClient = None
    ObjectId = None
    json_util = None
    PYMONGO_AVAILABLE = False
    logger.warning("pymongo not available. Install with 'pip install pymongo' to use MongoDB content resolver.")


class MongoDBContentResolver(ContentResolver):
    """Resolver for MongoDB content."""

    def __init__(self):
        """Initialize the MongoDB content resolver."""
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo is required for MongoDB content resolver")

        self.clients = {}  # Cache for MongoDB clients by connection string
        self.content_cache = {}  # Cache for retrieved content

    def resolve_content(self, content_location: str) -> str:
        """
        Resolve MongoDB content pointer to actual content.

        Args:
            content_location: JSON-formatted content location pointer

        Returns:
            Resolved content as string

        Raises:
            ValueError: If source is invalid or record not found
        """
        location_data = json.loads(content_location)

        source = location_data.get("source", "")
        if not source.startswith("mongodb://"):
            raise ValueError(f"Invalid MongoDB source: {source}")

        # Extract info from source identifier
        # Format: mongodb://database/collection/document_id[/field]
        parts = source.split('/')
        if len(parts) < 5:
            raise ValueError(f"Invalid MongoDB source format: {source}")

        connection_string = parts[0] + "//" + parts[2]  # mongodb://host:port
        database_name = parts[3]
        collection_name = parts[4]

        document_id = parts[5] if len(parts) > 5 else None
        field_path = '/'.join(parts[6:]) if len(parts) > 6 else None

        # Get MongoDB client
        client = self._get_client(connection_string)
        db = client[database_name]
        collection = db[collection_name]

        # Determine what part of the content to return based on element type
        element_type = location_data.get("type", "")

        # Check cache first
        cache_key = f"{source}:{element_type}"
        if cache_key in self.content_cache:
            logger.debug(f"Using cached content for: {cache_key}")
            return self.content_cache[cache_key]

        try:
            # Convert document_id if it looks like ObjectId
            if document_id and len(document_id) == 24 and all(c in '0123456789abcdefABCDEF' for c in document_id):
                try:
                    document_id = ObjectId(document_id)
                except Exception:
                    pass  # Keep as string if conversion fails

            # Query for the document
            document = collection.find_one({"_id": document_id})

            if not document:
                raise ValueError(f"Document not found: {document_id}")

            # Extract content based on element type and field path
            if element_type == "root" or not element_type:
                # Return full document content as JSON
                resolved_content = json.dumps(document, default=json_util.default, indent=2)
            elif element_type == "field" and field_path:
                # Extract specific field
                field_value = self._get_nested_field(document, field_path)

                if field_value is None:
                    raise ValueError(f"Field not found: {field_path}")

                # Convert field value to string
                if isinstance(field_value, (dict, list)):
                    resolved_content = json.dumps(field_value, default=json_util.default, indent=2)
                else:
                    resolved_content = str(field_value)
            else:
                # Default: return full document
                resolved_content = json.dumps(document, default=json_util.default, indent=2)

            # Cache the result
            self.content_cache[cache_key] = resolved_content

            return resolved_content

        except Exception as e:
            logger.error(f"Error resolving MongoDB content: {str(e)}")
            raise

    def supports_location(self, content_location: str) -> bool:
        """
        Check if this resolver supports the location.

        Args:
            content_location: Content location pointer

        Returns:
            True if supported, False otherwise
        """
        try:
            location_data = json.loads(content_location)
            source = location_data.get("source", "")
            # Source must be a MongoDB URI
            return source.startswith("mongodb://")
        except (json.JSONDecodeError, TypeError):
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
        # For MongoDB, we'll return the JSON content as bytes
        content = self.resolve_content(content_location)
        return content.encode('utf-8')

    def _get_client(self, connection_string: str) -> MongoClient:
        """
        Get or create a MongoDB client for the given connection string.

        Args:
            connection_string: MongoDB connection string

        Returns:
            MongoDB client
        """
        if connection_string in self.clients:
            return self.clients[connection_string]

        # Create a new client
        client = MongoClient(connection_string)

        # Test connection
        client.admin.command('ping')

        # Cache the client
        self.clients[connection_string] = client

        return client

    @staticmethod
    def _get_nested_field(document: Dict[str, Any], field_path: str) -> Any:
        """
        Get a nested field from a document using dot notation.

        Args:
            document: MongoDB document
            field_path: Field path using dot notation (e.g., 'user.profile.name')

        Returns:
            Field value or None if not found
        """
        # Handle array indexing and nested fields
        parts = field_path.replace('[', '.').replace(']', '').split('.')

        current = document
        for part in parts:
            if not part:  # Skip empty parts
                continue

            # Handle numeric indices for arrays
            if part.isdigit() and isinstance(current, list):
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            # Handle dictionary keys
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def __del__(self):
        """Clean up MongoDB connections on deletion."""
        for client in self.clients.values():
            try:
                client.close()
            except Exception as e:
                logger.warning(f"Error closing MongoDB connection: {str(e)}")
