"""
MongoDB adapter module for the document pointer system.

This module provides an adapter to retrieve content from MongoDB sources.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from .base import ContentSourceAdapter
from ..document_parser.document_type_detector import DocumentTypeDetector

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
    logger.warning("pymongo not available. Install with 'pip install pymongo' to use MongoDB adapter.")


class MongoDBAdapter(ContentSourceAdapter):
    """Adapter for MongoDB content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the MongoDB adapter."""
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo is required for MongoDB adapter")

        super().__init__(config)
        self.config = config or {}

        # MongoDB connection settings
        self.default_connection_string = self.config.get("connection_string")
        self.default_database = self.config.get("database")
        self.default_collection = self.config.get("collection")
        self.connection_timeout = self.config.get("connection_timeout", 30000)
        self.server_selection_timeout = self.config.get("server_selection_timeout", 30000)

        # Auth settings if not in connection string
        self.username = self.config.get("username")
        self.password = self.config.get("password")
        self.auth_source = self.config.get("auth_source", "admin")
        self.auth_mechanism = self.config.get("auth_mechanism", "SCRAM-SHA-256")

        # Cached connections
        self.clients = {}

        # Content cache
        self.content_cache = {}
        self.metadata_cache = {}

    def get_content(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get content from MongoDB.

        Args:
            location_data: Location data with MongoDB connection info

        Returns:
            Dictionary with content and metadata

        Raises:
            ValueError: If MongoDB document cannot be retrieved
        """
        # Parse location data
        source = location_data.get("source", "")
        if not source.startswith("mongodb://"):
            raise ValueError(f"Invalid MongoDB source: {source}")

        # Cache key using the full source string
        cache_key = source
        if cache_key in self.content_cache:
            return self.content_cache[cache_key]

        # Parse MongoDB URI and extract components
        parsed_data = self._parse_mongodb_uri(source)

        # Get client for this connection
        client = self._get_client(parsed_data["connection_string"])

        try:
            # Access database and collection
            db = client[parsed_data["database"]]
            collection = db[parsed_data["collection"]]

            # Prepare query for finding the document
            query = self._build_query(parsed_data)

            # Execute query
            document = collection.find_one(query)

            if not document:
                raise ValueError(f"Document not found for query: {query}")

            # Extract field if specified
            content, field_path = self._extract_field(document, parsed_data)

            # Determine content type
            if isinstance(content, dict) or isinstance(content, list):
                content_type = "json"
                str_content = json.dumps(content, default=json_util.default, indent=2)
            else:
                # Try to detect content type from the content itself
                str_content = str(content) if not isinstance(content, str) else content
                content_type = DocumentTypeDetector.detect_from_content(
                    str_content,
                    metadata={"field_path": field_path}
                )

            # Prepare metadata
            metadata = {
                "connection_string_masked": self._mask_connection_string(parsed_data["connection_string"]),
                "database": parsed_data["database"],
                "collection": parsed_data["collection"],
                "document_id": str(document.get("_id")),
                "document_size": len(json.dumps(document, default=json_util.default)),
                "field_path": field_path if field_path else None,
                "query": str(query),
                "timestamp": datetime.now().isoformat()
            }

            # Add field type information
            if field_path:
                field_type = type(content).__name__
                metadata["field_type"] = field_type

            # Create result
            result = {
                "content": str_content if not isinstance(content, (bytes, bytearray)) else content,
                "content_type": content_type,
                "metadata": metadata
            }

            # Cache the result
            self.content_cache[cache_key] = result
            self.metadata_cache[cache_key] = metadata

            return result

        except Exception as e:
            logger.error(f"Error retrieving MongoDB content: {str(e)}")
            raise ValueError(f"Error retrieving MongoDB content: {str(e)}")

    def supports_location(self, location_data: Dict[str, Any]) -> bool:
        """
        Check if this adapter supports the location.

        Args:
            location_data: Content location data

        Returns:
            True if supported, False otherwise
        """
        if not PYMONGO_AVAILABLE:
            return False

        source = location_data.get("source", "")
        # Source must be a MongoDB URI
        return source.startswith("mongodb://")

    def get_binary_content(self, location_data: Dict[str, Any]) -> bytes:
        """
        Get the MongoDB document as binary data.

        Args:
            location_data: Location data

        Returns:
            Binary content

        Raises:
            ValueError: If document cannot be retrieved as binary
        """
        # First try to get the content normally
        content_data = self.get_content(location_data)
        content = content_data.get("content")

        # If content is already binary, return it
        if isinstance(content, (bytes, bytearray)):
            return content

        # Otherwise, convert to JSON and then to bytes
        if isinstance(content, (dict, list)):
            return json.dumps(content, default=json_util.default).encode('utf-8')

        # Convert string content to bytes
        if isinstance(content, str):
            return content.encode('utf-8')

        # For other types, convert to string first
        return str(content).encode('utf-8')

    def get_metadata(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get metadata about the MongoDB document without retrieving the full content.

        Args:
            location_data: Location data

        Returns:
            Dictionary with metadata

        Raises:
            ValueError: If metadata cannot be retrieved
        """
        source = location_data.get("source", "")

        # Check cache first
        if source in self.metadata_cache:
            return self.metadata_cache[source]

        # Parse MongoDB URI and extract components
        parsed_data = self._parse_mongodb_uri(source)

        # Get client for this connection
        client = self._get_client(parsed_data["connection_string"])

        try:
            # Access database and collection
            db = client[parsed_data["database"]]
            collection = db[parsed_data["collection"]]

            # Prepare query for finding the document
            query = self._build_query(parsed_data)

            # Execute projection query to just get basic info without the full content
            projection = {"_id": 1}
            if parsed_data.get("field_path"):
                # Add field existence check
                field_parts = parsed_data["field_path"].split(".")
                field_to_check = field_parts[0]
                projection[field_to_check] = 1

            document = collection.find_one(query, projection)

            if not document:
                raise ValueError(f"Document not found for query: {query}")

            # Prepare metadata
            metadata = {
                "connection_string_masked": self._mask_connection_string(parsed_data["connection_string"]),
                "database": parsed_data["database"],
                "collection": parsed_data["collection"],
                "document_id": str(document.get("_id")),
                "field_path": parsed_data.get("field_path"),
                "query": str(query),
                "timestamp": datetime.now().isoformat()
            }

            # Get collection stats
            try:
                stats = db.command("collstats", parsed_data["collection"])
                metadata["collection_size"] = stats.get("size")
                metadata["document_count"] = stats.get("count")
                metadata["avg_document_size"] = stats.get("avgObjSize")
            except Exception:
                # If stats command fails, just continue without this info
                pass

            # Cache the metadata
            self.metadata_cache[source] = metadata

            return metadata

        except Exception as e:
            logger.error(f"Error retrieving MongoDB metadata: {str(e)}")
            raise ValueError(f"Error retrieving MongoDB metadata: {str(e)}")

    def resolve_uri(self, uri: str) -> Dict[str, Any]:
        """
        Parse a MongoDB URI into location data.

        Args:
            uri: MongoDB URI string

        Returns:
            Dictionary with parsed location data

        Raises:
            ValueError: If URI cannot be parsed
        """
        if not uri.startswith("mongodb://"):
            raise ValueError(f"Not a MongoDB URI: {uri}")

        # Parse URI into components
        parsed_data = self._parse_mongodb_uri(uri)

        # Build location data
        location_data = {
            "source": uri,
            "connection_string": parsed_data["connection_string"],
            "database": parsed_data["database"],
            "collection": parsed_data["collection"]
        }

        # Add document_id if available
        if "document_id" in parsed_data:
            location_data["document_id"] = parsed_data["document_id"]

        # Add field_path if available
        if "field_path" in parsed_data:
            location_data["field_path"] = parsed_data["field_path"]

        return location_data

    def cleanup(self):
        """
        Clean up MongoDB connections.

        This method should be called when the adapter is no longer needed.
        """
        for conn_id, client in self.clients.items():
            try:
                client.close()
                logger.debug(f"Closed MongoDB connection: {conn_id}")
            except Exception as e:
                logger.warning(f"Error closing MongoDB connection {conn_id}: {str(e)}")

        # Clear caches and connections
        self.clients = {}
        self.content_cache = {}
        self.metadata_cache = {}

    @staticmethod
    def _parse_mongodb_uri(uri: str) -> Dict[str, Any]:
        """
        Parse MongoDB URI into components.

        Expected format: mongodb://[username:password@]host[:port]/database/collection[/document_id][/field_path]

        Args:
            uri: MongoDB URI string

        Returns:
            Dictionary with parsed components
        """
        # Split off the connection string part
        if not uri.startswith("mongodb://"):
            raise ValueError(f"Invalid MongoDB URI: {uri}")

        # Find the first / after mongodb://
        base_uri_parts = uri.split("/")

        # We need at least mongodb://host/database/collection
        if len(base_uri_parts) < 5:
            raise ValueError(f"Invalid MongoDB URI format. Expected mongodb://host/database/collection: {uri}")

        # Extract connection string (mongodb://host)
        connection_string = f"{base_uri_parts[0]}//{base_uri_parts[2]}"

        # Extract database and collection
        database = base_uri_parts[3]
        collection = base_uri_parts[4]

        # Create result with essential parts
        result = {
            "connection_string": connection_string,
            "database": database,
            "collection": collection
        }

        # Check for document_id and field_path
        if len(base_uri_parts) > 5:
            result["document_id"] = base_uri_parts[5]

            # Check for field path
            if len(base_uri_parts) > 6:
                # Rejoin remaining parts as the field path
                result["field_path"] = "/".join(base_uri_parts[6:])

        return result

    def _get_client(self, connection_string: str) -> MongoClient:
        """
        Get or create MongoDB client for the given connection string.

        Args:
            connection_string: MongoDB connection string

        Returns:
            MongoDB client

        Raises:
            ValueError: If connection cannot be established
        """
        # Use a masked version as the cache key
        cache_key = self._mask_connection_string(connection_string)

        # Check if client exists in cache
        if cache_key in self.clients:
            return self.clients[cache_key]

        # Create client options
        client_options = {
            "connectTimeoutMS": self.connection_timeout,
            "serverSelectionTimeoutMS": self.server_selection_timeout
        }

        # Add authentication if needed and not in connection string
        if self.username and self.password and "@" not in connection_string:
            client_options["username"] = self.username
            client_options["password"] = self.password
            client_options["authSource"] = self.auth_source
            client_options["authMechanism"] = self.auth_mechanism

        try:
            # Create MongoDB client
            client = MongoClient(connection_string, **client_options)

            # Test connection with a ping
            client.admin.command("ping")

            # Cache client
            self.clients[cache_key] = client

            return client

        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise ValueError(f"Cannot connect to MongoDB: {str(e)}")

    @staticmethod
    def _build_query(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build MongoDB query from parsed URI data.

        Args:
            parsed_data: Parsed MongoDB URI data

        Returns:
            MongoDB query document
        """
        query = {}

        # Add document ID if specified
        if "document_id" in parsed_data:
            doc_id = parsed_data["document_id"]

            # Try to convert to ObjectId if possible
            if ObjectId.is_valid(doc_id):
                query["_id"] = ObjectId(doc_id)
            else:
                # Use as string ID
                query["_id"] = doc_id

        return query

    @staticmethod
    def _extract_field(document: Dict[str, Any], parsed_data: Dict[str, Any]) -> tuple:
        """
        Extract field from document using dot notation.

        Args:
            document: MongoDB document
            parsed_data: Parsed MongoDB URI data

        Returns:
            Tuple of (field_value, field_path)
        """
        field_path = parsed_data.get("field_path")

        if not field_path:
            # Return full document if no field specified
            return document, None

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
                    raise ValueError(f"Array index out of range: {part} in {field_path}")
            # Handle dictionary keys
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise ValueError(f"Field not found: {part} in {field_path}")

        return current, field_path

    @staticmethod
    def _mask_connection_string(connection_string: str) -> str:
        """
        Mask sensitive information in connection string for logging.

        Args:
            connection_string: MongoDB connection string

        Returns:
            Masked connection string
        """
        # Check if connection string contains credentials
        if "@" in connection_string:
            # Extract username:password part
            parts = connection_string.split("@")
            prefix = parts[0]
            suffix = "@".join(parts[1:])

            # Find where credentials are
            cred_parts = prefix.split("//")
            protocol = cred_parts[0] + "//"
            # credentials = cred_parts[1]

            # Mask credentials
            return f"{protocol}****:****@{suffix}"

        return connection_string
