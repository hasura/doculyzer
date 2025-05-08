"""
MongoDB Content Source for the document pointer system.

This module provides integration with MongoDB to ingest documents stored in collections.
"""

import json
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

# Import types for type checking only - these won't be imported at runtime
if TYPE_CHECKING:
    from pymongo import MongoClient
    from bson import ObjectId

    # Define type aliases for type checking
    MongoClientType = MongoClient
    ObjectIdType = ObjectId
else:
    # Runtime type aliases - use generic Python types
    MongoClientType = Any
    ObjectIdType = Any

from .base import ContentSource

logger = logging.getLogger(__name__)

# Define global flags for availability - these will be set at runtime
PYMONGO_AVAILABLE = False

# Try to import pymongo conditionally
try:
    import pymongo
    from pymongo import MongoClient
    from bson import ObjectId

    PYMONGO_AVAILABLE = True
except ImportError:
    logger.warning("pymongo not available. Install with 'pip install pymongo' to use MongoDB content source.")


class MongoDBContentSource(ContentSource):
    """Content source for MongoDB collections."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MongoDB content source.

        Args:
            config: Configuration dictionary containing MongoDB connection details
        """
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo is required for MongoDBContentSource but not available")

        super().__init__(config)

        # MongoDB connection details
        self.connection_string = config.get("connection_string", "mongodb://localhost:27017/")
        self.database_name = config.get("database_name", "")
        self.collection_name = config.get("collection_name", "")
        self.query = config.get("query", {})
        self.projection = config.get("projection", None)
        self.id_field = config.get("id_field", "_id")
        self.content_field = config.get("content_field", None)  # The field containing the main content
        self.timestamp_field = config.get("timestamp_field", "updated_at")  # For change detection
        self.limit = config.get("limit", 1000)
        self.sort_by = config.get("sort_by", [("_id", 1)])  # Default sort by _id ascending

        # Additional configuration
        self.include_patterns = config.get("include_patterns", [])
        self.exclude_patterns = config.get("exclude_patterns", [])
        self.follow_references = config.get("follow_references", False)
        self.reference_field = config.get("reference_field", None)
        self.max_link_depth = config.get("max_link_depth", 3)

        # Initialize MongoDB client
        self.client: Optional[MongoClientType] = None
        self.db = None
        self.collection = None

        # Initialize MongoDB connection
        if self.connection_string:
            try:
                self.client = MongoClient(self.connection_string)
                # Check connection
                self.client.admin.command('ping')
                # Get database and collection
                self.db = self.client[self.database_name]
                self.collection = self.db[self.collection_name]
                logger.debug(f"Successfully connected to MongoDB: {self.get_safe_connection_string()}")
            except Exception as e:
                logger.error(f"Error connecting to MongoDB: {str(e)}")
                raise

        # Cache for content
        self.content_cache = {}

    def get_safe_connection_string(self) -> str:
        """Return a safe version of the connection string with password masked."""
        if not self.connection_string:
            return "<no connection string>"

        try:
            parts = self.connection_string.split("://")
            if len(parts) != 2:
                return "<malformed connection string>"

            protocol = parts[0]
            connection_parts = parts[1].split("@")

            if len(connection_parts) == 2:
                # Connection string with authentication
                auth_parts = connection_parts[0].split(":")
                if len(auth_parts) == 2:
                    username = auth_parts[0]
                    masked_conn = f"{protocol}://{username}:****@{connection_parts[1]}"
                    return masked_conn

            # If we can't parse properly, return a generic masked string
            return f"{protocol}://*****"
        except Exception:
            return "<connection string parsing error>"

    def fetch_document(self, source_id: str) -> Dict[str, Any]:
        """
        Fetch document content from MongoDB.

        Args:
            source_id: Identifier for the document (usually the MongoDB _id)

        Returns:
            Dictionary containing document content and metadata
        """
        if not self.client:
            raise ValueError("MongoDB not configured")

        logger.debug(f"Fetching MongoDB document: {source_id}")

        try:
            # Extract MongoDB ID if source_id is a complex identifier
            mongo_id = self._extract_mongo_id(source_id)

            # Determine query to use based on ID field
            if self.id_field == "_id" and not isinstance(mongo_id, dict):
                try:
                    # Try to convert to ObjectId if it looks like one
                    if isinstance(mongo_id, str) and len(mongo_id) == 24 and all(
                            c in '0123456789abcdefABCDEF' for c in mongo_id):
                        query = {"_id": ObjectId(mongo_id)}
                    else:
                        query = {"_id": mongo_id}
                except Exception:
                    query = {self.id_field: mongo_id}
            else:
                query = {self.id_field: mongo_id}

            # Check cache first
            cache_key = str(query)
            if cache_key in self.content_cache:
                cache_entry = self.content_cache[cache_key]
                logger.debug(f"Using cached content for: {source_id}")
                return cache_entry

            # Get document from MongoDB
            document = self.collection.find_one(query, self.projection)

            if not document:
                raise ValueError(f"Document not found: {source_id}")

            # Format document ID for identification
            qualified_source = f"mongodb://{self.database_name}/{self.collection_name}/{self._get_doc_id_str(document)}"

            # Convert document to JSON string
            if self.content_field and self.content_field in document:
                content = document[self.content_field]
                # If content is not a string, convert to JSON
                if not isinstance(content, str):
                    content = json.dumps(content, default=self._json_serializer)
            else:
                # Use whole document as content
                content = json.dumps(document, default=self._json_serializer)

            # Generate metadata
            metadata = {
                "database": self.database_name,
                "collection": self.collection_name,
                "id_field": self.id_field,
                "id_value": self._get_doc_id_str(document),
                "timestamp_field": self.timestamp_field,
                "timestamp_value": self._get_timestamp(document),
                "fields": list(document.keys()),
                "content_field": self.content_field,
                "url": qualified_source
            }

            # Generate content hash for change detection
            content_hash = self.get_content_hash(content)

            # Create result
            result = {
                "id": qualified_source,
                "content": content,
                "metadata": metadata,
                "content_hash": content_hash,
                "content_type": "application/json"  # Specify content type as JSON
            }

            # Cache the content for faster access
            self.content_cache[cache_key] = result

            return result

        except ValueError:
            # Re-raise ValueError for not found
            raise
        except Exception as e:
            logger.error(f"Error fetching MongoDB document {source_id}: {str(e)}")
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List available documents in MongoDB.

        Returns:
            List of document identifiers and metadata
        """
        if not self.client:
            raise ValueError("MongoDB not configured")

        logger.debug(f"Listing MongoDB documents from {self.database_name}.{self.collection_name}")

        results = []
        try:
            # Execute query with projection, sort, and limit
            cursor = self.collection.find(
                self.query,
                projection=self.projection,
            ).sort(self.sort_by).limit(self.limit)

            for document in cursor:
                # Format document ID for identification
                qualified_source = f"mongodb://{self.database_name}/{self.collection_name}/{self._get_doc_id_str(document)}"

                # Create metadata
                metadata = {
                    "database": self.database_name,
                    "collection": self.collection_name,
                    "id_field": self.id_field,
                    "id_value": self._get_doc_id_str(document),
                    "timestamp_field": self.timestamp_field,
                    "timestamp_value": self._get_timestamp(document),
                    "fields": list(document.keys()),
                    "content_field": self.content_field,
                    "url": qualified_source
                }

                results.append({
                    "id": qualified_source,
                    "metadata": metadata
                })

            logger.info(f"Found {len(results)} MongoDB documents")
            return results

        except Exception as e:
            logger.error(f"Error listing MongoDB documents: {str(e)}")
            raise

    def has_changed(self, source_id: str, last_modified: Optional[float] = None) -> bool:
        """
        Check if a document has changed since last processing.

        Args:
            source_id: Identifier for the document
            last_modified: Timestamp of last known modification

        Returns:
            True if document has changed, False otherwise
        """
        if not self.client or not self.timestamp_field:
            # Can't determine changes without timestamp or connection
            return True

        logger.debug(f"Checking if MongoDB document has changed: {source_id}")

        try:
            # Extract MongoDB ID
            mongo_id = self._extract_mongo_id(source_id)

            # Determine query to use based on ID field
            if self.id_field == "_id" and not isinstance(mongo_id, dict):
                try:
                    # Try to convert to ObjectId if it looks like one
                    if isinstance(mongo_id, str) and len(mongo_id) == 24 and all(
                            c in '0123456789abcdefABCDEF' for c in mongo_id):
                        query = {"_id": ObjectId(mongo_id)}
                    else:
                        query = {"_id": mongo_id}
                except Exception:
                    query = {self.id_field: mongo_id}
            else:
                query = {self.id_field: mongo_id}

            # Check cache first
            cache_key = str(query)
            if cache_key in self.content_cache:
                cache_entry = self.content_cache[cache_key]
                cache_timestamp = cache_entry["metadata"].get("timestamp_value")

                if cache_timestamp and last_modified and cache_timestamp <= last_modified:
                    logger.debug(f"Document {source_id} unchanged according to cache")
                    return False

            # Get only the timestamp field
            projection = {self.timestamp_field: 1}
            document = self.collection.find_one(query, projection=projection)

            if not document:
                logger.debug(f"Document {source_id} not found")
                return False

            current_timestamp = self._get_timestamp(document)

            if current_timestamp and last_modified:
                changed = current_timestamp > last_modified
                logger.debug(f"Document {source_id} changed: {changed}")
                return changed

            # If no timestamp field or couldn't determine, consider it changed
            return True

        except Exception as e:
            logger.error(f"Error checking changes for {source_id}: {str(e)}")
            return True  # Assume changed if there's an error

    def follow_links(self, content: str, source_id: str, current_depth: int = 0,
                     global_visited_docs=None) -> List[Dict[str, Any]]:
        """
        Extract and follow references in MongoDB document.

        Args:
            content: Document content
            source_id: Identifier for the source document
            current_depth: Current depth of link following
            global_visited_docs: Global set of all visited document IDs

        Returns:
            List of linked documents
        """
        if current_depth >= self.max_link_depth or not self.follow_references or not self.reference_field:
            logger.debug(f"Not following references for {source_id}")
            return []

        # Initialize global visited set if not provided
        if global_visited_docs is None:
            global_visited_docs = set()

        # Add current document to global visited set
        global_visited_docs.add(source_id)

        logger.debug(f"Following references in MongoDB document {source_id} at depth {current_depth}")

        linked_docs = []

        try:
            # Parse content as JSON
            try:
                document = json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse content as JSON for {source_id}")
                return []

            # Extract references
            references = self._extract_references(document)

            # Process each reference
            for reference in references:
                reference_id = reference.get('id')
                if not reference_id:
                    continue

                # Format reference ID for identification
                qualified_ref = f"mongodb://{self.database_name}/{self.collection_name}/{reference_id}"

                # Skip if globally visited
                if qualified_ref in global_visited_docs:
                    logger.debug(f"Skipping globally visited reference: {qualified_ref}")
                    continue

                global_visited_docs.add(qualified_ref)

                try:
                    # Fetch the referenced document
                    linked_doc = self.fetch_document(reference_id)
                    linked_docs.append(linked_doc)
                    logger.debug(f"Successfully fetched referenced document: {qualified_ref}")

                    # Recursively follow references if not at max depth
                    if current_depth + 1 < self.max_link_depth:
                        logger.debug(
                            f"Recursively following references from {qualified_ref} at depth {current_depth + 1}")
                        nested_docs = self.follow_links(
                            linked_doc["content"],
                            linked_doc["id"],
                            current_depth + 1,
                            global_visited_docs
                        )
                        linked_docs.extend(nested_docs)
                except Exception as e:
                    logger.warning(f"Error following reference {reference_id} from {source_id}: {str(e)}")

            logger.debug(f"Completed following references from {source_id}: found {len(linked_docs)} linked documents")
            return linked_docs

        except Exception as e:
            logger.error(f"Error following references from MongoDB document {source_id}: {str(e)}")
            return []

    def _extract_references(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract references from a document.

        Args:
            document: Document data

        Returns:
            List of reference dictionaries
        """
        references = []

        # Function to recursively extract references
        def extract_from_dict(d, path=""):
            if not isinstance(d, dict):
                return

            # Check if current dict has reference field
            if self.reference_field in d:
                ref_value = d[self.reference_field]

                # Handle different reference formats
                if isinstance(ref_value, (str, int, ObjectId)):
                    references.append({
                        'id': str(ref_value),
                        'path': f"{path}.{self.reference_field}" if path else self.reference_field
                    })
                elif isinstance(ref_value, list):
                    # Handle array of references
                    for i, val in enumerate(ref_value):
                        if isinstance(val, (str, int, ObjectId)):
                            references.append({
                                'id': str(val),
                                'path': f"{path}.{self.reference_field}[{i}]" if path else f"{self.reference_field}[{i}]"
                            })
                        elif isinstance(val, dict):
                            # Handle complex references
                            if self.id_field in val:
                                references.append({
                                    'id': str(val[self.id_field]),
                                    'path': f"{path}.{self.reference_field}[{i}].{self.id_field}" if path else f"{self.reference_field}[{i}].{self.id_field}"
                                })

            # Recurse into nested dictionaries
            for key, value in d.items():
                new_path = f"{path}.{key}" if path else key

                if isinstance(value, dict):
                    extract_from_dict(value, new_path)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            extract_from_dict(item, f"{new_path}[{i}]")

        # Start recursive extraction
        extract_from_dict(document)
        return references

    @staticmethod
    def _extract_mongo_id(source_id: str) -> Any:
        """
        Extract MongoDB ID from source ID.

        Args:
            source_id: Source identifier

        Returns:
            MongoDB ID
        """
        # Check if it's already a MongoDB URI
        if source_id.startswith('mongodb://'):
            # Format: mongodb://database/collection/id
            parts = source_id.split('/')
            if len(parts) >= 5:
                return parts[4]  # Return the ID part

        # Otherwise, assume it's just the ID
        return source_id

    def _get_doc_id_str(self, document: Dict[str, Any]) -> str:
        """
        Get document ID as string.

        Args:
            document: MongoDB document

        Returns:
            String representation of document ID
        """
        doc_id = document.get(self.id_field, None)

        if doc_id is None:
            return "unknown"

        if isinstance(doc_id, ObjectId):
            return str(doc_id)

        return str(doc_id)

    def _get_timestamp(self, document: Dict[str, Any]) -> Optional[float]:
        """
        Get document timestamp.

        Args:
            document: MongoDB document

        Returns:
            Timestamp as float or None if not available
        """
        if not self.timestamp_field or self.timestamp_field not in document:
            return None

        timestamp = document[self.timestamp_field]

        # Handle different timestamp formats
        if isinstance(timestamp, (int, float)):
            # Assume it's already a timestamp
            return float(timestamp)
        elif hasattr(timestamp, 'timestamp'):
            # For datetime objects
            return timestamp.timestamp()

        # Try to convert string to timestamp
        try:
            import dateutil.parser
            dt = dateutil.parser.parse(str(timestamp))
            return dt.timestamp()
        except Exception:
            return None

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer to handle MongoDB-specific types."""
        if isinstance(obj, ObjectId):
            return str(obj)

        # Handle other MongoDB-specific types
        try:
            # noinspection PyUnresolvedReferences
            from bson import Timestamp, datetime
            if isinstance(obj, (Timestamp, datetime)):
                return str(obj)
        except ImportError:
            pass

        raise TypeError(f"Type {type(obj)} not serializable")

    def __del__(self):
        """Close MongoDB connection when object is deleted."""
        if self.client:
            try:
                self.client.close()
                logger.debug("Closed MongoDB connection")
            except Exception as e:
                logger.warning(f"Error closing MongoDB connection: {str(e)}")
