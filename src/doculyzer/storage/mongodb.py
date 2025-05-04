import logging
import os
from typing import Dict, Any, List, Optional, Tuple

import time

from .element_relationship import ElementRelationship

# Try to import MongoDB library
try:
    import pymongo
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, DuplicateKeyError
    import numpy as np

    PYMONGO_AVAILABLE = True
except ImportError:
    MongoClient = None
    ConnectionFailure = None
    DuplicateKeyError = None
    np = None
    logging.warning("pymongo not available. Install with 'pip install pymongo'.")
    PYMONGO_AVAILABLE = False

from .base import DocumentDatabase

logger = logging.getLogger(__name__)

# Try to import the config the same way SQLite does
try:
    from ..config import Config

    config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
except Exception as e:
    logger.warning(f"Error configuring MongoDB provider: {str(e)}")
    config = None


class MongoDBDocumentDatabase(DocumentDatabase):
    """MongoDB implementation of document database."""

    def get_outgoing_relationships(self, element_pk: int) -> List[ElementRelationship]:
        """
        Find all relationships where the specified element_pk is the source.

        Implementation for MongoDB database using aggregation pipeline to efficiently
        retrieve target element information.

        Args:
            element_pk: The primary key of the element

        Returns:
            List of ElementRelationship objects where the specified element is the source
        """
        if not self.db:
            raise ValueError("Database not initialized")

        relationships = []

        # Get the element to find its element_id and type
        element = self.get_element(element_pk)
        if not element:
            logger.warning(f"Element with PK {element_pk} not found")
            return []

        element_id = element.get("element_id")
        if not element_id:
            logger.warning(f"Element with PK {element_pk} has no element_id")
            return []

        element_type = element.get("element_type", "")

        try:
            # Use aggregation pipeline to join relationships with elements
            # This is similar to a SQL JOIN but using MongoDB's aggregation framework
            pipeline = [
                # Match relationships where this element is the source
                {"$match": {"source_id": element_id}},

                # Lookup target elements
                {"$lookup": {
                    "from": "elements",
                    "localField": "target_reference",
                    "foreignField": "element_id",
                    "as": "target_element"
                }},

                # Unwind target_element array (or preserve null with preserveNullAndEmptyArrays)
                {"$unwind": {
                    "path": "$target_element",
                    "preserveNullAndEmptyArrays": True
                }}
            ]

            # Execute the aggregation pipeline
            results = list(self.db.relationships.aggregate(pipeline))

            # Process results
            for result in results:
                # Remove MongoDB's _id field
                if "_id" in result:
                    del result["_id"]

                # Extract target element information if available
                target_element_pk = None
                target_element_type = None
                target_content_preview = None  # Added this field

                if "target_element" in result and result["target_element"]:
                    target_element = result["target_element"]
                    target_element_pk = target_element.get("element_pk")
                    target_element_type = target_element.get("element_type")
                    target_content_preview = target_element.get("content_preview", "")  # Added this field

                    # Remove the target_element object from the result
                    del result["target_element"]

                # Create enriched relationship
                relationship = ElementRelationship(
                    relationship_id=result.get("relationship_id", ""),
                    source_id=element_id,
                    source_element_pk=element_pk,
                    source_element_type=element_type,
                    relationship_type=result.get("relationship_type", ""),
                    target_reference=result.get("target_reference", ""),
                    target_element_pk=target_element_pk,
                    target_element_type=target_element_type,
                    target_content_preview=target_content_preview,  # Added this field
                    doc_id=result.get("doc_id"),
                    metadata=result.get("metadata", {}),
                    is_source=True
                )

                relationships.append(relationship)

            return relationships

        except Exception as e:
            logger.error(f"Error getting outgoing relationships for element {element_pk}: {str(e)}")
            return []

    def __init__(self, conn_params: Dict[str, Any]):
        """
        Initialize MongoDB document database.

        Args:
            conn_params: Connection parameters for MongoDB
                (host, port, username, password, db_name)
        """
        self.conn_params = conn_params
        self.client = None
        self.db = None
        self.vector_search = False
        self.embedding_generator = None
        self.vector_dimension = None
        if config:
            self.vector_dimension = config.config.get('embedding', {}).get('dimensions', 384)
        else:
            self.vector_dimension = 384  # Default if config not available

    def initialize(self) -> None:
        """Initialize the database by connecting and creating collections if they don't exist."""
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo is required for MongoDB support")

        # Extract connection parameters
        host = self.conn_params.get('host', 'localhost')
        port = self.conn_params.get('port', 27017)
        username = self.conn_params.get('username')
        password = self.conn_params.get('password')
        db_name = self.conn_params.get('db_name', 'doculyzer')

        # Build connection string
        connection_string = "mongodb://"
        if username and password:
            connection_string += f"{username}:{password}@"
        connection_string += f"{host}:{port}/{db_name}"

        # Add additional connection options
        options = self.conn_params.get('options', {})
        if options:
            option_str = "&".join(f"{k}={v}" for k, v in options.items())
            connection_string += f"?{option_str}"

        # Connect to MongoDB
        try:
            self.client = MongoClient(connection_string)
            # Ping the server to verify connection
            self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB at {host}:{port}")

            # Select database
            self.db = self.client[db_name]

            # Create collections and indexes if they don't exist
            self._create_collections()

            # Check if vector search is available
            self._check_vector_search()

            logger.info(f"Initialized MongoDB database with vector search: {self.vector_search}")

        except ConnectionFailure as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise

    def _check_vector_search(self) -> None:
        """Check if vector search capabilities are available."""
        try:
            # Check MongoDB version (5.0+ required for some vector features)
            server_info = self.client.server_info()
            version = server_info.get('version', '0.0.0')
            major_version = int(version.split('.')[0])

            if major_version >= 5:
                # Check if Atlas Vector Search is available
                try:
                    # Try getting list of search indexes to see if feature is available
                    # indexes = list(self.db.command({"listSearchIndexes": "elements"}))
                    self.vector_search = True
                    logger.info("MongoDB vector search is available")
                    return
                except Exception as e:
                    logger.debug(f"Vector search not available: {str(e)}")

            logger.info(f"MongoDB version: {version}, vector search unavailable")
            self.vector_search = False

        except Exception as e:
            logger.warning(f"Error checking vector search availability: {str(e)}")
            self.vector_search = False

    def _create_collections(self) -> None:
        """Create collections and indexes if they don't exist."""
        # Documents collection
        if "documents" not in self.db.list_collection_names():
            self.db.create_collection("documents")

        # Create indexes for documents collection
        self.db.documents.create_index("doc_id", unique=True)
        self.db.documents.create_index("source")

        # Elements collection - Updated to match SQLite schema
        if "elements" not in self.db.list_collection_names():
            self.db.create_collection("elements")

        # Create indexes for elements collection
        self.db.elements.create_index("element_id", unique=True)
        self.db.elements.create_index("element_pk", unique=True)
        self.db.elements.create_index("doc_id")
        self.db.elements.create_index("parent_id")
        self.db.elements.create_index("element_type")

        # Relationships collection
        if "relationships" not in self.db.list_collection_names():
            self.db.create_collection("relationships")

        # Create indexes for relationships collection
        self.db.relationships.create_index("relationship_id", unique=True)
        self.db.relationships.create_index("source_id")
        self.db.relationships.create_index("relationship_type")

        # Embeddings collection - Update to match SQLite schema
        if "embeddings" not in self.db.list_collection_names():
            self.db.create_collection("embeddings")

        # Create indexes for embeddings collection
        self.db.embeddings.create_index("element_pk", unique=True)

        # Processing history collection
        if "processing_history" not in self.db.list_collection_names():
            self.db.create_collection("processing_history")

        # Create indexes for processing history collection
        self.db.processing_history.create_index("source_id", unique=True)

        # Ensure counters collection exists for auto-incrementing element_pk
        if "counters" not in self.db.list_collection_names():
            self.db.create_collection("counters")
            # Initialize element_pk counter if it doesn't exist
            if not self.db.counters.find_one({"_id": "element_pk"}):
                self.db.counters.insert_one({"_id": "element_pk", "seq": 0})

        logger.info("Created MongoDB collections and indexes")

    def close(self) -> None:
        """Close the database connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None

    def get_last_processed_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about when a document was last processed."""
        if not self.db:
            raise ValueError("Database not initialized")

        try:
            history = self.db.processing_history.find_one({"source_id": source_id})
            if not history:
                return None

            # Remove MongoDB's _id field
            if "_id" in history:
                del history["_id"]

            return history
        except Exception as e:
            logger.error(f"Error getting processing history for {source_id}: {str(e)}")
            return None

    def update_processing_history(self, source_id: str, content_hash: str) -> None:
        """Update the processing history for a document."""
        if not self.db:
            raise ValueError("Database not initialized")

        try:
            # Check if record exists
            existing = self.db.processing_history.find_one({"source_id": source_id})
            processing_count = 1  # Default for new records

            if existing:
                processing_count = existing.get("processing_count", 0) + 1

            # Update or insert record
            self.db.processing_history.update_one(
                {"source_id": source_id},
                {
                    "$set": {
                        "content_hash": content_hash,
                        "last_modified": time.time(),
                        "processing_count": processing_count
                    }
                },
                upsert=True
            )

            logger.debug(f"Updated processing history for {source_id}")

        except Exception as e:
            logger.error(f"Error updating processing history for {source_id}: {str(e)}")

    def store_document(self, document: Dict[str, Any], elements: List[Dict[str, Any]],
                       relationships: List[Dict[str, Any]]) -> None:
        """
        Store a document with its elements and relationships.
        If a document with the same source already exists, update it instead.

        Args:
            document: Document metadata
            elements: Document elements
            relationships: Element relationships
        """
        if not self.db:
            raise ValueError("Database not initialized")

        source = document.get("source", "")
        content_hash = document.get("content_hash", "")

        # Check if document already exists with this source
        existing_doc = self.db.documents.find_one({"source": source}) if source else None

        if existing_doc:
            # Document exists, update it
            doc_id = existing_doc["doc_id"]
            document["doc_id"] = doc_id  # Use existing doc_id

            # Update all elements to use the existing doc_id
            for element in elements:
                element["doc_id"] = doc_id

            self.update_document(doc_id, document, elements, relationships)
            return

        # New document, proceed with creation
        doc_id = document["doc_id"]

        try:
            # Store document
            document_with_timestamps = {
                **document,
                "created_at": document.get("created_at", time.time()),
                "updated_at": document.get("updated_at", time.time())
            }

            self.db.documents.insert_one(document_with_timestamps)

            # Process elements with element_pk
            elements_to_insert = []
            for i, element in enumerate(elements):
                # Generate MongoDB compatible representation with element_pk
                mongo_element = {**element}

                # Generate a unique element_pk if not present
                if "element_pk" not in mongo_element:
                    # Use an auto-incrementing counter similar to SQLite
                    counter = self.db.counters.find_one_and_update(
                        {"_id": "element_pk"},
                        {"$inc": {"seq": 1}},
                        upsert=True,
                        return_document=True
                    )
                    mongo_element["element_pk"] = counter["seq"]
                    # Store it back into the original element for reference
                    element["element_pk"] = mongo_element["element_pk"]

                elements_to_insert.append(mongo_element)

            # Store elements in bulk if there are any
            if elements_to_insert:
                self.db.elements.insert_many(elements_to_insert)

            # Store relationships in bulk if there are any
            if relationships:
                self.db.relationships.insert_many(relationships)

            # Update processing history
            if source:
                self.update_processing_history(source, content_hash)

            logger.info(
                f"Stored document {doc_id} with {len(elements)} elements and {len(relationships)} relationships")

        except Exception as e:
            logger.error(f"Error storing document {doc_id}: {str(e)}")
            raise

    def update_document(self, doc_id: str, document: Dict[str, Any],
                        elements: List[Dict[str, Any]],
                        relationships: List[Dict[str, Any]]) -> None:
        """
        Update an existing document.
        """
        if not self.db:
            raise ValueError("Database not initialized")

        # Check if document exists
        existing_doc = self.db.documents.find_one({"doc_id": doc_id})
        if not existing_doc:
            raise ValueError(f"Document not found: {doc_id}")

        try:
            # Update document timestamps
            document["updated_at"] = time.time()
            if "created_at" not in document and "created_at" in existing_doc:
                document["created_at"] = existing_doc["created_at"]

            # Get all element IDs for this document
            element_ids = [element["element_id"] for element in
                           self.db.elements.find({"doc_id": doc_id}, {"element_id": 1})]

            # Delete all existing relationships related to this document's elements
            if element_ids:
                self.db.relationships.delete_many({"source_id": {"$in": element_ids}})

            # Delete all existing embeddings for this document's elements
            element_pks = [element["element_pk"] for element in
                           self.db.elements.find({"doc_id": doc_id}, {"element_pk": 1})]
            if element_pks:
                self.db.embeddings.delete_many({"element_pk": {"$in": element_pks}})

            # Delete all existing elements for this document
            self.db.elements.delete_many({"doc_id": doc_id})

            # Replace the document
            self.db.documents.replace_one({"doc_id": doc_id}, document)

            # Process elements with element_pk
            elements_to_insert = []
            for element in elements:
                # Generate MongoDB compatible representation
                mongo_element = {**element}

                # Generate a unique element_pk if not present
                if "element_pk" not in mongo_element:
                    # Use auto-incrementing counter
                    counter = self.db.counters.find_one_and_update(
                        {"_id": "element_pk"},
                        {"$inc": {"seq": 1}},
                        upsert=True,
                        return_document=True
                    )
                    mongo_element["element_pk"] = counter["seq"]
                    # Store it back for reference
                    element["element_pk"] = mongo_element["element_pk"]

                elements_to_insert.append(mongo_element)

            # Insert new elements
            if elements_to_insert:
                self.db.elements.insert_many(elements_to_insert)

            # Insert new relationships
            if relationships:
                self.db.relationships.insert_many(relationships)

            # Update processing history
            source = document.get("source", "")
            content_hash = document.get("content_hash", "")
            if source:
                self.update_processing_history(source, content_hash)

            logger.info(
                f"Updated document {doc_id} with {len(elements)} elements and {len(relationships)} relationships")

        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {str(e)}")
            raise

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        if not self.db:
            raise ValueError("Database not initialized")

        document = self.db.documents.find_one({"doc_id": doc_id})

        if not document:
            return None

        # Remove MongoDB's _id field
        if "_id" in document:
            del document["_id"]

        return document

    def get_document_elements(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get elements for a document, by doc_id or source."""
        if not self.db:
            raise ValueError("Database not initialized")

        # First try to get document by doc_id
        document = self.db.documents.find_one({"doc_id": doc_id})

        if not document:
            # If not found, try by source
            document = self.db.documents.find_one({"source": doc_id})
            if not document:
                return []
            doc_id = document["doc_id"]

        elements = list(self.db.elements.find({"doc_id": doc_id}))

        # Remove MongoDB's _id field from each element
        for element in elements:
            if "_id" in element:
                del element["_id"]

        return elements

    def get_document_relationships(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get relationships for a document."""
        if not self.db:
            raise ValueError("Database not initialized")

        # First get all element IDs for the document
        element_ids = [element["element_id"] for element in
                       self.db.elements.find({"doc_id": doc_id}, {"element_id": 1})]

        if not element_ids:
            return []

        # Find relationships involving these elements
        relationships = list(self.db.relationships.find({"source_id": {"$in": element_ids}}))

        # Remove MongoDB's _id field from each relationship
        for relationship in relationships:
            if "_id" in relationship:
                del relationship["_id"]

        return relationships

    def get_element(self, element_id_or_pk: str | int) -> Optional[Dict[str, Any]]:
        """
        Get element by ID or PK.

        Args:
            element_id_or_pk: Either the element_id (string) or element_pk (integer)
        """
        if not self.db:
            raise ValueError("Database not initialized")

        element = None

        # Try to interpret as element_pk (integer) first
        try:
            element_pk = int(element_id_or_pk)
            element = self.db.elements.find_one({"element_pk": element_pk})
        except (ValueError, TypeError):
            # If not an integer, treat as element_id (string)
            element = self.db.elements.find_one({"element_id": element_id_or_pk})

        if not element:
            return None

        # Remove MongoDB's _id field
        if "_id" in element:
            del element["_id"]

        return element

    def find_documents(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find documents matching query."""
        if not self.db:
            raise ValueError("Database not initialized")

        # Build MongoDB query
        mongo_query = {}

        if query:
            for key, value in query.items():
                if key == "metadata":
                    # Handle metadata queries
                    for meta_key, meta_value in value.items():
                        mongo_query[f"metadata.{meta_key}"] = meta_value
                else:
                    mongo_query[key] = value

        # Execute query
        documents = list(self.db.documents.find(mongo_query).limit(limit))

        # Remove MongoDB's _id field from each document
        for document in documents:
            if "_id" in document:
                del document["_id"]

        return documents

    def find_elements(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find elements matching query."""
        if not self.db:
            raise ValueError("Database not initialized")

        # Build MongoDB query
        mongo_query = {}

        if query:
            for key, value in query.items():
                if key == "metadata":
                    # Handle metadata queries
                    for meta_key, meta_value in value.items():
                        mongo_query[f"metadata.{meta_key}"] = meta_value
                elif key == "element_type" and isinstance(value, list):
                    # Handle list of element types
                    mongo_query["element_type"] = {"$in": value}
                elif isinstance(value, list):
                    # Handle other list values (like doc_id list)
                    mongo_query[key] = {"$in": value}
                else:
                    mongo_query[key] = value

        # Execute query
        elements = list(self.db.elements.find(mongo_query).limit(limit))

        # Remove MongoDB's _id field from each element
        for element in elements:
            if "_id" in element:
                del element["_id"]

        return elements

    def search_elements_by_content(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search elements by content preview."""
        if not self.db:
            raise ValueError("Database not initialized")

        # Create text search query
        elements = list(self.db.elements.find(
            {"content_preview": {"$regex": search_text, "$options": "i"}}
        ).limit(limit))

        # Remove MongoDB's _id field from each element
        for element in elements:
            if "_id" in element:
                del element["_id"]

        return elements

    def store_embedding(self, element_pk: int, embedding: List[float]) -> None:
        """
        Store embedding for an element.
        """
        if not self.db:
            raise ValueError("Database not initialized")

        # Verify element exists
        element = self.db.elements.find_one({"element_pk": element_pk})
        if not element:
            raise ValueError(f"Element not found: {element_pk}")

        # Update vector dimension based on actual data
        self.vector_dimension = max(self.vector_dimension, len(embedding))

        try:
            # Store or update embedding
            self.db.embeddings.update_one(
                {"element_pk": element_pk},
                {
                    "$set": {
                        "embedding": embedding,
                        "dimensions": len(embedding),
                        "created_at": time.time()
                    }
                },
                upsert=True
            )

            logger.debug(f"Stored embedding for element {element_pk}")

        except Exception as e:
            logger.error(f"Error storing embedding for {element_pk}: {str(e)}")
            raise

    def get_embedding(self, element_pk: int) -> Optional[List[float]]:
        """
        Get embedding for an element.
        """
        if not self.db:
            raise ValueError("Database not initialized")

        embedding_doc = self.db.embeddings.find_one({"element_pk": element_pk})
        if not embedding_doc:
            return None

        return embedding_doc.get("embedding")

    def search_by_embedding(self, query_embedding: List[float], limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by embedding similarity with optional filtering.
        Returns (element_pk, similarity) tuples for consistency with other implementations.
        """
        if not self.db:
            raise ValueError("Database not initialized")

        try:
            if self.vector_search:
                return self._search_by_vector_index(query_embedding, limit, filter_criteria)
            else:
                return self._search_by_cosine_similarity(query_embedding, limit, filter_criteria)
        except Exception as e:
            logger.error(f"Error searching by embedding: {str(e)}")
            # Fall back to cosine similarity
            return self._search_by_cosine_similarity(query_embedding, limit, filter_criteria)

    def _search_by_vector_index(self, query_embedding: List[float], limit: int = 10,
                                filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Search embeddings using MongoDB Atlas Vector Search with filtering."""
        try:
            # Define the vector search pipeline
            pipeline = [{
                "$vectorSearch": {
                    "index": "embeddings_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 5,  # Get more candidates for better results
                    "limit": limit * 10  # Get more results than needed to allow for filtering
                }
            }, {
                "$lookup": {
                    "from": "elements",
                    "localField": "element_pk",
                    "foreignField": "element_pk",
                    "as": "element"
                }
            }, {
                "$unwind": "$element"
            }, {
                "$lookup": {
                    "from": "documents",
                    "localField": "element.doc_id",
                    "foreignField": "doc_id",
                    "as": "document"
                }
            }, {
                "$unwind": {
                    "path": "$document",
                    "preserveNullAndEmptyArrays": True
                }
            }]

            # Add document lookup to support document-level filtering

            # Add filter stages if criteria provided
            if filter_criteria:
                match_conditions = {}

                for key, value in filter_criteria.items():
                    if key == "element_type" and isinstance(value, list):
                        # Handle list of allowed element types
                        match_conditions["element.element_type"] = {"$in": value}
                    elif key == "doc_id" and isinstance(value, list):
                        # Handle list of document IDs to include
                        match_conditions["element.doc_id"] = {"$in": value}
                    elif key == "exclude_doc_id" and isinstance(value, list):
                        # Handle list of document IDs to exclude
                        match_conditions["element.doc_id"] = {"$nin": value}
                    elif key == "exclude_doc_source" and isinstance(value, list):
                        # Handle list of document sources to exclude
                        match_conditions["document.source"] = {"$nin": value}
                    else:
                        # Simple equality filter
                        match_conditions[f"element.{key}"] = value

                if match_conditions:
                    pipeline.append({"$match": match_conditions})

            # Add projection and limit
            pipeline.extend([
                {
                    "$project": {
                        "_id": 0,
                        "element_pk": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$limit": limit
                }
            ])

            # Execute the search
            results = list(self.db.embeddings.aggregate(pipeline))

            # Format results as (element_pk, similarity_score)
            return [(doc["element_pk"], doc["score"]) for doc in results]

        except Exception as e:
            logger.error(f"Error using vector search index: {str(e)}")
            raise

    def _search_by_cosine_similarity(self, query_embedding: List[float], limit: int = 10,
                                     filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Fall back to calculating cosine similarity in Python with filtering."""
        # Begin building a pipeline for embeddings with element and document data
        pipeline = [
            {
                "$lookup": {
                    "from": "elements",
                    "localField": "element_pk",
                    "foreignField": "element_pk",
                    "as": "element"
                }
            },
            {
                "$unwind": "$element"
            },
            {
                "$lookup": {
                    "from": "documents",
                    "localField": "element.doc_id",
                    "foreignField": "doc_id",
                    "as": "document"
                }
            },
            {
                "$unwind": {
                    "path": "$document",
                    "preserveNullAndEmptyArrays": True
                }
            }
        ]

        # Add filter criteria if provided
        if filter_criteria:
            match_conditions = {}

            for key, value in filter_criteria.items():
                if key == "element_type" and isinstance(value, list):
                    # Handle list of allowed element types
                    match_conditions["element.element_type"] = {"$in": value}
                elif key == "doc_id" and isinstance(value, list):
                    # Handle list of document IDs to include
                    match_conditions["element.doc_id"] = {"$in": value}
                elif key == "exclude_doc_id" and isinstance(value, list):
                    # Handle list of document IDs to exclude
                    match_conditions["element.doc_id"] = {"$nin": value}
                elif key == "exclude_doc_source" and isinstance(value, list):
                    # Handle list of document sources to exclude
                    match_conditions["document.source"] = {"$nin": value}
                else:
                    # Simple equality filter
                    match_conditions[f"element.{key}"] = value

            if match_conditions:
                pipeline.append({"$match": match_conditions})

        # Add projection to get just what we need
        pipeline.append({
            "$project": {
                "_id": 0,
                "element_pk": 1,
                "embedding": 1
            }
        })

        # Execute aggregation to get filtered embeddings
        filtered_embeddings = list(self.db.embeddings.aggregate(pipeline))

        # Calculate cosine similarity for each embedding
        similarities = []
        query_array = np.array(query_embedding)

        for doc in filtered_embeddings:
            element_pk = doc["element_pk"]
            embedding = doc["embedding"]

            if embedding and len(embedding) == len(query_embedding):
                similarity = self._calculate_cosine_similarity(query_array, np.array(embedding))
                similarities.append((element_pk, similarity))

        # Sort by similarity (highest first) and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

    @staticmethod
    def _calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all associated elements and relationships."""
        if not self.db:
            raise ValueError("Database not initialized")

        # Check if document exists
        if not self.db.documents.find_one({"doc_id": doc_id}):
            return False

        try:
            # Get all element IDs for this document
            element_ids = [element["element_id"] for element in
                           self.db.elements.find({"doc_id": doc_id}, {"element_id": 1})]

            # Get all element PKs for this document
            element_pks = [element["element_pk"] for element in
                           self.db.elements.find({"doc_id": doc_id}, {"element_pk": 1})]

            # Delete embeddings for these elements
            if element_pks:
                self.db.embeddings.delete_many({"element_pk": {"$in": element_pks}})

            # Delete relationships involving these elements
            if element_ids:
                self.db.relationships.delete_many({"source_id": {"$in": element_ids}})

            # Delete elements
            self.db.elements.delete_many({"doc_id": doc_id})

            # Delete document
            self.db.documents.delete_one({"doc_id": doc_id})

            logger.info(f"Deleted document {doc_id} with {len(element_ids)} elements")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False

    def store_relationship(self, relationship: Dict[str, Any]) -> None:
        """
        Store a relationship between elements.

        Args:
            relationship: Relationship data with source_id, relationship_type, and target_reference
        """
        if not self.db:
            raise ValueError("Database not initialized")

        try:
            # Insert or update the relationship
            self.db.relationships.update_one(
                {"relationship_id": relationship["relationship_id"]},
                {"$set": relationship},
                upsert=True
            )
            logger.debug(f"Stored relationship {relationship['relationship_id']}")
        except Exception as e:
            logger.error(f"Error storing relationship: {str(e)}")
            raise

    def delete_relationships_for_element(self, element_id: str, relationship_type: str = None) -> None:
        """
        Delete relationships for an element.

        Args:
            element_id: Element ID
            relationship_type: Optional relationship type to filter by
        """
        if not self.db:
            raise ValueError("Database not initialized")

        try:
            # Build query for source relationships
            query = {"source_id": element_id}
            if relationship_type:
                query["relationship_type"] = relationship_type

            # Delete source relationships
            self.db.relationships.delete_many(query)

            # Build query for target relationships
            query = {"target_reference": element_id}
            if relationship_type:
                query["relationship_type"] = relationship_type

            # Delete target relationships
            self.db.relationships.delete_many(query)

            logger.debug(f"Deleted relationships for element {element_id}")
        except Exception as e:
            logger.error(f"Error deleting relationships for element {element_id}: {str(e)}")
            raise

    def create_vector_search_index(self) -> bool:
        """
        Create a vector search index for embeddings collection.
        This requires MongoDB Atlas.

        Returns:
            bool: True if index was created successfully, False otherwise
        """
        if not self.db:
            raise ValueError("Database not initialized")

        try:
            # Define the index
            index_definition = {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "embedding": {
                            "dimensions": self.vector_dimension,
                            "similarity": "cosine",
                            "type": "knnVector"
                        }
                    }
                }
            }

            # Create the index
            self.db.command({
                "createSearchIndex": "embeddings",
                "name": "embeddings_vector_index",
                "definition": index_definition
            })

            logger.info(f"Created vector search index with {self.vector_dimension} dimensions")
            self.vector_search = True
            return True

        except Exception as e:
            logger.error(f"Error creating vector search index: {str(e)}")
            return False

    def search_by_text(self, search_text: str, limit: int = 10,
                       filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by semantic similarity to the provided text.
        Returns (element_pk, similarity) tuples for consistency with other implementations.
        """
        if not self.db:
            raise ValueError("Database not initialized")

        try:
            if self.embedding_generator is None:
                from ..embeddings import get_embedding_generator
                self.embedding_generator = get_embedding_generator(config)

            # Generate embedding for the search text
            query_embedding = self.embedding_generator.generate(search_text)

            # Use the embedding to search, passing the filter criteria
            return self.search_by_embedding(query_embedding, limit, filter_criteria)

        except Exception as e:
            logger.error(f"Error in semantic search by text: {str(e)}")
            # Return empty list on error
            return []
