import logging
import os
import time
from typing import Dict, Any, List, Optional, Tuple

from ..config import Config

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

config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))


class MongoDBDocumentDatabase(DocumentDatabase):
    """MongoDB implementation of document database."""

    def __init__(self, conn_params: Dict[str, Any]):
        """
        Initialize MongoDB document database.

        Args:
            conn_params: Connection parameters for MongoDB
                (host, port, username, password, db_name)
        """
        self.config = None
        self.conn_params = conn_params
        self.client = None
        self.db = None
        self.vector_search = False
        self.vector_dimension = config.config.get('embedding', {}).get('dimensions', 384)

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

        # Elements collection
        if "elements" not in self.db.list_collection_names():
            self.db.create_collection("elements")

        # Create indexes for elements collection
        self.db.elements.create_index("element_id", unique=True)
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

        # Embeddings collection
        if "embeddings" not in self.db.list_collection_names():
            self.db.create_collection("embeddings")

        # Create indexes for embeddings collection
        self.db.embeddings.create_index("element_id", unique=True)

        # Processing history collection
        if "processing_history" not in self.db.list_collection_names():
            self.db.create_collection("processing_history")

        # Create indexes for processing history collection
        self.db.processing_history.create_index("source_id", unique=True)

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
        existing_doc = self.db.documents.find_one({"source": source})

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

            # Process elements
            elements_to_insert = []
            for element in elements:
                # Generate MongoDB compatible representation
                mongo_element = {**element}
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

            # Delete all existing relationships related to this document's elements
            element_ids = [element["element_id"] for element in
                           self.db.elements.find({"doc_id": doc_id}, {"element_id": 1})]

            if element_ids:
                self.db.relationships.delete_many({"source_id": {"$in": element_ids}})

            # Delete all existing embeddings for this document's elements
            if element_ids:
                self.db.embeddings.delete_many({"element_id": {"$in": element_ids}})

            # Delete all existing elements for this document
            self.db.elements.delete_many({"doc_id": doc_id})

            # Replace the document
            self.db.documents.replace_one({"doc_id": doc_id}, document)

            # Insert new elements
            if elements:
                self.db.elements.insert_many(elements)

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
        """Get elements for a document."""
        if not self.db:
            raise ValueError("Database not initialized")

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

    def get_element(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Get element by ID."""
        if not self.db:
            raise ValueError("Database not initialized")

        element = self.db.elements.find_one({"element_id": element_id})

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

    def store_embedding(self, element_id: str, embedding: List[float]) -> None:
        """Store embedding for an element."""
        if not self.db:
            raise ValueError("Database not initialized")

        # Verify element exists
        element = self.db.elements.find_one({"element_id": element_id})
        if not element:
            raise ValueError(f"Element not found: {element_id}")

        # Update vector dimension based on actual data
        self.vector_dimension = max(self.vector_dimension, len(embedding))

        try:
            # Store or update embedding
            self.db.embeddings.update_one(
                {"element_id": element_id},
                {
                    "$set": {
                        "embedding": embedding,
                        "dimensions": len(embedding),
                        "created_at": time.time()
                    }
                },
                upsert=True
            )

            logger.debug(f"Stored embedding for element {element_id}")

        except Exception as e:
            logger.error(f"Error storing embedding for {element_id}: {str(e)}")
            raise

    def get_embedding(self, element_id: str) -> Optional[List[float]]:
        """Get embedding for an element."""
        if not self.db:
            raise ValueError("Database not initialized")

        embedding_doc = self.db.embeddings.find_one({"element_id": element_id})
        if not embedding_doc:
            return None

        return embedding_doc.get("embedding")

    def search_by_embedding(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[str, float]]:
        """Search elements by embedding similarity."""
        if not self.db:
            raise ValueError("Database not initialized")

        try:
            if self.vector_search:
                return self._search_by_vector_index(query_embedding, limit)
            else:
                return self._search_by_cosine_similarity(query_embedding, limit)
        except Exception as e:
            logger.error(f"Error searching by embedding: {str(e)}")
            return self._search_by_cosine_similarity(query_embedding, limit)

    def _search_by_vector_index(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[str, float]]:
        """Search embeddings using MongoDB Atlas Vector Search."""
        try:
            # Define the vector search pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "embeddings_vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 5,  # Get more candidates for better results
                        "limit": limit
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "element_id": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            # Execute the search
            results = list(self.db.embeddings.aggregate(pipeline))

            # Format results
            return [(doc["element_id"], doc["score"]) for doc in results]

        except Exception as e:
            logger.error(f"Error using vector search index: {str(e)}")
            raise

    def _search_by_cosine_similarity(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[str, float]]:
        """Fall back to calculating cosine similarity in Python."""
        # Get all embeddings from MongoDB
        all_embeddings = list(self.db.embeddings.find({}, {"element_id": 1, "embedding": 1}))

        # Calculate cosine similarity for each embedding
        similarities = []
        query_array = np.array(query_embedding)

        for doc in all_embeddings:
            element_id = doc["element_id"]
            embedding = doc["embedding"]

            if embedding and len(embedding) == len(query_embedding):
                similarity = self._calculate_cosine_similarity(query_array, np.array(embedding))
                similarities.append((element_id, similarity))

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

            # Delete embeddings for these elements
            if element_ids:
                self.db.embeddings.delete_many({"element_id": {"$in": element_ids}})

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

    def search_by_text(self, search_text: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Search elements by semantic similarity to the provided text."""
        if not self.db:
            raise ValueError("Database not initialized")

        from doculyzer.embeddings import get_embedding_generator

        # Get the embedding generator from config
        embedding_generator = get_embedding_generator(self.config)

        # Generate embedding for the search text
        query_embedding = embedding_generator.generate(search_text)

        # Use the embedding to search
        return self.search_by_embedding(query_embedding, limit)
