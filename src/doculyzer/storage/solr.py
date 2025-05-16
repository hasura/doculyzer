import logging
import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from .element_relationship import ElementRelationship
from .element_element import ElementBase, ElementHierarchical
from .base import DocumentDatabase

logger = logging.getLogger(__name__)

# Define global flag for availability - will be set at runtime
PYSOLR_AVAILABLE = False

# Try to import SOLR library at runtime
try:
    import pysolr

    PYSOLR_AVAILABLE = True
except ImportError:
    logger.warning("pysolr not available. Install with 'pip install pysolr'.")


    # Create a placeholder for type checking
    class pysolr:
        class Solr:
            def __init__(self, *args, **kwargs):
                pass

# Try to import the config
try:
    from ..config import Config

    config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
except Exception as e:
    logger.warning(f"Error configuring SOLR provider: {str(e)}")
    config = None


class SolrDocumentDatabase(DocumentDatabase):
    """SOLR implementation of document database."""

    def __init__(self, conn_params: Dict[str, Any]):
        """
        Initialize SOLR document database.

        Args:
            conn_params: Connection parameters for SOLR
                (host, port, username, password, core_prefix)
        """
        self.conn_params = conn_params

        # Extract connection parameters
        host = conn_params.get('host', 'localhost')
        port = conn_params.get('port', 8983)
        username = conn_params.get('username')
        password = conn_params.get('password')
        self.core_prefix = conn_params.get('core_prefix', 'doculyzer')

        # Build base URL
        self.base_url = f"http://{host}:{port}/solr"
        if username and password:
            self.base_url = f"http://{username}:{password}@{host}:{port}/solr"

        # Define core names
        self.documents_core = f"{self.core_prefix}_documents"
        self.elements_core = f"{self.core_prefix}_elements"
        self.relationships_core = f"{self.core_prefix}_relationships"
        self.history_core = f"{self.core_prefix}_history"

        # Initialize SOLR clients to None - will be created in initialize()
        self.documents = None
        self.elements = None
        self.relationships = None
        self.history = None

        # Auto-increment counters
        self.element_pk_counter = 0

        # Configuration for vector search
        self.vector_dimension = conn_params.get('vector_dimension', 384)
        self.embedding_generator = None

    def initialize(self) -> None:
        """Initialize the database by connecting to SOLR and creating cores if needed."""
        if not PYSOLR_AVAILABLE:
            raise ImportError("pysolr is required for SOLR support")

        try:
            # Connect to each core
            self.documents = pysolr.Solr(f"{self.base_url}/{self.documents_core}", always_commit=True)
            self.elements = pysolr.Solr(f"{self.base_url}/{self.elements_core}", always_commit=True)
            self.relationships = pysolr.Solr(f"{self.base_url}/{self.relationships_core}", always_commit=True)
            self.history = pysolr.Solr(f"{self.base_url}/{self.history_core}", always_commit=True)

            # Check if cores exist by making a simple query
            try:
                self.documents.search("*:*", rows=1)
                logger.info(f"Connected to SOLR document core {self.documents_core}")
            except Exception as e:
                logger.warning(f"SOLR core {self.documents_core} may not exist: {str(e)}")
                logger.warning("Create cores using the SOLR admin UI with appropriate schema configuration.")

            # Initialize element_pk counter
            self._initialize_counter()

            logger.info("SOLR document database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing SOLR database: {str(e)}")
            raise

    def _initialize_counter(self) -> None:
        """Initialize the element_pk counter based on highest existing value."""
        try:
            # Search for highest element_pk
            results = self.elements.search("*:*", sort="element_pk desc", rows=1)
            if len(results) > 0:
                self.element_pk_counter = int(results.docs[0].get("element_pk", 0))
                logger.info(f"Initialized element_pk counter to {self.element_pk_counter}")
            else:
                self.element_pk_counter = 0
                logger.info("No existing elements found, element_pk counter set to 0")
        except Exception as e:
            logger.error(f"Error initializing counter: {str(e)}")
            self.element_pk_counter = 0

    def close(self) -> None:
        """Close the database connection."""
        # SOLR connections don't need explicit closing
        self.documents = None
        self.elements = None
        self.relationships = None
        self.history = None

    def get_last_processed_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about when a document was last processed."""
        if not self.history:
            raise ValueError("Database not initialized")

        try:
            results = self.history.search(f"source_id:{source_id}", rows=1)
            if len(results) == 0:
                return None

            # Convert SOLR doc to dict
            record = dict(results.docs[0])
            return record

        except Exception as e:
            logger.error(f"Error getting processing history for {source_id}: {str(e)}")
            return None

    def update_processing_history(self, source_id: str, content_hash: str) -> None:
        """Update the processing history for a document."""
        if not self.history:
            raise ValueError("Database not initialized")

        try:
            # Check if record exists
            existing = self.history.search(f"source_id:{source_id}", rows=1)
            processing_count = 1  # Default for new records

            if len(existing) > 0:
                processing_count = int(existing.docs[0].get("processing_count", 0)) + 1

            # Create or update record
            record = {
                "id": source_id,  # SOLR unique ID field
                "source_id": source_id,
                "content_hash": content_hash,
                "last_modified": time.time(),
                "processing_count": processing_count
            }

            self.history.add([record], commit=True)
            logger.debug(f"Updated processing history for {source_id}")

        except Exception as e:
            logger.error(f"Error updating processing history for {source_id}: {str(e)}")

    def store_document(self, document: Dict[str, Any], elements: List[Dict[str, Any]],
                       relationships: List[Dict[str, Any]]) -> None:
        """
        Store a document with its elements and relationships.

        Args:
            document: Document metadata
            elements: Document elements
            relationships: Element relationships
        """
        if not self.documents:
            raise ValueError("Database not initialized")

        source = document.get("source", "")
        content_hash = document.get("content_hash", "")

        # Check if document already exists with this source
        if source:
            existing_docs = self.documents.search(f"source:{source}", rows=1)
            if len(existing_docs) > 0:
                # Document exists, update it
                doc_id = existing_docs.docs[0]["doc_id"]
                document["doc_id"] = doc_id  # Use existing doc_id

                # Update all elements to use the existing doc_id
                for element in elements:
                    element["doc_id"] = doc_id

                self.update_document(doc_id, document, elements, relationships)
                return

        # New document, proceed with creation
        doc_id = document["doc_id"]

        try:
            # Prepare document for SOLR
            solr_document = {**document}
            solr_document["id"] = doc_id  # SOLR requires a unique 'id' field

            # Add timestamps
            solr_document["created_at"] = document.get("created_at", time.time())
            solr_document["updated_at"] = document.get("updated_at", time.time())

            # Convert metadata to JSON if it's a dict
            if isinstance(solr_document.get("metadata"), dict):
                solr_document["metadata_json"] = json.dumps(solr_document["metadata"])

            # Store document
            self.documents.add([solr_document])

            # Process elements
            solr_elements = []
            for element in elements:
                solr_element = {**element}

                # Generate element_pk if not present
                if "element_pk" not in solr_element:
                    self.element_pk_counter += 1
                    solr_element["element_pk"] = self.element_pk_counter
                    # Store back in original element
                    element["element_pk"] = solr_element["element_pk"]

                # Ensure element has a unique id for SOLR
                solr_element["id"] = solr_element["element_id"]

                # Extract full content if available
                # This will be indexed but not stored
                if "full_content" in element:
                    solr_element["full_text"] = element["full_content"]
                    # Don't store the full content
                    if "full_content" in solr_element:
                        del solr_element["full_content"]

                # Convert metadata to JSON if it's a dict
                if isinstance(solr_element.get("metadata"), dict):
                    solr_element["metadata_json"] = json.dumps(solr_element["metadata"])

                solr_elements.append(solr_element)

            # Store elements
            if solr_elements:
                self.elements.add(solr_elements)

            # Process relationships
            solr_relationships = []
            for rel in relationships:
                solr_rel = {**rel}

                # Ensure relationship has a unique id for SOLR
                solr_rel["id"] = solr_rel["relationship_id"]

                # Convert metadata to JSON if it's a dict
                if isinstance(solr_rel.get("metadata"), dict):
                    solr_rel["metadata_json"] = json.dumps(solr_rel["metadata"])

                solr_relationships.append(solr_rel)

            # Store relationships
            if solr_relationships:
                self.relationships.add(solr_relationships)

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

        Args:
            doc_id: Document ID
            document: Document metadata
            elements: Document elements
            relationships: Element relationships
        """
        if not self.documents:
            raise ValueError("Database not initialized")

        # Check if document exists
        existing_docs = self.documents.search(f"doc_id:{doc_id}", rows=1)
        if len(existing_docs) == 0:
            raise ValueError(f"Document not found: {doc_id}")

        try:
            # Update document timestamps
            document["updated_at"] = time.time()
            if "created_at" not in document:
                document["created_at"] = existing_docs.docs[0].get("created_at", time.time())

            # Prepare document for SOLR
            solr_document = {**document}
            solr_document["id"] = doc_id  # SOLR requires a unique 'id' field

            # Convert metadata to JSON if it's a dict
            if isinstance(solr_document.get("metadata"), dict):
                solr_document["metadata_json"] = json.dumps(solr_document["metadata"])

            # Delete existing document elements
            self.elements.delete(f"doc_id:{doc_id}")

            # Delete existing relationships for document elements
            element_ids = [f'"{element["element_id"]}"' for element in elements]
            if element_ids:
                element_ids_str = " OR ".join(element_ids)
                self.relationships.delete(f"source_id:({element_ids_str})")

            # Store updated document
            self.documents.add([solr_document])

            # Process elements
            solr_elements = []
            for element in elements:
                solr_element = {**element}

                # Generate element_pk if not present
                if "element_pk" not in solr_element:
                    self.element_pk_counter += 1
                    solr_element["element_pk"] = self.element_pk_counter
                    # Store back in original element
                    element["element_pk"] = solr_element["element_pk"]

                # Ensure element has a unique id for SOLR
                solr_element["id"] = solr_element["element_id"]

                # Extract full content if available
                # This will be indexed but not stored
                if "full_content" in element:
                    solr_element["full_text"] = element["full_content"]
                    # Don't store the full content
                    if "full_content" in solr_element:
                        del solr_element["full_content"]

                # Convert metadata to JSON if it's a dict
                if isinstance(solr_element.get("metadata"), dict):
                    solr_element["metadata_json"] = json.dumps(solr_element["metadata"])

                solr_elements.append(solr_element)

            # Store elements
            if solr_elements:
                self.elements.add(solr_elements)

            # Process relationships
            solr_relationships = []
            for rel in relationships:
                solr_rel = {**rel}

                # Ensure relationship has a unique id for SOLR
                solr_rel["id"] = solr_rel["relationship_id"]

                # Convert metadata to JSON if it's a dict
                if isinstance(solr_rel.get("metadata"), dict):
                    solr_rel["metadata_json"] = json.dumps(solr_rel["metadata"])

                solr_relationships.append(solr_rel)

            # Store relationships
            if solr_relationships:
                self.relationships.add(solr_relationships)

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
        """
        Get document metadata by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document metadata or None if not found
        """
        if not self.documents:
            raise ValueError("Database not initialized")

        try:
            # Try to get by doc_id
            results = self.documents.search(f"doc_id:{doc_id}", rows=1)

            if len(results) == 0:
                # Try to get by source field
                results = self.documents.search(f"source:{doc_id}", rows=1)

                if len(results) == 0:
                    return None

            # Convert SOLR doc to dict
            document = dict(results.docs[0])

            # Parse metadata_json if present
            if "metadata_json" in document and not document.get("metadata"):
                try:
                    document["metadata"] = json.loads(document["metadata_json"])
                except:
                    pass

            return document

        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {str(e)}")
            return None

    def get_document_elements(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get elements for a document.

        Args:
            doc_id: Document ID

        Returns:
            List of document elements
        """
        if not self.elements:
            raise ValueError("Database not initialized")

        try:
            # First try to get document by doc_id to handle case where source is provided
            document = self.get_document(doc_id)
            if document:
                doc_id = document["doc_id"]

            # Get elements
            results = self.elements.search(f"doc_id:{doc_id}", rows=10000)

            # Convert SOLR docs to dicts
            elements = []
            for doc in results.docs:
                element = dict(doc)

                # Parse metadata_json if present
                if "metadata_json" in element and not element.get("metadata"):
                    try:
                        element["metadata"] = json.loads(element["metadata_json"])
                    except:
                        pass

                elements.append(element)

            return elements

        except Exception as e:
            logger.error(f"Error getting document elements for {doc_id}: {str(e)}")
            return []

    def get_document_relationships(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get relationships for a document.

        Args:
            doc_id: Document ID

        Returns:
            List of document relationships
        """
        if not self.relationships or not self.elements:
            raise ValueError("Database not initialized")

        try:
            # Get all element IDs for this document
            elements = self.get_document_elements(doc_id)
            element_ids = [f'"{element["element_id"]}"' for element in elements]

            if not element_ids:
                return []

            # Find relationships involving these elements
            element_ids_str = " OR ".join(element_ids)
            results = self.relationships.search(f"source_id:({element_ids_str})", rows=10000)

            # Convert SOLR docs to dicts
            relationships = []
            for doc in results.docs:
                relationship = dict(doc)

                # Parse metadata_json if present
                if "metadata_json" in relationship and not relationship.get("metadata"):
                    try:
                        relationship["metadata"] = json.loads(relationship["metadata_json"])
                    except:
                        pass

                relationships.append(relationship)

            return relationships

        except Exception as e:
            logger.error(f"Error getting document relationships for {doc_id}: {str(e)}")
            return []

    def get_element(self, element_id_or_pk: Union[str, int]) -> Optional[Dict[str, Any]]:
        """
        Get element by ID or PK.

        Args:
            element_id_or_pk: Either the element_id (string) or element_pk (integer)

        Returns:
            Element data or None if not found
        """
        if not self.elements:
            raise ValueError("Database not initialized")

        try:
            # Try to interpret as element_pk (integer) first
            try:
                element_pk = int(element_id_or_pk)
                results = self.elements.search(f"element_pk:{element_pk}", rows=1)
            except (ValueError, TypeError):
                # If not an integer, treat as element_id (string)
                results = self.elements.search(f"element_id:{element_id_or_pk}", rows=1)

            if len(results) == 0:
                return None

            # Convert SOLR doc to dict
            element = dict(results.docs[0])

            # Parse metadata_json if present
            if "metadata_json" in element and not element.get("metadata"):
                try:
                    element["metadata"] = json.loads(element["metadata_json"])
                except:
                    pass

            return element

        except Exception as e:
            logger.error(f"Error getting element {element_id_or_pk}: {str(e)}")
            return None

    def find_documents(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find documents matching query.

        Args:
            query: Query parameters
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        if not self.documents:
            raise ValueError("Database not initialized")

        try:
            # Build SOLR query
            solr_query = "*:*"  # Default to all documents
            filter_queries = []

            if query:
                query_parts = []

                for key, value in query.items():
                    if key == "metadata":
                        # Handle metadata queries
                        for meta_key, meta_value in value.items():
                            # Use metadata_json for exact JSON structure
                            filter_queries.append(f'metadata_json:"*\\"{meta_key}\\":\\"{meta_value}\\"*"')
                    elif isinstance(value, list):
                        # Handle list values
                        values_str = " OR ".join([f'"{v}"' for v in value])
                        filter_queries.append(f"{key}:({values_str})")
                    else:
                        # Simple equality
                        query_parts.append(f'{key}:"{value}"')

                if query_parts:
                    solr_query = " AND ".join(query_parts)

            # Execute query
            params = {"rows": limit}
            if filter_queries:
                params["fq"] = filter_queries

            results = self.documents.search(solr_query, **params)

            # Convert SOLR docs to dicts
            documents = []
            for doc in results.docs:
                document = dict(doc)

                # Parse metadata_json if present
                if "metadata_json" in document and not document.get("metadata"):
                    try:
                        document["metadata"] = json.loads(document["metadata_json"])
                    except:
                        pass

                documents.append(document)

            return documents

        except Exception as e:
            logger.error(f"Error finding documents: {str(e)}")
            return []

    def find_elements(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements matching query.

        Args:
            query: Query parameters
            limit: Maximum number of results

        Returns:
            List of matching elements
        """
        if not self.elements:
            raise ValueError("Database not initialized")

        try:
            # Build SOLR query
            solr_query = "*:*"  # Default to all elements
            filter_queries = []

            if query:
                query_parts = []

                for key, value in query.items():
                    if key == "metadata":
                        # Handle metadata queries
                        for meta_key, meta_value in value.items():
                            # Use metadata_json for exact JSON structure
                            filter_queries.append(f'metadata_json:"*\\"{meta_key}\\":\\"{meta_value}\\"*"')
                    elif key == "element_type" and isinstance(value, list):
                        # Handle list of element types
                        values_str = " OR ".join([f'"{v}"' for v in value])
                        filter_queries.append(f"element_type:({values_str})")
                    elif isinstance(value, list):
                        # Handle other list values
                        values_str = " OR ".join([f'"{v}"' for v in value])
                        filter_queries.append(f"{key}:({values_str})")
                    else:
                        # Simple equality
                        query_parts.append(f'{key}:"{value}"')

                if query_parts:
                    solr_query = " AND ".join(query_parts)

            # Execute query
            params = {"rows": limit}
            if filter_queries:
                params["fq"] = filter_queries

            results = self.elements.search(solr_query, **params)

            # Convert SOLR docs to dicts
            elements = []
            for doc in results.docs:
                element = dict(doc)

                # Parse metadata_json if present
                if "metadata_json" in element and not element.get("metadata"):
                    try:
                        element["metadata"] = json.loads(element["metadata_json"])
                    except:
                        pass

                elements.append(element)

            return elements

        except Exception as e:
            logger.error(f"Error finding elements: {str(e)}")
            return []

    def search_elements_by_content(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search elements by content.

        Args:
            search_text: Text to search for
            limit: Maximum number of results

        Returns:
            List of matching elements
        """
        if not self.elements:
            raise ValueError("Database not initialized")

        try:
            # Build query to search both content_preview and full_text
            escaped_text = search_text.replace('"', '\\"')
            query = f'content_preview:"{escaped_text}" OR full_text:"{escaped_text}"'

            results = self.elements.search(query, rows=limit)

            # Convert SOLR docs to dicts
            elements = []
            for doc in results.docs:
                element = dict(doc)

                # Parse metadata_json if present
                if "metadata_json" in element and not element.get("metadata"):
                    try:
                        element["metadata"] = json.loads(element["metadata_json"])
                    except:
                        pass

                elements.append(element)

            return elements

        except Exception as e:
            logger.error(f"Error searching elements by content: {str(e)}")
            return []

    def store_embedding(self, element_pk: int, embedding: List[float]) -> None:
        """
        Store embedding for an element.

        Args:
            element_pk: Element ID
            embedding: Vector embedding
        """
        if not self.elements:
            raise ValueError("Database not initialized")

        try:
            # Get the element
            element = self.get_element(element_pk)
            if not element:
                raise ValueError(f"Element not found: {element_pk}")

            # Update the element with embedding
            element["embedding"] = embedding
            element["embedding_dimensions"] = len(embedding)

            # Add to SOLR (will replace existing document with same ID)
            self.elements.add([element])

            logger.debug(f"Stored embedding for element {element_pk}")

        except Exception as e:
            logger.error(f"Error storing embedding for element {element_pk}: {str(e)}")
            raise

    def get_embedding(self, element_pk: int) -> Optional[List[float]]:
        """
        Get embedding for an element.

        Args:
            element_pk: Element ID

        Returns:
            Vector embedding or None if not found
        """
        if not self.elements:
            raise ValueError("Database not initialized")

        try:
            element = self.get_element(element_pk)
            if not element:
                return None

            embedding = element.get("embedding")
            return embedding

        except Exception as e:
            logger.error(f"Error getting embedding for element {element_pk}: {str(e)}")
            return None

    def search_by_embedding(self, query_embedding: List[float], limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by embedding similarity with optional filtering.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_criteria: Optional dictionary with criteria to filter results
                             (e.g. {"element_type": ["header", "section"]})

        Returns:
            List of (element_pk, similarity_score) tuples for matching elements
        """
        if not self.elements:
            raise ValueError("Database not initialized")

        try:
            # Build KNN query
            vector_str = ",".join(str(v) for v in query_embedding)
            knn_query = f"{{!knn f=embedding topK={limit * 3}}} {vector_str}"

            # Add filter queries if needed
            params = {"rows": limit * 3}  # Get more results for better filtering

            if filter_criteria:
                fq = []
                for key, value in filter_criteria.items():
                    if key == "element_type" and isinstance(value, list):
                        # Handle list of element types
                        values_str = " OR ".join([f'"{v}"' for v in value])
                        fq.append(f"element_type:({values_str})")
                    elif key == "doc_id" and isinstance(value, list):
                        # Handle list of document IDs
                        values_str = " OR ".join([f'"{v}"' for v in value])
                        fq.append(f"doc_id:({values_str})")
                    elif key == "exclude_doc_id" and isinstance(value, list):
                        # Handle list of document IDs to exclude
                        values_str = " OR ".join([f'"{v}"' for v in value])
                        fq.append(f"-doc_id:({values_str})")
                    else:
                        # Simple equality
                        fq.append(f'{key}:"{value}"')

                if fq:
                    params["fq"] = fq

            # Execute query
            results = self.elements.search(knn_query, **params)

            # Format results as (element_pk, similarity_score) tuples
            # SOLR returns scores where higher is better
            element_scores = []
            for doc in results.docs:
                element_pk = int(doc["element_pk"])
                score = float(doc.get("score", 0.0))
                element_scores.append((element_pk, score))

            return element_scores[:limit]

        except Exception as e:
            logger.error(f"Error searching by embedding: {str(e)}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all associated elements and relationships.

        Args:
            doc_id: Document ID

        Returns:
            True if document was deleted, False otherwise
        """
        if not self.documents or not self.elements or not self.relationships:
            raise ValueError("Database not initialized")

        try:
            # Check if document exists
            document = self.get_document(doc_id)
            if not document:
                return False

            # Get all element IDs for this document
            elements = self.get_document_elements(doc_id)
            element_ids = [element["element_id"] for element in elements]

            # Delete relationships involving these elements
            if element_ids:
                element_ids_str = " OR ".join([f'"{eid}"' for eid in element_ids])
                self.relationships.delete(f"source_id:({element_ids_str})")

            # Delete elements
            self.elements.delete(f"doc_id:{doc_id}")

            # Delete document
            self.documents.delete(f"doc_id:{doc_id}")

            logger.info(f"Deleted document {doc_id} with {len(element_ids)} elements")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False

    def search_by_text(self, search_text: str, limit: int = 10,
                       filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by semantic similarity to the provided text.

        This method combines text-to-embedding conversion and embedding search
        into a single convenient operation. It implements a hybrid search approach
        that blends traditional text search with vector similarity search.

        Args:
            search_text: Text to search for semantically
            limit: Maximum number of results
            filter_criteria: Optional dictionary with criteria to filter results

        Returns:
            List of (element_pk, similarity_score) tuples
        """
        if not self.elements:
            raise ValueError("Database not initialized")

        try:
            # First, perform traditional text search
            escaped_text = search_text.replace('"', '\\"')
            text_query = f'content_preview:"{escaped_text}" OR full_text:"{escaped_text}"'

            # Add filter queries if needed
            params = {"rows": limit * 2}  # Get more results for better merging

            if filter_criteria:
                fq = []
                for key, value in filter_criteria.items():
                    if key == "element_type" and isinstance(value, list):
                        # Handle list of element types
                        values_str = " OR ".join([f'"{v}"' for v in value])
                        fq.append(f"element_type:({values_str})")
                    elif key == "doc_id" and isinstance(value, list):
                        # Handle list of document IDs
                        values_str = " OR ".join([f'"{v}"' for v in value])
                        fq.append(f"doc_id:({values_str})")
                    elif key == "exclude_doc_id" and isinstance(value, list):
                        # Handle list of document IDs to exclude
                        values_str = " OR ".join([f'"{v}"' for v in value])
                        fq.append(f"-doc_id:({values_str})")
                    else:
                        # Simple equality
                        fq.append(f'{key}:"{value}"')

                if fq:
                    params["fq"] = fq

            # Execute text search
            text_results = self.elements.search(text_query, **params)
            text_scores = {int(doc["element_pk"]): float(doc.get("score", 0.0))
                           for doc in text_results.docs}

            # If embedding generator available, also perform vector search
            vector_scores = {}
            try:
                # Import embedding generator on-demand if not already loaded
                if self.embedding_generator is None:
                    from ..embeddings import get_embedding_generator
                    from ..config import Config
                    # Try to get config from the module scope
                    config_instance = config or Config()
                    self.embedding_generator = get_embedding_generator(config_instance)

                # Generate embedding and perform vector search
                query_embedding = self.embedding_generator.generate(search_text)
                vector_results = self.search_by_embedding(query_embedding, limit, filter_criteria)
                vector_scores = {pk: score for pk, score in vector_results}

            except Exception as e:
                logger.warning(f"Vector search failed, falling back to text search: {str(e)}")

            # Merge results with a hybrid ranking strategy
            combined_scores = {}

            # Add text search results
            for pk, score in text_scores.items():
                combined_scores[pk] = {"text": score, "vector": 0.0}

            # Add vector search results
            for pk, score in vector_scores.items():
                if pk in combined_scores:
                    combined_scores[pk]["vector"] = score
                else:
                    combined_scores[pk] = {"text": 0.0, "vector": score}

            # Calculate final scores (weighted average)
            # Text weight: 0.3, Vector weight: 0.7
            results = []
            for pk, scores in combined_scores.items():
                # Normalize scores to account for different ranges
                text_score = scores["text"] / 10.0 if scores["text"] > 0 else 0  # SOLR text scores can be much higher
                vector_score = scores["vector"]

                # Calculate weighted score
                final_score = 0.3 * text_score + 0.7 * vector_score
                results.append((pk, final_score))

            # Sort by score (highest first) and limit results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Error in semantic search by text: {str(e)}")
            # Return empty list on error
            return []

    def get_outgoing_relationships(self, element_pk: int) -> List[ElementRelationship]:
        """
        Find all relationships where the specified element_pk is the source.

        Args:
            element_pk: The primary key of the element

        Returns:
            List of ElementRelationship objects where the specified element is the source
        """
        if not self.relationships or not self.elements:
            raise ValueError("Database not initialized")

        try:
            # Get the element to find its element_id
            element = self.get_element(element_pk)
            if not element:
                logger.warning(f"Element with PK {element_pk} not found")
                return []

            element_id = element.get("element_id")
            if not element_id:
                logger.warning(f"Element with PK {element_pk} has no element_id")
                return []

            element_type = element.get("element_type", "")

            # Search for relationships where this element is the source
            results = self.relationships.search(f'source_id:"{element_id}"', rows=10000)

            relationships = []
            for rel_doc in results.docs:
                # Get target element if it exists
                target_reference = rel_doc.get("target_reference", "")
                target_element = None
                target_element_pk = None
                target_element_type = None
                target_content_preview = None

                if target_reference:
                    target_element = self.get_element(target_reference)
                    if target_element:
                        target_element_pk = target_element.get("element_pk")
                        target_element_type = target_element.get("element_type")
                        target_content_preview = target_element.get("content_preview", "")

                # Parse metadata if it exists
                metadata = {}
                if "metadata_json" in rel_doc:
                    try:
                        metadata = json.loads(rel_doc["metadata_json"])
                    except:
                        metadata = rel_doc.get("metadata", {})

                # Create relationship object
                relationship = ElementRelationship(
                    relationship_id=rel_doc.get("relationship_id", ""),
                    source_id=element_id,
                    source_element_pk=element_pk,
                    source_element_type=element_type,
                    relationship_type=rel_doc.get("relationship_type", ""),
                    target_reference=target_reference,
                    target_element_pk=target_element_pk,
                    target_element_type=target_element_type,
                    target_content_preview=target_content_preview,
                    doc_id=rel_doc.get("doc_id"),
                    metadata=metadata,
                    is_source=True
                )

                relationships.append(relationship)

            return relationships

        except Exception as e:
            logger.error(f"Error getting outgoing relationships for element {element_pk}: {str(e)}")
            return []
