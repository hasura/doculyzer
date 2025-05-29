import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple, Union, TYPE_CHECKING

import time

# Import types for type checking only
if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    # Define type aliases for type checking
    VectorType = NDArray[np.float32]
else:
    # Runtime type aliases - use generic Python types
    VectorType = List[float]

from .element_relationship import ElementRelationship
from .base import DocumentDatabase
from .element_element import ElementType, ElementBase  # Import existing enum and ElementBase

logger = logging.getLogger(__name__)

# Define global flags for availability - will be set at runtime
PYSOLR_AVAILABLE = False
NUMPY_AVAILABLE = False

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

# Try to import NumPy conditionally
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("NumPy not available. Will use slower pure Python vector operations.")

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
        self.embeddings_core = f"{self.core_prefix}_embeddings"

        # Initialize SOLR clients to None - will be created in initialize()
        self.documents = None
        self.elements = None
        self.relationships = None
        self.history = None
        self.embeddings = None

        # Auto-increment counters
        self.element_pk_counter = 0

        # Configuration for vector search
        self.vector_dimension = conn_params.get('vector_dimension', 384)
        if config:
            self.vector_dimension = config.config.get('embedding', {}).get('dimensions', self.vector_dimension)

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
            self.embeddings = pysolr.Solr(f"{self.base_url}/{self.embeddings_core}", always_commit=True)

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
        self.embeddings = None

    def get_last_processed_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about when a document was last processed."""
        if not self.history:
            raise ValueError("Database not initialized")

        try:
            escaped_source = self._escape_solr_query(source_id)
            results = self.history.search(f"source_id:{escaped_source}", rows=1)
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
            escaped_source = self._escape_solr_query(source_id)
            existing = self.history.search(f"source_id:{escaped_source}", rows=1)
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
            escaped_source = self._escape_solr_query(source)
            existing_docs = self.documents.search(f"source:{escaped_source}", rows=1)
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
            solr_document = {**document, "id": doc_id, "created_at": document.get("created_at", time.time()),
                             "updated_at": document.get("updated_at", time.time())}

            # Add timestamps

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
        escaped_doc_id = self._escape_solr_query(doc_id)
        existing_docs = self.documents.search(f"doc_id:{escaped_doc_id}", rows=1)
        if len(existing_docs) == 0:
            raise ValueError(f"Document not found: {doc_id}")

        try:
            # Update document timestamps
            document["updated_at"] = time.time()
            if "created_at" not in document:
                document["created_at"] = existing_docs.docs[0].get("created_at", time.time())

            # Prepare document for SOLR
            solr_document = {**document, "id": doc_id}

            # Convert metadata to JSON if it's a dict
            if isinstance(solr_document.get("metadata"), dict):
                solr_document["metadata_json"] = json.dumps(solr_document["metadata"])

            # Get existing elements to clean up embeddings
            existing_elements = self.get_document_elements(doc_id)
            existing_element_pks = [int(elem.get("element_pk", 0)) for elem in existing_elements]

            # Delete existing document elements
            self.elements.delete(f"doc_id:{escaped_doc_id}")

            # Delete existing embeddings for document elements
            if existing_element_pks:
                element_pks_str = " OR ".join([str(pk) for pk in existing_element_pks])
                self.embeddings.delete(f"element_pk:({element_pks_str})")

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
            escaped_doc_id = self._escape_solr_query(doc_id)
            results = self.documents.search(f"doc_id:{escaped_doc_id}", rows=1)

            if len(results) == 0:
                # Try to get by source field
                escaped_source = self._escape_solr_query(doc_id)
                results = self.documents.search(f"source:{escaped_source}", rows=1)

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
            escaped_doc_id = self._escape_solr_query(doc_id)
            results = self.elements.search(f"doc_id:{escaped_doc_id}", rows=10000)

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
            element_ids = [self._escape_solr_query(element["element_id"]) for element in elements]

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
                escaped_element_id = self._escape_solr_query(element_id_or_pk)
                results = self.elements.search(f"element_id:{escaped_element_id}", rows=1)

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
        Find documents matching query with support for LIKE patterns.

        Args:
            query: Query parameters. Enhanced syntax supports:
                   - Exact matches: {"doc_type": "pdf"}
                   - LIKE patterns: {"source_like": "%reports%"} (converted to wildcard)
                   - Case-insensitive LIKE: {"source_ilike": "%REPORTS%"}
                   - List matching: {"doc_type": ["pdf", "docx"]}
                   - Metadata exact: {"metadata": {"author": "John"}}
                   - Metadata LIKE: {"metadata_like": {"title": "%annual%"}}
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
                        # Handle metadata exact matches
                        for meta_key, meta_value in value.items():
                            # Use metadata_json for exact JSON structure
                            escaped_key = self._escape_solr_query(meta_key)
                            escaped_value = self._escape_solr_query(str(meta_value))
                            filter_queries.append(f'metadata_json:"*\\"{escaped_key}\\":\\"{escaped_value}\\"*"')
                    elif key == "metadata_like":
                        # Handle metadata LIKE patterns
                        for meta_key, meta_value in value.items():
                            pattern = self._convert_like_to_wildcard(meta_value)
                            escaped_key = self._escape_solr_query(meta_key)
                            filter_queries.append(f'metadata_json:*\\"{escaped_key}\\"\\:*{pattern}*')
                    elif key == "metadata_ilike":
                        # Handle case-insensitive metadata LIKE patterns (same as metadata_like in SOLR)
                        for meta_key, meta_value in value.items():
                            pattern = self._convert_like_to_wildcard(meta_value)
                            escaped_key = self._escape_solr_query(meta_key)
                            filter_queries.append(f'metadata_json:*\\"{escaped_key}\\"\\:*{pattern}*')
                    elif key.endswith("_ilike"):
                        # Case-insensitive LIKE pattern
                        field_name = key[:-6]  # Remove '_ilike' suffix
                        pattern = self._convert_like_to_wildcard(value)
                        filter_queries.append(f"{field_name}:{pattern}")
                    elif key.endswith("_like"):
                        # LIKE pattern for regular fields
                        field_name = key[:-5]  # Remove '_like' suffix
                        pattern = self._convert_like_to_wildcard(value)
                        filter_queries.append(f"{field_name}:{pattern}")
                    elif isinstance(value, list):
                        # Handle list values
                        escaped_values = [self._escape_solr_query(str(v)) for v in value]
                        values_str = " OR ".join(escaped_values)
                        filter_queries.append(f"{key}:({values_str})")
                    else:
                        # Simple equality
                        escaped_value = self._escape_solr_query(str(value))
                        query_parts.append(f'{key}:{escaped_value}')

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
        Find elements matching query with support for LIKE patterns and ElementType enums.

        Args:
            query: Query parameters. Enhanced syntax supports:
                   - Exact matches: {"element_type": "header"}
                   - ElementType enums: {"element_type": ElementType.HEADER}
                   - Multiple enums: {"element_type": [ElementType.HEADER, ElementType.PARAGRAPH]}
                   - LIKE patterns: {"content_preview_like": "%important%"}
                   - Case-insensitive LIKE: {"content_preview_ilike": "%IMPORTANT%"}
                   - List matching: {"doc_id": ["doc1", "doc2"]}
                   - Metadata exact: {"metadata": {"section": "intro"}}
                   - Metadata LIKE: {"metadata_like": {"title": "%chapter%"}}
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
                        # Handle metadata exact matches
                        for meta_key, meta_value in value.items():
                            # Use metadata_json for exact JSON structure
                            escaped_key = self._escape_solr_query(meta_key)
                            escaped_value = self._escape_solr_query(str(meta_value))
                            filter_queries.append(f'metadata_json:"*\\"{escaped_key}\\":\\"{escaped_value}\\"*"')
                    elif key == "metadata_like":
                        # Handle metadata LIKE patterns
                        for meta_key, meta_value in value.items():
                            pattern = self._convert_like_to_wildcard(meta_value)
                            escaped_key = self._escape_solr_query(meta_key)
                            filter_queries.append(f'metadata_json:*\\"{escaped_key}\\"\\:*{pattern}*')
                    elif key == "metadata_ilike":
                        # Handle case-insensitive metadata LIKE patterns (same as metadata_like in SOLR)
                        for meta_key, meta_value in value.items():
                            pattern = self._convert_like_to_wildcard(meta_value)
                            escaped_key = self._escape_solr_query(meta_key)
                            filter_queries.append(f'metadata_json:*\\"{escaped_key}\\"\\:*{pattern}*')
                    elif key.endswith("_ilike"):
                        # Case-insensitive LIKE pattern
                        field_name = key[:-6]  # Remove '_ilike' suffix
                        pattern = self._convert_like_to_wildcard(value)
                        filter_queries.append(f"{field_name}:{pattern}")
                    elif key.endswith("_like"):
                        # LIKE pattern for regular fields
                        field_name = key[:-5]  # Remove '_like' suffix
                        pattern = self._convert_like_to_wildcard(value)
                        filter_queries.append(f"{field_name}:{pattern}")
                    elif key == "element_type":
                        # Handle ElementType enums, strings, and lists
                        type_values = self._prepare_element_type_query(value)
                        if type_values:
                            if len(type_values) == 1:
                                escaped_value = self._escape_solr_query(type_values[0])
                                filter_queries.append(f"element_type:{escaped_value}")
                            else:
                                escaped_values = [self._escape_solr_query(v) for v in type_values]
                                values_str = " OR ".join(escaped_values)
                                filter_queries.append(f"element_type:({values_str})")
                    elif isinstance(value, list):
                        # Handle other list values
                        escaped_values = [self._escape_solr_query(str(v)) for v in value]
                        values_str = " OR ".join(escaped_values)
                        filter_queries.append(f"{key}:({values_str})")
                    else:
                        # Simple equality
                        escaped_value = self._escape_solr_query(str(value))
                        query_parts.append(f'{key}:{escaped_value}')

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
            escaped_text = self._escape_solr_query(search_text)
            query = f'content_preview:{escaped_text} OR full_text:{escaped_text}'

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

            # Get all elements for this document to clean up embeddings
            elements = self.get_document_elements(doc_id)
            element_pks = [int(elem.get("element_pk", 0)) for elem in elements]
            element_ids = [elem["element_id"] for elem in elements]

            # Delete embeddings for these elements
            if element_pks:
                element_pks_str = " OR ".join([str(pk) for pk in element_pks])
                self.embeddings.delete(f"element_pk:({element_pks_str})")

            # Delete relationships involving these elements
            if element_ids:
                escaped_element_ids = [self._escape_solr_query(eid) for eid in element_ids]
                element_ids_str = " OR ".join(escaped_element_ids)
                self.relationships.delete(f"source_id:({element_ids_str})")

            # Delete elements
            escaped_doc_id = self._escape_solr_query(doc_id)
            self.elements.delete(f"doc_id:{escaped_doc_id}")

            # Delete document
            self.documents.delete(f"doc_id:{escaped_doc_id}")

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
        if not self.relationships:
            raise ValueError("Database not initialized")

        try:
            # Prepare relationship for SOLR
            solr_rel = {**relationship, "id": relationship["relationship_id"]}

            # Convert metadata to JSON if it's a dict
            if isinstance(solr_rel.get("metadata"), dict):
                solr_rel["metadata_json"] = json.dumps(solr_rel["metadata"])

            # Store relationship
            self.relationships.add([solr_rel])
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
        if not self.relationships:
            raise ValueError("Database not initialized")

        try:
            # Build query for source relationships
            escaped_element_id = self._escape_solr_query(element_id)
            source_query = f'source_id:{escaped_element_id}'
            if relationship_type:
                escaped_rel_type = self._escape_solr_query(relationship_type)
                source_query += f' AND relationship_type:{escaped_rel_type}'

            # Build query for target relationships
            target_query = f'target_reference:{escaped_element_id}'
            if relationship_type:
                escaped_rel_type = self._escape_solr_query(relationship_type)
                target_query += f' AND relationship_type:{escaped_rel_type}'

            # Delete source relationships
            self.relationships.delete(source_query)

            # Delete target relationships
            self.relationships.delete(target_query)

            logger.debug(f"Deleted relationships for element {element_id}")

        except Exception as e:
            logger.error(f"Error deleting relationships for element {element_id}: {str(e)}")
            raise

    # ========================================
    # ENHANCED EMBEDDING FUNCTIONS
    # ========================================

    def store_embedding(self, element_pk: int, embedding: VectorType) -> None:
        """
        Store embedding for an element.

        Args:
            element_pk: Element ID
            embedding: Vector embedding
        """
        if not self.embeddings:
            raise ValueError("Database not initialized")

        try:
            # Verify element exists
            element = self.get_element(element_pk)
            if not element:
                raise ValueError(f"Element not found: {element_pk}")

            # Create enhanced embedding document
            embedding_doc = {
                "id": str(element_pk),  # SOLR unique ID
                "element_pk": element_pk,
                "embedding": embedding,
                "dimensions": len(embedding),
                "topics": [],  # Default to empty topics
                "confidence": 1.0,  # Default confidence
                "created_at": time.time()
            }

            # Store in embeddings core
            self.embeddings.add([embedding_doc])
            logger.debug(f"Stored embedding for element {element_pk}")

        except Exception as e:
            logger.error(f"Error storing embedding for element {element_pk}: {str(e)}")
            raise

    def get_embedding(self, element_pk: int) -> Optional[VectorType]:
        """
        Get embedding for an element.

        Args:
            element_pk: Element ID

        Returns:
            Vector embedding or None if not found
        """
        if not self.embeddings:
            raise ValueError("Database not initialized")

        try:
            results = self.embeddings.search(f"element_pk:{element_pk}", rows=1)
            if len(results) == 0:
                return None

            embedding_doc = dict(results.docs[0])
            return embedding_doc.get("embedding")

        except Exception as e:
            logger.error(f"Error getting embedding for element {element_pk}: {str(e)}")
            return None

    def search_by_embedding(self, query_embedding: VectorType, limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by embedding similarity with optional filtering.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_criteria: Optional dictionary with criteria to filter results

        Returns:
            List of (element_pk, similarity_score) tuples for matching elements
        """
        if not self.embeddings:
            raise ValueError("Database not initialized")

        try:
            # For SOLR, we'll fetch all embeddings and compute similarity in Python
            # since SOLR's vector search capabilities vary by version
            return self._fallback_embedding_search(query_embedding, limit, filter_criteria)

        except Exception as e:
            logger.error(f"Error searching by embedding: {str(e)}")
            return []

    def _fallback_embedding_search(self, query_embedding: VectorType, limit: int = 10,
                                   filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Fallback implementation for embedding search using Python similarity calculation.
        """
        try:
            # Build SOLR query to get embeddings
            solr_query = "*:*"
            filter_queries = []

            # Add element filters if provided
            if filter_criteria:
                # Get element IDs that match the filter criteria
                matching_elements = self.find_elements(filter_criteria, limit=10000)
                if not matching_elements:
                    return []

                element_pks = [int(elem["element_pk"]) for elem in matching_elements]
                element_pks_str = " OR ".join([str(pk) for pk in element_pks])
                filter_queries.append(f"element_pk:({element_pks_str})")

            # Execute query to get all embeddings
            params = {"rows": 10000}  # Get large number for better results
            if filter_queries:
                params["fq"] = filter_queries

            results = self.embeddings.search(solr_query, **params)

            # Calculate similarities in Python
            similarities = []
            for doc in results.docs:
                element_pk = int(doc["element_pk"])
                embedding = doc.get("embedding", [])

                if not embedding:
                    continue

                try:
                    # Calculate cosine similarity
                    if NUMPY_AVAILABLE:
                        similarity = self._cosine_similarity_numpy(query_embedding, embedding)
                    else:
                        similarity = self._cosine_similarity_python(query_embedding, embedding)

                    similarities.append((element_pk, similarity))
                except Exception as e:
                    logger.warning(f"Error calculating similarity for element {element_pk}: {str(e)}")

            # Sort by similarity (highest first) and limit results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]

        except Exception as e:
            logger.error(f"Error in fallback embedding search: {str(e)}")
            return []

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
            escaped_text = self._escape_solr_query(search_text)
            text_query = f'content_preview:{escaped_text} OR full_text:{escaped_text}'

            # Add filter queries if needed
            params = {"rows": limit * 2}  # Get more results for better merging

            if filter_criteria:
                fq = []
                for key, value in filter_criteria.items():
                    if key == "element_type" and isinstance(value, list):
                        # Handle list of element types
                        escaped_values = [self._escape_solr_query(v) for v in value]
                        values_str = " OR ".join(escaped_values)
                        fq.append(f"element_type:({values_str})")
                    elif key == "doc_id" and isinstance(value, list):
                        # Handle list of document IDs
                        escaped_values = [self._escape_solr_query(v) for v in value]
                        values_str = " OR ".join(escaped_values)
                        fq.append(f"doc_id:({values_str})")
                    elif key == "exclude_doc_id" and isinstance(value, list):
                        # Handle list of document IDs to exclude
                        escaped_values = [self._escape_solr_query(v) for v in value]
                        values_str = " OR ".join(escaped_values)
                        fq.append(f"-doc_id:({values_str})")
                    else:
                        # Simple equality
                        escaped_value = self._escape_solr_query(str(value))
                        fq.append(f'{key}:{escaped_value}')

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
            escaped_element_id = self._escape_solr_query(element_id)
            results = self.relationships.search(f'source_id:{escaped_element_id}', rows=10000)

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

    # ========================================
    # NEW: ENHANCED SEARCH HELPER METHODS
    # ========================================

    @staticmethod
    def supports_like_patterns() -> bool:
        """
        Indicate whether this backend supports LIKE pattern matching.

        Returns:
            True since SOLR supports wildcard queries
        """
        return True

    @staticmethod
    def supports_case_insensitive_like() -> bool:
        """
        Indicate whether this backend supports case-insensitive LIKE (ILIKE).

        Returns:
            True since SOLR is case-insensitive by default
        """
        return True

    @staticmethod
    def supports_element_type_enums() -> bool:
        """
        Indicate whether this backend supports ElementType enum integration.

        Returns:
            True since SOLR implementation supports ElementType enums
        """
        return True

    def _prepare_element_type_query(self, element_types: Union[
        ElementType,
        List[ElementType],
        str,
        List[str],
        None
    ]) -> Optional[List[str]]:
        """
        Prepare element type values for queries using existing ElementType enum.

        Args:
            element_types: ElementType enum(s), string(s), or None

        Returns:
            List of string values for query, or None
        """
        if element_types is None:
            return None

        if isinstance(element_types, ElementType):
            return [element_types.value]
        elif isinstance(element_types, str):
            return [element_types]
        elif isinstance(element_types, list):
            result = []
            for et in element_types:
                if isinstance(et, ElementType):
                    result.append(et.value)
                elif isinstance(et, str):
                    result.append(et)
            return result if result else None

        return None

    @staticmethod
    def get_element_types_by_category() -> Dict[str, List[ElementType]]:
        """
        Get categorized lists of ElementType enums.

        Returns:
            Dictionary with categorized element types
        """
        return {
            "text_elements": [
                ElementType.HEADER,
                ElementType.PARAGRAPH,
                ElementType.BLOCKQUOTE,
                ElementType.TEXT_BOX
            ],

            "structural_elements": [
                ElementType.ROOT,
                ElementType.PAGE,
                ElementType.BODY,
                ElementType.PAGE_HEADER,
                ElementType.PAGE_FOOTER
            ],

            "list_elements": [
                ElementType.LIST,
                ElementType.LIST_ITEM
            ],

            "table_elements": [
                ElementType.TABLE,
                ElementType.TABLE_ROW,
                ElementType.TABLE_HEADER_ROW,
                ElementType.TABLE_CELL,
                ElementType.TABLE_HEADER
            ],

            "media_elements": [
                ElementType.IMAGE,
                ElementType.CHART,
                ElementType.SHAPE,
                ElementType.SHAPE_GROUP
            ],

            "code_elements": [
                ElementType.CODE_BLOCK
            ],

            "presentation_elements": [
                ElementType.SLIDE,
                ElementType.SLIDE_NOTES,
                ElementType.PRESENTATION_BODY,
                ElementType.SLIDE_MASTERS,
                ElementType.SLIDE_TEMPLATES,
                ElementType.SLIDE_LAYOUT,
                ElementType.SLIDE_MASTER
            ],

            "data_elements": [
                ElementType.JSON_OBJECT,
                ElementType.JSON_ARRAY,
                ElementType.JSON_FIELD,
                ElementType.JSON_ITEM
            ],

            "xml_elements": [
                ElementType.XML_ELEMENT,
                ElementType.XML_TEXT,
                ElementType.XML_LIST,
                ElementType.XML_OBJECT
            ]
        }

    def find_elements_by_category(self, category: str, **other_filters) -> List[Dict[str, Any]]:
        """
        Find elements by predefined category using ElementType enums.

        Args:
            category: Category name from get_element_types_by_category()
            **other_filters: Additional filter criteria

        Returns:
            List of matching elements

        Examples:
            find_elements_by_category("text_elements")
            find_elements_by_category("table_elements", content_preview_like="%data%")
        """
        categories = self.get_element_types_by_category()

        if category not in categories:
            available = list(categories.keys())
            raise ValueError(f"Unknown category: {category}. Available: {available}")

        element_types = categories[category]
        query = {"element_type": element_types}
        query.update(other_filters)

        return self.find_elements(query)

    def find_elements_ilike(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements with case-insensitive LIKE support.

        SOLR is case-insensitive by default.

        Args:
            query: Query parameters with _ilike suffix support
            limit: Maximum number of results

        Returns:
            List of matching elements
        """
        # SOLR is case-insensitive by default, so just use regular find_elements
        return self.find_elements(query, limit)

    def _convert_like_to_wildcard(self, like_pattern: str) -> str:
        """
        Convert SQL LIKE pattern to SOLR wildcard pattern.

        Args:
            like_pattern: SQL LIKE pattern (e.g., "%abc%", "abc_def")

        Returns:
            SOLR wildcard pattern
        """
        # Convert % to * (match any characters)
        pattern = like_pattern.replace('%', '*')
        # Convert _ to ? (match single character)
        pattern = pattern.replace('_', '?')
        return pattern

    def _escape_solr_query(self, query_string: str) -> str:
        """
        Escape special SOLR characters in query string.

        Args:
            query_string: String to escape

        Returns:
            Escaped string safe for SOLR queries
        """
        # SOLR special characters that need escaping
        # + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
        special_chars = ['+', '-', '&', '|', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\', '/']

        escaped = query_string
        for char in special_chars:
            escaped = escaped.replace(char, f'\\{char}')

        return f'"{escaped}"'

    # ========================================
    # TOPIC SUPPORT METHODS (Enhanced)
    # ========================================

    def supports_topics(self) -> bool:
        """
        Indicate whether this backend supports topic-aware embeddings.

        Returns:
            True since SOLR implementation now supports topics
        """
        return True

    def store_embedding_with_topics(self, element_pk: int, embedding: VectorType,
                                    topics: List[str], confidence: float = 1.0) -> None:
        """
        Store embedding for an element with topic assignments.

        Args:
            element_pk: Element primary key
            embedding: Vector embedding
            topics: List of topic strings (e.g., ['security.policy', 'compliance'])
            confidence: Overall confidence in this embedding/topic assignment
        """
        if not self.embeddings:
            raise ValueError("Database not initialized")

        try:
            # Verify element exists
            element = self.get_element(element_pk)
            if not element:
                raise ValueError(f"Element not found: {element_pk}")

            # Create enhanced embedding document
            embedding_doc = {
                "id": str(element_pk),  # SOLR unique ID
                "element_pk": element_pk,
                "embedding": embedding,
                "dimensions": len(embedding),
                "topics": topics,  # SOLR can handle multi-valued fields
                "topics_json": json.dumps(topics),  # Also store as JSON for complex queries
                "confidence": confidence,
                "created_at": time.time()
            }

            # Store in embeddings core
            self.embeddings.add([embedding_doc])
            logger.debug(f"Stored embedding with topics for element {element_pk}")

        except Exception as e:
            logger.error(f"Error storing embedding with topics for element {element_pk}: {str(e)}")
            raise

    def search_by_text_and_topics(self, search_text: str = None,
                                  include_topics: Optional[List[str]] = None,
                                  exclude_topics: Optional[List[str]] = None,
                                  min_confidence: float = 0.7,
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search elements by text with topic filtering using pattern matching.

        Args:
            search_text: Text to search for semantically (optional)
            include_topics: Topic patterns to include (e.g., ['security*', '*.policy*'])
            exclude_topics: Topic patterns to exclude (e.g., ['deprecated*'])
            min_confidence: Minimum confidence threshold for embeddings
            limit: Maximum number of results

        Returns:
            List of dictionaries with keys:
            - element_pk: Element primary key
            - similarity: Similarity score (if search_text provided)
            - confidence: Overall embedding confidence
            - topics: List of assigned topic strings
        """
        if not self.embeddings:
            raise ValueError("Database not initialized")

        try:
            # Generate embedding for search text if provided
            query_embedding = None
            if search_text:
                if self.embedding_generator is None:
                    from ..embeddings import get_embedding_generator
                    config_instance = config or Config()
                    self.embedding_generator = get_embedding_generator(config_instance)

                query_embedding = self.embedding_generator.generate(search_text)

            return self._search_by_text_and_topics_fallback(
                query_embedding, include_topics, exclude_topics, min_confidence, limit
            )

        except Exception as e:
            logger.error(f"Error in topic-aware search: {str(e)}")
            return []

    def _search_by_text_and_topics_fallback(self, query_embedding: Optional[VectorType] = None,
                                            include_topics: Optional[List[str]] = None,
                                            exclude_topics: Optional[List[str]] = None,
                                            min_confidence: float = 0.7,
                                            limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback search using Python similarity calculation with topic filtering."""

        try:
            # Build SOLR query with confidence filtering
            solr_query = f"confidence:[{min_confidence} TO *]"
            filter_queries = []

            # Add topic filtering
            if include_topics:
                # Convert patterns to SOLR wildcard patterns
                include_patterns = []
                for pattern in include_topics:
                    wildcard_pattern = self._convert_like_to_wildcard(pattern)
                    include_patterns.append(f"topics:{wildcard_pattern}")
                if include_patterns:
                    filter_queries.append(f"({' OR '.join(include_patterns)})")

            if exclude_topics:
                # Convert patterns to SOLR wildcard patterns
                exclude_patterns = []
                for pattern in exclude_topics:
                    wildcard_pattern = self._convert_like_to_wildcard(pattern)
                    exclude_patterns.append(f"-topics:{wildcard_pattern}")
                if exclude_patterns:
                    filter_queries.extend(exclude_patterns)

            # Execute query to get all embeddings above confidence threshold
            params = {"rows": 10000}
            if filter_queries:
                params["fq"] = filter_queries

            results = self.embeddings.search(solr_query, **params)

            # Process results in Python
            filtered_results = []
            for doc in results.docs:
                element_pk = int(doc["element_pk"])
                embedding = doc.get("embedding", [])
                confidence = float(doc.get("confidence", 1.0))

                # Parse topics
                topics = []
                if "topics" in doc:
                    topics = doc["topics"] if isinstance(doc["topics"], list) else [doc["topics"]]
                elif "topics_json" in doc:
                    try:
                        topics = json.loads(doc["topics_json"])
                    except:
                        topics = []

                result_dict = {
                    'element_pk': element_pk,
                    'confidence': confidence,
                    'topics': topics
                }

                # Calculate similarity if we have a query embedding
                if query_embedding and embedding:
                    try:
                        if NUMPY_AVAILABLE:
                            similarity = self._cosine_similarity_numpy(query_embedding, embedding)
                        else:
                            similarity = self._cosine_similarity_python(query_embedding, embedding)
                        result_dict['similarity'] = float(similarity)
                    except Exception as e:
                        logger.warning(f"Error calculating similarity for element {element_pk}: {str(e)}")
                        result_dict['similarity'] = 0.0
                else:
                    result_dict['similarity'] = 1.0  # No text search, all results have equal similarity

                filtered_results.append(result_dict)

            # Sort by similarity if we calculated it
            if query_embedding:
                filtered_results.sort(key=lambda x: x['similarity'], reverse=True)

            return filtered_results[:limit]

        except Exception as e:
            logger.error(f"Error in fallback topic search: {str(e)}")
            return []

    def get_topic_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about topic distribution across embeddings.

        Returns:
            Dictionary mapping topic strings to statistics
        """
        if not self.embeddings:
            raise ValueError("Database not initialized")

        try:
            # Get all embeddings with topics
            results = self.embeddings.search("topics:[* TO *]", rows=10000)

            topic_stats = {}
            for doc in results.docs:
                confidence = float(doc.get("confidence", 1.0))

                # Parse topics
                topics = []
                if "topics" in doc:
                    topics = doc["topics"] if isinstance(doc["topics"], list) else [doc["topics"]]
                elif "topics_json" in doc:
                    try:
                        topics = json.loads(doc["topics_json"])
                    except:
                        topics = []

                # Get document ID for this element
                element_pk = int(doc["element_pk"])
                element = self.get_element(element_pk)
                doc_id = element.get("doc_id") if element else None

                for topic in topics:
                    if topic not in topic_stats:
                        topic_stats[topic] = {
                            'embedding_count': 0,
                            'document_ids': set(),
                            'confidences': []
                        }

                    topic_stats[topic]['embedding_count'] += 1
                    topic_stats[topic]['confidences'].append(confidence)
                    if doc_id:
                        topic_stats[topic]['document_ids'].add(doc_id)

            # Calculate final statistics
            final_stats = {}
            for topic, stats in topic_stats.items():
                final_stats[topic] = {
                    'embedding_count': stats['embedding_count'],
                    'document_count': len(stats['document_ids']),
                    'avg_embedding_confidence': sum(stats['confidences']) / len(stats['confidences'])
                }

            return final_stats

        except Exception as e:
            logger.error(f"Error getting topic statistics: {str(e)}")
            return {}

    def get_embedding_topics(self, element_pk: int) -> List[str]:
        """
        Get topics assigned to a specific embedding.

        Args:
            element_pk: Element primary key

        Returns:
            List of topic strings assigned to this embedding
        """
        if not self.embeddings:
            raise ValueError("Database not initialized")

        try:
            results = self.embeddings.search(f"element_pk:{element_pk}", rows=1)
            if len(results) == 0:
                return []

            doc = results.docs[0]

            # Parse topics
            if "topics" in doc:
                topics = doc["topics"]
                return topics if isinstance(topics, list) else [topics]
            elif "topics_json" in doc:
                try:
                    return json.loads(doc["topics_json"])
                except:
                    return []

            return []

        except Exception as e:
            logger.error(f"Error getting topics for element {element_pk}: {str(e)}")
            return []

    # ========================================
    # HIERARCHY METHODS
    # ========================================

    def get_results_outline(self, elements: List[Tuple[int, float]]) -> List["ElementHierarchical"]:
        """
        For an arbitrary list of element pk search results, finds the root node of the source, and each
        ancestor element, to create a root -> element array of arrays like this:
        [(<parent element>, score, [children])]

        (Note score is None if the element was not in the results param)

        Then each additional element is analyzed, its hierarchy materialized, and merged into
        the final result.
        """
        from .element_element import ElementHierarchical

        # Dictionary to store element_pk -> score mapping for quick lookup
        element_scores = {element_pk: score for element_pk, score in elements}

        # Set to track processed element_pks to avoid duplicates
        processed_elements = set()

        # Final result structure
        result_tree: List[ElementHierarchical] = []

        # Process each element from the search results
        for element_pk, score in elements:
            if element_pk in processed_elements:
                continue

            # Find the complete ancestry path for this element
            ancestry_path = self._get_element_ancestry_path(element_pk)

            if not ancestry_path:
                continue

            # Mark this element as processed
            processed_elements.add(element_pk)

            # Start with the root level
            current_level = result_tree

            # Process each ancestor from root to the target element
            for i, ancestor in enumerate(ancestry_path):
                ancestor_pk = ancestor.element_pk

                # Check if this ancestor is already in the current level
                existing_idx = None
                for idx, existing_element in enumerate(current_level):
                    if existing_element.element_pk == ancestor_pk:
                        existing_idx = idx
                        break

                if existing_idx is not None:
                    # Ancestor exists, get its children
                    current_level = current_level[existing_idx].child_elements  # Get children list
                else:
                    # Ancestor doesn't exist, add it with its score (or None if not in search results)
                    ancestor_score = element_scores.get(ancestor_pk)
                    children = []
                    ancestor.score = ancestor_score
                    h_ancestor = ancestor.to_hierarchical()
                    h_ancestor.child_elements = children
                    current_level.append(h_ancestor)
                    current_level = children

        return result_tree

    def _get_element_ancestry_path(self, element_pk: int) -> List[ElementBase]:
        """
        Get the complete ancestry path for an element, from root to the element itself.

        Uses parent_id to find parents instead of relationships.
        """
        # Get the element
        element_dict = self.get_element(element_pk)
        if not element_dict:
            return []

        # Convert to ElementBase instance
        element = ElementBase(**element_dict)

        # Start building the ancestry path with the element itself
        ancestry = [element]

        # Track to avoid circular references
        visited = {element_pk}

        # Current element to process
        current_pk = element_pk

        # Traverse up the hierarchy using parent_id
        while True:
            # Get the current element
            current_element = self.get_element(current_pk)
            if not current_element:
                break

            # Get parent ID
            parent_id = current_element.get('parent_id')
            if not parent_id:
                break

            # Get the parent element
            parent_dict = self.get_element(parent_id)
            if not parent_dict:
                break

            # Check for circular references
            parent_pk = parent_dict.get('id') or parent_dict.get('pk') or parent_dict.get('element_id')
            if parent_pk in visited:
                break

            # Convert to ElementBase
            parent = ElementBase(**parent_dict)

            # Add to visited set
            visited.add(parent_pk)

            # Add parent to the beginning of the ancestry list (root first)
            ancestry.insert(0, parent)

            # Move up to the parent
            current_pk = parent_id

        return ancestry

    @staticmethod
    def _cosine_similarity_numpy(vec1: VectorType, vec2: VectorType) -> float:
        """
        Calculate cosine similarity between two vectors using NumPy.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (between -1 and 1)
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for this method but not available")

        # Make sure vectors are the same length
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]

        # Convert to numpy arrays
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        # Calculate dot product
        dot_product = np.dot(vec1_np, vec2_np)

        # Calculate magnitudes
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def _cosine_similarity_python(vec1: VectorType, vec2: VectorType) -> float:
        """
        Calculate cosine similarity between two vectors using pure Python.
        This is a fallback when NumPy is not available.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (between -1 and 1)
        """
        # Make sure vectors are the same length
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Calculate magnitudes
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        # Check for zero magnitudes
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Calculate cosine similarity
        return float(dot_product / (magnitude1 * magnitude2))
