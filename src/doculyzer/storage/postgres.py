import json
import logging
import os
from datetime import date
from typing import Optional, Dict, Any, List, Tuple, Union, TYPE_CHECKING

import time

# Import types for type checking only - these won't be imported at runtime
if TYPE_CHECKING:
    import psycopg2
    import psycopg2.extras
    import psycopg2.extensions
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    import numpy as np
    from numpy.typing import NDArray

    # Define type aliases for type checking
    VectorType = NDArray[np.float32]  # NumPy array type for vectors
    PostgresConnectionType = psycopg2.extensions.connection
    PostgresCursorType = psycopg2.extras.DictCursor
else:
    # Runtime type aliases - use generic Python types
    VectorType = List[float]  # Generic list of floats for vectors
    PostgresConnectionType = Any  # Generic type for PostgreSQL connection
    PostgresCursorType = Any  # Generic type for PostgreSQL cursor

from .base import DocumentDatabase
from .element_relationship import ElementRelationship
from .element_element import ElementType  # Import existing enum

logger = logging.getLogger(__name__)

# Define global flags for availability - these will be set at runtime
PSYCOPG2_AVAILABLE = False
NUMPY_AVAILABLE = False
PGVECTOR_AVAILABLE = False

# Try to import PostgreSQL library conditionally
try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.extensions
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    PSYCOPG2_AVAILABLE = True
except ImportError:
    logger.warning("psycopg2 not available. Install with 'pip install psycopg2-binary'.")

# Try to import NumPy conditionally
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("NumPy not available. Will use slower pure Python vector operations.")

# Try to import pgvector conditionally
try:
    import pgvector

    PGVECTOR_AVAILABLE = True
except ImportError:
    logger.debug("pgvector Python package not available. Will use native database support if available.")

# Try to import the config
try:
    from ..config import Config

    config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
except Exception as e:
    logger.warning(f"Error configuring PostgreSQL provider: {str(e)}")
    config = None


class PostgreSQLDocumentDatabase(DocumentDatabase):
    """PostgreSQL implementation of document database."""

    def __init__(self, conn_params: Dict[str, Any]):
        """
        Initialize PostgreSQL document database.

        Args:
            conn_params: Connection parameters for PostgreSQL
                (host, port, dbname, user, password)
        """
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL support")

        self.conn_params = conn_params
        self.conn: PostgresConnectionType = None
        self.cursor: PostgresCursorType = None
        self.vector_extension = None
        self.vector_dimension = config.config.get('embedding', {}).get('dimensions', 384) if config else 384
        self.embedding_generator = None

    def initialize(self) -> None:
        """Initialize the database by connecting and creating tables if they don't exist."""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL support")

        # Connect to PostgreSQL
        try:
            if "uri" in self.conn_params:
                # Use the URI if provided in connection parameters
                self.conn = psycopg2.connect(self.conn_params["uri"])
                logger.info("Connected to PostgreSQL using a URI.")
            else:
                # Use individual parameters if URI is not provided
                self.conn = psycopg2.connect(**self.conn_params)
                logger.info(
                    f"Connected to PostgreSQL at {self.conn_params.get('host', 'localhost')}:{self.conn_params.get('port', 5432)}"
                )

            # Set connection settings
            self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {str(e)}")
            raise

        # Discover vector extensions
        self._discover_vector_extensions()

        # Create tables if they don't exist
        self._create_tables()

        # Create vector column if vector extension is available
        if self.vector_extension:
            self._create_vector_column()

        logger.info(f"Initialized PostgreSQL database with vector extension: {self.vector_extension}")

    def _discover_vector_extensions(self) -> None:
        """Discover available vector search extensions."""
        try:
            # Check if pgvector is available
            self.cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_available_extensions WHERE name = 'vector'
                )
            """)
            pgvector_available = self.cursor.fetchone()[0]

            if pgvector_available:
                # Check if pgvector is installed
                self.cursor.execute("""
                    SELECT EXISTS(
                        SELECT 1 FROM pg_extension WHERE extname = 'vector'
                    )
                """)
                pgvector_installed = self.cursor.fetchone()[0]

                if not pgvector_installed:
                    try:
                        # Try to install pgvector
                        logger.info("Installing pgvector extension...")
                        self.cursor.execute("CREATE EXTENSION vector")
                        self.vector_extension = "pgvector"
                        logger.info("Successfully installed pgvector extension")
                    except Exception as e:
                        logger.warning(f"Failed to install pgvector extension: {str(e)}")
                else:
                    self.vector_extension = "pgvector"
                    logger.info("pgvector extension is already installed")
            else:
                logger.info("pgvector extension is not available on this PostgreSQL server")

            # Add checks for other vector extensions here if needed

        except Exception as e:
            logger.warning(f"Error discovering vector extensions: {str(e)}")
            self.vector_extension = None

    def close(self) -> None:
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
            self.cursor = None

        if self.conn:
            self.conn.close()
            self.conn = None

    def get_last_processed_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about when a document was last processed."""
        if not self.cursor:
            raise ValueError("Database not initialized")

        try:
            self.cursor.execute(
                """
                SELECT * FROM processing_history 
                WHERE source_id = %s
                """,
                (source_id,)
            )

            row = self.cursor.fetchone()
            if row is None:
                return None

            return {
                "source_id": row["source_id"],
                "content_hash": row["content_hash"],
                "last_modified": row["last_modified"],
                "processing_count": row["processing_count"]
            }
        except Exception as e:
            logger.error(f"Error getting processing history for {source_id}: {str(e)}")
            return None

    def update_processing_history(self, source_id: str, content_hash: str) -> None:
        """Update the processing history for a document."""
        if not self.cursor:
            raise ValueError("Database not initialized")

        try:
            # Check if record exists
            self.cursor.execute(
                "SELECT processing_count FROM processing_history WHERE source_id = %s",
                (source_id,)
            )

            row = self.cursor.fetchone()
            processing_count = 1  # Default for new records

            if row is not None:
                processing_count = row[0] + 1

                # Update existing record
                self.cursor.execute(
                    """
                    UPDATE processing_history
                    SET content_hash = %s, last_modified = %s, processing_count = %s
                    WHERE source_id = %s
                    """,
                    (content_hash, time.time(), processing_count, source_id)
                )
            else:
                # Insert new record
                self.cursor.execute(
                    """
                    INSERT INTO processing_history
                    (source_id, content_hash, last_modified, processing_count)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (source_id, content_hash, time.time(), processing_count)
                )

            self.conn.commit()
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
        if not self.cursor:
            raise ValueError("Database not initialized")

        source = document.get("source", "")
        content_hash = document.get("content_hash", "")

        # Check if document already exists with this source
        self.cursor.execute(
            "SELECT doc_id FROM documents WHERE source = %s",
            (source,)
        )
        existing_doc = self.cursor.fetchone()

        if existing_doc:
            # Document exists, update it
            doc_id = existing_doc[0]
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
            metadata = document.get("metadata", {})

            class CustomJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, date):
                        return obj.isoformat()  # Convert date to ISO 8601 string format (e.g., 'YYYY-MM-DD')
                    return super().default(obj)

            metadata_json = json.dumps(metadata, cls=CustomJSONEncoder)

            self.cursor.execute(
                """
                INSERT INTO documents 
                (doc_id, doc_type, source, content_hash, metadata, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    doc_id,
                    document.get("doc_type", ""),
                    source,
                    content_hash,
                    metadata_json,
                    document.get("created_at", time.time()),
                    document.get("updated_at", time.time())
                )
            )

            # Store elements
            for element in elements:
                element_id = element["element_id"]
                metadata_json = json.dumps(element.get("metadata", {}))
                content_preview = element.get("content_preview", "")
                if len(content_preview) > 100:
                    content_preview = content_preview[:100] + "..."

                self.cursor.execute(
                    """
                    INSERT INTO elements 
                    (element_id, doc_id, element_type, parent_id, content_preview, 
                     content_location, content_hash, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING element_pk
                    """,
                    (
                        element_id,
                        element.get("doc_id", ""),
                        element.get("element_type", ""),
                        element.get("parent_id", ""),
                        content_preview,
                        element.get("content_location", ""),
                        element.get("content_hash", ""),
                        metadata_json
                    )
                )

                # Get the PostgreSQL serial auto-increment ID
                element_pk = self.cursor.fetchone()[0]
                # Store it back into the dictionary
                element['element_pk'] = element_pk

            # Store relationships
            for relationship in relationships:
                relationship_id = relationship["relationship_id"]
                metadata_json = json.dumps(relationship.get("metadata", {}))

                self.cursor.execute(
                    """
                    INSERT INTO relationships 
                    (relationship_id, source_id, relationship_type, target_reference, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        relationship_id,
                        relationship.get("source_id", ""),
                        relationship.get("relationship_type", ""),
                        relationship.get("target_reference", ""),
                        metadata_json
                    )
                )

            # Commit transaction
            self.conn.commit()

            # Update processing history
            if source:
                self.update_processing_history(source, content_hash)

        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            logger.error(f"Error storing document {doc_id}: {str(e)}")
            raise

    def update_document(self, doc_id: str, document: Dict[str, Any],
                        elements: List[Dict[str, Any]],
                        relationships: List[Dict[str, Any]]) -> None:
        """
        Update an existing document.
        This will delete the old document and insert a new one.
        """
        if not self.cursor:
            raise ValueError("Database not initialized")

        # Check if document exists
        self.cursor.execute("SELECT doc_id FROM documents WHERE doc_id = %s", (doc_id,))
        if self.cursor.fetchone() is None:
            raise ValueError(f"Document not found: {doc_id}")

        try:
            # Get all element PKs for this document
            self.cursor.execute("SELECT element_pk FROM elements WHERE doc_id = %s", (doc_id,))
            element_pks = [row[0] for row in self.cursor.fetchall()]

            # Delete relationships related to this document's elements
            if element_pks:
                element_pks_str = ','.join(['%s'] * len(element_pks))
                self.cursor.execute(
                    f"DELETE FROM relationships WHERE source_id IN (SELECT element_id FROM elements WHERE element_pk IN ({element_pks_str}))",
                    element_pks)

            # Delete embeddings for this document's elements
            if element_pks:
                element_pks_str = ','.join(['%s'] * len(element_pks))
                self.cursor.execute(f"DELETE FROM embeddings WHERE element_pk IN ({element_pks_str})", element_pks)

            # Delete all elements for this document
            self.cursor.execute("DELETE FROM elements WHERE doc_id = %s", (doc_id,))

            # Delete the document itself
            self.cursor.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

            # Commit the deletion part of the transaction
            self.conn.commit()

            # Now use store_document to insert everything
            # This will also update the processing history
            self.store_document(document, elements, relationships)

        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            logger.error(f"Error updating document {doc_id}: {str(e)}")
            raise

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        if not self.cursor:
            raise ValueError("Database not initialized")

        self.cursor.execute(
            "SELECT * FROM documents WHERE doc_id = %s",
            (doc_id,)
        )

        row = self.cursor.fetchone()
        if row is None:
            return None

        doc = dict(row)

        # Convert metadata from JSON
        try:
            doc["metadata"] = doc["metadata"]
        except (json.JSONDecodeError, TypeError):
            doc["metadata"] = {}

        return doc

    def get_document_elements(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get elements for a document."""
        if not self.cursor:
            raise ValueError("Database not initialized")

        # Modified to handle doc_id being either an actual doc_id or a source
        self.cursor.execute(
            """
            SELECT e.* FROM elements e
            JOIN documents d ON e.doc_id = d.doc_id
            WHERE d.doc_id = %s OR d.source = %s
            ORDER BY e.element_id
            """,
            (doc_id, doc_id)
        )

        elements = []
        for row in self.cursor.fetchall():
            element = dict(row)

            # Convert metadata from JSON
            try:
                element["metadata"] = element["metadata"]
            except (json.JSONDecodeError, TypeError):
                element["metadata"] = {}

            elements.append(element)

        return elements

    def get_document_relationships(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get relationships for a document."""
        if not self.cursor:
            raise ValueError("Database not initialized")

        # First get all element IDs for the document
        self.cursor.execute(
            "SELECT element_id FROM elements WHERE doc_id = %s",
            (doc_id,)
        )

        element_ids = [row[0] for row in self.cursor.fetchall()]

        if not element_ids:
            return []

        # Create placeholders for SQL IN clause
        element_ids_str = ','.join(['%s'] * len(element_ids))

        # Find relationships involving these elements
        self.cursor.execute(
            f"SELECT * FROM relationships WHERE source_id IN ({element_ids_str})",
            element_ids
        )

        relationships = []
        for row in self.cursor.fetchall():
            relationship = dict(row)

            # Convert metadata from JSON
            try:
                relationship["metadata"] = relationship["metadata"]
            except (json.JSONDecodeError, TypeError):
                relationship["metadata"] = {}

            relationships.append(relationship)

        return relationships

    def get_element(self, element_id_or_pk: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Get element by ID or PK.

        Args:
            element_id_or_pk: Either the element_id (string) or element_pk (integer)

        Returns:
            Element data or None if not found
        """
        if not self.cursor:
            raise ValueError("Database not initialized")

        # Try to interpret as element_pk (integer) first
        try:
            element_pk = int(element_id_or_pk)
            self.cursor.execute(
                "SELECT * FROM elements WHERE element_pk = %s",
                (element_pk,)
            )
        except (ValueError, TypeError):
            # If not an integer, treat as element_id (string)
            self.cursor.execute(
                "SELECT * FROM elements WHERE element_id = %s",
                (element_id_or_pk,)
            )

        row = self.cursor.fetchone()
        if row is None:
            return None

        element = dict(row)

        # Convert metadata from JSON
        try:
            element["metadata"] = element["metadata"]
        except (json.JSONDecodeError, TypeError):
            element["metadata"] = {}

        return element

    def get_outgoing_relationships(self, element_pk: Union[int, str]) -> List[ElementRelationship]:
        """
        Find all relationships where the specified element_pk is the source.

        Implementation for PostgreSQL database using JOIN to efficiently retrieve target information.

        Args:
            element_pk: The primary key of the element

        Returns:
            List of ElementRelationship objects where the specified element is the source
        """
        if not self.cursor:
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
            # Find relationships with target element information using JOIN
            # This query joins the relationships table with the elements table
            # to get information about target elements in one go
            self.cursor.execute(
                """
                SELECT 
                    r.*,
                    t.element_pk as target_element_pk,
                    t.element_type as target_element_type,
                    t.content_preview as target_content_preview
                FROM 
                    relationships r
                LEFT JOIN 
                    elements t ON r.target_reference = t.element_id
                WHERE 
                    r.source_id = %s
                """,
                (element_id,)
            )

            for row in self.cursor.fetchall():
                # Convert to dictionary
                rel_dict = dict(row)

                # Convert metadata from JSON if it's in string format
                if isinstance(rel_dict.get("metadata"), str):
                    try:
                        rel_dict["metadata"] = rel_dict["metadata"]
                    except (json.JSONDecodeError, TypeError):
                        rel_dict["metadata"] = {}

                # Extract target element information from the joined query results
                target_element_pk = rel_dict.get("target_element_pk")
                target_element_type = rel_dict.get("target_element_type")
                target_content_preview = rel_dict.get("target_content_preview", "")

                # Create enriched relationship
                relationship = ElementRelationship(
                    relationship_id=rel_dict.get("relationship_id", ""),
                    source_id=element_id,
                    source_element_pk=element_pk if isinstance(element_pk, int) else element.get("element_pk"),
                    source_element_type=element_type,
                    relationship_type=rel_dict.get("relationship_type", ""),
                    target_reference=rel_dict.get("target_reference", ""),
                    target_element_pk=target_element_pk,
                    target_element_type=target_element_type,
                    target_content_preview=target_content_preview,
                    doc_id=rel_dict.get("doc_id"),
                    metadata=rel_dict.get("metadata", {}),
                    is_source=True
                )

                relationships.append(relationship)

            return relationships

        except Exception as e:
            logger.error(f"Error getting outgoing relationships for element {element_pk}: {str(e)}")
            return []

    def store_relationship(self, relationship: Dict[str, Any]) -> None:
        """
        Store a relationship between elements.

        Args:
            relationship: Relationship data with source_id, relationship_type, and target_reference
        """
        if not self.cursor:
            raise ValueError("Database not initialized")

        try:
            # Convert metadata to JSON
            metadata_json = json.dumps(relationship.get("metadata", {}))

            # Insert the relationship
            self.cursor.execute(
                """
                INSERT INTO relationships
                (relationship_id, source_id, relationship_type, target_reference, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (relationship_id) DO UPDATE
                SET source_id = EXCLUDED.source_id,
                    relationship_type = EXCLUDED.relationship_type,
                    target_reference = EXCLUDED.target_reference,
                    metadata = EXCLUDED.metadata
                """,
                (
                    relationship["relationship_id"],
                    relationship.get("source_id", ""),
                    relationship.get("relationship_type", ""),
                    relationship.get("target_reference", ""),
                    metadata_json
                )
            )

            # Commit the transaction
            self.conn.commit()
            logger.debug(f"Stored relationship {relationship['relationship_id']}")

        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            logger.error(f"Error storing relationship: {str(e)}")
            raise

    def delete_relationships_for_element(self, element_id: str, relationship_type: str = None) -> None:
        """
        Delete relationships for an element.

        Args:
            element_id: Element ID
            relationship_type: Optional relationship type to filter by
        """
        if not self.cursor:
            raise ValueError("Database not initialized")

        try:
            # Build query conditions
            conditions = ["source_id = %s"]
            params = [element_id]

            # Add relationship type filter if provided
            if relationship_type:
                conditions.append("relationship_type = %s")
                params.append(relationship_type)

            # Execute delete for source relationships
            self.cursor.execute(
                f"DELETE FROM relationships WHERE {' AND '.join(conditions)}",
                params
            )

            # Build query conditions for target relationships
            conditions = ["target_reference = %s"]
            params = [element_id]

            # Add relationship type filter if provided
            if relationship_type:
                conditions.append("relationship_type = %s")
                params.append(relationship_type)

            # Execute delete for target relationships
            self.cursor.execute(
                f"DELETE FROM relationships WHERE {' AND '.join(conditions)}",
                params
            )

            # Commit the transaction
            self.conn.commit()
            logger.debug(f"Deleted relationships for element {element_id}")

        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            logger.error(f"Error deleting relationships for element {element_id}: {str(e)}")
            raise

    def find_documents(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find documents matching query with support for LIKE patterns.

        Args:
            query: Query parameters. Use '_like' suffix for LIKE patterns.
                   Examples:
                   - {"doc_type": "pdf"} - exact match
                   - {"source_like": "%reports%"} - LIKE pattern
                   - {"source_ilike": "%REPORTS%"} - case-insensitive LIKE
                   - {"metadata": {"author": "John"}} - metadata exact match
                   - {"metadata_like": {"title": "%annual%"}} - metadata LIKE pattern
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        if not self.cursor:
            raise ValueError("Database not initialized")

        # Start with base query
        sql = "SELECT * FROM documents"
        params = []

        # Apply filters if provided
        if query:
            conditions = []

            for key, value in query.items():
                if key == "metadata":
                    # Metadata filters require special handling with JSONB (exact match)
                    for meta_key, meta_value in value.items():
                        conditions.append("metadata->>%s = %s")
                        params.extend([meta_key, str(meta_value)])
                elif key == "metadata_like":
                    # Metadata LIKE filters
                    for meta_key, meta_value in value.items():
                        conditions.append("metadata->>%s LIKE %s")
                        params.extend([meta_key, str(meta_value)])
                elif key == "metadata_ilike":
                    # Case-insensitive metadata LIKE filters
                    for meta_key, meta_value in value.items():
                        conditions.append("metadata->>%s ILIKE %s")
                        params.extend([meta_key, str(meta_value)])
                elif key.endswith("_ilike"):
                    # Case-insensitive LIKE pattern
                    field_name = key[:-6]  # Remove '_ilike' suffix
                    conditions.append(f"{field_name} ILIKE %s")
                    params.append(value)
                elif key.endswith("_like"):
                    # LIKE pattern for regular fields
                    field_name = key[:-5]  # Remove '_like' suffix
                    conditions.append(f"{field_name} LIKE %s")
                    params.append(value)
                elif isinstance(value, list):
                    # Handle list fields with IN clause
                    placeholders = ', '.join(['%s'] * len(value))
                    conditions.append(f"{key} IN ({placeholders})")
                    params.extend(value)
                else:
                    # Exact match for regular fields
                    conditions.append(f"{key} = %s")
                    params.append(value)

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        # Add limit
        sql += f" LIMIT %s"
        params.append(limit)

        # Execute query
        self.cursor.execute(sql, params)

        documents = []
        for row in self.cursor.fetchall():
            doc = dict(row)

            # Convert metadata from JSON
            try:
                doc["metadata"] = doc["metadata"]
            except (json.JSONDecodeError, TypeError):
                doc["metadata"] = {}

            documents.append(doc)

        return documents

    def find_elements(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements matching query with support for LIKE patterns and ElementType enums.

        Args:
            query: Query parameters. Use '_like' suffix for LIKE patterns.
                   Examples:
                   - {"element_type": "header"} - exact match with string
                   - {"element_type": ElementType.HEADER} - exact match with enum
                   - {"element_type": [ElementType.HEADER, ElementType.PARAGRAPH]} - enum list
                   - {"element_type_like": "head%"} - LIKE pattern
                   - {"element_type_ilike": "HEAD%"} - case-insensitive LIKE
                   - {"content_preview_like": "%important%"} - LIKE pattern
                   - {"doc_id": ["doc1", "doc2"]} - list for IN clause
                   - {"metadata": {"section": "intro"}} - metadata exact match
                   - {"metadata_like": {"title": "%summary%"}} - metadata LIKE pattern
            limit: Maximum number of results

        Returns:
            List of matching elements
        """
        if not self.cursor:
            raise ValueError("Database not initialized")

        # Start with base query
        sql = "SELECT * FROM elements"
        params = []

        # Apply filters if provided
        if query:
            conditions = []

            for key, value in query.items():
                if key == "metadata":
                    # Metadata filters require special handling with JSONB (exact match)
                    for meta_key, meta_value in value.items():
                        conditions.append("metadata->>%s = %s")
                        params.extend([meta_key, str(meta_value)])
                elif key == "metadata_like":
                    # Metadata LIKE filters
                    for meta_key, meta_value in value.items():
                        conditions.append("metadata->>%s LIKE %s")
                        params.extend([meta_key, str(meta_value)])
                elif key == "metadata_ilike":
                    # Case-insensitive metadata LIKE filters
                    for meta_key, meta_value in value.items():
                        conditions.append("metadata->>%s ILIKE %s")
                        params.extend([meta_key, str(meta_value)])
                elif key.endswith("_ilike"):
                    # Case-insensitive LIKE pattern
                    field_name = key[:-6]  # Remove '_ilike' suffix
                    conditions.append(f"{field_name} ILIKE %s")
                    params.append(value)
                elif key.endswith("_like"):
                    # LIKE pattern for regular fields
                    field_name = key[:-5]  # Remove '_like' suffix
                    conditions.append(f"{field_name} LIKE %s")
                    params.append(value)
                elif key == "element_type":
                    # Handle ElementType enums, strings, and lists
                    type_values = self._prepare_element_type_query(value)
                    if type_values:
                        if len(type_values) == 1:
                            conditions.append("element_type = %s")
                            params.append(type_values[0])
                        else:
                            placeholders = ', '.join(['%s'] * len(type_values))
                            conditions.append(f"element_type IN ({placeholders})")
                            params.extend(type_values)
                elif isinstance(value, list):
                    # Handle other list fields with IN clause
                    field_name = key
                    placeholders = ', '.join(['%s'] * len(value))
                    conditions.append(f"{field_name} IN ({placeholders})")
                    params.extend(value)
                else:
                    # Exact match for regular fields
                    conditions.append(f"{key} = %s")
                    params.append(value)

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        # Add limit
        sql += f" LIMIT %s"
        params.append(limit)

        # Execute query
        self.cursor.execute(sql, params)

        elements = []
        for row in self.cursor.fetchall():
            element = dict(row)

            # Convert metadata from JSON
            try:
                element["metadata"] = element["metadata"]
            except (json.JSONDecodeError, TypeError):
                element["metadata"] = {}

            elements.append(element)

        return elements

    def search_elements_by_content(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search elements by content preview."""
        if not self.cursor:
            raise ValueError("Database not initialized")

        self.cursor.execute(
            "SELECT * FROM elements WHERE content_preview LIKE %s LIMIT %s",
            (f"%{search_text}%", limit)
        )

        elements = []
        for row in self.cursor.fetchall():
            element = dict(row)

            # Convert metadata from JSON
            try:
                element["metadata"] = element["metadata"]
            except (json.JSONDecodeError, TypeError):
                element["metadata"] = {}

            elements.append(element)

        return elements

    def store_embedding(self, element_pk: int, embedding: VectorType) -> None:
        """Store embedding for an element."""
        if not self.cursor:
            raise ValueError("Database not initialized")

        # Verify element exists
        self.cursor.execute(
            "SELECT element_pk FROM elements WHERE element_pk = %s",
            (element_pk,)
        )

        if self.cursor.fetchone() is None:
            raise ValueError(f"Element not found: {element_pk}")

        # Update vector dimension based on actual data
        self.vector_dimension = max(self.vector_dimension, len(embedding))

        try:
            # Store embedding in the main embeddings table
            embedding_json = json.dumps(embedding)

            self.cursor.execute(
                """
                INSERT INTO embeddings 
                (element_pk, embedding, dimensions, topics, confidence, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (element_pk) DO UPDATE 
                SET embedding = EXCLUDED.embedding, 
                    dimensions = EXCLUDED.dimensions, 
                    topics = EXCLUDED.topics,
                    confidence = EXCLUDED.confidence,
                    created_at = EXCLUDED.created_at
                """,
                (
                    element_pk,
                    embedding_json,
                    len(embedding),
                    json.dumps([]),  # Default to empty topics
                    1.0,  # Default confidence
                    time.time()
                )
            )

            # If pgvector is available, also store in vector column
            if self.vector_extension == "pgvector":
                self.cursor.execute(
                    """
                    UPDATE embeddings
                    SET vector_embedding = %s::vector
                    WHERE element_pk = %s
                    """,
                    (embedding_json, element_pk)
                )

            self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing embedding for {element_pk}: {str(e)}")
            raise

    def get_embedding(self, element_pk: int) -> Optional[VectorType]:
        """Get embedding for an element."""
        if not self.cursor:
            raise ValueError("Database not initialized")

        self.cursor.execute(
            "SELECT embedding FROM embeddings WHERE element_pk = %s",
            (element_pk,)
        )

        row = self.cursor.fetchone()
        if row is None:
            return None

        try:
            return row["embedding"]
        except (json.JSONDecodeError, TypeError):
            return None

    def search_by_embedding(self, query_embedding: VectorType, limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by embedding similarity using available method.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_criteria: Optional dictionary with criteria to filter results
                            (e.g. {"element_type": ["header", "section"]})

        Returns:
            List of (element_pk, similarity_score) tuples
        """
        if not self.cursor:
            raise ValueError("Database not initialized")

        try:
            if self.vector_extension == "pgvector":
                return self._search_by_pgvector(query_embedding, limit, filter_criteria)
            else:
                return self._search_by_similarity_function(query_embedding, limit, filter_criteria)
        except Exception as e:
            logger.error(f"Error searching by embedding: {str(e)}")
            # Fall back to non-vector search
            return self._search_by_similarity_function(query_embedding, limit, filter_criteria)

    def _search_by_pgvector(self, query_embedding: VectorType, limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search embeddings using pgvector similarity with filtering.
        """
        vector_json = json.dumps(query_embedding)

        try:
            # Start building the query
            sql = """
            SELECT e.element_pk, 1 - (em.vector_embedding <=> %s::vector) as similarity
            FROM embeddings em
            JOIN elements e ON e.element_pk = em.element_pk
            JOIN documents d ON e.doc_id = d.doc_id
            """
            params = [vector_json]

            # Add WHERE clauses if we have filter criteria
            if filter_criteria:
                conditions = []
                for key, value in filter_criteria.items():
                    if key == "element_type" and isinstance(value, list):
                        # Handle list of allowed element types
                        placeholders = ', '.join(['%s'] * len(value))
                        conditions.append(f"e.element_type IN ({placeholders})")
                        params.extend(value)
                    elif key == "doc_id" and isinstance(value, list):
                        # Handle list of document IDs to include
                        placeholders = ', '.join(['%s'] * len(value))
                        conditions.append(f"e.doc_id IN ({placeholders})")
                        params.extend(value)
                    elif key == "exclude_doc_id" and isinstance(value, list):
                        # Handle list of document IDs to exclude
                        placeholders = ', '.join(['%s'] * len(value))
                        conditions.append(f"e.doc_id NOT IN ({placeholders})")
                        params.extend(value)
                    elif key == "exclude_doc_source" and isinstance(value, list):
                        # Handle list of document sources to exclude
                        placeholders = ', '.join(['%s'] * len(value))
                        conditions.append(f"d.source NOT IN ({placeholders})")
                        params.extend(value)
                    else:
                        # Simple equality filter
                        conditions.append(f"e.{key} = %s")
                        params.append(value)

                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)

            # FIXED ORDER BY CLAUSE - order by similarity DESC instead of by distance
            sql += """
            ORDER BY similarity DESC
            LIMIT %s
            """
            params.append(limit)

            # Execute the query
            self.cursor.execute(sql, params)

            # Return element_pk and similarity score
            return [(row[0], row[1]) for row in self.cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error using pgvector for search: {str(e)}")
            raise

    def _search_by_similarity_function(self, query_embedding: VectorType, limit: int = 10,
                                       filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Fall back to calculating similarity in Python with filtering.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_criteria: Optional dictionary with criteria to filter results

        Returns:
            List of (element_pk, similarity_score) tuples
        """
        # Build base query to get embeddings with possible filtering
        sql = """
        SELECT em.element_pk, em.embedding, e.element_type, e.doc_id, d.source
        FROM embeddings em
        JOIN elements e ON e.element_pk = em.element_pk
        JOIN documents d ON e.doc_id = d.doc_id
        """
        params = []

        # Add WHERE clauses if we have filter criteria
        if filter_criteria:
            conditions = []

            for key, value in filter_criteria.items():
                if key == "element_type" and isinstance(value, list):
                    # Handle list of allowed element types
                    placeholders = ', '.join(['%s'] * len(value))
                    conditions.append(f"e.element_type IN ({placeholders})")
                    params.extend(value)
                elif key == "doc_id" and isinstance(value, list):
                    # Handle list of document IDs to include
                    placeholders = ', '.join(['%s'] * len(value))
                    conditions.append(f"e.doc_id IN ({placeholders})")
                    params.extend(value)
                elif key == "exclude_doc_id" and isinstance(value, list):
                    # Handle list of document IDs to exclude
                    placeholders = ', '.join(['%s'] * len(value))
                    conditions.append(f"e.doc_id NOT IN ({placeholders})")
                    params.extend(value)
                elif key == "exclude_doc_source" and isinstance(value, list):
                    # Handle list of document sources to exclude
                    placeholders = ', '.join(['%s'] * len(value))
                    conditions.append(f"d.source NOT IN ({placeholders})")
                    params.extend(value)
                else:
                    # Simple equality filter
                    conditions.append(f"e.{key} = %s")
                    params.append(value)

            # Add WHERE clause if we have conditions
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        # Execute the query
        self.cursor.execute(sql, params)

        # Calculate similarities based on the available implementations
        if NUMPY_AVAILABLE:
            similarities = [
                (row["element_pk"], self._cosine_similarity_numpy(query_embedding, row["embedding"]))
                for row in self.cursor.fetchall()
            ]
        else:
            similarities = [
                (row["element_pk"], self._cosine_similarity_python(query_embedding, row["embedding"]))
                for row in self.cursor.fetchall()
            ]

        # Sort by similarity (highest first)
        similarities.sort(key=lambda row: row[1], reverse=True)

        return similarities[:limit]

    def search_by_text(self, search_text: str, limit: int = 10,
                       filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by semantic similarity to the provided text.

        This method combines text-to-embedding conversion and embedding search
        into a single convenient operation.

        Args:
            search_text: Text to search for semantically
            limit: Maximum number of results
            filter_criteria: Optional dictionary with criteria to filter results

        Returns:
            List of (element_pk, similarity_score) tuples
        """
        if not self.cursor:
            raise ValueError("Database not initialized")

        try:
            # Import necessary modules
            from ..embeddings import get_embedding_generator

            # Initialize embedding generator if not already done
            if self.embedding_generator is None:
                # Get config from the connection parameters
                # This assumes config is accessible, otherwise it would need to be passed in
                config_obj = self.conn_params.get('config')
                if not config_obj:
                    from ..config import Config
                    config_obj = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))

                self.embedding_generator = get_embedding_generator(config_obj)

            # Generate embedding for the search text
            query_embedding = self.embedding_generator.generate(search_text)

            # Use the embedding to search, passing filter criteria
            return self.search_by_embedding(query_embedding, limit, filter_criteria)

        except Exception as e:
            logger.error(f"Error in semantic search by text: {str(e)}")
            # Return empty list on error
            return []

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all associated elements and relationships."""
        if not self.cursor:
            raise ValueError("Database not initialized")

        # Check if document exists
        self.cursor.execute(
            "SELECT doc_id FROM documents WHERE doc_id = %s",
            (doc_id,)
        )

        if self.cursor.fetchone() is None:
            return False

        try:
            # Get all element PKs for this document
            self.cursor.execute(
                "SELECT element_pk FROM elements WHERE doc_id = %s",
                (doc_id,)
            )

            element_pks = [row[0] for row in self.cursor.fetchall()]

            # Get all element IDs for this document (for relationship deletion)
            self.cursor.execute(
                "SELECT element_id FROM elements WHERE doc_id = %s",
                (doc_id,)
            )
            element_ids = [row[0] for row in self.cursor.fetchall()]

            # Delete embeddings for these elements
            if element_pks:
                element_pks_str = ','.join(['%s'] * len(element_pks))
                self.cursor.execute(f"DELETE FROM embeddings WHERE element_pk IN ({element_pks_str})", element_pks)

            # Delete relationships involving these elements
            if element_ids:
                element_ids_str = ','.join(['%s'] * len(element_ids))
                self.cursor.execute(f"DELETE FROM relationships WHERE source_id IN ({element_ids_str})", element_ids)

            # Delete elements
            self.cursor.execute(
                "DELETE FROM elements WHERE doc_id = %s",
                (doc_id,)
            )

            # Delete document
            self.cursor.execute(
                "DELETE FROM documents WHERE doc_id = %s",
                (doc_id,)
            )

            # Commit transaction
            self.conn.commit()

            return True

        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        try:
            # Create the required schemas
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                doc_type TEXT,
                source TEXT,
                content_hash TEXT,
                metadata JSONB,
                created_at DOUBLE PRECISION,
                updated_at DOUBLE PRECISION
            )
            """)

            # Modified elements table with element_pk as serial
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS elements (
                element_pk SERIAL PRIMARY KEY,
                element_id TEXT UNIQUE NOT NULL,
                doc_id TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
                element_type TEXT,
                parent_id TEXT REFERENCES elements(element_id),
                content_preview TEXT,
                content_location TEXT,
                content_hash TEXT,
                metadata JSONB
            )
            """)

            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_elements_doc_id ON elements(doc_id)
            """)

            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_elements_parent_id ON elements(parent_id)
            """)

            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_elements_type ON elements(element_type)
            """)

            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id TEXT PRIMARY KEY,
                source_id TEXT REFERENCES elements(element_id) ON DELETE CASCADE,
                relationship_type TEXT,
                target_reference TEXT,
                metadata JSONB
            )
            """)

            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id)
            """)

            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type)
            """)

            # Modified embeddings table with topic support
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                element_pk INTEGER PRIMARY KEY REFERENCES elements(element_pk) ON DELETE CASCADE,
                embedding JSONB,
                dimensions INTEGER,
                topics JSONB DEFAULT '[]'::jsonb,
                confidence REAL DEFAULT 1.0,
                created_at DOUBLE PRECISION
            )
            """)

            # Add indexes for topic searching
            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_topics ON embeddings USING GIN (topics)
            """)

            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_confidence ON embeddings(confidence)
            """)

            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_history (
                source_id TEXT PRIMARY KEY,
                content_hash TEXT,
                last_modified DOUBLE PRECISION,
                processing_count INTEGER DEFAULT 1
            )
            """)

            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_processing_history_source_id ON processing_history(source_id)
            """)

            self.conn.commit()
            logger.info("Created core database tables")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error creating tables: {str(e)}")
            raise

    def _create_vector_column(self) -> None:
        """Create vector column for embeddings if pgvector is available."""
        if self.vector_extension != "pgvector":
            return

        try:
            # Add vector column to embeddings table
            self.cursor.execute(f"""
            ALTER TABLE embeddings 
            ADD COLUMN IF NOT EXISTS vector_embedding vector({self.vector_dimension})
            """)

            # Create index for vector similarity search
            # Using cosine distance by default (can be changed based on needs)
            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_vector_cosine ON embeddings 
            USING ivfflat (vector_embedding vector_cosine_ops)
            WITH (lists = 100)
            """)

            self.conn.commit()
            logger.info(f"Created vector column and index with dimension {self.vector_dimension}")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error creating vector column: {str(e)}")

    def _cosine_similarity(self, vec1: VectorType, vec2: VectorType) -> float:
        """
        Calculate cosine similarity between two vectors.
        Automatically uses NumPy if available, otherwise falls back to pure Python.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (between -1 and 1)
        """
        if NUMPY_AVAILABLE:
            return self._cosine_similarity_numpy(vec1, vec2)
        else:
            return self._cosine_similarity_python(vec1, vec2)

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

    # ========================================
    # NEW: ENHANCED SEARCH HELPER METHODS
    # ========================================

    def _prepare_element_type_query(self, element_types: Union[
        ElementType,
        List[ElementType],
        str,
        List[str],
        None
    ]) -> Optional[List[str]]:
        """
        Prepare element type values for database queries using existing ElementType enum.

        Args:
            element_types: ElementType enum(s), string(s), or None

        Returns:
            List of string values for database query, or None
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

    def get_element_types_by_category(self):
        """
        Get categorized lists of ElementType enums from your existing enum.

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
        Find elements by predefined category using your existing ElementType enum.

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
            raise ValueError(f"Unknown category: {category}. Available: {list(categories.keys())}")

        element_types = categories[category]
        query = {"element_type": element_types}
        query.update(other_filters)

        return self.find_elements(query)

    def find_elements_ilike(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements with case-insensitive LIKE support.

        PostgreSQL has native ILIKE support.

        Args:
            query: Query parameters with _ilike suffix support
            limit: Maximum number of results

        Returns:
            List of matching elements
        """
        # PostgreSQL has native ILIKE, so just use the regular find_elements method
        return self.find_elements(query, limit)

    def supports_like_patterns(self) -> bool:
        """PostgreSQL supports LIKE patterns."""
        return True

    def supports_case_insensitive_like(self) -> bool:
        """PostgreSQL has native ILIKE support."""
        return True

    def supports_element_type_enums(self) -> bool:
        """PostgreSQL supports ElementType enum integration."""
        return True

    def create_search_indexes(self):
        """
        Create additional indexes to optimize LIKE and enum searches.
        Call this after database initialization for better performance.
        """
        if not self.cursor:
            raise ValueError("Database not initialized")

        try:
            # Index for content preview LIKE searches
            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_elements_content_preview_gin 
            ON elements USING gin(content_preview gin_trgm_ops)
            """)

            # Index for element type searches
            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_elements_type_gin 
            ON elements USING gin(element_type gin_trgm_ops)
            """)

            # Index for document source LIKE searches
            self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_source_gin 
            ON documents USING gin(source gin_trgm_ops)
            """)

            # Create trigram extension if available for better LIKE performance
            try:
                self.cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                logger.info("Created trigram extension for better LIKE performance")
            except Exception as e:
                logger.debug(f"Could not create trigram extension: {str(e)}")

            self.conn.commit()
            logger.info("Created additional search optimization indexes for PostgreSQL")

        except Exception as e:
            logger.warning(f"Could not create search optimization indexes: {str(e)}")

    def find_elements_with_jsonb_path(self, json_path: str, value: Any,
                                      operator: str = "=", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements using PostgreSQL JSONB path expressions.

        Args:
            json_path: JSONB path expression (e.g., "author", "tags[0]", "details.title")
            value: Value to search for
            operator: Comparison operator ("=", "LIKE", "ILIKE", "!=", etc.)
            limit: Maximum number of results

        Returns:
            List of matching elements

        Examples:
            find_elements_with_jsonb_path("author", "John", "=")
            find_elements_with_jsonb_path("title", "%report%", "LIKE")
            find_elements_with_jsonb_path("tags", "important", "@>")  # JSONB contains
        """
        if not self.cursor:
            raise ValueError("Database not initialized")

        # Build the query using JSONB operators
        sql = f"""
        SELECT * FROM elements 
        WHERE metadata->>%s {operator} %s
        LIMIT %s
        """

        params = [json_path, value if operator.upper() not in ["LIKE", "ILIKE"] else str(value), limit]

        self.cursor.execute(sql, params)

        elements = []
        for row in self.cursor.fetchall():
            element = dict(row)
            try:
                element["metadata"] = element["metadata"]
            except (json.JSONDecodeError, TypeError):
                element["metadata"] = {}
            elements.append(element)

        return elements

    # ========================================
    # EXISTING TOPIC SUPPORT METHODS
    # ========================================

    def supports_topics(self) -> bool:
        """
        Indicate whether this backend supports topic-aware embeddings.

        Returns:
            True since PostgreSQL implementation supports topics
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
        if not self.cursor:
            raise ValueError("Database not initialized")

        # Verify element exists
        self.cursor.execute(
            "SELECT element_pk FROM elements WHERE element_pk = %s",
            (element_pk,)
        )

        if self.cursor.fetchone() is None:
            raise ValueError(f"Element not found: {element_pk}")

        # Update vector dimension based on actual data
        self.vector_dimension = max(self.vector_dimension, len(embedding))

        try:
            # Store embedding with topics in the main embeddings table
            embedding_json = json.dumps(embedding)
            topics_json = json.dumps(topics)

            self.cursor.execute(
                """
                INSERT INTO embeddings 
                (element_pk, embedding, dimensions, topics, confidence, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (element_pk) DO UPDATE 
                SET embedding = EXCLUDED.embedding, 
                    dimensions = EXCLUDED.dimensions,
                    topics = EXCLUDED.topics,
                    confidence = EXCLUDED.confidence,
                    created_at = EXCLUDED.created_at
                """,
                (
                    element_pk,
                    embedding_json,
                    len(embedding),
                    topics_json,
                    confidence,
                    time.time()
                )
            )

            # If pgvector is available, also store in vector column
            if self.vector_extension == "pgvector":
                self.cursor.execute(
                    """
                    UPDATE embeddings
                    SET vector_embedding = %s::vector
                    WHERE element_pk = %s
                    """,
                    (embedding_json, element_pk)
                )

            self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing embedding with topics for {element_pk}: {str(e)}")
            raise

    def search_by_text_and_topics(self, search_text: str = None,
                                  include_topics: Optional[List[str]] = None,
                                  exclude_topics: Optional[List[str]] = None,
                                  min_confidence: float = 0.7,
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search elements by text with topic filtering using LIKE patterns.

        Args:
            search_text: Text to search for semantically (optional)
            include_topics: Topic LIKE patterns to include (e.g., ['security%', '%.policy%'])
            exclude_topics: Topic LIKE patterns to exclude (e.g., ['deprecated%'])
            min_confidence: Minimum confidence threshold for embeddings
            limit: Maximum number of results

        Returns:
            List of dictionaries with keys:
            - element_pk: Element primary key
            - similarity: Similarity score (if search_text provided)
            - confidence: Overall embedding confidence
            - topics: List of assigned topic strings
        """
        if not self.cursor:
            raise ValueError("Database not initialized")

        try:
            # Generate embedding for search text if provided
            query_embedding = None
            if search_text:
                # Import necessary modules
                from ..embeddings import get_embedding_generator

                # Initialize embedding generator if not already done
                if self.embedding_generator is None:
                    config_obj = self.conn_params.get('config')
                    if not config_obj:
                        from ..config import Config
                        config_obj = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
                    self.embedding_generator = get_embedding_generator(config_obj)

                query_embedding = self.embedding_generator.generate(search_text)

            # Build the query based on whether we have search text and vector support
            if search_text and self.vector_extension == "pgvector":
                return self._search_by_text_and_topics_pgvector(
                    query_embedding, include_topics, exclude_topics, min_confidence, limit
                )
            else:
                return self._search_by_text_and_topics_fallback(
                    query_embedding, include_topics, exclude_topics, min_confidence, limit
                )

        except Exception as e:
            logger.error(f"Error in topic-aware search: {str(e)}")
            return []

    def _search_by_text_and_topics_pgvector(self, query_embedding: VectorType,
                                            include_topics: Optional[List[str]] = None,
                                            exclude_topics: Optional[List[str]] = None,
                                            min_confidence: float = 0.7,
                                            limit: int = 10) -> List[Dict[str, Any]]:
        """Search using pgvector with topic filtering."""
        vector_json = json.dumps(query_embedding)

        # Base query with similarity calculation
        sql = """
        SELECT 
            em.element_pk,
            1 - (em.vector_embedding <=> %s::vector) as similarity,
            em.confidence,
            em.topics
        FROM embeddings em
        WHERE em.confidence >= %s
        """
        params = [vector_json, min_confidence]

        # Add topic filtering conditions
        sql, params = self._add_topic_filters(sql, params, include_topics, exclude_topics)

        # Order by similarity and limit
        sql += " ORDER BY similarity DESC LIMIT %s"
        params.append(limit)

        self.cursor.execute(sql, params)

        results = []
        for row in self.cursor.fetchall():
            try:
                topics = row[3] if row[3] else []
            except (json.JSONDecodeError, TypeError):
                topics = []

            results.append({
                'element_pk': row[0],
                'similarity': float(row[1]),
                'confidence': float(row[2]),
                'topics': topics
            })

        return results

    def _search_by_text_and_topics_fallback(self, query_embedding: Optional[VectorType] = None,
                                            include_topics: Optional[List[str]] = None,
                                            exclude_topics: Optional[List[str]] = None,
                                            min_confidence: float = 0.7,
                                            limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback search using Python similarity calculation with topic filtering."""

        # Base query to get embeddings with topic filtering
        sql = """
        SELECT em.element_pk, em.embedding, em.confidence, em.topics
        FROM embeddings em
        WHERE em.confidence >= %s
        """
        params = [min_confidence]

        # Add topic filtering conditions
        sql, params = self._add_topic_filters(sql, params, include_topics, exclude_topics)

        self.cursor.execute(sql, params)

        # Calculate similarities if we have a query embedding
        results = []
        for row in self.cursor.fetchall():
            try:
                topics = row[3] if row[3] else []
            except (json.JSONDecodeError, TypeError):
                topics = []

            result = {
                'element_pk': row[0],
                'confidence': float(row[2]),
                'topics': topics
            }

            # Calculate similarity if we have a query embedding
            if query_embedding:
                try:
                    embedding = row[1]
                    if NUMPY_AVAILABLE:
                        similarity = self._cosine_similarity_numpy(query_embedding, embedding)
                    else:
                        similarity = self._cosine_similarity_python(query_embedding, embedding)
                    result['similarity'] = float(similarity)
                except Exception as e:
                    logger.warning(f"Error calculating similarity for element {row[0]}: {str(e)}")
                    result['similarity'] = 0.0
            else:
                result['similarity'] = 1.0  # No text search, all results have equal similarity

            results.append(result)

        # Sort by similarity if we calculated it
        if query_embedding:
            results.sort(key=lambda x: x['similarity'], reverse=True)

        return results[:limit]

    def _add_topic_filters(self, sql: str, params: List,
                           include_topics: Optional[List[str]] = None,
                           exclude_topics: Optional[List[str]] = None) -> tuple[str, List]:
        """Add topic filtering conditions to SQL query."""

        # Add include topic filters
        if include_topics:
            include_conditions = []
            for topic_pattern in include_topics:
                # Check if any topic in the topics JSON array matches the pattern
                include_conditions.append("""
                    EXISTS (
                        SELECT 1 FROM jsonb_array_elements_text(em.topics) AS topic
                        WHERE topic LIKE %s
                    )
                """)
                params.append(topic_pattern)

            if include_conditions:
                sql += " AND (" + " OR ".join(include_conditions) + ")"

        # Add exclude topic filters
        if exclude_topics:
            exclude_conditions = []
            for topic_pattern in exclude_topics:
                # Check that no topic in the topics JSON array matches the pattern
                exclude_conditions.append("""
                    NOT EXISTS (
                        SELECT 1 FROM jsonb_array_elements_text(em.topics) AS topic
                        WHERE topic LIKE %s
                    )
                """)
                params.append(topic_pattern)

            if exclude_conditions:
                sql += " AND " + " AND ".join(exclude_conditions)

        return sql, params

    def get_topic_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about topic distribution across embeddings.

        Returns:
            Dictionary mapping topic strings to statistics:
            {
                'security.policy': {
                    'embedding_count': int,
                    'document_count': int,
                    'avg_embedding_confidence': float
                }
            }
        """
        if not self.cursor:
            raise ValueError("Database not initialized")

        try:
            # Query to get topic statistics using PostgreSQL JSON functions
            self.cursor.execute("""
                WITH topic_expanded AS (
                    SELECT 
                        jsonb_array_elements_text(topics) AS topic,
                        confidence,
                        e.doc_id
                    FROM embeddings em
                    JOIN elements e ON em.element_pk = e.element_pk
                    WHERE topics IS NOT NULL AND jsonb_array_length(topics) > 0
                )
                SELECT 
                    topic,
                    COUNT(*) as embedding_count,
                    COUNT(DISTINCT doc_id) as document_count,
                    AVG(confidence) as avg_confidence
                FROM topic_expanded
                GROUP BY topic
                ORDER BY embedding_count DESC
            """)

            statistics = {}
            for row in self.cursor.fetchall():
                statistics[row[0]] = {
                    'embedding_count': int(row[1]),
                    'document_count': int(row[2]),
                    'avg_embedding_confidence': float(row[3])
                }

            return statistics

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
        if not self.cursor:
            raise ValueError("Database not initialized")

        try:
            self.cursor.execute(
                "SELECT topics FROM embeddings WHERE element_pk = %s",
                (element_pk,)
            )

            row = self.cursor.fetchone()
            if row is None or row[0] is None:
                return []

            try:
                return row[0] if isinstance(row[0], list) else []
            except (json.JSONDecodeError, TypeError):
                return []

        except Exception as e:
            logger.error(f"Error getting topics for element {element_pk}: {str(e)}")
            return []

    # ========================================
    # EXISTING HIERARCHY METHODS (unchanged)
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
        from .element_element import ElementBase, ElementHierarchical

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

    def _get_element_ancestry_path(self, element_pk: int) -> List["ElementBase"]:
        """
        Get the complete ancestry path for an element, from root to the element itself.

        Uses parent_id to find parents instead of relationships.
        """
        from .element_element import ElementBase

        # Get the element
        element_dict = self.get_element(element_pk)
        if not element_dict:
            return []

        # Convert to ElementElement instance
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

            # Convert to ElementElement
            parent = ElementBase(**parent_dict)

            # Add to visited set
            visited.add(parent_pk)

            # Add parent to the beginning of the ancestry list (root first)
            ancestry.insert(0, parent)

            # Move up to the parent
            current_pk = parent_id

        return ancestry
