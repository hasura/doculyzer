import json
import logging
import os
from typing import Optional, Dict, Any, List, Tuple

import time

from .base import DocumentDatabase
from .element_relationship import ElementRelationship
from ..config import Config

logger = logging.getLogger(__name__)

# Try to import PostgreSQL library
try:
    # noinspection PyUnresolvedReferences
    import psycopg2
    # noinspection PyUnresolvedReferences
    import psycopg2.extras
    # noinspection PyUnresolvedReferences
    import psycopg2.extensions
    # noinspection PyUnresolvedReferences
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    PSYCOPG2_AVAILABLE = True
except ImportError:
    logger.warning("psycopg2 not available. Install with 'pip install psycopg2-binary'.")
    PSYCOPG2_AVAILABLE = False

config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))


class PostgreSQLDocumentDatabase(DocumentDatabase):
    """PostgreSQL implementation of document database."""

    """
    Key updates to the PostgreSQL implementation:

    1. Improved get_element method to handle both element_id (string) and element_pk (integer)
    2. Enhanced get_outgoing_relationships method to use JOIN for efficiently retrieving target information
    3. Updated search_by_embedding methods to return element_pk instead of element_id for consistency
    """

    def get_element(self, element_id_or_pk: str | int) -> Optional[Dict[str, Any]]:
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
            element["metadata"] = json.loads(element["metadata"])
        except (json.JSONDecodeError, TypeError):
            element["metadata"] = {}

        return element

    def get_outgoing_relationships(self, element_pk: int) -> List[ElementRelationship]:
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
                        rel_dict["metadata"] = json.loads(rel_dict["metadata"])
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
                    source_element_pk=element_pk,
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

    def _search_by_pgvector(self, query_embedding: List[float], limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search embeddings using pgvector similarity with filtering.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_criteria: Optional dictionary with criteria to filter results

        Returns:
            List of (element_pk, similarity_score) tuples
        """
        # Convert embedding to JSON array for casting to vector
        embedding_json = json.dumps(query_embedding)

        try:
            # Start building the query
            sql = """
            SELECT e.element_pk, 1 - (em.vector_embedding <=> %s::vector) as similarity
            FROM embeddings em
            JOIN elements e ON e.element_pk = em.element_pk
            JOIN documents d ON e.doc_id = d.doc_id
            """
            params = [embedding_json]

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

            # Add ORDER BY and LIMIT
            sql += """
            ORDER BY em.vector_embedding <=> %s::vector
            LIMIT %s
            """
            params.extend([embedding_json, limit])

            # Execute the query
            self.cursor.execute(sql, params)

            # Return element_pk instead of element_id for consistency with changes
            return [(row["element_pk"], row["similarity"]) for row in self.cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error using pgvector for search: {str(e)}")
            raise

    def _search_by_similarity_function(self, query_embedding: List[float], limit: int = 10,
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

        similarities = [
            (row["element_pk"], self._cosine_similarity(query_embedding, row["embedding"]))
            for row in self.cursor.fetchall()
        ]
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

    def __init__(self, conn_params: Dict[str, Any]):
        """
        Initialize PostgreSQL document database.

        Args:
            conn_params: Connection parameters for PostgreSQL
                (host, port, dbname, user, password)
        """
        self.conn_params = conn_params
        self.conn = None
        self.cursor = None
        self.vector_extension = None
        self.vector_dimension = config.config.get('embedding', {}).get('dimensions', 384)
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
            metadata_json = json.dumps(document.get("metadata", {}))

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
            doc["metadata"] = json.loads(doc["metadata"])
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
                element["metadata"] = json.loads(element["metadata"])
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
                relationship["metadata"] = json.loads(relationship["metadata"])
            except (json.JSONDecodeError, TypeError):
                relationship["metadata"] = {}

            relationships.append(relationship)

        return relationships

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
        """Find documents matching query."""
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
                    # Metadata filters require special handling with JSONB
                    for meta_key, meta_value in value.items():
                        conditions.append(f"metadata->>'%s' = %s")
                        params.extend([meta_key, str(meta_value)])
                else:
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
                doc["metadata"] = json.loads(doc["metadata"])
            except (json.JSONDecodeError, TypeError):
                doc["metadata"] = {}

            documents.append(doc)

        return documents

    def find_elements(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find elements matching query."""
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
                    # Metadata filters require special handling with JSONB
                    for meta_key, meta_value in value.items():
                        conditions.append(f"metadata->>'%s' = %s")
                        params.extend([meta_key, str(meta_value)])
                elif key == "element_type" and isinstance(value, list):
                    # Handle list of element types
                    placeholders = ', '.join(['%s'] * len(value))
                    conditions.append(f"element_type IN ({placeholders})")
                    params.extend(value)
                else:
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
                element["metadata"] = json.loads(element["metadata"])
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
                element["metadata"] = json.loads(element["metadata"])
            except (json.JSONDecodeError, TypeError):
                element["metadata"] = {}

            elements.append(element)

        return elements

    def store_embedding(self, element_pk: int, embedding: List[float]) -> None:
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
                (element_pk, embedding, dimensions, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (element_pk) DO UPDATE 
                SET embedding = EXCLUDED.embedding, 
                    dimensions = EXCLUDED.dimensions, 
                    created_at = EXCLUDED.created_at
                """,
                (
                    element_pk,
                    embedding_json,
                    len(embedding),
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

    def get_embedding(self, element_pk: int) -> Optional[List[float]]:
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
            return json.loads(row["embedding"])
        except (json.JSONDecodeError, TypeError):
            return None

    def search_by_embedding(self, query_embedding: List[float], limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> list[tuple[int, float]]:
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

            # Modified embeddings table with element_pk as reference
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                element_pk INTEGER PRIMARY KEY REFERENCES elements(element_pk) ON DELETE CASCADE,
                embedding JSONB,
                dimensions INTEGER,
                created_at DOUBLE PRECISION
            )
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

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np

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
