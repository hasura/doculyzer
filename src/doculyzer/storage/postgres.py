import json
import logging
import os
import time
from typing import Optional, Dict, Any, List, Tuple

from .base import DocumentDatabase
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

    def initialize(self) -> None:
        """Initialize the database by connecting and creating tables if they don't exist."""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL support")

        # Connect to PostgreSQL
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            logger.info(
                f"Connected to PostgreSQL at {self.conn_params.get('host', 'localhost')}:{self.conn_params.get('port', 5432)}")
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

        self.cursor.execute(
            "SELECT * FROM elements WHERE doc_id = %s ORDER BY element_id",
            (doc_id,)
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

    def get_element(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Get element by ID."""
        if not self.cursor:
            raise ValueError("Database not initialized")

        self.cursor.execute(
            "SELECT * FROM elements WHERE element_id = %s",
            (element_id,)
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

    def search_by_embedding(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[str, float]]:
        """Search elements by embedding similarity using available method."""
        if not self.cursor:
            raise ValueError("Database not initialized")

        try:
            if self.vector_extension == "pgvector":
                return self._search_by_pgvector(query_embedding, limit)
            else:
                return self._search_by_similarity_function(query_embedding, limit)
        except Exception as e:
            logger.error(f"Error searching by embedding: {str(e)}")
            # Fall back to non-vector search
            return self._search_by_similarity_function(query_embedding, limit)

    def _search_by_pgvector(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[str, float]]:
        """Search embeddings using pgvector similarity."""
        # Convert embedding to JSON array for casting to vector
        embedding_json = json.dumps(query_embedding)

        try:
            # Use cosine distance by default
            self.cursor.execute(
                """
                SELECT e.element_id, 1 - (em.vector_embedding <=> %s::vector) as similarity
                FROM embeddings em
                JOIN elements e ON e.element_pk = em.element_pk
                ORDER BY em.vector_embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding_json, embedding_json, limit)
            )

            return [(row["element_id"], row["similarity"]) for row in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error using pgvector for search: {str(e)}")
            raise

    def _search_by_similarity_function(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[str, float]]:
        """Fall back to calculating similarity in Python."""
        # Get all embeddings
        self.cursor.execute("""
            SELECT em.element_pk, em.embedding, e.element_id
            FROM embeddings em
            JOIN elements e ON e.element_pk = em.element_pk
        """)

        embeddings = []
        for row in self.cursor.fetchall():
            try:
                embedding = json.loads(row["embedding"])
                embeddings.append((row["element_id"], embedding))
            except (json.JSONDecodeError, TypeError):
                continue

        # Calculate cosine similarity for each embedding
        similarities = []
        for element_id, embedding in embeddings:
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((element_id, similarity))

        # Sort by similarity (highest first) and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

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

    def search_by_text(self, search_text: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Search elements by semantic similarity to the provided text.

        This method combines text-to-embedding conversion and embedding search
        into a single convenient operation.

        Args:
            search_text: Text to search for semantically
            limit: Maximum number of results

        Returns:
            List of (element_id, similarity_score) tuples
        """
        if not self.cursor:
            raise ValueError("Database not initialized")

        try:
            # Import necessary modules
            from doculyzer.embeddings import get_embedding_generator

            # Get config from the connection parameters
            # This assumes config is accessible, otherwise it would need to be passed in
            config = self.conn_params.get('config')
            if not config:
                from doculyzer import Config
                config = Config()

            # Get the embedding generator
            embedding_generator = get_embedding_generator(config)

            # Generate embedding for the search text
            query_embedding = embedding_generator.generate(search_text)

            # Use the embedding to search
            return self.search_by_embedding(query_embedding, limit)

        except Exception as e:
            logger.error(f"Error in semantic search by text: {str(e)}")
            # Return empty list on error
            return []
