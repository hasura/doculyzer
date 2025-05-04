import datetime
import json
import logging
import os
from typing import Optional, Dict, Any, List, Tuple

import time

from .element_relationship import ElementRelationship

logger = logging.getLogger(__name__)

# Try to import the preferred SQLite library based on config
try:
    from ..config import Config

    config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
    # Check if we should use sqlean
    use_sqlean = config.config.get("storage", {}).get("sqlite_extensions", {}).get("use_sqlean", False)

    if use_sqlean:
        try:
            # noinspection PyPackageRequirements
            import sqlean as sqlite3

            logger.info("Using sqlean as SQLite provider (with extension support)")
        except ImportError:
            logger.warning("sqlean requested but not installed. Falling back to standard sqlite3.")
            import sqlite3
    else:
        import sqlite3
except Exception as e:
    logger.warning(f"Error configuring SQLite provider: {str(e)}. Using standard sqlite3.")
    import sqlite3

# Try to import vector search extensions
try:
    # noinspection PyUnresolvedReferences
    import sqlite_vec

    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

try:
    # noinspection PyUnresolvedReferences
    import sqlite_vss

    SQLITE_VSS_AVAILABLE = True
except ImportError:
    SQLITE_VSS_AVAILABLE = False

from .base import DocumentDatabase


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()  # Convert date/datetime to ISO 8601 string
        return super().default(obj)


class SQLiteDocumentDatabase(DocumentDatabase):
    """SQLite implementation of document database."""

    def get_outgoing_relationships(self, element_pk: int) -> List[ElementRelationship]:
        """
        Find all relationships where the specified element_pk is the source.

        Implementation for SQLite database using JOIN to efficiently retrieve target information.

        Args:
            element_pk: The primary key of the element

        Returns:
            List of ElementRelationship objects where the specified element is the source
        """

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

        # Find relationships with target element information using JOIN
        # This query joins the relationships table with the elements table
        # to get information about target elements in one go
        self.cursor = self.conn.execute(
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
                r.source_id = ?
            """,
            (element_id,)
        )

        for row in self.cursor.fetchall():
            # Convert to dictionary
            rel_dict = dict(row)

            # Remove SQLite's rowid if present
            if "rowid" in rel_dict:
                del rel_dict["rowid"]

            # Convert metadata from JSON if needed
            try:
                if isinstance(rel_dict.get("metadata"), str):
                    rel_dict["metadata"] = json.loads(rel_dict["metadata"])
            except (json.JSONDecodeError, TypeError):
                rel_dict["metadata"] = {}

            # Extract target element information from the joined query results
            target_element_pk = rel_dict.get("target_element_pk")
            target_element_type = rel_dict.get("target_element_type")

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
                target_content_preview=rel_dict.get("target_content_preview", ""),
                doc_id=rel_dict.get("doc_id"),
                metadata=rel_dict.get("metadata", {}),
                is_source=True
            )

            relationships.append(relationship)

        return relationships

    def __init__(self, db_path: str):
        """
        Initialize SQLite document database.

        Args:
            db_path: Path to SQLite database file
        """
        self.cursor = None
        self.db_path = db_path
        self.conn = None
        self.vector_extension = None
        self.embedding_generator = None

    def initialize(self) -> None:
        """Initialize the database by creating tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(os.path.join(self.db_path, 'document_db.sqlite'))
        self.conn.row_factory = sqlite3.Row

        # Check if extension loading is supported
        auto_discover = config.config.get("storage", {}).get("sqlite_extensions", {}).get("auto_discover", True)
        extension_loading_supported = True

        try:
            self.conn.enable_load_extension(True)
        except (AttributeError, sqlite3.OperationalError) as e:
            extension_loading_supported = False
            logger.warning(f"SQLite extension loading not supported: {str(e)}")

            if isinstance(sqlite3.__name__, str) and "sqlean" not in sqlite3.__name__:
                logger.info("Consider using sqlean.py for SQLite extension support.")
                logger.info("Set storage.sqlite_extensions.use_sqlean to True in your config file.")

        # Only attempt to load extensions if supported and auto-discover is enabled
        if extension_loading_supported and auto_discover:
            self._load_vector_extensions()
        else:
            self.vector_extension = None
            logger.info("Using native vector search implementation (no extensions)")

        self._create_tables()
        self._create_vector_tables()

    def _load_vector_extensions(self):
        """Load available vector search extensions."""
        try:
            # Try sqlite-vec first (newer, runs anywhere)
            if SQLITE_VEC_AVAILABLE:
                try:
                    sqlite_vec.load(self.conn)
                    self.vector_extension = "vec0"
                    logger.info("SQLite vector search extension 'vec0' loaded successfully")
                    return
                except Exception as e:
                    logger.debug(f"Failed to load sqlite-vec extension: {str(e)}")

            # Try sqlite-vss as fallback
            if SQLITE_VSS_AVAILABLE:
                try:
                    sqlite_vss.load(self.conn)
                    self.vector_extension = "vss0"
                    logger.info("SQLite vector search extension 'vss0' loaded successfully")
                    return
                except Exception as e:
                    logger.debug(f"Failed to load sqlite-vss extension: {str(e)}")

            logger.info("SQLite vector search extensions not available. Using native implementation.")
            self.vector_extension = None
        except Exception as e:
            logger.info(f"Error loading SQLite extensions: {str(e)}. Using native implementation.")
            self.vector_extension = None
        finally:
            # Disable extension loading after we're done
            self.conn.enable_load_extension(False)

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_last_processed_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about when a document was last processed."""
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            cursor = self.conn.execute(
                """
                SELECT * FROM processing_history 
                WHERE source_id = ?
                """,
                (source_id,)
            )

            row = cursor.fetchone()
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
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            # Check if record exists
            cursor = self.conn.execute(
                "SELECT processing_count FROM processing_history WHERE source_id = ?",
                (source_id,)
            )

            row = cursor.fetchone()
            processing_count = 1  # Default for new records

            if row is not None:
                processing_count = row[0] + 1

                # Update existing record
                cursor.execute(
                    """
                    UPDATE processing_history
                    SET content_hash = ?, last_modified = ?, processing_count = ?
                    WHERE source_id = ?
                    """,
                    (content_hash, time.time(), processing_count, source_id)
                )
            else:
                # Insert new record
                cursor.execute(
                    """
                    INSERT INTO processing_history
                    (source_id, content_hash, last_modified, processing_count)
                    VALUES (?, ?, ?, ?)
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
        if not self.conn:
            raise ValueError("Database not initialized")

        source = document.get("source", "")
        content_hash = document.get("content_hash", "")

        # Check if document already exists with this source
        cursor = self.conn.execute(
            "SELECT doc_id FROM documents WHERE source = ?",
            (source,)
        )
        existing_doc = cursor.fetchone()

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

        # Begin transaction
        cursor.execute("BEGIN TRANSACTION")

        try:
            # Store document
            metadata_json = json.dumps(document.get("metadata", {}), cls=DateTimeEncoder)

            cursor.execute(
                """
                INSERT OR REPLACE INTO documents 
                (doc_id, doc_type, source, content_hash, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
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

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO elements 
                    (element_id, doc_id, element_type, parent_id, content_preview, 
                     content_location, content_hash, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

                # Get the last inserted auto-increment ID
                generated_pk = cursor.lastrowid
                # Store it back into the dictionary
                element['element_pk'] = generated_pk

            # Store relationships
            for relationship in relationships:
                relationship_id = relationship["relationship_id"]
                metadata_json = json.dumps(relationship.get("metadata", {}))

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO relationships 
                    (relationship_id, source_id, relationship_type, target_reference, metadata)
                    VALUES (?, ?, ?, ?, ?)
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
            cursor.execute("COMMIT TRANSACTION")

            # Update processing history (after successful commit)
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
        Update an existing document by removing it and then using store_document to reinsert.
        This avoids foreign key constraint issues during partial updates.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        # source = document.get("source", "")
        # content_hash = document.get("content_hash", "")

        # Check if document exists
        cursor = self.conn.execute("SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,))
        if cursor.fetchone() is None:
            raise ValueError(f"Document not found: {doc_id}")

        # Begin transaction
        self.conn.execute("BEGIN TRANSACTION")

        try:
            # Temporarily disable foreign key constraints
            self.conn.execute("PRAGMA foreign_keys = OFF")

            # Get all elements for this document
            cursor = self.conn.execute("SELECT element_id FROM elements WHERE doc_id = ?", (doc_id,))
            element_ids = [row[0] for row in cursor.fetchall()]

            # Delete relationships related to this document's elements
            if element_ids:
                placeholders = ', '.join(['?'] * len(element_ids))
                self.conn.execute(f"DELETE FROM relationships WHERE source_id IN ({placeholders})", element_ids)

            # Delete embeddings for this document's elements
            if element_ids:
                placeholders = ', '.join(['?'] * len(element_ids))
                self.conn.execute(f"DELETE FROM embeddings WHERE element_id IN ({placeholders})", element_ids)

                # Also delete from extension tables if they exist
                if self.vector_extension == "vec0":
                    try:
                        self.conn.execute(f"DELETE FROM embeddings_vec WHERE rowid IN ({placeholders})", element_ids)
                    except Exception as e:
                        logger.debug(f"Error cleaning up embeddings_vec: {str(e)}")
                elif self.vector_extension == "vss0":
                    try:
                        self.conn.execute(f"DELETE FROM embeddings_vss WHERE rowid IN ({placeholders})", element_ids)
                    except Exception as e:
                        logger.debug(f"Error cleaning up embeddings_vss: {str(e)}")

            # Delete all elements for this document
            self.conn.execute("DELETE FROM elements WHERE doc_id = ?", (doc_id,))

            # Delete the document itself
            self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

            # Re-enable foreign key constraints
            self.conn.execute("PRAGMA foreign_keys = ON")

            # Commit the deletion part of the transaction
            self.conn.commit()

            # Now use store_document to insert everything
            # This will also update the processing history
            self.store_document(document, elements, relationships)

        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            # Re-enable foreign keys in case of error
            self.conn.execute("PRAGMA foreign_keys = ON")
            logger.error(f"Error updating document {doc_id}: {str(e)}")
            raise
        finally:
            # Ensure foreign keys are re-enabled
            self.conn.execute("PRAGMA foreign_keys = ON")

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        if not self.conn:
            raise ValueError("Database not initialized")

        cursor = self.conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?",
            (doc_id,)
        )

        row = cursor.fetchone()
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
        if not self.conn:
            raise ValueError("Database not initialized")

        cursor = self.conn.execute(
            "select e.* from documents join main.elements e on documents.doc_id = e.doc_id WHERE documents.source = ? OR documents.doc_id = ?",
            (doc_id, doc_id)
        )

        elements = []
        for row in cursor:
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
        if not self.conn:
            raise ValueError("Database not initialized")

        # First get all element IDs for the document
        cursor = self.conn.execute(
            "SELECT element_id FROM elements WHERE doc_id = ?",
            (doc_id,)
        )

        element_ids = [row[0] for row in cursor]

        if not element_ids:
            return []

        # Create placeholders for SQL IN clause
        placeholders = ', '.join(['?'] * len(element_ids))

        # Find relationships involving these elements
        cursor = self.conn.execute(
            f"SELECT * FROM relationships WHERE source_id IN ({placeholders})",
            element_ids
        )

        relationships = []
        for row in cursor:
            relationship = dict(row)

            # Convert metadata from JSON
            try:
                relationship["metadata"] = json.loads(relationship["metadata"])
            except (json.JSONDecodeError, TypeError):
                relationship["metadata"] = {}

            relationships.append(relationship)

        return relationships

    def get_element(self, element_pk: int | str) -> Optional[Dict[str, Any]]:
        """Get element by ID."""
        if not self.conn:
            raise ValueError("Database not initialized")

        cursor = self.conn.execute(
            "SELECT * FROM elements WHERE element_pk = ? OR element_id = ?",
            (element_pk if str(element_pk).isnumeric() else -1, str(element_pk))
        )

        row = cursor.fetchone()
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
        if not self.conn:
            raise ValueError("Database not initialized")

        # Start with base query
        sql = "SELECT * FROM documents"
        params = []

        # Apply filters if provided
        if query:
            conditions = []

            for key, value in query.items():
                if key == "metadata":
                    # Metadata filters require special handling
                    for meta_key, meta_value in value.items():
                        # Use JSON_EXTRACT to query JSON metadata
                        conditions.append(f"JSON_EXTRACT(metadata, '$.{meta_key}') = ?")
                        params.append(json.dumps(meta_value))
                else:
                    conditions.append(f"{key} = ?")
                    params.append(value)

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        # Add limit
        sql += f" LIMIT {limit}"

        # Execute query
        cursor = self.conn.execute(sql, params)

        documents = []
        for row in cursor:
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
        if not self.conn:
            raise ValueError("Database not initialized")

        # Start with base query
        sql = "SELECT * FROM elements"
        params = []

        # Apply filters if provided
        if query:
            conditions = []

            for key, value in query.items():
                if key == "metadata":
                    # Metadata filters require special handling
                    for meta_key, meta_value in value.items():
                        # Use JSON_EXTRACT to query JSON metadata
                        conditions.append(f"JSON_EXTRACT(metadata, '$.{meta_key}') = ?")
                        params.append(json.dumps(meta_value))
                else:
                    conditions.append(f"{key} = ?")
                    params.append(value)

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        # Add limit
        sql += f" LIMIT {limit}"

        # Execute query
        cursor = self.conn.execute(sql, params)

        elements = []
        for row in cursor:
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
        if not self.conn:
            raise ValueError("Database not initialized")

        cursor = self.conn.execute(
            "SELECT * FROM elements WHERE content_preview LIKE ? LIMIT ?",
            (f"%{search_text}%", limit)
        )

        elements = []
        for row in cursor:
            element = dict(row)

            # Convert metadata from JSON
            try:
                element["metadata"] = json.loads(element["metadata"])
            except (json.JSONDecodeError, TypeError):
                element["metadata"] = {}

            elements.append(element)

        return elements

    def store_embedding(self, element_pk: str, embedding: List[float]) -> None:
        """Store embedding for an element."""
        if not self.conn:
            raise ValueError("Database not initialized")

        # Verify element exists
        cursor = self.conn.execute(
            "SELECT element_pk FROM elements WHERE element_pk = ?",
            (element_pk,)
        )

        if cursor.fetchone() is None:
            raise ValueError(f"Element not found: {element_pk}")

        # Store embedding in the main embeddings table
        embedding_blob = self._encode_embedding(embedding)

        self.conn.execute(
            """
            INSERT OR REPLACE INTO embeddings 
            (element_pk, embedding, dimensions, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                element_pk,
                embedding_blob,
                len(embedding),
                time.time()
            )
        )

        # Store embedding in extension tables if available
        if self.vector_extension:
            # Convert embedding to the required format
            embedding_json = json.dumps(embedding)

            try:
                if self.vector_extension == "vec0":
                    # For sqlite-vec
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO embeddings_vec (rowid, embedding)
                        VALUES (?, ?)
                        """,
                        (element_pk, embedding_json)
                    )
                elif self.vector_extension == "vss0":
                    # For sqlite-vss
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO embeddings_vss (rowid, embedding)
                        VALUES (?, ?)
                        """,
                        (element_pk, embedding_json)
                    )
            except Exception as e:
                logger.warning(f"Error storing embedding in extension table: {str(e)}")

        self.conn.commit()

    def get_embedding(self, element_pk: int) -> Optional[List[float]]:
        """Get embedding for an element."""
        if not self.conn:
            raise ValueError("Database not initialized")

        cursor = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE element_pk = ?",
            (element_pk,)
        )

        row = cursor.fetchone()
        if row is None:
            return None

        return self._decode_embedding(row["embedding"])

    def search_by_embedding(self, query_embedding: List[float], limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> list[tuple[str, float]] | list[Any]:
        """Search elements by embedding similarity using available method."""
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            if self.vector_extension == "vec0":
                return self._search_by_vec_extension(query_embedding, limit, filter_criteria)
            elif self.vector_extension == "vss0":
                return self._search_by_vss_extension(query_embedding, limit, filter_criteria)
            else:
                # Use native implementation
                return self._search_by_embedding_native(query_embedding, limit, filter_criteria)
        except Exception as e:
            logger.error(f"Error searching by embedding: {str(e)}")
            # Fall back to native implementation
            try:
                return self._search_by_embedding_native(query_embedding, limit, filter_criteria)
            except Exception as e2:
                logger.error(f"Error in fallback search: {str(e2)}")
                return []

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all associated elements and relationships."""
        if not self.conn:
            raise ValueError("Database not initialized")

        # Check if document exists
        cursor = self.conn.execute(
            "SELECT doc_id FROM documents WHERE doc_id = ?",
            (doc_id,)
        )

        if cursor.fetchone() is None:
            return False

        # Begin transaction
        self.conn.execute("BEGIN TRANSACTION")

        try:
            # Get all element IDs for this document
            cursor = self.conn.execute(
                "SELECT element_id FROM elements WHERE doc_id = ?",
                (doc_id,)
            )

            element_ids = [row[0] for row in cursor]

            # Delete embeddings for these elements
            if element_ids:
                # Create placeholders for SQL IN clause
                placeholders = ', '.join(['?'] * len(element_ids))

                self.conn.execute(
                    f"DELETE FROM embeddings WHERE element_id IN ({placeholders})",
                    element_ids
                )

                # Delete from extension tables if they exist
                if self.vector_extension == "vec0":
                    try:
                        self.conn.execute(f"DELETE FROM embeddings_vec WHERE rowid IN ({placeholders})", element_ids)
                    except Exception as e:
                        logger.debug(f"Error cleaning up embeddings_vec: {str(e)}")
                elif self.vector_extension == "vss0":
                    try:
                        self.conn.execute(f"DELETE FROM embeddings_vss WHERE rowid IN ({placeholders})", element_ids)
                    except Exception as e:
                        logger.debug(f"Error cleaning up embeddings_vss: {str(e)}")

                # Delete relationships involving these elements
                self.conn.execute(
                    f"DELETE FROM relationships WHERE source_id IN ({placeholders})",
                    element_ids
                )

            # Delete elements
            self.conn.execute(
                "DELETE FROM elements WHERE doc_id = ?",
                (doc_id,)
            )

            # Delete document
            self.conn.execute(
                "DELETE FROM documents WHERE doc_id = ?",
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

    def store_relationship(self, relationship: Dict[str, Any]) -> None:
        """
        Store a relationship between elements.

        Args:
            relationship: Relationship data
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            # Convert metadata to JSON
            metadata_json = json.dumps(relationship.get("metadata", {}))

            self.conn.execute(
                """
                INSERT OR REPLACE INTO relationships
                (relationship_id, source_id, relationship_type, target_reference, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    relationship["relationship_id"],
                    relationship.get("source_id", ""),
                    relationship.get("relationship_type", ""),
                    relationship.get("target_reference", ""),
                    metadata_json
                )
            )

            self.conn.commit()
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
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            # Start with basic query to delete source relationships
            query = "DELETE FROM relationships WHERE source_id = ?"
            params = [element_id]

            # Add relationship type filter if provided
            if relationship_type:
                query += " AND relationship_type = ?"
                params.append(relationship_type)

            # Delete source relationships
            self.conn.execute(query, params)

            # Also delete relationships where this element is the target
            query = "DELETE FROM relationships WHERE target_reference = ?"
            params = [element_id]

            # Add relationship type filter if provided
            if relationship_type:
                query += " AND relationship_type = ?"
                params.append(relationship_type)

            # Delete target relationships
            self.conn.execute(query, params)

            self.conn.commit()
        except Exception as e:
            logger.error(f"Error deleting relationships for element {element_id}: {str(e)}")
            raise

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        # Documents table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            doc_type TEXT,
            source TEXT,
            content_hash TEXT,
            metadata TEXT,
            created_at REAL,
            updated_at REAL
        )
        """)

        # --- MODIFIED: Elements table ---
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS elements (
            element_pk INTEGER PRIMARY KEY AUTOINCREMENT, -- New auto-increment integer PK
            element_id TEXT UNIQUE NOT NULL,             -- Original string ID, now unique & indexed
            doc_id TEXT,
            element_type TEXT,
            parent_id TEXT,                              -- Still TEXT, references element_id
            content_preview TEXT,
            content_location TEXT,
            content_hash TEXT,
            metadata TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE,
            -- parent_id (TEXT) now correctly references element_id (TEXT UNIQUE)
            FOREIGN KEY (parent_id) REFERENCES elements (element_id)
        )
        """)
        # --- END MODIFICATION ---

        # Create index on doc_id for faster lookups (remains unchanged)
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_elements_doc_id ON elements (doc_id)
        """)

        # Create index on parent_id (TEXT) for faster lookups (remains unchanged)
        # This is useful if you often query elements by their parent's string ID
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_elements_parent_id ON elements (parent_id)
        """)

        # Create index on element_type for faster lookups (remains unchanged)
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_elements_type ON elements (element_type)
        """)

        # --- MODIFIED: Relationships table ---
        # Foreign key remains correct as source_id (TEXT) references elements.element_id (TEXT UNIQUE)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            relationship_id TEXT PRIMARY KEY,
            source_id TEXT,
            relationship_type TEXT,
            target_reference TEXT,
            metadata TEXT,
            FOREIGN KEY (source_id) REFERENCES elements (element_id) ON DELETE CASCADE
        )
        """)
        # --- END MODIFICATION ---

        # Create index on source_id for faster lookups (remains unchanged)
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships (source_id)
        """)

        # Create index on relationship_type for faster lookups (remains unchanged)
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships (relationship_type)
        """)

        # --- MODIFIED: Embeddings table ---
        # Foreign key remains correct as element_id (TEXT) references elements.element_id (TEXT UNIQUE)
        # If you wanted this to reference the integer PK, you'd change element_id here to
        # element_pk INTEGER and update the FOREIGN KEY accordingly.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
            element_pk INTEGER PRIMARY KEY, -- Links to elements.element_pk (INTEGER)
            embedding BLOB,
            dimensions INTEGER,
            created_at REAL,
            FOREIGN KEY (element_pk) REFERENCES elements (element_pk) ON DELETE CASCADE
        )
        """)
        # --- END MODIFICATION ---

        # Add processing history table (remains unchanged)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS processing_history (
            source_id TEXT PRIMARY KEY,
            content_hash TEXT,
            last_modified REAL,
            processing_count INTEGER DEFAULT 1
        )
        """)

        # Add index on source_id for faster lookups (remains unchanged)
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_processing_history_source_id ON processing_history (source_id)
        """)

        # Enable foreign key constraints (remains unchanged)
        self.conn.execute("PRAGMA foreign_keys = ON")

        # Commit changes (remains unchanged)
        self.conn.commit()

    def _create_vector_tables(self) -> None:
        """Create vector tables based on available extension."""
        try:
            # Get dimensions from existing embeddings or use default
            dimensions = self._get_embedding_dimensions()

            if self.vector_extension == "vec0":
                # For sqlite-vec
                self.conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_vec 
                USING vec0(embedding FLOAT[{dimensions}])
                """)
                logger.info(f"Created vector table using vec0 with {dimensions} dimensions")

            elif self.vector_extension == "vss0":
                # For sqlite-vss
                self.conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_vss 
                USING vss0(embedding({dimensions}))
                """)
                logger.info(f"Created vector table using vss0 with {dimensions} dimensions")

            # Populate vector table with existing embeddings
            self._populate_vector_tables()

        except Exception as e:
            logger.error(f"Error creating vector tables: {str(e)}")

    @staticmethod
    def _get_embedding_dimensions() -> int:
        """Get embedding dimensions from config or use default."""
        return config.config.get('embedding', {}).get('dimensions', 384)

    def _populate_vector_tables(self) -> None:
        """Populate vector tables with existing embeddings."""
        try:
            # Check if we have any embeddings to populate
            cursor = self.conn.execute("SELECT COUNT(*) as count FROM embeddings")
            row = cursor.fetchone()
            if not row or row["count"] == 0:
                return

            # Get all embeddings
            cursor = self.conn.execute("SELECT element_pk, embedding FROM embeddings")

            # Begin transaction
            self.conn.execute("BEGIN TRANSACTION")

            try:
                # Process each embedding
                for row in cursor:
                    element_pk = row["element_pk"]
                    embedding_blob = row["embedding"]
                    embedding = self._decode_embedding(embedding_blob)
                    embedding_json = json.dumps(embedding)

                    if self.vector_extension == "vec0":
                        # First check if a row with this rowid already exists
                        check_cursor = self.conn.execute(
                            "SELECT rowid FROM embeddings_vec WHERE rowid = ?",
                            (element_pk,)
                        )

                        if check_cursor.fetchone():
                            # Update existing record
                            self.conn.execute(
                                """
                                UPDATE embeddings_vec 
                                SET embedding = ?
                                WHERE rowid = ?
                                """,
                                (embedding_json, element_pk)
                            )
                        else:
                            # Insert new record
                            self.conn.execute(
                                """
                                INSERT INTO embeddings_vec (rowid, embedding)
                                VALUES (?, ?)
                                """,
                                (element_pk, embedding_json)
                            )
                    elif self.vector_extension == "vss0":
                        self.conn.execute(
                            """
                            INSERT OR REPLACE INTO embeddings_vss (rowid, embedding)
                            VALUES (?, ?)
                            """,
                            (element_pk, embedding_json)
                        )

                # Commit transaction
                self.conn.commit()
                logger.info("Successfully populated vector tables with existing embeddings")

            except Exception as e:
                # Rollback on error
                self.conn.rollback()
                logger.error(f"Error populating vector tables: {str(e)}")

        except Exception as e:
            logger.error(f"Error getting embeddings for vector tables: {str(e)}")

    def _search_by_vec_extension(self, query_embedding: List[float], limit: int = 10,
                                 filter_criteria: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        """Use the vec0 extension for vector search with filtering."""
        # Convert embedding to JSON string
        query_json = json.dumps(query_embedding)

        try:
            # Start building the query
            query = """
            SELECT
                e.element_pk,
                vec_results.distance
            FROM
                embeddings_vec AS vec_results
            JOIN elements AS e ON 
                vec_results.rowid = e.element_pk 
            JOIN documents AS d ON
                d.doc_id = e.doc_id
            WHERE
                vec_results.embedding MATCH ? AND k = ?
            """
            params = [query_json, limit]

            # Add filter criteria if provided
            if filter_criteria:
                for key, value in filter_criteria.items():
                    if key == "element_type" and isinstance(value, list):
                        # Handle list of allowed element types
                        placeholders = ', '.join(['?'] * len(value))
                        query += f" AND e.element_type IN ({placeholders})"
                        params.extend(value)
                    elif key == "doc_id" and isinstance(value, list):
                        # Handle list of document IDs to include
                        placeholders = ', '.join(['?'] * len(value))
                        query += f" AND e.doc_id IN ({placeholders})"
                        params.extend(value)
                    elif key == "exclude_doc_id" and isinstance(value, list):
                        # Handle list of document IDs to exclude
                        placeholders = ', '.join(['?'] * len(value))
                        query += f" AND e.doc_id NOT IN ({placeholders})"
                        params.extend(value)
                    elif key == "exclude_doc_source" and isinstance(value, list):
                        # Handle list of document IDs to exclude
                        placeholders = ', '.join(['?'] * len(value))
                        query += f" AND d.source NOT IN ({placeholders})"
                        params.extend(value)
                    else:
                        # Simple equality filter
                        query += f" AND e.{key} = ?"
                        params.append(value)

            # Add order
            query += " ORDER BY vec_results.distance"

            # Execute query
            cursor = self.conn.execute(query, params)

            return [(row["element_pk"], 1 - row["distance"]) for row in cursor]
        except Exception as e:
            logger.error(f"Error using vec0 extension for search: {str(e)}")
            raise

    def _search_by_vss_extension(self, query_embedding: List[float], limit: int = 10,
                                 filter_criteria: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        """Use the vss0 extension for vector search with filtering."""
        # Convert embedding to JSON string
        query_json = json.dumps(query_embedding)

        try:
            # Start building the query
            query = """
            SELECT e.element_pk, vss_search(ev.embedding, ?) AS similarity
            FROM embeddings_vss ev
            JOIN elements e ON e.element_pk = ev.rowid
            JOIN documents AS d ON
                d.doc_id = e.doc_id
            WHERE 1=1
            """
            params = [query_json]

            # Add filter criteria if provided
            if filter_criteria:
                for key, value in filter_criteria.items():
                    if key == "element_type" and isinstance(value, list):
                        # Handle list of allowed element types
                        placeholders = ', '.join(['?'] * len(value))
                        query += f" AND e.element_type IN ({placeholders})"
                        params.extend(value)
                    elif key == "doc_id" and isinstance(value, list):
                        # Handle list of document IDs to include
                        placeholders = ', '.join(['?'] * len(value))
                        query += f" AND e.doc_id IN ({placeholders})"
                        params.extend(value)
                    elif key == "exclude_doc_id" and isinstance(value, list):
                        # Handle list of document IDs to exclude
                        placeholders = ', '.join(['?'] * len(value))
                        query += f" AND e.doc_id NOT IN ({placeholders})"
                        params.extend(value)
                    elif key == "exclude_doc_source" and isinstance(value, list):
                        # Handle list of document IDs to exclude
                        placeholders = ', '.join(['?'] * len(value))
                        query += f" AND d.source NOT IN ({placeholders})"
                        params.extend(value)
                    else:
                        # Simple equality filter
                        query += f" AND e.{key} = ?"
                        params.append(value)

            # Add order and limit
            query += " ORDER BY similarity DESC LIMIT ?"
            params.append(limit)

            # Execute query
            cursor = self.conn.execute(query, params)

            return [(row["element_pk"], row["similarity"]) for row in cursor]
        except Exception as e:
            logger.error(f"Error using vss0 extension for search: {str(e)}")
            raise

    def _search_by_embedding_native(self, query_embedding: List[float], limit: int = 10,
                                    filter_criteria: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        """Fall back to native cosine similarity implementation with filtering."""
        # Register cosine similarity function if needed
        self._register_similarity_function()

        # Convert query embedding to blob
        query_blob = self._encode_embedding(query_embedding)

        # Start building the query
        query = """
        SELECT e.element_pk, cosine_similarity(emb.embedding, ?) AS similarity
        FROM embeddings emb
        JOIN elements e ON emb.element_pk = e.element_pk
        JOIN documents d ON e.doc_id = d.doc_id
        WHERE emb.dimensions = ?
        """
        params = [query_blob, len(query_embedding)]

        # Add filter criteria if provided
        if filter_criteria:
            for key, value in filter_criteria.items():
                if key == "element_type" and isinstance(value, list):
                    # Handle list of allowed element types
                    placeholders = ', '.join(['?'] * len(value))
                    query += f" AND e.element_type IN ({placeholders})"
                    params.extend(value)
                elif key == "doc_id" and isinstance(value, list):
                    # Handle list of document IDs to include
                    placeholders = ', '.join(['?'] * len(value))
                    query += f" AND e.doc_id IN ({placeholders})"
                    params.extend(value)
                elif key == "exclude_doc_id" and isinstance(value, list):
                    # Handle list of document IDs to exclude
                    placeholders = ', '.join(['?'] * len(value))
                    query += f" AND e.doc_id NOT IN ({placeholders})"
                    params.extend(value)
                elif key == "exclude_doc_source" and isinstance(value, list):
                    # Handle list of document IDs to exclude
                    placeholders = ', '.join(['?'] * len(value))
                    query += f" AND d.source NOT IN ({placeholders})"
                    params.extend(value)
                else:
                    # Simple equality filter
                    query += f" AND e.{key} = ?"
                    params.append(value)

        # Add order and limit
        query += " ORDER BY similarity DESC LIMIT ?"
        params.append(limit)

        # Execute search query
        cursor = self.conn.execute(query, params)

        return [(row["element_pk"], row["similarity"]) for row in cursor]

    def _register_similarity_function(self) -> None:
        """Register cosine similarity function with SQLite."""
        import numpy as np

        def cosine_similarity(blob1, blob2):
            # Decode embeddings
            vec1 = self._decode_embedding(blob1)
            vec2 = self._decode_embedding(blob2)

            if not vec1 or not vec2 or len(vec1) != len(vec2):
                return 0.0

            # Convert to numpy arrays
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))

        # Register function
        self.conn.create_function("cosine_similarity", 2, cosine_similarity)

    @staticmethod
    def _encode_embedding(embedding: List[float]) -> bytes:
        """Encode embedding as binary blob."""
        import numpy as np

        # Convert to numpy array and then to bytes
        return np.array(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def _decode_embedding(blob: bytes) -> List[float]:
        """Decode embedding from binary blob."""
        import numpy as np

        # Convert from bytes to numpy array and then to list
        return np.frombuffer(blob, dtype=np.float32).tolist()

    def search_by_text(self, search_text: str, limit: int = 10,
                       filter_criteria: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        """
        Search elements by semantic similarity to the provided text.

        This method combines text-to-embedding conversion and embedding search
        into a single convenient operation.

        Args:
            search_text: Text to search for semantically
            limit: Maximum number of results
            filter_criteria: Optional dictionary with criteria to filter results

        Returns:
            List of (element_id, similarity_score) tuples
        """
        if not self.conn:
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
