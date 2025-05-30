"""
Complete SQLite Implementation with Structured Search Support

This module provides a comprehensive SQLite implementation of the DocumentDatabase
with full structured search capabilities, matching the PostgreSQL implementation.
It leverages SQLite's JSON functions, custom similarity functions, and advanced
SQL queries to provide comprehensive search functionality.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Union, TYPE_CHECKING

import time

from .element_element import ElementHierarchical, ElementBase

# Import types for type checking only - these won't be imported at runtime
if TYPE_CHECKING:
    import sqlite3
    import numpy as np
    from numpy.typing import NDArray
    import sqlite_vec
    import sqlite_vss

    # Define type aliases for type checking
    VectorType = NDArray[np.float32]  # NumPy array type for vectors
    SQLiteConnectionType = sqlite3.Connection  # SQLite connection type
    SQLiteCursorType = sqlite3.Cursor  # SQLite cursor type
else:
    # Runtime type aliases - use generic Python types
    VectorType = List[float]  # Generic list of floats for vectors
    SQLiteConnectionType = Any  # Generic type for SQLite connection
    SQLiteCursorType = Any  # Generic type for SQLite cursor

from .base import DocumentDatabase
from .element_relationship import ElementRelationship
from .element_element import ElementType  # Import existing enum

# Import structured search components
from .structured_search import (
    StructuredSearchQuery, SearchCriteriaGroup, BackendCapabilities, SearchCapability,
    UnsupportedSearchError, TextSearchCriteria, EmbeddingSearchCriteria, DateSearchCriteria,
    TopicSearchCriteria, MetadataSearchCriteria, ElementSearchCriteria,
    LogicalOperator, DateRangeOperator, SimilarityOperator
)

logger = logging.getLogger(__name__)

# Define global flags for availability - these will be set at runtime
SQLITE3_AVAILABLE = False
SQLITE_SQLEAN_AVAILABLE = False
SQLITE_VEC_AVAILABLE = False
SQLITE_VSS_AVAILABLE = False
NUMPY_AVAILABLE = False

# Try to import the config
try:
    from ..config import Config
    config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
except Exception as e:
    logger.warning(f"Error configuring SQLite provider: {str(e)}. Using default settings.")
    config = None

# Try to import SQLite libraries conditionally
try:
    # Check if we should use sqlean based on config
    use_sqlean = config.config.get("storage", {}).get("sqlite_extensions", {}).get("use_sqlean",
                                                                                   False) if config else False

    if use_sqlean:
        try:
            # Try to import sqlean
            import sqlean as sqlite3
            SQLITE_SQLEAN_AVAILABLE = True
            logger.info("Using sqlean as SQLite provider (with extension support)")
        except ImportError:
            logger.warning("sqlean requested but not installed. Falling back to standard sqlite3.")
            import sqlite3
            SQLITE3_AVAILABLE = True
    else:
        import sqlite3
        SQLITE3_AVAILABLE = True
except ImportError:
    logger.warning("sqlite3 not available. This is unusual as it's part of Python standard library.")

# Try to import vector search extensions conditionally
try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
    logger.info("sqlite_vec extension available")
except ImportError:
    logger.debug("sqlite_vec extension not available")

try:
    import sqlite_vss
    SQLITE_VSS_AVAILABLE = True
    logger.info("sqlite_vss extension available")
except ImportError:
    logger.debug("sqlite_vss extension not available")

# Try to import numpy conditionally
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("NumPy not available. Fallback vector operations will be used.")


class SQLiteDocumentDatabase(DocumentDatabase):
    """SQLite implementation of document database with comprehensive structured search support."""

    def __init__(self, db_path: str):
        """
        Initialize SQLite document database.

        Args:
            db_path: Path to SQLite database file
        """
        if not SQLITE3_AVAILABLE and not SQLITE_SQLEAN_AVAILABLE:
            raise ImportError("Neither sqlite3 nor sqlean is available")

        self.db_path = db_path
        self.conn: SQLiteConnectionType = None
        self.vector_extension = None
        self.embedding_generator = None
        self.vector_dimension = config.config.get('embedding', {}).get('dimensions', 384) if config else 384

    # ========================================
    # STRUCTURED SEARCH IMPLEMENTATION
    # ========================================

    def get_backend_capabilities(self) -> BackendCapabilities:
        """
        SQLite supports most search capabilities with some limitations.
        """
        supported = {
            # Core search types
            SearchCapability.TEXT_SIMILARITY,
            SearchCapability.EMBEDDING_SIMILARITY,
            # SearchCapability.FULL_TEXT_SEARCH,  # Would need FTS extension

            # Date capabilities
            SearchCapability.DATE_FILTERING,
            SearchCapability.DATE_RANGE_QUERIES,
            SearchCapability.FISCAL_YEAR_DATES,
            SearchCapability.RELATIVE_DATES,
            SearchCapability.DATE_AGGREGATIONS,

            # Topic capabilities
            SearchCapability.TOPIC_FILTERING,
            SearchCapability.TOPIC_LIKE_PATTERNS,
            SearchCapability.TOPIC_CONFIDENCE_FILTERING,

            # Metadata capabilities
            SearchCapability.METADATA_EXACT,
            SearchCapability.METADATA_LIKE,
            SearchCapability.METADATA_RANGE,
            SearchCapability.METADATA_EXISTS,
            SearchCapability.NESTED_METADATA,

            # Element capabilities
            SearchCapability.ELEMENT_TYPE_FILTERING,
            SearchCapability.ELEMENT_HIERARCHY,
            SearchCapability.ELEMENT_RELATIONSHIPS,

            # Logical operations
            SearchCapability.LOGICAL_AND,
            SearchCapability.LOGICAL_OR,
            SearchCapability.LOGICAL_NOT,
            SearchCapability.NESTED_QUERIES,

            # Scoring and ranking
            SearchCapability.CUSTOM_SCORING,
            SearchCapability.SIMILARITY_THRESHOLDS,
            SearchCapability.BOOST_FACTORS,
            SearchCapability.SCORE_COMBINATION,

            # Advanced features
            SearchCapability.FACETED_SEARCH,
            # SearchCapability.RESULT_HIGHLIGHTING,  # Limited in SQLite
        }

        # Add vector search if extensions are available
        if self.vector_extension:
            supported.add(SearchCapability.VECTOR_SEARCH)

        return BackendCapabilities(supported)

    def execute_structured_search(self, query: StructuredSearchQuery) -> List[Dict[str, Any]]:
        """
        Execute a structured search query using SQLite's capabilities.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        # Validate query support
        missing = self.validate_query_support(query)
        if missing:
            raise UnsupportedSearchError(missing)

        try:
            # Execute the root criteria group
            raw_results = self._execute_criteria_group(query.criteria_group)

            # Process and enrich results
            final_results = self._process_search_results(raw_results, query)

            # Apply pagination
            start_idx = query.offset
            end_idx = start_idx + query.limit

            return final_results[start_idx:end_idx]

        except Exception as e:
            logger.error(f"Error executing structured search: {str(e)}")
            return []

    def _execute_criteria_group(self, group: SearchCriteriaGroup) -> List[Dict[str, Any]]:
        """Execute a single criteria group and return scored results."""

        # Collect results from all criteria in this group
        all_results = []

        # Execute individual criteria
        if group.text_criteria:
            text_results = self._execute_text_criteria(group.text_criteria)
            all_results.append(("text", text_results))

        if group.embedding_criteria:
            embedding_results = self._execute_embedding_criteria(group.embedding_criteria)
            all_results.append(("embedding", embedding_results))

        if group.date_criteria:
            date_results = self._execute_date_criteria(group.date_criteria)
            all_results.append(("date", date_results))

        if group.topic_criteria:
            topic_results = self._execute_topic_criteria(group.topic_criteria)
            all_results.append(("topic", topic_results))

        if group.metadata_criteria:
            metadata_results = self._execute_metadata_criteria(group.metadata_criteria)
            all_results.append(("metadata", metadata_results))

        if group.element_criteria:
            element_results = self._execute_element_criteria(group.element_criteria)
            all_results.append(("element", element_results))

        # Execute sub-groups recursively
        for sub_group in group.sub_groups:
            sub_results = self._execute_criteria_group(sub_group)
            all_results.append(("subgroup", sub_results))

        # Combine results based on the group's logical operator
        return self._combine_results(all_results, group.operator)

    def _execute_text_criteria(self, criteria: TextSearchCriteria) -> List[Dict[str, Any]]:
        """Execute text similarity search using embeddings."""
        try:
            # Generate embedding for the query text
            query_embedding = self._generate_embedding(criteria.query_text)

            # Perform similarity search
            similarity_results = self.search_by_embedding(
                query_embedding,
                limit=1000,  # Get many results for filtering
                filter_criteria=None
            )

            # Filter by similarity threshold and operator
            filtered_results = []
            for element_pk, similarity in similarity_results:
                if self._compare_similarity(similarity, criteria.similarity_threshold, criteria.similarity_operator):
                    filtered_results.append({
                        'element_pk': element_pk,
                        'scores': {
                            'text_similarity': similarity * criteria.boost_factor
                        }
                    })

            return filtered_results

        except Exception as e:
            logger.error(f"Error executing text criteria: {str(e)}")
            return []

    def _execute_embedding_criteria(self, criteria: EmbeddingSearchCriteria) -> List[Dict[str, Any]]:
        """Execute direct embedding vector search."""
        try:
            similarity_results = self.search_by_embedding(
                criteria.embedding_vector,
                limit=1000,
                filter_criteria=None
            )

            filtered_results = []
            for element_pk, similarity in similarity_results:
                if self._compare_similarity(similarity, criteria.similarity_threshold, criteria.similarity_operator):
                    filtered_results.append({
                        'element_pk': element_pk,
                        'scores': {
                            'embedding_similarity': similarity * criteria.boost_factor
                        }
                    })

            return filtered_results

        except Exception as e:
            logger.error(f"Error executing embedding criteria: {str(e)}")
            return []

    def _execute_date_criteria(self, criteria: DateSearchCriteria) -> List[Dict[str, Any]]:
        """Execute date-based filtering using SQLite date functions."""
        try:
            # Build date filter based on operator
            if criteria.operator == DateRangeOperator.WITHIN:
                element_pks = self._get_element_pks_in_date_range(criteria.start_date, criteria.end_date)

            elif criteria.operator == DateRangeOperator.AFTER:
                element_pks = self._get_element_pks_in_date_range(criteria.exact_date, None)

            elif criteria.operator == DateRangeOperator.BEFORE:
                element_pks = self._get_element_pks_in_date_range(None, criteria.exact_date)

            elif criteria.operator == DateRangeOperator.EXACTLY:
                # For exactly, we need a tight range around the date
                start_of_day = criteria.exact_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day = criteria.exact_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                element_pks = self._get_element_pks_in_date_range(start_of_day, end_of_day)

            elif criteria.operator == DateRangeOperator.RELATIVE_DAYS:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=criteria.relative_value)
                element_pks = self._get_element_pks_in_date_range(start_date, end_date)

            elif criteria.operator == DateRangeOperator.RELATIVE_MONTHS:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=criteria.relative_value * 30)  # Approximate
                element_pks = self._get_element_pks_in_date_range(start_date, end_date)

            elif criteria.operator == DateRangeOperator.FISCAL_YEAR:
                # Assume fiscal year starts in July (customize as needed)
                start_date = datetime(criteria.year - 1, 7, 1)
                end_date = datetime(criteria.year, 6, 30, 23, 59, 59)
                element_pks = self._get_element_pks_in_date_range(start_date, end_date)

            elif criteria.operator == DateRangeOperator.CALENDAR_YEAR:
                start_date = datetime(criteria.year, 1, 1)
                end_date = datetime(criteria.year, 12, 31, 23, 59, 59)
                element_pks = self._get_element_pks_in_date_range(start_date, end_date)

            elif criteria.operator == DateRangeOperator.QUARTER:
                quarter_starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
                quarter_ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}

                start_month, start_day = quarter_starts[criteria.quarter]
                end_month, end_day = quarter_ends[criteria.quarter]

                start_date = datetime(criteria.year, start_month, start_day)
                end_date = datetime(criteria.year, end_month, end_day, 23, 59, 59)
                element_pks = self._get_element_pks_in_date_range(start_date, end_date)

            # Also filter by specificity levels if needed
            if criteria.specificity_levels:
                element_pks = self._filter_by_specificity(element_pks, criteria.specificity_levels)

            # Convert to result format
            results = []
            for element_pk in element_pks:
                results.append({
                    'element_pk': element_pk,
                    'scores': {
                        'date_relevance': 1.0  # Could calculate date relevance score
                    }
                })

            return results

        except Exception as e:
            logger.error(f"Error executing date criteria: {str(e)}")
            return []

    def _execute_topic_criteria(self, criteria: TopicSearchCriteria) -> List[Dict[str, Any]]:
        """Execute topic-based filtering using SQLite JSON functions."""
        try:
            topic_results = self.search_by_text_and_topics(
                search_text=None,
                include_topics=criteria.include_topics,
                exclude_topics=criteria.exclude_topics,
                min_confidence=criteria.min_confidence,
                limit=1000
            )

            results = []
            for result in topic_results:
                results.append({
                    'element_pk': result['element_pk'],
                    'scores': {
                        'topic_confidence': result['confidence'] * criteria.boost_factor
                    }
                })

            return results

        except Exception as e:
            logger.error(f"Error executing topic criteria: {str(e)}")
            return []

    def _execute_metadata_criteria(self, criteria: MetadataSearchCriteria) -> List[Dict[str, Any]]:
        """Execute metadata-based filtering using SQLite JSON functions."""
        try:
            # Build SQL query for metadata filtering
            sql = "SELECT element_pk FROM elements WHERE 1=1"
            params = []

            # Add exact matches
            for key, value in criteria.exact_matches.items():
                sql += " AND JSON_EXTRACT(metadata, ?) = ?"
                params.extend([f'$.{key}', json.dumps(value)])

            # Add LIKE patterns
            for key, pattern in criteria.like_patterns.items():
                sql += " AND JSON_EXTRACT(metadata, ?) LIKE ?"
                params.extend([f'$.{key}', pattern])

            # Add range filters
            for key, range_filter in criteria.range_filters.items():
                if 'gte' in range_filter:
                    sql += " AND CAST(JSON_EXTRACT(metadata, ?) AS REAL) >= ?"
                    params.extend([f'$.{key}', range_filter['gte']])
                if 'lte' in range_filter:
                    sql += " AND CAST(JSON_EXTRACT(metadata, ?) AS REAL) <= ?"
                    params.extend([f'$.{key}', range_filter['lte']])
                if 'gt' in range_filter:
                    sql += " AND CAST(JSON_EXTRACT(metadata, ?) AS REAL) > ?"
                    params.extend([f'$.{key}', range_filter['gt']])
                if 'lt' in range_filter:
                    sql += " AND CAST(JSON_EXTRACT(metadata, ?) AS REAL) < ?"
                    params.extend([f'$.{key}', range_filter['lt']])

            # Add exists filters (check if JSON key exists)
            for key in criteria.exists_filters:
                sql += " AND JSON_EXTRACT(metadata, ?) IS NOT NULL"
                params.append(f'$.{key}')

            sql += " LIMIT 1000"

            cursor = self.conn.execute(sql, params)
            element_pks = [row[0] for row in cursor.fetchall()]

            results = []
            for element_pk in element_pks:
                results.append({
                    'element_pk': element_pk,
                    'scores': {
                        'metadata_relevance': 1.0
                    }
                })

            return results

        except Exception as e:
            logger.error(f"Error executing metadata criteria: {str(e)}")
            return []

    def _execute_element_criteria(self, criteria: ElementSearchCriteria) -> List[Dict[str, Any]]:
        """Execute element-based filtering using SQLite."""
        try:
            # Build SQL query for element filtering
            sql = "SELECT element_pk FROM elements WHERE 1=1"
            params = []

            # Add element type filter
            if criteria.element_types:
                type_values = self._prepare_element_type_query(criteria.element_types)
                if type_values:
                    if len(type_values) == 1:
                        sql += " AND element_type = ?"
                        params.append(type_values[0])
                    else:
                        placeholders = ', '.join(['?'] * len(type_values))
                        sql += f" AND element_type IN ({placeholders})"
                        params.extend(type_values)

            # Add document ID filters
            if criteria.doc_ids:
                placeholders = ', '.join(['?'] * len(criteria.doc_ids))
                sql += f" AND doc_id IN ({placeholders})"
                params.extend(criteria.doc_ids)

            if criteria.exclude_doc_ids:
                placeholders = ', '.join(['?'] * len(criteria.exclude_doc_ids))
                sql += f" AND doc_id NOT IN ({placeholders})"
                params.extend(criteria.exclude_doc_ids)

            # Add content length filters
            if criteria.content_length_min is not None:
                sql += " AND LENGTH(content_preview) >= ?"
                params.append(criteria.content_length_min)

            if criteria.content_length_max is not None:
                sql += " AND LENGTH(content_preview) <= ?"
                params.append(criteria.content_length_max)

            # Add parent element filters
            if criteria.parent_element_ids:
                placeholders = ', '.join(['?'] * len(criteria.parent_element_ids))
                sql += f" AND parent_id IN ({placeholders})"
                params.extend(criteria.parent_element_ids)

            sql += " LIMIT 1000"

            cursor = self.conn.execute(sql, params)
            element_pks = [row[0] for row in cursor.fetchall()]

            results = []
            for element_pk in element_pks:
                results.append({
                    'element_pk': element_pk,
                    'scores': {
                        'element_match': 1.0
                    }
                })

            return results

        except Exception as e:
            logger.error(f"Error executing element criteria: {str(e)}")
            return []

    # ========================================
    # CORE DATABASE OPERATIONS
    # ========================================

    def initialize(self) -> None:
        """Initialize the database by creating tables if they don't exist."""
        if not SQLITE3_AVAILABLE and not SQLITE_SQLEAN_AVAILABLE:
            raise ImportError("Neither sqlite3 nor sqlean is available")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)

        # Connect to database
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        # Enable foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = ON")

        # Check if extension loading is supported
        auto_discover = config.config.get("storage", {}).get("sqlite_extensions", {}).get("auto_discover",
                                                                                          True) if config else True
        extension_loading_supported = True

        try:
            self.conn.enable_load_extension(True)
        except (AttributeError, sqlite3.OperationalError) as e:
            extension_loading_supported = False
            logger.warning(f"SQLite extension loading not supported: {str(e)}")

            if not SQLITE_SQLEAN_AVAILABLE:
                logger.info("Consider using sqlean.py for SQLite extension support.")
                logger.info("Set storage.sqlite_extensions.use_sqlean to True in your config file.")

        # Only attempt to load extensions if supported and auto-discover is enabled
        if extension_loading_supported and auto_discover:
            self._load_vector_extensions()
        else:
            self.vector_extension = None
            logger.info("Using native vector search implementation (no extensions)")

        self._create_tables()
        logger.info(f"Initialized SQLite database at {self.db_path}")

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
            try:
                self.conn.enable_load_extension(False)
            except Exception:
                pass

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
                "SELECT * FROM processing_history WHERE source_id = ?",
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
                self.conn.execute(
                    """
                    UPDATE processing_history
                    SET content_hash = ?, last_modified = ?, processing_count = ?
                    WHERE source_id = ?
                    """,
                    (content_hash, time.time(), processing_count, source_id)
                )
            else:
                # Insert new record
                self.conn.execute(
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
                       relationships: List[Dict[str, Any]],
                       element_dates: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> None:
        """
        Store a document with its elements, relationships, and extracted dates.
        If a document with the same source already exists, update it instead.
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

            self.update_document(doc_id, document, elements, relationships, element_dates)
            return

        # New document, proceed with creation
        doc_id = document["doc_id"]

        try:
            # Store document
            metadata = document.get("metadata", {})
            metadata_json = json.dumps(metadata, default=self._json_default)

            self.conn.execute(
                """
                INSERT INTO documents 
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
                metadata_json = json.dumps(element.get("metadata", {}), default=self._json_default)
                content_preview = element.get("content_preview", "")
                if len(content_preview) > 500:
                    content_preview = content_preview[:500] + "..."

                cursor = self.conn.execute(
                    """
                    INSERT INTO elements 
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

                # Get the SQLite auto-increment ID
                element_pk = cursor.lastrowid
                # Store it back into the dictionary
                element['element_pk'] = element_pk

            # Store relationships
            for relationship in relationships:
                relationship_id = relationship["relationship_id"]
                metadata_json = json.dumps(relationship.get("metadata", {}), default=self._json_default)

                self.conn.execute(
                    """
                    INSERT INTO relationships 
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

            # Store extracted dates if provided
            if element_dates:
                try:
                    for element_id, dates_list in element_dates.items():
                        if dates_list:  # Only store if there are dates for this element
                            self.store_element_dates(element_id, dates_list)
                            logger.debug(f"Stored {len(dates_list)} dates for element {element_id}")
                except Exception as e:
                    logger.warning(f"Error storing element dates: {str(e)}")
                    # Don't fail the entire document storage for date storage errors

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
                        relationships: List[Dict[str, Any]],
                        element_dates: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> None:
        """
        Update an existing document.
        This will delete the old document and insert a new one.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        # Check if document exists
        cursor = self.conn.execute("SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,))
        if cursor.fetchone() is None:
            raise ValueError(f"Document not found: {doc_id}")

        try:
            # Get all element PKs for this document
            cursor = self.conn.execute("SELECT element_pk FROM elements WHERE doc_id = ?", (doc_id,))
            element_pks = [row[0] for row in cursor.fetchall()]

            # Delete relationships related to this document's elements
            if element_pks:
                element_pks_placeholders = ','.join(['?'] * len(element_pks))
                self.conn.execute(
                    f"DELETE FROM relationships WHERE source_id IN (SELECT element_id FROM elements WHERE element_pk IN ({element_pks_placeholders}))",
                    element_pks)

            # Delete embeddings for this document's elements
            if element_pks:
                element_pks_placeholders = ','.join(['?'] * len(element_pks))
                self.conn.execute(f"DELETE FROM embeddings WHERE element_pk IN ({element_pks_placeholders})",
                                  element_pks)

            # Delete dates for this document's elements
            if element_pks:
                element_pks_placeholders = ','.join(['?'] * len(element_pks))
                self.conn.execute(f"DELETE FROM element_dates WHERE element_pk IN ({element_pks_placeholders})",
                                  element_pks)

            # Delete all elements for this document
            self.conn.execute("DELETE FROM elements WHERE doc_id = ?", (doc_id,))

            # Delete the document itself
            self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

            # Commit the deletion part of the transaction
            self.conn.commit()

            # Now use store_document to insert everything
            # This will also update the processing history
            self.store_document(document, elements, relationships, element_dates)

        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            logger.error(f"Error updating document {doc_id}: {str(e)}")
            raise

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
            doc["metadata"] = json.loads(doc["metadata"]) if doc["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            doc["metadata"] = {}

        return doc

    def get_document_elements(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get elements for a document."""
        if not self.conn:
            raise ValueError("Database not initialized")

        # Modified to handle doc_id being either an actual doc_id or a source
        cursor = self.conn.execute(
            """
            SELECT e.* FROM elements e
            JOIN documents d ON e.doc_id = d.doc_id
            WHERE d.doc_id = ? OR d.source = ?
            ORDER BY e.element_id
            """,
            (doc_id, doc_id)
        )

        elements = []
        for row in cursor.fetchall():
            element = dict(row)

            # Convert metadata from JSON
            try:
                element["metadata"] = json.loads(element["metadata"]) if element["metadata"] else {}
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

        element_ids = [row[0] for row in cursor.fetchall()]

        if not element_ids:
            return []

        # Create placeholders for SQL IN clause
        element_ids_placeholders = ','.join(['?'] * len(element_ids))

        # Find relationships involving these elements
        cursor = self.conn.execute(
            f"SELECT * FROM relationships WHERE source_id IN ({element_ids_placeholders})",
            element_ids
        )

        relationships = []
        for row in cursor.fetchall():
            relationship = dict(row)

            # Convert metadata from JSON
            try:
                relationship["metadata"] = json.loads(relationship["metadata"]) if relationship["metadata"] else {}
            except (json.JSONDecodeError, TypeError):
                relationship["metadata"] = {}

            relationships.append(relationship)

        return relationships

    def get_element(self, element_id_or_pk: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Get element by ID or PK.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        # Try to interpret as element_pk (integer) first
        try:
            element_pk = int(element_id_or_pk)
            cursor = self.conn.execute(
                "SELECT * FROM elements WHERE element_pk = ?",
                (element_pk,)
            )
        except (ValueError, TypeError):
            # If not an integer, treat as element_id (string)
            cursor = self.conn.execute(
                "SELECT * FROM elements WHERE element_id = ?",
                (element_id_or_pk,)
            )

        row = cursor.fetchone()
        if row is None:
            return None

        element = dict(row)

        # Convert metadata from JSON
        try:
            element["metadata"] = json.loads(element["metadata"]) if element["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            element["metadata"] = {}

        return element

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

        try:
            # Get all element PKs for this document
            cursor = self.conn.execute(
                "SELECT element_pk FROM elements WHERE doc_id = ?",
                (doc_id,)
            )

            element_pks = [row[0] for row in cursor.fetchall()]

            # Get all element IDs for this document (for relationship deletion)
            cursor = self.conn.execute(
                "SELECT element_id FROM elements WHERE doc_id = ?",
                (doc_id,)
            )
            element_ids = [row[0] for row in cursor.fetchall()]

            # Delete embeddings for these elements
            if element_pks:
                element_pks_placeholders = ','.join(['?'] * len(element_pks))
                self.conn.execute(f"DELETE FROM embeddings WHERE element_pk IN ({element_pks_placeholders})",
                                  element_pks)

            # Delete dates for these elements
            if element_pks:
                element_pks_placeholders = ','.join(['?'] * len(element_pks))
                self.conn.execute(f"DELETE FROM element_dates WHERE element_pk IN ({element_pks_placeholders})",
                                  element_pks)

            # Delete relationships involving these elements
            if element_ids:
                element_ids_placeholders = ','.join(['?'] * len(element_ids))
                self.conn.execute(f"DELETE FROM relationships WHERE source_id IN ({element_ids_placeholders})",
                                  element_ids)

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

    # ========================================
    # LEGACY SEARCH METHODS
    # ========================================

    def find_documents(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find documents matching query with support for LIKE patterns.
        """
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
                    # Metadata filters require special handling with JSON (exact match)
                    for meta_key, meta_value in value.items():
                        conditions.append("JSON_EXTRACT(metadata, ?) = ?")
                        params.extend([f'$.{meta_key}', json.dumps(meta_value)])
                elif key == "metadata_like":
                    # Metadata LIKE filters
                    for meta_key, meta_value in value.items():
                        conditions.append("JSON_EXTRACT(metadata, ?) LIKE ?")
                        params.extend([f'$.{meta_key}', str(meta_value)])
                elif key.endswith("_like"):
                    # LIKE pattern for regular fields
                    field_name = key[:-5]  # Remove '_like' suffix
                    conditions.append(f"{field_name} LIKE ?")
                    params.append(value)
                elif isinstance(value, list):
                    # Handle list fields with IN clause
                    placeholders = ', '.join(['?'] * len(value))
                    conditions.append(f"{key} IN ({placeholders})")
                    params.extend(value)
                else:
                    # Exact match for regular fields
                    conditions.append(f"{key} = ?")
                    params.append(value)

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        # Add limit
        sql += f" LIMIT ?"
        params.append(limit)

        # Execute query
        cursor = self.conn.execute(sql, params)

        documents = []
        for row in cursor.fetchall():
            doc = dict(row)

            # Convert metadata from JSON
            try:
                doc["metadata"] = json.loads(doc["metadata"]) if doc["metadata"] else {}
            except (json.JSONDecodeError, TypeError):
                doc["metadata"] = {}

            documents.append(doc)

        return documents

    def find_elements(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements matching query with support for LIKE patterns and ElementType enums.
        """
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
                    # Metadata filters require special handling with JSON (exact match)
                    for meta_key, meta_value in value.items():
                        conditions.append("JSON_EXTRACT(metadata, ?) = ?")
                        params.extend([f'$.{meta_key}', json.dumps(meta_value)])
                elif key == "metadata_like":
                    # Metadata LIKE filters
                    for meta_key, meta_value in value.items():
                        conditions.append("JSON_EXTRACT(metadata, ?) LIKE ?")
                        params.extend([f'$.{meta_key}', str(meta_value)])
                elif key.endswith("_like"):
                    # LIKE pattern for regular fields
                    field_name = key[:-5]  # Remove '_like' suffix
                    conditions.append(f"{field_name} LIKE ?")
                    params.append(value)
                elif key == "element_type":
                    # Handle ElementType enums, strings, and lists
                    type_values = self._prepare_element_type_query(value)
                    if type_values:
                        if len(type_values) == 1:
                            conditions.append("element_type = ?")
                            params.append(type_values[0])
                        else:
                            placeholders = ', '.join(['?'] * len(type_values))
                            conditions.append(f"element_type IN ({placeholders})")
                            params.extend(type_values)
                elif isinstance(value, list):
                    # Handle other list fields with IN clause
                    field_name = key
                    placeholders = ', '.join(['?'] * len(value))
                    conditions.append(f"{field_name} IN ({placeholders})")
                    params.extend(value)
                else:
                    # Exact match for regular fields
                    conditions.append(f"{key} = ?")
                    params.append(value)

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        # Add limit
        sql += f" LIMIT ?"
        params.append(limit)

        # Execute query
        cursor = self.conn.execute(sql, params)

        elements = []
        for row in cursor.fetchall():
            element = dict(row)

            # Convert metadata from JSON
            try:
                element["metadata"] = json.loads(element["metadata"]) if element["metadata"] else {}
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
        for row in cursor.fetchall():
            element = dict(row)

            # Convert metadata from JSON
            try:
                element["metadata"] = json.loads(element["metadata"]) if element["metadata"] else {}
            except (json.JSONDecodeError, TypeError):
                element["metadata"] = {}

            elements.append(element)

        return elements

    # ========================================
    # EMBEDDING SEARCH METHODS
    # ========================================

    def store_embedding(self, element_pk: int, embedding: VectorType) -> None:
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

        # Update vector dimension based on actual data
        self.vector_dimension = max(self.vector_dimension, len(embedding))

        try:
            # Store embedding in the main embeddings table
            embedding_json = json.dumps(embedding)

            self.conn.execute(
                """
                INSERT OR REPLACE INTO embeddings 
                (element_pk, embedding, dimensions, topics, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
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

            self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing embedding for {element_pk}: {str(e)}")
            raise

    def get_embedding(self, element_pk: int) -> Optional[VectorType]:
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

        try:
            return json.loads(row[0]) if row[0] else None
        except (json.JSONDecodeError, TypeError):
            return None

    def search_by_embedding(self, query_embedding: VectorType, limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by embedding similarity using available method.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            if self.vector_extension:
                return self._search_by_vector_extension(query_embedding, limit, filter_criteria)
            else:
                return self._search_by_similarity_function(query_embedding, limit, filter_criteria)
        except Exception as e:
            logger.error(f"Error searching by embedding: {str(e)}")
            # Fall back to similarity function
            return self._search_by_similarity_function(query_embedding, limit, filter_criteria)

    def search_by_text(self, search_text: str, limit: int = 10,
                       filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by semantic similarity to the provided text.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            # Generate embedding for the search text
            query_embedding = self._generate_embedding(search_text)

            # Use the embedding to search, passing filter criteria
            return self.search_by_embedding(query_embedding, limit, filter_criteria)

        except Exception as e:
            logger.error(f"Error in semantic search by text: {str(e)}")
            # Return empty list on error
            return []

    def get_outgoing_relationships(self, element_pk: Union[int, str]) -> List[ElementRelationship]:
        """
        Find all relationships where the specified element_pk is the source.
        """
        if not self.conn:
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
            cursor = self.conn.execute(
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

            for row in cursor.fetchall():
                # Convert to dictionary
                rel_dict = dict(row)

                # Convert metadata from JSON if it's in string format
                try:
                    rel_dict["metadata"] = json.loads(rel_dict["metadata"]) if rel_dict["metadata"] else {}
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

    # ========================================
    # DATE STORAGE AND SEARCH METHODS
    # ========================================

    def store_element_dates(self, element_id: str, dates: List[Dict[str, Any]]) -> None:
        """
        Store extracted dates associated with an element with comprehensive temporal analysis.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        # Get element_pk from element_id
        element = self.get_element(element_id)
        if not element:
            raise ValueError(f"Element not found: {element_id}")

        element_pk = element.get('element_pk')
        if not element_pk:
            raise ValueError(f"Element {element_id} has no element_pk")

        try:
            for date_info in dates:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO element_dates 
                    (element_pk, original_text, iso_string, timestamp_value, 
                     year_value, month_value, day_value, hour_value, minute_value, second_value,
                     start_position, end_position, context,
                     decade_value, century_value, quarter_value, season_value, day_of_week_value,
                     day_of_year_value, week_of_year_value, is_weekend, is_holiday_season,
                     fiscal_quarter_value, fiscal_year_value, academic_semester_value, academic_year_value,
                     time_of_day_value, is_business_hours,
                     is_relative, is_partial, date_type, relative_reference, specificity_level,
                     date_format_detected, locale_hint, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                            ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        element_pk,
                        # Core fields
                        date_info.get('original_text', ''),
                        date_info.get('iso_string'),
                        date_info.get('timestamp'),
                        # Basic components (nullable)
                        date_info.get('year'),
                        date_info.get('month'),
                        date_info.get('day'),
                        date_info.get('hour'),
                        date_info.get('minute'),
                        date_info.get('second'),
                        date_info.get('start_position', -1),
                        date_info.get('end_position', -1),
                        date_info.get('context', ''),
                        # Extended temporal concepts (nullable)
                        date_info.get('decade'),
                        date_info.get('century'),
                        date_info.get('quarter'),
                        date_info.get('season'),
                        date_info.get('day_of_week'),
                        date_info.get('day_of_year'),
                        date_info.get('week_of_year'),
                        date_info.get('is_weekend'),
                        date_info.get('is_holiday_season'),
                        # Business/Academic periods (nullable)
                        date_info.get('fiscal_quarter'),
                        date_info.get('fiscal_year'),
                        date_info.get('academic_semester'),
                        date_info.get('academic_year'),
                        # Time categories (nullable)
                        date_info.get('time_of_day'),
                        date_info.get('is_business_hours'),
                        # Contextual information
                        date_info.get('is_relative', False),
                        date_info.get('is_partial', False),
                        date_info.get('date_type', 'absolute'),
                        date_info.get('relative_reference', ''),
                        date_info.get('specificity_level', 'full'),
                        # Format detection
                        date_info.get('date_format_detected', ''),
                        date_info.get('locale_hint', 'US'),
                        time.time()
                    )
                )

            self.conn.commit()
            logger.debug(f"Stored {len(dates)} comprehensive dates for element {element_id}")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing comprehensive dates for element {element_id}: {str(e)}")
            raise

    def get_element_dates(self, element_id: str) -> List[Dict[str, Any]]:
        """
        Get all dates associated with an element with comprehensive temporal analysis.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        # Get element_pk from element_id
        element = self.get_element(element_id)
        if not element:
            return []

        element_pk = element.get('element_pk')
        if not element_pk:
            return []

        try:
            cursor = self.conn.execute(
                """
                SELECT original_text, iso_string, timestamp_value, 
                       year_value, month_value, day_value, hour_value, minute_value, second_value,
                       start_position, end_position, context,
                       decade_value, century_value, quarter_value, season_value, day_of_week_value,
                       day_of_year_value, week_of_year_value, is_weekend, is_holiday_season,
                       fiscal_quarter_value, fiscal_year_value, academic_semester_value, academic_year_value,
                       time_of_day_value, is_business_hours,
                       is_relative, is_partial, date_type, relative_reference, specificity_level,
                       date_format_detected, locale_hint, created_at
                FROM element_dates 
                WHERE element_pk = ?
                ORDER BY timestamp_value
                """,
                (element_pk,)
            )

            dates = []
            for row in cursor.fetchall():
                date_dict = {
                    # Core fields
                    'original_text': row[0],
                    'iso_string': row[1],
                    'timestamp': row[2],
                    # Basic components (nullable)
                    'year': row[3],
                    'month': row[4],
                    'day': row[5],
                    'hour': row[6],
                    'minute': row[7],
                    'second': row[8],
                    'start_position': row[9],
                    'end_position': row[10],
                    'context': row[11],
                    # Extended temporal concepts (nullable)
                    'decade': row[12],
                    'century': row[13],
                    'quarter': row[14],
                    'season': row[15],
                    'day_of_week': row[16],
                    'day_of_year': row[17],
                    'week_of_year': row[18],
                    'is_weekend': row[19],
                    'is_holiday_season': row[20],
                    # Business/Academic periods (nullable)
                    'fiscal_quarter': row[21],
                    'fiscal_year': row[22],
                    'academic_semester': row[23],
                    'academic_year': row[24],
                    # Time categories (nullable)
                    'time_of_day': row[25],
                    'is_business_hours': row[26],
                    # Contextual information
                    'is_relative': row[27],
                    'is_partial': row[28],
                    'date_type': row[29],
                    'relative_reference': row[30],
                    'specificity_level': row[31],
                    # Format detection
                    'date_format_detected': row[32],
                    'locale_hint': row[33],
                    'created_at': row[34]
                }
                dates.append(date_dict)

            return dates

        except Exception as e:
            logger.error(f"Error getting comprehensive dates for element {element_id}: {str(e)}")
            return []

    def store_embedding_with_dates(self, element_id: str, embedding: List[float],
                                   dates: List[Dict[str, Any]]) -> None:
        """
        Store both embedding and dates for an element in a single operation.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        # Get element_pk from element_id
        element = self.get_element(element_id)
        if not element:
            raise ValueError(f"Element not found: {element_id}")

        element_pk = element.get('element_pk')
        if not element_pk:
            raise ValueError(f"Element {element_id} has no element_pk")

        try:
            # Store embedding
            self.store_embedding(element_pk, embedding)

            # Store comprehensive dates if any
            if dates:
                self.store_element_dates(element_id, dates)

            logger.debug(f"Stored embedding and {len(dates)} comprehensive dates for element {element_id}")

        except Exception as e:
            logger.error(f"Error storing embedding with comprehensive dates for element {element_id}: {str(e)}")
            raise

    def delete_element_dates(self, element_id: str) -> bool:
        """
        Delete all dates associated with an element.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        # Get element_pk from element_id
        element = self.get_element(element_id)
        if not element:
            return False

        element_pk = element.get('element_pk')
        if not element_pk:
            return False

        try:
            cursor = self.conn.execute(
                "DELETE FROM element_dates WHERE element_pk = ?",
                (element_pk,)
            )

            deleted_count = cursor.rowcount
            self.conn.commit()

            logger.debug(f"Deleted {deleted_count} dates for element {element_id}")
            return deleted_count > 0

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error deleting dates for element {element_id}: {str(e)}")
            return False

    def search_elements_by_date_range(self, start_date: datetime, end_date: datetime,
                                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements that contain dates within a specified range.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        start_timestamp = start_date.timestamp()
        end_timestamp = end_date.timestamp()

        try:
            cursor = self.conn.execute(
                """
                SELECT DISTINCT e.* 
                FROM elements e
                JOIN element_dates ed ON e.element_pk = ed.element_pk
                WHERE ed.timestamp_value >= ? AND ed.timestamp_value <= ?
                ORDER BY e.element_pk
                LIMIT ?
                """,
                (start_timestamp, end_timestamp, limit)
            )

            elements = []
            for row in cursor.fetchall():
                element = dict(row)
                # Convert metadata from JSON
                try:
                    element["metadata"] = json.loads(element["metadata"]) if element["metadata"] else {}
                except (json.JSONDecodeError, TypeError):
                    element["metadata"] = {}
                elements.append(element)

            return elements

        except Exception as e:
            logger.error(f"Error searching elements by date range: {str(e)}")
            return []

    def search_by_text_and_date_range(self,
                                      search_text: str,
                                      start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None,
                                      limit: int = 10) -> List[Tuple[int, float]]:
        """
        Search elements by semantic similarity AND date range.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            # Generate embedding for search text
            query_embedding = self._generate_embedding(search_text)

            # Build filter criteria for date range
            filter_criteria = {}
            if start_date or end_date:
                # Get element_pks that have dates in the range
                date_sql = "SELECT DISTINCT element_pk FROM element_dates WHERE 1=1"
                date_params = []

                if start_date:
                    date_sql += " AND timestamp_value >= ?"
                    date_params.append(start_date.timestamp())

                if end_date:
                    date_sql += " AND timestamp_value <= ?"
                    date_params.append(end_date.timestamp())

                cursor = self.conn.execute(date_sql, date_params)
                element_pks_with_dates = [row[0] for row in cursor.fetchall()]

                if not element_pks_with_dates:
                    return []  # No elements have dates in range

                filter_criteria['element_pk_list'] = element_pks_with_dates

            return self._search_by_embedding_with_filter(query_embedding, limit, filter_criteria)

        except Exception as e:
            logger.error(f"Error in text and date range search: {str(e)}")
            return []

    def search_by_embedding_and_date_range(self,
                                           query_embedding: List[float],
                                           start_date: Optional[datetime] = None,
                                           end_date: Optional[datetime] = None,
                                           limit: int = 10) -> List[Tuple[int, float]]:
        """
        Search elements by embedding similarity AND date range.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            # Build filter criteria for date range
            filter_criteria = {}
            if start_date or end_date:
                # Get element_pks that have dates in the range
                date_sql = "SELECT DISTINCT element_pk FROM element_dates WHERE 1=1"
                date_params = []

                if start_date:
                    date_sql += " AND timestamp_value >= ?"
                    date_params.append(start_date.timestamp())

                if end_date:
                    date_sql += " AND timestamp_value <= ?"
                    date_params.append(end_date.timestamp())

                cursor = self.conn.execute(date_sql, date_params)
                element_pks_with_dates = [row[0] for row in cursor.fetchall()]

                if not element_pks_with_dates:
                    return []  # No elements have dates in range

                filter_criteria['element_pk_list'] = element_pks_with_dates

            return self._search_by_embedding_with_filter(query_embedding, limit, filter_criteria)

        except Exception as e:
            logger.error(f"Error in embedding and date range search: {str(e)}")
            return []

    def get_elements_with_dates(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all elements that have associated dates.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            cursor = self.conn.execute(
                """
                SELECT DISTINCT e.* 
                FROM elements e
                JOIN element_dates ed ON e.element_pk = ed.element_pk
                ORDER BY e.element_pk
                LIMIT ?
                """,
                (limit,)
            )

            elements = []
            for row in cursor.fetchall():
                element = dict(row)
                # Convert metadata from JSON
                try:
                    element["metadata"] = json.loads(element["metadata"]) if element["metadata"] else {}
                except (json.JSONDecodeError, TypeError):
                    element["metadata"] = {}
                elements.append(element)

            return elements

        except Exception as e:
            logger.error(f"Error getting elements with dates: {str(e)}")
            return []

    def get_date_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about dates in the database with enhanced temporal analysis.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            # Get basic statistics
            cursor = self.conn.execute("""
                SELECT 
                    COUNT(DISTINCT element_pk) as elements_with_dates,
                    COUNT(*) as total_date_extractions,
                    MIN(iso_string) as earliest_date,
                    MAX(iso_string) as latest_date
                FROM element_dates
                WHERE timestamp_value IS NOT NULL
            """)

            basic_stats = cursor.fetchone()

            # Get year distribution
            cursor = self.conn.execute("""
                SELECT year_value, COUNT(DISTINCT element_pk) as element_count
                FROM element_dates
                WHERE year_value IS NOT NULL
                GROUP BY year_value
                ORDER BY element_count DESC
                LIMIT 10
            """)

            year_stats = cursor.fetchall()

            # Get most common years by total extractions
            cursor = self.conn.execute("""
                SELECT year_value, COUNT(*) as extraction_count
                FROM element_dates
                WHERE year_value IS NOT NULL
                GROUP BY year_value
                ORDER BY extraction_count DESC
                LIMIT 10
            """)

            common_years = cursor.fetchall()

            # Get season distribution
            cursor = self.conn.execute("""
                SELECT season_value, COUNT(DISTINCT element_pk) as element_count
                FROM element_dates
                WHERE season_value IS NOT NULL
                GROUP BY season_value
                ORDER BY element_count DESC
            """)

            season_stats = cursor.fetchall()

            # Get quarter distribution
            cursor = self.conn.execute("""
                SELECT quarter_value, COUNT(DISTINCT element_pk) as element_count
                FROM element_dates
                WHERE quarter_value IS NOT NULL
                GROUP BY quarter_value
                ORDER BY quarter_value
            """)

            quarter_stats = cursor.fetchall()

            # Get specificity level distribution
            cursor = self.conn.execute("""
                SELECT specificity_level, COUNT(*) as extraction_count
                FROM element_dates
                WHERE specificity_level IS NOT NULL
                GROUP BY specificity_level
                ORDER BY extraction_count DESC
            """)

            specificity_stats = cursor.fetchall()

            return {
                'total_elements_with_dates': int(basic_stats[0]) if basic_stats[0] else 0,
                'total_date_extractions': int(basic_stats[1]) if basic_stats[1] else 0,
                'earliest_date': basic_stats[2] if basic_stats[2] else None,
                'latest_date': basic_stats[3] if basic_stats[3] else None,
                'most_common_years': [(int(row[0]), int(row[1])) for row in common_years],
                'elements_by_year': {int(row[0]): int(row[1]) for row in year_stats},
                'season_distribution': {row[0]: int(row[1]) for row in season_stats},
                'quarter_distribution': {int(row[0]): int(row[1]) for row in quarter_stats},
                'specificity_distribution': {row[0]: int(row[1]) for row in specificity_stats}
            }

        except Exception as e:
            logger.error(f"Error getting enhanced date statistics: {str(e)}")
            return {}

    # ========================================
    # TOPIC SUPPORT METHODS
    # ========================================

    def supports_topics(self) -> bool:
        """SQLite implementation supports topic-aware embeddings."""
        return True

    def store_embedding_with_topics(self, element_pk: int, embedding: VectorType,
                                    topics: List[str], confidence: float = 1.0) -> None:
        """Store embedding for an element with topic assignments."""
        if not self.conn:
            raise ValueError("Database not initialized")

        # Verify element exists
        cursor = self.conn.execute(
            "SELECT element_pk FROM elements WHERE element_pk = ?",
            (element_pk,)
        )

        if cursor.fetchone() is None:
            raise ValueError(f"Element not found: {element_pk}")

        # Update vector dimension based on actual data
        self.vector_dimension = max(self.vector_dimension, len(embedding))

        try:
            # Store embedding with topics in the main embeddings table
            embedding_json = json.dumps(embedding)
            topics_json = json.dumps(topics)

            self.conn.execute(
                """
                INSERT OR REPLACE INTO embeddings 
                (element_pk, embedding, dimensions, topics, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
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
        """Search elements by text with topic filtering using LIKE patterns."""
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            # Generate embedding for search text if provided
            query_embedding = None
            if search_text:
                query_embedding = self._generate_embedding(search_text)

            # Build the query based on whether we have search text and vector support
            if search_text and self.vector_extension:
                return self._search_by_text_and_topics_extension(
                    query_embedding, include_topics, exclude_topics, min_confidence, limit
                )
            else:
                return self._search_by_text_and_topics_fallback(
                    query_embedding, include_topics, exclude_topics, min_confidence, limit
                )

        except Exception as e:
            logger.error(f"Error in topic-aware search: {str(e)}")
            return []

    def get_topic_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about topic distribution across embeddings."""
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            # Query to get topic statistics using SQLite JSON functions
            cursor = self.conn.execute("""
                WITH topic_expanded AS (
                    SELECT 
                        json_each.value AS topic,
                        em.confidence,
                        e.doc_id
                    FROM embeddings em
                    JOIN elements e ON em.element_pk = e.element_pk
                    JOIN json_each(em.topics) ON json_each.value IS NOT NULL
                    WHERE em.topics IS NOT NULL AND json_array_length(em.topics) > 0
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
            for row in cursor.fetchall():
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
        """Get topics assigned to a specific embedding."""
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            cursor = self.conn.execute(
                "SELECT topics FROM embeddings WHERE element_pk = ?",
                (element_pk,)
            )

            row = cursor.fetchone()
            if row is None or row[0] is None:
                return []

            try:
                return json.loads(row[0]) if row[0] else []
            except (json.JSONDecodeError, TypeError):
                return []

        except Exception as e:
            logger.error(f"Error getting topics for element {element_pk}: {str(e)}")
            return []

    # ========================================
    # HELPER METHODS
    # ========================================

    def _json_default(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _generate_embedding(self, search_text: str) -> List[float]:
        """Generate embedding for search text."""
        try:
            if self.embedding_generator is None:
                # Conditional import for embedding generator
                try:
                    from ..embeddings import get_embedding_generator
                    self.embedding_generator = get_embedding_generator(config)
                except ImportError as e:
                    logger.error(f"Error importing embedding generator: {str(e)}")
                    raise ValueError("Embedding generator not available - embedding libraries may not be installed")

            return self.embedding_generator.generate(search_text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def _get_element_pks_in_date_range(self, start_date: Optional[datetime],
                                       end_date: Optional[datetime]) -> List[int]:
        """Get element_pks that have dates within the specified range."""
        if not (start_date or end_date):
            return []

        date_sql = "SELECT DISTINCT element_pk FROM element_dates WHERE 1=1"
        date_params = []

        if start_date:
            date_sql += " AND timestamp_value >= ?"
            date_params.append(start_date.timestamp())

        if end_date:
            date_sql += " AND timestamp_value <= ?"
            date_params.append(end_date.timestamp())

        cursor = self.conn.execute(date_sql, date_params)
        return [row[0] for row in cursor.fetchall()]

    def _filter_by_specificity(self, element_pks: List[int],
                               allowed_levels: List[str]) -> List[int]:
        """Filter element PKs by date specificity levels."""
        if not element_pks or not allowed_levels:
            return element_pks

        # Query to get element PKs that have dates with allowed specificity levels
        placeholders = ', '.join(['?'] * len(element_pks))
        specificity_placeholders = ', '.join(['?'] * len(allowed_levels))

        cursor = self.conn.execute(f"""
            SELECT DISTINCT ed.element_pk
            FROM element_dates ed
            WHERE ed.element_pk IN ({placeholders})
            AND ed.specificity_level IN ({specificity_placeholders})
        """, element_pks + allowed_levels)

        return [row[0] for row in cursor.fetchall()]

    def _search_by_similarity_function(self, query_embedding: VectorType, limit: int = 10,
                                       filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Fall back to calculating similarity in Python with filtering.
        """
        # Build base query to get embeddings with possible filtering
        sql = """
        SELECT em.element_pk, em.embedding, e.element_type, e.doc_id
        FROM embeddings em
        JOIN elements e ON e.element_pk = em.element_pk
        """
        params = []

        # Add WHERE clauses if we have filter criteria
        if filter_criteria:
            conditions = []

            for key, value in filter_criteria.items():
                if key == "element_type" and isinstance(value, list):
                    # Handle list of allowed element types
                    placeholders = ', '.join(['?'] * len(value))
                    conditions.append(f"e.element_type IN ({placeholders})")
                    params.extend(value)
                elif key == "doc_id" and isinstance(value, list):
                    # Handle list of document IDs to include
                    placeholders = ', '.join(['?'] * len(value))
                    conditions.append(f"e.doc_id IN ({placeholders})")
                    params.extend(value)
                elif key == "exclude_doc_id" and isinstance(value, list):
                    # Handle list of document IDs to exclude
                    placeholders = ', '.join(['?'] * len(value))
                    conditions.append(f"e.doc_id NOT IN ({placeholders})")
                    params.extend(value)
                elif key == "element_pk_list" and isinstance(value, list):
                    # Filter by specific element_pks (for date range filtering)
                    if value:  # Only add condition if list is not empty
                        placeholders = ', '.join(['?'] * len(value))
                        conditions.append(f"em.element_pk IN ({placeholders})")
                        params.extend(value)
                else:
                    # Simple equality filter
                    conditions.append(f"e.{key} = ?")
                    params.append(value)

            # Add WHERE clause if we have conditions
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        # Execute the query
        cursor = self.conn.execute(sql, params)

        # Calculate similarities based on the available implementations
        if NUMPY_AVAILABLE:
            similarities = [
                (row[0], self._cosine_similarity_numpy(query_embedding, json.loads(row[1])))
                for row in cursor.fetchall()
            ]
        else:
            similarities = [
                (row[0], self._cosine_similarity_python(query_embedding, json.loads(row[1])))
                for row in cursor.fetchall()
            ]

        # Sort by similarity (highest first)
        similarities.sort(key=lambda row: row[1], reverse=True)

        return similarities[:limit]

    def _search_by_embedding_with_filter(self, query_embedding: VectorType, limit: int = 10,
                                         filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Enhanced embedding search with support for element_pk filtering for date ranges.
        """
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            if self.vector_extension:
                return self._search_by_vector_extension_with_filter(query_embedding, limit, filter_criteria)
            else:
                return self._search_by_similarity_function_with_filter(query_embedding, limit, filter_criteria)
        except Exception as e:
            logger.error(f"Error searching by embedding with filter: {str(e)}")
            return []

    def _search_by_similarity_function_with_filter(self, query_embedding: VectorType, limit: int = 10,
                                                   filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Fall back to calculating similarity in Python with enhanced filtering including element_pk lists.
        """
        return self._search_by_similarity_function(query_embedding, limit, filter_criteria)

    def _search_by_vector_extension(self, query_embedding: VectorType, limit: int = 10,
                                    filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Search using vector extension."""
        # Implementation would depend on which extension is loaded
        # For now, fall back to similarity function
        return self._search_by_similarity_function(query_embedding, limit, filter_criteria)

    def _search_by_vector_extension_with_filter(self, query_embedding: VectorType, limit: int = 10,
                                                filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Search using vector extension with filtering."""
        # Implementation would depend on which extension is loaded
        # For now, fall back to similarity function
        return self._search_by_similarity_function_with_filter(query_embedding, limit, filter_criteria)

    @staticmethod
    def _cosine_similarity_numpy(vec1: VectorType, vec2: VectorType) -> float:
        """Calculate cosine similarity between two vectors using NumPy."""
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
        """Calculate cosine similarity between two vectors using pure Python."""
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

    def _search_by_text_and_topics_extension(self, query_embedding: VectorType,
                                             include_topics: Optional[List[str]] = None,
                                             exclude_topics: Optional[List[str]] = None,
                                             min_confidence: float = 0.7,
                                             limit: int = 10) -> List[Dict[str, Any]]:
        """Search using vector extension with topic filtering."""
        # For now, fall back to Python implementation
        return self._search_by_text_and_topics_fallback(
            query_embedding, include_topics, exclude_topics, min_confidence, limit
        )

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
        WHERE em.confidence >= ?
        """
        params = [min_confidence]

        # Add topic filtering conditions
        if include_topics:
            topic_conditions = []
            for topic_pattern in include_topics:
                # Check if any topic in the topics JSON array matches the pattern
                topic_conditions.append("""
                    EXISTS (
                        SELECT 1 FROM json_each(em.topics) 
                        WHERE json_each.value LIKE ?
                    )
                """)
                params.append(topic_pattern)

            if topic_conditions:
                sql += " AND (" + " OR ".join(topic_conditions) + ")"

        # Add exclude topic filters
        if exclude_topics:
            exclude_conditions = []
            for topic_pattern in exclude_topics:
                # Check that no topic in the topics JSON array matches the pattern
                exclude_conditions.append("""
                    NOT EXISTS (
                        SELECT 1 FROM json_each(em.topics) 
                        WHERE json_each.value LIKE ?
                    )
                """)
                params.append(topic_pattern)

            if exclude_conditions:
                sql += " AND " + " AND ".join(exclude_conditions)

        cursor = self.conn.execute(sql, params)

        # Calculate similarities if we have a query embedding
        results = []
        for row in cursor.fetchall():
            try:
                topics = json.loads(row[3]) if row[3] else []
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
                    embedding = json.loads(row[1])
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

    @staticmethod
    def _compare_similarity(similarity: float, threshold: float,
                            operator: SimilarityOperator) -> bool:
        """Compare similarity score against threshold using specified operator."""
        if operator == SimilarityOperator.GREATER_THAN:
            return similarity > threshold
        elif operator == SimilarityOperator.GREATER_EQUAL:
            return similarity >= threshold
        elif operator == SimilarityOperator.LESS_THAN:
            return similarity < threshold
        elif operator == SimilarityOperator.LESS_EQUAL:
            return similarity <= threshold
        elif operator == SimilarityOperator.EQUALS:
            return abs(similarity - threshold) < 0.001  # Small epsilon for float comparison
        return False

    @staticmethod
    def _intersect_results(result_sets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Find intersection of multiple result sets."""
        if not result_sets:
            return []

        # Get element_pks from all sets and combine scores
        element_pk_sets = []
        element_scores = {}  # element_pk -> combined scores

        for result_set in result_sets:
            pk_set = set()
            for result in result_set:
                element_pk = result['element_pk']
                pk_set.add(element_pk)

                # Accumulate scores
                if element_pk not in element_scores:
                    element_scores[element_pk] = {}

                for score_type, score_value in result.get('scores', {}).items():
                    if score_type not in element_scores[element_pk]:
                        element_scores[element_pk][score_type] = []
                    element_scores[element_pk][score_type].append(score_value)

            element_pk_sets.append(pk_set)

        # Find intersection
        common_pks = element_pk_sets[0]
        for pk_set in element_pk_sets[1:]:
            common_pks = common_pks.intersection(pk_set)

        # Build result list
        results = []
        for element_pk in common_pks:
            results.append({
                'element_pk': element_pk,
                'scores': element_scores[element_pk]
            })

        return results

    @staticmethod
    def _union_results(result_sets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Find union of multiple result sets."""
        element_scores = {}  # element_pk -> combined scores

        for result_set in result_sets:
            for result in result_set:
                element_pk = result['element_pk']

                if element_pk not in element_scores:
                    element_scores[element_pk] = {}

                for score_type, score_value in result.get('scores', {}).items():
                    if score_type not in element_scores[element_pk]:
                        element_scores[element_pk][score_type] = []
                    element_scores[element_pk][score_type].append(score_value)

        # Build result list
        results = []
        for element_pk, scores in element_scores.items():
            results.append({
                'element_pk': element_pk,
                'scores': scores
            })

        return results

    @staticmethod
    def _subtract_results(base_set: List[Dict[str, Any]],
                          subtract_sets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Subtract multiple sets from base set."""
        base_pks = {result['element_pk'] for result in base_set}

        # Collect all PKs to subtract
        subtract_pks = set()
        for subtract_set in subtract_sets:
            for result in subtract_set:
                subtract_pks.add(result['element_pk'])

        # Return base results that are not in subtract sets
        final_pks = base_pks - subtract_pks

        return [result for result in base_set if result['element_pk'] in final_pks]

    def _process_search_results(self, raw_results: List[Dict[str, Any]],
                                query: StructuredSearchQuery) -> List[Dict[str, Any]]:
        """Process and enrich search results."""

        # Calculate combined scores
        for result in raw_results:
            result['final_score'] = self._calculate_combined_score(
                result.get('scores', {}),
                query.score_combination,
                query.custom_weights
            )

        # Sort by final score
        raw_results.sort(key=lambda x: x['final_score'], reverse=True)

        # Enrich with element details
        enriched_results = []
        for result in raw_results:
            element_pk = result['element_pk']
            element = self.get_element(element_pk)

            if not element:
                continue

            enriched_result = {
                'element_pk': element_pk,
                'element_id': element.get('element_id'),
                'doc_id': element.get('doc_id'),
                'element_type': element.get('element_type'),
                'content_preview': element.get('content_preview'),
                'final_score': result['final_score']
            }

            if query.include_similarity_scores:
                enriched_result['scores'] = result.get('scores', {})

            if query.include_metadata:
                enriched_result['metadata'] = element.get('metadata', {})

            if query.include_topics:
                enriched_result['topics'] = self.get_embedding_topics(element_pk)

            if query.include_element_dates:
                element_id = element.get('element_id')
                if element_id:
                    enriched_result['extracted_dates'] = self.get_element_dates(element_id)
                    enriched_result['date_count'] = len(enriched_result['extracted_dates'])

            enriched_results.append(enriched_result)

        return enriched_results

    @staticmethod
    def _calculate_combined_score(scores: Dict[str, List[float]],
                                  combination_method: str,
                                  weights: Dict[str, float]) -> float:
        """Calculate final combined score from multiple score types."""

        if not scores:
            return 0.0

        # Average scores of the same type
        avg_scores = {}
        for score_type, score_list in scores.items():
            if score_list:
                avg_scores[score_type] = sum(score_list) / len(score_list)

        if not avg_scores:
            return 0.0

        if combination_method == "multiply":
            final_score = 1.0
            for score_type, score in avg_scores.items():
                weight = weights.get(score_type, 1.0)
                final_score *= (score * weight)
            return final_score

        elif combination_method == "add":
            final_score = 0.0
            for score_type, score in avg_scores.items():
                weight = weights.get(score_type, 1.0)
                final_score += (score * weight)
            return final_score

        elif combination_method == "max":
            weighted_scores = []
            for score_type, score in avg_scores.items():
                weight = weights.get(score_type, 1.0)
                weighted_scores.append(score * weight)
            return max(weighted_scores)

        elif combination_method == "weighted_avg":
            total_weighted_score = 0.0
            total_weight = 0.0
            for score_type, score in avg_scores.items():
                weight = weights.get(score_type, 1.0)
                total_weighted_score += (score * weight)
                total_weight += weight
            return total_weighted_score / total_weight if total_weight > 0 else 0.0

        return 0.0

    def _combine_results(self, all_results: List[Tuple[str, List[Dict[str, Any]]]],
                         operator: LogicalOperator) -> List[Dict[str, Any]]:
        """Combine results from multiple criteria using logical operators."""

        if not all_results:
            return []

        if len(all_results) == 1:
            return all_results[0][1]  # Return the single result set

        # Extract just the result lists
        result_sets = [results for _, results in all_results]

        if operator == LogicalOperator.AND:
            return self._intersect_results(result_sets)
        elif operator == LogicalOperator.OR:
            return self._union_results(result_sets)
        elif operator == LogicalOperator.NOT:
            # NOT operation: first set minus all other sets
            if len(result_sets) >= 2:
                return self._subtract_results(result_sets[0], result_sets[1:])
            else:
                return result_sets[0]

        return []

    def _prepare_element_type_query(self, element_types: Union[
        ElementType,
        List[ElementType],
        str,
        List[str],
        None
    ]) -> Optional[List[str]]:
        """
        Prepare element type values for database queries using existing ElementType enum.
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

    # ========================================
    # HIERARCHY METHODS
    # ========================================

    def get_results_outline(self, elements: List[Tuple[int, float]]) -> List["ElementHierarchical"]:
        """
        For search results, create a hierarchical outline showing element ancestry.
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
                    current_level = current_level[existing_idx].child_elements
                else:
                    # Ancestor doesn't exist, add it with its score
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
        """
        from .element_element import ElementBase

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

    # ========================================
    # TABLE CREATION
    # ========================================

    def _create_tables(self) -> None:
        """Create database tables including the enhanced element_dates table."""
        try:
            # Create the required schemas (existing tables)
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

            # Modified elements table with element_pk as auto-increment
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS elements (
                element_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                element_id TEXT UNIQUE NOT NULL,
                doc_id TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
                element_type TEXT,
                parent_id TEXT REFERENCES elements(element_id),
                content_preview TEXT,
                content_location TEXT,
                content_hash TEXT,
                metadata TEXT
            )
            """)

            # ENHANCED: Create comprehensive element_dates table with all temporal analysis fields
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS element_dates (
                date_id INTEGER PRIMARY KEY AUTOINCREMENT,
                element_pk INTEGER REFERENCES elements(element_pk) ON DELETE CASCADE,

                -- Core fields (always present)
                original_text TEXT NOT NULL,
                iso_string TEXT,
                timestamp_value REAL,
                start_position INTEGER DEFAULT -1,
                end_position INTEGER DEFAULT -1,
                context TEXT DEFAULT '',

                -- Basic date components (nullable when not specific enough)
                year_value INTEGER,
                month_value INTEGER,
                day_value INTEGER,
                hour_value INTEGER,
                minute_value INTEGER,
                second_value INTEGER,

                -- Extended temporal concepts (nullable when not applicable)
                decade_value INTEGER,
                century_value INTEGER,
                quarter_value INTEGER,
                season_value TEXT,
                day_of_week_value TEXT,
                day_of_year_value INTEGER,
                week_of_year_value INTEGER,
                is_weekend INTEGER,
                is_holiday_season INTEGER,

                -- Business/Academic periods (nullable when not specific enough)
                fiscal_quarter_value INTEGER,
                fiscal_year_value INTEGER,
                academic_semester_value TEXT,
                academic_year_value TEXT,

                -- Time of day categories (nullable when no time specified)
                time_of_day_value TEXT,
                is_business_hours INTEGER,

                -- Contextual information
                is_relative INTEGER DEFAULT 0,
                is_partial INTEGER DEFAULT 0,
                date_type TEXT DEFAULT 'absolute',
                relative_reference TEXT DEFAULT '',
                specificity_level TEXT DEFAULT 'full',

                -- Format detection
                date_format_detected TEXT DEFAULT '',
                locale_hint TEXT DEFAULT 'US',

                created_at REAL NOT NULL
            )
            """)

            # Create indexes for existing tables
            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_elements_doc_id ON elements(doc_id)
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_elements_parent_id ON elements(parent_id)
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_elements_type ON elements(element_type)
            """)

            # ENHANCED: Create comprehensive indexes for element_dates table
            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_element_dates_element_pk ON element_dates(element_pk)
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_element_dates_timestamp ON element_dates(timestamp_value)
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_element_dates_year ON element_dates(year_value)
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_element_dates_year_month ON element_dates(year_value, month_value)
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_element_dates_iso_string ON element_dates(iso_string)
            """)

            # NEW: Enhanced indexes for temporal analysis
            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_element_dates_decade ON element_dates(decade_value)
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_element_dates_quarter ON element_dates(quarter_value)
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_element_dates_season ON element_dates(season_value)
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_element_dates_specificity ON element_dates(specificity_level)
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_element_dates_date_type ON element_dates(date_type)
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_element_dates_fiscal_year ON element_dates(fiscal_year_value)
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_element_dates_academic_year ON element_dates(academic_year_value)
            """)

            # Continue with existing tables...
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id TEXT PRIMARY KEY,
                source_id TEXT REFERENCES elements(element_id) ON DELETE CASCADE,
                relationship_type TEXT,
                target_reference TEXT,
                metadata TEXT
            )
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id)
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type)
            """)

            # Modified embeddings table with topic support
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                element_pk INTEGER PRIMARY KEY REFERENCES elements(element_pk) ON DELETE CASCADE,
                embedding TEXT,
                dimensions INTEGER,
                topics TEXT DEFAULT '[]',
                confidence REAL DEFAULT 1.0,
                created_at REAL
            )
            """)

            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_history (
                source_id TEXT PRIMARY KEY,
                content_hash TEXT,
                last_modified REAL,
                processing_count INTEGER DEFAULT 1
            )
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_processing_history_source_id ON processing_history(source_id)
            """)

            self.conn.commit()
            logger.info(
                "Created core database tables including enhanced element_dates with comprehensive temporal analysis")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error creating tables: {str(e)}")
            raise


if __name__ == "__main__":
    # Example demonstrating structured search with SQLite
    db = SQLiteDocumentDatabase("./test_db/document_db.sqlite")
    db.initialize()

    # Show backend capabilities
    capabilities = db.get_backend_capabilities()
    print(f"SQLite supports {len(capabilities.supported)} capabilities:")
    for cap in sorted(capabilities.get_supported_list()):
        print(f"  ✓ {cap}")

    # Example structured search
    from .structured_search import SearchQueryBuilder, LogicalOperator

    query = (SearchQueryBuilder()
             .with_operator(LogicalOperator.AND)
             .text_search("machine learning algorithms", similarity_threshold=0.8)
             .last_days(30)
             .topics(include=["ml%", "ai%"])
             .element_types(["header", "paragraph"])
             .include_dates(True)
             .include_topics_in_results(True)
             .build())

    print(f"\nExecuting structured search...")
    print(f"Query capabilities required: {len(query.get_required_capabilities())}")

    # Validate query
    missing = db.validate_query_support(query)
    if missing:
        print(f"Missing capabilities: {[m.value for m in missing]}")
    else:
        print("Query fully supported!")

        # Execute the search
        results = db.execute_structured_search(query)
        print(f"Found {len(results)} results")

        for result in results[:3]:  # Show first 3 results
            print(f"  - {result['element_id']}: {result['final_score']:.3f}")

    print("\nSQLite DocumentDatabase with structured search capabilities ready!")
    print("Features include:")
    print("- Complete structured search with logical operators")
    print("- Topic-aware embeddings with JSON support")
    print("- Comprehensive date storage and temporal analysis")
    print("- Optional vector extension integration")
    print("- Hierarchical element relationships")
    print("- Metadata filtering with JSON operators")
    print("- All base DocumentDatabase methods implemented")
