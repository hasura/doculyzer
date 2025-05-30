"""
SQLAlchemy implementation of document database with comprehensive structured search support.

This module provides a SQLAlchemy-based storage backend for the document pointer system,
with full structured search capabilities matching the PostgreSQL implementation.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Union, TYPE_CHECKING

import time

from .element_element import ElementHierarchical

# Import types for type checking only - these won't be imported at runtime
if TYPE_CHECKING:
    from sqlalchemy import (
        create_engine, Column, ForeignKey, String, Integer, Float, Text, LargeBinary, func, text,
        Engine
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship, scoped_session, Session
    import numpy as np
    from numpy.typing import NDArray

    # Define type aliases for type checking
    VectorType = NDArray[np.float32]
    SQLAlchemyEngineType = Engine
    SQLAlchemySessionType = Session
else:
    # Runtime type aliases
    VectorType = List[float]
    SQLAlchemyEngineType = Any
    SQLAlchemySessionType = Any

from .base import DocumentDatabase
from .element_relationship import ElementRelationship
from .element_element import ElementType, ElementBase

# Import structured search components
from .structured_search import (
    StructuredSearchQuery, SearchCriteriaGroup, BackendCapabilities, SearchCapability,
    UnsupportedSearchError, TextSearchCriteria, EmbeddingSearchCriteria, DateSearchCriteria,
    TopicSearchCriteria, MetadataSearchCriteria, ElementSearchCriteria,
    LogicalOperator, DateRangeOperator, SimilarityOperator
)

logger = logging.getLogger(__name__)

# Define global flags for availability
SQLALCHEMY_AVAILABLE = False
NUMPY_AVAILABLE = False
PGVECTOR_AVAILABLE = False
SQLITE_VEC_AVAILABLE = False
SQLITE_VSS_AVAILABLE = False

# Try to import SQLAlchemy conditionally at runtime
try:
    from sqlalchemy import (
        create_engine, Column, ForeignKey, String, Integer, Float, Text, LargeBinary, func, text
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship, scoped_session

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    logger.warning("SQLAlchemy not available. Install with 'pip install sqlalchemy'.")
    create_engine = None
    Column = None
    ForeignKey = None
    declarative_base = None
    sessionmaker = None
    relationship = None
    scoped_session = None

# Try to import NumPy conditionally at runtime
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("NumPy not available. Will use fallback vector operations.")

# Try to import pgvector conditionally
try:
    import pgvector
    PGVECTOR_AVAILABLE = True
except ImportError:
    logger.debug("pgvector not available. PostgreSQL vector operations will use native implementation.")

# Try to import sqlite-vec conditionally
try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    logger.debug("sqlite-vec not available.")

# Try to import sqlite-vss conditionally
try:
    import sqlite_vss
    SQLITE_VSS_AVAILABLE = True
except ImportError:
    logger.debug("sqlite-vss not available.")

# Try to import the config
try:
    from ..config import Config
    config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
except Exception as e:
    logger.warning(f"Error configuring SQLAlchemy provider: {str(e)}")
    config = None

# Create declarative base only if SQLAlchemy is available
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()

    # Define ORM models
    class Document(Base):
        """Document model for SQLAlchemy ORM."""
        __tablename__ = 'documents'

        doc_id = Column(String(255), primary_key=True)
        doc_type = Column(String(50))
        source = Column(String(1024))
        content_hash = Column(String(255))
        metadata_ = Column('metadata', Text)
        created_at = Column(Float)
        updated_at = Column(Float)

        # Relationships
        elements = relationship("Element", back_populates="document", cascade="all, delete-orphan")

    class Element(Base):
        """Element model for SQLAlchemy ORM."""
        __tablename__ = 'elements'

        element_pk = Column(Integer, primary_key=True, autoincrement=True)
        element_id = Column(String(255), unique=True, nullable=False)
        doc_id = Column(String(255), ForeignKey('documents.doc_id', ondelete='CASCADE'))
        element_type = Column(String(50))
        parent_id = Column(String(255), ForeignKey('elements.element_id'))
        content_preview = Column(Text)
        content_location = Column(Text)
        content_hash = Column(String(255))
        metadata_ = Column('metadata', Text)

        # Relationships
        document = relationship("Document", back_populates="elements")
        embedding = relationship("Embedding", uselist=False, back_populates="element", cascade="all, delete-orphan")
        relationships_as_source = relationship("Relationship", foreign_keys="Relationship.source_id",
                                               cascade="all, delete-orphan")
        children = relationship("Element", backref="parent", remote_side=[element_id])
        dates = relationship("ElementDate", back_populates="element", cascade="all, delete-orphan")

    class Relationship(Base):
        """Relationship model for SQLAlchemy ORM."""
        __tablename__ = 'relationships'

        relationship_id = Column(String(255), primary_key=True)
        source_id = Column(String(255), ForeignKey('elements.element_id', ondelete='CASCADE'))
        relationship_type = Column(String(50))
        target_reference = Column(String(255))
        metadata_ = Column('metadata', Text)

    class Embedding(Base):
        """Enhanced Embedding model with topic support for SQLAlchemy ORM."""
        __tablename__ = 'embeddings'

        element_pk = Column(Integer, ForeignKey('elements.element_pk', ondelete='CASCADE'), primary_key=True)
        embedding = Column(LargeBinary)
        dimensions = Column(Integer)
        topics = Column(Text)  # JSON array of topic strings
        confidence = Column(Float, default=1.0)
        created_at = Column(Float)

        # Relationships
        element = relationship("Element", back_populates="embedding")

    class ElementDate(Base):
        """Element dates model for SQLAlchemy ORM."""
        __tablename__ = 'element_dates'

        id = Column(Integer, primary_key=True, autoincrement=True)
        element_pk = Column(Integer, ForeignKey('elements.element_pk', ondelete='CASCADE'))
        element_id = Column(String(255), ForeignKey('elements.element_id', ondelete='CASCADE'))
        timestamp_value = Column(Float)
        date_text = Column(Text)
        specificity_level = Column(String(20))
        metadata_ = Column('metadata', Text)

        # Relationships
        element = relationship("Element", back_populates="dates")

    class ProcessingHistory(Base):
        """Processing history model for SQLAlchemy ORM."""
        __tablename__ = 'processing_history'

        source_id = Column(String(1024), primary_key=True)
        content_hash = Column(String(255))
        last_modified = Column(Float)
        processing_count = Column(Integer, default=1)
else:
    # Define placeholder classes if SQLAlchemy is not available
    Document = None
    Element = None
    Relationship = None
    Embedding = None
    ElementDate = None
    ProcessingHistory = None
    Base = None


class SQLAlchemyDocumentDatabase(DocumentDatabase):
    """SQLAlchemy implementation with comprehensive structured search support."""

    def __init__(self, db_uri: str, echo: bool = False):
        """
        Initialize SQLAlchemy document database.

        Args:
            db_uri: Database URI (e.g. 'sqlite:///path/to/database.db',
                                 'postgresql://user:pass@localhost/dbname')
            echo: Whether to echo SQL statements
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for SQLAlchemyDocumentDatabase")

        self.config = None
        self.db_uri = db_uri
        self.echo = echo
        self.engine: SQLAlchemyEngineType = None
        self.Session = None
        self.session: SQLAlchemySessionType = None
        self._vector_extension = None
        self._vector_dimension = config.config.get('embedding', {}).get('dimensions', 384) if config else 384
        self.embedding_generator = None

    # ========================================
    # STRUCTURED SEARCH IMPLEMENTATION
    # ========================================

    def get_backend_capabilities(self) -> BackendCapabilities:
        """
        SQLAlchemy supports comprehensive search capabilities across multiple databases.
        """
        supported = {
            # Core search types
            SearchCapability.TEXT_SIMILARITY,
            SearchCapability.EMBEDDING_SIMILARITY,
            SearchCapability.FULL_TEXT_SEARCH,

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
            SearchCapability.RESULT_HIGHLIGHTING,
        }

        # Add vector search if available
        if self._vector_extension in ["pgvector", "vec0", "vss0"]:
            supported.add(SearchCapability.VECTOR_SEARCH)

        return BackendCapabilities(supported)

    def execute_structured_search(self, query: StructuredSearchQuery) -> List[Dict[str, Any]]:
        """
        Execute a structured search query using SQLAlchemy ORM.
        """
        if not self.session:
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
        """Execute date-based filtering using SQLAlchemy ORM."""
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
        """Execute topic-based filtering using SQLAlchemy ORM."""
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
        """Execute metadata-based filtering using SQLAlchemy ORM."""
        try:
            # Build SQLAlchemy query for metadata filtering
            query = self.session.query(Element.element_pk)

            # Add exact matches using database-specific JSON operators
            for key, value in criteria.exact_matches.items():
                if self.db_uri.startswith('postgresql'):
                    # PostgreSQL JSONB operator
                    query = query.filter(text(f"metadata_->>'{key}' = :value_{key}").params(**{f"value_{key}": str(value)}))
                elif self.db_uri.startswith('sqlite'):
                    # SQLite JSON1 extension
                    query = query.filter(text(f"json_extract(metadata_, '$.{key}') = :value_{key}").params(**{f"value_{key}": str(value)}))
                else:
                    # Fallback to LIKE search
                    query = query.filter(Element.metadata_.like(f'%"{key}"%"{value}"%'))

            # Add LIKE patterns
            for key, pattern in criteria.like_patterns.items():
                if self.db_uri.startswith('postgresql'):
                    query = query.filter(text(f"metadata_->>'{key}' LIKE :pattern_{key}").params(**{f"pattern_{key}": pattern}))
                elif self.db_uri.startswith('sqlite'):
                    query = query.filter(text(f"json_extract(metadata_, '$.{key}') LIKE :pattern_{key}").params(**{f"pattern_{key}": pattern}))
                else:
                    query = query.filter(Element.metadata_.like(f'%{pattern}%'))

            # Add range filters
            for key, range_filter in criteria.range_filters.items():
                if self.db_uri.startswith('postgresql'):
                    if 'gte' in range_filter:
                        query = query.filter(text(f"(metadata_->>'{key}')::numeric >= :gte_{key}").params(**{f"gte_{key}": range_filter['gte']}))
                    if 'lte' in range_filter:
                        query = query.filter(text(f"(metadata_->>'{key}')::numeric <= :lte_{key}").params(**{f"lte_{key}": range_filter['lte']}))
                    if 'gt' in range_filter:
                        query = query.filter(text(f"(metadata_->>'{key}')::numeric > :gt_{key}").params(**{f"gt_{key}": range_filter['gt']}))
                    if 'lt' in range_filter:
                        query = query.filter(text(f"(metadata_->>'{key}')::numeric < :lt_{key}").params(**{f"lt_{key}": range_filter['lt']}))
                elif self.db_uri.startswith('sqlite'):
                    if 'gte' in range_filter:
                        query = query.filter(text(f"CAST(json_extract(metadata_, '$.{key}') AS REAL) >= :gte_{key}").params(**{f"gte_{key}": range_filter['gte']}))
                    if 'lte' in range_filter:
                        query = query.filter(text(f"CAST(json_extract(metadata_, '$.{key}') AS REAL) <= :lte_{key}").params(**{f"lte_{key}": range_filter['lte']}))
                    if 'gt' in range_filter:
                        query = query.filter(text(f"CAST(json_extract(metadata_, '$.{key}') AS REAL) > :gt_{key}").params(**{f"gt_{key}": range_filter['gt']}))
                    if 'lt' in range_filter:
                        query = query.filter(text(f"CAST(json_extract(metadata_, '$.{key}') AS REAL) < :lt_{key}").params(**{f"lt_{key}": range_filter['lt']}))

            # Add exists filters
            for key in criteria.exists_filters:
                if self.db_uri.startswith('postgresql'):
                    query = query.filter(text(f"metadata_ ? '{key}'"))
                elif self.db_uri.startswith('sqlite'):
                    query = query.filter(text(f"json_extract(metadata_, '$.{key}') IS NOT NULL"))
                else:
                    query = query.filter(Element.metadata_.like(f'%"{key}"%'))

            # Execute query and limit results
            element_pks = [row[0] for row in query.limit(1000).all()]

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
        """Execute element-based filtering using SQLAlchemy ORM."""
        try:
            # Build SQLAlchemy query for element filtering
            query = self.session.query(Element.element_pk)

            # Add element type filter
            if criteria.element_types:
                type_values = self.prepare_element_type_query(criteria.element_types)
                if type_values:
                    if len(type_values) == 1:
                        query = query.filter(Element.element_type == type_values[0])
                    else:
                        query = query.filter(Element.element_type.in_(type_values))

            # Add document ID filters
            if criteria.doc_ids:
                query = query.filter(Element.doc_id.in_(criteria.doc_ids))

            if criteria.exclude_doc_ids:
                query = query.filter(~Element.doc_id.in_(criteria.exclude_doc_ids))

            # Add content length filters
            if criteria.content_length_min is not None:
                query = query.filter(func.length(Element.content_preview) >= criteria.content_length_min)

            if criteria.content_length_max is not None:
                query = query.filter(func.length(Element.content_preview) <= criteria.content_length_max)

            # Add parent element filters
            if criteria.parent_element_ids:
                query = query.filter(Element.parent_id.in_(criteria.parent_element_ids))

            # Execute query and limit results
            element_pks = [row[0] for row in query.limit(1000).all()]

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

    def _generate_embedding(self, search_text: str) -> List[float]:
        """Generate embedding for search text."""
        try:
            from ..embeddings import get_embedding_generator

            if self.embedding_generator is None:
                config_obj = self.config
                if not config_obj:
                    from ..config import Config
                    config_obj = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
                self.embedding_generator = get_embedding_generator(config_obj)

            return self.embedding_generator.generate(search_text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def _get_element_pks_in_date_range(self, start_date: Optional[datetime],
                                       end_date: Optional[datetime]) -> List[int]:
        """Get element_pks that have dates within the specified range."""
        if not (start_date or end_date):
            return []

        # Build query using SQLAlchemy ORM
        query = self.session.query(ElementDate.element_pk.distinct())

        if start_date:
            query = query.filter(ElementDate.timestamp_value >= start_date.timestamp())

        if end_date:
            query = query.filter(ElementDate.timestamp_value <= end_date.timestamp())

        return [row[0] for row in query.all()]

    def _filter_by_specificity(self, element_pks: List[int],
                               allowed_levels: List[str]) -> List[int]:
        """Filter element PKs by date specificity levels."""
        if not element_pks or not allowed_levels:
            return element_pks

        # Query using SQLAlchemy ORM
        query = self.session.query(ElementDate.element_pk.distinct()).filter(
            ElementDate.element_pk.in_(element_pks),
            ElementDate.specificity_level.in_(allowed_levels)
        )

        return [row[0] for row in query.all()]

    # ========================================
    # CORE DATABASE OPERATIONS (existing methods)
    # ========================================

    def initialize(self) -> None:
        """Initialize the database by creating tables if they don't exist."""
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for SQLAlchemyDocumentDatabase")

        # Create directory if it's a sqlite file
        if self.db_uri.startswith('sqlite:///'):
            db_path = self.db_uri.replace('sqlite:///', '')
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        # Create engine
        self.engine = create_engine(self.db_uri, echo=self.echo)

        # Create session factory
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        self.session = self.Session()

        # Create tables
        Base.metadata.create_all(self.engine)

        # Check for vector extension support
        self._check_vector_extension()

        logger.info(f"Initialized SQLAlchemy database with URI: {self.db_uri}")

    def _check_vector_extension(self) -> None:
        """Check for vector extension support in the database."""
        if self.db_uri.startswith('postgresql'):
            try:
                # Check for pgvector
                result = self.session.execute(
                    text("SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'vector')"))
                pgvector_available = result.scalar()

                if pgvector_available:
                    # Check if installed
                    result = self.session.execute(
                        text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"))
                    pgvector_installed = result.scalar()

                    if not pgvector_installed:
                        try:
                            # Try to install
                            self.session.execute(text("CREATE EXTENSION vector"))
                            self.session.commit()
                            self._vector_extension = "pgvector"
                            logger.info("Installed pgvector extension")
                        except Exception as e:
                            logger.warning(f"Failed to install pgvector extension: {str(e)}")
                    else:
                        self._vector_extension = "pgvector"
                        logger.info("Using pgvector extension")
            except Exception as e:
                logger.warning(f"Error checking for vector extension: {str(e)}")
        elif self.db_uri.startswith('sqlite'):
            # Check for sqlite vector extensions
            if SQLITE_VEC_AVAILABLE:
                self._vector_extension = "vec0"
                logger.info("Using sqlite-vec extension")
                return
            elif SQLITE_VSS_AVAILABLE:
                self._vector_extension = "vss0"
                logger.info("Using sqlite-vss extension")
                return

            logger.info("No vector extensions found, using native implementation")

    def close(self) -> None:
        """Close the database connection."""
        if self.session:
            self.session.close()
            self.session = None

        if self.engine:
            self.engine.dispose()
            self.engine = None

    # [Include all existing methods from the original implementation]
    # For brevity, I'm including key ones but all others remain the same

    def get_element(self, element_id_or_pk: Union[int, str]) -> Optional[Dict[str, Any]]:
        """Get element by ID or PK."""
        if not self.session:
            raise ValueError("Database not initialized")

        # Try to interpret as element_pk (integer) first
        try:
            element_pk = int(element_id_or_pk)
            element = self.session.query(Element).filter_by(element_pk=element_pk).first()
        except (ValueError, TypeError):
            # If not an integer, treat as element_id (string)
            element = self.session.query(Element).filter_by(element_id=element_id_or_pk).first()

        if not element:
            return None

        # Convert to dictionary
        result = {
            "element_id": element.element_id,
            "element_pk": element.element_pk,
            "doc_id": element.doc_id,
            "element_type": element.element_type,
            "parent_id": element.parent_id,
            "content_preview": element.content_preview,
            "content_location": element.content_location,
            "content_hash": element.content_hash
        }

        # Parse metadata JSON
        try:
            result["metadata"] = json.loads(element.metadata_)
        except (json.JSONDecodeError, TypeError):
            result["metadata"] = {}

        return result

    # ========================================
    # DATE STORAGE AND SEARCH METHODS
    # ========================================

    def store_element_dates(self, element_id: str, dates: List[Dict[str, Any]]) -> None:
        """Store extracted dates associated with an element."""
        if not self.session:
            raise ValueError("Database not initialized")

        # Get element to find its PK
        element = self.session.query(Element).filter_by(element_id=element_id).first()
        if not element:
            raise ValueError(f"Element not found: {element_id}")

        try:
            # Store each date
            for date_dict in dates:
                date_record = ElementDate(
                    element_pk=element.element_pk,
                    element_id=element_id,
                    timestamp_value=date_dict.get('timestamp'),
                    date_text=date_dict.get('date_text', ''),
                    specificity_level=date_dict.get('specificity_level', 'day'),
                    metadata_=json.dumps(date_dict.get('metadata', {}))
                )
                self.session.add(date_record)

            self.session.commit()
            logger.debug(f"Stored {len(dates)} dates for element {element_id}")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error storing dates for element {element_id}: {str(e)}")
            raise

    def get_element_dates(self, element_id: str) -> List[Dict[str, Any]]:
        """Get all dates associated with an element."""
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            dates = self.session.query(ElementDate).filter_by(element_id=element_id).all()

            result = []
            for date_record in dates:
                date_dict = {
                    'timestamp': date_record.timestamp_value,
                    'date_text': date_record.date_text,
                    'specificity_level': date_record.specificity_level
                }

                # Parse metadata
                try:
                    date_dict['metadata'] = json.loads(date_record.metadata_)
                except (json.JSONDecodeError, TypeError):
                    date_dict['metadata'] = {}

                result.append(date_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting dates for element {element_id}: {str(e)}")
            return []

    def store_embedding_with_dates(self, element_id: str, embedding: List[float],
                                   dates: List[Dict[str, Any]]) -> None:
        """Store both embedding and dates for an element in a single operation."""
        if not self.session:
            raise ValueError("Database not initialized")

        # Get element to find its PK
        element = self.session.query(Element).filter_by(element_id=element_id).first()
        if not element:
            raise ValueError(f"Element not found: {element_id}")

        try:
            self.session.begin()

            # Store embedding
            self.store_embedding(element.element_pk, embedding)

            # Store dates
            self.store_element_dates(element_id, dates)

            self.session.commit()
            logger.debug(f"Stored embedding and {len(dates)} dates for element {element_id}")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error storing embedding and dates for element {element_id}: {str(e)}")
            raise

    def delete_element_dates(self, element_id: str) -> bool:
        """Delete all dates associated with an element."""
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            deleted_count = self.session.query(ElementDate).filter_by(element_id=element_id).delete()
            self.session.commit()

            return deleted_count > 0

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting dates for element {element_id}: {str(e)}")
            return False

    def search_elements_by_date_range(self, start_date: datetime, end_date: datetime,
                                      limit: int = 100) -> List[Dict[str, Any]]:
        """Find elements that contain dates within a specified range."""
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            # Query using JOIN to get element details
            query = self.session.query(Element).join(ElementDate).filter(
                ElementDate.timestamp_value >= start_date.timestamp(),
                ElementDate.timestamp_value <= end_date.timestamp()
            ).distinct().limit(limit)

            elements = query.all()

            result = []
            for element in elements:
                element_dict = {
                    "element_id": element.element_id,
                    "element_pk": element.element_pk,
                    "doc_id": element.doc_id,
                    "element_type": element.element_type,
                    "parent_id": element.parent_id,
                    "content_preview": element.content_preview,
                    "content_location": element.content_location,
                    "content_hash": element.content_hash
                }

                # Parse metadata
                try:
                    element_dict["metadata"] = json.loads(element.metadata_)
                except (json.JSONDecodeError, TypeError):
                    element_dict["metadata"] = {}

                result.append(element_dict)

            return result

        except Exception as e:
            logger.error(f"Error searching elements by date range: {str(e)}")
            return []

    def search_by_text_and_date_range(self, search_text: str,
                                      start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None,
                                      limit: int = 10) -> List[Tuple[int, float]]:
        """Search elements by semantic similarity AND date range."""
        try:
            # Generate embedding for search text
            query_embedding = self._generate_embedding(search_text)

            # Get elements in date range
            if start_date and end_date:
                date_element_pks = self._get_element_pks_in_date_range(start_date, end_date)

                # Use date filtering in embedding search
                filter_criteria = {"element_pk": date_element_pks}
                return self.search_by_embedding(query_embedding, limit, filter_criteria)
            else:
                return self.search_by_embedding(query_embedding, limit)

        except Exception as e:
            logger.error(f"Error in text and date range search: {str(e)}")
            return []

    def search_by_embedding_and_date_range(self, query_embedding: List[float],
                                           start_date: Optional[datetime] = None,
                                           end_date: Optional[datetime] = None,
                                           limit: int = 10) -> List[Tuple[int, float]]:
        """Search elements by embedding similarity AND date range."""
        try:
            # Get elements in date range
            if start_date and end_date:
                date_element_pks = self._get_element_pks_in_date_range(start_date, end_date)

                # Use date filtering in embedding search
                filter_criteria = {"element_pk": date_element_pks}
                return self.search_by_embedding(query_embedding, limit, filter_criteria)
            else:
                return self.search_by_embedding(query_embedding, limit)

        except Exception as e:
            logger.error(f"Error in embedding and date range search: {str(e)}")
            return []

    def get_elements_with_dates(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all elements that have associated dates."""
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            # Query using JOIN to get elements with dates
            query = self.session.query(Element).join(ElementDate).distinct().limit(limit)

            elements = query.all()

            result = []
            for element in elements:
                element_dict = {
                    "element_id": element.element_id,
                    "element_pk": element.element_pk,
                    "doc_id": element.doc_id,
                    "element_type": element.element_type,
                    "parent_id": element.parent_id,
                    "content_preview": element.content_preview,
                    "content_location": element.content_location,
                    "content_hash": element.content_hash
                }

                # Parse metadata
                try:
                    element_dict["metadata"] = json.loads(element.metadata_)
                except (json.JSONDecodeError, TypeError):
                    element_dict["metadata"] = {}

                result.append(element_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting elements with dates: {str(e)}")
            return []

    def get_date_statistics(self) -> Dict[str, Any]:
        """Get statistics about dates in the database."""
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            # Count total dates
            total_dates = self.session.query(ElementDate).count()

            # Count elements with dates
            elements_with_dates = self.session.query(ElementDate.element_pk.distinct()).count()

            # Get date range
            min_date_result = self.session.query(func.min(ElementDate.timestamp_value)).scalar()
            max_date_result = self.session.query(func.max(ElementDate.timestamp_value)).scalar()

            # Count by specificity level
            specificity_counts = {}
            specificity_query = self.session.query(
                ElementDate.specificity_level,
                func.count(ElementDate.id)
            ).group_by(ElementDate.specificity_level).all()

            for level, count in specificity_query:
                specificity_counts[level] = count

            return {
                'total_dates': total_dates,
                'elements_with_dates': elements_with_dates,
                'earliest_date': datetime.fromtimestamp(min_date_result) if min_date_result else None,
                'latest_date': datetime.fromtimestamp(max_date_result) if max_date_result else None,
                'specificity_distribution': specificity_counts
            }

        except Exception as e:
            logger.error(f"Error getting date statistics: {str(e)}")
            return {}

    # [Continue with all other existing methods from the original implementation]
    # The rest of the methods (store_document, find_documents, search_by_embedding, etc.)
    # remain exactly the same as in the original SQLAlchemy implementation

    # For the complete implementation, include all remaining methods here...

    def get_last_processed_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about when a document was last processed."""
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            history = self.session.query(ProcessingHistory).filter_by(source_id=source_id).first()

            if not history:
                return None

            return {
                "source_id": history.source_id,
                "content_hash": history.content_hash,
                "last_modified": history.last_modified,
                "processing_count": history.processing_count
            }
        except Exception as e:
            logger.error(f"Error getting processing history for {source_id}: {str(e)}")
            return None

    def update_processing_history(self, source_id: str, content_hash: str) -> None:
        """Update the processing history for a document."""
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            # Check if record exists
            history = self.session.query(ProcessingHistory).filter_by(source_id=source_id).first()

            if history:
                # Update existing record
                history.content_hash = content_hash
                history.last_modified = time.time()
                history.processing_count += 1
            else:
                # Create new record
                history = ProcessingHistory(
                    source_id=source_id,
                    content_hash=content_hash,
                    last_modified=time.time(),
                    processing_count=1
                )
                self.session.add(history)

            self.session.commit()
            logger.debug(f"Updated processing history for {source_id}")

        except Exception as e:
            self.session.rollback()
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
        if not self.session:
            raise ValueError("Database not initialized")

        source = document.get("source", "")
        content_hash = document.get("content_hash", "")

        # Check if document already exists with this source
        existing_doc = None
        if source:
            existing_doc = self.session.query(Document).filter_by(source=source).first()

        if existing_doc:
            # Document exists with same source, update it
            doc_id = existing_doc.doc_id
            document["doc_id"] = doc_id  # Use existing doc_id

            # Update all elements to use the existing doc_id
            for element in elements:
                element["doc_id"] = doc_id

            self.update_document(doc_id, document, elements, relationships)
            return

        try:
            # Start a transaction
            self.session.begin()

            # Create document record
            doc_id = document["doc_id"]
            doc_record = Document(
                doc_id=doc_id,
                doc_type=document.get("doc_type", ""),
                source=source,
                content_hash=content_hash,
                metadata_=json.dumps(document.get("metadata", {})),
                created_at=document.get("created_at", time.time()),
                updated_at=document.get("updated_at", time.time())
            )
            self.session.add(doc_record)
            self.session.flush()  # Flush to get doc_id if it's generated

            # Store elements
            element_records = {}
            for element in elements:
                element_id = element["element_id"]
                element_record = Element(
                    element_id=element_id,
                    doc_id=element.get("doc_id", doc_id),
                    element_type=element.get("element_type", ""),
                    parent_id=element.get("parent_id"),
                    content_preview=element.get("content_preview", ""),
                    content_location=element.get("content_location", ""),
                    content_hash=element.get("content_hash", ""),
                    metadata_=json.dumps(element.get("metadata", {}))
                )
                self.session.add(element_record)
                element_records[element_id] = element_record

            # Flush to get element PKs
            self.session.flush()

            # Update the original elements with their PKs
            for element in elements:
                element_id = element["element_id"]
                if element_id in element_records:
                    element["element_pk"] = element_records[element_id].element_pk

            # Store relationships
            for relationship in relationships:
                relationship_id = relationship["relationship_id"]
                relationship_record = Relationship(
                    relationship_id=relationship_id,
                    source_id=relationship.get("source_id", ""),
                    relationship_type=relationship.get("relationship_type", ""),
                    target_reference=relationship.get("target_reference", ""),
                    metadata_=json.dumps(relationship.get("metadata", {}))
                )
                self.session.add(relationship_record)

            # Commit the transaction
            self.session.commit()

            # Update processing history
            if source:
                self.update_processing_history(source, content_hash)

            logger.info(
                f"Stored document {doc_id} with {len(elements)} elements and {len(relationships)} relationships")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error storing document {document.get('doc_id')}: {str(e)}")
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
        if not self.session:
            raise ValueError("Database not initialized")

        # Check if document exists
        existing_doc = self.session.query(Document).filter_by(doc_id=doc_id).first()
        if not existing_doc:
            raise ValueError(f"Document not found: {doc_id}")

        source = document.get("source", "")
        content_hash = document.get("content_hash", "")

        try:
            # Start a transaction
            self.session.begin()

            # Delete existing relationships
            self.session.query(Relationship).filter(
                Relationship.source_id.in_(
                    self.session.query(Element.element_id).filter_by(doc_id=doc_id)
                )
            ).delete(synchronize_session=False)

            # Delete existing embeddings
            self.session.query(Embedding).filter(
                Embedding.element_pk.in_(
                    self.session.query(Element.element_pk).filter_by(doc_id=doc_id)
                )
            ).delete(synchronize_session=False)

            # Delete existing element dates
            self.session.query(ElementDate).filter(
                ElementDate.element_pk.in_(
                    self.session.query(Element.element_pk).filter_by(doc_id=doc_id)
                )
            ).delete(synchronize_session=False)

            # Delete existing elements
            self.session.query(Element).filter_by(doc_id=doc_id).delete(synchronize_session=False)

            # Update document record
            existing_doc.doc_type = document.get("doc_type", existing_doc.doc_type)
            existing_doc.source = source
            existing_doc.content_hash = content_hash
            existing_doc.metadata_ = json.dumps(document.get("metadata", {}))
            existing_doc.updated_at = time.time()

            # Store elements
            element_records = {}
            for element in elements:
                element_id = element["element_id"]
                element_record = Element(
                    element_id=element_id,
                    doc_id=doc_id,
                    element_type=element.get("element_type", ""),
                    parent_id=element.get("parent_id"),
                    content_preview=element.get("content_preview", ""),
                    content_location=element.get("content_location", ""),
                    content_hash=element.get("content_hash", ""),
                    metadata_=json.dumps(element.get("metadata", {}))
                )
                self.session.add(element_record)
                element_records[element_id] = element_record

            # Flush to get element PKs
            self.session.flush()

            # Update the original elements with their PKs
            for element in elements:
                element_id = element["element_id"]
                if element_id in element_records:
                    element["element_pk"] = element_records[element_id].element_pk

            # Store relationships
            for relationship in relationships:
                relationship_id = relationship["relationship_id"]
                relationship_record = Relationship(
                    relationship_id=relationship_id,
                    source_id=relationship.get("source_id", ""),
                    relationship_type=relationship.get("relationship_type", ""),
                    target_reference=relationship.get("target_reference", ""),
                    metadata_=json.dumps(relationship.get("metadata", {}))
                )
                self.session.add(relationship_record)

            # Commit the transaction
            self.session.commit()

            # Update processing history
            if source:
                self.update_processing_history(source, content_hash)

            logger.info(
                f"Updated document {doc_id} with {len(elements)} elements and {len(relationships)} relationships")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating document {doc_id}: {str(e)}")
            raise

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        if not self.session:
            raise ValueError("Database not initialized")

        document = self.session.query(Document).filter_by(doc_id=doc_id).first()
        if not document:
            return None

        # Convert to dictionary
        result = {
            "doc_id": document.doc_id,
            "doc_type": document.doc_type,
            "source": document.source,
            "content_hash": document.content_hash,
            "created_at": document.created_at,
            "updated_at": document.updated_at
        }

        # Parse metadata JSON
        try:
            result["metadata"] = json.loads(document.metadata_)
        except (json.JSONDecodeError, TypeError):
            result["metadata"] = {}

        return result

    def get_document_elements(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get elements for a document."""
        if not self.session:
            raise ValueError("Database not initialized")

        # Modified to handle doc_id being either an actual doc_id or a source
        elements = self.session.query(Element).join(
            Document, Element.doc_id == Document.doc_id
        ).filter(
            (Document.doc_id == doc_id) | (Document.source == doc_id)
        ).order_by(Element.element_id).all()

        result = []
        for element in elements:
            # Convert to dictionary
            element_dict = {
                "element_id": element.element_id,
                "element_pk": element.element_pk,
                "doc_id": element.doc_id,
                "element_type": element.element_type,
                "parent_id": element.parent_id,
                "content_preview": element.content_preview,
                "content_location": element.content_location,
                "content_hash": element.content_hash
            }

            # Parse metadata JSON
            try:
                element_dict["metadata"] = json.loads(element.metadata_)
            except (json.JSONDecodeError, TypeError):
                element_dict["metadata"] = {}

            result.append(element_dict)

        return result

    def get_document_relationships(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get relationships for a document."""
        if not self.session:
            raise ValueError("Database not initialized")

        # Get all element IDs for this document
        element_ids = [row[0] for row in
                       self.session.query(Element.element_id).filter_by(doc_id=doc_id).all()]

        if not element_ids:
            return []

        # Get relationships involving these elements
        relationships = self.session.query(Relationship).filter(
            Relationship.source_id.in_(element_ids)
        ).all()

        result = []
        for relationship in relationships:
            # Convert to dictionary
            rel_dict = {
                "relationship_id": relationship.relationship_id,
                "source_id": relationship.source_id,
                "relationship_type": relationship.relationship_type,
                "target_reference": relationship.target_reference
            }

            # Parse metadata JSON
            try:
                rel_dict["metadata"] = json.loads(relationship.metadata_)
            except (json.JSONDecodeError, TypeError):
                rel_dict["metadata"] = {}

            result.append(rel_dict)

        return result

    def get_outgoing_relationships(self, element_pk: Union[int, str]) -> List[ElementRelationship]:
        """
        Find all relationships where the specified element_pk is the source.

        Implementation for SQLAlchemy database using JOIN to efficiently retrieve target information.

        Args:
            element_pk: The primary key of the element or element_id

        Returns:
            List of ElementRelationship objects where the specified element is the source
        """
        if not self.session:
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
            # Use SQLAlchemy's query builder to create a JOIN query
            # This joins relationships with elements to get target element information
            query = self.session.query(
                Relationship,
                Element.element_pk.label('target_element_pk'),
                Element.element_type.label('target_element_type'),
                Element.content_preview.label('target_content_preview')
            ).outerjoin(
                Element,
                Relationship.target_reference == Element.element_id
            ).filter(
                Relationship.source_id == element_id
            )

            # Execute the query
            results = query.all()

            # Process results
            for row in results:
                rel = row[0]  # The Relationship object
                target_element_pk = row[1]  # The target element_pk from the join
                target_element_type = row[2]  # The target element_type from the join
                target_content_preview = row[3]  # The target content_preview from the join

                # Convert the relationship to a dictionary for easier processing
                rel_dict = {
                    "relationship_id": rel.relationship_id,
                    "source_id": rel.source_id,
                    "relationship_type": rel.relationship_type,
                    "target_reference": rel.target_reference,
                    "doc_id": getattr(rel, "doc_id", None)
                }

                # Convert metadata from JSON
                try:
                    rel_dict["metadata"] = json.loads(rel.metadata_)
                except (json.JSONDecodeError, TypeError):
                    rel_dict["metadata"] = {}

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

    def find_documents(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find documents matching query with support for LIKE patterns.

        Args:
            query: Query parameters. Enhanced syntax supports:
                   - Exact matches: {"doc_type": "pdf"}
                   - LIKE patterns: {"source_like": "%reports%"}
                   - Case-insensitive LIKE: {"source_ilike": "%REPORTS%"} (if supported)
                   - List matching: {"doc_type": ["pdf", "docx"]}
                   - Metadata exact: {"metadata": {"author": "John"}}
                   - Metadata LIKE: {"metadata_like": {"title": "%annual%"}}
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        if not self.session:
            raise ValueError("Database not initialized")

        if query is None:
            query = {}

        # Build query
        db_query = self.session.query(Document)

        # Apply filters
        for key, value in query.items():
            if key == "metadata":
                # Handle metadata exact matches
                for meta_key, meta_value in value.items():
                    # Use database-specific JSON extraction
                    if self.db_uri.startswith('postgresql'):
                        # PostgreSQL JSONB operator
                        json_filter = text(f"metadata_->>'{meta_key}' = :meta_value")
                        db_query = db_query.filter(json_filter.params(meta_value=str(meta_value)))
                    elif self.db_uri.startswith('sqlite'):
                        # SQLite JSON1 extension
                        json_filter = text(f"json_extract(metadata_, '$.{meta_key}') = :meta_value")
                        db_query = db_query.filter(json_filter.params(meta_value=str(meta_value)))
                    else:
                        # Fallback to simple text search
                        db_query = db_query.filter(Document.metadata_.like(f'%"{meta_key}"%"{meta_value}"%'))
            elif key == "metadata_like":
                # Handle metadata LIKE patterns
                for meta_key, meta_value in value.items():
                    if self.db_uri.startswith('postgresql'):
                        json_filter = text(f"metadata_->>'{meta_key}' LIKE :meta_value")
                        db_query = db_query.filter(json_filter.params(meta_value=str(meta_value)))
                    elif self.db_uri.startswith('sqlite'):
                        json_filter = text(f"json_extract(metadata_, '$.{meta_key}') LIKE :meta_value")
                        db_query = db_query.filter(json_filter.params(meta_value=str(meta_value)))
                    else:
                        # Fallback to simple text search
                        db_query = db_query.filter(Document.metadata_.like(f'%{meta_value}%'))
            elif key == "metadata_ilike":
                # Handle case-insensitive metadata LIKE patterns
                for meta_key, meta_value in value.items():
                    if self.db_uri.startswith('postgresql'):
                        json_filter = text(f"metadata_->>'{meta_key}' ILIKE :meta_value")
                        db_query = db_query.filter(json_filter.params(meta_value=str(meta_value)))
                    else:
                        # Fallback to case-insensitive LIKE
                        if self.db_uri.startswith('sqlite'):
                            json_filter = text(f"json_extract(metadata_, '$.{meta_key}') LIKE :meta_value")
                            db_query = db_query.filter(json_filter.params(meta_value=str(meta_value)))
                        else:
                            db_query = db_query.filter(Document.metadata_.like(f'%{meta_value}%'))
            elif key.endswith("_like"):
                # LIKE pattern for regular fields
                field_name = key[:-5]  # Remove '_like' suffix
                if hasattr(Document, field_name):
                    db_query = db_query.filter(getattr(Document, field_name).like(value))
            elif key.endswith("_ilike"):
                # Case-insensitive LIKE pattern
                field_name = key[:-6]  # Remove '_ilike' suffix
                if hasattr(Document, field_name):
                    if self.db_uri.startswith('postgresql'):
                        # PostgreSQL has native ILIKE support
                        db_query = db_query.filter(getattr(Document, field_name).op('ILIKE')(value))
                    else:
                        # Fallback to case-insensitive LIKE
                        db_query = db_query.filter(func.lower(getattr(Document, field_name)).like(func.lower(value)))
            elif hasattr(Document, key):
                if isinstance(value, list):
                    # Handle list of values (IN condition)
                    db_query = db_query.filter(getattr(Document, key).in_(value))
                else:
                    # Simple equality
                    db_query = db_query.filter(getattr(Document, key) == value)

        # Apply limit
        db_query = db_query.limit(limit)

        # Execute query
        documents = db_query.all()

        # Convert to dictionaries
        result = []
        for document in documents:
            doc_dict = {
                "doc_id": document.doc_id,
                "doc_type": document.doc_type,
                "source": document.source,
                "content_hash": document.content_hash,
                "created_at": document.created_at,
                "updated_at": document.updated_at
            }

            # Parse metadata JSON
            try:
                doc_dict["metadata"] = json.loads(document.metadata_)
            except (json.JSONDecodeError, TypeError):
                doc_dict["metadata"] = {}

            result.append(doc_dict)

        return result

    def find_elements(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements matching query with support for LIKE patterns and ElementType enums.

        Args:
            query: Query parameters. Enhanced syntax supports:
                   - Exact matches: {"element_type": "header"}
                   - ElementType enums: {"element_type": ElementType.HEADER}
                   - Multiple enums: {"element_type": [ElementType.HEADER, ElementType.PARAGRAPH]}
                   - LIKE patterns: {"content_preview_like": "%summary%"}
                   - Case-insensitive LIKE: {"content_preview_ilike": "%SUMMARY%"} (if supported)
                   - List matching: {"doc_id": ["doc1", "doc2"]}
                   - Metadata exact: {"metadata": {"section": "intro"}}
                   - Metadata LIKE: {"metadata_like": {"title": "%chapter%"}}
            limit: Maximum number of results

        Returns:
            List of matching elements
        """
        if not self.session:
            raise ValueError("Database not initialized")

        if query is None:
            query = {}

        # Build query
        db_query = self.session.query(Element)

        # Apply filters
        for key, value in query.items():
            if key == "metadata":
                # Handle metadata exact matches
                for meta_key, meta_value in value.items():
                    # Use database-specific JSON extraction
                    if self.db_uri.startswith('postgresql'):
                        # PostgreSQL JSONB operator
                        json_filter = text(f"metadata_->>'{meta_key}' = :meta_value")
                        db_query = db_query.filter(json_filter.params(meta_value=str(meta_value)))
                    elif self.db_uri.startswith('sqlite'):
                        # SQLite JSON1 extension
                        json_filter = text(f"json_extract(metadata_, '$.{meta_key}') = :meta_value")
                        db_query = db_query.filter(json_filter.params(meta_value=str(meta_value)))
                    else:
                        # Fallback to simple text search
                        db_query = db_query.filter(Element.metadata_.like(f'%"{meta_key}"%"{meta_value}"%'))
            elif key == "metadata_like":
                # Handle metadata LIKE patterns
                for meta_key, meta_value in value.items():
                    if self.db_uri.startswith('postgresql'):
                        json_filter = text(f"metadata_->>'{meta_key}' LIKE :meta_value")
                        db_query = db_query.filter(json_filter.params(meta_value=str(meta_value)))
                    elif self.db_uri.startswith('sqlite'):
                        json_filter = text(f"json_extract(metadata_, '$.{meta_key}') LIKE :meta_value")
                        db_query = db_query.filter(json_filter.params(meta_value=str(meta_value)))
                    else:
                        # Fallback to simple text search
                        db_query = db_query.filter(Element.metadata_.like(f'%{meta_value}%'))
            elif key == "metadata_ilike":
                # Handle case-insensitive metadata LIKE patterns
                for meta_key, meta_value in value.items():
                    if self.db_uri.startswith('postgresql'):
                        json_filter = text(f"metadata_->>'{meta_key}' ILIKE :meta_value")
                        db_query = db_query.filter(json_filter.params(meta_value=str(meta_value)))
                    else:
                        # Fallback to case-insensitive LIKE
                        if self.db_uri.startswith('sqlite'):
                            json_filter = text(f"json_extract(metadata_, '$.{meta_key}') LIKE :meta_value")
                            db_query = db_query.filter(json_filter.params(meta_value=str(meta_value)))
                        else:
                            db_query = db_query.filter(Element.metadata_.like(f'%{meta_value}%'))
            elif key.endswith("_like"):
                # LIKE pattern for regular fields
                field_name = key[:-5]  # Remove '_like' suffix
                if hasattr(Element, field_name):
                    db_query = db_query.filter(getattr(Element, field_name).like(value))
            elif key.endswith("_ilike"):
                # Case-insensitive LIKE pattern
                field_name = key[:-6]  # Remove '_ilike' suffix
                if hasattr(Element, field_name):
                    if self.db_uri.startswith('postgresql'):
                        # PostgreSQL has native ILIKE support
                        db_query = db_query.filter(getattr(Element, field_name).op('ILIKE')(value))
                    else:
                        # Fallback to case-insensitive LIKE
                        db_query = db_query.filter(func.lower(getattr(Element, field_name)).like(func.lower(value)))
            elif key == "element_type":
                # Handle ElementType enums, strings, and lists
                type_values = self.prepare_element_type_query(value)
                if type_values:
                    if len(type_values) == 1:
                        db_query = db_query.filter(Element.element_type == type_values[0])
                    else:
                        db_query = db_query.filter(Element.element_type.in_(type_values))
            elif hasattr(Element, key):
                if isinstance(value, list):
                    # Handle list of values (IN condition)
                    db_query = db_query.filter(getattr(Element, key).in_(value))
                else:
                    # Simple equality
                    db_query = db_query.filter(getattr(Element, key) == value)

        # Apply limit
        db_query = db_query.limit(limit)

        # Execute query
        elements = db_query.all()

        # Convert to dictionaries
        result = []
        for element in elements:
            element_dict = {
                "element_id": element.element_id,
                "element_pk": element.element_pk,
                "doc_id": element.doc_id,
                "element_type": element.element_type,
                "parent_id": element.parent_id,
                "content_preview": element.content_preview,
                "content_location": element.content_location,
                "content_hash": element.content_hash
            }

            # Parse metadata JSON
            try:
                element_dict["metadata"] = json.loads(element.metadata_)
            except (json.JSONDecodeError, TypeError):
                element_dict["metadata"] = {}

            result.append(element_dict)

        return result

    def search_elements_by_content(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search elements by content preview."""
        if not self.session:
            raise ValueError("Database not initialized")

        # Use LIKE operator for text search
        elements = self.session.query(Element).filter(
            Element.content_preview.like(f"%{search_text}%")
        ).limit(limit).all()

        # Convert to dictionaries
        result = []
        for element in elements:
            element_dict = {
                "element_id": element.element_id,
                "element_pk": element.element_pk,
                "doc_id": element.doc_id,
                "element_type": element.element_type,
                "parent_id": element.parent_id,
                "content_preview": element.content_preview,
                "content_location": element.content_location,
                "content_hash": element.content_hash
            }

            # Parse metadata JSON
            try:
                element_dict["metadata"] = json.loads(element.metadata_)
            except (json.JSONDecodeError, TypeError):
                element_dict["metadata"] = {}

            result.append(element_dict)

        return result

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all associated elements and relationships."""
        if not self.session:
            raise ValueError("Database not initialized")

        # Check if document exists
        document = self.session.query(Document).filter_by(doc_id=doc_id).first()
        if not document:
            return False

        try:
            # Start transaction
            self.session.begin()

            # Delete the document (cascading delete will handle elements,
            # relationships, and embeddings due to our relationship configurations)
            self.session.delete(document)

            # Commit changes
            self.session.commit()

            logger.info(f"Deleted document {doc_id}")
            return True

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False

    # ========================================
    # ENHANCED EMBEDDING FUNCTIONS
    # ========================================

    def store_embedding(self, element_pk: Union[int, str], embedding: VectorType) -> None:
        """Store embedding for an element."""
        if not self.session:
            raise ValueError("Database not initialized")

        # Verify element exists
        element = self.session.query(Element).filter_by(element_pk=element_pk).first()
        if not element:
            raise ValueError(f"Element not found: {element_pk}")

        # Update vector dimension
        self._vector_dimension = max(self._vector_dimension, len(embedding))

        try:
            # Encode embedding as binary
            embedding_blob = self._encode_embedding(embedding)

            # Check if embedding already exists
            existing = self.session.query(Embedding).filter_by(element_pk=element_pk).first()

            if existing:
                # Update existing embedding
                existing.embedding = embedding_blob
                existing.dimensions = len(embedding)
                existing.topics = json.dumps([])  # Default to empty topics
                existing.confidence = 1.0  # Default confidence
                existing.created_at = time.time()
            else:
                # Create new embedding
                new_embedding = Embedding(
                    element_pk=element_pk,
                    embedding=embedding_blob,
                    dimensions=len(embedding),
                    topics=json.dumps([]),  # Default to empty topics
                    confidence=1.0,  # Default confidence
                    created_at=time.time()
                )
                self.session.add(new_embedding)

            # Commit changes
            self.session.commit()

            # Handle vector extension specific storage
            if self._vector_extension == "pgvector" and self.db_uri.startswith('postgresql'):
                self._store_pgvector_embedding(element_pk, embedding)

            logger.debug(f"Stored embedding for element {element_pk} with {len(embedding)} dimensions")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error storing embedding for {element_pk}: {str(e)}")
            raise

    def _store_pgvector_embedding(self, element_pk: Union[int, str], embedding: VectorType) -> None:
        """Store embedding using pgvector extension."""
        if not PGVECTOR_AVAILABLE:
            logger.warning("pgvector module not available, skipping vector storage")
            return

        try:
            # Convert embedding to string for pgvector
            embedding_str = json.dumps(embedding)

            # Execute raw SQL to update the vector column
            self.session.execute(text(
                f"UPDATE embeddings SET vector_embedding = :embedding::vector WHERE element_pk = :pk"
            ), {"embedding": embedding_str, "pk": element_pk})

            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error storing pgvector embedding: {str(e)}")

    def get_embedding(self, element_pk: Union[int, str]) -> Optional[VectorType]:
        """Get embedding for an element."""
        if not self.session:
            raise ValueError("Database not initialized")

        embedding_record = self.session.query(Embedding).filter_by(element_pk=element_pk).first()
        if not embedding_record:
            return None

        # Decode binary embedding
        return self._decode_embedding(embedding_record.embedding)

    def search_by_embedding(self, query_embedding: VectorType, limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by embedding similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_criteria: Optional dictionary with criteria to filter results
                            (e.g. {"element_type": ["header", "section"]})

        Returns:
            List of (element_pk, similarity_score) tuples for matching elements
        """
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            if self._vector_extension == "pgvector" and self.db_uri.startswith('postgresql') and PGVECTOR_AVAILABLE:
                return self._search_by_pgvector(query_embedding, limit, filter_criteria)
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

    def _search_by_pgvector(self, query_embedding: VectorType, limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Use pgvector for similarity search with filtering.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_criteria: Optional filtering criteria

        Returns:
            List of (element_pk, similarity_score) tuples
        """
        if not PGVECTOR_AVAILABLE:
            raise ImportError("pgvector module is required for pgvector search but not available")

        # Convert embedding to JSON string for pgvector
        embedding_json = json.dumps(query_embedding)

        try:
            # Start building SQL query
            sql_query = """
            SELECT e.element_pk, 1 - (em.vector_embedding <=> :query::vector) as similarity
            FROM embeddings em
            JOIN elements e ON e.element_pk = em.element_pk
            JOIN documents d ON e.doc_id = d.doc_id
            WHERE 1=1
            """

            params = {"query": embedding_json}

            # Apply filters if provided
            if filter_criteria:
                filter_conditions = []

                for key, value in filter_criteria.items():
                    # Handle element_type list
                    if key == "element_type" and isinstance(value, list):
                        placeholders = []
                        for i, elem_type in enumerate(value):
                            param_name = f"elem_type_{i}"
                            params[param_name] = elem_type
                            placeholders.append(f":elem_type_{i}")
                        if placeholders:
                            filter_conditions.append(f"e.element_type IN ({', '.join(placeholders)})")

                    # Handle document ID inclusion
                    elif key == "doc_id" and isinstance(value, list):
                        placeholders = []
                        for i, doc_id in enumerate(value):
                            param_name = f"doc_id_{i}"
                            params[param_name] = doc_id
                            placeholders.append(f":doc_id_{i}")
                        if placeholders:
                            filter_conditions.append(f"e.doc_id IN ({', '.join(placeholders)})")

                    # Handle document ID exclusion
                    elif key == "exclude_doc_id" and isinstance(value, list):
                        placeholders = []
                        for i, doc_id in enumerate(value):
                            param_name = f"exclude_doc_id_{i}"
                            params[param_name] = doc_id
                            placeholders.append(f":exclude_doc_id_{i}")
                        if placeholders:
                            filter_conditions.append(f"e.doc_id NOT IN ({', '.join(placeholders)})")

                    # Handle document source exclusion
                    elif key == "exclude_doc_source" and isinstance(value, list):
                        placeholders = []
                        for i, source in enumerate(value):
                            param_name = f"exclude_doc_source_{i}"
                            params[param_name] = source
                            placeholders.append(f":exclude_doc_source_{i}")
                        if placeholders:
                            filter_conditions.append(f"d.source NOT IN ({', '.join(placeholders)})")

                    # Handle single value equality
                    elif hasattr(Element, key):
                        params[key] = value
                        filter_conditions.append(f"e.{key} = :{key}")

                # Add filter conditions to SQL
                for condition in filter_conditions:
                    sql_query += f" AND {condition}"

            # Add ordering and limit
            sql_query += " ORDER BY em.vector_embedding <=> :query::vector LIMIT :limit"
            params["limit"] = limit

            # Execute query
            result = self.session.execute(text(sql_query), params)

            # Return element_pk instead of element_id for consistency
            return [(row[0], row[1]) for row in result]

        except Exception as e:
            logger.error(f"Error using pgvector search: {str(e)}")
            raise

    def _search_by_embedding_native(self, query_embedding: VectorType, limit: int = 10,
                                    filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Use native Python implementation for similarity search with filtering.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_criteria: Optional filtering criteria

        Returns:
            List of (element_pk, similarity_score) tuples
        """
        # Start building the query
        query = self.session.query(Embedding, Element.element_pk, Element.element_type, Element.doc_id, Document.source)
        query = query.join(Element, Embedding.element_pk == Element.element_pk)
        query = query.join(Document, Element.doc_id == Document.doc_id)

        # Apply filters if provided
        if filter_criteria:
            for key, value in filter_criteria.items():
                if key == "element_type" and isinstance(value, list):
                    # Handle list of element types
                    query = query.filter(Element.element_type.in_(value))
                elif key == "doc_id" and isinstance(value, list):
                    # Handle list of document IDs to include
                    query = query.filter(Element.doc_id.in_(value))
                elif key == "exclude_doc_id" and isinstance(value, list):
                    # Handle list of document IDs to exclude
                    query = query.filter(~Element.doc_id.in_(value))
                elif key == "exclude_doc_source" and isinstance(value, list):
                    # Handle list of document sources to exclude
                    query = query.filter(~Document.source.in_(value))
                elif hasattr(Element, key):
                    # Simple equality filter
                    query = query.filter(getattr(Element, key) == value)

        # Execute query to get filtered elements with embeddings
        records = query.all()

        # Calculate similarities
        similarities = []

        if NUMPY_AVAILABLE:
            # Use numpy for faster calculation
            return self._search_by_embedding_numpy(query_embedding, records, limit)
        else:
            # Use fallback implementation without numpy
            return self._search_by_embedding_fallback(query_embedding, records, limit)

    def _search_by_embedding_numpy(self, query_embedding: VectorType, records: List, limit: int = 10) -> List[
        Tuple[int, float]]:
        """Use NumPy for faster embedding similarity calculation."""
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for this method but not available")

        # Convert query embedding to numpy array
        query_np = np.array(query_embedding)

        similarities = []
        for record in records:
            embedding_record, element_pk, _, _, _ = record
            embedding = self._decode_embedding(embedding_record.embedding)

            if len(embedding) != len(query_embedding):
                # Skip if dimensions don't match
                continue

            embedding_np = np.array(embedding)
            similarity = self._cosine_similarity_numpy(query_np, embedding_np)
            # Return element_pk instead of element_id for consistency
            similarities.append((element_pk, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        return similarities[:limit]

    def _search_by_embedding_fallback(self, query_embedding: VectorType, records: List, limit: int = 10) -> List[
        Tuple[int, float]]:
        """Pure Python fallback for embedding similarity calculation."""
        similarities = []
        for record in records:
            embedding_record, element_pk, _, _, _ = record
            embedding = self._decode_embedding(embedding_record.embedding)

            if len(embedding) != len(query_embedding):
                # Skip if dimensions don't match
                continue

            similarity = self._cosine_similarity_fallback(query_embedding, embedding)
            # Return element_pk instead of element_id for consistency
            similarities.append((element_pk, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top results
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
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            # Import necessary modules conditionally
            try:
                # Try to get embedding generator
                if self.embedding_generator is None:
                    from ..embeddings import get_embedding_generator

                    # Get config from the connection parameters or load from path
                    config_obj = self.config
                    if not config_obj:
                        try:
                            from ..config import Config
                            config_obj = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
                        except Exception as e:
                            logger.warning(f"Error loading config: {str(e)}. Using default config.")
                            config_obj = Config()

                    # Get the embedding generator
                    self.embedding_generator = get_embedding_generator(config_obj)

                if self.embedding_generator is None:
                    raise ValueError("Could not initialize embedding generator")

                # Generate embedding for the search text
                query_embedding = self.embedding_generator.generate(search_text)

                # Use the embedding to search, passing the filter criteria
                return self.search_by_embedding(query_embedding, limit, filter_criteria)
            except ImportError as e:
                logger.error(f"Error importing embedding generator: {str(e)}")
                raise ValueError("Embedding generator not available - embedding libraries may not be installed")

        except Exception as e:
            logger.error(f"Error in semantic search by text: {str(e)}")
            # Return empty list on error
            return []

    # ========================================
    # ENHANCED SEARCH HELPER METHODS
    # ========================================

    @staticmethod
    def supports_like_patterns() -> bool:
        """
        Indicate whether this backend supports LIKE pattern matching.

        Returns:
            True - SQLAlchemy supports LIKE patterns across databases
        """
        return True

    @staticmethod
    def supports_case_insensitive_like() -> bool:
        """
        Indicate whether this backend supports case-insensitive LIKE (ILIKE).

        Returns:
            True - We can implement case-insensitive LIKE across databases
        """
        return True

    @staticmethod
    def supports_element_type_enums() -> bool:
        """
        Indicate whether this backend supports ElementType enum integration.

        Returns:
            True - SQLAlchemy implementation supports ElementType enums
        """
        return True

    @staticmethod
    def prepare_element_type_query(element_types: Union[
        ElementType,
        List[ElementType],
        str,
        List[str],
        None
    ]) -> Optional[List[str]]:
        """
        Prepare element type values for database queries.

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

        SQLAlchemy implementation supports case-insensitive LIKE across databases.

        Args:
            query: Query parameters with _ilike suffix support
            limit: Maximum number of results

        Returns:
            List of matching elements
        """
        # SQLAlchemy implementation handles case-insensitive LIKE natively
        return self.find_elements(query, limit)

    # ========================================
    # TOPIC SUPPORT METHODS
    # ========================================

    def supports_topics(self) -> bool:
        """
        Indicate whether this backend supports topic-aware embeddings.

        Returns:
            True since SQLAlchemy implementation now supports topics
        """
        return True

    def store_embedding_with_topics(self, element_pk: Union[int, str], embedding: VectorType,
                                    topics: List[str], confidence: float = 1.0) -> None:
        """
        Store embedding for an element with topic assignments.

        Args:
            element_pk: Element primary key
            embedding: Vector embedding
            topics: List of topic strings (e.g., ['security.policy', 'compliance'])
            confidence: Overall confidence in this embedding/topic assignment
        """
        if not self.session:
            raise ValueError("Database not initialized")

        # Verify element exists
        element = self.session.query(Element).filter_by(element_pk=element_pk).first()
        if not element:
            raise ValueError(f"Element not found: {element_pk}")

        # Update vector dimension
        self._vector_dimension = max(self._vector_dimension, len(embedding))

        try:
            # Encode embedding as binary
            embedding_blob = self._encode_embedding(embedding)

            # Check if embedding already exists
            existing = self.session.query(Embedding).filter_by(element_pk=element_pk).first()

            if existing:
                # Update existing embedding
                existing.embedding = embedding_blob
                existing.dimensions = len(embedding)
                existing.topics = json.dumps(topics)
                existing.confidence = confidence
                existing.created_at = time.time()
            else:
                # Create new embedding
                new_embedding = Embedding(
                    element_pk=element_pk,
                    embedding=embedding_blob,
                    dimensions=len(embedding),
                    topics=json.dumps(topics),
                    confidence=confidence,
                    created_at=time.time()
                )
                self.session.add(new_embedding)

            # Commit changes
            self.session.commit()

            # Handle vector extension specific storage
            if self._vector_extension == "pgvector" and self.db_uri.startswith('postgresql'):
                self._store_pgvector_embedding(element_pk, embedding)

            logger.debug(f"Stored embedding with topics for element {element_pk}")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error storing embedding with topics for {element_pk}: {str(e)}")
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
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            # Generate embedding for search text if provided
            query_embedding = None
            if search_text:
                if self.embedding_generator is None:
                    from ..embeddings import get_embedding_generator
                    config_obj = config or Config()
                    self.embedding_generator = get_embedding_generator(config_obj)

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
            # Start building query with confidence filtering
            query = self.session.query(Embedding).filter(Embedding.confidence >= min_confidence)

            # Execute query to get all embeddings above confidence threshold
            embeddings = query.all()

            # Process results in Python with topic filtering
            results = []
            for embedding_record in embeddings:
                element_pk = embedding_record.element_pk
                confidence = embedding_record.confidence

                # Parse topics
                topics = []
                if embedding_record.topics:
                    try:
                        topics = json.loads(embedding_record.topics)
                    except (json.JSONDecodeError, TypeError):
                        topics = []

                # Apply topic filtering
                if not self._matches_topic_filters(topics, include_topics, exclude_topics):
                    continue

                result_dict = {
                    'element_pk': element_pk,
                    'confidence': float(confidence),
                    'topics': topics
                }

                # Calculate similarity if we have a query embedding
                if query_embedding:
                    try:
                        embedding = self._decode_embedding(embedding_record.embedding)
                        if NUMPY_AVAILABLE:
                            similarity = self._cosine_similarity_numpy(
                                np.array(query_embedding), np.array(embedding)
                            )
                        else:
                            similarity = self._cosine_similarity_fallback(query_embedding, embedding)
                        result_dict['similarity'] = float(similarity)
                    except Exception as e:
                        logger.warning(f"Error calculating similarity for element {element_pk}: {str(e)}")
                        result_dict['similarity'] = 0.0
                else:
                    result_dict['similarity'] = 1.0  # No text search, all results have equal similarity

                results.append(result_dict)

            # Sort by similarity if we calculated it
            if query_embedding:
                results.sort(key=lambda x: x['similarity'], reverse=True)

            return results[:limit]

        except Exception as e:
            logger.error(f"Error in fallback topic search: {str(e)}")
            return []

    @staticmethod
    def _matches_topic_filters(topics: List[str],
                               include_topics: Optional[List[str]] = None,
                               exclude_topics: Optional[List[str]] = None) -> bool:
        """Check if topics match the include/exclude filters using pattern matching."""
        import fnmatch

        # Check include filters - at least one must match
        if include_topics:
            include_match = False
            for topic in topics:
                for pattern in include_topics:
                    if fnmatch.fnmatch(topic, pattern):
                        include_match = True
                        break
                if include_match:
                    break

            if not include_match:
                return False

        # Check exclude filters - none should match
        if exclude_topics:
            for topic in topics:
                for pattern in exclude_topics:
                    if fnmatch.fnmatch(topic, pattern):
                        return False

        return True

    def get_topic_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about topic distribution across embeddings.

        Returns:
            Dictionary mapping topic strings to statistics
        """
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            # Get all embeddings with topics
            embeddings = self.session.query(Embedding).filter(Embedding.topics.isnot(None)).all()

            topic_stats = {}
            for embedding_record in embeddings:
                confidence = embedding_record.confidence

                # Parse topics
                topics = []
                if embedding_record.topics:
                    try:
                        topics = json.loads(embedding_record.topics)
                    except (json.JSONDecodeError, TypeError):
                        topics = []

                # Get document ID for this element
                element = self.session.query(Element).filter_by(element_pk=embedding_record.element_pk).first()
                doc_id = element.doc_id if element else None

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

    def get_embedding_topics(self, element_pk: Union[int, str]) -> List[str]:
        """
        Get topics assigned to a specific embedding.

        Args:
            element_pk: Element primary key

        Returns:
            List of topic strings assigned to this embedding
        """
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            embedding_record = self.session.query(Embedding).filter_by(element_pk=element_pk).first()
            if not embedding_record or not embedding_record.topics:
                return []

            try:
                return json.loads(embedding_record.topics)
            except (json.JSONDecodeError, TypeError):
                return []

        except Exception as e:
            logger.error(f"Error getting topics for element {element_pk}: {str(e)}")
            return []

    # ========================================
    # HIERARCHY METHODS
    # ========================================

    def get_results_outline(self, elements: List[Tuple[int, float]]) -> List[ElementHierarchical]:
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

    # ========================================
    # UTILITY METHODS
    # ========================================

    @staticmethod
    def _encode_embedding(embedding: VectorType) -> bytes:
        """Encode embedding as binary blob."""
        if NUMPY_AVAILABLE:
            # Use numpy for efficient encoding
            return np.array(embedding, dtype=np.float32).tobytes()
        else:
            # Pure Python implementation using struct
            import struct
            # Pack each float into a binary string
            return b''.join(struct.pack('f', float(val)) for val in embedding)

    @staticmethod
    def _decode_embedding(blob: bytes) -> VectorType:
        """Decode embedding from binary blob."""
        if NUMPY_AVAILABLE:
            # Use numpy for efficient decoding
            return np.frombuffer(blob, dtype=np.float32).tolist()
        else:
            # Pure Python implementation using struct
            import struct
            # Calculate how many floats are in the blob (assuming 4 bytes per float)
            float_count = len(blob) // 4
            # Unpack the binary data into floats
            return list(struct.unpack(f'{float_count}f', blob))

    @staticmethod
    def _cosine_similarity_numpy(vec1: 'np.ndarray', vec2: 'np.ndarray') -> float:
        """Calculate cosine similarity between two vectors using numpy."""
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for this method but not available")

        # Calculate dot product
        dot_product = np.dot(vec1, vec2)

        # Calculate magnitudes
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def _cosine_similarity_fallback(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors without numpy."""
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Calculate magnitudes
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5

        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0

        # Calculate cosine similarity
        return float(dot_product / (mag1 * mag2))


if __name__ == "__main__":
    # Example demonstrating structured search with SQLAlchemy
    db_uri = 'sqlite:///test_doculyzer.db'

    db = SQLAlchemyDocumentDatabase(db_uri)
    db.initialize()

    # Show backend capabilities
    capabilities = db.get_backend_capabilities()
    print(f"SQLAlchemy supports {len(capabilities.supported)} capabilities:")
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
