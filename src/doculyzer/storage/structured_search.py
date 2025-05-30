"""
Core Structured Search System

This module provides a backend-agnostic structured search system that can work
with any document database backend. It defines the query language, data structures,
and building tools without any backend-specific implementation details.

Key Components:
- Search criteria data structures
- Logical operators and query composition
- Backend capability system
- Query builder with fluent interface
- Serialization support

Usage:
    from structured_search import SearchQueryBuilder, LogicalOperator

    query = (SearchQueryBuilder()
             .text_search("machine learning")
             .last_days(30)
             .topics(include=["ml%", "ai%"])
             .build())
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Literal, Set

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class LogicalOperator(Enum):
    """Logical operators for combining search criteria."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class DateRangeOperator(Enum):
    """Operators for date-based filtering."""
    WITHIN = "within"  # Between start and end dates
    BEFORE = "before"  # Before specified date
    AFTER = "after"  # After specified date
    EXACTLY = "exactly"  # Exactly on specified date
    RELATIVE_DAYS = "relative_days"  # Within last N days
    RELATIVE_MONTHS = "relative_months"  # Within last N months
    FISCAL_YEAR = "fiscal_year"  # Within fiscal year
    CALENDAR_YEAR = "calendar_year"  # Within calendar year
    QUARTER = "quarter"  # Within specific quarter


class SimilarityOperator(Enum):
    """Operators for similarity threshold comparisons."""
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    EQUALS = "="


class SearchCapability(Enum):
    """Enumeration of search capabilities that backends can support."""

    # Core search types
    TEXT_SIMILARITY = "text_similarity"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    VECTOR_SEARCH = "vector_search"
    FULL_TEXT_SEARCH = "full_text_search"

    # Date capabilities
    DATE_FILTERING = "date_filtering"
    DATE_RANGE_QUERIES = "date_range_queries"
    FISCAL_YEAR_DATES = "fiscal_year_dates"
    RELATIVE_DATES = "relative_dates"
    DATE_AGGREGATIONS = "date_aggregations"

    # Topic capabilities
    TOPIC_FILTERING = "topic_filtering"
    TOPIC_LIKE_PATTERNS = "topic_like_patterns"
    TOPIC_CONFIDENCE_FILTERING = "topic_confidence_filtering"

    # Metadata capabilities
    METADATA_EXACT = "metadata_exact"
    METADATA_LIKE = "metadata_like"
    METADATA_RANGE = "metadata_range"
    METADATA_EXISTS = "metadata_exists"
    NESTED_METADATA = "nested_metadata"

    # Element capabilities
    ELEMENT_TYPE_FILTERING = "element_type_filtering"
    ELEMENT_HIERARCHY = "element_hierarchy"
    ELEMENT_RELATIONSHIPS = "element_relationships"

    # Logical operations
    LOGICAL_AND = "logical_and"
    LOGICAL_OR = "logical_or"
    LOGICAL_NOT = "logical_not"
    NESTED_QUERIES = "nested_queries"

    # Scoring and ranking
    CUSTOM_SCORING = "custom_scoring"
    SIMILARITY_THRESHOLDS = "similarity_thresholds"
    BOOST_FACTORS = "boost_factors"
    SCORE_COMBINATION = "score_combination"

    # Advanced features
    FACETED_SEARCH = "faceted_search"
    AUTOCOMPLETE = "autocomplete"
    SPELL_CORRECTION = "spell_correction"
    RESULT_HIGHLIGHTING = "result_highlighting"


# ============================================================================
# BACKEND CAPABILITY SYSTEM
# ============================================================================

class BackendCapabilities:
    """Represents the capabilities of a specific backend."""

    def __init__(self, supported_capabilities: Set[SearchCapability]):
        self.supported = supported_capabilities

    def supports(self, capability: SearchCapability) -> bool:
        """Check if backend supports a specific capability."""
        return capability in self.supported

    def supports_all(self, capabilities: List[SearchCapability]) -> bool:
        """Check if backend supports all specified capabilities."""
        return all(cap in self.supported for cap in capabilities)

    def supports_any(self, capabilities: List[SearchCapability]) -> bool:
        """Check if backend supports any of the specified capabilities."""
        return any(cap in self.supported for cap in capabilities)

    def missing_capabilities(self, required: List[SearchCapability]) -> List[SearchCapability]:
        """Return list of missing capabilities from required set."""
        return [cap for cap in required if cap not in self.supported]

    def get_supported_list(self) -> List[str]:
        """Get list of supported capability names."""
        return [cap.value for cap in self.supported]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'supported_capabilities': self.get_supported_list(),
            'total_count': len(self.supported)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackendCapabilities':
        """Create from dictionary."""
        supported = {SearchCapability(cap) for cap in data['supported_capabilities']}
        return cls(supported)


class UnsupportedSearchError(Exception):
    """Raised when a backend cannot execute a requested search."""

    def __init__(self, missing_capabilities: List[SearchCapability]):
        self.missing_capabilities = missing_capabilities
        cap_names = [cap.value for cap in missing_capabilities]
        super().__init__(f"Backend does not support: {', '.join(cap_names)}")


# ============================================================================
# SEARCH CRITERIA DATA STRUCTURES
# ============================================================================

@dataclass
class TextSearchCriteria:
    """Criteria for semantic text search using embeddings."""
    query_text: str
    similarity_threshold: float = 0.7
    similarity_operator: SimilarityOperator = SimilarityOperator.GREATER_EQUAL
    boost_factor: float = 1.0
    search_fields: List[str] = field(default_factory=list)  # Specific fields to search

    def __post_init__(self):
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        if not self.query_text.strip():
            raise ValueError("Query text cannot be empty")


@dataclass
class EmbeddingSearchCriteria:
    """Criteria for direct embedding vector search (ANN)."""
    embedding_vector: List[float]
    similarity_threshold: float = 0.7
    similarity_operator: SimilarityOperator = SimilarityOperator.GREATER_EQUAL
    distance_metric: Literal["cosine", "euclidean", "dot_product"] = "cosine"
    boost_factor: float = 1.0

    def __post_init__(self):
        if not self.embedding_vector:
            raise ValueError("Embedding vector cannot be empty")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")


@dataclass
class DateSearchCriteria:
    """Criteria for date-based filtering with multiple operators."""
    operator: DateRangeOperator

    # For absolute date ranges
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    exact_date: Optional[datetime] = None

    # For relative dates
    relative_value: Optional[int] = None  # N days/months back

    # For fiscal/calendar periods
    year: Optional[int] = None
    quarter: Optional[int] = None  # 1-4

    # Additional constraints
    include_partial_dates: bool = True
    specificity_levels: List[str] = field(default_factory=lambda: [
        "full", "date_only", "month_only", "quarter_only", "year_only"
    ])

    def __post_init__(self):
        self._validate_operator_requirements()

    def _validate_operator_requirements(self):
        """Validate that required fields are provided for the operator."""
        if self.operator == DateRangeOperator.WITHIN:
            if not (self.start_date and self.end_date):
                raise ValueError("WITHIN operator requires both start_date and end_date")
        elif self.operator in [DateRangeOperator.BEFORE, DateRangeOperator.AFTER, DateRangeOperator.EXACTLY]:
            if not self.exact_date:
                raise ValueError(f"{self.operator.value} operator requires exact_date")
        elif self.operator in [DateRangeOperator.RELATIVE_DAYS, DateRangeOperator.RELATIVE_MONTHS]:
            if not self.relative_value or self.relative_value <= 0:
                raise ValueError(f"{self.operator.value} operator requires positive relative_value")
        elif self.operator == DateRangeOperator.QUARTER:
            if not (self.year and self.quarter):
                raise ValueError("QUARTER operator requires both year and quarter")
            if not 1 <= self.quarter <= 4:
                raise ValueError("Quarter must be between 1 and 4")
        elif self.operator in [DateRangeOperator.FISCAL_YEAR, DateRangeOperator.CALENDAR_YEAR]:
            if not self.year:
                raise ValueError(f"{self.operator.value} operator requires year")


@dataclass
class TopicSearchCriteria:
    """Criteria for topic-based filtering with LIKE patterns."""
    include_topics: List[str] = field(default_factory=list)
    exclude_topics: List[str] = field(default_factory=list)
    require_all_included: bool = False  # AND vs OR for include_topics
    min_confidence: float = 0.7
    boost_factor: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("Min confidence must be between 0.0 and 1.0")
        if not self.include_topics and not self.exclude_topics:
            raise ValueError("Must specify at least one topic to include or exclude")


@dataclass
class MetadataSearchCriteria:
    """Criteria for metadata-based filtering."""
    exact_matches: Dict[str, Any] = field(default_factory=dict)
    like_patterns: Dict[str, str] = field(default_factory=dict)
    range_filters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    exists_filters: List[str] = field(default_factory=list)

    # Example range_filters:
    # {"created_at": {"gte": timestamp, "lte": timestamp}}
    # {"word_count": {"gt": 100, "lt": 5000}}

    def __post_init__(self):
        if not any([self.exact_matches, self.like_patterns, self.range_filters, self.exists_filters]):
            raise ValueError("Must specify at least one metadata filter")


@dataclass
class ElementSearchCriteria:
    """Criteria for element-specific filtering."""
    element_types: List[str] = field(default_factory=list)
    doc_ids: List[str] = field(default_factory=list)
    exclude_doc_ids: List[str] = field(default_factory=list)
    doc_sources: List[str] = field(default_factory=list)
    parent_element_ids: List[str] = field(default_factory=list)
    content_length_min: Optional[int] = None
    content_length_max: Optional[int] = None

    def __post_init__(self):
        if self.content_length_min is not None and self.content_length_min < 0:
            raise ValueError("Minimum content length cannot be negative")
        if self.content_length_max is not None and self.content_length_max < 0:
            raise ValueError("Maximum content length cannot be negative")
        if (self.content_length_min is not None and
                self.content_length_max is not None and
                self.content_length_min > self.content_length_max):
            raise ValueError("Minimum content length cannot be greater than maximum")


# ============================================================================
# QUERY COMPOSITION
# ============================================================================

@dataclass
class SearchCriteriaGroup:
    """A group of search criteria with a logical operator."""
    operator: LogicalOperator = LogicalOperator.AND

    # Core search criteria
    text_criteria: Optional[TextSearchCriteria] = None
    embedding_criteria: Optional[EmbeddingSearchCriteria] = None
    date_criteria: Optional[DateSearchCriteria] = None
    topic_criteria: Optional[TopicSearchCriteria] = None
    metadata_criteria: Optional[MetadataSearchCriteria] = None
    element_criteria: Optional[ElementSearchCriteria] = None

    # Nested groups for complex logic
    sub_groups: List['SearchCriteriaGroup'] = field(default_factory=list)

    def __post_init__(self):
        self._validate_group()

    def _validate_group(self):
        """Validate that the group has at least one criterion."""
        criteria_count = sum([
            self.text_criteria is not None,
            self.embedding_criteria is not None,
            self.date_criteria is not None,
            self.topic_criteria is not None,
            self.metadata_criteria is not None,
            self.element_criteria is not None,
            len(self.sub_groups) > 0
        ])

        if criteria_count == 0:
            raise ValueError("SearchCriteriaGroup must have at least one criterion or sub-group")

    def get_required_capabilities(self) -> Set[SearchCapability]:
        """Get the capabilities required by this criteria group."""
        required = set()

        # Operator capabilities
        if self.operator == LogicalOperator.AND:
            required.add(SearchCapability.LOGICAL_AND)
        elif self.operator == LogicalOperator.OR:
            required.add(SearchCapability.LOGICAL_OR)
        elif self.operator == LogicalOperator.NOT:
            required.add(SearchCapability.LOGICAL_NOT)

        # Individual criteria capabilities
        if self.text_criteria:
            required.add(SearchCapability.TEXT_SIMILARITY)
            if self.text_criteria.similarity_operator != SimilarityOperator.GREATER_EQUAL:
                required.add(SearchCapability.SIMILARITY_THRESHOLDS)
            if self.text_criteria.boost_factor != 1.0:
                required.add(SearchCapability.BOOST_FACTORS)

        if self.embedding_criteria:
            required.add(SearchCapability.EMBEDDING_SIMILARITY)
            required.add(SearchCapability.VECTOR_SEARCH)
            if self.embedding_criteria.similarity_operator != SimilarityOperator.GREATER_EQUAL:
                required.add(SearchCapability.SIMILARITY_THRESHOLDS)

        if self.date_criteria:
            required.add(SearchCapability.DATE_FILTERING)
            if self.date_criteria.operator in [DateRangeOperator.WITHIN, DateRangeOperator.BEFORE,
                                               DateRangeOperator.AFTER]:
                required.add(SearchCapability.DATE_RANGE_QUERIES)
            if self.date_criteria.operator == DateRangeOperator.FISCAL_YEAR:
                required.add(SearchCapability.FISCAL_YEAR_DATES)
            if self.date_criteria.operator in [DateRangeOperator.RELATIVE_DAYS, DateRangeOperator.RELATIVE_MONTHS]:
                required.add(SearchCapability.RELATIVE_DATES)

        if self.topic_criteria:
            required.add(SearchCapability.TOPIC_FILTERING)
            if self.topic_criteria.include_topics or self.topic_criteria.exclude_topics:
                required.add(SearchCapability.TOPIC_LIKE_PATTERNS)
            if self.topic_criteria.min_confidence != 0.7:
                required.add(SearchCapability.TOPIC_CONFIDENCE_FILTERING)

        if self.metadata_criteria:
            if self.metadata_criteria.exact_matches:
                required.add(SearchCapability.METADATA_EXACT)
            if self.metadata_criteria.like_patterns:
                required.add(SearchCapability.METADATA_LIKE)
            if self.metadata_criteria.range_filters:
                required.add(SearchCapability.METADATA_RANGE)
            if self.metadata_criteria.exists_filters:
                required.add(SearchCapability.METADATA_EXISTS)

        if self.element_criteria:
            if self.element_criteria.element_types:
                required.add(SearchCapability.ELEMENT_TYPE_FILTERING)

        # Sub-group capabilities
        if self.sub_groups:
            required.add(SearchCapability.NESTED_QUERIES)
            for sub_group in self.sub_groups:
                required.update(sub_group.get_required_capabilities())

        return required


@dataclass
class StructuredSearchQuery:
    """Top-level structured search query."""
    criteria_group: SearchCriteriaGroup

    # Result configuration
    limit: int = 10
    offset: int = 0
    include_element_dates: bool = False
    include_metadata: bool = True
    include_topics: bool = False
    include_similarity_scores: bool = True
    include_highlighting: bool = False

    # Scoring and ranking
    score_combination: Literal["multiply", "add", "max", "weighted_avg"] = "weighted_avg"
    custom_weights: Dict[str, float] = field(default_factory=lambda: {
        "text_similarity": 1.0,
        "embedding_similarity": 1.0,
        "topic_confidence": 0.5,
        "date_relevance": 0.3
    })

    # Query metadata
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self._validate_query()

    def _validate_query(self):
        """Validate query parameters."""
        if self.limit <= 0:
            raise ValueError("Limit must be positive")
        if self.offset < 0:
            raise ValueError("Offset cannot be negative")
        if self.score_combination not in ["multiply", "add", "max", "weighted_avg"]:
            raise ValueError("Invalid score combination method")

    def get_required_capabilities(self) -> List[SearchCapability]:
        """Get all capabilities required by this query."""
        required = set(self.criteria_group.get_required_capabilities())

        # Add scoring capabilities if needed
        if self.score_combination != "weighted_avg" or any(w != 1.0 for w in self.custom_weights.values()):
            required.add(SearchCapability.CUSTOM_SCORING)

        if self.include_highlighting:
            required.add(SearchCapability.RESULT_HIGHLIGHTING)

        return list(required)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "created_at": self.created_at.isoformat(),
            "criteria_group": self._criteria_group_to_dict(self.criteria_group),
            "limit": self.limit,
            "offset": self.offset,
            "include_element_dates": self.include_element_dates,
            "include_metadata": self.include_metadata,
            "include_topics": self.include_topics,
            "include_similarity_scores": self.include_similarity_scores,
            "include_highlighting": self.include_highlighting,
            "score_combination": self.score_combination,
            "custom_weights": self.custom_weights
        }

    def _criteria_group_to_dict(self, group: SearchCriteriaGroup) -> Dict[str, Any]:
        """Convert criteria group to dictionary recursively."""
        result = {"operator": group.operator.value}

        if group.text_criteria:
            result["text_criteria"] = {
                "query_text": group.text_criteria.query_text,
                "similarity_threshold": group.text_criteria.similarity_threshold,
                "similarity_operator": group.text_criteria.similarity_operator.value,
                "boost_factor": group.text_criteria.boost_factor,
                "search_fields": group.text_criteria.search_fields
            }

        if group.embedding_criteria:
            result["embedding_criteria"] = {
                "embedding_vector": group.embedding_criteria.embedding_vector,
                "similarity_threshold": group.embedding_criteria.similarity_threshold,
                "similarity_operator": group.embedding_criteria.similarity_operator.value,
                "distance_metric": group.embedding_criteria.distance_metric,
                "boost_factor": group.embedding_criteria.boost_factor
            }

        if group.date_criteria:
            result["date_criteria"] = {
                "operator": group.date_criteria.operator.value,
                "start_date": group.date_criteria.start_date.isoformat() if group.date_criteria.start_date else None,
                "end_date": group.date_criteria.end_date.isoformat() if group.date_criteria.end_date else None,
                "exact_date": group.date_criteria.exact_date.isoformat() if group.date_criteria.exact_date else None,
                "relative_value": group.date_criteria.relative_value,
                "year": group.date_criteria.year,
                "quarter": group.date_criteria.quarter,
                "include_partial_dates": group.date_criteria.include_partial_dates,
                "specificity_levels": group.date_criteria.specificity_levels
            }

        if group.topic_criteria:
            result["topic_criteria"] = {
                "include_topics": group.topic_criteria.include_topics,
                "exclude_topics": group.topic_criteria.exclude_topics,
                "require_all_included": group.topic_criteria.require_all_included,
                "min_confidence": group.topic_criteria.min_confidence,
                "boost_factor": group.topic_criteria.boost_factor
            }

        if group.metadata_criteria:
            result["metadata_criteria"] = {
                "exact_matches": group.metadata_criteria.exact_matches,
                "like_patterns": group.metadata_criteria.like_patterns,
                "range_filters": group.metadata_criteria.range_filters,
                "exists_filters": group.metadata_criteria.exists_filters
            }

        if group.element_criteria:
            result["element_criteria"] = {
                "element_types": group.element_criteria.element_types,
                "doc_ids": group.element_criteria.doc_ids,
                "exclude_doc_ids": group.element_criteria.exclude_doc_ids,
                "doc_sources": group.element_criteria.doc_sources,
                "parent_element_ids": group.element_criteria.parent_element_ids,
                "content_length_min": group.element_criteria.content_length_min,
                "content_length_max": group.element_criteria.content_length_max
            }

        if group.sub_groups:
            result["sub_groups"] = [self._criteria_group_to_dict(sg) for sg in group.sub_groups]

        return result


# ============================================================================
# FLUENT QUERY BUILDER
# ============================================================================

class SearchQueryBuilder:
    """Builder for constructing complex search queries fluently."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the builder to start a new query."""
        self._root_group = SearchCriteriaGroup()
        self._current_group = self._root_group
        self._group_stack = []
        self._query_config = {
            "limit": 10,
            "offset": 0,
            "include_element_dates": False,
            "include_metadata": True,
            "include_topics": False,
            "include_similarity_scores": True,
            "include_highlighting": False,
            "score_combination": "weighted_avg",
            "custom_weights": {
                "text_similarity": 1.0,
                "embedding_similarity": 1.0,
                "topic_confidence": 0.5,
                "date_relevance": 0.3
            }
        }
        return self

    # Logical operators
    def with_operator(self, operator: LogicalOperator):
        """Set the logical operator for the current group."""
        self._current_group.operator = operator
        return self

    # Text search
    def text_search(self, query_text: str, similarity_threshold: float = 0.7,
                    boost_factor: float = 1.0, search_fields: List[str] = None):
        """Add text search criteria."""
        self._current_group.text_criteria = TextSearchCriteria(
            query_text=query_text,
            similarity_threshold=similarity_threshold,
            boost_factor=boost_factor,
            search_fields=search_fields or []
        )
        return self

    # Embedding search
    def embedding_search(self, embedding_vector: List[float],
                         similarity_threshold: float = 0.7,
                         distance_metric: str = "cosine",
                         boost_factor: float = 1.0):
        """Add embedding search criteria."""
        self._current_group.embedding_criteria = EmbeddingSearchCriteria(
            embedding_vector=embedding_vector,
            similarity_threshold=similarity_threshold,
            distance_metric=distance_metric,
            boost_factor=boost_factor
        )
        return self

    # Date filters
    def date_range(self, start_date: datetime, end_date: datetime,
                   include_partial: bool = True):
        """Add date range criteria."""
        self._current_group.date_criteria = DateSearchCriteria(
            operator=DateRangeOperator.WITHIN,
            start_date=start_date,
            end_date=end_date,
            include_partial_dates=include_partial
        )
        return self

    def date_after(self, date: datetime):
        """Add date after criteria."""
        self._current_group.date_criteria = DateSearchCriteria(
            operator=DateRangeOperator.AFTER,
            exact_date=date
        )
        return self

    def date_before(self, date: datetime):
        """Add date before criteria."""
        self._current_group.date_criteria = DateSearchCriteria(
            operator=DateRangeOperator.BEFORE,
            exact_date=date
        )
        return self

    def last_days(self, days: int):
        """Add relative date criteria for last N days."""
        self._current_group.date_criteria = DateSearchCriteria(
            operator=DateRangeOperator.RELATIVE_DAYS,
            relative_value=days
        )
        return self

    def last_months(self, months: int):
        """Add relative date criteria for last N months."""
        self._current_group.date_criteria = DateSearchCriteria(
            operator=DateRangeOperator.RELATIVE_MONTHS,
            relative_value=months
        )
        return self

    def fiscal_year(self, year: int):
        """Add fiscal year criteria."""
        self._current_group.date_criteria = DateSearchCriteria(
            operator=DateRangeOperator.FISCAL_YEAR,
            year=year
        )
        return self

    def calendar_year(self, year: int):
        """Add calendar year criteria."""
        self._current_group.date_criteria = DateSearchCriteria(
            operator=DateRangeOperator.CALENDAR_YEAR,
            year=year
        )
        return self

    def quarter(self, year: int, quarter: int):
        """Add quarter criteria."""
        self._current_group.date_criteria = DateSearchCriteria(
            operator=DateRangeOperator.QUARTER,
            year=year,
            quarter=quarter
        )
        return self

    # Topic filters
    def topics(self, include: List[str] = None, exclude: List[str] = None,
               require_all: bool = False, min_confidence: float = 0.7,
               boost_factor: float = 1.0):
        """Add topic criteria."""
        self._current_group.topic_criteria = TopicSearchCriteria(
            include_topics=include or [],
            exclude_topics=exclude or [],
            require_all_included=require_all,
            min_confidence=min_confidence,
            boost_factor=boost_factor
        )
        return self

    # Element filters
    def element_types(self, types: List[str]):
        """Add element type filtering."""
        if not self._current_group.element_criteria:
            self._current_group.element_criteria = ElementSearchCriteria()
        self._current_group.element_criteria.element_types = types
        return self

    def doc_ids(self, doc_ids: List[str]):
        """Add document ID filtering."""
        if not self._current_group.element_criteria:
            self._current_group.element_criteria = ElementSearchCriteria()
        self._current_group.element_criteria.doc_ids = doc_ids
        return self

    def exclude_doc_ids(self, doc_ids: List[str]):
        """Add document ID exclusion."""
        if not self._current_group.element_criteria:
            self._current_group.element_criteria = ElementSearchCriteria()
        self._current_group.element_criteria.exclude_doc_ids = doc_ids
        return self

    def content_length(self, min_length: int = None, max_length: int = None):
        """Add content length filtering."""
        if not self._current_group.element_criteria:
            self._current_group.element_criteria = ElementSearchCriteria()
        if min_length is not None:
            self._current_group.element_criteria.content_length_min = min_length
        if max_length is not None:
            self._current_group.element_criteria.content_length_max = max_length
        return self

    # Metadata filters
    def metadata_exact(self, **kwargs):
        """Add exact metadata matches."""
        if not self._current_group.metadata_criteria:
            self._current_group.metadata_criteria = MetadataSearchCriteria()
        self._current_group.metadata_criteria.exact_matches.update(kwargs)
        return self

    def metadata_like(self, **kwargs):
        """Add metadata LIKE patterns."""
        if not self._current_group.metadata_criteria:
            self._current_group.metadata_criteria = MetadataSearchCriteria()
        self._current_group.metadata_criteria.like_patterns.update(kwargs)
        return self

    def metadata_range(self, field: str, gte: Any = None, lte: Any = None,
                       gt: Any = None, lt: Any = None):
        """Add metadata range filter."""
        if not self._current_group.metadata_criteria:
            self._current_group.metadata_criteria = MetadataSearchCriteria()

        range_filter = {}
        if gte is not None:
            range_filter["gte"] = gte
        if lte is not None:
            range_filter["lte"] = lte
        if gt is not None:
            range_filter["gt"] = gt
        if lt is not None:
            range_filter["lt"] = lt

        self._current_group.metadata_criteria.range_filters[field] = range_filter
        return self

    def metadata_exists(self, fields: List[str]):
        """Add metadata existence filters."""
        if not self._current_group.metadata_criteria:
            self._current_group.metadata_criteria = MetadataSearchCriteria()
        self._current_group.metadata_criteria.exists_filters.extend(fields)
        return self

    # Group management
    def begin_group(self, operator: LogicalOperator = LogicalOperator.AND):
        """Begin a new nested group."""
        new_group = SearchCriteriaGroup(operator=operator)
        self._current_group.sub_groups.append(new_group)
        self._group_stack.append(self._current_group)
        self._current_group = new_group
        return self

    def end_group(self):
        """End the current nested group."""
        if self._group_stack:
            self._current_group = self._group_stack.pop()
        return self

    # Result configuration
    def limit(self, limit: int):
        """Set result limit."""
        self._query_config["limit"] = limit
        return self

    def offset(self, offset: int):
        """Set result offset."""
        self._query_config["offset"] = offset
        return self

    def include_dates(self, include: bool = True):
        """Include extracted dates in results."""
        self._query_config["include_element_dates"] = include
        return self

    def include_topics_in_results(self, include: bool = True):
        """Include topics in results."""
        self._query_config["include_topics"] = include
        return self

    def include_highlighting(self, include: bool = True):
        """Include result highlighting."""
        self._query_config["include_highlighting"] = include
        return self

    def score_combination(self, method: str, weights: Dict[str, float] = None):
        """Set score combination method and weights."""
        self._query_config["score_combination"] = method
        if weights:
            self._query_config["custom_weights"].update(weights)
        return self

    def build(self) -> StructuredSearchQuery:
        """Build the final structured search query."""
        return StructuredSearchQuery(
            criteria_group=self._root_group,
            **self._query_config
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_query_capabilities(query: StructuredSearchQuery,
                                backend_capabilities: BackendCapabilities) -> List[SearchCapability]:
    """
    Validate that a backend can execute a given query.

    Args:
        query: The structured search query to validate
        backend_capabilities: The capabilities of the target backend

    Returns:
        List of missing capabilities (empty if fully supported)
    """
    required = query.get_required_capabilities()
    return backend_capabilities.missing_capabilities(required)


def get_common_query_patterns() -> Dict[str, StructuredSearchQuery]:
    """Get a dictionary of common query patterns for reference."""

    patterns = {"simple_text": (SearchQueryBuilder()
                                .text_search("example query")
                                .limit(10)
                                .build()), "recent_content": (SearchQueryBuilder()
                                                              .text_search("important updates")
                                                              .last_days(7)
                                                              .element_types(["header", "paragraph"])
                                                              .build()), "topic_search": (SearchQueryBuilder()
                                                                                          .topics(
        include=["technology%", "innovation%"])
                                                                                          .last_months(3)
                                                                                          .build()),
                "complex_logic": (SearchQueryBuilder()
                                  .with_operator(LogicalOperator.AND)
                                  .begin_group(LogicalOperator.OR)
                                  .text_search("artificial intelligence")
                                  .topics(include=["ai%", "ml%"])
                                  .end_group()
                                  .begin_group(LogicalOperator.NOT)
                                  .topics(include=["deprecated%"])
                                  .end_group()
                                  .build())}

    # Simple text search

    # Recent content search

    # Topic-based search

    # Complex logical search

    return patterns


# ============================================================================
# EXAMPLES AND DOCUMENTATION
# ============================================================================

def demonstrate_query_building():
    """Demonstrate various query building patterns."""

    print("=== Structured Search Query Examples ===\n")

    # Example 1: Simple text search
    print("1. Simple text search:")
    query1 = (SearchQueryBuilder()
              .text_search("machine learning algorithms", similarity_threshold=0.8)
              .limit(20)
              .build())
    print(f"   Query ID: {query1.query_id}")
    print(f"   Required capabilities: {[c.value for c in query1.get_required_capabilities()]}")
    print()

    # Example 2: Date-filtered search
    print("2. Date-filtered search:")
    query2 = (SearchQueryBuilder()
              .text_search("quarterly reports")
              .quarter(2024, 3)
              .element_types(["header", "paragraph"])
              .include_dates(True)
              .build())
    print(f"   Query ID: {query2.query_id}")
    print()

    # Example 3: Complex nested logic
    print("3. Complex nested logic:")
    query3 = (SearchQueryBuilder()
              .with_operator(LogicalOperator.AND)
              .begin_group(LogicalOperator.OR)
              .text_search("security policy")
              .topics(include=["security%", "policy%"])
              .end_group()
              .begin_group(LogicalOperator.NOT)
              .metadata_like(status="%draft%")
              .end_group()
              .last_days(30)
              .build())
    print(f"   Query ID: {query3.query_id}")
    print(f"   Required capabilities: {len(query3.get_required_capabilities())} total")
    print()

    # Example 4: Metadata and range filters
    print("4. Metadata and range filters:")
    query4 = (SearchQueryBuilder()
              .metadata_exact(department="engineering", priority="high")
              .metadata_range("word_count", gte=100, lte=5000)
              .content_length(min_length=50)
              .build())
    print(f"   Query ID: {query4.query_id}")
    print()


if __name__ == "__main__":
    demonstrate_query_building()
