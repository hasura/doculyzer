"""
Pydantic Structured Search Query System (Pydantic v2) - FIXED VERSION

This module provides Pydantic models for structured search queries with automatic
validation, serialization, and JSON schema generation. It bridges the gap between
the core structured search system and API/serialization requirements.

SEARCH TYPE NAMING:
- SemanticSearchRequest: Natural language text -> embedding -> vector similarity (ANN)
- VectorSearchRequest: Direct pre-computed vector -> vector similarity (ANN)
- TopicSearchRequest: Topic/category pattern matching (LIKE patterns)
- MetadataSearchRequest: Structured field matching (exact, range, exists)
- DateSearchRequest: Temporal filtering on extracted dates
- ElementSearchRequest: Document structure filtering (types, length, etc.)

Note: The underlying core system uses 'text_criteria' and 'embedding_criteria' field names
for compatibility, but the Pydantic API uses more accurate naming for the search types.

Key Features:
- Pydantic v2 models with automatic validation
- JSON serialization/deserialization
- OpenAPI schema generation for APIs
- Conversion utilities between dataclasses and Pydantic models
- Search execution function

Usage:
    from pydantic_search import SearchQueryRequest, execute_search

    query = SearchQueryRequest(
        criteria_group=SearchCriteriaGroupRequest(
            semantic_search=SemanticSearchRequest(
                query_text="machine learning",
                similarity_threshold=0.8
            ),
            date_search=DateSearchRequest(
                operator="relative_days",
                relative_value=30
            )
        ),
        limit=20
    )

    results = execute_search(query, database)
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, ValidationError
from typing import Optional, List, Union, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)

# Import the core structured search components for conversion
# Note: These imports should be updated to match your actual module structure
try:
    from .structured_search import (
        StructuredSearchQuery as CoreStructuredSearchQuery,
        SearchCriteriaGroup as CoreSearchCriteriaGroup,
        TextSearchCriteria as CoreTextSearchCriteria,
        EmbeddingSearchCriteria as CoreEmbeddingSearchCriteria,
        DateSearchCriteria as CoreDateSearchCriteria,
        TopicSearchCriteria as CoreTopicSearchCriteria,
        MetadataSearchCriteria as CoreMetadataSearchCriteria,
        ElementSearchCriteria as CoreElementSearchCriteria,
        LogicalOperator,
        DateRangeOperator,
        SimilarityOperator
    )
except ImportError:
    # Fallback for when core modules are not available
    CoreStructuredSearchQuery = None
    CoreSearchCriteriaGroup = None
    CoreTextSearchCriteria = None
    CoreEmbeddingSearchCriteria = None
    CoreDateSearchCriteria = None
    CoreTopicSearchCriteria = None
    CoreMetadataSearchCriteria = None
    CoreElementSearchCriteria = None
    LogicalOperator = None
    DateRangeOperator = None
    SimilarityOperator = None


# ============================================================================
# PYDANTIC ENUMS (reuse core enums but make them serializable)
# ============================================================================

class LogicalOperatorEnum(str, Enum):
    """Logical operators for combining search criteria in complex queries."""
    AND = "AND"  # All criteria must match (intersection)
    OR = "OR"    # Any criteria can match (union)
    NOT = "NOT"  # Exclude matching criteria (negation)


class DateRangeOperatorEnum(str, Enum):
    """Date filtering operators supporting various temporal query patterns."""
    WITHIN = "within"              # Between two specific dates (inclusive range)
    BEFORE = "before"              # Earlier than specified date (exclusive)
    AFTER = "after"                # Later than specified date (exclusive)
    EXACTLY = "exactly"            # Exact date match (precise)
    RELATIVE_DAYS = "relative_days"      # Within last N days from now
    RELATIVE_MONTHS = "relative_months"  # Within last N months from now
    FISCAL_YEAR = "fiscal_year"          # Within organization's fiscal year
    CALENDAR_YEAR = "calendar_year"      # Within standard calendar year
    QUARTER = "quarter"                  # Within specific quarter (Q1-Q4)


class SimilarityOperatorEnum(str, Enum):
    """Comparison operators for semantic similarity thresholds in vector searches."""
    GREATER_THAN = ">"   # Similarity must exceed threshold
    GREATER_EQUAL = ">=" # Similarity must meet or exceed threshold (default)
    LESS_THAN = "<"      # Similarity must be below threshold
    LESS_EQUAL = "<="    # Similarity must be at or below threshold
    EQUALS = "="         # Similarity must exactly match threshold


class ScoreCombinationEnum(str, Enum):
    """Methods for combining multiple relevance scores into final ranking."""
    MULTIPLY = "multiply"       # Multiplicative combination (penalizes low scores)
    ADD = "add"                # Additive combination (simple sum)
    MAX = "max"                # Take the highest individual score
    WEIGHTED_AVG = "weighted_avg"  # Weighted average (balanced, default)


# ============================================================================
# PYDANTIC SEARCH CRITERIA MODELS
# ============================================================================

class SemanticSearchRequest(BaseModel):
    """
    Semantic text search using natural language queries and vector embeddings.

    This search type converts your query text into embeddings and finds semantically
    similar content, even when exact keywords don't match. Perfect for conceptual
    searches like "budget planning strategies" matching "financial planning methods".
    """
    model_config = ConfigDict(
        title="Semantic Text Search Configuration",
        json_schema_extra={
            "examples": [
                {
                    "query_text": "machine learning algorithms",
                    "similarity_threshold": 0.8,
                    "similarity_operator": ">=",
                    "boost_factor": 1.5,
                    "search_fields": []
                },
                {
                    "query_text": "quarterly financial performance analysis",
                    "similarity_threshold": 0.75,
                    "similarity_operator": ">=",
                    "boost_factor": 2.0,
                    "search_fields": ["title", "summary", "conclusion"]
                }
            ]
        }
    )

    query_text: str = Field(
        ...,
        title="Search Query Text",
        description="Natural language query text that will be converted to embeddings for semantic search. "
                   "Use descriptive phrases rather than keywords for best results. "
                   "Examples: 'budget planning strategies', 'team performance metrics', 'security vulnerabilities'",
        min_length=1,
        max_length=1000,
        examples=["machine learning algorithms", "quarterly budget analysis", "customer satisfaction trends"]
    )

    similarity_threshold: float = Field(
        default=0.7,
        title="Minimum Similarity Score",
        description="Minimum cosine similarity score (0.0-1.0) required for results. "
                   "Higher values (0.8+) return more precise matches, lower values (0.6-) return broader results. "
                   "Recommended: 0.7 for balanced results, 0.8+ for precision, 0.6- for recall",
        ge=0.0,
        le=1.0,
        examples=[0.6, 0.7, 0.75, 0.8, 0.85]
    )

    similarity_operator: SimilarityOperatorEnum = Field(
        default=SimilarityOperatorEnum.GREATER_EQUAL,
        title="Similarity Comparison Method",
        description="How to compare similarity scores against the threshold. "
                   "'>=' (default) includes scores at or above threshold, "
                   "'>' excludes exact threshold matches, "
                   "'<=' finds dissimilar content below threshold"
    )

    boost_factor: float = Field(
        default=1.0,
        title="Relevance Score Multiplier",
        description="Multiplier for text search scores in final ranking (must be positive). "
                   "Values > 1.0 increase importance of text similarity, "
                   "values < 1.0 decrease importance relative to other criteria. "
                   "Use 2.0+ to prioritize text relevance, 0.5 for secondary importance",
        gt=0.0,
        le=10.0,
        examples=[0.5, 1.0, 1.5, 2.0, 3.0]
    )

    search_fields: List[str] = Field(
        default_factory=list,
        title="Target Content Fields",
        description="Specific document fields to search within. Empty list searches all available fields. "
                   "Common fields: 'title', 'content', 'summary', 'abstract', 'conclusion', 'keywords'. "
                   "Restricting fields can improve precision and performance",
        examples=[[], ["title", "summary"], ["content"], ["title", "abstract", "conclusion"]]
    )


class VectorSearchRequest(BaseModel):
    """
    Direct vector similarity search using pre-computed embedding vectors.

    For when you already have embedding vectors and want to perform Approximate Nearest
    Neighbor (ANN) search directly. More efficient than semantic search if you've
    pre-computed embeddings. Uses vector similarity algorithms (cosine, Euclidean, etc.)
    to find the most similar content in the embedding space.
    """
    model_config = ConfigDict(
        title="Direct Vector Similarity Search",
        json_schema_extra={
            "examples": [
                {
                    "embedding_vector": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "similarity_threshold": 0.8,
                    "similarity_operator": ">=",
                    "distance_metric": "cosine",
                    "boost_factor": 1.0
                },
                {
                    "embedding_vector": [0.8, -0.2, 0.5, 0.1, -0.3],
                    "similarity_threshold": 0.75,
                    "similarity_operator": ">=",
                    "distance_metric": "euclidean",
                    "boost_factor": 1.5
                }
            ]
        }
    )

    embedding_vector: List[float] = Field(
        ...,
        title="Pre-computed Embedding Vector",
        description="Pre-computed embedding vector for direct similarity search. "
                   "Must be a list of floating-point numbers representing the vector. "
                   "Vector dimensions must match the index being searched",
        min_length=1
    )
    similarity_threshold: float = Field(
        default=0.7,
        title="Minimum Similarity Threshold",
        description="Minimum similarity threshold for vector matching (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    similarity_operator: SimilarityOperatorEnum = Field(
        default=SimilarityOperatorEnum.GREATER_EQUAL,
        title="Similarity Comparison Operator",
        description="Operator for comparing similarity scores against threshold"
    )
    distance_metric: Literal["cosine", "euclidean", "dot_product"] = Field(
        default="cosine",
        title="Vector Distance Metric",
        description="Distance metric algorithm for vector similarity calculation. "
                   "Cosine is most common for normalized vectors"
    )
    boost_factor: float = Field(
        default=1.0,
        title="Vector Score Boost Factor",
        description="Multiplier for vector similarity scores in final ranking",
        gt=0.0
    )


class DateSearchRequest(BaseModel):
    """
    Temporal filtering for documents based on extracted dates and time periods.

    Supports various date patterns from absolute ranges to relative periods and
    fiscal calendars. Handles partial dates (like "Q2 2024") and fuzzy temporal
    references. Essential for time-sensitive searches and chronological analysis.
    """
    model_config = ConfigDict(
        title="Date-Based Document Filtering",
        json_schema_extra={
            "examples": [
                {
                    "operator": "relative_days",
                    "relative_value": 30,
                    "include_partial_dates": True,
                    "specificity_levels": ["full", "date_only", "month_only"]
                },
                {
                    "operator": "within",
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-03-31T23:59:59Z",
                    "include_partial_dates": True
                },
                {
                    "operator": "quarter",
                    "year": 2024,
                    "quarter": 2,
                    "include_partial_dates": True
                }
            ]
        }
    )

    operator: DateRangeOperatorEnum = Field(
        ...,
        title="Date Filtering Method",
        description="Type of date comparison to perform. Choose based on your temporal search needs: "
                   "'relative_days/months' for recent content, 'within' for specific ranges, "
                   "'quarter/fiscal_year' for business periods, 'before/after' for chronological boundaries"
    )

    # Absolute date range fields
    start_date: Optional[datetime] = Field(
        default=None,
        title="Range Start Date",
        description="Beginning of date range for 'within' operator (inclusive). "
                   "Must be provided with end_date. Should be timezone-aware or UTC. "
                   "Example: '2024-01-01T00:00:00Z' for start of year",
        examples=["2024-01-01T00:00:00Z", "2024-06-15T09:00:00Z"]
    )

    end_date: Optional[datetime] = Field(
        default=None,
        title="Range End Date",
        description="End of date range for 'within' operator (inclusive). "
                   "Must be provided with start_date and should be after start_date. "
                   "Example: '2024-12-31T23:59:59Z' for end of year",
        examples=["2024-12-31T23:59:59Z", "2024-06-30T17:00:00Z"]
    )

    exact_date: Optional[datetime] = Field(
        default=None,
        title="Specific Target Date",
        description="Precise date for 'exactly', 'before', or 'after' operators. "
                   "Used as the comparison point for chronological filtering. "
                   "Time component matters for precision",
        examples=["2024-03-15T12:00:00Z", "2024-12-25T00:00:00Z"]
    )

    # Relative date fields
    relative_value: Optional[int] = Field(
        default=None,
        title="Relative Time Quantity",
        description="Number of days or months for relative date operators. "
                   "For 'relative_days': 7=last week, 30=last month, 90=last quarter. "
                   "For 'relative_months': 1=last month, 3=last quarter, 12=last year. "
                   "Must be positive integer",
        gt=0,
        le=3650,  # ~10 years max
        examples=[7, 14, 30, 60, 90, 180, 365]
    )

    # Business period fields
    year: Optional[int] = Field(
        default=None,
        title="Target Year",
        description="Year for fiscal_year, calendar_year, or quarter operators. "
                   "Must be 4-digit year. For fiscal years, represents the ending year "
                   "(e.g., FY2024 might be July 2023 - June 2024)",
        ge=1900,
        le=2100,
        examples=[2023, 2024, 2025]
    )

    quarter: Optional[int] = Field(
        default=None,
        title="Business Quarter (1-4)",
        description="Quarter number for 'quarter' operator (1=Q1, 2=Q2, 3=Q3, 4=Q4). "
                   "Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec for calendar quarters. "
                   "Fiscal quarters may differ based on organization's fiscal year start",
        ge=1,
        le=4,
        examples=[1, 2, 3, 4]
    )

    # Advanced date matching options
    include_partial_dates: bool = Field(
        default=True,
        title="Include Imprecise Dates",
        description="Whether to include documents with partial or vague dates like '2024', 'Q2', 'Summer 2024'. "
                   "True (recommended) for comprehensive results, False for precise dates only. "
                   "Helps find documents with approximate temporal references"
    )

    specificity_levels: List[str] = Field(
        default_factory=lambda: ["full", "date_only", "month_only", "quarter_only", "year_only"],
        title="Allowed Date Precision Levels",
        description="Types of date specificity to include in search results. "
                   "'full'=complete timestamp, 'date_only'=YYYY-MM-DD, 'month_only'=YYYY-MM, "
                   "'quarter_only'=Q1 2024, 'year_only'=2024. "
                   "Remove levels to exclude less precise dates",
        examples=[
            ["full", "date_only"],
            ["full", "date_only", "month_only"],
            ["quarter_only", "year_only"]
        ]
    )

    @model_validator(mode='after')
    def validate_date_operator_requirements(self):
        """Validate that required fields are provided for the operator."""
        operator = self.operator

        if operator == DateRangeOperatorEnum.WITHIN:
            if not (self.start_date and self.end_date):
                raise ValueError("WITHIN operator requires both start_date and end_date")
        elif operator in [DateRangeOperatorEnum.BEFORE, DateRangeOperatorEnum.AFTER, DateRangeOperatorEnum.EXACTLY]:
            if not self.exact_date:
                raise ValueError(f"{operator} operator requires exact_date")
        elif operator in [DateRangeOperatorEnum.RELATIVE_DAYS, DateRangeOperatorEnum.RELATIVE_MONTHS]:
            if not self.relative_value:
                raise ValueError(f"{operator} operator requires relative_value")
        elif operator == DateRangeOperatorEnum.QUARTER:
            if not (self.year and self.quarter):
                raise ValueError("QUARTER operator requires both year and quarter")
        elif operator in [DateRangeOperatorEnum.FISCAL_YEAR, DateRangeOperatorEnum.CALENDAR_YEAR]:
            if not self.year:
                raise ValueError(f"{operator} operator requires year")

        return self


class TopicSearchRequest(BaseModel):
    """
    Content filtering by topic classification using pattern matching.

    Searches documents by their assigned topics/categories rather than content text.
    Topics are typically generated by classification algorithms or manual tagging.
    Supports wildcard patterns for flexible matching and confidence thresholds
    for quality control. Essential for categorical document discovery.
    """
    model_config = ConfigDict(
        title="Topic-Based Content Classification Search",
        json_schema_extra={
            "examples": [
                {
                    "include_topics": ["machine-learning%", "artificial-intelligence%", "data-science%"],
                    "exclude_topics": ["deprecated%", "draft%"],
                    "require_all_included": False,
                    "min_confidence": 0.8,
                    "boost_factor": 1.5
                },
                {
                    "include_topics": ["finance%", "quarterly-report%"],
                    "exclude_topics": ["preliminary%", "confidential%"],
                    "require_all_included": True,
                    "min_confidence": 0.75,
                    "boost_factor": 2.0
                },
                {
                    "include_topics": ["technology", "innovation", "strategy"],
                    "min_confidence": 0.9
                }
            ]
        }
    )

    include_topics: List[str] = Field(
        default_factory=list,
        title="Required Topic Patterns",
        description="List of topic patterns that documents should contain. Supports SQL LIKE wildcards: "
                   "'%' matches any characters, '_' matches single character. "
                   "Examples: 'technology%' matches 'technology-trends', 'technology-review', etc. "
                   "Use exact names for precise matching or patterns for category matching",
        examples=[
            ["machine-learning", "data-science"],
            ["finance%", "budget%", "cost%"],
            ["security-policy", "authentication%"],
            ["quarterly%", "annual%"]
        ]
    )

    exclude_topics: List[str] = Field(
        default_factory=list,
        title="Excluded Topic Patterns",
        description="List of topic patterns to exclude from results. Documents matching any excluded pattern "
                   "will be filtered out regardless of include_topics matches. Same wildcard support as include_topics. "
                   "Useful for removing drafts, deprecated content, or sensitive categories",
        examples=[
            ["deprecated%", "obsolete%"],
            ["draft%", "preliminary%"],
            ["confidential%", "internal-only%"],
            []
        ]
    )

    require_all_included: bool = Field(
        default=False,
        title="Require All Topics (AND vs OR)",
        description="Logical combination mode for include_topics list. "
                   "False (default): Document needs ANY of the included topics (OR logic - more permissive). "
                   "True: Document needs ALL of the included topics (AND logic - more restrictive). "
                   "Choose based on whether topics are alternatives or requirements"
    )

    min_confidence: float = Field(
        default=0.7,
        title="Minimum Topic Confidence Score",
        description="Minimum confidence threshold for topic classifications (0.0-1.0). "
                   "Only topics assigned with confidence >= this threshold will be considered. "
                   "Higher values (0.8+) ensure high-quality topic matches, lower values (0.6-) include uncertain classifications. "
                   "Recommended: 0.7 for balanced results, 0.8+ for precision",
        ge=0.0,
        le=1.0,
        examples=[0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    )

    boost_factor: float = Field(
        default=1.0,
        title="Topic Score Boost Multiplier",
        description="Multiplier for topic-based relevance scores in final ranking (must be positive). "
                   "Values > 1.0 increase importance of topic matching relative to other criteria. "
                   "Values < 1.0 decrease topic importance. Use 2.0+ to prioritize topic relevance, "
                   "0.5 for secondary topic filtering",
        gt=0.0,
        le=10.0,
        examples=[0.5, 1.0, 1.5, 2.0, 3.0]
    )

    @model_validator(mode='after')
    def validate_topics(self):
        """Validate that at least one topic filter is specified."""
        if not self.include_topics and not self.exclude_topics:
            raise ValueError("Must specify at least one topic to include or exclude")
        return self


class MetadataSearchRequest(BaseModel):
    """
    Document metadata filtering for structured field-based searches.

    Filters documents based on their metadata properties like author, department,
    creation date, tags, etc. Supports exact matching, pattern matching with wildcards,
    numeric range queries, and existence checks. Essential for structured document
    discovery based on document properties rather than content.
    """
    model_config = ConfigDict(
        title="Document Metadata Field Filtering",
        json_schema_extra={
            "examples": [
                {
                    "exact_matches": {"department": "engineering", "priority": "high"},
                    "like_patterns": {"title": "%quarterly%"},
                    "range_filters": {"word_count": {"gte": 100, "lte": 5000}},
                    "exists_filters": ["author", "created_date"]
                },
                {
                    "exact_matches": {"status": "published", "document_type": "report"},
                    "range_filters": {"created_date": {"gte": "2024-01-01", "lte": "2024-12-31"}},
                    "exists_filters": ["reviewer", "approval_date"]
                }
            ]
        }
    )

    exact_matches: Dict[str, Any] = Field(
        default_factory=dict,
        title="Exact Field Matches",
        description="Dictionary of metadata fields that must match exactly. "
                   "Key is field name, value is the required value. "
                   "Example: {'department': 'engineering', 'status': 'approved'}"
    )
    like_patterns: Dict[str, str] = Field(
        default_factory=dict,
        title="Pattern Matching Fields",
        description="Dictionary of metadata fields to match using SQL LIKE patterns. "
                   "Supports '%' wildcard for any characters and '_' for single character. "
                   "Example: {'title': '%quarterly%', 'filename': '%.pdf'}"
    )
    range_filters: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        title="Numeric Range Filters",
        description="Dictionary of range queries for numeric/date fields. "
                   "Each value is a dict with 'gte', 'lte', 'gt', 'lt' operators. "
                   "Example: {'word_count': {'gte': 100, 'lte': 5000}}"
    )
    exists_filters: List[str] = Field(
        default_factory=list,
        title="Required Field Existence",
        description="List of metadata field names that must exist (not null/empty). "
                   "Useful for ensuring documents have required metadata. "
                   "Example: ['author', 'created_date', 'department']"
    )

    @model_validator(mode='after')
    def validate_metadata_filters(self):
        """Validate that at least one metadata filter is specified."""
        if not any([self.exact_matches, self.like_patterns, self.range_filters, self.exists_filters]):
            raise ValueError("Must specify at least one metadata filter")
        return self


class ElementSearchRequest(BaseModel):
    """
    Document structure and element-specific filtering.

    Filters based on document structure elements like element types (paragraph, header),
    specific document IDs, content length constraints, and hierarchical relationships.
    Essential for focusing searches on specific document sections or document sets.
    """
    model_config = ConfigDict(
        title="Document Structure Element Filtering",
        json_schema_extra={
            "examples": [
                {
                    "element_types": ["header", "paragraph"],
                    "doc_ids": ["doc123", "doc456"],
                    "content_length_min": 50,
                    "content_length_max": 1000
                },
                {
                    "element_types": ["table", "list_item"],
                    "exclude_doc_ids": ["draft_001", "template_base"],
                    "doc_sources": ["quarterly_reports%", "policy_docs%"],
                    "content_length_min": 20
                }
            ]
        }
    )

    element_types: List[str] = Field(
        default_factory=list,
        title="Document Element Types",
        description="List of element types to include in search results. "
                   "Common types: 'paragraph', 'header', 'table', 'list_item', 'caption'. "
                   "Empty list includes all element types"
    )
    doc_ids: List[str] = Field(
        default_factory=list,
        title="Include Document IDs",
        description="Specific document IDs to include in search. "
                   "Only elements from these documents will be returned. "
                   "Empty list includes all documents"
    )
    exclude_doc_ids: List[str] = Field(
        default_factory=list,
        title="Exclude Document IDs",
        description="Document IDs to exclude from search results. "
                   "Elements from these documents will be filtered out. "
                   "Useful for removing drafts or test documents"
    )
    doc_sources: List[str] = Field(
        default_factory=list,
        title="Document Source Patterns",
        description="Document source patterns to filter by. "
                   "Supports pattern matching for document origins or collections. "
                   "Example: ['reports%', 'policies%']"
    )
    parent_element_ids: List[str] = Field(
        default_factory=list,
        title="Parent Element IDs",
        description="Filter by parent element relationships in document hierarchy. "
                   "Useful for finding elements within specific document sections"
    )
    content_length_min: Optional[int] = Field(
        default=None,
        title="Minimum Content Length",
        description="Minimum character length for element content. "
                   "Useful for filtering out very short elements like headers or captions",
        ge=0
    )
    content_length_max: Optional[int] = Field(
        default=None,
        title="Maximum Content Length",
        description="Maximum character length for element content. "
                   "Useful for filtering out very long elements or focusing on summaries",
        ge=0
    )

    @field_validator('content_length_max')
    def validate_content_length_range(self, v, info):
        """Validate that max length is not less than min length."""
        min_length = info.data.get('content_length_min')
        if min_length is not None and v is not None and v < min_length:
            raise ValueError("Maximum content length cannot be less than minimum")
        return v


# ============================================================================
# NESTED SEARCH CRITERIA GROUP
# ============================================================================

class SearchCriteriaGroupRequest(BaseModel):
    """
    Logical grouping of multiple search criteria with boolean operators.

    Enables complex query composition like "(text_search OR topic_search) AND date_search AND NOT metadata_search".
    Groups can be nested to create sophisticated search logic. Each group combines its criteria
    using the specified logical operator before being combined with parent group criteria.
    """
    model_config = ConfigDict(
        title="Search Criteria Logical Group",
        json_schema_extra={
            "examples": [
                {
                    "operator": "AND",
                    "semantic_search": {
                        "query_text": "machine learning",
                        "similarity_threshold": 0.8
                    },
                    "date_search": {
                        "operator": "relative_days",
                        "relative_value": 30
                    },
                    "element_search": {
                        "element_types": ["header", "paragraph"]
                    }
                },
                {
                    "operator": "OR",
                    "semantic_search": {
                        "query_text": "artificial intelligence"
                    },
                    "topic_search": {
                        "include_topics": ["ai%", "ml%", "deep-learning%"]
                    },
                    "sub_groups": [
                        {
                            "operator": "AND",
                            "metadata_search": {
                                "exact_matches": {"department": "research"}
                            },
                            "date_search": {
                                "operator": "relative_months",
                                "relative_value": 6
                            }
                        }
                    ]
                }
            ]
        }
    )

    operator: LogicalOperatorEnum = Field(
        default=LogicalOperatorEnum.AND,
        title="Logical Combination Operator",
        description="How to combine criteria within this group. "
                   "AND: All criteria must match (intersection - more restrictive). "
                   "OR: Any criteria can match (union - more permissive). "
                   "NOT: Exclude documents matching criteria (negation - filtering)"
    )

    # Core search criteria with detailed descriptions
    semantic_search: Optional[SemanticSearchRequest] = Field(
        default=None,
        title="Semantic Text Search",
        description="Natural language semantic search using embeddings. Best for conceptual queries "
                   "where you want to find content with similar meaning, not just exact keywords. "
                   "Automatically converts your text to embeddings for vector similarity search. "
                   "Example: 'budget planning' can match 'financial strategy' content"
    )

    vector_search: Optional[VectorSearchRequest] = Field(
        default=None,
        title="Direct Vector Search",
        description="Pre-computed embedding vector search for when you already have embeddings. "
                   "More efficient than semantic search if you've already generated embeddings. "
                   "Uses Approximate Nearest Neighbor (ANN) algorithms for similarity search"
    )

    date_search: Optional[DateSearchRequest] = Field(
        default=None,
        title="Temporal Document Filtering",
        description="Filter documents by dates extracted from content or metadata. "
                   "Supports relative periods (last 30 days), absolute ranges, business quarters, "
                   "and fiscal years. Essential for time-sensitive document discovery"
    )

    topic_search: Optional[TopicSearchRequest] = Field(
        default=None,
        title="Topic-Based Content Filtering",
        description="Filter by document topics using pattern matching. Topics are typically "
                   "pre-classified categories or tags. Supports wildcards (%) for flexible matching. "
                   "Good for broad categorical filtering like 'technology%' or 'finance%'"
    )

    metadata_search: Optional[MetadataSearchRequest] = Field(
        default=None,
        title="Document Metadata Filtering",
        description="Filter by document metadata fields like author, department, status, etc. "
                   "Supports exact matches, pattern matching, range queries, and existence checks. "
                   "Perfect for structured filtering on document properties"
    )

    element_search: Optional[ElementSearchRequest] = Field(
        default=None,
        title="Document Structure Filtering",
        description="Filter by document structure elements like element types, document IDs, "
                   "content length, or hierarchical relationships. Use to focus on specific "
                   "document sections (headers, paragraphs) or document sets"
    )

    # Nested logical groups
    sub_groups: List['SearchCriteriaGroupRequest'] = Field(
        default_factory=list,
        title="Nested Criteria Groups",
        description="Child groups for complex logical expressions. Each sub-group is evaluated "
                   "independently then combined with this group's operator. Enables queries like "
                   "'(A OR B) AND (C OR D)' through nested group structure. No depth limit"
    )

    @model_validator(mode='after')
    def validate_has_criteria(self):
        """Validate that the group has at least one criterion or subgroup."""
        criteria_count = sum([
            self.semantic_search is not None,
            self.vector_search is not None,
            self.date_search is not None,
            self.topic_search is not None,
            self.metadata_search is not None,
            self.element_search is not None,
            len(self.sub_groups) > 0
        ])

        if criteria_count == 0:
            raise ValueError("SearchCriteriaGroup must have at least one criterion or sub-group")

        return self


# Update forward reference for Pydantic v2
SearchCriteriaGroupRequest.model_rebuild()


# ============================================================================
# MAIN SEARCH QUERY REQUEST MODEL
# ============================================================================

class SearchQueryRequest(BaseModel):
    """
    Complete structured search query configuration for complex document retrieval.

    This is the top-level search request that combines multiple search criteria with
    result formatting options and scoring parameters. Supports everything from simple
    text searches to complex multi-criteria queries with custom ranking algorithms.
    """
    model_config = ConfigDict(
        title="Structured Document Search Query",
        json_schema_extra={
            "examples": [
                {
                    "criteria_group": {
                        "operator": "AND",
                        "semantic_search": {
                            "query_text": "machine learning algorithms",
                            "similarity_threshold": 0.8,
                            "boost_factor": 2.0
                        },
                        "date_search": {
                            "operator": "relative_days",
                            "relative_value": 30
                        },
                        "element_search": {
                            "element_types": ["header", "paragraph"]
                        }
                    },
                    "limit": 20,
                    "include_element_dates": True,
                    "include_similarity_scores": True,
                    "score_combination": "weighted_avg",
                    "custom_weights": {
                        "text_similarity": 2.0,
                        "date_relevance": 0.5
                    }
                },
                {
                    "criteria_group": {
                        "operator": "OR",
                        "topic_search": {
                            "include_topics": ["technology%", "innovation%"],
                            "min_confidence": 0.8
                        },
                        "metadata_search": {
                            "exact_matches": {"department": "engineering"},
                            "range_filters": {"priority": {"gte": 8}}
                        }
                    },
                    "limit": 50,
                    "offset": 20,
                    "include_metadata": True,
                    "include_topics": True,
                    "include_highlighting": True
                }
            ]
        }
    )

    criteria_group: SearchCriteriaGroupRequest = Field(
        ...,
        title="Main Search Criteria",
        description="Root criteria group containing all search logic. This defines what documents to find "
                   "using various search types (text, date, topic, metadata, etc.) combined with logical operators. "
                   "All search functionality is defined within this criteria group structure"
    )

    # Result pagination and limits
    limit: int = Field(
        default=10,
        title="Maximum Results Count",
        description="Maximum number of search results to return in a single response. "
                   "Higher values provide more results but may impact performance. "
                   "Typical values: 10-20 for UI display, 50-100 for analysis, 1000 for bulk operations",
        gt=0,
        le=1000,
        examples=[10, 20, 50, 100, 500]
    )

    offset: int = Field(
        default=0,
        title="Result Pagination Offset",
        description="Number of results to skip from the beginning (for pagination). "
                   "Used with limit for pagination: page 1 = offset 0, page 2 = offset=limit, etc. "
                   "Must be non-negative. Large offsets may impact performance",
        ge=0,
        examples=[0, 10, 20, 50, 100]
    )

    # Result enrichment options
    include_element_dates: bool = Field(
        default=False,
        title="Include Extracted Date Information",
        description="Whether to include detailed information about dates extracted from document content. "
                   "Provides date parsing details, confidence scores, and temporal context. "
                   "Useful for chronological analysis but increases response size"
    )

    include_metadata: bool = Field(
        default=True,
        title="Include Document Metadata",
        description="Whether to include document metadata fields (author, tags, etc.) in results. "
                   "True (recommended) provides rich context, False reduces response size. "
                   "Metadata often contains essential document properties for result evaluation"
    )

    include_topics: bool = Field(
        default=False,
        title="Include Topic Classifications",
        description="Whether to include topic/category information for each result. "
                   "Shows topic labels and confidence scores assigned to documents. "
                   "Helpful for understanding content categorization and thematic analysis"
    )

    include_similarity_scores: bool = Field(
        default=True,
        title="Include Relevance Scores",
        description="Whether to include detailed similarity/relevance scores in results. "
                   "Shows individual component scores (text, topic, date) and final combined score. "
                   "Essential for understanding ranking and tuning search parameters"
    )

    include_highlighting: bool = Field(
        default=False,
        title="Include Content Highlighting",
        description="Whether to include highlighted excerpts showing why content matched. "
                   "Marks relevant passages in search results for better user understanding. "
                   "Increases response size but improves result comprehension"
    )

    # Advanced scoring configuration
    score_combination: ScoreCombinationEnum = Field(
        default=ScoreCombinationEnum.WEIGHTED_AVG,
        title="Score Combination Method",
        description="Algorithm for combining multiple relevance scores into final ranking. "
                   "'weighted_avg' (recommended): balanced combination using custom weights. "
                   "'multiply': penalizes low scores heavily. 'add': simple sum. 'max': highest individual score"
    )

    custom_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "text_similarity": 1.0,
            "embedding_similarity": 1.0,
            "topic_confidence": 0.5,
            "date_relevance": 0.3
        },
        title="Score Component Weights",
        description="Relative importance weights for different relevance factors. "
                   "Higher weights increase importance in final ranking. "
                   "text_similarity: semantic text matching, topic_confidence: topic classification strength, "
                   "date_relevance: temporal relevance, embedding_similarity: vector similarity",
        examples=[
            {"text_similarity": 2.0, "topic_confidence": 1.0, "date_relevance": 0.5},
            {"text_similarity": 1.0, "embedding_similarity": 1.0, "metadata_relevance": 0.3}
        ]
    )

    # Query tracking
    query_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        title="Unique Query Identifier",
        description="Automatically generated unique identifier for this search query. "
                   "Used for tracking, logging, and debugging. Useful for correlating "
                   "requests with responses in distributed systems",
        examples=["123e4567-e89b-12d3-a456-426614174000"]
    )


# ============================================================================
# SEARCH RESULT MODELS
# ============================================================================

class ExtractedDateInfo(BaseModel):
    """
    Information about dates extracted from document content.

    Contains parsed date information with various levels of precision and context.
    Used for temporal analysis and date-based search result enrichment.
    """
    original_text: str = Field(
        ...,
        title="Original Date Text",
        description="Original date text as found in the document content",
        examples=["March 15, 2024", "Q2 2024", "Summer 2023", "2024-03-15"]
    )
    iso_string: Optional[str] = Field(
        default=None,
        title="ISO Date String",
        description="Date in ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ) if fully parseable",
        examples=["2024-03-15T00:00:00Z", "2024-06-30T23:59:59Z"]
    )
    timestamp: Optional[float] = Field(
        default=None,
        title="Unix Timestamp",
        description="Unix timestamp (seconds since epoch) for the parsed date",
        examples=[1710460800.0, 1719792000.0]
    )
    year: Optional[int] = Field(
        default=None,
        title="Year Component",
        description="Four-digit year extracted from the date",
        examples=[2023, 2024, 2025]
    )
    month: Optional[int] = Field(
        default=None,
        title="Month Component",
        description="Month number (1-12) if extractable",
        ge=1,
        le=12,
        examples=[1, 6, 12]
    )
    day: Optional[int] = Field(
        default=None,
        title="Day Component",
        description="Day of month (1-31) if extractable",
        ge=1,
        le=31,
        examples=[1, 15, 31]
    )
    quarter: Optional[int] = Field(
        default=None,
        title="Business Quarter",
        description="Business quarter (1-4) if extractable from context",
        ge=1,
        le=4,
        examples=[1, 2, 3, 4]
    )
    season: Optional[str] = Field(
        default=None,
        title="Season Name",
        description="Season name if extracted from seasonal references",
        examples=["Spring", "Summer", "Fall", "Winter"]
    )
    specificity_level: str = Field(
        default="full",
        title="Date Precision Level",
        description="Level of date precision that was extractable",
        examples=["full", "date_only", "month_only", "quarter_only", "year_only"]
    )
    context: str = Field(
        default="",
        title="Surrounding Text Context",
        description="Text context surrounding the date for disambiguation",
        examples=["published on", "due by", "quarterly report for"]
    )


class SearchResultItem(BaseModel):
    """
    Individual search result representing a matching document element.

    Contains the core document content plus enriched metadata, relevance scores,
    and optional extracted information. Each result represents a specific element
    (paragraph, header, table, etc.) from a document that matched the search criteria.
    """
    model_config = ConfigDict(
        title="Search Result Document Element",
        json_schema_extra={
            "examples": [
                {
                    "element_pk": 12345,
                    "element_id": "doc1_elem5",
                    "doc_id": "document_001",
                    "element_type": "paragraph",
                    "content_preview": "Machine learning algorithms have revolutionized data analysis by enabling automated pattern recognition...",
                    "final_score": 0.87,
                    "scores": {
                        "text_similarity": 0.85,
                        "topic_confidence": 0.92,
                        "date_relevance": 0.75
                    },
                    "metadata": {
                        "author": "Dr. Sarah Smith",
                        "department": "Research",
                        "created_date": "2024-03-15",
                        "document_type": "technical_report"
                    },
                    "topics": ["machine-learning", "data-science", "algorithms"],
                    "date_count": 3
                },
                {
                    "element_pk": 67890,
                    "element_id": "doc2_header1",
                    "doc_id": "document_002",
                    "element_type": "header",
                    "content_preview": "Quarterly Performance Analysis - Q3 2024",
                    "final_score": 0.78,
                    "scores": {
                        "text_similarity": 0.72,
                        "date_relevance": 0.95
                    },
                    "metadata": {
                        "section": "executive_summary",
                        "page_number": 1,
                        "word_count": 8
                    }
                }
            ]
        }
    )

    # Core document identification
    element_pk: int = Field(
        ...,
        title="Element Primary Key",
        description="Unique database identifier for this document element. "
                   "Used for internal tracking and can be used for follow-up queries. "
                   "Guaranteed to be unique across the entire document corpus",
        examples=[12345, 67890, 54321]
    )

    element_id: str = Field(
        ...,
        title="Element Identifier",
        description="Human-readable identifier combining document and element information. "
                   "Typically formatted as 'doc_id_element_number' or similar. "
                   "Useful for debugging and cross-referencing with source documents",
        examples=["doc1_elem5", "report_2024_header3", "policy_doc_para12"]
    )

    doc_id: str = Field(
        ...,
        title="Source Document ID",
        description="Identifier of the parent document containing this element. "
                   "Multiple elements can share the same doc_id. "
                   "Use for grouping results by document or retrieving full documents",
        examples=["document_001", "quarterly_report_q3_2024", "policy_handbook_v2"]
    )

    element_type: str = Field(
        ...,
        title="Content Element Type",
        description="Type of document element that matched the search criteria. "
                   "Common types: 'paragraph', 'header', 'table', 'list_item', 'caption', 'footer'. "
                   "Helps understand the structural context of the match",
        examples=["paragraph", "header", "table", "list_item", "caption", "abstract"]
    )

    # Content and relevance
    content_preview: str = Field(
        ...,
        title="Content Preview Text",
        description="Preview of the actual content that matched the search criteria. "
                   "May be truncated for long elements. Contains the text used for similarity matching. "
                   "Length typically limited to 200-500 characters for display purposes",
        examples=[
            "Machine learning algorithms have revolutionized data analysis...",
            "Q3 2024 Performance Summary",
            "The new security policy requires all employees to use two-factor authentication..."
        ]
    )

    final_score: float = Field(
        ...,
        title="Combined Relevance Score",
        description="Final calculated relevance score after combining all search criteria scores. "
                   "Range typically 0.0-1.0, where 1.0 is perfect match. "
                   "Used for ranking results. Higher scores indicate better matches",
        ge=0.0,
        le=1.0,
        examples=[0.87, 0.78, 0.92, 0.65]
    )

    # Optional enriched data
    scores: Optional[Dict[str, float]] = Field(
        default=None,
        title="Individual Score Components",
        description="Breakdown of relevance scores by search criteria type. "
                   "Shows contribution of text_similarity, topic_confidence, date_relevance, etc. "
                   "Useful for understanding why a result ranked highly and tuning search parameters",
        examples=[
            {"text_similarity": 0.85, "topic_confidence": 0.92, "date_relevance": 0.75},
            {"text_similarity": 0.72, "embedding_similarity": 0.89},
            {"topic_confidence": 0.95, "metadata_relevance": 0.88}
        ]
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        title="Document Element Metadata",
        description="Structured metadata associated with this document element. "
                   "Can include author, creation date, department, tags, document properties, etc. "
                   "Content varies by document type and indexing system configuration",
        examples=[
            {"author": "Dr. Smith", "department": "Research", "created_date": "2024-03-15"},
            {"section": "introduction", "page_number": 3, "word_count": 245},
            {"priority": "high", "status": "approved", "last_modified": "2024-05-20"}
        ]
    )

    topics: Optional[List[str]] = Field(
        default=None,
        title="Associated Topic Classifications",
        description="List of topics or categories assigned to this document element. "
                   "Typically generated by topic modeling or manual classification. "
                   "Helps understand the thematic content and can be used for further filtering",
        examples=[
            ["machine-learning", "data-science", "algorithms"],
            ["finance", "quarterly-report", "performance"],
            ["security", "policy", "authentication"]
        ]
    )

    extracted_dates: Optional[List[ExtractedDateInfo]] = Field(
        default=None,
        title="Detailed Date Extraction Results",
        description="Comprehensive information about dates found in this element's content. "
                   "Includes original text, parsed timestamps, confidence levels, and context. "
                   "Only present when include_element_dates=True in search request"
    )

    date_count: Optional[int] = Field(
        default=None,
        title="Number of Extracted Dates",
        description="Total count of dates extracted from this element's content. "
                   "Quick indicator of temporal richness without full date details. "
                   "Useful for sorting by temporal content density",
        ge=0,
        examples=[0, 1, 3, 5, 12]
    )


class SearchResponse(BaseModel):
    """Search response model containing results and metadata."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "query_id": "123e4567-e89b-12d3-a456-426614174000",
                "total_results": 45,
                "returned_results": 20,
                "execution_time_ms": 125.5,
                "results": [
                    {
                        "element_pk": 12345,
                        "element_id": "doc1_elem5",
                        "doc_id": "doc1",
                        "element_type": "paragraph",
                        "content_preview": "Machine learning algorithms...",
                        "final_score": 0.87
                    }
                ]
            }
        }
    )

    success: bool = Field(..., description="Whether the search was successful")
    query_id: str = Field(..., description="Unique query identifier")
    total_results: int = Field(..., description="Total number of results found")
    returned_results: int = Field(..., description="Number of results returned")
    execution_time_ms: Optional[float] = Field(default=None, description="Query execution time in milliseconds")

    # Results
    results: List[SearchResultItem] = Field(..., description="Search result items")

    # Query echo for reference
    query_summary: Optional[Dict[str, Any]] = Field(default=None, description="Summary of the executed query")

    # Error information
    error_message: Optional[str] = Field(default=None, description="Error message if search failed")
    missing_capabilities: Optional[List[str]] = Field(None, description="Backend capabilities missing for this query")


# ============================================================================
# CONVERSION UTILITIES
# ============================================================================

def pydantic_to_core_query(pydantic_query: SearchQueryRequest):
    """Convert Pydantic query model to core structured search query."""

    if CoreStructuredSearchQuery is None:
        raise ImportError("Core search modules not available. Cannot convert to core query.")

    def convert_criteria_group(pydantic_group: SearchCriteriaGroupRequest):
        """Convert Pydantic criteria group to core criteria group."""
        if not CoreSearchCriteriaGroup:
            raise ImportError("CoreSearchCriteriaGroup not available")

        core_group = CoreSearchCriteriaGroup(
            operator=LogicalOperator(pydantic_group.operator.value)
        )

        # Convert individual criteria (map new Pydantic names to existing core names)
        if pydantic_group.semantic_search:
            ts = pydantic_group.semantic_search
            if CoreTextSearchCriteria:
                core_group.text_criteria = CoreTextSearchCriteria(
                    query_text=ts.query_text,
                    similarity_threshold=ts.similarity_threshold,
                    similarity_operator=SimilarityOperator(ts.similarity_operator.value),
                    boost_factor=ts.boost_factor,
                    search_fields=ts.search_fields
                )

        if pydantic_group.vector_search:
            es = pydantic_group.vector_search
            if CoreEmbeddingSearchCriteria:
                core_group.embedding_criteria = CoreEmbeddingSearchCriteria(
                    embedding_vector=es.embedding_vector,
                    similarity_threshold=es.similarity_threshold,
                    similarity_operator=SimilarityOperator(es.similarity_operator.value),
                    distance_metric=es.distance_metric,
                    boost_factor=es.boost_factor
                )

        if pydantic_group.date_search:
            ds = pydantic_group.date_search
            if CoreDateSearchCriteria:
                core_group.date_criteria = CoreDateSearchCriteria(
                    operator=DateRangeOperator(ds.operator.value),
                    start_date=ds.start_date,
                    end_date=ds.end_date,
                    exact_date=ds.exact_date,
                    relative_value=ds.relative_value,
                    year=ds.year,
                    quarter=ds.quarter,
                    include_partial_dates=ds.include_partial_dates,
                    specificity_levels=ds.specificity_levels
                )

        if pydantic_group.topic_search:
            tops = pydantic_group.topic_search
            if CoreTopicSearchCriteria:
                core_group.topic_criteria = CoreTopicSearchCriteria(
                    include_topics=tops.include_topics,
                    exclude_topics=tops.exclude_topics,
                    require_all_included=tops.require_all_included,
                    min_confidence=tops.min_confidence,
                    boost_factor=tops.boost_factor
                )

        if pydantic_group.metadata_search:
            ms = pydantic_group.metadata_search
            if CoreMetadataSearchCriteria:
                core_group.metadata_criteria = CoreMetadataSearchCriteria(
                    exact_matches=ms.exact_matches,
                    like_patterns=ms.like_patterns,
                    range_filters=ms.range_filters,
                    exists_filters=ms.exists_filters
                )

        if pydantic_group.element_search:
            els = pydantic_group.element_search
            if CoreElementSearchCriteria:
                core_group.element_criteria = CoreElementSearchCriteria(
                    element_types=els.element_types,
                    doc_ids=els.doc_ids,
                    exclude_doc_ids=els.exclude_doc_ids,
                    doc_sources=els.doc_sources,
                    parent_element_ids=els.parent_element_ids,
                    content_length_min=els.content_length_min,
                    content_length_max=els.content_length_max
                )

        # Convert subgroups recursively
        for sub_group in pydantic_group.sub_groups:
            core_group.sub_groups.append(convert_criteria_group(sub_group))

        return core_group

    # Convert the main criteria group
    core_criteria_group = convert_criteria_group(pydantic_query.criteria_group)

    # Create the core structured query
    if CoreStructuredSearchQuery:
        core_query = CoreStructuredSearchQuery(
            criteria_group=core_criteria_group,
            limit=pydantic_query.limit,
            offset=pydantic_query.offset,
            include_element_dates=pydantic_query.include_element_dates,
            include_metadata=pydantic_query.include_metadata,
            include_topics=pydantic_query.include_topics,
            include_similarity_scores=pydantic_query.include_similarity_scores,
            include_highlighting=pydantic_query.include_highlighting,
            score_combination=pydantic_query.score_combination.value,
            custom_weights=pydantic_query.custom_weights,
            query_id=pydantic_query.query_id,
            created_at=datetime.now()
        )
        return core_query
    else:
        raise ImportError("CoreStructuredSearchQuery not available")


def core_results_to_pydantic(core_results: List[Dict[str, Any]],
                             query_id: str,
                             execution_time_ms: Optional[float] = None) -> SearchResponse:
    """Convert core search results to Pydantic response model."""

    # Convert results to Pydantic models
    pydantic_results: List[SearchResultItem] = []
    for result in core_results:
        # Convert extracted dates if present
        extracted_dates: Optional[List[ExtractedDateInfo]] = None
        if result.get('extracted_dates'):
            extracted_dates = [
                ExtractedDateInfo(
                    original_text=date_info.get('original_text', ''),
                    iso_string=date_info.get('iso_string'),
                    timestamp=date_info.get('timestamp'),
                    year=date_info.get('year'),
                    month=date_info.get('month'),
                    day=date_info.get('day'),
                    quarter=date_info.get('quarter'),
                    season=date_info.get('season'),
                    specificity_level=date_info.get('specificity_level', 'full'),
                    context=date_info.get('context', '')
                )
                for date_info in result['extracted_dates']
                if isinstance(date_info, dict)
            ]

        pydantic_result = SearchResultItem(
            element_pk=result.get('element_pk', 0),
            element_id=result.get('element_id', ''),
            doc_id=result.get('doc_id', ''),
            element_type=result.get('element_type', ''),
            content_preview=result.get('content_preview', ''),
            final_score=result.get('final_score', 0.0),
            scores=result.get('scores'),
            metadata=result.get('metadata'),
            topics=result.get('topics'),
            extracted_dates=extracted_dates,
            date_count=result.get('date_count')
        )
        pydantic_results.append(pydantic_result)

    return SearchResponse(
        success=True,
        query_id=query_id,
        total_results=len(core_results),
        returned_results=len(core_results),
        execution_time_ms=execution_time_ms,
        results=pydantic_results,
        query_summary=None,
        error_message=None,
        missing_capabilities=None
    )


# ============================================================================
# MAIN SEARCH EXECUTION FUNCTION
# ============================================================================

def execute_search(query: SearchQueryRequest,
                   database: Any,  # Generic type to avoid import dependency
                   validate_capabilities: bool = True) -> SearchResponse:
    """
    Execute a structured search query against a document database.

    Args:
        query: Pydantic search query request
        database: Document database implementation
        validate_capabilities: Whether to validate backend capabilities before execution

    Returns:
        Pydantic search response with results and metadata
    """
    import time

    start_time = time.time()

    try:
        # Check if core modules are available
        if CoreStructuredSearchQuery is None:
            return SearchResponse(
                success=False,
                query_id=query.query_id,
                total_results=0,
                returned_results=0,
                execution_time_ms=None,
                results=[],
                query_summary=None,
                error_message="Core search modules not available",
                missing_capabilities=None
            )

        # Convert Pydantic query to core query
        core_query = pydantic_to_core_query(query)

        # Validate backend capabilities if requested
        if validate_capabilities and hasattr(database, 'validate_query_support'):
            missing_capabilities = database.validate_query_support(core_query)
            if missing_capabilities:
                return SearchResponse(
                    success=False,
                    query_id=query.query_id,
                    total_results=0,
                    returned_results=0,
                    execution_time_ms=None,
                    results=[],
                    query_summary=None,
                    error_message="Backend does not support required capabilities",
                    missing_capabilities=[cap.value for cap in missing_capabilities]
                )

        # Execute the search
        if hasattr(database, 'execute_structured_search'):
            core_results = database.execute_structured_search(core_query)
        else:
            raise AttributeError("Database does not support structured search")

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Convert results to Pydantic response
        response = core_results_to_pydantic(core_results, query.query_id, execution_time_ms)

        # Add query summary
        response.query_summary = {
            "total_criteria": 1 if query.criteria_group.semantic_search or query.criteria_group.vector_search or
                                 query.criteria_group.date_search or query.criteria_group.topic_search or
                                 query.criteria_group.metadata_search or query.criteria_group.element_search else 0,
            "has_sub_groups": len(query.criteria_group.sub_groups) > 0,
            "search_types": []
        }

        # Add search types to summary
        if query.criteria_group.semantic_search:
            response.query_summary["search_types"].append("semantic")
        if query.criteria_group.vector_search:
            response.query_summary["search_types"].append("vector")
        if query.criteria_group.date_search:
            response.query_summary["search_types"].append("date")
        if query.criteria_group.topic_search:
            response.query_summary["search_types"].append("topic")
        if query.criteria_group.metadata_search:
            response.query_summary["search_types"].append("metadata")
        if query.criteria_group.element_search:
            response.query_summary["search_types"].append("element")

        return response

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            success=False,
            query_id=query.query_id,
            total_results=0,
            returned_results=0,
            execution_time_ms=execution_time_ms,
            results=[],
            query_summary=None,
            error_message=str(e),
            missing_capabilities=None
        )


# ============================================================================
# DESERIALIZATION UTILITIES AND EXAMPLES
# ============================================================================

def deserialize_search_query(data: Union[Dict[str, Any], str]) -> SearchQueryRequest:
    """
    Deserialize dictionary or JSON string into SearchQueryRequest.

    Args:
        data: Dictionary or JSON string containing search query data

    Returns:
        SearchQueryRequest: Validated and parsed search query

    Raises:
        ValidationError: If data doesn't match expected schema
        ValueError: If JSON string is malformed
    """
    try:
        if isinstance(data, str):
            return SearchQueryRequest.model_validate_json(data)
        else:
            return SearchQueryRequest.model_validate(data)
    except Exception as e:
        logger.error(f"Failed to deserialize search query: {e}")
        raise


def create_query_from_dict_examples() -> List[SearchQueryRequest]:
    """Demonstrate various ways to create SearchQueryRequest from dictionaries."""

    print("=== SearchQueryRequest Deserialization Examples ===\n")

    # Example 1: Simple dictionary deserialization
    print("1. Simple dictionary -> SearchQueryRequest:")
    simple_dict = {
        "criteria_group": {
            "operator": "AND",
            "semantic_search": {
                "query_text": "artificial intelligence trends",
                "similarity_threshold": 0.75,
                "boost_factor": 1.5
            },
            "date_search": {
                "operator": "relative_days",
                "relative_value": 30
            }
        },
        "limit": 25,
        "include_similarity_scores": True
    }

    # Method 1: model_validate (recommended)
    query1 = SearchQueryRequest.model_validate(simple_dict)
    print(f"   Created with model_validate(): {query1.query_id[:8]}...")
    print(f"   Text query: {query1.criteria_group.semantic_search.query_text}")
    print(f"   Date filter: Last {query1.criteria_group.date_search.relative_value} days")
    print()

    # Example 2: JSON string deserialization
    print("2. JSON string -> SearchQueryRequest:")
    json_data = '''
    {
        "criteria_group": {
            "operator": "OR",
            "topic_search": {
                "include_topics": ["technology%", "innovation%"],
                "min_confidence": 0.8,
                "boost_factor": 2.0
            },
            "metadata_search": {
                "exact_matches": {"department": "research", "status": "published"},
                "range_filters": {"word_count": {"gte": 100, "lte": 5000}}
            }
        },
        "limit": 50,
        "include_topics": true,
        "include_metadata": true,
        "score_combination": "weighted_avg",
        "custom_weights": {
            "topic_confidence": 2.0,
            "metadata_relevance": 1.0
        }
    }
    '''

    query2 = SearchQueryRequest.model_validate_json(json_data)
    print(f"   Created from JSON: {query2.query_id[:8]}...")
    print(f"   Topic patterns: {query2.criteria_group.topic_search.include_topics}")
    print(f"   Metadata filters: {list(query2.criteria_group.metadata_search.exact_matches.keys())}")
    print()

    # Example 3: Complex nested query deserialization
    print("3. Complex nested query -> SearchQueryRequest:")
    complex_dict = {
        "criteria_group": {
            "operator": "AND",
            "sub_groups": [
                {
                    "operator": "OR",
                    "semantic_search": {
                        "query_text": "machine learning algorithms",
                        "similarity_threshold": 0.8
                    },
                    "vector_search": {
                        "embedding_vector": [0.1, 0.2, 0.3, 0.4, 0.5],  # Example vector
                        "similarity_threshold": 0.75,
                        "distance_metric": "cosine"
                    }
                },
                {
                    "operator": "NOT",
                    "topic_search": {
                        "include_topics": ["deprecated%", "obsolete%"]
                    }
                }
            ],
            "date_search": {
                "operator": "quarter",
                "year": 2024,
                "quarter": 3
            }
        },
        "limit": 100,
        "offset": 20,
        "include_element_dates": True,
        "include_highlighting": True
    }

    query3 = SearchQueryRequest.model_validate(complex_dict)
    print(f"   Created complex query: {query3.query_id[:8]}...")
    print(f"   Sub-groups: {len(query3.criteria_group.sub_groups)}")
    print(f"   Date filter: Q{query3.criteria_group.date_search.quarter} {query3.criteria_group.date_search.year}")
    print(f"   Vector search: {len(query3.criteria_group.sub_groups[0].vector_search.embedding_vector)} dimensions")
    print()

    # Example 4: Error handling demonstration
    print("4. Error handling for invalid data:")
    invalid_dict = {
        "criteria_group": {
            "operator": "INVALID_OPERATOR",  # This will cause validation error
            "semantic_search": {
                "query_text": "",  # Empty text will cause validation error
                "similarity_threshold": 1.5  # Out of range
            }
        },
        "limit": -5  # Negative limit will cause validation error
    }

    try:
        SearchQueryRequest.model_validate(invalid_dict)
    except ValidationError as e:
        print(f"   Validation failed as expected: {len(e.errors())} errors found")
        for error in e.errors()[:2]:  # Show first 2 errors
            print(f"   - {error['loc']}: {error['msg']}")
    print()

    return [query1, query2, query3]


def serialize_and_deserialize_roundtrip() -> None:
    """Demonstrate serialization -> deserialization roundtrip."""

    print("=== Serialization/Deserialization Roundtrip ===\n")

    # Create original query
    original = SearchQueryRequest(
        criteria_group=SearchCriteriaGroupRequest(
            operator=LogicalOperatorEnum.AND,
            semantic_search=SemanticSearchRequest(
                query_text="data science methodologies",
                similarity_threshold=0.85,
                boost_factor=1.8
            ),
            topic_search=TopicSearchRequest(
                include_topics=["data-science%", "analytics%"],
                exclude_topics=["draft%"],
                min_confidence=0.8
            )
        ),
        limit=30,
        include_similarity_scores=True
    )

    print("1. Original query:")
    print(f"   Query ID: {original.query_id}")
    print(f"   Text: {original.criteria_group.semantic_search.query_text}")
    print()

    # Serialize to dict
    serialized_dict = original.model_dump()
    print("2. Serialized to dictionary:")
    print(f"   Keys: {list(serialized_dict.keys())}")
    print(f"   Criteria group operator: {serialized_dict['criteria_group']['operator']}")
    print()

    # Serialize to JSON
    serialized_json = original.model_dump_json(indent=2)
    print("3. Serialized to JSON:")
    print(f"   JSON length: {len(serialized_json)} characters")
    print()

    # Deserialize back from dict
    restored_from_dict = SearchQueryRequest.model_validate(serialized_dict)
    print("4. Restored from dictionary:")
    print(f"   Query ID matches: {original.query_id == restored_from_dict.query_id}")
    print(f"   Text query matches: {original.criteria_group.semantic_search.query_text == restored_from_dict.criteria_group.semantic_search.query_text}")
    print()

    # Deserialize back from JSON
    restored_from_json = SearchQueryRequest.model_validate_json(serialized_json)
    print("5. Restored from JSON:")
    print(f"   Query ID matches: {original.query_id == restored_from_json.query_id}")
    print(f"   Similarity threshold: {restored_from_json.criteria_group.semantic_search.similarity_threshold}")
    print()


# ============================================================================
# CONVENIENCE FUNCTIONS (Updated)
# ============================================================================

def create_simple_search(query_text: str,
                         days_back: Optional[int] = None,
                         element_types: Optional[List[str]] = None,
                         limit: int = 10) -> SearchQueryRequest:
    """Create a simple search query from basic parameters."""

    criteria_group = SearchCriteriaGroupRequest(
        operator=LogicalOperatorEnum.AND,
        semantic_search=SemanticSearchRequest(query_text=query_text)
    )

    if days_back:
        criteria_group.date_search = DateSearchRequest(
            operator=DateRangeOperatorEnum.RELATIVE_DAYS,
            relative_value=days_back
        )

    if element_types:
        criteria_group.element_search = ElementSearchRequest(
            element_types=element_types
        )

    return SearchQueryRequest(
        criteria_group=criteria_group,
        limit=limit,
        include_element_dates=True
    )


def create_topic_search(include_topics: List[str],
                        exclude_topics: Optional[List[str]] = None,
                        min_confidence: float = 0.7,
                        limit: int = 10) -> SearchQueryRequest:
    """Create a topic-based search query."""

    criteria_group = SearchCriteriaGroupRequest(
        operator=LogicalOperatorEnum.AND,
        topic_search=TopicSearchRequest(
            include_topics=include_topics,
            exclude_topics=exclude_topics or [],
            min_confidence=min_confidence
        )
    )

    return SearchQueryRequest(
        criteria_group=criteria_group,
        limit=limit,
        include_topics=True
    )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def demonstrate_pydantic_search() -> None:
    """Demonstrate usage of the Pydantic search system."""

    print("=== Pydantic v2 Structured Search Examples ===\n")

    # Example 1: Simple text search
    simple_query = create_simple_search(
        query_text="machine learning algorithms",
        days_back=30,
        element_types=["header", "paragraph"],
        limit=20
    )

    print("1. Simple search query:")
    print(simple_query.model_dump_json(indent=2))
    print()

    # Example 2: Complex nested query
    complex_query = SearchQueryRequest(
        criteria_group=SearchCriteriaGroupRequest(
            operator=LogicalOperatorEnum.AND,
            sub_groups=[
                SearchCriteriaGroupRequest(
                    operator=LogicalOperatorEnum.OR,
                    semantic_search=SemanticSearchRequest(
                        query_text="artificial intelligence",
                        similarity_threshold=0.8
                    ),
                    topic_search=TopicSearchRequest(
                        include_topics=["ai%", "ml%"]
                    )
                ),
                SearchCriteriaGroupRequest(
                    operator=LogicalOperatorEnum.NOT,
                    topic_search=TopicSearchRequest(
                        include_topics=["deprecated%"]
                    )
                )
            ]
        ),
        limit=50,
        include_element_dates=True,
        include_topics=True
    )

    print("2. Complex nested query:")
    print("Query ID:", complex_query.query_id)
    print("Has", len(complex_query.criteria_group.sub_groups), "sub-groups")
    print()

    # Example 3: Topic search
    topic_query = create_topic_search(
        include_topics=["technology%", "innovation%"],
        exclude_topics=["deprecated%", "obsolete%"],
        min_confidence=0.8
    )

    print("3. Topic search query:")
    print("Include topics:", topic_query.criteria_group.topic_search.include_topics)
    print("Exclude topics:", topic_query.criteria_group.topic_search.exclude_topics)
    print()


if __name__ == "__main__":
    demonstrate_pydantic_search()
