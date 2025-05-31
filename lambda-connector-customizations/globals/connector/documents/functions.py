"""
functions.py

This is an example of how you can use the Python SDK's built-in Function connector to easily write Python code.
When you add a Python Lambda connector to your Hasura project, this file is generated for you!

In this file you'll find code examples that will help you get up to speed with the usage of the Hasura lambda connector.
If you are an old pro and already know what is going on you can get rid of these example functions and start writing your own code.
"""
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Literal
import json

import aiohttp
from hasura_ndc import start
from hasura_ndc.function_connector import FunctionConnector
from hasura_ndc.instrumentation import \
    with_active_span  # If you aren't planning on adding additional tracing spans, you don't need this!
from opentelemetry.trace import \
    get_tracer, \
    get_current_span  # If you aren't planning on adding additional tracing spans, you don't need this either!
from pydantic import \
    Field, BaseModel, \
    model_validator  # You only need this import if you plan to have complex inputs/outputs, which function similar to how frameworks like FastAPI do


# ============================================================================
# STRUCTURED SEARCH PYDANTIC MODELS (replicated from server)
# ============================================================================

class LogicalOperatorEnum(str, Enum):
    """Logical operators for combining search criteria in complex queries."""
    AND = "AND"  # All criteria must match (intersection)
    OR = "OR"  # Any criteria can match (union)
    NOT = "NOT"  # Exclude matching criteria (negation)


class DateRangeOperatorEnum(str, Enum):
    """Date filtering operators supporting various temporal query patterns."""
    WITHIN = "within"  # Between two specific dates (inclusive range)
    BEFORE = "before"  # Earlier than specified date (exclusive)
    AFTER = "after"  # Later than specified date (exclusive)
    EXACTLY = "exactly"  # Exact date match (precise)
    RELATIVE_DAYS = "relative_days"  # Within last N days from now
    RELATIVE_MONTHS = "relative_months"  # Within last N months from now
    FISCAL_YEAR = "fiscal_year"  # Within organization's fiscal year
    CALENDAR_YEAR = "calendar_year"  # Within standard calendar year
    QUARTER = "quarter"  # Within specific quarter (Q1-Q4)


class SimilarityOperatorEnum(str, Enum):
    """Comparison operators for semantic similarity thresholds in vector searches."""
    GREATER_THAN = ">"  # Similarity must exceed threshold
    GREATER_EQUAL = ">="  # Similarity must meet or exceed threshold (default)
    LESS_THAN = "<"  # Similarity must be below threshold
    LESS_EQUAL = "<="  # Similarity must be at or below threshold
    EQUALS = "="  # Similarity must exactly match threshold


class ScoreCombinationEnum(str, Enum):
    """Methods for combining multiple relevance scores into final ranking."""
    MULTIPLY = "multiply"  # Multiplicative combination (penalizes low scores)
    ADD = "add"  # Additive combination (simple sum)
    MAX = "max"  # Take the highest individual score
    WEIGHTED_AVG = "weighted_avg"  # Weighted average (balanced, default)


class SemanticSearchRequest(BaseModel):
    """Semantic text search using natural language queries and vector embeddings."""

    query_text: str = Field(
        ...,
        title="Search Query Text",
        description="Natural language query text for semantic search",
        min_length=1,
        max_length=1000
    )

    similarity_threshold: float = Field(
        default=0.7,
        title="Minimum Similarity Score",
        description="Minimum cosine similarity score (0.0-1.0) required for results",
        ge=0.0,
        le=1.0
    )

    similarity_operator: SimilarityOperatorEnum = Field(
        default=SimilarityOperatorEnum.GREATER_EQUAL,
        title="Similarity Comparison Method"
    )

    boost_factor: float = Field(
        default=1.0,
        title="Relevance Score Multiplier",
        description="Multiplier for text search scores in final ranking",
        gt=0.0,
        le=10.0
    )

    search_fields: List[str] = Field(
        default_factory=list,
        title="Target Content Fields",
        description="Specific document fields to search within"
    )


class VectorSearchRequest(BaseModel):
    """Direct vector similarity search using pre-computed embedding vectors."""

    embedding_vector: List[float] = Field(
        ...,
        title="Pre-computed Embedding Vector",
        description="Pre-computed embedding vector for direct similarity search",
        min_length=1
    )

    similarity_threshold: float = Field(
        default=0.7,
        title="Minimum Similarity Threshold",
        ge=0.0,
        le=1.0
    )

    similarity_operator: SimilarityOperatorEnum = Field(
        default=SimilarityOperatorEnum.GREATER_EQUAL
    )

    distance_metric: Literal["cosine", "euclidean", "dot_product"] = Field(
        default="cosine",
        title="Vector Distance Metric"
    )

    boost_factor: float = Field(
        default=1.0,
        title="Vector Score Boost Factor",
        gt=0.0
    )


class DateSearchRequest(BaseModel):
    """Temporal filtering for documents based on extracted dates and time periods."""

    operator: DateRangeOperatorEnum = Field(
        ...,
        title="Date Filtering Method"
    )

    # Absolute date range fields
    start_date: Optional[datetime] = Field(
        default=None,
        title="Range Start Date"
    )

    end_date: Optional[datetime] = Field(
        default=None,
        title="Range End Date"
    )

    exact_date: Optional[datetime] = Field(
        default=None,
        title="Specific Target Date"
    )

    # Relative date fields
    relative_value: Optional[int] = Field(
        default=None,
        title="Relative Time Quantity",
        gt=0,
        le=3650
    )

    # Business period fields
    year: Optional[int] = Field(
        default=None,
        title="Target Year",
        ge=1900,
        le=2100
    )

    quarter: Optional[int] = Field(
        default=None,
        title="Business Quarter (1-4)",
        ge=1,
        le=4
    )

    # Advanced date matching options
    include_partial_dates: bool = Field(
        default=True,
        title="Include Imprecise Dates"
    )

    specificity_levels: List[str] = Field(
        default_factory=lambda: ["full", "date_only", "month_only", "quarter_only", "year_only"],
        title="Allowed Date Precision Levels"
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
    """Content filtering by topic classification using pattern matching."""

    include_topics: List[str] = Field(
        default_factory=list,
        title="Required Topic Patterns",
        description="List of topic patterns that documents should contain"
    )

    exclude_topics: List[str] = Field(
        default_factory=list,
        title="Excluded Topic Patterns",
        description="List of topic patterns to exclude from results"
    )

    require_all_included: bool = Field(
        default=False,
        title="Require All Topics (AND vs OR)"
    )

    min_confidence: float = Field(
        default=0.7,
        title="Minimum Topic Confidence Score",
        ge=0.0,
        le=1.0
    )

    boost_factor: float = Field(
        default=1.0,
        title="Topic Score Boost Multiplier",
        gt=0.0,
        le=10.0
    )

    @model_validator(mode='after')
    def validate_topics(self):
        """Validate that at least one topic filter is specified."""
        if not self.include_topics and not self.exclude_topics:
            raise ValueError("Must specify at least one topic to include or exclude")
        return self


class MetadataSearchRequest(BaseModel):
    """Document metadata filtering for structured field-based searches."""

    exact_matches: Dict[str, Any] = Field(
        default_factory=dict,
        title="Exact Field Matches"
    )

    like_patterns: Dict[str, str] = Field(
        default_factory=dict,
        title="Pattern Matching Fields"
    )

    range_filters: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        title="Numeric Range Filters"
    )

    exists_filters: List[str] = Field(
        default_factory=list,
        title="Required Field Existence"
    )

    @model_validator(mode='after')
    def validate_metadata_filters(self):
        """Validate that at least one metadata filter is specified."""
        if not any([self.exact_matches, self.like_patterns, self.range_filters, self.exists_filters]):
            raise ValueError("Must specify at least one metadata filter")
        return self


class ElementSearchRequest(BaseModel):
    """Document structure and element-specific filtering."""

    element_types: List[str] = Field(
        default_factory=list,
        title="Document Element Types"
    )

    doc_ids: List[str] = Field(
        default_factory=list,
        title="Include Document IDs"
    )

    exclude_doc_ids: List[str] = Field(
        default_factory=list,
        title="Exclude Document IDs"
    )

    doc_sources: List[str] = Field(
        default_factory=list,
        title="Document Source Patterns"
    )

    parent_element_ids: List[str] = Field(
        default_factory=list,
        title="Parent Element IDs"
    )

    content_length_min: Optional[int] = Field(
        default=None,
        title="Minimum Content Length",
        ge=0
    )

    content_length_max: Optional[int] = Field(
        default=None,
        title="Maximum Content Length",
        ge=0
    )


class SearchCriteriaGroupRequest(BaseModel):
    """Logical grouping of multiple search criteria with boolean operators."""

    operator: LogicalOperatorEnum = Field(
        default=LogicalOperatorEnum.AND,
        title="Logical Combination Operator"
    )

    # Core search criteria
    semantic_search: Optional[SemanticSearchRequest] = Field(
        default=None,
        title="Semantic Text Search"
    )

    vector_search: Optional[VectorSearchRequest] = Field(
        default=None,
        title="Direct Vector Search"
    )

    date_search: Optional[DateSearchRequest] = Field(
        default=None,
        title="Temporal Document Filtering"
    )

    topic_search: Optional[TopicSearchRequest] = Field(
        default=None,
        title="Topic-Based Content Filtering"
    )

    metadata_search: Optional[MetadataSearchRequest] = Field(
        default=None,
        title="Document Metadata Filtering"
    )

    element_search: Optional[ElementSearchRequest] = Field(
        default=None,
        title="Document Structure Filtering"
    )

    # Nested logical groups
    sub_groups: List['SearchCriteriaGroupRequest'] = Field(
        default_factory=list,
        title="Nested Criteria Groups"
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


# Update forward reference
SearchCriteriaGroupRequest.model_rebuild()


class StructuredSearchRequest(BaseModel):
    """Complete structured search query configuration for complex document retrieval."""

    criteria_group: SearchCriteriaGroupRequest = Field(
        ...,
        title="Main Search Criteria"
    )

    # Result pagination and limits
    limit: int = Field(
        default=10,
        title="Maximum Results Count",
        gt=0,
        le=1000
    )

    offset: int = Field(
        default=0,
        title="Result Pagination Offset",
        ge=0
    )

    # Result enrichment options
    include_element_dates: bool = Field(
        default=False,
        title="Include Extracted Date Information"
    )

    include_metadata: bool = Field(
        default=True,
        title="Include Document Metadata"
    )

    include_topics: bool = Field(
        default=False,
        title="Include Topic Classifications"
    )

    include_similarity_scores: bool = Field(
        default=True,
        title="Include Relevance Scores"
    )

    include_highlighting: bool = Field(
        default=False,
        title="Include Content Highlighting"
    )

    # Advanced scoring configuration
    score_combination: ScoreCombinationEnum = Field(
        default=ScoreCombinationEnum.WEIGHTED_AVG,
        title="Score Combination Method"
    )

    custom_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "text_similarity": 1.0,
            "embedding_similarity": 1.0,
            "topic_confidence": 0.5,
            "date_relevance": 0.3
        },
        title="Score Component Weights"
    )

    # Query tracking
    query_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        title="Unique Query Identifier"
    )


class SimpleStructuredSearchRequest(BaseModel):
    """Simplified structured search request for common patterns."""

    query_text: str = Field(
        ...,
        title="Search Query Text",
        min_length=1
    )

    limit: int = Field(
        default=10,
        title="Maximum Results",
        gt=0,
        le=100
    )

    similarity_threshold: float = Field(
        default=0.7,
        title="Minimum Similarity Score",
        ge=0.0,
        le=1.0
    )

    include_topics: Optional[List[str]] = Field(
        default=None,
        title="Topic Patterns to Include"
    )

    exclude_topics: Optional[List[str]] = Field(
        default=None,
        title="Topic Patterns to Exclude"
    )

    days_back: Optional[int] = Field(
        default=None,
        title="Filter to Last N Days",
        gt=0
    )

    element_types: Optional[List[str]] = Field(
        default=None,
        title="Element Types to Filter"
    )


# ============================================================================
# EXISTING ELEMENT FLAT MODEL
# ============================================================================

class ElementFlat(BaseModel):
    element_pk: int = Field(
        description="Primary key of the element in the document store. Used for direct element retrieval and unique identification."
    )
    score: Optional[float] = Field(
        default=None,
        description="Semantic similarity score (-1 to 1) indicating how closely the content matches conceptually. Higher scores mean more relevant matches."
    )
    element_id: str = Field(
        default="",
        description="Unique identifier of the specific matching element within its document. Used for element-level referencing and relationships."
    )
    element_type: str = Field(
        default="",
        description="Type of document element (e.g., section, paragraph, table, list) indicating the structural context of the match. Helps understand content organization."
    )
    content_preview: str = Field(
        default="",
        description="Abbreviated preview of the matching content, suitable for display. May be truncated for large elements. Use text field for complete content."
    )
    doc_id: str = Field(
        default="",
        description="Unique identifier of the document containing this match. Use to locate or reference source documents."
    )
    content_location: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Location details of the content within its document, such as page numbers, sections, or coordinates. Format varies by document type."
    )
    source: Optional[str] = Field(
        default=None,
        description="Origin or system source of the document. Useful for tracking document provenance and filtering results by system."
    )
    text: Optional[str] = Field(
        default=None,
        description="Complete text content of the matching element when resolve_text is true. Provides full context of the match without structural formatting."
    )
    content: Optional[str] = Field(
        default=None,
        description="Complete structured content of the matching element when resolve_content is true. Preserves document-specific formatting and structure."
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="Identifier of the element's parent when include_parents is true. Used to understand containment relationships and document hierarchy."
    )
    path: Optional[str] = Field(
        default=None,
        description="Full path showing element's location in document hierarchy. Format depends on document type. Useful for understanding context and navigation."
    )
    content_hash: Optional[str] = Field(
        default=None,
        description="Hash identifier of the element's content. Useful for tracking content changes, versioning, or deduplication."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional document or element metadata key-value pairs. May include custom attributes, tags, or system-specific information."
    )


# ============================================================================
# CONNECTOR AND CONFIGURATION
# ============================================================================

connector = FunctionConnector()

# This last section shows you how to add OTEL tracing to any of your functions!
tracer = get_tracer("document_search.server") # You only need a tracer if you plan to add additional Otel spans

# Configuration for the web server
SEARCH_SERVER_URL = os.environ.get('DOCUMENTS_URI', 'http://localhost:5000')
SEARCH_API_KEY = os.environ.get('SEARCH_API_KEY')  # Optional API key


# ============================================================================
# STRUCTURED SEARCH FUNCTIONS
# ============================================================================

@connector.register_query
async def search_structured_query(
        structured_query: str = Field(
            ...,
            description="""Complete structured search configuration as a dictionary with criteria groups and logical operators.

Expected structure:
{
    "criteria_group": {
        "operator": "AND" | "OR" | "NOT",  # Logical combination operator
        "semantic_search": {  # Optional: Semantic text search
            "query_text": "string",  # Natural language query (1-1000 chars)
            "similarity_threshold": 0.7,  # Min similarity score (0.0-1.0)
            "similarity_operator": ">=" | ">" | "<" | "<=" | "=",
            "boost_factor": 1.0,  # Score multiplier (0.0-10.0)
            "search_fields": ["field1", "field2"]  # Optional: specific fields
        },
        "vector_search": {  # Optional: Direct vector similarity search
            "embedding_vector": [0.1, 0.2, ...],  # Pre-computed embedding
            "similarity_threshold": 0.7,
            "similarity_operator": ">=",
            "distance_metric": "cosine" | "euclidean" | "dot_product",
            "boost_factor": 1.0
        },
        "date_search": {  # Optional: Temporal filtering
            "operator": "within" | "before" | "after" | "exactly" | "relative_days" | "relative_months" | "fiscal_year" | "calendar_year" | "quarter",
            "start_date": "2024-01-01T00:00:00Z",  # For "within" operator
            "end_date": "2024-12-31T23:59:59Z",    # For "within" operator
            "exact_date": "2024-06-15T12:00:00Z",  # For "before"/"after"/"exactly"
            "relative_value": 30,  # For "relative_days"/"relative_months" (1-3650)
            "year": 2024,  # For year-based operators (1900-2100)
            "quarter": 2,  # For "quarter" operator (1-4)
            "include_partial_dates": true,
            "specificity_levels": ["full", "date_only", "month_only", "quarter_only", "year_only"]
        },
        "topic_search": {  # Optional: Topic-based filtering
            "include_topics": ["ai%", "machine-learning%"],  # Topic patterns to include (supports % wildcards)
            "exclude_topics": ["deprecated%", "draft%"],     # Topic patterns to exclude
            "require_all_included": false,  # true = AND logic, false = OR logic
            "min_confidence": 0.7,  # Min topic confidence (0.0-1.0)
            "boost_factor": 1.0
        },
        "metadata_search": {  # Optional: Document metadata filtering
            "exact_matches": {"author": "john.doe", "status": "published"},
            "like_patterns": {"title": "%report%", "department": "finance%"},
            "range_filters": {"page_count": {"min": 10, "max": 100}},
            "exists_filters": ["created_date", "last_modified"]
        },
        "element_search": {  # Optional: Document structure filtering
            "element_types": ["paragraph", "header", "list_item"],
            "doc_ids": ["doc1", "doc2"],  # Include specific documents
            "exclude_doc_ids": ["draft_doc1"],  # Exclude specific documents
            "doc_sources": ["system1%", "wiki%"],  # Source patterns
            "parent_element_ids": ["section1", "chapter2"],
            "content_length_min": 50,  # Min content length in characters
            "content_length_max": 5000   # Max content length in characters
        },
        "sub_groups": [  # Optional: Nested criteria groups
            {
                "operator": "OR",
                "semantic_search": {...},
                "topic_search": {...}
            }
        ]
    },
    "limit": 10,  # Max results (1-1000)
    "offset": 0,   # Pagination offset (>=0)
    "include_element_dates": false,  # Include extracted date info
    "include_metadata": true,        # Include document metadata
    "include_topics": false,         # Include topic classifications
    "include_similarity_scores": true,  # Include relevance scores
    "include_highlighting": false,   # Include content highlighting
    "score_combination": "weighted_avg" | "multiply" | "add" | "max",  # Score combination method
    "custom_weights": {  # Score component weights
        "text_similarity": 1.0,
        "embedding_similarity": 1.0,
        "topic_confidence": 0.5,
        "date_relevance": 0.3
    },
    "query_id": "optional-unique-id"  # Query tracking ID
}

Example simple query:
{
    "criteria_group": {
        "operator": "AND",
        "semantic_search": {
            "query_text": "machine learning algorithms",
            "similarity_threshold": 0.7
        },
        "topic_search": {
            "include_topics": ["ai%", "ml%"],
            "exclude_topics": ["deprecated%"]
        }
    },
    "limit": 20
}

Example complex query with date filtering:
{
    "criteria_group": {
        "operator": "AND",
        "sub_groups": [
            {
                "operator": "OR",
                "semantic_search": {"query_text": "security policy"},
                "topic_search": {"include_topics": ["security%"]}
            },
            {
                "operator": "AND",
                "date_search": {
                    "operator": "relative_days",
                    "relative_value": 90
                },
                "metadata_search": {
                    "exact_matches": {"status": "active"}
                }
            }
        ]
    },
    "limit": 15,
    "include_topics": true
}"""
        ),
        resolve_text: Optional[bool] = Field(
            default=False,
            description="Whether to resolve text content in search tree"
        ),
        resolve_content: Optional[bool] = Field(
            default=False,
            description="Whether to resolve raw content in search tree"
        ),
        flat: Optional[bool] = Field(
            default=False,
            description="Whether to return flat results instead of hierarchical"
        ),
        include_parents: Optional[bool] = Field(
            default=True,
            description="Whether to include parent elements in flat results"
        )
) -> List[ElementFlat]:
    """
    Execute a complex structured search with multiple criteria types and logical operators.

    This function provides the full power of structured search, allowing you to combine:
    - Semantic text search with embedding similarity
    - Topic-based filtering with pattern matching
    - Date-based filtering with relative and absolute ranges
    - Metadata and element structure filtering
    - Nested criteria groups with logical operators (AND, OR, NOT)

    The structured query uses a criteria group system where you can nest multiple search
    types and combine them with logical operators for sophisticated search logic.

    Examples:
    - Find documents that match "AI research" AND have topics like "machine-learning%"
      AND were created in the last 30 days
    - Find content that matches "security policy" OR has topics "security%"
      BUT NOT topics "deprecated%"
    - Complex nested: ((text_search OR topic_search) AND date_search) AND NOT exclude_criteria

    Parameters:
    :param structured_query: Complete structured search configuration dictionary with criteria groups
    :param resolve_text: Whether to materialize text content in the search tree
    :param resolve_content: Whether to materialize raw content in the search tree
    :param flat: Whether to return flat results instead of hierarchical document structure
    :param include_parents: Whether to include parent elements when using flat results

    Returns:
    List of ElementFlat objects with search results and optional materialized content.
    When text/content resolution is enabled, the search tree preserves document hierarchy
    with fully materialized content accessible via the text and content fields.
    """

    async def work(_structured_query, _resolve_text, _resolve_content, _flat, _include_parents) -> List[ElementFlat]:
        _span = get_current_span()

        # Parse JSON string to dictionary
        try:
            if isinstance(_structured_query, str):
                structured_query_dict = json.loads(_structured_query)
            else:
                raise ValueError("structured_query must be a JSON string")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in structured_query: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing structured_query: {str(e)}")

        # Basic validation of required structure
        if not isinstance(structured_query_dict, dict):
            raise ValueError("structured_query must be a JSON object")

        if "criteria_group" not in structured_query_dict:
            raise ValueError("structured_query must contain 'criteria_group' field")

        # Set defaults
        if not isinstance(_resolve_text, bool):
            _resolve_text = False
        if not isinstance(_resolve_content, bool):
            _resolve_content = False
        if not isinstance(_flat, bool):
            _flat = False
        if not isinstance(_include_parents, bool):
            _include_parents = True

        # Prepare request headers
        headers = {
            'Content-Type': 'application/json'
        }
        if SEARCH_API_KEY:
            headers['X-API-Key'] = SEARCH_API_KEY

        # Build query parameters for content materialization
        params = {
            'text': str(_resolve_text).lower(),
            'content': str(_resolve_content).lower(),
            'flat': str(_flat).lower(),
            'include_parents': str(_include_parents).lower()
        }

        # Use the parsed structured query dict as payload
        payload = structured_query_dict.copy()

        # Add a query_id if not present
        if 'query_id' not in payload:
            payload['query_id'] = str(uuid.uuid4())

        try:
            # Make HTTP request to the structured search endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{SEARCH_SERVER_URL}/api/search/structured",
                        json=payload,
                        headers=headers,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=60)  # Longer timeout for complex queries
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Structured search server error {response.status}: {error_text}")

                    response_data = await response.json()

        except Exception as e:
            _span.set_attribute("error", str(e))
            _span.set_attribute("search_error", "HTTP request failed")
            raise Exception(f"Failed to execute structured search: {str(e)}")

        try:
            search_tree = response_data.get('search_tree', [])
            search_tree = [ElementFlat(**item) for item in search_tree]

            _span.set_attribute("result_count", len(search_tree))
            _span.set_attribute("query_id", payload.get('query_id', 'unknown'))
            _span.set_attribute("search_type", "structured")

            return search_tree

        except Exception as e:
            _span.set_attribute("error", str(e))
            _span.set_attribute("processing_error", "Failed to process structured search results")
            raise Exception(f"Failed to parse structured search results: {str(e)}")

    return await with_active_span(
        tracer,
        "Structured Search",
        lambda span: work(
            _structured_query=structured_query,
            _resolve_text=resolve_text,
            _resolve_content=resolve_content,
            _flat=flat,
            _include_parents=include_parents
        ),
        {
            "query_id": "extracted_from_parsed_json",
            "search_type": "structured",
            "resolve_text": str(resolve_text),
            "resolve_content": str(resolve_content),
            "flat": str(flat),
            "include_parents": str(include_parents)
        }
    )


@connector.register_query
async def search_simple_structured(
        query_text: str = Field(
            ...,
            description="Natural language search query (required)"
        ),
        limit: Optional[int] = Field(
            default=10,
            description="Maximum number of results to return"
        ),
        similarity_threshold: Optional[float] = Field(
            default=0.7,
            description="Minimum similarity score threshold (0.0-1.0)"
        ),
        include_topics: Optional[List[str]] = Field(
            default=None,
            description="Topic patterns to include (supports % wildcards). Example: ['ai%', 'machine-learning%']"
        ),
        exclude_topics: Optional[List[str]] = Field(
            default=None,
            description="Topic patterns to exclude (supports % wildcards). Example: ['deprecated%', 'draft%']"
        ),
        days_back: Optional[int] = Field(
            default=None,
            description="Filter to documents from last N days. Example: 30 for last month"
        ),
        element_types: Optional[List[str]] = Field(
            default=None,
            description="Filter by element types. Example: ['paragraph', 'header', 'list_item']"
        ),
        resolve_text: Optional[bool] = Field(
            default=False,
            description="Whether to resolve text content in search tree"
        ),
        resolve_content: Optional[bool] = Field(
            default=False,
            description="Whether to resolve raw content in search tree"
        ),
        flat: Optional[bool] = Field(
            default=False,
            description="Whether to return flat results instead of hierarchical"
        ),
        include_parents: Optional[bool] = Field(
            default=True,
            description="Whether to include parent elements in flat results"
        )
) -> List[ElementFlat]:
    """
    Simplified structured search interface for common search patterns.

    This function provides an easy-to-use interface for structured search without requiring
    you to construct complex criteria group objects. It automatically builds structured
    queries from common parameters like topic filtering, date ranges, and element types.

    Common use cases:
    - Search for "machine learning" in documents with AI topics from the last 30 days
    - Find "security policies" excluding deprecated content
    - Locate "customer support" procedures in paragraph elements only
    - Search recent documentation (last 60 days) about "API integration"

    The function combines semantic text search with optional filters:
    - Topic filtering: Include/exclude documents based on topic classifications
    - Date filtering: Limit to documents from recent time periods
    - Element filtering: Focus on specific document element types
    - Content materialization: Optionally resolve full text/content with document hierarchy

    Parameters:
    :param query_text: Natural language search query (required)
    :param limit: Maximum results to return (default: 10)
    :param similarity_threshold: Minimum semantic similarity score 0.0-1.0 (default: 0.7)
    :param include_topics: Topic patterns to include, supports wildcards (optional)
    :param exclude_topics: Topic patterns to exclude, supports wildcards (optional)
    :param days_back: Filter to documents from last N days (optional)
    :param element_types: Filter by document element types (optional)
    :param resolve_text: Whether to materialize text content (default: False)
    :param resolve_content: Whether to materialize raw content (default: False)
    :param flat: Whether to return flat vs hierarchical results (default: False)
    :param include_parents: Whether to include parent elements in flat results (default: True)

    Returns:
    List of ElementFlat objects representing search results with optional materialized content.
    """

    def extract_value(param, default_value=None):
        """Extract actual value from potential Field object"""
        if hasattr(param, 'default'):
            # This is a Field object, get its default value
            return param.default if param.default is not ... else default_value
        return param if param is not None else default_value

    def safe_bool(value, default=False):
        """Safely convert to boolean"""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes')
        return bool(value)

    def safe_int(value, default=None):
        """Safely convert to integer"""
        if value is None:
            return default
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def safe_float(value, default=None):
        """Safely convert to float"""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def safe_list(value, default=None):
        """Safely convert to list"""
        if value is None:
            return default or []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            # Try to parse JSON array or comma-separated values
            try:
                import json
                return json.loads(value)
            except:
                return [item.strip() for item in value.split(',') if item.strip()]
        return default or []

    async def work(_query_text, _limit, _similarity_threshold, _include_topics, _exclude_topics,
                   _days_back, _element_types, _resolve_text, _resolve_content, _flat, _include_parents) -> List[ElementFlat]:
        _span = get_current_span()

        # Extract actual values from potential Field objects and apply defaults
        _query_text = extract_value(_query_text, "")
        _limit = safe_int(extract_value(_limit), 10)
        _similarity_threshold = safe_float(extract_value(_similarity_threshold), 0.7)
        _include_topics = safe_list(extract_value(_include_topics), [])
        _exclude_topics = safe_list(extract_value(_exclude_topics), [])
        _days_back = safe_int(extract_value(_days_back), None)
        _element_types = safe_list(extract_value(_element_types), [])
        _resolve_text = safe_bool(extract_value(_resolve_text), False)
        _resolve_content = safe_bool(extract_value(_resolve_content), False)
        _flat = safe_bool(extract_value(_flat), False)
        _include_parents = safe_bool(extract_value(_include_parents), True)

        # Validate required parameters
        if not _query_text or not isinstance(_query_text, str):
            raise ValueError("query_text is required and must be a non-empty string")

        # Prepare request headers
        headers = {
            'Content-Type': 'application/json'
        }
        if SEARCH_API_KEY:
            headers['X-API-Key'] = SEARCH_API_KEY

        # Build query parameters for content materialization
        params = {
            'text': str(_resolve_text).lower(),
            'content': str(_resolve_content).lower(),
            'flat': str(_flat).lower(),
            'include_parents': str(_include_parents).lower()
        }

        # Build simple structured request payload
        payload = {
            'query_text': _query_text,
            'limit': _limit,
            'similarity_threshold': _similarity_threshold
        }

        # Add optional filters only if they have values
        if _include_topics:
            payload['include_topics'] = _include_topics
        if _exclude_topics:
            payload['exclude_topics'] = _exclude_topics
        if _days_back is not None:
            payload['days_back'] = _days_back
        if _element_types:
            payload['element_types'] = _element_types

        try:
            # Make HTTP request to the simple structured search endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{SEARCH_SERVER_URL}/api/search/structured/simple",
                        json=payload,
                        headers=headers,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Simple structured search server error {response.status}: {error_text}")

                    response_data = await response.json()

        except Exception as e:
            _span.set_attribute("error", str(e))
            _span.set_attribute("search_error", "HTTP request failed")
            raise Exception(f"Failed to execute simple structured search: {str(e)}")

        try:
            search_tree = response_data.get('search_tree', [])
            search_tree = [ElementFlat(**item) for item in search_tree]

            _span.set_attribute("result_count", len(search_tree))
            _span.set_attribute("query_text", _query_text)
            _span.set_attribute("search_type", "simple_structured")

            return search_tree

        except Exception as e:
            _span.set_attribute("error", str(e))
            _span.set_attribute("processing_error", "Failed to process simple structured search results")
            raise Exception(f"Failed to parse simple structured search results: {str(e)}")

    return await with_active_span(
        tracer,
        "Simple Structured Search",
        lambda span: work(
            _query_text=query_text,
            _limit=limit,
            _similarity_threshold=similarity_threshold,
            _include_topics=include_topics,
            _exclude_topics=exclude_topics,
            _days_back=days_back,
            _element_types=element_types,
            _resolve_text=resolve_text,
            _resolve_content=resolve_content,
            _flat=flat,
            _include_parents=include_parents
        ),
        {
            "query_text": str(query_text),
            "search_type": "simple_structured"
        }
    )


# ============================================================================
# CONVENIENCE FUNCTIONS FOR STRUCTURED SEARCH
# ============================================================================

@connector.register_query
async def search_with_topics_and_dates(
        query_text: str = Field(
            ...,
            description="Natural language search query (required)"
        ),
        include_topics: List[str] = Field(
            ...,
            description="Topic patterns to include (required). Example: ['ai%', 'machine-learning%']"
        ),
        days_back: int = Field(
            ...,
            description="Filter to documents from last N days (required). Example: 30 for last month"
        ),
        exclude_topics: Optional[List[str]] = Field(
            default=None,
            description="Topic patterns to exclude. Example: ['deprecated%', 'draft%']"
        ),
        limit: Optional[int] = Field(
            default=10,
            description="Maximum number of results to return"
        ),
        similarity_threshold: Optional[float] = Field(
            default=0.7,
            description="Minimum similarity score threshold (0.0-1.0)"
        ),
        resolve_text: Optional[bool] = Field(
            default=True,
            description="Whether to resolve text content in search tree"
        )
) -> List[ElementFlat]:
    """
    Convenience function for searching with topic and date filtering.

    Common pattern for finding recent documents on specific topics.
    Equivalent to simple structured search but with required topic and date filters.
    """
    return await search_simple_structured(
        query_text=query_text,
        include_topics=include_topics,
        exclude_topics=exclude_topics,
        days_back=days_back,
        limit=limit,
        similarity_threshold=similarity_threshold,
        resolve_text=resolve_text
    )


@connector.register_query
async def search_recent_by_topics(
        include_topics: List[str] = Field(
            ...,
            description="Topic patterns to include (required). Example: ['ai%', 'machine-learning%']"
        ),
        days_back: int = Field(
            default=30,
            description="Filter to documents from last N days"
        ),
        exclude_topics: Optional[List[str]] = Field(
            default=None,
            description="Topic patterns to exclude. Example: ['deprecated%', 'draft%']"
        ),
        limit: Optional[int] = Field(
            default=20,
            description="Maximum number of results to return"
        ),
        min_confidence: Optional[float] = Field(
            default=0.8,
            description="Minimum confidence threshold for topic results (0.0-1.0)"
        )
) -> List[ElementFlat]:
    """
    Topic-only search for recent documents without text query.

    Useful for browsing recent content by category or discovering new documents
    in specific topic areas.
    """
    # Build a structured query with topic and date criteria only
    criteria_group = SearchCriteriaGroupRequest(
        operator=LogicalOperatorEnum.AND,
        topic_search=TopicSearchRequest(
            include_topics=include_topics,
            exclude_topics=exclude_topics or [],
            min_confidence=min_confidence
        ),
        date_search=DateSearchRequest(
            operator=DateRangeOperatorEnum.RELATIVE_DAYS,
            relative_value=days_back
        )
    )

    structured_query = StructuredSearchRequest(
        criteria_group=criteria_group,
        limit=limit,
        include_topics=True,
        include_element_dates=True
    )

    return await search_structured_query(
        structured_query=json.dumps(structured_query.model_dump()),
        resolve_text=True,
        flat=True
    )


# ============================================================================
# EXISTING SEARCH FUNCTIONS (unchanged)
# ============================================================================

@connector.register_query
async def search_document_detail(
        search_for: str = Field(
            ...,
            description="Natural language text to search with. Can be a question, description, or topic. The search uses semantic similarity, so exact word matches aren't needed."
        ),
        include_parents: Optional[bool] = Field(
            default=None,
            description="Include containing elements (e.g., sections containing matching paragraphs) to provide fuller context. Parent elements help understand where matches fit in the document structure. Defaults to False."
        ),
        just_documents: Optional[bool] = Field(
            default=None,
            description="Include only the top level documents for the search."
        ),
        resolve_content: Optional[bool] = Field(
            default=None,
            description="Include complete structured content of matching elements. Useful when document structure (like XML or JSON) contains important context beyond plain text. Defaults to False."
        ),
        resolve_text: Optional[bool] = Field(
            default=None,
            description="Include complete text content of matching elements. Useful when previews are insufficient and full text context is needed. Defaults to False."
        ),
        limit: Optional[int] = Field(
            default=None,
            description="Maximum results to return. Higher limits find more matches but may include less relevant content. Consider balancing with min_score to maintain relevance quality. Defaults to 10."
        ),
        min_score: Optional[float] = Field(
            default=None,
            description="Semantic similarity threshold (-1 to 1). Higher values ensure closer conceptual matches: 0.7+ for exact concepts, 0.5+ for closely related, 0.3+ for broadly related, 0.1+ for exploratory searches. Defaults to 0."
        ),
        include_topics: Optional[List[str]] = Field(
            default=None,
            description="A list of topics to include in the search. Includes ANY document that matches ANY topic. Uses a LIKE syntax where % matches any text. For example if the document topics included the source you might look for wikipedia articles setting this to: [\"%wikipedia%\"]"
        ),
        exclude_topics: Optional[List[str]] = Field(
            default=None,
            description="A list of topics to include in the search. Excludes ANY document that matches ANY topic"
        )
) -> List[ElementFlat]:

    """
    This performs semantic similarity search to find relevant content across documents, returning both the matching elements
    and their related context. The search identifies conceptually similar content even when exact words don't match, making
    it ideal for finding relevant documentation based on natural language descriptions or questions.

    The function breaks documents into searchable elements (paragraphs, lists, tables, etc.) and understands their relationships:
    - Structural relationships: parent/child elements (e.g., a section containing paragraphs), siblings (adjacent elements)
    - Explicit relationships: links or references between elements (if the document format supports it)
    - Semantic relationships: elements with similar meaning or topic, even if using different words

    Common uses:
    - Finding relevant documentation for customer inquiries
    - Locating policy information based on topic descriptions
    - Discovering related content across document sections
    - Matching technical documentation to user questions

    Parameters:
    :param include_topics: a list of topics to include in the search. Includes ANY document that matches ANY topic. Uses a LIKE syntax where % matches any text. For example if the document topics included the source you might look for wikipedia articles setting this to: ["%wikipedia%"]
    :param exclude_topics: a list of topics to exclude from the search. Excludes ANY document that matches ANY topic.
    :param resolve_text: This will provide the complete textual version of the matching element.
    :param resolve_content: This will provide the complete content (meaning any structural decorators or tags like formatting) of the matching element.
    :param search_for: Natural language text to search with. Can be a question ("How do I dispute a charge?"),
        description ("Customer asking about wire transfers"), or topic ("account security policies").
        The search uses semantic similarity, so exact word matches aren't needed.
    :param min_score: Similarity threshold (-1 to 1). Higher values mean closer conceptual matches:
        - 0.7+: Nearly exact concept matches
        - 0.5+: Closely related content
        - 0.3+: Broadly related content
        - 0.1+: Exploratory searches
        Defaults to 0.
    :param include_parents: Whether to include containing elements (e.g., the section containing a matching paragraph)
        to provide fuller context. Useful when matching content is part of a larger relevant section. Defaults to False.
    :param limit: Maximum number of results to return. Higher limits find more matches but may include less relevant content.
        Defaults to 10.
    :param just_documents: If True, returns only the highest scoring element per document, with path and content_location
        updated to match the top-level document. Useful for document-level searches.

    Returns:
    A SearchResults object containing matching elements with:
    - doc_id: Identifier of the containing document
    - element_type: Type of matching element (section, paragraph, list, etc.)
    - text: Full text of the matching element when available
    - content_preview: Preview of the matching content (may be truncated)
    - score: Semantic similarity score (-1 to 1) indicating conceptual relevance
    - path: Full element path showing location in document hierarchy

    Example matches for "customer asking about wire transfer fees":
    - Exact match: Section about wire transfer fee schedule (score: 0.85)
    - Related: Paragraph about international transfer costs (score: 0.65)
    - Broader: Table of all service fees (score: 0.45)
    """
    async def work(_search_for, _limit, _min_score, _include_parents, _resolve_text, _resolve_content, _just_documents, _include_topics, _exclude_topics) -> List[ElementFlat]:
        _span = get_current_span()

        # Set defaults
        if not isinstance(_limit, int):
            _limit = 10

        if not isinstance(_min_score, float):
            _min_score = 0.0

        if not isinstance(_include_parents, bool):
            _include_parents = False

        if not isinstance(_resolve_text, bool):
            _resolve_text = False

        if not isinstance(_resolve_content, bool):
            _resolve_content = False

        if not isinstance(_just_documents, bool):
            _just_documents = False

        if not isinstance(_include_topics, list):
            _include_topics = []
        if not isinstance(_exclude_topics, list):
            _exclude_topics = []

        # Prepare request headers
        headers = {
            'Content-Type': 'application/json'
        }
        if SEARCH_API_KEY:
            headers['X-API-Key'] = SEARCH_API_KEY

        if _just_documents:
            _include_parents = False

        # Prepare request payload
        payload = {
            'query': _search_for,
            'limit': _limit,
            'include_parents': _include_parents,
            'min_score': _min_score,
            'text': _resolve_text,
            'content': _resolve_content,
            'flat': True,
            'include_topics': _include_topics,
            'exclude_topics': _exclude_topics,
        }

        try:
            # Make HTTP request to the search server
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{SEARCH_SERVER_URL}/api/search",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Search server error {response.status}: {error_text}")

                    response_data = await response.json()

        except Exception as e:
            _span.set_attribute("error", str(e))
            _span.set_attribute("search_error", "HTTP request failed")
            raise Exception(f"Failed to search documents: {str(e)}")

        try:
            search_tree = response_data.get('search_tree', [])
            search_tree = [ElementFlat(**item) for item in search_tree]

            # Process results for just_documents if needed
            if _just_documents:
                # Group by document and find highest scoring element
                doc_map = {}
                for element in search_tree:
                    doc_id = element.doc_id

                    if doc_id not in doc_map or (element.score if element.score is not None else -1) > doc_map[doc_id].score:
                        doc_map[doc_id] = element

                # Extract the top elements as a list
                top_elements = list(doc_map.values())

                search_tree = top_elements

            _span.set_attribute("result_count", len(search_tree))
            _span.set_attributes(payload)
            return search_tree

        except Exception as e:
            _span.set_attribute("error", str(e))
            _span.set_attribute("processing_error", "Failed to process search results")
            raise Exception(f"Failed to serialize search documents: {str(e)}")

    return await with_active_span(
        tracer,
        "Search Documents",
        lambda span: work(
            _search_for=search_for, _limit=limit, _min_score=min_score,
            _include_parents=include_parents, _resolve_text=resolve_text,
            _resolve_content=resolve_content, _just_documents=just_documents,
            _include_topics=include_topics, _exclude_topics=exclude_topics),
        {
            "search_for": search_for,
            "limit": str(limit),
            "min_score": str(min_score),
            "resolve_text": str(resolve_text),
            "resolve_content": str(resolve_content),
            "include_parents": str(include_parents),
            "just_documents": str(just_documents),
        })

@connector.register_query
async def search_top_document_matches(
        search_for: str = Field(
            ...,
            description="Natural language text to search with. Can be a question, description, or topic. The search uses semantic similarity, so exact word matches aren't needed."
        ),
        resolve_content: Optional[bool] = Field(
            default=None,
            description="Include complete structured content of matching elements. Useful when document structure (like XML or JSON) contains important context beyond plain text. Defaults to False."
        ),
        resolve_text: Optional[bool] = Field(
            default=True,  # Default to True for this function
            description="Include complete text content of matching elements. Always defaults to True for this query to provide full context of the best matches."
        ),
        limit: Optional[int] = Field(
            default=None,
            description="Maximum results to return. Higher limits find more matches but may include less relevant content. Consider balancing with min_score to maintain relevance quality. Defaults to 10."
        ),
        min_score: Optional[float] = Field(
            default=None,
            description="Semantic similarity threshold (-1 to 1). Higher values ensure closer conceptual matches: 0.7+ for exact concepts, 0.5+ for closely related, 0.3+ for broadly related, 0.1+ for exploratory searches. Defaults to 0."
        ),
        include_topics: Optional[List[str]] = Field(
            default=None,
            description="A list of topics to include in the search. Includes ANY document that matches ANY topic. Uses a LIKE syntax where % matches any text. For example if the document topics included the source you might look for wikipedia articles setting this to: [\"%wikipedia%\"]"
        ),
        exclude_topics: Optional[List[str]] = Field(
            default=None,
            description="A list of topics to include in the search. Excludes ANY document that matches ANY topic"
        )
) -> List[ElementFlat]:
    """
    This function performs a document-level semantic search, returning the best matching element from each document.

    It always sets just_documents=True and calls search_document_detail, which:
    1. Searches at the document level
    2. For each document, selects only the element with the highest score
    3. Standardizes path and content_location to match the top-level document

    This approach is useful when you want document-level results but with the most relevant content
    from each document highlighted, while maintaining a document-oriented result structure.

    Parameters are the same as search_document_detail except:
    - just_documents is always set to True
    - resolve_text defaults to True to provide the complete text of the best matches

    Returns:
    A list of ElementFlat objects representing the top matching element from each document.
    """
    # Simply call the original query with just_documents=True
    return await search_document_detail(
        search_for=search_for,
        include_parents=False,
        just_documents=True,
        resolve_content=resolve_content,
        resolve_text=resolve_text if resolve_text is not None else True,  # Default to True if None
        limit=limit,
        min_score=min_score,
        include_topics=include_topics,
        exclude_topics=exclude_topics,
    )

@connector.register_query
async def search_top_document_matches_with_defaults(
        search_for: str = Field(
            ...,
            description="Natural language text to search with. Uses default settings: resolve_content=True, resolve_text=True, limit=10, min_score=0.3"
        )
) -> List[ElementFlat]:
    return await search_top_document_matches(search_for=search_for, resolve_content=True, resolve_text=True, limit=10, min_score=0.3)


if __name__ == "__main__":
    start(connector)
