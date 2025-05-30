"""
Enhanced DocumentDatabase Abstract Base Class

This module provides the revised abstract base class for document database implementations
with integrated structured search support. All backend implementations must inherit from
this class and implement both legacy and structured search methods.

The class bridges the gap between the original document storage API and the new
structured search system, ensuring backward compatibility while enabling advanced
search capabilities.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

# Import existing element types
from .element_element import ElementBase, ElementType, ElementHierarchical
from .element_relationship import ElementRelationship
# Import the structured search components
from .structured_search import (
    StructuredSearchQuery, BackendCapabilities, SearchCapability,
    validate_query_capabilities
)


class DocumentDatabase(ABC):
    """
    Abstract base class for document database implementations with comprehensive
    search support including both legacy methods and structured search capabilities.
    """

    # ========================================
    # CORE DATABASE OPERATIONS (unchanged)
    # ========================================

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the database."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass

    # ========================================
    # DOCUMENT STORAGE OPERATIONS (unchanged)
    # ========================================

    @abstractmethod
    def store_document(self, document: Dict[str, Any], elements: List[Dict[str, Any]],
                       relationships: List[Dict[str, Any]]) -> None:
        """
        Store a document with its elements and relationships.

        Args:
            document: Document metadata
            elements: Document elements
            relationships: Element relationships
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document metadata or None if not found
        """
        pass

    @abstractmethod
    def get_last_processed_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about when a document was last processed.

        Args:
            source_id: Source identifier for the document

        Returns:
            Dictionary with last_modified and content_hash, or None if not found
        """
        pass

    @abstractmethod
    def update_processing_history(self, source_id: str, content_hash: str) -> None:
        """
        Update the processing history for a document.

        Args:
            source_id: Source identifier for the document
            content_hash: Hash of the document content
        """
        pass

    @abstractmethod
    def get_document_elements(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get elements for a document.

        Args:
            doc_id: Document ID

        Returns:
            List of document elements
        """
        pass

    @abstractmethod
    def get_document_relationships(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get relationships for a document.

        Args:
            doc_id: Document ID

        Returns:
            List of document relationships
        """
        pass

    @abstractmethod
    def get_element(self, element_id: str | int) -> Optional[Dict[str, Any]]:
        """
        Get element by ID.

        Args:
            element_id: Element ID

        Returns:
            Element data or None if not found
        """
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all associated elements and relationships.

        Args:
            doc_id: Document ID

        Returns:
            True if document was deleted, False otherwise
        """
        pass

    # ========================================
    # LEGACY SEARCH METHODS (unchanged but now optional)
    # ========================================

    @abstractmethod
    def find_documents(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find documents matching query with support for LIKE patterns.

        Args:
            query: Query parameters with enhanced syntax support
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        pass

    @abstractmethod
    def find_elements(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements matching query with support for LIKE patterns and ElementType enums.

        Args:
            query: Query parameters with enhanced syntax support
            limit: Maximum number of results

        Returns:
            List of matching elements
        """
        pass

    @abstractmethod
    def search_elements_by_content(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search elements by content preview.

        Args:
            search_text: Text to search for
            limit: Maximum number of results

        Returns:
            List of matching elements
        """
        pass

    # ========================================
    # EMBEDDING SEARCH METHODS (unchanged)
    # ========================================

    @abstractmethod
    def store_embedding(self, element_id: str, embedding: List[float]) -> None:
        """
        Store embedding for an element.

        Args:
            element_id: Element ID
            embedding: Vector embedding
        """
        pass

    @abstractmethod
    def get_embedding(self, element_id: str) -> Optional[List[float]]:
        """
        Get embedding for an element.

        Args:
            element_id: Element ID

        Returns:
            Vector embedding or None if not found
        """
        pass

    @abstractmethod
    def search_by_embedding(self, query_embedding: List[float], limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by embedding similarity with optional filtering.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_criteria: Optional dictionary with criteria to filter results

        Returns:
            List of (element_id, similarity_score) tuples for matching elements
        """
        pass

    @abstractmethod
    def search_by_text(self, search_text: str, limit: int = 10,
                       filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by semantic similarity to the provided text.

        Args:
            search_text: Text to search for semantically
            limit: Maximum number of results
            filter_criteria: Optional dictionary with criteria to filter results

        Returns:
            List of (element_id, similarity_score) tuples
        """
        pass

    @abstractmethod
    def get_outgoing_relationships(self, element_pk: int) -> List[ElementRelationship]:
        """Find all relationships where the specified element_pk is the source."""
        pass

    # ========================================
    # DATE STORAGE AND SEARCH METHODS (unchanged)
    # ========================================

    @abstractmethod
    def store_element_dates(self, element_id: str, dates: List[Dict[str, Any]]) -> None:
        """
        Store extracted dates associated with an element.

        Args:
            element_id: Element ID
            dates: List of date dictionaries from ExtractedDate.to_dict()
        """
        pass

    @abstractmethod
    def get_element_dates(self, element_id: str) -> List[Dict[str, Any]]:
        """
        Get all dates associated with an element.

        Args:
            element_id: Element ID

        Returns:
            List of date dictionaries, empty list if none found
        """
        pass

    @abstractmethod
    def store_embedding_with_dates(self, element_id: str, embedding: List[float],
                                   dates: List[Dict[str, Any]]) -> None:
        """
        Store both embedding and dates for an element in a single operation.

        Args:
            element_id: Element ID
            embedding: Vector embedding
            dates: List of extracted date dictionaries
        """
        pass

    @abstractmethod
    def delete_element_dates(self, element_id: str) -> bool:
        """
        Delete all dates associated with an element.

        Args:
            element_id: Element ID

        Returns:
            True if dates were deleted, False if none existed
        """
        pass

    @abstractmethod
    def search_elements_by_date_range(self, start_date: datetime, end_date: datetime,
                                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements that contain dates within a specified range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            limit: Maximum number of results

        Returns:
            List of element dictionaries that contain dates in the range
        """
        pass

    @abstractmethod
    def search_by_text_and_date_range(self,
                                      search_text: str,
                                      start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None,
                                      limit: int = 10) -> List[Tuple[int, float]]:
        """
        Search elements by semantic similarity AND date range.

        Args:
            search_text: Text to search for semantically
            start_date: Optional start of date range
            end_date: Optional end of date range
            limit: Maximum number of results

        Returns:
            List of (element_id, similarity_score) tuples
        """
        pass

    @abstractmethod
    def search_by_embedding_and_date_range(self,
                                           query_embedding: List[float],
                                           start_date: Optional[datetime] = None,
                                           end_date: Optional[datetime] = None,
                                           limit: int = 10) -> List[Tuple[int, float]]:
        """
        Search elements by embedding similarity AND date range.

        Args:
            query_embedding: Query embedding vector
            start_date: Optional start of date range
            end_date: Optional end of date range
            limit: Maximum number of results

        Returns:
            List of (element_id, similarity_score) tuples
        """
        pass

    @abstractmethod
    def get_elements_with_dates(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all elements that have associated dates.

        Args:
            limit: Maximum number of results

        Returns:
            List of element dictionaries that have dates
        """
        pass

    @abstractmethod
    def get_date_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about dates in the database.

        Returns:
            Dictionary with date statistics
        """
        pass

    # ========================================
    # NEW: STRUCTURED SEARCH SYSTEM (required for all backends)
    # ========================================

    @abstractmethod
    def get_backend_capabilities(self) -> BackendCapabilities:
        """
        Return the capabilities supported by this backend.

        This method must be implemented by each backend to declare what
        search features it supports. The structured search system uses
        this information to validate queries and provide appropriate
        error messages.

        Returns:
            BackendCapabilities object describing supported features

        Example:
            ```python
            def get_backend_capabilities(self) -> BackendCapabilities:
                supported = {
                    SearchCapability.TEXT_SIMILARITY,
                    SearchCapability.DATE_FILTERING,
                    SearchCapability.LOGICAL_AND,
                    SearchCapability.LOGICAL_OR,
                }
                return BackendCapabilities(supported)
            ```
        """
        pass

    @abstractmethod
    def execute_structured_search(self, query: StructuredSearchQuery) -> List[Dict[str, Any]]:
        """
        Execute a structured search query.

        This is the main entry point for the new structured search system.
        Backends must implement this method to handle complex search queries
        with logical operators, multiple criteria types, and custom scoring.

        Args:
            query: Structured search query object containing all search criteria

        Returns:
            List of search results with the following structure:
            [
                {
                    'element_pk': int,
                    'element_id': str,
                    'doc_id': str,
                    'element_type': str,
                    'content_preview': str,
                    'final_score': float,
                    'scores': Dict[str, float],  # Individual score components
                    'metadata': Dict[str, Any],  # If include_metadata=True
                    'topics': List[str],         # If include_topics=True
                    'extracted_dates': List[Dict[str, Any]],  # If include_element_dates=True
                }
            ]

        Raises:
            UnsupportedSearchError: If query uses unsupported capabilities

        Implementation Notes:
            - Backends should first validate the query using validate_query_support()
            - Results should be sorted by final_score in descending order
            - Honor the limit and offset parameters in the query
            - Include additional fields based on query configuration flags
        """
        pass

    def validate_query_support(self, query: StructuredSearchQuery) -> List[SearchCapability]:
        """
        Validate that this backend can execute the given query.

        This method analyzes the query to determine what capabilities are required
        and compares them against the backend's declared capabilities.

        Args:
            query: Structured search query to validate

        Returns:
            List of missing capabilities (empty if fully supported)

        Example:
            ```python
            missing = db.validate_query_support(complex_query)
            if missing:
                print(f"Cannot execute query. Missing: {[c.value for c in missing]}")
            else:
                results = db.execute_structured_search(complex_query)
            ```
        """
        return validate_query_capabilities(query, self.get_backend_capabilities())

    def is_query_supported(self, query: StructuredSearchQuery) -> bool:
        """
        Check if a query is fully supported by this backend.

        Args:
            query: Structured search query to check

        Returns:
            True if query is fully supported, False otherwise
        """
        return len(self.validate_query_support(query)) == 0

    def get_supported_capabilities_list(self) -> List[str]:
        """
        Get a list of capability names supported by this backend.

        Returns:
            List of capability names as strings
        """
        return self.get_backend_capabilities().get_supported_list()

    # ========================================
    # ENHANCED CONVENIENCE METHODS
    # ========================================

    def unified_search(self,
                      search_text: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      element_types: Optional[List[str]] = None,
                      doc_ids: Optional[List[str]] = None,
                      include_topics: Optional[List[str]] = None,
                      exclude_topics: Optional[List[str]] = None,
                      metadata_filters: Optional[Dict[str, Any]] = None,
                      limit: int = 10,
                      include_element_dates: bool = False) -> List[Dict[str, Any]]:
        """
        Unified search method that builds and executes a structured query.

        This is a convenience method that builds a StructuredSearchQuery from
        simple parameters and executes it. For more complex queries with nested
        logic, use SearchQueryBuilder directly.

        Args:
            search_text: Optional text for semantic similarity search
            start_date: Optional start of date range filter
            end_date: Optional end of date range filter
            element_types: Optional list of element types to filter by
            doc_ids: Optional list of document IDs to filter by
            include_topics: Optional topics to include (LIKE patterns)
            exclude_topics: Optional topics to exclude (LIKE patterns)
            metadata_filters: Optional metadata key-value filters
            limit: Maximum number of results
            include_element_dates: Whether to include extracted dates in results

        Returns:
            List of search results

        Example:
            ```python
            results = db.unified_search(
                search_text="machine learning",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                element_types=["header", "paragraph"],
                limit=20
            )
            ```
        """
        from .structured_search import SearchQueryBuilder

        builder = SearchQueryBuilder()

        # Add text search if provided
        if search_text:
            builder.text_search(search_text)

        # Add date range if provided
        if start_date and end_date:
            builder.date_range(start_date, end_date)
        elif start_date:
            builder.date_after(start_date)
        elif end_date:
            builder.date_before(end_date)

        # Add element type filter
        if element_types:
            builder.element_types(element_types)

        # Add document ID filter
        if doc_ids:
            builder.doc_ids(doc_ids)

        # Add topic filters
        if include_topics or exclude_topics:
            builder.topics(include=include_topics, exclude=exclude_topics)

        # Add metadata filters
        if metadata_filters:
            builder.metadata_exact(**metadata_filters)

        # Configure result options
        builder.limit(limit)
        if include_element_dates:
            builder.include_dates(True)

        # Build and execute query
        query = builder.build()
        return self.execute_structured_search(query)

    def search_with_date_range(self, search_text: str, start_date: datetime,
                              end_date: datetime, **kwargs) -> List[Dict[str, Any]]:
        """
        Convenience method for text search with date range.

        Args:
            search_text: Text to search for
            start_date: Start of date range
            end_date: End of date range
            **kwargs: Additional parameters for unified_search

        Returns:
            List of search results
        """
        return self.unified_search(
            search_text=search_text,
            start_date=start_date,
            end_date=end_date,
            include_element_dates=True,
            **kwargs
        )

    def search_recent_content(self, search_text: str, days_back: int = 30,
                             **kwargs) -> List[Dict[str, Any]]:
        """
        Search for content from the last N days.

        Args:
            search_text: Text to search for
            days_back: Number of days to look back
            **kwargs: Additional parameters for unified_search

        Returns:
            List of search results
        """
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        return self.unified_search(
            search_text=search_text,
            start_date=start_date,
            end_date=end_date,
            include_element_dates=True,
            **kwargs
        )

    def search_by_year(self, search_text: str, year: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for content from a specific year.

        Args:
            search_text: Text to search for
            year: Year to search in
            **kwargs: Additional parameters for unified_search

        Returns:
            List of search results
        """
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)

        return self.unified_search(
            search_text=search_text,
            start_date=start_date,
            end_date=end_date,
            include_element_dates=True,
            **kwargs
        )

    def search_quarterly_content(self, search_text: str, year: int, quarter: int,
                               **kwargs) -> List[Dict[str, Any]]:
        """
        Search for content from a specific quarter.

        Args:
            search_text: Text to search for
            year: Year
            quarter: Quarter (1-4)
            **kwargs: Additional parameters for unified_search

        Returns:
            List of search results
        """
        quarter_starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
        quarter_ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}

        start_month, start_day = quarter_starts[quarter]
        end_month, end_day = quarter_ends[quarter]

        start_date = datetime(year, start_month, start_day)
        end_date = datetime(year, end_month, end_day, 23, 59, 59)

        return self.unified_search(
            search_text=search_text,
            start_date=start_date,
            end_date=end_date,
            include_element_dates=True,
            **kwargs
        )

    # ========================================
    # EXISTING HELPER METHODS (unchanged)
    # ========================================

    @staticmethod
    def supports_like_patterns() -> bool:
        """
        Indicate whether this backend supports LIKE pattern matching.

        Returns:
            True if LIKE patterns are supported, False otherwise
        """
        return True  # Default: assume LIKE support

    @staticmethod
    def supports_case_insensitive_like() -> bool:
        """
        Indicate whether this backend supports case-insensitive LIKE (ILIKE).

        Returns:
            True if ILIKE patterns are supported, False otherwise
        """
        return False  # Default: assume no ILIKE support

    @staticmethod
    def supports_element_type_enums() -> bool:
        """
        Indicate whether this backend supports ElementType enum integration.

        Returns:
            True if ElementType enums are supported, False otherwise
        """
        return True  # Default: assume enum support

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
        """
        categories = self.get_element_types_by_category()

        if category not in categories:
            available = list(categories.keys())
            raise ValueError(f"Unknown category: {category}. Available: {available}")

        element_types = categories[category]
        query = {"element_type": element_types}
        query.update(other_filters)

        return self.find_elements(query)

    # ========================================
    # TOPIC SUPPORT METHODS (unchanged)
    # ========================================

    def supports_topics(self) -> bool:
        """
        Indicate whether this backend supports topic-aware embeddings.

        Returns:
            True if topics are supported, False otherwise
        """
        return SearchCapability.TOPIC_FILTERING in self.get_backend_capabilities().supported

    def store_embedding_with_topics(self, element_pk: int, embedding: List[float],
                                    topics: List[str], confidence: float = 1.0) -> None:
        """
        Store embedding for an element with topic assignments.

        Default implementation falls back to regular embedding storage.
        Backends with topic support should override this method.

        Args:
            element_pk: Element primary key
            embedding: Vector embedding
            topics: List of topic strings
            confidence: Overall confidence in this embedding/topic assignment
        """
        # Default implementation: fallback to regular embedding storage
        self.store_embedding(element_pk, embedding)

    def search_by_text_and_topics(self, search_text: str = None,
                                  include_topics: Optional[List[str]] = None,
                                  exclude_topics: Optional[List[str]] = None,
                                  min_confidence: float = 0.7,
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search elements by text with topic filtering.

        Default implementation falls back to regular text search.
        Backends with topic support should override this method.

        Args:
            search_text: Text to search for semantically
            include_topics: Topic LIKE patterns to include
            exclude_topics: Topic LIKE patterns to exclude
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results

        Returns:
            List of search results
        """
        # Default implementation: fallback to regular text search
        if search_text:
            results = self.search_by_text(search_text, limit)
            return [
                {
                    'element_pk': element_pk,
                    'similarity': similarity,
                    'confidence': 1.0,
                    'topics': []
                }
                for element_pk, similarity in results
            ]
        else:
            return []

    def get_topic_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about topic distribution across embeddings.

        Default implementation returns empty statistics.
        Backends with topic support should override this method.

        Returns:
            Dictionary mapping topic strings to statistics
        """
        return {}

    def get_embedding_topics(self, element_pk: int) -> List[str]:
        """
        Get topics assigned to a specific embedding.

        Default implementation returns empty list.
        Backends with topic support should override this method.

        Args:
            element_pk: Element primary key

        Returns:
            List of topic strings assigned to this embedding
        """
        return []

    # ========================================
    # DATE UTILITY METHODS (unchanged)
    # ========================================

    def supports_date_storage(self) -> bool:
        """
        Indicate whether this backend supports date storage.

        Returns:
            True if date storage is supported, False otherwise
        """
        return SearchCapability.DATE_FILTERING in self.get_backend_capabilities().supported

    def get_date_range_for_element(self, element_id: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Get the date range (earliest, latest) for an element.

        Args:
            element_id: Element ID

        Returns:
            Tuple of (earliest_date, latest_date) or None if no dates
        """
        dates = self.get_element_dates(element_id)
        if not dates:
            return None

        timestamps = [d['timestamp'] for d in dates if 'timestamp' in d and d['timestamp'] is not None]
        if not timestamps:
            return None

        earliest = datetime.fromtimestamp(min(timestamps))
        latest = datetime.fromtimestamp(max(timestamps))
        return earliest, latest

    def count_dates_for_element(self, element_id: str) -> int:
        """
        Count the number of dates associated with an element.

        Args:
            element_id: Element ID

        Returns:
            Number of dates associated with the element
        """
        dates = self.get_element_dates(element_id)
        return len(dates)

    def get_elements_by_year(self, year: int, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get elements that contain dates from a specific year.

        Args:
            year: Year to search for
            limit: Maximum number of results

        Returns:
            List of element dictionaries
        """
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        return self.search_elements_by_date_range(start_date, end_date, limit)

    def get_elements_by_month(self, year: int, month: int, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get elements that contain dates from a specific month.

        Args:
            year: Year
            month: Month (1-12)
            limit: Maximum number of results

        Returns:
            List of element dictionaries
        """
        import calendar
        start_date = datetime(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        end_date = datetime(year, month, last_day, 23, 59, 59)
        return self.search_elements_by_date_range(start_date, end_date, limit)

    def update_element_dates(self, element_id: str, dates: List[Dict[str, Any]]) -> None:
        """
        Update dates for an element (delete old, store new).

        Args:
            element_id: Element ID
            dates: New list of date dictionaries
        """
        self.delete_element_dates(element_id)
        self.store_element_dates(element_id, dates)

    # ========================================
    # HIERARCHY METHODS (unchanged)
    # ========================================

    def get_results_outline(self, elements: List[Tuple[int, float]]) -> List[ElementHierarchical]:
        """
        For search results, create a hierarchical outline showing element ancestry.

        Args:
            elements: List of (element_pk, score) tuples from search results

        Returns:
            List of ElementHierarchical objects representing the hierarchy
        """
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

    def _get_element_ancestry_path(self, element_pk: int) -> List[ElementBase]:
        """
        Get the complete ancestry path for an element, from root to the element itself.

        Args:
            element_pk: Element primary key

        Returns:
            List of ElementBase objects representing the ancestry path
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
