import json
import logging
import os
from typing import List, Optional, Dict, Any, Tuple, Set, Union

from pydantic import BaseModel, Field, PrivateAttr

from .adapter import create_content_resolver, ContentResolver
from .config import Config
from .storage import ElementRelationship, DocumentDatabase, ElementHierarchical, ElementFlat, flatten_hierarchy
# Import the Pydantic models
from .storage.search import (
    SearchQueryRequest,
    SearchCriteriaGroupRequest,
    SemanticSearchRequest,
    TopicSearchRequest,
    DateSearchRequest,
    ElementSearchRequest,
    # SearchResultItem,
    LogicalOperatorEnum,
    DateRangeOperatorEnum
)

logger = logging.getLogger(__name__)

_config = Config(os.environ.get('DOCULYZER_CONFIG_PATH', 'config.yaml'))


class SearchResultItem(BaseModel):
    """Pydantic model for a single search result item."""
    element_pk: int
    similarity: float
    confidence: Optional[float] = None  # For topic search results
    topics: Optional[List[str]] = None  # For topic search results
    _db: Optional[DocumentDatabase] = PrivateAttr()
    _resolver: Optional[ContentResolver] = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._db = _config.get_document_database()
        self._resolver = create_content_resolver(_config)

    @property
    def doc_id(self) -> Optional[str]:
        return self._db.get_element(self.element_pk).get("doc_id", None)

    @property
    def element_id(self) -> Optional[str]:
        return self._db.get_element(self.element_pk).get("element_id", None)

    @property
    def element_type(self) -> Optional[str]:
        return self._db.get_element(self.element_pk).get("element_type", None)

    @property
    def parent_id(self) -> Optional[str]:
        return self._db.get_element(self.element_pk).get("parent_id", None)

    @property
    def content_preview(self) -> Optional[str]:
        return self._db.get_element(self.element_pk).get("content_preview", None)

    @property
    def metadata(self) -> Optional[dict]:
        m = self._db.get_element(self.element_pk).get("metadata")
        if m is None:
            return {}
        if isinstance(m, str):
            json.loads(m)
        if isinstance(m, dict):
            return m

    @property
    def content(self) -> Optional[str]:
        """
        A dynamic property that calls resolver.resolve_content() to return its value.
        """
        if self._resolver and self.element_pk:
            return self._resolver.resolve_content(self._db.get_element(self.element_pk).get("content_location"),
                                                  text=False)
        return None

    @property
    def text(self) -> Optional[str]:
        """
        A dynamic property that calls resolver.resolve_content() to return its value.
        """
        if self._resolver and self.element_pk:
            return self._resolver.resolve_content(self._db.get_element(self.element_pk).get("content_location"),
                                                  text=True)
        return None


class SearchResults(BaseModel):
    """Pydantic model for search results collection."""
    results: List[SearchResultItem] = Field(default_factory=list)
    total_results: int = 0
    query: Optional[str] = None
    filter_criteria: Optional[Dict[str, Any]] = None
    # Topic filtering criteria
    include_topics: Optional[List[str]] = None
    exclude_topics: Optional[List[str]] = None
    min_confidence: Optional[float] = None
    search_type: str = "embedding"  # Can be "embedding", "text", "content", "topic", "structured"
    min_score: float = 0.0  # Minimum score threshold used
    documents: List[str] = Field(default_factory=list)  # Unique list of document sources from the results
    search_tree: Optional[List[ElementHierarchical | ElementFlat]] = None
    # Track whether content was resolved during search
    content_resolved: bool = False
    text_resolved: bool = False
    # Topic-related metadata
    supports_topics: bool = False
    topic_statistics: Optional[Dict[str, Any]] = None
    # Structured search metadata
    query_id: Optional[str] = None
    execution_time_ms: Optional[float] = None

    @classmethod
    def from_tuples(cls, tuples: List[Tuple[int, float]],
                    flat: bool = False,
                    include_parents: bool = True,
                    query: Optional[str] = None,
                    filter_criteria: Optional[Dict[str, Any]] = None,
                    include_topics: Optional[List[str]] = None,
                    exclude_topics: Optional[List[str]] = None,
                    min_confidence: Optional[float] = None,
                    search_type: str = "embedding",
                    min_score: float = 0.0,
                    search_tree: Optional[List[ElementHierarchical]] = None,
                    documents: Optional[List[str]] = None,
                    content_resolved: bool = False,
                    text_resolved: bool = False,
                    supports_topics: bool = False,
                    topic_statistics: Optional[Dict[str, Any]] = None,
                    query_id: Optional[str] = None,
                    execution_time_ms: Optional[float] = None) -> "SearchResults":
        """
        Create a SearchResults object from a list of (element_pk, similarity) tuples.

        Args:
            query_id
            execution_time_ms
            flat
            include_parents
            tuples: List of (element_pk, similarity) tuples
            query: Optional query string that produced these results
            filter_criteria: Optional dictionary of filter criteria
            include_topics: Topic patterns that were included
            exclude_topics: Topic patterns that were excluded
            min_confidence: Minimum confidence threshold for topic results
            search_type: Type of search performed
            min_score: Minimum score threshold used
            documents: List of unique document sources
            search_tree: Optional tree structure representing the search results
            content_resolved: Whether content was resolved during search
            text_resolved: Whether text was resolved during search
            supports_topics: Whether the backend supports topics
            topic_statistics: Topic distribution statistics

        Returns:
            SearchResults object
        """
        results = [SearchResultItem(element_pk=pk, similarity=similarity) for pk, similarity in tuples]
        if flat and include_parents:
            s = flatten_hierarchy(search_tree)
        elif flat and not include_parents:
            s = [r for r in flatten_hierarchy(search_tree) if r.score is not None]
        else:
            s = search_tree or []
        return cls(
            results=results,
            total_results=len(results),
            query=query,
            filter_criteria=filter_criteria,
            include_topics=include_topics,
            exclude_topics=exclude_topics,
            min_confidence=min_confidence,
            search_type=search_type,
            min_score=min_score,
            documents=documents or [],
            search_tree=s,
            content_resolved=content_resolved,
            text_resolved=text_resolved,
            supports_topics=supports_topics,
            topic_statistics=topic_statistics,
            query_id=query_id,
            execution_time_ms=execution_time_ms
        )


class SearchResult(BaseModel):
    """Pydantic model for storing search result data in a flat structure with relationships."""
    # Similarity score
    similarity: float
    # Topic fields (optional)
    confidence: Optional[float] = None
    topics: Optional[List[str]] = None

    # Element fields
    element_pk: int = Field(default=-1,
                            title="Element primary key, used to get additional information about an element.")
    element_id: str = Field(default="", title="Element natural key.")
    element_type: str = Field(default="", title="Element type.",
                              examples=["body", "div", "header", "table", "table_row"])
    content_preview: Optional[str] = Field(default=None,
                                           title="Short version of the element's content, used for previewing.")
    content_location: Optional[str] = Field(default=None,
                                            title="URI to the location of element's content, if available.")

    # Document fields
    doc_id: str = Field(default="", title="Document natural key.")
    doc_type: str = Field(default="", title="Document type.", examples=["pdf", "docx", "html", "text", "markdown"])
    source: Optional[str] = Field(default=None, title="URI to the original document source, if available.")

    # Outgoing relationships
    outgoing_relationships: List[ElementRelationship] = Field(default_factory=list)

    # Resolved content
    resolved_content: Optional[str] = None
    resolved_text: Optional[str] = None

    # Error information (if content resolution fails)
    resolution_error: Optional[str] = None

    def get_relationship_count(self) -> int:
        """Get the number of outgoing relationships for this element."""
        return len(self.outgoing_relationships)

    def get_relationships_by_type(self) -> Dict[str, List[ElementRelationship]]:
        """Group outgoing relationships by relationship type."""
        result = {}
        for rel in self.outgoing_relationships:
            rel_type = rel.relationship_type
            if rel_type not in result:
                result[rel_type] = []
            result[rel_type].append(rel)
        return result

    def get_contained_elements(self) -> List[ElementRelationship]:
        """Get elements that this element contains (container relationships)."""
        container_types = ["contains", "contains_row", "contains_cell", "contains_item"]
        return [rel for rel in self.outgoing_relationships if rel.relationship_type in container_types]

    def get_linked_elements(self) -> List[ElementRelationship]:
        """Get elements that this element links to (explicit links)."""
        return [rel for rel in self.outgoing_relationships if rel.relationship_type == "link"]

    def get_semantic_relationships(self) -> List[ElementRelationship]:
        """Get elements that are semantically similar to this element."""
        return [rel for rel in self.outgoing_relationships if rel.relationship_type == "semantic_similarity"]


class SearchHelper:
    """Helper class for semantic search operations with singleton pattern."""

    _instance = None
    _db = None
    _content_resolver = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(SearchHelper, cls).__new__(cls)
            cls._initialize_dependencies()
        return cls._instance

    @classmethod
    def _initialize_dependencies(cls):
        """Initialize database and content resolver if not already initialized."""
        if cls._db is None:
            cls._db = _config.get_document_database()
            cls._db.initialize()
            logger.info("Database initialized as singleton")

        if cls._content_resolver is None:
            cls._content_resolver = create_content_resolver(_config)
            logger.info("Content resolver initialized as singleton")

    @classmethod
    def get_database(cls):
        """Get the singleton database instance."""
        if cls._db is None:
            cls._initialize_dependencies()
        return cls._db

    @classmethod
    def get_content_resolver(cls):
        """Get the singleton content resolver instance."""
        if cls._content_resolver is None:
            cls._initialize_dependencies()
        return cls._content_resolver

    # NEW STRUCTURED SEARCH METHODS

    @classmethod
    def execute_structured_search(cls, query: SearchQueryRequest,
                                  text: bool = False,
                                  content: bool = False,
                                  flat: bool = False,
                                  include_parents: bool = True) -> SearchResults:
        """
        Execute a structured search using Pydantic models with SearchHelper enhancements.

        Args:
            query: SearchQueryRequest object with structured search criteria
            text: Whether to resolve text content for results
            content: Whether to resolve content for results
            flat: Whether to return flat results
            include_parents: Whether to include parent elements

        Returns:
            SearchResults object with results, search tree, and materialized content
        """
        # Ensure database is initialized
        db = cls.get_database()
        resolver = cls.get_content_resolver()

        logger.debug(f"Executing structured search with query ID: {query.query_id}")

        try:
            # Import the execute_search function from pydantic_search
            from .pydantic_search import execute_search

            # Execute the search using the existing structured search system
            pydantic_response = execute_search(query, db, validate_capabilities=True)

            if not pydantic_response.success:
                logger.error(f"Structured search failed: {pydantic_response.error_message}")
                return SearchResults(
                    results=[],
                    total_results=0,
                    search_type="structured",
                    query_id=query.query_id,
                    execution_time_ms=pydantic_response.execution_time_ms
                )

            # Convert Pydantic results to tuples for SearchHelper processing
            result_tuples = [(item.element_pk, item.final_score) for item in pydantic_response.results]

            # Build search tree and resolve content if requested (SearchHelper value-add)
            def resolve_elements(items: List[ElementHierarchical]):
                for item in items:
                    if item.child_elements:
                        resolve_elements(item.child_elements)
                    if text and item.content_location:
                        try:
                            item.text = resolver.resolve_content(item.content_location, text=True)
                        except Exception as e:
                            logger.warning(f"Failed to resolve text for {item.content_location}: {e}")
                    if content and item.content_location:
                        try:
                            item.content = resolver.resolve_content(item.content_location, text=False)
                        except Exception as e:
                            logger.warning(f"Failed to resolve content for {item.content_location}: {e}")

            # Get document outline/hierarchy
            search_tree = db.get_results_outline(result_tuples) if result_tuples else []

            # Resolve content if requested
            if text or content:
                resolve_elements(search_tree)

            # Get document sources for these elements
            document_sources = cls._get_document_sources_for_elements([pk for pk, _ in result_tuples])

            # Convert SearchResultItems from Pydantic format
            search_result_items = []
            for pydantic_item in pydantic_response.results:
                search_item = SearchResultItem(
                    element_pk=pydantic_item.element_pk,
                    similarity=pydantic_item.final_score,
                    confidence=getattr(pydantic_item, 'confidence', None),
                    topics=getattr(pydantic_item, 'topics', None)
                )
                search_result_items.append(search_item)

            # Extract query text from criteria group for logging
            query_text = cls._extract_query_text_from_request(query)

            # Handle flat vs hierarchical results
            if flat and include_parents:
                final_search_tree = flatten_hierarchy(search_tree)
            elif flat and not include_parents:
                final_search_tree = [r for r in flatten_hierarchy(search_tree) if r.score is not None]
            else:
                final_search_tree = search_tree

            return SearchResults(
                results=search_result_items,
                total_results=pydantic_response.total_results,
                query=query_text,
                search_type="structured",
                documents=document_sources,
                search_tree=final_search_tree,
                query_id=pydantic_response.query_id,
                execution_time_ms=pydantic_response.execution_time_ms,
                content_resolved=content,
                text_resolved=text,
                supports_topics=db.supports_topics()
            )

        except ImportError:
            logger.error("Pydantic search module not available")
            return SearchResults(
                results=[],
                total_results=0,
                search_type="structured",
                query_id=query.query_id,
                execution_time_ms=None
            )
        except Exception as e:
            logger.error(f"Error executing structured search: {str(e)}")
            return SearchResults(
                results=[],
                total_results=0,
                search_type="structured",
                query_id=query.query_id,
                execution_time_ms=None
            )

    # ENHANCED CONVENIENCE METHODS

    @classmethod
    def search_structured(cls, query: Union[SearchQueryRequest, Dict[str, Any]],
                          text: bool = False,
                          content: bool = False,
                          flat: bool = False,
                          include_parents: bool = True) -> SearchResults:
        """
        Convenience method for structured search that accepts either Pydantic model or dict.

        Args:
            query: SearchQueryRequest object or dictionary that can be converted to one
            text: Whether to resolve text content for results
            content: Whether to resolve content for results
            flat: Whether to return flat results
            include_parents: Whether to include parent elements

        Returns:
            SearchResults object
        """
        if isinstance(query, dict):
            query = SearchQueryRequest.model_validate(query)

        return cls.execute_structured_search(query, text=text, content=content,
                                             flat=flat, include_parents=include_parents)

    @classmethod
    def search_simple_structured(cls,
                                 query_text: str,
                                 limit: int = 10,
                                 similarity_threshold: float = 0.7,
                                 include_topics: Optional[List[str]] = None,
                                 exclude_topics: Optional[List[str]] = None,
                                 days_back: Optional[int] = None,
                                 element_types: Optional[List[str]] = None,
                                 text: bool = False,
                                 content: bool = False,
                                 flat: bool = False,
                                 include_parents: bool = True) -> SearchResults:
        """
        Create and execute a simple structured search query with content materialization.

        Args:
            query_text: Natural language search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            include_topics: Topic patterns to include
            exclude_topics: Topic patterns to exclude
            days_back: Filter to documents from last N days
            element_types: Filter by element types
            text: Whether to resolve text content for results
            content: Whether to resolve content for results
            flat: Whether to return flat results
            include_parents: Whether to include parent elements

        Returns:
            SearchResults object with materialized content and search tree
        """
        # Build criteria group
        criteria_group = SearchCriteriaGroupRequest(
            operator=LogicalOperatorEnum.AND,
            semantic_search=SemanticSearchRequest(
                query_text=query_text,
                similarity_threshold=similarity_threshold
            )
        )

        # Add topic search if specified
        if include_topics or exclude_topics:
            criteria_group.topic_search = TopicSearchRequest(
                include_topics=include_topics or [],
                exclude_topics=exclude_topics or []
            )

        # Add date search if specified
        if days_back:
            criteria_group.date_search = DateSearchRequest(
                operator=DateRangeOperatorEnum.RELATIVE_DAYS,
                relative_value=days_back
            )

        # Add element search if specified
        if element_types:
            criteria_group.element_search = ElementSearchRequest(
                element_types=element_types
            )

        # Create and execute query
        query = SearchQueryRequest(
            criteria_group=criteria_group,
            limit=limit,
            include_similarity_scores=True
        )

        return cls.execute_structured_search(query, text=text, content=content,
                                             flat=flat, include_parents=include_parents)

    @classmethod
    def _extract_query_text_from_request(cls, query: SearchQueryRequest) -> Optional[str]:
        """Extract query text from SearchQueryRequest for logging."""
        return cls._extract_query_text_from_criteria_group(query.criteria_group)

    @classmethod
    def _extract_query_text_from_criteria_group(cls, criteria_group: SearchCriteriaGroupRequest) -> Optional[str]:
        """Extract query text from criteria group for logging."""
        if criteria_group.semantic_search:
            return criteria_group.semantic_search.query_text

        for sub_group in criteria_group.sub_groups:
            text = cls._extract_query_text_from_criteria_group(sub_group)
            if text:
                return text

        return None

    # ORIGINAL METHODS (kept for backward compatibility)

    @classmethod
    def search_by_text(
            cls,
            query_text: str,
            limit: int = 10,
            filter_criteria: Dict[str, Any] = None,
            include_topics: Optional[List[str]] = None,
            exclude_topics: Optional[List[str]] = None,
            min_confidence: Optional[float] = None,
            min_score: float = 0.0,
            text: bool = False,
            content: bool = False,
            flat: bool = False,
            include_parents: bool = True,
    ) -> SearchResults:
        """
        Search for elements similar to the query text and return raw results.

        Args:
            query_text: The text to search for
            limit: Maximum number of results to return
            filter_criteria: Optional filtering criteria for the search
            include_topics: Topic LIKE patterns to include (e.g., ['security%', '%.policy%'])
            exclude_topics: Topic LIKE patterns to exclude (e.g., ['deprecated%'])
            min_confidence: Minimum confidence threshold for topic results
            min_score: Minimum similarity score threshold (default 0.0)
            text: Whether to resolve text content for results
            content: Whether to resolve content for results
            flat: Whether to return flat results
            include_parents: Whether to include parent elements

        Returns:
            SearchResults object with element_pk and similarity scores
        """
        # Ensure database is initialized
        db = cls.get_database()
        resolver = cls.get_content_resolver()

        logger.debug(f"Searching for text: {query_text} with min_score: {min_score}")

        # Check if topic filtering is requested and supported
        supports_topics = db.supports_topics()
        using_topics = include_topics or exclude_topics or min_confidence is not None

        if using_topics and supports_topics:
            # Use topic-aware search
            logger.debug(
                f"Using topic search - include: {include_topics}, exclude: {exclude_topics}, min_confidence: {min_confidence}")

            topic_results = db.search_by_text_and_topics(
                search_text=query_text,
                include_topics=include_topics,
                exclude_topics=exclude_topics,
                min_confidence=min_confidence or 0.7,
                limit=limit
            )

            # Convert topic results to tuples format for search tree generation
            filtered_elements = [(result['element_pk'], result.get('similarity', 1.0)) for result in topic_results]

            # Build search tree and resolve content if requested
            def resolve_elements(items: List[ElementHierarchical]):
                for item in items:
                    if item.child_elements:
                        resolve_elements(item.child_elements)
                    if text:
                        item.text = resolver.resolve_content(item.content_location, text=True)
                    if content:
                        item.content = resolver.resolve_content(item.content_location, text=False)

            search_tree = db.get_results_outline(filtered_elements)
            resolve_elements(search_tree)

            # Get document sources and topic statistics
            element_pks = [result['element_pk'] for result in topic_results]
            document_sources = cls._get_document_sources_for_elements(element_pks)
            topic_statistics = db.get_topic_statistics()

            # Create SearchResultItems with topic information
            results = []
            for result in topic_results:
                item = SearchResultItem(
                    element_pk=result['element_pk'],
                    similarity=result.get('similarity', 1.0),
                    confidence=result.get('confidence'),
                    topics=result.get('topics', [])
                )
                results.append(item)

            return SearchResults(
                results=results,
                total_results=len(results),
                query=query_text,
                filter_criteria=filter_criteria,
                include_topics=include_topics,
                exclude_topics=exclude_topics,
                min_confidence=min_confidence,
                search_type="topic",
                min_score=min_score,
                documents=document_sources,
                search_tree=flatten_hierarchy(search_tree) if flat and include_parents else [r for r in
                                                                                             flatten_hierarchy(
                                                                                                 search_tree) if
                                                                                             r.score is not None] if flat and not include_parents else search_tree or [],
                content_resolved=content,
                text_resolved=text,
                supports_topics=supports_topics,
                topic_statistics=topic_statistics
            )
        else:
            # Use regular text search
            if using_topics and not supports_topics:
                logger.warning("Topic filtering requested but not supported by database backend")

            # Perform the regular search
            similar_elements = db.search_by_text(query_text, limit=limit * 2, filter_criteria=filter_criteria)
            logger.info(f"Found {len(similar_elements)} similar elements before score filtering")

            # Filter by minimum score
            filtered_elements = [elem for elem in similar_elements if elem[1] >= min_score]
            logger.info(f"Found {len(filtered_elements)} elements after score filtering (threshold: {min_score})")

            # Apply limit after filtering
            filtered_elements = filtered_elements[:limit]

            def resolve_elements(items: List[ElementHierarchical]):
                for item in items:
                    if item.child_elements:
                        resolve_elements(item.child_elements)
                    if text:
                        item.text = resolver.resolve_content(item.content_location, text=True)
                    if content:
                        item.content = resolver.resolve_content(item.content_location, text=False)

            search_tree = db.get_results_outline(filtered_elements)
            resolve_elements(search_tree)

            # Get document sources for these elements
            document_sources = cls._get_document_sources_for_elements([pk for pk, _ in filtered_elements])

            # Convert to SearchResults
            return SearchResults.from_tuples(
                tuples=filtered_elements,
                query=query_text,
                filter_criteria=filter_criteria,
                include_topics=include_topics,
                exclude_topics=exclude_topics,
                min_confidence=min_confidence,
                search_type="text",
                min_score=min_score,
                documents=document_sources,
                search_tree=search_tree,
                flat=flat,
                include_parents=include_parents,
                content_resolved=content,
                text_resolved=text,
                supports_topics=supports_topics
            )

    @classmethod
    def _get_document_sources_for_elements(cls, element_pks: List[int]) -> List[str]:
        """
        Get unique document sources for a list of element primary keys.

        Args:
            element_pks: List of element primary keys

        Returns:
            List of unique document sources
        """
        if not element_pks:
            return []

        db = cls.get_database()
        unique_sources: Set[str] = set()

        for pk in element_pks:
            # Get the element
            element = db.get_element(pk)
            if not element:
                continue

            # Get the document
            doc_id = element.get("doc_id", "")
            document = db.get_document(doc_id)
            if not document:
                continue

            # Add the source if it exists
            source = document.get("source")
            if source:
                unique_sources.add(source)

        return list(unique_sources)

    @classmethod
    def search_with_content(
            cls,
            query_text: str,
            limit: int = 10,
            filter_criteria: Dict[str, Any] = None,
            include_topics: Optional[List[str]] = None,
            exclude_topics: Optional[List[str]] = None,
            min_confidence: Optional[float] = None,
            resolve_content: bool = True,
            include_relationships: bool = True,
            min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for elements similar to the query text and return enriched results.

        Args:
            query_text: The text to search for
            limit: Maximum number of results to return
            filter_criteria: Optional filtering criteria for the search
            include_topics: Topic LIKE patterns to include (e.g., ['security%', '%.policy%'])
            exclude_topics: Topic LIKE patterns to exclude (e.g., ['deprecated%'])
            min_confidence: Minimum confidence threshold for topic results
            resolve_content: Whether to resolve the original content
            include_relationships: Whether to include outgoing relationships
            min_score: Minimum similarity score threshold (default 0.0)

        Returns:
            List of SearchResult objects with element, document, and content information
        """
        # Ensure dependencies are initialized
        db = cls.get_database()
        content_resolver = cls.get_content_resolver()

        logger.debug(f"Searching for text: {query_text} with min_score: {min_score}")

        # Perform the search - get raw results first
        search_results = cls.search_by_text(
            query_text,
            limit=limit,
            filter_criteria=filter_criteria,
            include_topics=include_topics,
            exclude_topics=exclude_topics,
            min_confidence=min_confidence,
            min_score=min_score
        )

        logger.info(f"Found {len(search_results.results)} similar elements after filtering")

        results = []

        # Process each search result
        for item in search_results.results:
            element_pk = item.element_pk
            similarity = item.similarity

            # Get the element
            element = db.get_element(element_pk)
            if not element:
                logger.warning(f"Could not find element with PK: {element_pk}")
                continue

            # Get the document
            doc_id = element.get("doc_id", "")
            document = db.get_document(doc_id)
            if not document:
                logger.warning(f"Could not find document with ID: {doc_id}")
                document = {}  # Use empty dict to avoid None errors

            # Get outgoing relationships if requested
            outgoing_relationships = []
            if include_relationships:
                try:
                    outgoing_relationships = db.get_outgoing_relationships(element_pk)
                    logger.debug(f"Found {len(outgoing_relationships)} outgoing relationships for element {element_pk}")
                except Exception as e:
                    logger.error(f"Error getting outgoing relationships: {str(e)}")

            # Create result object with element and document fields
            result = SearchResult(
                # Similarity score
                similarity=similarity,
                # Topic fields (if available)
                confidence=item.confidence,
                topics=item.topics,

                # Element fields
                element_pk=element_pk,
                element_id=element.get("element_id", ""),
                element_type=element.get("element_type", ""),
                content_preview=element.get("content_preview", ""),
                content_location=element.get("content_location", ""),

                # Document fields
                doc_id=doc_id,
                doc_type=document.get("doc_type", ""),
                source=document.get("source", ""),

                # Outgoing relationships
                outgoing_relationships=outgoing_relationships,

                # Default values for content fields
                resolved_content=None,
                resolved_text=None,
                resolution_error=None
            )

            # Try to resolve content if requested
            if resolve_content:
                content_location = element.get("content_location")
                if content_location and content_resolver.supports_location(content_location):
                    try:
                        result.resolved_content = content_resolver.resolve_content(content_location, text=False)
                        result.resolved_text = content_resolver.resolve_content(content_location, text=True)
                    except Exception as e:
                        logger.error(f"Error resolving content: {str(e)}")
                        result.resolution_error = str(e)

            results.append(result)

        return results


# UPDATED CONVENIENCE FUNCTIONS

def search_structured(query: Union[SearchQueryRequest, Dict[str, Any]],
                      text: bool = False,
                      content: bool = False,
                      flat: bool = False,
                      include_parents: bool = True) -> SearchResults:
    """
    Execute a structured search using Pydantic models.
    Uses singleton instances of database and content resolver.

    Args:
        query: SearchQueryRequest object or dictionary that can be converted to one
        text: Whether to resolve text content for results
        content: Whether to resolve content for results
        flat: Whether to return flat results
        include_parents: Whether to include parent elements

    Returns:
        SearchResults object with materialized content and search tree
    """
    return SearchHelper.search_structured(query, text=text, content=content,
                                          flat=flat, include_parents=include_parents)


def search_simple_structured(query_text: str,
                             limit: int = 10,
                             similarity_threshold: float = 0.7,
                             include_topics: Optional[List[str]] = None,
                             exclude_topics: Optional[List[str]] = None,
                             days_back: Optional[int] = None,
                             element_types: Optional[List[str]] = None,
                             text: bool = False,
                             content: bool = False,
                             flat: bool = False,
                             include_parents: bool = True) -> SearchResults:
    """
    Create and execute a simple structured search query with content materialization.
    Uses singleton instances of database.

    Args:
        query_text: Natural language search query
        limit: Maximum number of results
        similarity_threshold: Minimum similarity score
        include_topics: Topic patterns to include
        exclude_topics: Topic patterns to exclude
        days_back: Filter to documents from last N days
        element_types: Filter by element types
        text: Whether to resolve text content for results
        content: Whether to resolve content for results
        flat: Whether to return flat results
        include_parents: Whether to include parent elements

    Returns:
        SearchResults object with materialized content and search tree
    """
    return SearchHelper.search_simple_structured(
        query_text=query_text,
        limit=limit,
        similarity_threshold=similarity_threshold,
        include_topics=include_topics,
        exclude_topics=exclude_topics,
        days_back=days_back,
        element_types=element_types,
        text=text,
        content=content,
        flat=flat,
        include_parents=include_parents
    )


def create_simple_search_query(query_text: str,
                               days_back: Optional[int] = None,
                               element_types: Optional[List[str]] = None,
                               limit: int = 10,
                               similarity_threshold: float = 0.7) -> SearchQueryRequest:
    """Create a simple SearchQueryRequest from basic parameters."""

    criteria_group = SearchCriteriaGroupRequest(
        operator=LogicalOperatorEnum.AND,
        semantic_search=SemanticSearchRequest(
            query_text=query_text,
            similarity_threshold=similarity_threshold
        )
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
        include_element_dates=bool(days_back),
        include_similarity_scores=True
    )


def create_topic_search_query(include_topics: List[str],
                              exclude_topics: Optional[List[str]] = None,
                              min_confidence: float = 0.7,
                              limit: int = 10) -> SearchQueryRequest:
    """Create a topic-based SearchQueryRequest."""

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
        include_topics=True,
        include_similarity_scores=True
    )


# ORIGINAL CONVENIENCE FUNCTIONS (maintained for backward compatibility)

def search_with_content(
        query_text: str,
        limit: int = 10,
        filter_criteria: Dict[str, Any] = None,
        include_topics: Optional[List[str]] = None,
        exclude_topics: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        resolve_content: bool = True,
        include_relationships: bool = True,
        min_score: float = 0.0
) -> List[SearchResult]:
    """
    Search for elements similar to the query text and return enriched results.
    Uses singleton instances of database and content resolver.

    Args:
        query_text: The text to search for
        limit: Maximum number of results to return
        filter_criteria: Optional filtering criteria for the search
        include_topics: Topic LIKE patterns to include (e.g., ['security%', '%.policy%'])
        exclude_topics: Topic LIKE patterns to exclude (e.g., ['deprecated%'])
        min_confidence: Minimum confidence threshold for topic results
        resolve_content: Whether to resolve the original content
        include_relationships: Whether to include outgoing relationships
        min_score: Minimum similarity score threshold (default 0.0)

    Returns:
        List of SearchResult objects with element, document, and content information
    """
    return SearchHelper.search_with_content(
        query_text=query_text,
        limit=limit,
        filter_criteria=filter_criteria,
        include_topics=include_topics,
        exclude_topics=exclude_topics,
        min_confidence=min_confidence,
        resolve_content=resolve_content,
        include_relationships=include_relationships,
        min_score=min_score
    )


# Convenience function that uses the singleton helper for raw search results
def search_by_text(
        query_text: str,
        limit: int = 10,
        filter_criteria: Dict[str, Any] = None,
        include_topics: Optional[List[str]] = None,
        exclude_topics: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        min_score: float = 0.0,
        text: bool = False,
        content: bool = False,
        flat: bool = False,
        include_parents: bool = True,
) -> SearchResults:
    """
    Search for elements similar to the query text and return raw results.
    Uses singleton instances of database.

    Args:
        query_text: The text to search for
        limit: Maximum number of results to return
        filter_criteria: Optional filtering criteria for the search
        include_topics: Topic LIKE patterns to include (e.g., ['security%', '%.policy%'])
        exclude_topics: Topic LIKE patterns to exclude (e.g., ['deprecated%'])
        min_confidence: Minimum confidence threshold for topic results
        min_score: Minimum similarity score threshold (default 0.0)
        text: Whether to resolve text content for results
        content: Whether to resolve content for results
        flat: Whether to return flat results
        include_parents: Whether to include parent elements

    Returns:
        SearchResults object with element_pk and similarity scores
    """
    return SearchHelper.search_by_text(
        query_text=query_text,
        limit=limit,
        filter_criteria=filter_criteria,
        include_topics=include_topics,
        exclude_topics=exclude_topics,
        min_confidence=min_confidence,
        min_score=min_score,
        text=text,
        content=content,
        flat=flat,
        include_parents=include_parents
    )


# Get document sources from SearchResults
def get_document_sources(search_results: SearchResults) -> List[str]:
    """
    Extract document sources from search results.

    Args:
        search_results: SearchResults object

    Returns:
        List of document sources
    """
    return search_results.documents


# Helper functions for topic management
def get_element_topics(element_pk: int) -> List[str]:
    """
    Get topics assigned to a specific element.

    Args:
        element_pk: Element primary key

    Returns:
        List of topic strings assigned to this element
    """
    db = SearchHelper.get_database()
    return db.get_embedding_topics(element_pk)


def get_topic_statistics() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics about topic distribution across embeddings.

    Returns:
        Dictionary mapping topic strings to statistics
    """
    db = SearchHelper.get_database()
    return db.get_topic_statistics()


def supports_topics() -> bool:
    """
    Check if the current database backend supports topics.

    Returns:
        True if topics are supported, False otherwise
    """
    db = SearchHelper.get_database()
    return db.supports_topics()


# EXAMPLE USAGE:
"""
# Example 1: Using the new structured search methods with content materialization

# Create a structured query using Pydantic models
query = SearchQueryRequest(
    criteria_group=SearchCriteriaGroupRequest(
        operator=LogicalOperatorEnum.AND,
        semantic_search=SemanticSearchRequest(
            query_text="machine learning algorithms",
            similarity_threshold=0.8,
            boost_factor=2.0
        ),
        topic_search=TopicSearchRequest(
            include_topics=['ai%', 'ml%'],
            exclude_topics=['deprecated%'],
            min_confidence=0.8
        ),
        date_search=DateSearchRequest(
            operator=DateRangeOperatorEnum.RELATIVE_DAYS,
            relative_value=30
        )
    ),
    limit=20,
    include_similarity_scores=True,
    include_topics=True
)

# Execute the structured search with content materialization
results = search_structured(query, text=True, content=True)
print(f"Found {results.total_results} results in {results.execution_time_ms}ms")
print(f"Search tree has {len(results.search_tree)} top-level elements")

# Access materialized content in the search tree
for tree_item in results.search_tree:
    if hasattr(tree_item, 'text') and tree_item.text:
        print(f"Materialized text: {tree_item.text[:100]}...")
    if hasattr(tree_item, 'content') and tree_item.content:
        print(f"Materialized content available: {len(tree_item.content)} chars")

# Example 2: Using the simple structured search with content materialization
results = search_simple_structured(
    query_text="data science methodologies",
    limit=15,
    similarity_threshold=0.75,
    include_topics=['data-science%', 'analytics%'],
    exclude_topics=['draft%'],
    days_back=60,
    element_types=['paragraph', 'header'],
    text=True,        # Materialize text content
    content=False,    # Don't materialize raw content
    flat=True,        # Return flat results
    include_parents=True
)

print(f"Text was resolved: {results.text_resolved}")
print(f"Content was resolved: {results.content_resolved}")
print(f"Search tree is flat: {all(hasattr(item, 'score') for item in results.search_tree)}")

# Example 3: Creating queries programmatically
simple_query = create_simple_search_query(
    query_text="artificial intelligence trends",
    days_back=7,
    element_types=['paragraph'],
    similarity_threshold=0.8
)

# Execute with document hierarchy (not flat)
ai_results = search_structured(simple_query, text=True, flat=False, include_parents=True)

# Navigate the hierarchical search tree
for tree_item in ai_results.search_tree:
    print(f"Top-level element: {tree_item.element_type}")
    if hasattr(tree_item, 'child_elements'):
        for child in tree_item.child_elements:
            print(f"  Child: {child.element_type}")
            if hasattr(child, 'text') and child.text:
                print(f"    Text: {child.text[:50]}...")

# Example 4: Working with dictionaries (useful for API integrations)
query_dict = {
    "criteria_group": {
        "operator": "AND",
        "semantic_search": {
            "query_text": "quarterly financial analysis",
            "similarity_threshold": 0.8
        },
        "metadata_search": {
            "exact_matches": {"department": "finance"},
            "exists_filters": ["approval_date"]
        }
    },
    "limit": 25,
    "include_metadata": True
}

# Execute with both text and content materialization
results = search_structured(query_dict, text=True, content=True)

# Access both the search results and the search tree
for result_item in results.results:
    print(f"Result: {result_item.element_pk} (score: {result_item.similarity})")
    
    # Access materialized content via the SearchResultItem properties
    if result_item.text:
        print(f"  Text: {result_item.text[:100]}...")
    if result_item.content:
        print(f"  Content: {result_item.content[:100]}...")

# Example 5: Complex nested query with content materialization
complex_query = SearchQueryRequest(
    criteria_group=SearchCriteriaGroupRequest(
        operator=LogicalOperatorEnum.AND,
        sub_groups=[
            SearchCriteriaGroupRequest(
                operator=LogicalOperatorEnum.OR,
                semantic_search=SemanticSearchRequest(
                    query_text="machine learning",
                    similarity_threshold=0.7
                ),
                topic_search=TopicSearchRequest(
                    include_topics=['ai%', 'ml%', 'deep-learning%']
                )
            ),
            SearchCriteriaGroupRequest(
                operator=LogicalOperatorEnum.NOT,
                topic_search=TopicSearchRequest(
                    include_topics=['deprecated%', 'obsolete%']
                )
            )
        ],
        date_search=DateSearchRequest(
            operator=DateRangeOperatorEnum.QUARTER,
            year=2024,
            quarter=3
        )
    ),
    limit=50,
    include_element_dates=True,
    include_topics=True,
    include_similarity_scores=True
)

# Execute complex query with selective content materialization
complex_results = search_structured(complex_query, 
                                  text=True,          # Get text for display
                                  content=False,      # Skip raw content for performance
                                  flat=False,         # Keep hierarchical structure
                                  include_parents=True)

print(f"Complex search found {complex_results.total_results} results")
print(f"Document sources: {complex_results.documents}")
print(f"Supports topics: {complex_results.supports_topics}")

# Example 6: Performance-conscious search (flat results, no content materialization)
fast_results = search_simple_structured(
    query_text="security vulnerabilities",
    limit=100,
    similarity_threshold=0.6,
    include_topics=['security%'],
    flat=True,              # Faster flat results
    include_parents=False,  # Only scored elements
    text=False,             # No content materialization for speed
    content=False
)

print(f"Fast search returned {len(fast_results.results)} results")
print(f"All results have scores: {all(hasattr(item, 'score') for item in fast_results.search_tree if hasattr(item, 'score'))}")
"""
