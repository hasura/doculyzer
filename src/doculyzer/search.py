import json
import logging
import os
from typing import List, Optional, Dict, Any, Tuple, Set

from pydantic import BaseModel, Field, PrivateAttr

from .adapter import create_content_resolver, ContentResolver
from .config import Config
from .storage import ElementRelationship, DocumentDatabase, ElementHierarchical, ElementFlat, flatten_hierarchy

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
    search_type: str = "embedding"  # Can be "embedding", "text", "content", "topic"
    min_score: float = 0.0  # Minimum score threshold used
    documents: List[str] = Field(default_factory=list)  # Unique list of document sources from the results
    search_tree: Optional[List[ElementHierarchical | ElementFlat]] = None
    # Track whether content was resolved during search
    content_resolved: bool = False
    text_resolved: bool = False
    # Topic-related metadata
    supports_topics: bool = False
    topic_statistics: Optional[Dict[str, Any]] = None

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
                    topic_statistics: Optional[Dict[str, Any]] = None) -> "SearchResults":
        """
        Create a SearchResults object from a list of (element_pk, similarity) tuples.

        Args:
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
            topic_statistics=topic_statistics
        )

    @classmethod
    def from_topic_results(cls, topic_results: List[Dict[str, Any]],
                           query: Optional[str] = None,
                           include_topics: Optional[List[str]] = None,
                           exclude_topics: Optional[List[str]] = None,
                           min_confidence: Optional[float] = None,
                           documents: Optional[List[str]] = None,
                           supports_topics: bool = False,
                           topic_statistics: Optional[Dict[str, Any]] = None) -> "SearchResults":
        """
        Create a SearchResults object from topic search results.

        Args:
            topic_results: List of dictionaries from database search_by_text_and_topics
            query: Optional query string that produced these results
            include_topics: Topic patterns that were included
            exclude_topics: Topic patterns that were excluded
            min_confidence: Minimum confidence threshold used
            documents: List of unique document sources
            supports_topics: Whether the backend supports topics
            topic_statistics: Topic distribution statistics

        Returns:
            SearchResults object
        """
        results = []
        for result in topic_results:
            item = SearchResultItem(
                element_pk=result['element_pk'],
                similarity=result.get('similarity', 0.0),
                confidence=result.get('confidence'),
                topics=result.get('topics', [])
            )
            results.append(item)

        return cls(
            results=results,
            total_results=len(results),
            query=query,
            include_topics=include_topics,
            exclude_topics=exclude_topics,
            min_confidence=min_confidence,
            search_type="topic",
            documents=documents or [],
            supports_topics=supports_topics,
            topic_statistics=topic_statistics
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


# Convenience function that uses the singleton helper
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


# Example usage:
"""
# Using topic filters with existing search methods

# Regular text search with topic filtering
results = search_by_text(
    "security policy",
    include_topics=['security%', '%.policy%'],
    exclude_topics=['deprecated%'],
    min_confidence=0.8,
    limit=10
)

print(f"Found {results.total_results} results")
print(f"Supports topics: {results.supports_topics}")
print(f"Topic statistics: {results.topic_statistics}")

for item in results.results:
    print(f"Element PK: {item.element_pk}, Similarity: {item.similarity:.4f}")
    if item.topics:
        print(f"Topics: {item.topics}")
    if item.confidence:
        print(f"Confidence: {item.confidence}")

# Enhanced search with content and topic filtering
results = search_with_content(
    "security policy",
    include_topics=['security%'],
    exclude_topics=['draft%', 'deprecated%'],
    min_confidence=0.7,
    resolve_content=True,
    min_score=0.5,
    limit=15
)

for result in results:
    print(f"Result: {result.element_type} (Score: {result.similarity:.4f})")
    if result.topics:
        print(f"Topics: {result.topics}")
    if result.confidence:
        print(f"Confidence: {result.confidence}")
    print(f"Preview: {result.content_preview}")

# Topic-only search (no text query)
results = search_by_text(
    "",  # Empty query for topic-only search
    include_topics=['security%'],
    min_confidence=0.8,
    limit=20
)

# Check topic support and get statistics
if supports_topics():
    stats = get_topic_statistics()
    print(f"Topic statistics: {stats}")

    # Get topics for a specific element
    topics = get_element_topics(123)
    print(f"Element 123 topics: {topics}")
else:
    print("Topics not supported - falling back to regular search")
"""
