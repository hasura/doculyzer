import logging
import os
from typing import List, Optional, Dict, Any, Tuple

from pydantic import BaseModel, Field

from .adapter import create_content_resolver
from .config import Config
from .storage import ElementRelationship

logger = logging.getLogger(__name__)

_config = Config(os.environ.get('DOCULYZER_CONFIG_PATH', 'config.yaml'))


class SearchResultItem(BaseModel):
    """Pydantic model for a single search result item."""
    element_pk: int
    similarity: float


class SearchResults(BaseModel):
    """Pydantic model for search results collection."""
    results: List[SearchResultItem] = Field(default_factory=list)
    total_results: int = 0
    query: Optional[str] = None
    filter_criteria: Optional[Dict[str, Any]] = None
    search_type: str = "embedding"  # Can be "embedding", "text", "content"
    min_confidence: float = 0.0  # Minimum confidence threshold used

    @classmethod
    def from_tuples(cls, tuples: List[Tuple[int, float]], query: Optional[str] = None,
                   filter_criteria: Optional[Dict[str, Any]] = None,
                   search_type: str = "embedding",
                   min_confidence: float = 0.0) -> "SearchResults":
        """
        Create a SearchResults object from a list of (element_pk, similarity) tuples.

        Args:
            tuples: List of (element_pk, similarity) tuples
            query: Optional query string that produced these results
            filter_criteria: Optional dictionary of filter criteria
            search_type: Type of search performed
            min_confidence: Minimum confidence threshold used

        Returns:
            SearchResults object
        """
        results = [SearchResultItem(element_pk=pk, similarity=similarity) for pk, similarity in tuples]
        return cls(
            results=results,
            total_results=len(results),
            query=query,
            filter_criteria=filter_criteria,
            search_type=search_type,
            min_confidence=min_confidence
        )


class SearchResult(BaseModel):
    """Pydantic model for storing search result data in a flat structure with relationships."""
    # Similarity score
    similarity: float

    # Element fields
    element_pk: int = Field(default=-1, title="Element primary key, used to get additional information about an element.")
    element_id: str = Field(default="", title="Element natural key.")
    element_type: str = Field(default="", title="Element type.", examples=["body","div","header","table","table_row"])
    content_preview: str | None = Field(default=None, title="Short version of the element's content, used for previewing.")
    content_location: str | None = Field(default=None, title="URI to the location of element's content, if available.")

    # Document fields
    doc_id: str = Field(default="", title="Document natural key.")
    doc_type: str = Field(default="", title="Document type.", examples=["pdf", "docx", "html", "text", "markdown"])
    source: str | None = Field(default=None, title="URI to the original document source, if available.")

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
            min_confidence: float = 0.7
    ) -> SearchResults:
        """
        Search for elements similar to the query text and return raw results.

        Args:
            query_text: The text to search for
            limit: Maximum number of results to return
            filter_criteria: Optional filtering criteria for the search
            min_confidence: Minimum similarity score threshold (default 0.7)

        Returns:
            SearchResults object with element_pk and similarity scores
        """
        # Ensure database is initialized
        db = cls.get_database()

        logger.debug(f"Searching for text: {query_text} with min_confidence: {min_confidence}")

        # Perform the search
        similar_elements = db.search_by_text(query_text, limit=limit * 2, filter_criteria=filter_criteria)
        logger.info(f"Found {len(similar_elements)} similar elements before confidence filtering")

        # Filter by minimum confidence
        filtered_elements = [elem for elem in similar_elements if elem[1] >= min_confidence]
        logger.info(f"Found {len(filtered_elements)} elements after confidence filtering (threshold: {min_confidence})")

        # Apply limit after filtering
        # filtered_elements.reverse()
        filtered_elements = filtered_elements[:limit]

        # Convert to SearchResults
        return SearchResults.from_tuples(
            tuples=filtered_elements,
            query=query_text,
            filter_criteria=filter_criteria,
            search_type="text",
            min_confidence=min_confidence
        )

    @classmethod
    def search_with_content(
            cls,
            query_text: str,
            limit: int = 10,
            filter_criteria: Dict[str, Any] = None,
            resolve_content: bool = True,
            include_relationships: bool = True,
            min_confidence: float = 0.7
    ) -> List[SearchResult]:
        """
        Search for elements similar to the query text and return enriched results.

        Args:
            query_text: The text to search for
            limit: Maximum number of results to return
            filter_criteria: Optional filtering criteria for the search
            resolve_content: Whether to resolve the original content
            include_relationships: Whether to include outgoing relationships
            min_confidence: Minimum similarity score threshold (default 0.7)

        Returns:
            List of SearchResult objects with element, document, and content information
        """
        # Ensure dependencies are initialized
        db = cls.get_database()
        content_resolver = cls.get_content_resolver()

        logger.debug(f"Searching for text: {query_text} with min_confidence: {min_confidence}")

        # Perform the search - get raw results first
        search_results = cls.search_by_text(
            query_text,
            limit=limit,
            filter_criteria=filter_criteria,
            min_confidence=min_confidence
        )
        similar_elements = [(item.element_pk, item.similarity) for item in search_results.results]
        logger.info(f"Found {len(similar_elements)} similar elements after confidence filtering")

        results = []

        # Process each search result
        for element_pk, similarity in similar_elements:
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
        resolve_content: bool = True,
        include_relationships: bool = True,
        min_confidence: float = 0.7
) -> List[SearchResult]:
    """
    Search for elements similar to the query text and return enriched results.
    Uses singleton instances of database and content resolver.

    Args:
        query_text: The text to search for
        limit: Maximum number of results to return
        filter_criteria: Optional filtering criteria for the search
        resolve_content: Whether to resolve the original content
        include_relationships: Whether to include outgoing relationships
        min_confidence: Minimum similarity score threshold (default 0.7)

    Returns:
        List of SearchResult objects with element, document, and content information
    """
    return SearchHelper.search_with_content(
        query_text=query_text,
        limit=limit,
        filter_criteria=filter_criteria,
        resolve_content=resolve_content,
        include_relationships=include_relationships,
        min_confidence=min_confidence
    )


# Convenience function that uses the singleton helper for raw search results
def search_by_text(
        query_text: str,
        limit: int = 10,
        filter_criteria: Dict[str, Any] = None,
        min_confidence: float = 0.7
) -> SearchResults:
    """
    Search for elements similar to the query text and return raw results.
    Uses singleton instances of database.

    Args:
        query_text: The text to search for
        limit: Maximum number of results to return
        filter_criteria: Optional filtering criteria for the search
        min_confidence: Minimum similarity score threshold (default 0.7)

    Returns:
        SearchResults object with element_pk and similarity scores
    """
    return SearchHelper.search_by_text(
        query_text=query_text,
        limit=limit,
        filter_criteria=filter_criteria,
        min_confidence=min_confidence
    )


# Example usage:
"""
# Using the singleton-based convenience function with confidence threshold
results = search_with_content("search query", min_confidence=0.7)

# Print results with relationship information
for i, result in enumerate(results):
    print(f"Result {i+1}: {result.element_type} (Score: {result.similarity:.4f})")
    print(f"Preview: {result.content_preview}")

    # Print relationships summary
    rel_count = result.get_relationship_count()
    print(f"Outgoing relationships: {rel_count}")

    if rel_count > 0:
        # Group by type
        by_type = result.get_relationships_by_type()
        for rel_type, rels in by_type.items():
            print(f"  - {rel_type}: {len(rels)}")

        # Print contained elements
        contained = result.get_contained_elements()
        if contained:
            print(f"Contains {len(contained)} elements:")
            for rel in contained[:3]:  # Show just the first few
                print(f"  - {rel.target_element_type or 'Unknown'}: {rel.target_reference}")

    if result.resolved_content:
        print(f"Content: {result.resolved_content[:100]}...")
    print("---")

# Raw search results with higher confidence threshold
search_results = search_by_text("search query", limit=5, min_confidence=0.8)
print(f"Found {search_results.total_results} results for '{search_results.query}' with confidence >= {search_results.min_confidence}")
for item in search_results.results:
    print(f"Element PK: {item.element_pk}, Similarity: {item.similarity:.4f}")

# Search with filters and lower confidence threshold
results = search_with_content(
    "search query",
    limit=20,
    filter_criteria={"element_type": ["header", "paragraph"]},
    resolve_content=True,
    include_relationships=False,
    min_confidence=0.5  # Lower threshold to get more results
)
"""
