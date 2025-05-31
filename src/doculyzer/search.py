import json
import logging
import os
import time
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


class DocumentMaterializationOptions(BaseModel):
    """Configuration for document materialization in search results."""
    include_full_document: bool = False
    include_document_outline: bool = False
    include_document_statistics: bool = False
    document_format: Optional[str] = None  # 'text', 'markdown', 'html', 'json', 'yaml', 'xml'
    include_full_text: bool = True  # Whether to include full text in documents
    batch_documents: bool = True  # Whether to use batch loading for efficiency
    max_document_length: Optional[int] = None  # Truncate document content if longer
    join_elements: bool = True  # Whether to join elements in full text
    element_separator: str = '\n\n'  # Separator for joining elements


class MaterializedDocument(BaseModel):
    """Container for materialized document content in various formats."""
    doc_id: str
    title: Optional[str] = None
    source: Optional[str] = None
    doc_type: Optional[str] = None

    # Document structure
    complete_document: Optional[Dict[str, Any]] = None
    outline: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None

    # Formatted content
    formatted_content: Optional[str] = None
    format_type: Optional[str] = None

    # Full text
    full_text: Optional[str] = None

    # Metadata
    element_count: int = 0
    relationship_count: int = 0
    has_full_text: bool = False
    text_length: int = 0

    # Error information
    materialization_error: Optional[str] = None


class SearchResults(BaseModel):
    """Pydantic model for search results collection with document materialization."""
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

    # NEW: Document materialization fields
    materialized_documents: Dict[str, MaterializedDocument] = Field(default_factory=dict)
    materialization_options: Optional[DocumentMaterializationOptions] = None
    materialization_time_ms: Optional[float] = None

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
                    execution_time_ms: Optional[float] = None,
                    # NEW: Document materialization parameters
                    materialized_documents: Optional[Dict[str, MaterializedDocument]] = None,
                    materialization_options: Optional[DocumentMaterializationOptions] = None,
                    materialization_time_ms: Optional[float] = None) -> "SearchResults":
        """
        Create a SearchResults object from a list of (element_pk, similarity) tuples.

        Args:
            tuples: List of (element_pk, similarity) tuples
            flat: Whether to flatten hierarchy
            include_parents: Whether to include parent elements
            query: Optional query string that produced these results
            filter_criteria: Optional dictionary of filter criteria
            include_topics: Topic patterns that were included
            exclude_topics: Topic patterns that were excluded
            min_confidence: Minimum confidence threshold for topic results
            search_type: Type of search performed
            min_score: Minimum score threshold used
            search_tree: Optional tree structure representing the search results
            documents: List of unique document sources
            content_resolved: Whether content was resolved during search
            text_resolved: Whether text was resolved during search
            supports_topics: Whether the backend supports topics
            topic_statistics: Topic distribution statistics
            query_id: Unique query identifier
            execution_time_ms: Query execution time
            materialized_documents: Dictionary of materialized documents
            materialization_options: Document materialization configuration
            materialization_time_ms: Time spent on document materialization

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
            execution_time_ms=execution_time_ms,
            materialized_documents=materialized_documents or {},
            materialization_options=materialization_options,
            materialization_time_ms=materialization_time_ms
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

    # DOCUMENT MATERIALIZATION METHODS

    @classmethod
    def _materialize_documents(cls,
                               doc_ids: List[str],
                               options: DocumentMaterializationOptions) -> Dict[str, MaterializedDocument]:
        """
        Materialize documents according to the specified options.

        Args:
            doc_ids: List of document IDs to materialize
            options: Materialization options

        Returns:
            Dictionary mapping doc_id to MaterializedDocument
        """
        if not doc_ids or not (options.include_full_document or options.document_format or
                               options.include_document_outline or options.include_document_statistics):
            return {}

        db = cls.get_database()
        materialized = {}

        try:
            if options.batch_documents and len(doc_ids) > 1:
                # Use batch loading for efficiency
                batch_documents = db.get_documents_batch(doc_ids, options.include_full_text)

                for doc_id, complete_doc in batch_documents.items():
                    materialized[doc_id] = cls._create_materialized_document(
                        doc_id, complete_doc, options
                    )
            else:
                # Load documents individually
                for doc_id in doc_ids:
                    complete_doc = db.get_complete_document(doc_id, options.include_full_text)
                    if complete_doc:
                        materialized[doc_id] = cls._create_materialized_document(
                            doc_id, complete_doc, options
                        )

        except Exception as e:
            logger.error(f"Error materializing documents: {str(e)}")
            # Create error entries for failed documents
            for doc_id in doc_ids:
                if doc_id not in materialized:
                    materialized[doc_id] = MaterializedDocument(
                        doc_id=doc_id,
                        materialization_error=str(e)
                    )

        return materialized

    @classmethod
    def _create_materialized_document(cls,
                                      doc_id: str,
                                      complete_doc: Dict[str, Any],
                                      options: DocumentMaterializationOptions) -> MaterializedDocument:
        """
        Create a MaterializedDocument from complete document data.

        Args:
            doc_id: Document ID
            complete_doc: Complete document structure from get_complete_document
            options: Materialization options

        Returns:
            MaterializedDocument object
        """
        db = cls.get_database()
        document_meta = complete_doc.get('document', {})

        materialized = MaterializedDocument(
            doc_id=doc_id,
            title=document_meta.get('title'),
            source=document_meta.get('source'),
            doc_type=document_meta.get('doc_type'),
            element_count=complete_doc.get('element_count', 0),
            relationship_count=complete_doc.get('relationship_count', 0),
            has_full_text=complete_doc.get('has_full_text', False)
        )

        try:
            # Include complete document structure if requested
            if options.include_full_document:
                materialized.complete_document = complete_doc

            # Include document outline if requested
            if options.include_document_outline:
                materialized.outline = db.get_document_outline(doc_id)

            # Include document statistics if requested
            if options.include_document_statistics:
                materialized.statistics = db.get_document_statistics(doc_id)
                if materialized.statistics:
                    materialized.text_length = materialized.statistics.get('total_characters', 0)

            # Generate formatted content if format is specified
            if options.document_format:
                formatted_content = db.extract_document_content(doc_id, options.document_format)
                if formatted_content:
                    # Apply length limit if specified
                    if options.max_document_length and len(formatted_content) > options.max_document_length:
                        formatted_content = formatted_content[
                                            :options.max_document_length] + "\n[... content truncated ...]"

                    materialized.formatted_content = formatted_content
                    materialized.format_type = options.document_format

            # Generate full text if not included in formatted content
            if not options.document_format or options.document_format not in ['text', 'markdown']:
                full_text = db.get_document_full_text(
                    doc_id,
                    join_elements=options.join_elements,
                    element_separator=options.element_separator
                )
                if full_text:
                    # Apply length limit if specified
                    if options.max_document_length and len(full_text) > options.max_document_length:
                        full_text = full_text[:options.max_document_length] + "\n[... content truncated ...]"

                    materialized.full_text = full_text

        except Exception as e:
            logger.error(f"Error creating materialized document for {doc_id}: {str(e)}")
            materialized.materialization_error = str(e)

        return materialized

    @classmethod
    def _extract_document_ids_from_results(cls, results: List[SearchResultItem]) -> List[str]:
        """Extract unique document IDs from search results."""
        doc_ids = set()
        for item in results:
            if item.doc_id:
                doc_ids.add(item.doc_id)
        return list(doc_ids)

    # NEW STRUCTURED SEARCH METHODS

    @classmethod
    def execute_structured_search(cls, query: SearchQueryRequest,
                                  text: bool = False,
                                  content: bool = False,
                                  flat: bool = False,
                                  include_parents: bool = True,
                                  # NEW: Document materialization options
                                  include_full_document: bool = False,
                                  document_format: Optional[str] = None,
                                  include_document_outline: bool = False,
                                  include_document_statistics: bool = False,
                                  max_document_length: Optional[int] = None,
                                  batch_documents: bool = True) -> SearchResults:
        """
        Execute a structured search using Pydantic models with SearchHelper enhancements.

        Args:
            query: SearchQueryRequest object with structured search criteria
            text: Whether to resolve text content for results
            content: Whether to resolve content for results
            flat: Whether to return flat results
            include_parents: Whether to include parent elements
            include_full_document: Whether to include complete document structure
            document_format: Format for document content ('text', 'markdown', 'html', 'json', 'yaml', 'xml')
            include_document_outline: Whether to include document outline/hierarchy
            include_document_statistics: Whether to include document statistics
            max_document_length: Maximum length for document content (truncate if longer)
            batch_documents: Whether to use batch loading for efficiency

        Returns:
            SearchResults object with results, search tree, materialized content, and documents
        """
        # Ensure database is initialized
        db = cls.get_database()
        resolver = cls.get_content_resolver()

        logger.debug(f"Executing structured search with query ID: {query.query_id}")

        start_time = time.time()

        try:
            # Import the execute_search function from pydantic_search
            from .storage.search import execute_search

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

            # NEW: Handle document materialization
            materialization_options = None
            materialized_documents = {}
            materialization_time_ms = None

            if (include_full_document or document_format or include_document_outline or
                    include_document_statistics):
                mat_start = time.time()

                materialization_options = DocumentMaterializationOptions(
                    include_full_document=include_full_document,
                    document_format=document_format,
                    include_document_outline=include_document_outline,
                    include_document_statistics=include_document_statistics,
                    max_document_length=max_document_length,
                    batch_documents=batch_documents
                )

                # Extract unique document IDs from search results
                doc_ids = cls._extract_document_ids_from_results(search_result_items)

                # Materialize the documents
                materialized_documents = cls._materialize_documents(doc_ids, materialization_options)

                materialization_time_ms = (time.time() - mat_start) * 1000

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
                supports_topics=db.supports_topics(),
                materialized_documents=materialized_documents,
                materialization_options=materialization_options,
                materialization_time_ms=materialization_time_ms
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
                          include_parents: bool = True,
                          # NEW: Document materialization options
                          include_full_document: bool = False,
                          document_format: Optional[str] = None,
                          include_document_outline: bool = False,
                          include_document_statistics: bool = False,
                          max_document_length: Optional[int] = None,
                          batch_documents: bool = True) -> SearchResults:
        """
        Convenience method for structured search that accepts either Pydantic model or dict.

        Args:
            query: SearchQueryRequest object or dictionary that can be converted to one
            text: Whether to resolve text content for results
            content: Whether to resolve content for results
            flat: Whether to return flat results
            include_parents: Whether to include parent elements
            include_full_document: Whether to include complete document structure
            document_format: Format for document content
            include_document_outline: Whether to include document outline
            include_document_statistics: Whether to include document statistics
            max_document_length: Maximum length for document content
            batch_documents: Whether to use batch loading

        Returns:
            SearchResults object
        """
        if isinstance(query, dict):
            query = SearchQueryRequest.model_validate(query)

        return cls.execute_structured_search(
            query,
            text=text,
            content=content,
            flat=flat,
            include_parents=include_parents,
            include_full_document=include_full_document,
            document_format=document_format,
            include_document_outline=include_document_outline,
            include_document_statistics=include_document_statistics,
            max_document_length=max_document_length,
            batch_documents=batch_documents
        )

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
                                 include_parents: bool = True,
                                 # NEW: Document materialization options
                                 include_full_document: bool = False,
                                 document_format: Optional[str] = None,
                                 include_document_outline: bool = False,
                                 include_document_statistics: bool = False,
                                 max_document_length: Optional[int] = None,
                                 batch_documents: bool = True) -> SearchResults:
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
            include_full_document: Whether to include complete document structure
            document_format: Format for document content
            include_document_outline: Whether to include document outline
            include_document_statistics: Whether to include document statistics
            max_document_length: Maximum length for document content
            batch_documents: Whether to use batch loading

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

        # Configure result options
        query = SearchQueryRequest(
            criteria_group=criteria_group,
            limit=limit,
            include_similarity_scores=True,
            include_element_dates=bool(days_back)
        )

        # Build and execute query
        return cls.execute_structured_search(
            query,
            text=text,
            content=content,
            flat=flat,
            include_parents=include_parents,
            include_full_document=include_full_document,
            document_format=document_format,
            include_document_outline=include_document_outline,
            include_document_statistics=include_document_statistics,
            max_document_length=max_document_length,
            batch_documents=batch_documents
        )

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

    # ENHANCED SEARCH WITH DOCUMENTS

    @classmethod
    def search_with_documents(cls,
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
                              # NEW: Document materialization options
                              include_full_document: bool = False,
                              document_format: Optional[str] = None,
                              include_document_outline: bool = False,
                              include_document_statistics: bool = False,
                              max_document_length: Optional[int] = None,
                              batch_documents: bool = True) -> SearchResults:
        """
        Enhanced search with document materialization capabilities.

        Args:
            # Standard search parameters
            query_text: The text to search for
            limit: Maximum number of results to return
            filter_criteria: Optional filtering criteria for the search
            include_topics: Topic LIKE patterns to include
            exclude_topics: Topic LIKE patterns to exclude
            min_confidence: Minimum confidence threshold for topic results
            min_score: Minimum similarity score threshold
            text: Whether to resolve text content for results
            content: Whether to resolve content for results
            flat: Whether to return flat results
            include_parents: Whether to include parent elements

            # NEW: Document materialization parameters
            include_full_document: Whether to include complete document structure
            document_format: Format for document content ('text', 'markdown', 'html', 'json', 'yaml', 'xml')
            include_document_outline: Whether to include document outline/hierarchy
            include_document_statistics: Whether to include document statistics
            max_document_length: Maximum length for document content (truncate if longer)
            batch_documents: Whether to use batch loading for efficiency

        Returns:
            SearchResults with materialized documents
        """
        start_time = time.time()

        # Perform the standard search
        search_results = cls.search_by_text(
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

        # Create materialization options
        materialization_options = DocumentMaterializationOptions(
            include_full_document=include_full_document,
            document_format=document_format,
            include_document_outline=include_document_outline,
            include_document_statistics=include_document_statistics,
            max_document_length=max_document_length,
            batch_documents=batch_documents
        )

        # Materialize documents if requested
        materialized_documents = {}
        materialization_time_ms = None

        if (include_full_document or document_format or include_document_outline or
                include_document_statistics):
            mat_start = time.time()

            # Extract unique document IDs from search results
            doc_ids = cls._extract_document_ids_from_results(search_results.results)

            # Materialize the documents
            materialized_documents = cls._materialize_documents(doc_ids, materialization_options)

            materialization_time_ms = (time.time() - mat_start) * 1000

        # Update the search results with materialization data
        search_results.materialized_documents = materialized_documents
        search_results.materialization_options = materialization_options
        search_results.materialization_time_ms = materialization_time_ms

        total_time = (time.time() - start_time) * 1000
        logger.info(f"Enhanced search completed in {total_time:.2f}ms "
                    f"(search: {search_results.execution_time_ms or 0:.2f}ms, "
                    f"materialization: {materialization_time_ms or 0:.2f}ms)")

        return search_results

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
                      include_parents: bool = True,
                      # NEW: Document materialization options
                      include_full_document: bool = False,
                      document_format: Optional[str] = None,
                      include_document_outline: bool = False,
                      include_document_statistics: bool = False,
                      max_document_length: Optional[int] = None,
                      batch_documents: bool = True) -> SearchResults:
    """
    Execute a structured search using Pydantic models.
    Uses singleton instances of database and content resolver.

    Args:
        query: SearchQueryRequest object or dictionary that can be converted to one
        text: Whether to resolve text content for results
        content: Whether to resolve content for results
        flat: Whether to return flat results
        include_parents: Whether to include parent elements
        include_full_document: Whether to include complete document structure
        document_format: Format for document content
        include_document_outline: Whether to include document outline
        include_document_statistics: Whether to include document statistics
        max_document_length: Maximum length for document content
        batch_documents: Whether to use batch loading

    Returns:
        SearchResults object with materialized content and search tree
    """
    return SearchHelper.search_structured(
        query,
        text=text,
        content=content,
        flat=flat,
        include_parents=include_parents,
        include_full_document=include_full_document,
        document_format=document_format,
        include_document_outline=include_document_outline,
        include_document_statistics=include_document_statistics,
        max_document_length=max_document_length,
        batch_documents=batch_documents
    )


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
                             include_parents: bool = True,
                             # NEW: Document materialization options
                             include_full_document: bool = False,
                             document_format: Optional[str] = None,
                             include_document_outline: bool = False,
                             include_document_statistics: bool = False,
                             max_document_length: Optional[int] = None,
                             batch_documents: bool = True) -> SearchResults:
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
        include_full_document: Whether to include complete document structure
        document_format: Format for document content
        include_document_outline: Whether to include document outline
        include_document_statistics: Whether to include document statistics
        max_document_length: Maximum length for document content
        batch_documents: Whether to use batch loading

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
        include_parents=include_parents,
        include_full_document=include_full_document,
        document_format=document_format,
        include_document_outline=include_document_outline,
        include_document_statistics=include_document_statistics,
        max_document_length=max_document_length,
        batch_documents=batch_documents
    )


def search_with_documents(query_text: str,
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
                          # Document materialization options
                          include_full_document: bool = False,
                          document_format: Optional[str] = None,
                          include_document_outline: bool = False,
                          include_document_statistics: bool = False,
                          max_document_length: Optional[int] = None,
                          batch_documents: bool = True) -> SearchResults:
    """
    Search with complete document materialization capabilities.

    This function combines semantic search with full document materialization,
    allowing you to get search results along with complete document content
    in various formats.

    Example:
        # Get search results with documents as formatted markdown
        results = search_with_documents(
            query_text="machine learning algorithms",
            limit=10,
            document_format="markdown",
            include_document_statistics=True,
            max_document_length=5000
        )

        # Access materialized documents
        for doc_id, doc in results.materialized_documents.items():
            print(f"Document: {doc.title}")
            print(f"Markdown content: {doc.formatted_content}")
            print(f"Statistics: {doc.statistics}")
    """
    return SearchHelper.search_with_documents(
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
        include_parents=include_parents,
        include_full_document=include_full_document,
        document_format=document_format,
        include_document_outline=include_document_outline,
        include_document_statistics=include_document_statistics,
        max_document_length=max_document_length,
        batch_documents=batch_documents
    )


def get_document_in_format(doc_id: str,
                           format_type: str = 'text',
                           include_outline: bool = False,
                           include_statistics: bool = False,
                           include_full_text: bool = True,
                           max_length: Optional[int] = None) -> MaterializedDocument:
    """
    Get a single document in the specified format.

    This is a convenience function for getting formatted document content
    without performing a search.

    Args:
        doc_id: Document ID to retrieve
        format_type: Output format ('text', 'markdown', 'html', 'json', 'yaml', 'xml')
        include_outline: Whether to include document outline
        include_statistics: Whether to include document statistics
        include_full_text: Whether to include full text content
        max_length: Maximum length for content (truncate if longer)

    Returns:
        MaterializedDocument with formatted content

    Example:
        # Get document as markdown with outline
        doc = get_document_in_format(
            doc_id="doc_123",
            format_type="markdown",
            include_outline=True,
            max_length=5000
        )

        print(f"Markdown: {doc.formatted_content}")
        print(f"Outline: {doc.outline}")
    """
    db = SearchHelper.get_database()

    options = DocumentMaterializationOptions(
        include_full_document=True,
        document_format=format_type,
        include_document_outline=include_outline,
        include_document_statistics=include_statistics,
        include_full_text=include_full_text,
        max_document_length=max_length
    )

    complete_doc = db.get_complete_document(doc_id, include_full_text)
    if not complete_doc:
        return MaterializedDocument(
            doc_id=doc_id,
            materialization_error=f"Document {doc_id} not found"
        )

    return SearchHelper._create_materialized_document(doc_id, complete_doc, options)


def get_documents_batch_formatted(doc_ids: List[str],
                                  format_type: str = 'text',
                                  include_outline: bool = False,
                                  include_statistics: bool = False,
                                  include_full_text: bool = True,
                                  max_length: Optional[int] = None) -> Dict[str, MaterializedDocument]:
    """
    Get multiple documents in the specified format using batch loading.

    Args:
        doc_ids: List of document IDs to retrieve
        format_type: Output format for all documents
        include_outline: Whether to include document outlines
        include_statistics: Whether to include document statistics
        include_full_text: Whether to include full text content
        max_length: Maximum length for content

    Returns:
        Dictionary mapping doc_id to MaterializedDocument

    Example:
        # Get multiple documents as HTML
        docs = get_documents_batch_formatted(
            doc_ids=["doc_1", "doc_2", "doc_3"],
            format_type="html",
            include_statistics=True
        )

        for doc_id, doc in docs.items():
            print(f"Document {doc_id}: {doc.statistics['total_words']} words")
    """
    options = DocumentMaterializationOptions(
        include_full_document=True,
        document_format=format_type,
        include_document_outline=include_outline,
        include_document_statistics=include_statistics,
        include_full_text=include_full_text,
        max_document_length=max_length,
        batch_documents=True
    )

    return SearchHelper._materialize_documents(doc_ids, options)


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
# Example 1: Search with document materialization as markdown
results = search_with_documents(
    query_text="machine learning best practices",
    limit=10,
    document_format="markdown",
    include_document_statistics=True,
    max_document_length=5000
)

print(f"Found {results.total_results} results")
print(f"Materialized {len(results.materialized_documents)} documents")

for doc_id, doc in results.materialized_documents.items():
    print(f"\nDocument: {doc.title}")
    print(f"Words: {doc.statistics.get('total_words', 0) if doc.statistics else 'N/A'}")
    print(f"Markdown preview: {doc.formatted_content[:200]}...")

# Example 2: Structured search with HTML document materialization
query = SearchQueryRequest(
    criteria_group=SearchCriteriaGroupRequest(
        operator=LogicalOperatorEnum.AND,
        semantic_search=SemanticSearchRequest(
            query_text="quarterly financial results",
            similarity_threshold=0.8
        ),
        date_search=DateSearchRequest(
            operator=DateRangeOperatorEnum.QUARTER,
            year=2024,
            quarter=3
        )
    ),
    limit=15
)

results = search_structured(
    query=query,
    document_format="html",
    include_document_outline=True,
    include_document_statistics=True,
    text=True  # Also get element text
)

for doc_id, doc in results.materialized_documents.items():
    print(f"\nDocument: {doc.title}")
    print(f"Format: {doc.format_type}")
    print(f"HTML length: {len(doc.formatted_content) if doc.formatted_content else 0}")
    if doc.outline:
        print(f"Outline elements: {doc.outline.get('total_elements', 0)}")

# Example 3: Get single document in multiple formats
doc_markdown = get_document_in_format("doc_123", "markdown", include_outline=True)
doc_html = get_document_in_format("doc_123", "html", include_statistics=True)
doc_text = get_document_in_format("doc_123", "text", max_length=1000)

print(f"Markdown: {len(doc_markdown.formatted_content or '')} chars")
print(f"HTML: {len(doc_html.formatted_content or '')} chars") 
print(f"Text (truncated): {len(doc_text.formatted_content or '')} chars")

# Example 4: Batch load documents in JSON format
doc_ids = ["doc_1", "doc_2", "doc_3", "doc_4"]
docs_json = get_documents_batch_formatted(
    doc_ids=doc_ids,
    format_type="json",
    include_statistics=True
)

for doc_id, doc in docs_json.items():
    print(f"Document {doc_id}: {doc.element_count} elements")
    if doc.statistics:
        print(f"  Characters: {doc.statistics.get('total_characters', 0)}")
        print(f"  Element types: {list(doc.statistics.get('element_types', {}).keys())}")

# Example 5: Performance comparison - search with and without documents
import time

# Fast search (no document materialization)
start = time.time()
fast_results = search_by_text("data analysis", limit=20)
fast_time = time.time() - start

# Search with document materialization
start = time.time()
rich_results = search_with_documents(
    query_text="data analysis",
    limit=20,
    document_format="markdown",
    include_document_statistics=True,
    batch_documents=True
)
rich_time = time.time() - start

print(f"Fast search: {fast_time:.3f}s")
print(f"Rich search: {rich_time:.3f}s (materialization: {rich_results.materialization_time_ms:.1f}ms)")
print(f"Documents materialized: {len(rich_results.materialized_documents)}")
"""
