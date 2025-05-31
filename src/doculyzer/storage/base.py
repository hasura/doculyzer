"""
Enhanced DocumentDatabase Abstract Base Class

This module provides the revised abstract base class for document database implementations
with integrated structured search support, comprehensive full-text search configuration,
and convenient complete document retrieval methods.

All backend implementations must inherit from this class and implement both legacy and
structured search methods.

The class bridges the gap between the original document storage API and the new
structured search system, ensuring backward compatibility while enabling advanced
search capabilities and flexible full-text storage options.
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
    search support including both legacy methods, structured search capabilities,
    flexible full-text storage configuration, and convenient document retrieval methods.
    """

    def __init__(self, conn_params: Dict[str, Any]):
        """
        Initialize the document database with connection parameters and full-text configuration.

        Args:
            conn_params: Connection parameters specific to each backend, with common options:

                Full-text storage and indexing options:
                - store_full_text: Whether to store full text for retrieval (default: True)
                - index_full_text: Whether to index full text for search (default: True)
                - compress_full_text: Whether to enable compression for stored text (default: False)
                - full_text_max_length: Maximum length for full text, truncate if longer (default: None)

                Common configuration patterns:
                - Search + Storage: store_full_text=True, index_full_text=True (default, best search quality)
                - Search only: store_full_text=False, index_full_text=True (saves storage space)
                - Storage only: store_full_text=True, index_full_text=False (for retrieval without search)
                - Neither: store_full_text=False, index_full_text=False (minimal storage, preview only)

        Note:
            Backends that don't support full-text search will ignore index_full_text setting
            but should still respect store_full_text for content storage optimization.
        """
        # Store connection parameters for backend use
        self.conn_params = conn_params

        # Full-text configuration options (standardized across all backends)
        self.store_full_text = conn_params.get('store_full_text', True)
        self.index_full_text = conn_params.get('index_full_text', True)
        self.compress_full_text = conn_params.get('compress_full_text', False)
        self.full_text_max_length = conn_params.get('full_text_max_length', None)

    # ========================================
    # CORE DATABASE OPERATIONS
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
    # DOCUMENT STORAGE OPERATIONS
    # ========================================

    @abstractmethod
    def store_document(self, document: Dict[str, Any], elements: List[Dict[str, Any]],
                       relationships: List[Dict[str, Any]]) -> None:
        """
        Store a document with its elements and relationships.

        Implementation Notes:
            - Should respect full-text configuration options when processing 'full_content' field
            - Apply full_text_max_length truncation if configured
            - Store/index full text based on store_full_text/index_full_text settings

        Args:
            document: Document metadata
            elements: Document elements (may contain 'full_content' field)
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
            elements: Document elements (may contain 'full_content' field)
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
    # LEGACY SEARCH METHODS
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
        Search elements by content preview and optionally full text.

        Implementation Notes:
            - Should search content_preview for all backends
            - Should also search full_text field if index_full_text=True and backend supports it
            - Backends without full-text search should fall back to content_preview only

        Args:
            search_text: Text to search for
            limit: Maximum number of results

        Returns:
            List of matching elements
        """
        pass

    # ========================================
    # EMBEDDING SEARCH METHODS
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

        Implementation Notes:
            - Should use both content_preview and full_text (if available) for best results
            - Backends should adapt search fields based on index_full_text configuration

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
    # DATE STORAGE AND SEARCH METHODS
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
    # STRUCTURED SEARCH SYSTEM (required for all backends)
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
                if self.supports_full_text_search():
                    supported.add(SearchCapability.FULL_TEXT_SEARCH)
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

        Implementation Notes:
            - Should respect full-text configuration when executing text criteria
            - Should adapt search fields based on index_full_text setting

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
    # COMPLETE DOCUMENT RETRIEVAL METHODS (concrete implementations)
    # ========================================

    def get_complete_document(self, doc_id: str, include_full_text: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get a complete document with all its elements and relationships in one call.

        Args:
            doc_id: Document ID
            include_full_text: Whether to include full text content in elements

        Returns:
            Complete document structure or None if not found:
            {
                'document': {...},           # Document metadata
                'elements': [...],           # All document elements
                'relationships': [...],      # All document relationships
                'element_count': int,        # Number of elements
                'relationship_count': int,   # Number of relationships
                'has_full_text': bool,       # Whether full text is available
                'storage_config': {...}     # Full-text storage configuration
            }
        """
        # Get document metadata
        document = self.get_document(doc_id)
        if not document:
            return None

        # Get all components
        elements = self.get_document_elements(doc_id)
        relationships = self.get_document_relationships(doc_id)

        # Filter full text if requested
        if not include_full_text:
            for element in elements:
                element.pop('full_content', None)

        # Check if any elements have full text
        has_full_text = any('full_content' in elem and elem['full_content'] for elem in elements)

        # Get storage configuration for context
        storage_config = self.get_text_storage_config()

        return {
            'document': document,
            'elements': elements,
            'relationships': relationships,
            'element_count': len(elements),
            'relationship_count': len(relationships),
            'has_full_text': has_full_text,
            'storage_config': storage_config
        }

    def get_document_full_text(self, doc_id: str, join_elements: bool = True,
                              element_separator: str = '\n\n') -> Optional[str]:
        """
        Get the complete full text content of a document by combining all elements.

        Args:
            doc_id: Document ID
            join_elements: Whether to join all element content into one string
            element_separator: Separator to use when joining elements

        Returns:
            Complete document text or None if not found
        """
        elements = self.get_document_elements(doc_id)
        if not elements:
            return None

        # Sort elements by position/order if available
        if elements and any('position' in elem for elem in elements):
            elements.sort(key=lambda x: x.get('position', 0))

        # Extract full text from elements
        text_parts = []
        for element in elements:
            if 'full_content' in element and element['full_content']:
                text_parts.append(element['full_content'])
            elif 'content_preview' in element and element['content_preview']:
                text_parts.append(element['content_preview'])

        if not text_parts:
            return None

        if join_elements:
            return element_separator.join(text_parts)
        else:
            return text_parts

    def get_document_outline(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a hierarchical outline of a document showing structure.

        Args:
            doc_id: Document ID

        Returns:
            Document outline structure or None if not found:
            {
                'document_id': str,
                'title': str,
                'outline': [...],           # Hierarchical element structure
                'total_elements': int,
                'element_types': {...},     # Count by element type
                'text_length': int          # Total text length
            }
        """
        document = self.get_document(doc_id)
        if not document:
            return None

        elements = self.get_document_elements(doc_id)

        # Create outline using existing hierarchy method
        element_tuples = []
        for elem in elements:
            element_pk = elem.get('pk') or elem.get('id')
            if element_pk:
                element_tuples.append((element_pk, 1.0))

        outline = self.get_results_outline(element_tuples) if element_tuples else []

        # Count element types and calculate text length
        element_types = {}
        total_text_length = 0

        for element in elements:
            elem_type = element.get('element_type', 'unknown')
            element_types[elem_type] = element_types.get(elem_type, 0) + 1

            # Count text length
            content = element.get('full_content') or element.get('content_preview', '')
            if content:
                total_text_length += len(content)

        return {
            'document_id': doc_id,
            'title': document.get('title', 'Untitled'),
            'outline': outline,
            'total_elements': len(elements),
            'element_types': element_types,
            'text_length': total_text_length
        }

    def get_documents_batch(self, doc_ids: List[str], include_full_text: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Get multiple complete documents in one call for efficiency.

        Args:
            doc_ids: List of document IDs to retrieve
            include_full_text: Whether to include full text content

        Returns:
            Dictionary mapping doc_id to complete document structure
        """
        results = {}

        for doc_id in doc_ids:
            complete_doc = self.get_complete_document(doc_id, include_full_text)
            if complete_doc:
                results[doc_id] = complete_doc

        return results

    def extract_document_content(self, doc_id: str, format_type: str = 'text') -> Optional[str]:
        """
        Extract and format document content while preserving basic structural relationships.

        This creates a readable representation of the content but does not preserve
        formatting, layout, or visual design from the original document. It focuses
        on extracting textual content with structural hints.

        Args:
            doc_id: Document ID
            format_type: Output format ('text', 'markdown', 'html')

        Returns:
            Formatted document content or None if not possible
        """
        elements = self.get_document_elements(doc_id)
        if not elements:
            return None

        # Get document metadata to determine source format
        document = self.get_document(doc_id)
        source_format = document.get('doc_type', '').lower() if document else 'unknown'

        # Sort elements by position/order if available
        if elements and any('position' in elem for elem in elements):
            elements.sort(key=lambda x: x.get('position', 0))

        # Choose extraction strategy based on format and source
        if format_type == 'text':
            return self._extract_document_content_as_text(elements, source_format)
        elif format_type == 'markdown':
            return self._format_content_as_markdown(elements, source_format)
        elif format_type == 'html':
            return self._structure_content_as_html(elements, source_format)
        else:
            raise ValueError(f"Unsupported format_type: {format_type}. Use 'text', 'markdown', or 'html'")

    # Deprecated method name for backward compatibility
    def reconstruct_original_document(self, doc_id: str, format_type: str = 'text') -> Optional[str]:
        """
        DEPRECATED: Use extract_document_content() instead.

        This method name is misleading as it suggests faithful reconstruction,
        which is not possible given that we only extract textual content.
        """
        return self.extract_document_content(doc_id, format_type)

    def _get_element_content_with_fallback(self, element: Dict[str, Any]) -> str:
        """Get element content with smart fallback and metadata integration."""
        content = element.get('full_content') or element.get('content_preview', '')

        # For elements that might not have text content, use metadata
        if not content:
            metadata = element.get('metadata', {})
            if 'alt_text' in metadata:
                content = f"[{metadata['alt_text']}]"
            elif 'title' in metadata:
                content = metadata['title']
            elif 'filename' in metadata:
                content = f"[{metadata['filename']}]"

        return content

    def _get_element_type_enum(self, element: Dict[str, Any]) -> ElementType:
        """Get ElementType enum from element data with fallback."""
        elem_type_str = element.get('element_type', '').lower()
        try:
            # Try to match the string to an enum value
            for elem_type in ElementType:
                if elem_type.value.lower() == elem_type_str:
                    return elem_type
            return ElementType.UNKNOWN
        except:
            return ElementType.UNKNOWN

    def _extract_document_content_as_text(self, elements: List[Dict[str, Any]], source_format: str = '') -> str:
        """Extract document content as plain text with structural preservation."""
        text_parts = []
        current_list_level = 0
        in_table = False
        slide_number = 0

        for i, element in enumerate(elements):
            elem_type = self._get_element_type_enum(element)
            content = self._get_element_content_with_fallback(element)
            metadata = element.get('metadata', {})

            if not content:
                continue

            # Format-specific element handling
            if source_format in ['pptx', 'ppt']:
                if elem_type == ElementType.SLIDE:
                    slide_number += 1
                    text_parts.append(f"\n--- SLIDE {slide_number} ---\n")
                    if content:
                        text_parts.append(f"{content}\n")
                    continue
                elif elem_type == ElementType.SLIDE_NOTES:
                    text_parts.append(f"\nSpeaker Notes: {content}\n")
                    continue

            elif source_format in ['docx', 'doc']:
                if elem_type == ElementType.PAGE_HEADER:
                    text_parts.append(f"[HEADER: {content}]\n")
                    continue
                elif elem_type == ElementType.PAGE_FOOTER:
                    text_parts.append(f"[FOOTER: {content}]\n")
                    continue

            elif source_format == 'pdf':
                if elem_type == ElementType.PAGE:
                    page_num = metadata.get('page_number', '?')
                    text_parts.append(f"\n--- PAGE {page_num} ---\n")
                    if content:
                        text_parts.append(f"{content}\n")
                    continue

            # Universal element type handling using enum
            if elem_type == ElementType.HEADER:
                level = metadata.get('level', 1)
                underline_char = '=' if level <= 1 else '-'
                text_parts.append(f"\n{content}\n{underline_char * min(len(content), 50)}\n")

            elif elem_type == ElementType.PARAGRAPH:
                text_parts.append(f"{content}\n")

            elif elem_type == ElementType.LIST_ITEM:
                indent = metadata.get('level', 0)
                bullet = '•' if metadata.get('list_type') != 'ordered' else f"{metadata.get('number', '1')}."
                text_parts.append(f"{'  ' * indent}{bullet} {content}")

            elif elem_type == ElementType.BLOCKQUOTE:
                text_parts.append(f"> {content}\n")

            elif elem_type in [ElementType.TABLE, ElementType.TABLE_ROW, ElementType.TABLE_CELL]:
                if elem_type == ElementType.TABLE and not in_table:
                    text_parts.append(f"\n[TABLE]")
                    in_table = True
                text_parts.append(f"{content}")
                if elem_type == ElementType.TABLE_ROW:
                    text_parts.append("| ")
                elif elem_type == ElementType.TABLE_CELL:
                    text_parts.append(" |")

            elif elem_type in [ElementType.IMAGE, ElementType.CHART, ElementType.SHAPE]:
                alt_text = metadata.get('alt_text', metadata.get('title', elem_type.value.upper()))
                size_info = ''
                if 'width' in metadata and 'height' in metadata:
                    size_info = f" ({metadata['width']}x{metadata['height']})"
                text_parts.append(f"[{alt_text.upper()}{size_info}]")

            elif elem_type == ElementType.CODE_BLOCK:
                lang = metadata.get('language', '')
                text_parts.append(f"\n[CODE{f' ({lang})' if lang else ''}]\n{content}\n[/CODE]\n")

            elif elem_type == ElementType.TEXT_BOX:
                text_parts.append(f"[TEXT BOX: {content}]")

            else:
                # Check if element ended a table
                if in_table and elem_type not in [ElementType.TABLE, ElementType.TABLE_ROW,
                                                 ElementType.TABLE_CELL, ElementType.TABLE_HEADER]:
                    text_parts.append(f"[/TABLE]\n")
                    in_table = False
                text_parts.append(content)

        # Close any open structures
        if in_table:
            text_parts.append(f"[/TABLE]")

        return '\n'.join(text_parts)

    def _format_content_as_markdown(self, elements: List[Dict[str, Any]], source_format: str = '') -> str:
        """Format document content as Markdown with structural preservation."""
        md_parts = []
        in_list = False
        in_table = False
        slide_number = 0
        table_headers = []

        for element in elements:
            elem_type = self._get_element_type_enum(element)
            content = self._get_element_content_with_fallback(element)
            metadata = element.get('metadata', {})

            if not content:
                continue

            # Close list if we're no longer in list items
            if in_list and elem_type not in [ElementType.LIST_ITEM, ElementType.LIST]:
                md_parts.append("")  # Add blank line after list
                in_list = False

            # Format-specific handling
            if source_format in ['pptx', 'ppt']:
                if elem_type == ElementType.SLIDE:
                    slide_number += 1
                    md_parts.append(f"\n---\n\n# Slide {slide_number}")
                    if content.strip():
                        md_parts.append(f"\n{content}\n")
                    continue
                elif elem_type == ElementType.SLIDE_NOTES:
                    md_parts.append(f"\n> **Speaker Notes:** {content}\n")
                    continue

            elif source_format in ['docx', 'doc']:
                if elem_type in [ElementType.PAGE_HEADER, ElementType.PAGE_FOOTER]:
                    element_name = elem_type.value.replace('_', ' ').title()
                    md_parts.append(f"\n*{element_name}: {content}*\n")
                    continue

            # Universal element handling using enum
            if elem_type == ElementType.HEADER:
                level = metadata.get('level', 1)
                md_parts.append(f"{'#' * min(level, 6)} {content}\n")

            elif elem_type == ElementType.PARAGRAPH:
                md_parts.append(f"{content}\n")

            elif elem_type == ElementType.LIST_ITEM:
                indent = metadata.get('level', 0)
                list_type = metadata.get('list_type', 'bullet')
                if list_type == 'ordered':
                    number = metadata.get('number', 1)
                    md_parts.append(f"{'  ' * indent}{number}. {content}")
                else:
                    md_parts.append(f"{'  ' * indent}- {content}")
                in_list = True

            elif elem_type == ElementType.BLOCKQUOTE:
                md_parts.append(f"> {content}\n")

            elif elem_type == ElementType.TABLE:
                in_table = True
                md_parts.append(f"\n<!-- Table: {metadata.get('title', 'Untitled')} -->")
                continue

            elif elem_type == ElementType.TABLE_HEADER_ROW:
                if in_table:
                    headers = content.split('|') if '|' in content else [content]
                    table_headers = [h.strip() for h in headers if h.strip()]
                    md_parts.append(f"| {' | '.join(table_headers)} |")
                    md_parts.append(f"| {' | '.join(['---'] * len(table_headers))} |")

            elif elem_type == ElementType.TABLE_ROW:
                if in_table:
                    cells = content.split('|') if '|' in content else [content]
                    cells = [c.strip() for c in cells if c.strip()]
                    md_parts.append(f"| {' | '.join(cells)} |")

            elif elem_type in [ElementType.IMAGE, ElementType.CHART, ElementType.SHAPE]:
                alt_text = metadata.get('alt_text', metadata.get('title', elem_type.value.title()))
                image_url = metadata.get('src', metadata.get('path', ''))
                if image_url:
                    md_parts.append(f"![{alt_text}]({image_url})")
                else:
                    md_parts.append(f"![{alt_text}]()")

            elif elem_type == ElementType.CODE_BLOCK:
                lang = metadata.get('language', '')
                md_parts.append(f"```{lang}\n{content}\n```\n")

            elif elem_type == ElementType.TEXT_BOX:
                md_parts.append(f"\n> **Text Box:** {content}\n")

            else:
                if in_table and elem_type not in [ElementType.TABLE, ElementType.TABLE_ROW,
                                                 ElementType.TABLE_CELL, ElementType.TABLE_HEADER]:
                    md_parts.append("")  # End table
                    in_table = False
                md_parts.append(content)

        return '\n'.join(md_parts)

    def _structure_content_as_html(self, elements: List[Dict[str, Any]], source_format: str = '') -> str:
        """Structure document content as HTML with basic layout preservation."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "    <meta charset=\"UTF-8\">",
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">",
            "    <title>Document Content</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }",
            "        .slide { border: 2px solid #ccc; margin: 20px 0; padding: 20px; page-break-after: always; }",
            "        .slide-notes { background: #f0f0f0; padding: 10px; margin: 10px 0; font-style: italic; }",
            "        .text-box { border: 1px solid #999; padding: 10px; margin: 10px 0; background: #f9f9f9; }",
            "        .page-break { border-top: 3px dashed #ccc; margin: 20px 0; text-align: center; color: #666; }",
            "        table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "        th { background-color: #f2f2f2; }",
            "    </style>",
            "</head>",
            "<body>"
        ]

        in_list = False
        list_type = 'ul'
        in_table = False
        slide_number = 0

        for element in elements:
            elem_type = self._get_element_type_enum(element)
            content = self._get_element_content_with_fallback(element)
            metadata = element.get('metadata', {})

            if not content:
                continue

            # Escape HTML content
            escaped_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            # Close lists when appropriate
            if in_list and elem_type not in [ElementType.LIST_ITEM, ElementType.LIST]:
                html_parts.append(f"</{list_type}>")
                in_list = False

            # Format-specific handling
            if source_format in ['pptx', 'ppt']:
                if elem_type == ElementType.SLIDE:
                    if slide_number > 0:  # Close previous slide
                        html_parts.append("</div>")
                    slide_number += 1
                    html_parts.append(f"<div class=\"slide\">")
                    html_parts.append(f"<h2>Slide {slide_number}</h2>")
                    if escaped_content.strip():
                        html_parts.append(f"<p>{escaped_content}</p>")
                    continue
                elif elem_type == ElementType.SLIDE_NOTES:
                    html_parts.append(f"<div class=\"slide-notes\"><strong>Speaker Notes:</strong> {escaped_content}</div>")
                    continue

            elif source_format in ['docx', 'doc']:
                if elem_type == ElementType.PAGE_HEADER:
                    html_parts.append(f"<header><em>Header: {escaped_content}</em></header>")
                    continue
                elif elem_type == ElementType.PAGE_FOOTER:
                    html_parts.append(f"<footer><em>Footer: {escaped_content}</em></footer>")
                    continue

            # Universal element handling using enum
            if elem_type == ElementType.HEADER:
                level = metadata.get('level', 1)
                html_parts.append(f"<h{min(level, 6)}>{escaped_content}</h{min(level, 6)}>")

            elif elem_type == ElementType.PARAGRAPH:
                html_parts.append(f"<p>{escaped_content}</p>")

            elif elem_type == ElementType.LIST_ITEM:
                new_list_type = 'ol' if metadata.get('list_type') == 'ordered' else 'ul'
                if not in_list:
                    html_parts.append(f"<{new_list_type}>")
                    in_list = True
                    list_type = new_list_type
                elif list_type != new_list_type:
                    html_parts.append(f"</{list_type}>")
                    html_parts.append(f"<{new_list_type}>")
                    list_type = new_list_type
                html_parts.append(f"<li>{escaped_content}</li>")

            elif elem_type == ElementType.BLOCKQUOTE:
                html_parts.append(f"<blockquote><p>{escaped_content}</p></blockquote>")

            elif elem_type == ElementType.TABLE:
                if not in_table:
                    title = metadata.get('title', '')
                    html_parts.append(f"<table>")
                    if title:
                        html_parts.append(f"<caption>{title}</caption>")
                    in_table = True

            elif elem_type == ElementType.TABLE_HEADER_ROW:
                if in_table:
                    html_parts.append("<thead><tr>")
                    cells = escaped_content.split('|') if '|' in escaped_content else [escaped_content]
                    for cell in cells:
                        if cell.strip():
                            html_parts.append(f"<th>{cell.strip()}</th>")
                    html_parts.append("</tr></thead><tbody>")

            elif elem_type == ElementType.TABLE_ROW:
                if in_table:
                    html_parts.append("<tr>")
                    cells = escaped_content.split('|') if '|' in escaped_content else [escaped_content]
                    for cell in cells:
                        if cell.strip():
                            html_parts.append(f"<td>{cell.strip()}</td>")
                    html_parts.append("</tr>")

            elif elem_type in [ElementType.IMAGE, ElementType.CHART, ElementType.SHAPE]:
                alt_text = metadata.get('alt_text', metadata.get('title', elem_type.value.title()))
                src = metadata.get('src', metadata.get('path', ''))
                width = metadata.get('width', '')
                height = metadata.get('height', '')
                style = f"width: {width}px; height: {height}px;" if width and height else ""
                html_parts.append(f"<img src=\"{src}\" alt=\"{alt_text}\" style=\"{style}\">")

            elif elem_type == ElementType.CODE_BLOCK:
                lang = metadata.get('language', '')
                html_parts.append(f"<pre><code class=\"language-{lang}\">{escaped_content}</code></pre>")

            elif elem_type == ElementType.TEXT_BOX:
                html_parts.append(f"<div class=\"text-box\">{escaped_content}</div>")

            else:
                if in_table and elem_type not in [ElementType.TABLE, ElementType.TABLE_ROW,
                                                 ElementType.TABLE_CELL, ElementType.TABLE_HEADER]:
                    html_parts.append("</tbody></table>")
                    in_table = False
                html_parts.append(f"<div class=\"{elem_type.value}\">{escaped_content}</div>")

        # Close any open structures
        if in_list:
            html_parts.append(f"</{list_type}>")
        if in_table:
            html_parts.append("</tbody></table>")
        if source_format in ['pptx', 'ppt'] and slide_number > 0:
            html_parts.append("</div>")  # Close last slide

        html_parts.extend(["</body>", "</html>"])
        return '\n'.join(html_parts)

    # ========================================
    # DOCUMENT FORMAT DETECTION AND CONVERSION UTILITIES
    # ========================================

    def get_document_format_info(self, doc_id: str) -> Dict[str, Any]:
        """
        Get detailed information about document format and element type distribution.

        Args:
            doc_id: Document ID

        Returns:
            Dictionary with format analysis:
            {
                'source_format': str,
                'detected_format': str,
                'element_distribution': {...},
                'format_specific_elements': [...],
                'reconstruction_recommendations': [...]
            }
        """
        document = self.get_document(doc_id)
        elements = self.get_document_elements(doc_id)

        if not document or not elements:
            return {}

        source_format = document.get('doc_type', '').lower()
        element_types = [elem.get('element_type', '') for elem in elements]
        element_distribution = {}

        for elem_type in element_types:
            element_distribution[elem_type] = element_distribution.get(elem_type, 0) + 1

        # Detect format based on element types if source format is unknown
        detected_format = self._detect_format_from_elements(element_types, source_format)

        # Find format-specific elements
        format_specific_elements = self._get_format_specific_elements(element_types, detected_format)

        # Generate reconstruction recommendations
        recommendations = self._get_reconstruction_recommendations(element_distribution, detected_format)

        return {
            'source_format': source_format,
            'detected_format': detected_format,
            'element_distribution': element_distribution,
            'format_specific_elements': format_specific_elements,
            'reconstruction_recommendations': recommendations
        }

    def _detect_format_from_elements(self, element_types: List[str], source_format: str) -> str:
        """Detect document format based on element type patterns using ElementType enum."""
        if source_format and source_format != 'unknown':
            return source_format

        # Convert string types to enum for comparison
        element_enums = []
        for elem_type_str in element_types:
            try:
                for elem_type in ElementType:
                    if elem_type.value.lower() == elem_type_str.lower():
                        element_enums.append(elem_type)
                        break
            except:
                continue

        # PowerPoint indicators
        pptx_indicators = [ElementType.SLIDE, ElementType.SLIDE_NOTES, ElementType.PRESENTATION_BODY]
        if any(elem in element_enums for elem in pptx_indicators):
            return 'pptx'

        # Word document indicators
        docx_indicators = [ElementType.PAGE_HEADER, ElementType.PAGE_FOOTER]
        if any(elem in element_enums for elem in docx_indicators):
            return 'docx'

        # PDF indicators
        pdf_indicators = [ElementType.PAGE]
        if any(elem in element_enums for elem in pdf_indicators):
            return 'pdf'

        # JSON indicators
        json_indicators = [ElementType.JSON_OBJECT, ElementType.JSON_ARRAY, ElementType.JSON_FIELD]
        if any(elem in element_enums for elem in json_indicators):
            return 'json'

        # XML indicators
        xml_indicators = [ElementType.XML_ELEMENT, ElementType.XML_TEXT, ElementType.XML_LIST]
        if any(elem in element_enums for elem in xml_indicators):
            return 'xml'

        # Default based on common elements
        if ElementType.PARAGRAPH in element_enums and ElementType.HEADER in element_enums:
            return 'document'  # Generic document

        return 'unknown'

    def _get_format_specific_elements(self, element_types: List[str], format_type: str) -> List[str]:
        """Get list of format-specific elements found in the document using ElementType enum."""
        # Convert string types to enum for comparison
        element_enums = set()
        for elem_type_str in element_types:
            try:
                for elem_type in ElementType:
                    if elem_type.value.lower() == elem_type_str.lower():
                        element_enums.add(elem_type)
                        break
            except:
                continue

        format_specific = {
            'pptx': {ElementType.SLIDE, ElementType.SLIDE_NOTES, ElementType.PRESENTATION_BODY,
                    ElementType.SLIDE_MASTERS, ElementType.SLIDE_LAYOUT, ElementType.SLIDE_MASTER,
                    ElementType.SHAPE, ElementType.SHAPE_GROUP},
            'docx': {ElementType.PAGE_HEADER, ElementType.PAGE_FOOTER, ElementType.COMMENT,
                    ElementType.TEXT_BOX, ElementType.BODY},
            'pdf': {ElementType.PAGE},
            'json': {ElementType.JSON_OBJECT, ElementType.JSON_ARRAY, ElementType.JSON_FIELD,
                    ElementType.JSON_ITEM},
            'xml': {ElementType.XML_ELEMENT, ElementType.XML_TEXT, ElementType.XML_LIST,
                   ElementType.XML_OBJECT},
            'html': {ElementType.BODY}  # Limited HTML-specific elements in the enum
        }

        specific_enums = format_specific.get(format_type, set())
        found_specific = element_enums.intersection(specific_enums)

        return [elem.value for elem in found_specific]

    def _get_reconstruction_recommendations(self, element_distribution: Dict[str, int],
                                           format_type: str) -> List[str]:
        """Generate recommendations for document reconstruction."""
        recommendations = []

        if format_type == 'pptx':
            slide_count = element_distribution.get('slide', 0)
            notes_count = element_distribution.get('slide_notes', 0)

            recommendations.append(f"PowerPoint presentation with {slide_count} slides detected")
            if notes_count > 0:
                recommendations.append(f"Speaker notes found on {notes_count} slides")
            recommendations.append("Recommend 'pptx_html' format for best presentation layout")
            recommendations.append("Use 'markdown' format for readable slide content export")

        elif format_type == 'docx':
            page_elements = sum(element_distribution.get(elem, 0)
                              for elem in ['page_header', 'page_footer', 'page_break'])
            if page_elements > 0:
                recommendations.append("Word document with page layout elements detected")
                recommendations.append("Recommend 'docx_html' format to preserve page structure")

            footnote_count = element_distribution.get('footnote', 0)
            if footnote_count > 0:
                recommendations.append(f"Document contains {footnote_count} footnotes")
                recommendations.append("Footnotes will be preserved in markdown and HTML formats")

        elif format_type == 'pdf':
            page_count = element_distribution.get('page', 0)
            if page_count > 0:
                recommendations.append(f"PDF with {page_count} pages detected")
                recommendations.append("Page breaks will be marked in reconstruction")

        # Table handling recommendations
        table_count = element_distribution.get('table', 0)
        if table_count > 0:
            recommendations.append(f"Document contains {table_count} tables")
            recommendations.append("Tables will be preserved in markdown and HTML formats")

        # Image handling recommendations
        image_count = sum(element_distribution.get(elem, 0)
                         for elem in ['image', 'chart', 'shape'])
        if image_count > 0:
            recommendations.append(f"Document contains {image_count} images/charts/shapes")
            recommendations.append("Images will become alt-text references in text format")
            recommendations.append("Use HTML format to preserve image references")

        return recommendations

    @staticmethod
    def get_element_type_mappings() -> Dict[str, Dict[ElementType, str]]:
        """
        Get mappings for converting ElementType enums between different output formats.

        Returns:
            Dictionary mapping output formats to ElementType conversions
        """
        return {
            'text': {
                ElementType.SLIDE: 'section',
                ElementType.SLIDE_NOTES: 'note',
                ElementType.PAGE_HEADER: 'header',
                ElementType.PAGE_FOOTER: 'footer',
                ElementType.IMAGE: 'reference',
                ElementType.CHART: 'reference',
                ElementType.SHAPE: 'reference',
                ElementType.TEXT_BOX: 'aside'
            },
            'markdown': {
                ElementType.SLIDE: 'header',
                ElementType.SLIDE_NOTES: 'blockquote',
                ElementType.PAGE_HEADER: 'italic',
                ElementType.PAGE_FOOTER: 'italic',
                ElementType.IMAGE: 'image',
                ElementType.CHART: 'image',
                ElementType.SHAPE: 'image',
                ElementType.TEXT_BOX: 'blockquote'
            },
            'html': {
                ElementType.SLIDE: 'section',
                ElementType.SLIDE_NOTES: 'aside',
                ElementType.PAGE_HEADER: 'header',
                ElementType.PAGE_FOOTER: 'footer',
                ElementType.IMAGE: 'img',
                ElementType.CHART: 'img',
                ElementType.SHAPE: 'img',
                ElementType.TEXT_BOX: 'div'
            }
        }

    def get_content_extraction_preview(self, doc_id: str, format_type: str = 'text',
                                      max_length: int = 1000) -> Optional[str]:
        """
        Get a preview of content extraction without generating the full output.

        Args:
            doc_id: Document ID
            format_type: Output format ('text', 'markdown', 'html')
            max_length: Maximum length of preview

        Returns:
            Preview of extracted content
        """
        full_extraction = self.extract_document_content(doc_id, format_type)
        if not full_extraction:
            return None

        if len(full_extraction) <= max_length:
            return full_extraction

        # Try to cut at a reasonable boundary
        preview = full_extraction[:max_length]

        # Find the last complete line, sentence, or word
        for boundary in ['\n\n', '\n', '. ', ' ']:
            last_boundary = preview.rfind(boundary)
            if last_boundary > max_length * 0.8:  # Don't cut too much
                preview = preview[:last_boundary + len(boundary)]
                break

        return preview + "\n[... content truncated ...]"

    # Deprecated method name for backward compatibility
    def get_reconstruction_preview(self, doc_id: str, format_type: str = 'text',
                                   max_length: int = 1000) -> Optional[str]:
        """DEPRECATED: Use get_content_extraction_preview() instead."""
        return self.get_content_extraction_preview(doc_id, format_type, max_length)

    def validate_content_extraction_capability(self, doc_id: str) -> Dict[str, Any]:
        """
        Validate how well document content can be extracted in different formats.

        Args:
            doc_id: Document ID

        Returns:
            Dictionary with content extraction capability assessment
        """
        document = self.get_document(doc_id)
        elements = self.get_document_elements(doc_id)

        if not document or not elements:
            return {'error': 'Document not found'}

        element_type_strs = [elem.get('element_type', '') for elem in elements]
        element_types = []
        for elem_type_str in element_type_strs:
            try:
                for elem_type in ElementType:
                    if elem_type.value.lower() == elem_type_str.lower():
                        element_types.append(elem_type)
                        break
            except:
                element_types.append(ElementType.UNKNOWN)

        has_full_text = any('full_content' in elem and elem['full_content']
                           for elem in elements)

        # Assess extraction quality for each format
        assessment = {
            'document_id': doc_id,
            'total_elements': len(elements),
            'has_full_text': has_full_text,
            'format_assessments': {}
        }

        # Text format assessment
        text_quality = 'high' if has_full_text else 'medium'
        text_supported = [ElementType.HEADER, ElementType.PARAGRAPH, ElementType.LIST_ITEM,
                         ElementType.BLOCKQUOTE, ElementType.CODE_BLOCK]
        text_unsupported = [ElementType.IMAGE, ElementType.CHART, ElementType.SHAPE]

        if any(elem_type in element_types for elem_type in text_unsupported):
            text_quality = 'medium'  # Images become references

        assessment['format_assessments']['text'] = {
            'quality': text_quality,
            'supported_elements': len([e for e in element_types if e in text_supported]),
            'unsupported_elements': len([e for e in element_types if e in text_unsupported])
        }

        # Markdown format assessment
        markdown_quality = 'high'
        markdown_supported = [ElementType.HEADER, ElementType.PARAGRAPH, ElementType.LIST_ITEM,
                             ElementType.BLOCKQUOTE, ElementType.CODE_BLOCK, ElementType.TABLE,
                             ElementType.IMAGE]
        markdown_support_count = len([e for e in element_types if e in markdown_supported])

        assessment['format_assessments']['markdown'] = {
            'quality': markdown_quality,
            'supported_elements': markdown_support_count,
            'table_support': ElementType.TABLE in element_types,
            'image_support': any(e in element_types for e in [ElementType.IMAGE, ElementType.CHART])
        }

        # HTML format assessment
        html_quality = 'high'  # HTML can handle most elements
        assessment['format_assessments']['html'] = {
            'quality': html_quality,
            'supported_elements': len(element_types),  # HTML can handle everything as divs
            'css_styling': True,
            'preserves_structure': True
        }

        return assessment

    # Deprecated method name for backward compatibility
    def validate_reconstruction_capability(self, doc_id: str) -> Dict[str, Any]:
        """DEPRECATED: Use validate_content_extraction_capability() instead."""
        return self.validate_content_extraction_capability(doc_id)

    def _reconstruct_as_docx_html(self, elements: List[Dict[str, Any]]) -> str:
        """Reconstruct as HTML optimized for DOCX-style documents."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <meta charset=\"UTF-8\">",
            "    <title>DOCX Document</title>",
            "    <style>",
            "        body { font-family: 'Times New Roman', serif; margin: 1in; line-height: 1.15; }",
            "        .page-header, .page-footer { border-bottom: 1px solid #ccc; padding: 10px 0; font-size: 90%; }",
            "        .footnote { font-size: 80%; vertical-align: super; }",
            "        .page-break { page-break-before: always; }",
            "        table { border-collapse: collapse; width: 100%; }",
            "        th, td { border: 1px solid black; padding: 4px; }",
            "    </style>",
            "</head>",
            "<body>"
        ]

        for element in elements:
            elem_type = element.get('element_type', '')
            content = self._get_element_content_with_fallback(element)
            metadata = element.get('metadata', {})

            if not content:
                continue

            escaped_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            if elem_type == 'page_header':
                html_parts.append(f"<div class=\"page-header\">{escaped_content}</div>")
            elif elem_type == 'page_footer':
                html_parts.append(f"<div class=\"page-footer\">{escaped_content}</div>")
            elif elem_type == 'page_break':
                html_parts.append(f"<div class=\"page-break\"></div>")
            elif elem_type == 'footnote':
                footnote_num = metadata.get('number', '?')
                html_parts.append(f"<span class=\"footnote\">{footnote_num}</span> {escaped_content}")
            else:
                # Use standard HTML reconstruction for other elements
                pass

        html_parts.extend(["</body>", "</html>"])
        return '\n'.join(html_parts)

    def _reconstruct_as_pptx_html(self, elements: List[Dict[str, Any]]) -> str:
        """Reconstruct as HTML optimized for PPTX-style presentations."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <meta charset=\"UTF-8\">",
            "    <title>PowerPoint Presentation</title>",
            "    <style>",
            "        body { font-family: 'Calibri', sans-serif; margin: 0; background: #f0f0f0; }",
            "        .slide { width: 800px; height: 600px; margin: 20px auto; background: white; ",
            "                padding: 40px; border: 1px solid #ccc; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }",
            "        .slide-title { font-size: 2em; font-weight: bold; margin-bottom: 20px; }",
            "        .slide-notes { background: #ffffcc; padding: 15px; margin-top: 20px; ",
            "                      border-left: 4px solid #ffcc00; font-style: italic; }",
            "        .shape, .text-box { border: 2px dashed #999; padding: 10px; margin: 10px 0; }",
            "    </style>",
            "</head>",
            "<body>"
        ]

        slide_number = 0
        current_slide_open = False

        for element in elements:
            elem_type = element.get('element_type', '')
            content = self._get_element_content_with_fallback(element)
            metadata = element.get('metadata', {})

            if not content:
                continue

            escaped_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            if elem_type == 'slide':
                if current_slide_open:
                    html_parts.append("</div>")
                slide_number += 1
                html_parts.append(f"<div class=\"slide\">")
                html_parts.append(f"<div class=\"slide-title\">Slide {slide_number}</div>")
                if escaped_content.strip():
                    html_parts.append(f"<p>{escaped_content}</p>")
                current_slide_open = True

            elif elem_type == 'slide_notes':
                html_parts.append(f"<div class=\"slide-notes\"><strong>Notes:</strong> {escaped_content}</div>")

            elif elem_type in ['shape', 'text_box']:
                shape_type = metadata.get('shape_type', 'box')
                html_parts.append(f"<div class=\"{elem_type}\" data-shape-type=\"{shape_type}\">{escaped_content}</div>")

            elif elem_type == 'chart':
                chart_type = metadata.get('chart_type', 'unknown')
                html_parts.append(f"<div class=\"chart\"><strong>[{chart_type.upper()} CHART]</strong><br>{escaped_content}</div>")

            else:
                # Handle other elements normally
                if elem_type == 'paragraph':
                    html_parts.append(f"<p>{escaped_content}</p>")
                elif elem_type.startswith('h') or elem_type in ['header', 'title']:
                    html_parts.append(f"<h3>{escaped_content}</h3>")
                else:
                    html_parts.append(f"<div class=\"{elem_type}\">{escaped_content}</div>")

        if current_slide_open:
            html_parts.append("</div>")

        html_parts.extend(["</body>", "</html>"])
        return '\n'.join(html_parts)

    def get_document_statistics(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed statistics about a document.

        Args:
            doc_id: Document ID

        Returns:
            Document statistics or None if not found:
            {
                'document_id': str,
                'total_elements': int,
                'total_characters': int,
                'total_words': int,
                'element_types': {...},
                'has_full_text': bool,
                'has_relationships': bool,
                'has_embeddings': bool,
                'has_dates': bool,
                'average_element_length': float,
                'longest_element_length': int
            }
        """
        document = self.get_document(doc_id)
        if not document:
            return None

        elements = self.get_document_elements(doc_id)
        relationships = self.get_document_relationships(doc_id)

        # Calculate statistics
        total_characters = 0
        total_words = 0
        element_types = {}
        has_full_text = False
        has_embeddings = False
        has_dates = False
        element_lengths = []

        for element in elements:
            # Count element types
            elem_type = element.get('element_type', 'unknown')
            element_types[elem_type] = element_types.get(elem_type, 0) + 1

            # Check for full text
            if 'full_content' in element and element['full_content']:
                has_full_text = True
                content = element['full_content']
            else:
                content = element.get('content_preview', '')

            # Count characters and words
            if content:
                content_len = len(content)
                total_characters += content_len
                element_lengths.append(content_len)
                total_words += len(content.split())

            # Check for embeddings
            element_id = element.get('id') or element.get('pk')
            if element_id:
                try:
                    if self.get_embedding(str(element_id)):
                        has_embeddings = True
                except:
                    pass  # Ignore errors checking embeddings

                # Check for dates
                try:
                    if self.get_element_dates(str(element_id)):
                        has_dates = True
                except:
                    pass  # Ignore errors checking dates

        # Calculate averages
        avg_element_length = sum(element_lengths) / len(element_lengths) if element_lengths else 0
        longest_element_length = max(element_lengths) if element_lengths else 0

        return {
            'document_id': doc_id,
            'total_elements': len(elements),
            'total_characters': total_characters,
            'total_words': total_words,
            'element_types': element_types,
            'has_full_text': has_full_text,
            'has_relationships': len(relationships) > 0,
            'has_embeddings': has_embeddings,
            'has_dates': has_dates,
            'average_element_length': round(avg_element_length, 2),
            'longest_element_length': longest_element_length
        }

    def export_document(self, doc_id: str, format_type: str = 'json',
                       include_full_text: bool = True) -> Optional[str]:
        """
        Export a complete document in the specified format.

        Args:
            doc_id: Document ID
            format_type: Export format ('json', 'yaml', 'xml')
            include_full_text: Whether to include full text content

        Returns:
            Serialized document data or None if not found
        """
        complete_doc = self.get_complete_document(doc_id, include_full_text)
        if not complete_doc:
            return None

        if format_type == 'json':
            import json
            return json.dumps(complete_doc, indent=2, default=str)
        elif format_type == 'yaml':
            try:
                import yaml
                return yaml.dump(complete_doc, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML is required for YAML export")
        elif format_type == 'xml':
            return self._export_as_xml(complete_doc)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _export_as_xml(self, complete_doc: Dict[str, Any]) -> str:
        """Export document as XML."""
        def dict_to_xml(d, root_name='item'):
            xml_parts = [f"<{root_name}>"]
            for key, value in d.items():
                if isinstance(value, dict):
                    xml_parts.append(dict_to_xml(value, key))
                elif isinstance(value, list):
                    xml_parts.append(f"<{key}>")
                    for item in value:
                        if isinstance(item, dict):
                            xml_parts.append(dict_to_xml(item, 'item'))
                        else:
                            xml_parts.append(f"<item>{str(item)}</item>")
                    xml_parts.append(f"</{key}>")
                else:
                    # Escape XML content
                    escaped_value = str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    xml_parts.append(f"<{key}>{escaped_value}</{key}>")
            xml_parts.append(f"</{root_name}>")
            return '\n'.join(xml_parts)

        return f'<?xml version="1.0" encoding="UTF-8"?>\n{dict_to_xml(complete_doc, "document")}'

    # ========================================
    # FULL-TEXT SEARCH CONFIGURATION AND UTILITIES
    # ========================================

    def supports_full_text_search(self) -> bool:
        """
        Indicate whether this backend supports full-text search capabilities.

        This is used by the base class to provide consistent behavior across backends.
        Backends with full-text search should override this to return True.

        Returns:
            True if backend supports full-text search, False otherwise

        Example:
            ```python
            # In PostgreSQL implementation
            def supports_full_text_search(self) -> bool:
                return True  # PostgreSQL has excellent full-text search

            # In simple key-value store implementation
            def supports_full_text_search(self) -> bool:
                return False  # Only basic string matching
            ```
        """
        return SearchCapability.FULL_TEXT_SEARCH in self.get_backend_capabilities().supported

    def get_text_storage_config(self) -> Dict[str, Any]:
        """
        Get current text storage and indexing configuration.

        This provides a consistent interface across all backends to inspect
        the current full-text configuration and capabilities.

        Returns:
            Dictionary with current text storage settings:
            {
                'store_full_text': bool,
                'index_full_text': bool,
                'compress_full_text': bool,
                'full_text_max_length': int | None,
                'search_capabilities': {
                    'can_search_full_text': bool,
                    'can_retrieve_full_text': bool,
                    'search_fields': List[str]
                }
            }

        Example:
            ```python
            config = db.get_text_storage_config()
            if config['search_capabilities']['can_search_full_text']:
                print("Full-text search enabled")
            print(f"Available search fields: {config['search_capabilities']['search_fields']}")
            ```
        """
        return {
            'store_full_text': self.store_full_text,
            'index_full_text': self.index_full_text,
            'compress_full_text': self.compress_full_text,
            'full_text_max_length': self.full_text_max_length,
            'search_capabilities': {
                'can_search_full_text': self.index_full_text and self.supports_full_text_search(),
                'can_retrieve_full_text': self.store_full_text,
                'search_fields': self._get_available_search_fields()
            }
        }

    def get_storage_size_estimate(self) -> Dict[str, str]:
        """
        Get estimated storage usage based on current configuration.

        This helps users understand the storage implications of their
        full-text configuration choices.

        Returns:
            Dictionary with storage estimates:
            {
                'full_text_storage': str,    # 'High', 'Medium', 'Low', 'None'
                'full_text_index': str,      # 'High', 'Medium', 'Low', 'None'
                'overall_storage': str,      # Overall assessment
                'compression_enabled': bool
            }

        Example:
            ```python
            estimates = db.get_storage_size_estimate()
            print(f"Storage impact: {estimates['overall_storage']}")
            if estimates['compression_enabled']:
                print("Compression is helping reduce storage usage")
            ```
        """
        estimates = {
            'full_text_storage': 'High' if self.store_full_text else 'None',
            'full_text_index': 'High' if (self.index_full_text and self.supports_full_text_search()) else 'None',
            'compression_enabled': self.compress_full_text
        }

        # Calculate overall storage impact
        if not self.store_full_text and not (self.index_full_text and self.supports_full_text_search()):
            estimates['overall_storage'] = 'Minimal (preview only)'
        elif not self.store_full_text:
            estimates['overall_storage'] = 'Medium (search index only)'
        elif not (self.index_full_text and self.supports_full_text_search()):
            estimates['overall_storage'] = 'Medium (storage only)'
        elif self.compress_full_text:
            estimates['overall_storage'] = 'High (compressed full text)'
        else:
            estimates['overall_storage'] = 'High (full text + search)'

        return estimates

    def _get_available_search_fields(self) -> List[str]:
        """
        Get list of fields available for text search based on configuration.

        Returns:
            List of field names that can be searched
        """
        fields = ['content_preview']  # Always available

        if self.index_full_text and self.supports_full_text_search():
            fields.append('full_text')

        return fields

    def get_full_text_usage_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for optimizing full-text usage based on current configuration.

        Returns:
            Dictionary with optimization recommendations
        """
        config = self.get_text_storage_config()
        storage = self.get_storage_size_estimate()

        recommendations = {
            'current_config': 'optimal',
            'suggestions': [],
            'warnings': []
        }

        # Check for common configuration issues
        if self.index_full_text and not self.supports_full_text_search():
            recommendations['warnings'].append(
                "index_full_text=True but backend doesn't support full-text search. "
                "Consider setting index_full_text=False to avoid confusion."
            )

        if not self.store_full_text and not self.index_full_text:
            recommendations['suggestions'].append(
                "Minimal configuration detected. Consider enabling store_full_text=True "
                "if you need to retrieve original content."
            )

        if self.store_full_text and not self.index_full_text and self.supports_full_text_search():
            recommendations['suggestions'].append(
                "Consider enabling index_full_text=True to enable full-text search capabilities."
            )

        if storage['overall_storage'] == 'High' and not self.compress_full_text:
            recommendations['suggestions'].append(
                "High storage usage detected. Consider enabling compress_full_text=True "
                "to reduce storage requirements."
            )

        if self.full_text_max_length is None:
            recommendations['suggestions'].append(
                "No length limit set for full text. Consider setting full_text_max_length "
                "to prevent very large documents from consuming excessive storage."
            )

        return recommendations

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
    # EXISTING HELPER METHODS
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
    # TOPIC SUPPORT METHODS
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
    # DATE UTILITY METHODS
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
    # HIERARCHY METHODS
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
