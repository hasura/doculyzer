from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

from .element_element import ElementBase
from .element_element import ElementHierarchical
from .element_relationship import ElementRelationship


class DocumentDatabase(ABC):
    """Abstract base class for document database implementations."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the database."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass

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
    def find_documents(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find documents matching query.

        Args:
            query: Query parameters
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        pass

    @abstractmethod
    def find_elements(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements matching query.

        Args:
            query: Query parameters
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
                             (e.g. {"element_type": ["header", "section"]})

        Returns:
            List of (element_id, similarity_score) tuples for matching elements
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

    @abstractmethod
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
            List of (element_id, similarity_score) tuples
        """
        pass

    @abstractmethod
    def get_outgoing_relationships(self, element_pk: int) -> List[ElementRelationship]:
        """Find all relationships where the specified element_pk is the source."""
        pass

    # ========================================
    # NEW: SIMPLIFIED TOPIC SUPPORT
    # ========================================

    def supports_topics(self) -> bool:
        """
        Indicate whether this backend supports topic-aware embeddings.

        Returns:
            True if topics are supported, False otherwise
        """
        return False  # Default: no topic support

    def store_embedding_with_topics(self, element_pk: int, embedding: List[float],
                                    topics: List[str], confidence: float = 1.0) -> None:
        """
        Store embedding for an element with topic assignments.

        Args:
            element_pk: Element primary key
            embedding: Vector embedding
            topics: List of topic strings (e.g., ['security.policy', 'compliance'])
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
        Search elements by text with topic filtering using LIKE patterns.

        Args:
            search_text: Text to search for semantically (optional)
            include_topics: Topic LIKE patterns to include (e.g., ['security%', '%.policy%'])
            exclude_topics: Topic LIKE patterns to exclude (e.g., ['deprecated%'])
            min_confidence: Minimum confidence threshold for embeddings
            limit: Maximum number of results

        Returns:
            List of dictionaries with keys:
            - element_pk: Element primary key
            - similarity: Similarity score (if search_text provided)
            - confidence: Overall embedding confidence
            - topics: List of assigned topic strings
        """
        # Default implementation: fallback to regular text search
        if include_topics or exclude_topics:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Topic filtering not supported by this backend, performing regular search")

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
            # No search text and no topic support - return empty
            return []

    def get_topic_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about topic distribution across embeddings.

        Returns:
            Dictionary mapping topic strings to statistics:
            {
                'security.policy': {
                    'embedding_count': int,
                    'document_count': int,
                    'avg_embedding_confidence': float
                }
            }
        """
        return {}  # Default: empty statistics

    def get_embedding_topics(self, element_pk: int) -> List[str]:
        """
        Get topics assigned to a specific embedding.

        Args:
            element_pk: Element primary key

        Returns:
            List of topic strings assigned to this embedding
        """
        return []  # Default: no topics

    # ========================================
    # EXISTING HIERARCHY METHODS (unchanged)
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

        # Convert to ElementElement instance
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

            # Convert to ElementElement
            parent = ElementBase(**parent_dict)

            # Add to visited set
            visited.add(parent_pk)

            # Add parent to the beginning of the ancestry list (root first)
            ancestry.insert(0, parent)

            # Move up to the parent
            current_pk = parent_id

        return ancestry
