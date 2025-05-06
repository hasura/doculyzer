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
        """
        Find all relationships where the specified element_pk is the source.

        This method returns only the outgoing relationships - those where
        the specified element is the source and points to other elements or references.

        The returned relationships may include:
        - Structural relationships (contains, next_sibling, etc.)
        - Explicit links (links to other elements)
        - Semantic similarity relationships (similar content)

        Args:
            element_pk: The primary key of the element

        Returns:
            List of ElementRelationship objects where the specified element is the source
        """
        pass

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
        element = ElementBase.from_dict(element_dict)

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
            parent = ElementBase.from_dict(parent_dict)

            # Add to visited set
            visited.add(parent_pk)

            # Add parent to the beginning of the ancestry list (root first)
            ancestry.insert(0, parent)

            # Move up to the parent
            current_pk = parent_id

        return ancestry
