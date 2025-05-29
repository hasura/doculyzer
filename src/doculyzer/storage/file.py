import glob
import json
import logging
import os
import fnmatch
from typing import Optional, Dict, Any, List, Tuple, Union, TYPE_CHECKING

import time

# Import types for type checking only
if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    # Define type aliases for type checking
    VectorType = NDArray[np.float32]
else:
    # Runtime type aliases - use generic Python types
    VectorType = List[float]

from .base import DocumentDatabase
from .element_relationship import ElementRelationship
from .element_element import ElementType, ElementBase  # Import existing enum and ElementBase

logger = logging.getLogger(__name__)

# Define global flags for availability - these will be set at runtime
NUMPY_AVAILABLE = False

# Try to import NumPy conditionally
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("NumPy not available. Will use slower pure Python vector operations.")

# Try to import the config the same way SQLite does
try:
    from ..config import Config

    config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
except Exception as e:
    logger.warning(f"Error configuring File provider: {str(e)}")
    config = None


class FileDocumentDatabase(DocumentDatabase):
    """File-based implementation of document database."""

    def get_outgoing_relationships(self, element_pk: int) -> List[ElementRelationship]:
        """
        Find all relationships where the specified element_pk is the source.

        Implementation for File-based database, optimized to look up target elements efficiently.

        Args:
            element_pk: The primary key of the element

        Returns:
            List of ElementRelationship objects where the specified element is the source
        """
        relationships = []

        # Get the element to find its element_id and type
        element = self.get_element(element_pk)
        if not element:
            logger.warning(f"Element with PK {element_pk} not found")
            return []

        element_id = element.get("element_id")
        if not element_id:
            logger.warning(f"Element with PK {element_pk} has no element_id")
            return []

        element_type = element.get("element_type", "")

        # Create a lookup map of element_id to (element_pk, element_type, content_preview)
        # This is similar to what we get with SQL JOIN
        element_lookup = {}
        for eid, elem in self.elements.items():
            if "element_pk" in elem:
                element_lookup[eid] = (
                    elem.get("element_pk"),
                    elem.get("element_type"),
                    elem.get("content_preview", "")  # Added content_preview
                )

        # Find relationships where the element is the source (outgoing relationships)
        outgoing_relationships = [
            rel for rel in self.relationships.values()
            if rel.get("source_id") == element_id
        ]

        # Process relationships with target element lookup
        for rel in outgoing_relationships:
            target_reference = rel.get("target_reference", "")
            target_element_pk = None
            target_element_type = None
            target_content_preview = None  # Added this field

            # Look up target element information if available
            if target_reference in element_lookup:
                target_element_pk, target_element_type, target_content_preview = element_lookup[target_reference]

            # Create enriched relationship
            relationship = ElementRelationship(
                relationship_id=rel.get("relationship_id", ""),
                source_id=element_id,
                source_element_pk=element_pk,
                source_element_type=element_type,
                relationship_type=rel.get("relationship_type", ""),
                target_reference=target_reference,
                target_element_pk=target_element_pk,
                target_element_type=target_element_type,
                target_content_preview=target_content_preview,  # Added this field
                doc_id=rel.get("doc_id"),
                metadata=rel.get("metadata", {}),
                is_source=True
            )

            relationships.append(relationship)

        return relationships

    def __init__(self, storage_path: str):
        """
        Initialize file-based document database.

        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = storage_path
        self.documents = {}
        self.elements = {}
        self.element_pks = {}  # Map element_id to element_pk
        self.next_element_pk = 1  # Starting auto-increment value
        self.relationships = {}
        self.embeddings = {}  # Now stores enhanced embedding data with topics
        self.processing_history = {}  # Dictionary to track processing history
        self.embedding_generator = None

    def initialize(self) -> None:
        """Initialize the database by loading existing data."""
        os.makedirs(self.storage_path, exist_ok=True)

        # Create subdirectories
        os.makedirs(os.path.join(self.storage_path, 'documents'), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, 'elements'), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, 'relationships'), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, 'embeddings'), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, 'processing_history'), exist_ok=True)

        # Load existing data if available
        self._load_documents()
        self._load_elements()
        self._load_relationships()
        self._load_embeddings()
        self._load_processing_history()

        logger.info(f"Loaded {len(self.documents)} documents, "
                    f"{len(self.elements)} elements, "
                    f"{len(self.relationships)} relationships, "
                    f"{len(self.embeddings)} embeddings, "
                    f"{len(self.processing_history)} processing history records")

    def close(self) -> None:
        """Close the database (no-op for file-based database)."""
        pass

    def get_last_processed_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about when a document was last processed."""
        # Convert source_id to a safe filename
        safe_id = self._get_safe_filename(source_id)

        # Look up in processing history
        return self.processing_history.get(safe_id)

    def update_processing_history(self, source_id: str, content_hash: str) -> None:
        """Update the processing history for a document."""
        # Convert source_id to a safe filename
        safe_id = self._get_safe_filename(source_id)

        # Create history record
        history_record = {
            "source_id": source_id,
            "content_hash": content_hash,
            "last_modified": time.time(),
            "processing_count": self.processing_history.get(safe_id, {}).get("processing_count", 0) + 1
        }

        # Store in memory and on disk
        self.processing_history[safe_id] = history_record
        self._save_processing_history(safe_id)

        logger.debug(f"Updated processing history for {source_id}")

    def store_document(self, document: Dict[str, Any], elements: List[Dict[str, Any]],
                       relationships: List[Dict[str, Any]]) -> None:
        """Store a document with its elements and relationships."""
        doc_id = document["doc_id"]
        source_id = document.get("source", "")
        content_hash = document.get("content_hash", "")

        # Store document
        self.documents[doc_id] = document
        self._save_document(doc_id)

        # Store elements
        for element in elements:
            element_id = element["element_id"]

            # Assign an auto-increment element_pk
            element_pk = self.next_element_pk
            self.next_element_pk += 1
            element["element_pk"] = element_pk

            # Store mapping
            self.element_pks[element_id] = element_pk

            # Store element
            self.elements[element_id] = element
            self._save_element(element_id)

        # Store relationships
        for relationship in relationships:
            relationship_id = relationship["relationship_id"]
            self.relationships[relationship_id] = relationship
            self._save_relationship(relationship_id)

        # Update processing history
        if source_id:
            self.update_processing_history(source_id, content_hash)

    def update_document(self, doc_id: str, document: Dict[str, Any],
                        elements: List[Dict[str, Any]],
                        relationships: List[Dict[str, Any]]) -> None:
        """Update an existing document."""
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        source_id = document.get("source", "")
        content_hash = document.get("content_hash", "")

        # Update document
        self.documents[doc_id] = document
        self._save_document(doc_id)

        # Get existing elements and relationships for this document
        existing_elements = {e["element_id"]: e for e in self.get_document_elements(doc_id)}
        existing_relationships = {r["relationship_id"]: r
                                  for r in self.get_document_relationships(doc_id)}

        # Update elements - retain unchanged ones
        updated_element_ids = set()
        for element in elements:
            element_id = element["element_id"]
            updated_element_ids.add(element_id)

            if element_id in existing_elements:
                # Check if element has changed
                if self._has_element_changed(element, existing_elements[element_id]):
                    # Preserve the element_pk from the existing element
                    element["element_pk"] = existing_elements[element_id].get("element_pk")
                    self.elements[element_id] = element
                    self._save_element(element_id)
            else:
                # New element - assign an auto-increment element_pk
                element_pk = self.next_element_pk
                self.next_element_pk += 1
                element["element_pk"] = element_pk

                # Store mapping
                self.element_pks[element_id] = element_pk

                # Store element
                self.elements[element_id] = element
                self._save_element(element_id)

        # Remove elements that no longer exist
        for element_id in existing_elements:
            if element_id not in updated_element_ids:
                if element_id in self.elements:
                    # Get the element_pk for embedding deletion
                    element_pk = self.elements[element_id].get("element_pk")

                    # Delete element
                    del self.elements[element_id]
                    self._delete_element_file(element_id)

                    # Remove from element_pks mapping
                    if element_id in self.element_pks:
                        del self.element_pks[element_id]

                    # Also remove any embeddings
                    if element_pk in self.embeddings:
                        del self.embeddings[element_pk]
                        self._delete_embedding_file(element_pk)

        # Update relationships - similar approach
        updated_relationship_ids = set()
        for relationship in relationships:
            relationship_id = relationship["relationship_id"]
            updated_relationship_ids.add(relationship_id)

            if relationship_id in existing_relationships:
                # Check if relationship has changed
                if self._has_relationship_changed(relationship, existing_relationships[relationship_id]):
                    self.relationships[relationship_id] = relationship
                    self._save_relationship(relationship_id)
            else:
                # New relationship
                self.relationships[relationship_id] = relationship
                self._save_relationship(relationship_id)

        # Remove relationships that no longer exist
        for relationship_id in existing_relationships:
            if relationship_id not in updated_relationship_ids:
                if relationship_id in self.relationships:
                    del self.relationships[relationship_id]
                    self._delete_relationship_file(relationship_id)

        # Update processing history
        if source_id:
            self.update_processing_history(source_id, content_hash)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        return self.documents.get(doc_id)

    def get_document_elements(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get elements for a document by doc_id or source.
        Updated to match other implementations.
        """
        # First try to find document by doc_id
        document = self.documents.get(doc_id)

        # If not found by doc_id, try to find by source
        if not document:
            for doc in self.documents.values():
                if doc.get("source") == doc_id:
                    document = doc
                    doc_id = document["doc_id"]
                    break

        # If still not found, return empty list
        if not document:
            return []

        return [element for element in self.elements.values()
                if element.get("doc_id") == doc_id]

    def get_document_relationships(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get relationships for a document."""
        # First get all element IDs for the document
        element_ids = {element["element_id"] for element in self.get_document_elements(doc_id)}

        # Find relationships involving these elements
        return [relationship for relationship in self.relationships.values()
                if relationship.get("source_id") in element_ids]

    def store_relationship(self, relationship: Dict[str, Any]) -> None:
        """
        Store a relationship between elements.

        Args:
            relationship: Relationship data with source_id, relationship_type, and target_reference
        """
        relationship_id = relationship["relationship_id"]
        self.relationships[relationship_id] = relationship
        self._save_relationship(relationship_id)
        logger.debug(f"Stored relationship {relationship_id}")

    def delete_relationships_for_element(self, element_id: str, relationship_type: str = None) -> None:
        """
        Delete relationships for an element.

        Args:
            element_id: Element ID
            relationship_type: Optional relationship type to filter by
        """
        # Find all relationships where this element is the source
        source_relationships = [rel_id for rel_id, rel in self.relationships.items()
                                if rel.get("source_id") == element_id and
                                (relationship_type is None or rel.get("relationship_type") == relationship_type)]

        # Find all relationships where this element is the target
        target_relationships = [rel_id for rel_id, rel in self.relationships.items()
                                if rel.get("target_reference") == element_id and
                                (relationship_type is None or rel.get("relationship_type") == relationship_type)]

        # Delete all matching relationships
        for rel_id in source_relationships + target_relationships:
            if rel_id in self.relationships:
                del self.relationships[rel_id]
                self._delete_relationship_file(rel_id)

        logger.debug(f"Deleted relationships for element {element_id}")

    def get_element(self, element_id_or_pk: Union[str, int]) -> Optional[Dict[str, Any]]:
        """
        Get element by ID or PK.
        Updated to handle either element_id or element_pk.

        Args:
            element_id_or_pk: Either the element_id (string) or element_pk (integer)
        """
        # Try to interpret as element_pk first
        try:
            element_pk = int(element_id_or_pk)
            # Find element with matching element_pk
            for element in self.elements.values():
                if element.get("element_pk") == element_pk:
                    return element
        except (ValueError, TypeError):
            # If not an integer, treat as element_id
            if element_id_or_pk in self.elements:
                return self.elements[element_id_or_pk]

        return None

    def find_documents(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find documents matching query with support for LIKE patterns.

        Args:
            query: Query parameters. Enhanced syntax supports:
                   - Exact matches: {"doc_type": "pdf"}
                   - LIKE patterns: {"source_like": "%reports%"} (converted to fnmatch)
                   - Case-insensitive LIKE: {"source_ilike": "%REPORTS%"}
                   - List matching: {"doc_type": ["pdf", "docx"]}
                   - Metadata exact: {"metadata": {"author": "John"}}
                   - Metadata LIKE: {"metadata_like": {"title": "%annual%"}}
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        if query is None:
            query = {}

        results = []

        for doc in self.documents.values():
            if self._matches_document_query(doc, query):
                results.append(doc)
                if len(results) >= limit:
                    break

        return results

    def find_elements(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements matching query with support for LIKE patterns and ElementType enums.

        Args:
            query: Query parameters. Enhanced syntax supports:
                   - Exact matches: {"element_type": "header"}
                   - ElementType enums: {"element_type": ElementType.HEADER}
                   - Multiple enums: {"element_type": [ElementType.HEADER, ElementType.PARAGRAPH]}
                   - LIKE patterns: {"content_preview_like": "%important%"}
                   - Case-insensitive LIKE: {"content_preview_ilike": "%IMPORTANT%"}
                   - List matching: {"doc_id": ["doc1", "doc2"]}
                   - Metadata exact: {"metadata": {"section": "intro"}}
                   - Metadata LIKE: {"metadata_like": {"title": "%chapter%"}}
            limit: Maximum number of results

        Returns:
            List of matching elements
        """
        if query is None:
            query = {}

        results = []

        for element in self.elements.values():
            if self._matches_element_query(element, query):
                results.append(element)
                if len(results) >= limit:
                    break

        return results

    def search_elements_by_content(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search elements by content preview."""
        results = []
        search_text_lower = search_text.lower()

        for element in self.elements.values():
            content_preview = element.get("content_preview", "").lower()

            if search_text_lower in content_preview:
                results.append(element)

                if len(results) >= limit:
                    break

        return results

    def store_embedding(self, element_pk: int, embedding: VectorType) -> None:
        """Store embedding for an element."""
        # Verify element pk exists in some element
        found = False
        for element in self.elements.values():
            if element.get("element_pk") == element_pk:
                found = True
                break

        if not found:
            raise ValueError(f"Element pk not found: {element_pk}")

        # Store as enhanced embedding structure
        embedding_data = {
            "embedding": embedding,
            "dimensions": len(embedding),
            "topics": [],  # Default to empty topics
            "confidence": 1.0,  # Default confidence
            "created_at": time.time()
        }

        self.embeddings[element_pk] = embedding_data
        self._save_embedding(element_pk)

    def get_embedding(self, element_pk: int) -> Optional[VectorType]:
        """Get embedding for an element."""
        embedding_data = self.embeddings.get(element_pk)
        if embedding_data is None:
            return None

        # Handle both old format (direct list) and new format (dict with metadata)
        if isinstance(embedding_data, list):
            return embedding_data
        elif isinstance(embedding_data, dict):
            return embedding_data.get("embedding")

        return None

    def search_by_embedding(self, query_embedding: VectorType, limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by embedding similarity with optional filtering.
        Updated to handle enhanced embedding format and better similarity calculation.
        """
        if NUMPY_AVAILABLE:
            query_embedding_np = np.array(query_embedding)

        results = []

        # Get all element_pks with embeddings
        element_pks_with_embeddings = list(self.embeddings.keys())

        # Build a dict of element_pk to element for easier lookup
        element_pk_to_element = {}
        for element in self.elements.values():
            if "element_pk" in element:
                element_pk_to_element[element["element_pk"]] = element

        # Apply filtering if specified
        if filter_criteria:
            filtered_element_pks = []
            for element_pk in element_pks_with_embeddings:
                # Get the associated element
                element = element_pk_to_element.get(element_pk)
                if not element:
                    continue

                # Check if element matches all criteria
                if self._matches_filter_criteria(element, filter_criteria):
                    filtered_element_pks.append(element_pk)
        else:
            filtered_element_pks = element_pks_with_embeddings

        # Calculate similarity for each filtered embedding
        for element_pk in filtered_element_pks:
            embedding_data = self.embeddings[element_pk]

            # Handle both old format (direct list) and new format (dict with metadata)
            if isinstance(embedding_data, list):
                embedding = embedding_data
            elif isinstance(embedding_data, dict):
                embedding = embedding_data.get("embedding", [])
            else:
                continue

            # Calculate cosine similarity using the appropriate method
            if NUMPY_AVAILABLE:
                similarity = self._cosine_similarity_numpy(query_embedding, embedding)
            else:
                similarity = self._cosine_similarity_python(query_embedding, embedding)

            # Return element_pk instead of element_id for consistency
            results.append((element_pk, similarity))

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all associated elements and relationships."""
        if doc_id not in self.documents:
            return False

        # Get elements to delete
        elements_to_delete = self.get_document_elements(doc_id)
        element_ids = {element["element_id"] for element in elements_to_delete}
        element_pks = {element.get("element_pk") for element in elements_to_delete if
                       element.get("element_pk") is not None}

        # Get relationships to delete
        relationships_to_delete = self.get_document_relationships(doc_id)
        relationship_ids = {rel["relationship_id"] for rel in relationships_to_delete}

        # Delete elements
        for element_id in element_ids:
            if element_id in self.elements:
                # Remove element_pk mapping
                if element_id in self.element_pks:
                    del self.element_pks[element_id]

                # Delete element
                del self.elements[element_id]
                self._delete_element_file(element_id)

        # Delete embeddings
        for element_pk in element_pks:
            if element_pk in self.embeddings:
                del self.embeddings[element_pk]
                self._delete_embedding_file(element_pk)

        # Delete relationships
        for relationship_id in relationship_ids:
            if relationship_id in self.relationships:
                del self.relationships[relationship_id]
                self._delete_relationship_file(relationship_id)

        # Delete document
        del self.documents[doc_id]
        self._delete_document_file(doc_id)

        return True

    def search_by_text(self, search_text: str, limit: int = 10,
                       filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by semantic similarity to the provided text.
        Updated to return (element_pk, similarity) tuples for consistency.

        Args:
            search_text: Text to search for semantically
            limit: Maximum number of results
            filter_criteria: Optional dictionary with criteria to filter results

        Returns:
            List of (element_pk, similarity_score) tuples
        """
        try:
            if self.embedding_generator is None:
                from ..embeddings import get_embedding_generator
                self.embedding_generator = get_embedding_generator(config)

            # Generate embedding for the search text
            query_embedding = self.embedding_generator.generate(search_text)

            # Use the embedding to search, passing the filter criteria
            return self.search_by_embedding(query_embedding, limit, filter_criteria)

        except Exception as e:
            logger.error(f"Error in semantic search by text: {str(e)}")
            # Return empty list on error
            return []

    # ========================================
    # NEW: ENHANCED SEARCH HELPER METHODS
    # ========================================

    @staticmethod
    def supports_like_patterns() -> bool:
        """
        Indicate whether this backend supports LIKE pattern matching.

        Returns:
            True since File implementation supports fnmatch patterns
        """
        return True

    @staticmethod
    def supports_case_insensitive_like() -> bool:
        """
        Indicate whether this backend supports case-insensitive LIKE (ILIKE).

        Returns:
            True since File implementation can handle case-insensitive patterns
        """
        return True

    @staticmethod
    def supports_element_type_enums() -> bool:
        """
        Indicate whether this backend supports ElementType enum integration.

        Returns:
            True since File implementation supports ElementType enums
        """
        return True

    def _prepare_element_type_query(self, element_types: Union[
        ElementType,
        List[ElementType],
        str,
        List[str],
        None
    ]) -> Optional[List[str]]:
        """
        Prepare element type values for queries using existing ElementType enum.

        Args:
            element_types: ElementType enum(s), string(s), or None

        Returns:
            List of string values for query, or None
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

        Examples:
            find_elements_by_category("text_elements")
            find_elements_by_category("table_elements", content_preview_like="%data%")
        """
        categories = self.get_element_types_by_category()

        if category not in categories:
            available = list(categories.keys())
            raise ValueError(f"Unknown category: {category}. Available: {available}")

        element_types = categories[category]
        query = {"element_type": element_types}
        query.update(other_filters)

        return self.find_elements(query)

    def find_elements_ilike(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements with case-insensitive LIKE support.

        File implementation supports case-insensitive patterns natively.

        Args:
            query: Query parameters with _ilike suffix support
            limit: Maximum number of results

        Returns:
            List of matching elements
        """
        # File implementation supports case-insensitive patterns, so just use regular find_elements
        return self.find_elements(query, limit)

    def _convert_like_to_fnmatch(self, like_pattern: str) -> str:
        """
        Convert SQL LIKE pattern to fnmatch pattern.

        Args:
            like_pattern: SQL LIKE pattern (e.g., "%abc%", "abc_def")

        Returns:
            fnmatch pattern
        """
        # Convert % to * (match any characters)
        pattern = like_pattern.replace('%', '*')
        # Convert _ to ? (match single character)
        pattern = pattern.replace('_', '?')
        return pattern

    def _matches_document_query(self, doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if a document matches the query criteria."""
        for key, value in query.items():
            if key == "metadata":
                # Handle metadata exact matches
                for meta_key, meta_value in value.items():
                    if meta_key not in doc.get("metadata", {}) or doc["metadata"][meta_key] != meta_value:
                        return False
            elif key == "metadata_like":
                # Handle metadata LIKE patterns
                for meta_key, meta_value in value.items():
                    if meta_key not in doc.get("metadata", {}):
                        return False
                    pattern = self._convert_like_to_fnmatch(meta_value)
                    if not fnmatch.fnmatch(str(doc["metadata"][meta_key]), pattern):
                        return False
            elif key == "metadata_ilike":
                # Handle case-insensitive metadata LIKE patterns
                for meta_key, meta_value in value.items():
                    if meta_key not in doc.get("metadata", {}):
                        return False
                    pattern = self._convert_like_to_fnmatch(meta_value)
                    if not fnmatch.fnmatch(str(doc["metadata"][meta_key]).lower(), pattern.lower()):
                        return False
            elif key.endswith("_ilike"):
                # Case-insensitive LIKE pattern
                field_name = key[:-6]  # Remove '_ilike' suffix
                if field_name not in doc:
                    return False
                pattern = self._convert_like_to_fnmatch(value)
                if not fnmatch.fnmatch(str(doc[field_name]).lower(), pattern.lower()):
                    return False
            elif key.endswith("_like"):
                # LIKE pattern for regular fields
                field_name = key[:-5]  # Remove '_like' suffix
                if field_name not in doc:
                    return False
                pattern = self._convert_like_to_fnmatch(value)
                if not fnmatch.fnmatch(str(doc[field_name]), pattern):
                    return False
            elif isinstance(value, list):
                # Handle list fields with IN clause
                if doc.get(key) not in value:
                    return False
            else:
                # Exact match for regular fields
                if key not in doc or doc[key] != value:
                    return False

        return True

    def _matches_element_query(self, element: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if an element matches the query criteria."""
        for key, value in query.items():
            if key == "metadata":
                # Handle metadata exact matches
                for meta_key, meta_value in value.items():
                    if meta_key not in element.get("metadata", {}) or element["metadata"][meta_key] != meta_value:
                        return False
            elif key == "metadata_like":
                # Handle metadata LIKE patterns
                for meta_key, meta_value in value.items():
                    if meta_key not in element.get("metadata", {}):
                        return False
                    pattern = self._convert_like_to_fnmatch(meta_value)
                    if not fnmatch.fnmatch(str(element["metadata"][meta_key]), pattern):
                        return False
            elif key == "metadata_ilike":
                # Handle case-insensitive metadata LIKE patterns
                for meta_key, meta_value in value.items():
                    if meta_key not in element.get("metadata", {}):
                        return False
                    pattern = self._convert_like_to_fnmatch(meta_value)
                    if not fnmatch.fnmatch(str(element["metadata"][meta_key]).lower(), pattern.lower()):
                        return False
            elif key.endswith("_ilike"):
                # Case-insensitive LIKE pattern
                field_name = key[:-6]  # Remove '_ilike' suffix
                if field_name not in element:
                    return False
                pattern = self._convert_like_to_fnmatch(value)
                if not fnmatch.fnmatch(str(element[field_name]).lower(), pattern.lower()):
                    return False
            elif key.endswith("_like"):
                # LIKE pattern for regular fields
                field_name = key[:-5]  # Remove '_like' suffix
                if field_name not in element:
                    return False
                pattern = self._convert_like_to_fnmatch(value)
                if not fnmatch.fnmatch(str(element[field_name]), pattern):
                    return False
            elif key == "element_type":
                # Handle ElementType enums, strings, and lists
                type_values = self._prepare_element_type_query(value)
                if type_values:
                    if element.get("element_type") not in type_values:
                        return False
                else:
                    return False
            elif isinstance(value, list):
                # Handle other list fields with IN clause
                if element.get(key) not in value:
                    return False
            else:
                # Exact match for regular fields
                if key not in element or element[key] != value:
                    return False

        return True

    def _matches_filter_criteria(self, element: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if an element matches filter criteria for embedding search."""
        for key, value in filter_criteria.items():
            if key == "element_type" and isinstance(value, list):
                # Handle list of allowed element types
                if element.get("element_type") not in value:
                    return False
            elif key == "doc_id" and isinstance(value, list):
                # Handle list of document IDs to include
                if element.get("doc_id") not in value:
                    return False
            elif key == "exclude_doc_id" and isinstance(value, list):
                # Handle list of document IDs to exclude
                if element.get("doc_id") in value:
                    return False
            elif key == "exclude_doc_source" and isinstance(value, list):
                # Handle list of document sources to exclude
                doc_id = element.get("doc_id")
                if doc_id:
                    doc = self.documents.get(doc_id)
                    if doc and doc.get("source") in value:
                        return False
            else:
                # Simple equality filter
                if element.get(key) != value:
                    return False

        return True

    # ========================================
    # TOPIC SUPPORT METHODS (Enhanced)
    # ========================================

    def supports_topics(self) -> bool:
        """
        Indicate whether this backend supports topic-aware embeddings.

        Returns:
            True since File implementation now supports topics
        """
        return True

    def store_embedding_with_topics(self, element_pk: int, embedding: VectorType,
                                    topics: List[str], confidence: float = 1.0) -> None:
        """
        Store embedding for an element with topic assignments.

        Args:
            element_pk: Element primary key
            embedding: Vector embedding
            topics: List of topic strings (e.g., ['security.policy', 'compliance'])
            confidence: Overall confidence in this embedding/topic assignment
        """
        # Verify element pk exists in some element
        found = False
        for element in self.elements.values():
            if element.get("element_pk") == element_pk:
                found = True
                break

        if not found:
            raise ValueError(f"Element pk not found: {element_pk}")

        try:
            # Store enhanced embedding structure with topics
            embedding_data = {
                "embedding": embedding,
                "dimensions": len(embedding),
                "topics": topics,
                "confidence": confidence,
                "created_at": time.time()
            }

            self.embeddings[element_pk] = embedding_data
            self._save_embedding(element_pk)

        except Exception as e:
            logger.error(f"Error storing embedding with topics for {element_pk}: {str(e)}")
            raise

    def search_by_text_and_topics(self, search_text: str = None,
                                  include_topics: Optional[List[str]] = None,
                                  exclude_topics: Optional[List[str]] = None,
                                  min_confidence: float = 0.7,
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search elements by text with topic filtering using pattern matching.

        Args:
            search_text: Text to search for semantically (optional)
            include_topics: Topic patterns to include (e.g., ['security*', '*.policy*'])
            exclude_topics: Topic patterns to exclude (e.g., ['deprecated*'])
            min_confidence: Minimum confidence threshold for embeddings
            limit: Maximum number of results

        Returns:
            List of dictionaries with keys:
            - element_pk: Element primary key
            - similarity: Similarity score (if search_text provided)
            - confidence: Overall embedding confidence
            - topics: List of assigned topic strings
        """
        try:
            # Generate embedding for search text if provided
            query_embedding = None
            if search_text:
                if self.embedding_generator is None:
                    from ..embeddings import get_embedding_generator
                    self.embedding_generator = get_embedding_generator(config)

                query_embedding = self.embedding_generator.generate(search_text)

            return self._search_by_text_and_topics_fallback(
                query_embedding, include_topics, exclude_topics, min_confidence, limit
            )

        except Exception as e:
            logger.error(f"Error in topic-aware search: {str(e)}")
            return []

    def _search_by_text_and_topics_fallback(self, query_embedding: Optional[VectorType] = None,
                                            include_topics: Optional[List[str]] = None,
                                            exclude_topics: Optional[List[str]] = None,
                                            min_confidence: float = 0.7,
                                            limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback search using Python similarity calculation with topic filtering."""
        results = []

        for element_pk, embedding_data in self.embeddings.items():
            # Handle both old format (direct list) and new format (dict with metadata)
            if isinstance(embedding_data, list):
                # Old format - create default metadata
                embedding = embedding_data
                confidence = 1.0
                topics = []
            elif isinstance(embedding_data, dict):
                embedding = embedding_data.get("embedding", [])
                confidence = embedding_data.get("confidence", 1.0)
                topics = embedding_data.get("topics", [])
            else:
                continue

            # Check confidence threshold
            if confidence < min_confidence:
                continue

            # Apply topic filtering
            if not self._matches_topic_filters(topics, include_topics, exclude_topics):
                continue

            result = {
                'element_pk': element_pk,
                'confidence': float(confidence),
                'topics': topics
            }

            # Calculate similarity if we have a query embedding
            if query_embedding:
                try:
                    if NUMPY_AVAILABLE:
                        similarity = self._cosine_similarity_numpy(query_embedding, embedding)
                    else:
                        similarity = self._cosine_similarity_python(query_embedding, embedding)
                    result['similarity'] = float(similarity)
                except Exception as e:
                    logger.warning(f"Error calculating similarity for element {element_pk}: {str(e)}")
                    result['similarity'] = 0.0
            else:
                result['similarity'] = 1.0  # No text search, all results have equal similarity

            results.append(result)

        # Sort by similarity if we calculated it
        if query_embedding:
            results.sort(key=lambda x: x['similarity'], reverse=True)

        return results[:limit]

    @staticmethod
    def _matches_topic_filters(topics: List[str],
                               include_topics: Optional[List[str]] = None,
                               exclude_topics: Optional[List[str]] = None) -> bool:
        """Check if topics match the include/exclude filters using pattern matching."""
        # Check include filters - at least one must match
        if include_topics:
            include_match = False
            for topic in topics:
                for pattern in include_topics:
                    # Convert LIKE pattern to fnmatch pattern
                    fnmatch_pattern = pattern.replace('%', '*').replace('_', '?')
                    if fnmatch.fnmatch(topic, fnmatch_pattern):
                        include_match = True
                        break
                if include_match:
                    break

            if not include_match:
                return False

        # Check exclude filters - none should match
        if exclude_topics:
            for topic in topics:
                for pattern in exclude_topics:
                    # Convert LIKE pattern to fnmatch pattern
                    fnmatch_pattern = pattern.replace('%', '*').replace('_', '?')
                    if fnmatch.fnmatch(topic, fnmatch_pattern):
                        return False

        return True

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
        try:
            topic_stats = {}

            for element_pk, embedding_data in self.embeddings.items():
                # Handle both old format (direct list) and new format (dict with metadata)
                if isinstance(embedding_data, dict):
                    topics = embedding_data.get("topics", [])
                    confidence = embedding_data.get("confidence", 1.0)

                    # Get document for this element
                    element = self.get_element(element_pk)
                    doc_id = element.get("doc_id") if element else None

                    for topic in topics:
                        if topic not in topic_stats:
                            topic_stats[topic] = {
                                'embedding_count': 0,
                                'document_ids': set(),
                                'confidences': []
                            }

                        topic_stats[topic]['embedding_count'] += 1
                        topic_stats[topic]['confidences'].append(confidence)
                        if doc_id:
                            topic_stats[topic]['document_ids'].add(doc_id)

            # Calculate final statistics
            final_stats = {}
            for topic, stats in topic_stats.items():
                final_stats[topic] = {
                    'embedding_count': stats['embedding_count'],
                    'document_count': len(stats['document_ids']),
                    'avg_embedding_confidence': sum(stats['confidences']) / len(stats['confidences'])
                }

            return final_stats

        except Exception as e:
            logger.error(f"Error getting topic statistics: {str(e)}")
            return {}

    def get_embedding_topics(self, element_pk: int) -> List[str]:
        """
        Get topics assigned to a specific embedding.

        Args:
            element_pk: Element primary key

        Returns:
            List of topic strings assigned to this embedding
        """
        try:
            embedding_data = self.embeddings.get(element_pk)
            if embedding_data is None:
                return []

            # Handle both old format (direct list) and new format (dict with metadata)
            if isinstance(embedding_data, dict):
                return embedding_data.get("topics", [])
            else:
                return []  # Old format has no topics

        except Exception as e:
            logger.error(f"Error getting topics for element {element_pk}: {str(e)}")
            return []

    # ========================================
    # HIERARCHY METHODS
    # ========================================

    def get_results_outline(self, elements: List[Tuple[int, float]]) -> List["ElementHierarchical"]:
        """
        For an arbitrary list of element pk search results, finds the root node of the source, and each
        ancestor element, to create a root -> element array of arrays like this:
        [(<parent element>, score, [children])]

        (Note score is None if the element was not in the results param)

        Then each additional element is analyzed, its hierarchy materialized, and merged into
        the final result.
        """
        from .element_element import ElementHierarchical

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

    def _cosine_similarity(self, vec1: VectorType, vec2: VectorType) -> float:
        """
        Calculate cosine similarity between two vectors.
        Automatically uses NumPy if available, otherwise falls back to pure Python.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (between -1 and 1)
        """
        if NUMPY_AVAILABLE:
            return self._cosine_similarity_numpy(vec1, vec2)
        else:
            return self._cosine_similarity_python(vec1, vec2)

    @staticmethod
    def _cosine_similarity_numpy(vec1: VectorType, vec2: VectorType) -> float:
        """
        Calculate cosine similarity between two vectors using NumPy.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (between -1 and 1)
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for this method but not available")

        # Make sure vectors are the same length
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]

        # Convert to numpy arrays
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        # Calculate dot product
        dot_product = np.dot(vec1_np, vec2_np)

        # Calculate magnitudes
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def _cosine_similarity_python(vec1: VectorType, vec2: VectorType) -> float:
        """
        Calculate cosine similarity between two vectors using pure Python.
        This is a fallback when NumPy is not available.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (between -1 and 1)
        """
        # Make sure vectors are the same length
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Calculate magnitudes
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        # Check for zero magnitudes
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Calculate cosine similarity
        return float(dot_product / (magnitude1 * magnitude2))

    def _load_processing_history(self) -> None:
        """Load processing history from files."""
        history_files = glob.glob(os.path.join(self.storage_path, 'processing_history', '*.json'))

        for file_path in history_files:
            try:
                with open(file_path, 'r') as f:
                    history = json.load(f)

                # Use the safe filename as the key
                filename = os.path.basename(file_path)
                safe_id = os.path.splitext(filename)[0]
                self.processing_history[safe_id] = history

            except Exception as e:
                logger.error(f"Error loading processing history from {file_path}: {str(e)}")

    def _save_processing_history(self, safe_id: str) -> None:
        """Save processing history to file."""
        if safe_id not in self.processing_history:
            return

        file_path = os.path.join(self.storage_path, 'processing_history', f"{safe_id}.json")

        try:
            with open(file_path, 'w') as f:
                json.dump(self.processing_history[safe_id], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving processing history to {file_path}: {str(e)}")

    @staticmethod
    def _get_safe_filename(source_id: str) -> str:
        """Convert source_id to a safe filename by hashing it."""
        import hashlib
        return hashlib.md5(source_id.encode('utf-8')).hexdigest()

    def _load_documents(self) -> None:
        """Load documents from files."""
        document_files = glob.glob(os.path.join(self.storage_path, 'documents', '*.json'))

        for file_path in document_files:
            try:
                with open(file_path, 'r') as f:
                    document = json.load(f)

                if "doc_id" in document:
                    self.documents[document["doc_id"]] = document
            except Exception as e:
                logger.error(f"Error loading document from {file_path}: {str(e)}")

    def _load_elements(self) -> None:
        """Load elements from files."""
        element_files = glob.glob(os.path.join(self.storage_path, 'elements', '*.json'))
        max_pk = 0  # Track the highest element_pk

        for file_path in element_files:
            try:
                with open(file_path, 'r') as f:
                    element = json.load(f)

                if "element_id" in element:
                    # Ensure element has element_pk
                    if "element_pk" not in element:
                        element["element_pk"] = self.next_element_pk
                        self.next_element_pk += 1

                    # Track the highest element_pk
                    max_pk = max(max_pk, element["element_pk"])

                    # Store element and mapping
                    self.elements[element["element_id"]] = element
                    self.element_pks[element["element_id"]] = element["element_pk"]
            except Exception as e:
                logger.error(f"Error loading element from {file_path}: {str(e)}")

        # Update next_element_pk to be one more than the highest seen
        self.next_element_pk = max_pk + 1

    def _load_relationships(self) -> None:
        """Load relationships from files."""
        relationship_files = glob.glob(os.path.join(self.storage_path, 'relationships', '*.json'))

        for file_path in relationship_files:
            try:
                with open(file_path, 'r') as f:
                    relationship = json.load(f)

                if "relationship_id" in relationship:
                    self.relationships[relationship["relationship_id"]] = relationship
            except Exception as e:
                logger.error(f"Error loading relationship from {file_path}: {str(e)}")

    def _load_embeddings(self) -> None:
        """Load embeddings from files, supporting both old and new formats."""
        # Load old format (.npy files)
        if NUMPY_AVAILABLE:
            embedding_files = glob.glob(os.path.join(self.storage_path, 'embeddings', '*.npy'))

            for file_path in embedding_files:
                try:
                    # Extract element_pk from filename
                    filename = os.path.basename(file_path)
                    element_pk = int(os.path.splitext(filename)[0])

                    # Load embedding and convert to new format
                    embedding = np.load(file_path).tolist()
                    embedding_data = {
                        "embedding": embedding,
                        "dimensions": len(embedding),
                        "topics": [],
                        "confidence": 1.0,
                        "created_at": time.time()
                    }
                    self.embeddings[element_pk] = embedding_data
                except Exception as e:
                    logger.error(f"Error loading old format embedding from {file_path}: {str(e)}")

        # Load new format (.json files) - these take precedence
        embedding_json_files = glob.glob(os.path.join(self.storage_path, 'embeddings', '*.json'))

        for file_path in embedding_json_files:
            try:
                # Extract element_pk from filename
                filename = os.path.basename(file_path)
                element_pk = int(os.path.splitext(filename)[0])

                # Load enhanced embedding data
                with open(file_path, 'r') as f:
                    embedding_data = json.load(f)

                self.embeddings[element_pk] = embedding_data
            except Exception as e:
                logger.error(f"Error loading new format embedding from {file_path}: {str(e)}")

    def _save_document(self, doc_id: str) -> None:
        """Save document to file."""
        if doc_id not in self.documents:
            return

        file_path = os.path.join(self.storage_path, 'documents', f"{doc_id}.json")

        try:
            with open(file_path, 'w') as f:
                json.dump(self.documents[doc_id], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving document to {file_path}: {str(e)}")

    def _save_element(self, element_id: str) -> None:
        """Save element to file."""
        if element_id not in self.elements:
            return

        file_path = os.path.join(self.storage_path, 'elements', f"{element_id}.json")

        try:
            with open(file_path, 'w') as f:
                json.dump(self.elements[element_id], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving element to {file_path}: {str(e)}")

    def _save_relationship(self, relationship_id: str) -> None:
        """Save relationship to file."""
        if relationship_id not in self.relationships:
            return

        file_path = os.path.join(self.storage_path, 'relationships', f"{relationship_id}.json")

        try:
            with open(file_path, 'w') as f:
                json.dump(self.relationships[relationship_id], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving relationship to {file_path}: {str(e)}")

    def _save_embedding(self, element_pk: int) -> None:
        """Save embedding to file in new JSON format."""
        if element_pk not in self.embeddings:
            return

        # Save in new JSON format
        file_path = os.path.join(self.storage_path, 'embeddings', f"{element_pk}.json")

        try:
            with open(file_path, 'w') as f:
                json.dump(self.embeddings[element_pk], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving embedding to {file_path}: {str(e)}")

        # Also clean up old .npy file if it exists
        old_file_path = os.path.join(self.storage_path, 'embeddings', f"{element_pk}.npy")
        try:
            if os.path.exists(old_file_path):
                os.remove(old_file_path)
        except Exception as e:
            logger.warning(f"Could not remove old embedding file {old_file_path}: {str(e)}")

    def _delete_document_file(self, doc_id: str) -> None:
        """Delete document file."""
        file_path = os.path.join(self.storage_path, 'documents', f"{doc_id}.json")

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting document file {file_path}: {str(e)}")

    def _delete_element_file(self, element_id: str) -> None:
        """Delete element file."""
        file_path = os.path.join(self.storage_path, 'elements', f"{element_id}.json")

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting element file {file_path}: {str(e)}")

    def _delete_relationship_file(self, relationship_id: str) -> None:
        """Delete relationship file."""
        file_path = os.path.join(self.storage_path, 'relationships', f"{relationship_id}.json")

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting relationship file {file_path}: {str(e)}")

    def _delete_embedding_file(self, element_pk: int) -> None:
        """Delete embedding files (both old and new formats)."""
        # Delete new JSON format
        json_file_path = os.path.join(self.storage_path, 'embeddings', f"{element_pk}.json")
        try:
            if os.path.exists(json_file_path):
                os.remove(json_file_path)
        except Exception as e:
            logger.error(f"Error deleting embedding JSON file {json_file_path}: {str(e)}")

        # Delete old .npy format if it exists
        npy_file_path = os.path.join(self.storage_path, 'embeddings', f"{element_pk}.npy")
        try:
            if os.path.exists(npy_file_path):
                os.remove(npy_file_path)
        except Exception as e:
            logger.error(f"Error deleting embedding NPY file {npy_file_path}: {str(e)}")

    @staticmethod
    def _has_element_changed(new_element: Dict[str, Any],
                             old_element: Dict[str, Any]) -> bool:
        """Check if element has changed."""
        # Check content hash first
        if new_element.get("content_hash") != old_element.get("content_hash"):
            return True

        # Check other fields
        for field in ["element_type", "parent_id", "content_preview", "content_location"]:
            if new_element.get(field) != old_element.get(field):
                return True

        # Check metadata
        new_metadata = new_element.get("metadata", {})
        old_metadata = old_element.get("metadata", {})

        if set(new_metadata.keys()) != set(old_metadata.keys()):
            return True

        for key, value in new_metadata.items():
            if old_metadata.get(key) != value:
                return True

        return False

    @staticmethod
    def _has_relationship_changed(new_rel: Dict[str, Any],
                                  old_rel: Dict[str, Any]) -> bool:
        """Check if relationship has changed."""
        for field in ["source_id", "relationship_type", "target_reference"]:
            if new_rel.get(field) != old_rel.get(field):
                return True

        # Check metadata
        new_metadata = new_rel.get("metadata", {})
        old_metadata = old_rel.get("metadata", {})

        if set(new_metadata.keys()) != set(old_metadata.keys()):
            return True

        for key, value in new_metadata.items():
            if old_metadata.get(key) != value:
                return True

        return False
