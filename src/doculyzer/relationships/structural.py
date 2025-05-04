import uuid
from enum import Enum
from typing import Dict, Any, List

from .base import RelationshipDetector
from ..storage import ElementType


class RelationshipType(Enum):
    """Enumeration of structural relationship types."""

    HAS_TEMPORAL_INFORMATION = "has_temporal_information"
    TEMPORAL_RELATIONSHIP = "temporal_relationship"
    RELATED_TO = "related_to"
    DESCRIBED_BY = "described_by"
    DESCRIBES = "describes"
    NEXT_SIBLING = "next_sibling"
    PREVIOUS_SIBLING = "previous_sibling"
    LINK = "link"
    REFERENCED_BY = "referenced_by"
    SEMANTIC_SIMILARITY = "semantic_similarity"

    CONTAINS = "contains"
    CONTAINS_TABLE_HEADER = "contains_table_header"
    CONTAINS_LIST_ITEM = "contains_list_item"
    CONTAINS_TABLE_CELL = "contains_table_cell"
    CONTAINS_ARRAY_ITEM = "contains_array_item"  # New type for array items
    CONTAINS_ROW = "contains_row"
    CONTAINS_CELL = "contains_cell"
    CONTAINS_ITEM = "contains_item"
    CONTAINS_TEXT = "contains_text"  # Added to handle text content
    CONTAINS_TABLE_ROW = "contains_table_row"
    CONTAINS_NOTES = "contains_notes"

    CONTAINED_BY = "contained_by"

    @classmethod
    def is_parent_type(cls, rel_type: 'RelationshipType') -> bool:
        """Check if this is a parent relationship type."""
        return rel_type in [
            cls.CONTAINS,
            cls.CONTAINS_TABLE_HEADER,
            cls.CONTAINS_LIST_ITEM,
            cls.CONTAINS_TABLE_CELL,
            cls.CONTAINS_ARRAY_ITEM,
            cls.CONTAINS_ROW,
            cls.CONTAINS_CELL,
            cls.CONTAINS_ITEM,
            cls.CONTAINS_TEXT,
            cls.CONTAINS_TABLE_ROW,
            cls.CONTAINS_NOTES,
        ]

    @classmethod
    def is_child_type(cls, rel_type: 'RelationshipType') -> bool:
        """Check if this is a child relationship type."""
        return rel_type == cls.CONTAINED_BY

    @classmethod
    def is_sibling_type(cls, rel_type: 'RelationshipType') -> bool:
        """Check if this is a sibling relationship type."""
        return rel_type in [cls.NEXT_SIBLING, cls.PREVIOUS_SIBLING]

    @classmethod
    def is_semantic_type(cls, rel_type: 'RelationshipType') -> bool:
        """Check if this is a semantic relationship type."""
        return rel_type == cls.SEMANTIC_SIMILARITY

    @classmethod
    def is_link_type(cls, rel_type: 'RelationshipType') -> bool:
        """Check if this is a link relationship type."""
        return rel_type in [cls.LINK, cls.REFERENCED_BY]


class StructuralRelationshipDetector(RelationshipDetector):
    """Detector for structural relationships between elements."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the structural relationship detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    def detect_relationships(self, document: Dict[str, Any],
                             elements: List[Dict[str, Any]],
                             links: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect structural relationships between elements not found by parser.
        Generally, this is just creates sibling relationships between elements, or cleans up anything
        not handled by the parser.
        """
        relationships = []
        doc_id = document["doc_id"]

        # Create parent-child mapping
        parent_children = {}
        for element in elements:
            parent_id = element.get("parent_id")
            element_id = element["element_id"]

            if parent_id:
                if parent_id not in parent_children:
                    parent_children[parent_id] = []

                parent_children[parent_id].append(element_id)

        # Create sibling relationships
        for parent_id, children in parent_children.items():
            # Skip if only one child
            if len(children) <= 1:
                continue

            # Create relationships between consecutive siblings
            for i in range(len(children) - 1):
                prev_id = children[i]
                next_id = children[i + 1]

                # Create relationships
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": prev_id,
                    "relationship_type": RelationshipType.NEXT_SIBLING.value,
                    "target_reference": next_id,
                    "metadata": {
                        "confidence": 1.0
                    }
                }

                relationships.append(relationship)

                # Create reverse relationship
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": next_id,
                    "relationship_type": RelationshipType.PREVIOUS_SIBLING.value,
                    "target_reference": prev_id,
                    "metadata": {
                        "confidence": 1.0
                    }
                }

                relationships.append(relationship)

        return relationships

    @staticmethod
    def _get_section_elements(header: Dict[str, Any], _all_headers: List[Dict[str, Any]],
                              all_elements: List[Dict[str, Any]]) -> List[str]:
        """
        Get elements that belong to a header's section.

        A section includes all elements that:
        1. Come after this header
        2. Come before the next header of equal or higher level
        3. Are not headers themselves

        Args:
            header: Header element
            _all_headers: List of all headers
            all_elements: List of all elements

        Returns:
            List of element IDs in the section
        """
        header_id = header["element_id"]
        header_level = header.get("metadata", {}).get("level", 0)

        # Create list of elements in document order
        # This assumes elements are provided in document order
        element_ids = [e["element_id"] for e in all_elements]

        # Find index of this header
        try:
            header_index = element_ids.index(header_id)
        except ValueError:
            return []

        section_element_ids = []

        # Iterate through elements after this header
        for i in range(header_index + 1, len(element_ids)):
            element_id = element_ids[i]
            element = next((e for e in all_elements if e["element_id"] == element_id), None)

            if not element:
                continue

            # Stop at next header of equal or higher level
            if element.get("element_type") == ElementType.HEADER.value:
                element_level = element.get("metadata", {}).get("level", 0)
                if element_level <= header_level:
                    break

            # Add non-header elements to section
            if element.get("element_type") != ElementType.HEADER.value:
                section_element_ids.append(element_id)

        return section_element_ids

    @staticmethod
    def _generate_id(prefix: str = "") -> str:
        """
        Generate a unique ID.

        Args:
            prefix: Optional prefix for the ID

        Returns:
            Unique ID string
        """
        return f"{prefix}{uuid.uuid4()}"
