from dataclasses import field
from enum import Enum
from typing import Dict, Any, Optional, List

from pydantic import BaseModel


class RelationshipCategory(Enum):
    """Enumeration of relationship categories based on how they were created."""
    STRUCTURAL = "STRUCTURAL"  # Relationships derived from document structure
    EXPLICIT_LINK = "EXPLICIT"  # Explicit links found in the document
    SEMANTIC = "SEMANTIC"  # Semantic similarity relationships
    UNKNOWN = "UNKNOWN"  # Unknown or custom relationship types


# Helper functions for working with element collections


class ElementRelationship(BaseModel):
    """
    Class for representing relationships between elements.
    Designed to avoid name collisions with SQLAlchemy's Relationship class.
    Enriched with PK values for both source and target elements.
    """
    # Primary identifier
    relationship_id: str

    # Source element identifiers
    source_id: str

    # Relationship type
    relationship_type: str

    # Target element identifiers
    target_reference: str
    target_element_pk: Optional[int] = None
    target_element_type: Optional[str] = None
    target_content_preview: Optional[str] = None

    # Optional doc_id for the relationship
    doc_id: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Direction flag - whether the specified element is the source (True) or target (False)
    # This is especially useful when retrieving relationships for a specific element
    is_source: bool = True
    source_element_pk: Optional[int] = None
    source_element_type: Optional[str] = None

    def __str__(self) -> str:
        """String representation of the relationship."""
        source_info = f"{self.source_id}({self.source_element_pk})" if self.source_element_pk else self.source_id
        target_info = f"{self.target_reference}({self.target_element_pk})" if self.target_element_pk else self.target_reference

        direction = "-->" if self.is_source else "<--"
        return f"{source_info} {direction}[{self.relationship_type}]{direction} {target_info}"

    def get_category(self) -> RelationshipCategory:
        """
        Determine the category of this relationship based on its type.

        Returns:
            RelationshipCategory enum value
        """
        # Structural relationship types from StructuralRelationshipDetector
        structural_types = [
            "next_sibling", "previous_sibling", "contains", "contained_by",
            "contains_row", "contains_cell", "contains_item"
        ]

        # Explicit link types from ExplicitLinkDetector
        explicit_link_types = ["link", "referenced_by"]

        # Semantic similarity types from SemanticRelationshipDetector
        semantic_types = ["semantic_similarity"]

        # Check relationship type
        rel_type = self.relationship_type.lower()

        if rel_type in structural_types:
            return RelationshipCategory.STRUCTURAL
        elif rel_type in explicit_link_types:
            return RelationshipCategory.EXPLICIT_LINK
        elif rel_type in semantic_types:
            return RelationshipCategory.SEMANTIC
        else:
            return RelationshipCategory.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for storage."""
        return {
            "relationship_id": self.relationship_id,
            "source_id": self.source_id,
            "source_element_pk": self.source_element_pk,
            "source_element_type": self.source_element_type,
            "relationship_type": self.relationship_type,
            "target_reference": self.target_reference,
            "target_element_pk": self.target_element_pk,
            "target_element_type": self.target_element_type,
            "target_content_preview": self.target_content_preview,
            "doc_id": self.doc_id,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], is_source: bool = True) -> 'ElementRelationship':
        """
        Create an ElementRelationship instance from a dictionary.

        Args:
            data: Dictionary containing relationship data
            is_source: Whether the specified element is the source (True) or target (False)

        Returns:
            ElementRelationship instance
        """
        return cls(
            relationship_id=data.get("relationship_id", ""),
            source_id=data.get("source_id", ""),
            source_element_pk=data.get("source_element_pk"),
            source_element_type=data.get("source_element_type"),
            relationship_type=data.get("relationship_type", ""),
            target_reference=data.get("target_reference", ""),
            target_element_pk=data.get("target_element_pk"),
            target_element_type=data.get("target_element_type"),
            doc_id=data.get("doc_id"),
            metadata=data.get("metadata", {}),
            is_source=is_source
        )

    def get_related_element_id(self) -> str:
        """
        Get the ID of the related element, regardless of direction.

        Returns:
            Element ID of the related element (source or target)
        """
        return self.target_reference if self.is_source else self.source_id

    def get_related_element_pk(self) -> Optional[int]:
        """
        Get the PK of the related element, regardless of direction.

        Returns:
            Element PK of the related element (source or target)
        """
        return self.target_element_pk if self.is_source else self.source_element_pk

    def is_bidirectional(self) -> bool:
        """Check if this is a bidirectional relationship based on type."""
        # Semantic similarity is bidirectional
        if self.get_category() == RelationshipCategory.SEMANTIC:
            return True

        # Some specific structural relationships are bidirectional
        bidirectional_types = [
            "semantic_similarity", "synonym", "related",
            "similar", "equivalent"
        ]
        return self.relationship_type.lower() in bidirectional_types

    def get_confidence(self) -> float:
        """Get the confidence score for this relationship."""
        return self.metadata.get("confidence", 0.0)

    def get_similarity(self) -> float:
        """Get the similarity score for semantic relationships."""
        if self.get_category() == RelationshipCategory.SEMANTIC:
            return self.metadata.get("similarity", 0.0)
        return 0.0


# Helper functions for working with relationship collections

def get_structural_relationships(relationships: List[ElementRelationship]) -> List[ElementRelationship]:
    """
    Filter relationships to get only structural relationships.

    Args:
        relationships: List of ElementRelationship objects

    Returns:
        List of structural relationships
    """
    return [r for r in relationships if r.get_category() == RelationshipCategory.STRUCTURAL]


def get_explicit_links(relationships: List[ElementRelationship]) -> List[ElementRelationship]:
    """
    Filter relationships to get only explicit links.

    Args:
        relationships: List of ElementRelationship objects

    Returns:
        List of explicit link relationships
    """
    return [r for r in relationships if r.get_category() == RelationshipCategory.EXPLICIT_LINK]


def get_semantic_relationships(relationships: List[ElementRelationship]) -> List[ElementRelationship]:
    """
    Filter relationships to get only semantic similarity relationships.

    Args:
        relationships: List of ElementRelationship objects

    Returns:
        List of semantic similarity relationships
    """
    return [r for r in relationships if r.get_category() == RelationshipCategory.SEMANTIC]


def get_container_relationships(relationships: List[ElementRelationship]) -> List[ElementRelationship]:
    """
    Get container relationships (contains, contains_row, contains_cell, contains_item).

    Args:
        relationships: List of ElementRelationship objects

    Returns:
        List of container relationships
    """
    container_types = ["contains", "contains_row", "contains_cell", "contains_item"]
    return [r for r in relationships if r.relationship_type in container_types]


def get_sibling_relationships(relationships: List[ElementRelationship]) -> List[ElementRelationship]:
    """
    Get sibling relationships (next_sibling, previous_sibling).

    Args:
        relationships: List of ElementRelationship objects

    Returns:
        List of sibling relationships
    """
    sibling_types = ["next_sibling", "previous_sibling"]
    return [r for r in relationships if r.relationship_type in sibling_types]


def sort_relationships_by_confidence(relationships: List[ElementRelationship]) -> List[ElementRelationship]:
    """
    Sort relationships by confidence (highest first).

    Args:
        relationships: List of ElementRelationship objects

    Returns:
        Sorted list of relationships
    """
    return sorted(relationships, key=lambda r: r.get_confidence(), reverse=True)


def sort_semantic_relationships_by_similarity(relationships: List[ElementRelationship]) -> List[ElementRelationship]:
    """
    Sort semantic relationships by similarity (highest first).

    Args:
        relationships: List of ElementRelationship objects

    Returns:
        Sorted list of semantic relationships
    """
    semantic_rels = get_semantic_relationships(relationships)
    return sorted(semantic_rels, key=lambda r: r.get_similarity(), reverse=True)
