from dataclasses import field
from enum import Enum
from typing import Dict, Any, Optional, List, cast

from pydantic import BaseModel


class RelationshipCategory(Enum):
    """Enumeration of relationship categories based on how they were created."""
    STRUCTURAL = "STRUCTURAL"  # Relationships derived from document structure
    EXPLICIT_LINK = "EXPLICIT"  # Explicit links found in the document
    SEMANTIC = "SEMANTIC"  # Semantic similarity relationships
    UNKNOWN = "UNKNOWN"  # Unknown or custom relationship types


class ElementType(Enum):
    """Enumeration of common element types."""
    ROOT = "root"
    HEADER = "header"
    PARAGRAPH = "paragraph"
    PAGE = "page"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_HEADER_ROW = "table_header_row"
    TABLE_CELL = "table_cell"
    TABLE_HEADER = "table_header"
    IMAGE = "image"
    CODE_BLOCK = "code_block"
    BLOCKQUOTE = "blockquote"
    XML_ELEMENT = "xml_element"
    XML_TEXT = "xml_text"
    XML_LIST = "xml_list"
    XML_OBJECT = "xml_object"
    PRESENTATION_BODY = "presentation_body"
    TEXT_BOX = "text_box"
    SLIDE = "slide"
    SLIDE_NOTES = "slide_notes"
    COMMENT = "comment"
    CHART = "chart"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    BODY = "body"
    HEADERS = "headers"
    FOOTERS = "footers"
    COMMENTS = "comments"
    JSON_OBJECT = "json_object"
    JSON_ARRAY = "json_array"
    JSON_FIELD = "json_field"
    JSON_ITEM = "json_item"
    LINE = "line"
    RANGE = "range"
    SUBSTRING = "substring"
    SHAPE_GROUP = "shape_group"
    SHAPE = "shape"
    COMMENTS_CONTAINER = "comments_container"
    SLIDE_MASTERS = "slide_masters"
    SLIDE_TEMPLATES = "slide_templates"
    SLIDE_LAYOUT = "slide_layout"
    SLIDE_MASTER = "slide_master"
    UNKNOWN = "unknown"


class ElementElement(BaseModel):
    """
    Class for representing document elements.
    Provides methods for accessing and manipulating element data.
    """
    # Primary identifier
    element_pk: int  # Auto-increment primary key
    element_id: str

    # Document identifier
    doc_id: str

    # Element characteristics
    element_type: str
    parent_id: Optional[str] = None
    content_preview: str
    content_location: str
    content_hash: str

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    child_elements: List["ElementElement"] = field(default_factory=list)

    def __str__(self) -> str:
        """String representation of the element."""
        return f"{self.element_type}({self.element_pk}): {self.content_preview[:50]}{'...' if len(self.content_preview) > 50 else ''}"

    def get_element_type_enum(self) -> ElementType:
        """
        Get the element type as an enum value.

        Returns:
            ElementType enum value
        """
        try:
            return cast(ElementType, ElementType[self.element_type.upper()])
        except (KeyError, AttributeError):
            return ElementType.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for storage."""
        return {
            "element_pk": self.element_pk,
            "element_id": self.element_id,
            "doc_id": self.doc_id,
            "element_type": self.element_type,
            "parent_id": self.parent_id,
            "content_preview": self.content_preview,
            "content_location": self.content_location,
            "content_hash": self.content_hash,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ElementElement':
        """
        Create an ElementElement instance from a dictionary.

        Args:
            data: Dictionary containing element data

        Returns:
            ElementElement instance
        """
        # Convert the metadata_ field if it exists
        metadata = data.get("metadata", {})
        if "metadata_" in data and not metadata:
            metadata = data.get("metadata_", {})

        return cls(
            element_pk=data.get("element_pk", 0),
            element_id=data.get("element_id", ""),
            doc_id=data.get("doc_id", ""),
            element_type=data.get("element_type", ""),
            parent_id=data.get("parent_id"),
            content_preview=data.get("content_preview", ""),
            content_location=data.get("content_location", ""),
            content_hash=data.get("content_hash", ""),
            metadata=metadata
        )

    def is_root(self) -> bool:
        """Check if this is a root element."""
        return self.element_type.lower() == "root"

    def is_container(self) -> bool:
        """Check if this is a container element."""
        container_types = [
            "root", "div", "article", "section",
            "list", "table", "page", "xml_list", "xml_object",
            "table_header", "table_header_row", "presentation_body",
            "slide", "comments_container", "comments", "json_array",
            "json_object", "slide_masters", "slide_templates",
            "headers", "footers", "page_header", "page_footer", "body"
        ]
        return self.element_type.lower() in container_types

    def is_leaf(self) -> bool:
        """Check if this is a leaf element (not a container)."""
        return not self.is_container()

    def has_parent(self) -> bool:
        """Check if this element has a parent."""
        return self.parent_id is not None and self.parent_id != ""

    def get_level(self) -> Optional[int]:
        """
        Get the header level if this is a header element.

        Returns:
            Header level (1-6) or None if not a header
        """
        if self.element_type.lower() == "header":
            return self.metadata.get("level")
        return None

    def get_content_type(self) -> str:
        """
        Get the content type based on the element type.

        Returns:
            Content type string
        """
        element_type = self.element_type.lower()

        if element_type == "header":
            return "heading"
        elif element_type == "paragraph":
            return "text"
        elif element_type in ["list", "list_item"]:
            return "list"
        elif element_type in ["table", "table_row", "table_cell"]:
            return "table"
        elif element_type == "image":
            return "image"
        elif element_type == "code_block":
            return "code"
        elif element_type == "blockquote":
            return "quote"
        else:
            return "unknown"

    def get_language(self) -> Optional[str]:
        """
        Get the programming language if this is a code block.

        Returns:
            Language string or None if not a code block or language not specified
        """
        if self.element_type.lower() == "code_block":
            return self.metadata.get("language")
        return None


# Helper functions for working with element collections

def filter_elements_by_type(elements: List[ElementElement], element_type: str) -> List[ElementElement]:
    """
    Filter elements by type.

    Args:
        elements: List of ElementElement objects
        element_type: Element type to filter for

    Returns:
        List of elements matching the specified type
    """
    return [e for e in elements if e.element_type.lower() == element_type.lower()]


def get_root_elements(elements: List[ElementElement]) -> List[ElementElement]:
    """
    Get all root elements from a list.

    Args:
        elements: List of ElementElement objects

    Returns:
        List of root elements
    """
    return [e for e in elements if e.is_root()]


def get_container_elements(elements: List[ElementElement]) -> List[ElementElement]:
    """
    Get all container elements from a list.

    Args:
        elements: List of ElementElement objects

    Returns:
        List of container elements
    """
    return [e for e in elements if e.is_container()]


def get_leaf_elements(elements: List[ElementElement]) -> List[ElementElement]:
    """
    Get all leaf elements from a list.

    Args:
        elements: List of ElementElement objects

    Returns:
        List of leaf elements
    """
    return [e for e in elements if e.is_leaf()]


def get_child_elements(elements: List[ElementElement], parent_id: str) -> List[ElementElement]:
    """
    Get all direct children of a specific element.

    Args:
        elements: List of ElementElement objects
        parent_id: ID of the parent element

    Returns:
        List of child elements
    """
    return [e for e in elements if e.parent_id == parent_id]


def build_element_hierarchy(elements: List[ElementElement]) -> Dict[str, List[ElementElement]]:
    """
    Build a hierarchy map of parent IDs to child elements.

    Args:
        elements: List of ElementElement objects

    Returns:
        Dictionary mapping parent_id to list of child elements
    """
    hierarchy = {}

    for element in elements:
        if element.parent_id:
            if element.parent_id not in hierarchy:
                hierarchy[element.parent_id] = []

            hierarchy[element.parent_id].append(element)

    return hierarchy


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
