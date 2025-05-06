import json
from dataclasses import field
from enum import Enum
from typing import Optional, Dict, Any, List, cast

from pydantic import BaseModel


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
    metadata: str
    score: Optional[float] = None
    child_elements: List["ElementElement"] = field(default_factory=list)

    def __str__(self) -> str:
        """String representation of the element."""
        return f"{self.element_type}({self.element_pk}): {self.content_preview[:50]}{'...' if len(self.content_preview) > 50 else ''}"

    def get_element_type_enum(self) -> "ElementType":
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
            metadata=json.dumps(metadata)
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
            return json.loads(self.metadata).get("level")
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
            return json.loads(self.metadata).get("language")
        return None


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
