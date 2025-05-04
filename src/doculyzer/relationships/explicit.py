import uuid
from typing import Dict, Any, List, Optional

from .base import RelationshipDetector
from ..storage import ElementType


class ExplicitLinkDetector(RelationshipDetector):
    """Detector for explicit links extracted by the parser."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the explicit link detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    def detect_relationships(self, document: Dict[str, Any],
                             elements: List[Dict[str, Any]],
                             links: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Convert extracted links into relationships.

        Args:
            document: Document metadata
            elements: Document elements
            links: List of links extracted by the parser

        Returns:
            List of detected relationships
        """
        relationships = []
        doc_id = document["doc_id"]

        # If no links provided, return empty list
        if not links:
            return relationships

        # Create element ID to element mapping for easier lookup
        # element_map = {element["element_id"]: element for element in elements}

        # Process each link
        for link in links:
            source_id = link.get("source_id")
            link_text = link.get("link_text", "")
            link_target = link.get("link_target", "")
            link_type = link.get("link_type", "")

            # Skip if missing required data
            if not source_id or not link_target:
                continue

            # Create relationship
            relationship_id = self._generate_id("rel_")

            relationship = {
                "relationship_id": relationship_id,
                "doc_id": doc_id,
                "source_id": source_id,
                "relationship_type": "link",
                "target_reference": link_target,
                "metadata": {
                    "text": link_text,
                    "url": link_target,
                    "link_type": link_type,
                    "confidence": 1.0  # Explicit links have full confidence
                }
            }

            relationships.append(relationship)

            # Try to find target element in the same document
            target_element = self._find_target_element(link_target, link_text, elements)

            if target_element:
                # Create bidirectional relationship
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": target_element["element_id"],
                    "relationship_type": "referenced_by",
                    "target_reference": source_id,
                    "metadata": {
                        "text": link_text,
                        "confidence": 1.0
                    }
                }

                relationships.append(relationship)

        return relationships

    @staticmethod
    def _find_target_element(link_target: str, link_text: str,
                             elements: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find target element for a link.

        Args:
            link_target: Link target
            link_text: Link text
            elements: Document elements

        Returns:
            Target element or None if not found
        """
        # Check for element ID
        if link_target.startswith('#'):
            # Internal anchor link
            target_id = link_target[1:]

            for element in elements:
                if element.get("element_id") == target_id:
                    return element

                # Check metadata for ID
                metadata = element.get("metadata", {})
                if metadata.get("id") == target_id:
                    return element

        # Check for header text match
        for element in elements:
            if element.get("element_type") == ElementType.HEADER.value:
                header_text = element.get("metadata", {}).get("text", "")

                if header_text and (header_text == link_text or header_text == link_target):
                    return element

        # Check for file name in link_target (for cross-document links)
        if '.' in link_target and not link_target.startswith(('http://', 'https://')):
            # This might be a link to another document
            # We can't resolve this here, but it could be handled by a higher-level component
            pass

        return None

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
