from abc import ABC, abstractmethod
from typing import Dict, Any, List


class RelationshipDetector(ABC):
    """Abstract base class for relationship detectors."""

    @abstractmethod
    def detect_relationships(self, document: Dict[str, Any],
                             elements: List[Dict[str, Any]],
                             links: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect relationships between elements in a document.

        Args:
            document: Document metadata
            elements: Document elements
            links: Optional list of links extracted by the parser

        Returns:
            List of detected relationships
        """
        pass
