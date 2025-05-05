import logging
from typing import List, Dict, Any

from .base import RelationshipDetector

logger = logging.getLogger(__name__)


class CompositeRelationshipDetector(RelationshipDetector):
    """Combines multiple relationship detectors."""

    def __init__(self, detectors: List[RelationshipDetector]):
        """
        Initialize the composite relationship detector.

        Args:
            detectors: List of relationship detectors
        """
        self.detectors = detectors

    def detect_relationships(self, document: Dict[str, Any],
                             elements: List[Dict[str, Any]],
                             links: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run all detectors and combine their results."""
        all_relationships = []

        for detector in self.detectors:
            try:
                relationships = detector.detect_relationships(document, elements, links)
                all_relationships.extend(relationships)
            except Exception as e:
                logger.error(f"Error in detector {detector.__class__.__name__}: {str(e)}")

        return all_relationships
