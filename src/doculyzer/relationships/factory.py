"""
Relationship Detector Module for the document pointer system.

This module detects relationships between document elements,
including explicit links, semantic relationships, and structural connections.
"""

import logging
from typing import Dict, Any

from .base import RelationshipDetector
from .composite import CompositeRelationshipDetector
from .explicit import ExplicitLinkDetector
from .semantic import SemanticRelationshipDetector
from .structural import StructuralRelationshipDetector

logger = logging.getLogger(__name__)


def create_relationship_detector(config: Dict[str, Any], embedding_generator=None) -> RelationshipDetector:
    """
    Factory function to create a relationship detector from configuration.

    Args:
        config: Configuration dictionary
        embedding_generator: Optional embedding generator for semantic relationships

    Returns:
        RelationshipDetector instance
    """
    detectors = [ExplicitLinkDetector(config)]

    # Add explicit link detector (always enabled to handle parser-extracted links)

    # Add structural relationship detector
    if config.get("structural", True):
        detectors.append(StructuralRelationshipDetector(config))

    # Add semantic relationship detector if embeddings are enabled
    if config.get("semantic", False) and embedding_generator:
        semantic_config = config.get("semantic_config", {})
        detectors.append(SemanticRelationshipDetector(embedding_generator, semantic_config))

    # Return composite detector
    return CompositeRelationshipDetector(detectors)
