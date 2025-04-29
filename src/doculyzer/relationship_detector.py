"""
Relationship Detector Module for the document pointer system.

This module detects relationships between document elements,
including explicit links, semantic relationships, and structural connections.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


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
            if element.get("element_type") == "header":
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


class SemanticRelationshipDetector(RelationshipDetector):
    """Detector for semantic relationships between elements using embeddings."""

    def __init__(self, embedding_generator, config: Dict[str, Any] = None):
        """
        Initialize the semantic relationship detector.

        Args:
            embedding_generator: Embedding generator
            config: Configuration dictionary
        """
        self.embedding_generator = embedding_generator
        self.config = config or {}
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.max_relationships = self.config.get("max_relationships", 5)

    def detect_relationships(self, document: Dict[str, Any],
                             elements: List[Dict[str, Any]],
                             links: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect semantic relationships between elements."""
        relationships = []
        doc_id = document["doc_id"]

        # Skip if no elements
        if not elements:
            return relationships

        # Generate embeddings for all elements
        element_texts = {}
        elements_to_embed = []

        for element in elements:
            element_id = element["element_id"]
            element_type = element["element_type"]
            content_preview = element.get("content_preview", "")

            # Skip root element and elements without content
            if element_type == "root" or not content_preview:
                continue

            element_texts[element_id] = content_preview
            elements_to_embed.append((element_id, content_preview))

        # Skip if no elements to embed
        if not elements_to_embed:
            return relationships

        # Generate embeddings
        element_ids = [e[0] for e in elements_to_embed]
        texts = [e[1] for e in elements_to_embed]

        try:
            embeddings = self.embedding_generator.generate_batch(texts)

            # Create mapping of element ID to embedding
            element_embeddings = {
                element_id: embedding
                for element_id, embedding in zip(element_ids, embeddings)
            }

            # Calculate pairwise similarities
            similarities = self._calculate_similarities(element_embeddings)

            # Create relationships for similar elements
            for (source_id, target_id), similarity in similarities:
                # Skip if similarity is below threshold
                if similarity < self.similarity_threshold:
                    continue

                # Create relationship
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": source_id,
                    "relationship_type": "semantic_similarity",
                    "target_reference": target_id,
                    "metadata": {
                        "similarity": float(similarity),
                        "confidence": float(similarity),
                        "source_text": element_texts.get(source_id, "")[:50],
                        "target_text": element_texts.get(target_id, "")[:50]
                    }
                }

                relationships.append(relationship)

        except Exception as e:
            logger.error(f"Error detecting semantic relationships: {str(e)}")

        return relationships

    def _calculate_similarities(self, element_embeddings: Dict[str, List[float]]) -> List[
        Tuple[Tuple[str, str], float]]:
        """
        Calculate pairwise similarities between elements.

        Args:
            element_embeddings: Dict mapping element ID to embedding

        Returns:
            List of ((source_id, target_id), similarity) tuples, sorted by similarity
        """
        element_ids = list(element_embeddings.keys())
        similarities = []

        # Calculate similarities for all pairs
        for i, source_id in enumerate(element_ids):
            source_embedding = np.array(element_embeddings[source_id])

            # Only calculate for elements after this one (avoid duplicates)
            for target_id in element_ids[i + 1:]:
                target_embedding = np.array(element_embeddings[target_id])

                # Calculate cosine similarity
                similarity = self._cosine_similarity(source_embedding, target_embedding)

                # Add to results
                similarities.append(((source_id, target_id), similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Limit to max relationships
        return similarities[:self.max_relationships * len(element_ids)]

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (between -1 and 1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def _generate_id(prefix: str = "") -> str:
        """Generate a unique ID."""
        return f"{prefix}{uuid.uuid4()}"


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
        """Detect structural relationships between elements."""
        relationships = []
        doc_id = document["doc_id"]

        # Create element ID to element mapping for easier lookup
        # element_map = {element["element_id"]: element for element in elements}

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
                    "relationship_type": "next_sibling",
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
                    "relationship_type": "previous_sibling",
                    "target_reference": prev_id,
                    "metadata": {
                        "confidence": 1.0
                    }
                }

                relationships.append(relationship)

        # Create section relationships (header -> content elements)
        # Find all headers
        headers = [element for element in elements if element.get("element_type") == "header"]

        # Sort headers by their level (h1, h2, etc.)
        headers.sort(key=lambda h: h.get("metadata", {}).get("level", 0))

        # Process each header
        for header in headers:
            header_id = header["element_id"]
            header_level = header.get("metadata", {}).get("level", 0)

            # Find elements that should be in this header's section
            section_elements = self._get_section_elements(header, headers, elements)

            # Create relationships from header to section elements
            for element_id in section_elements:
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": header_id,
                    "relationship_type": "contains",
                    "target_reference": element_id,
                    "metadata": {
                        "confidence": 1.0,
                        "section_level": header_level
                    }
                }

                relationships.append(relationship)

                # Create reverse relationship
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": element_id,
                    "relationship_type": "contained_by",
                    "target_reference": header_id,
                    "metadata": {
                        "confidence": 1.0,
                        "section_level": header_level
                    }
                }

                relationships.append(relationship)

        # Create table relationships (table -> rows -> cells)
        tables = [element for element in elements if element.get("element_type") == "table"]

        for table in tables:
            table_id = table["element_id"]

            # Find table rows
            rows = [element for element in elements
                    if element.get("element_type") == "table_row" and element.get("parent_id") == table_id]

            # Sort rows by index
            rows.sort(key=lambda r: r.get("metadata", {}).get("row", 0))

            # Create table -> row relationships
            for row in rows:
                row_id = row["element_id"]

                # Create relationship
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": table_id,
                    "relationship_type": "contains_row",
                    "target_reference": row_id,
                    "metadata": {
                        "confidence": 1.0,
                        "row_index": row.get("metadata", {}).get("row", 0)
                    }
                }

                relationships.append(relationship)

                # Find cells in this row
                cells = [element for element in elements
                         if element.get("element_type") in ("table_cell", "table_header")
                         and element.get("parent_id") == row_id]

                # Sort cells by column
                cells.sort(key=lambda c: c.get("metadata", {}).get("col", 0))

                # Create row -> cell relationships
                for cell in cells:
                    cell_id = cell["element_id"]

                    # Create relationship
                    relationship_id = self._generate_id("rel_")

                    relationship = {
                        "relationship_id": relationship_id,
                        "doc_id": doc_id,
                        "source_id": row_id,
                        "relationship_type": "contains_cell",
                        "target_reference": cell_id,
                        "metadata": {
                            "confidence": 1.0,
                            "col_index": cell.get("metadata", {}).get("col", 0)
                        }
                    }

                    relationships.append(relationship)

        # Create list relationships (list -> list items)
        lists = [element for element in elements if element.get("element_type") == "list"]

        for list_element in lists:
            list_id = list_element["element_id"]

            # Find list items
            items = [element for element in elements
                     if element.get("element_type") == "list_item" and element.get("parent_id") == list_id]

            # Sort items by index
            items.sort(key=lambda ii: ii.get("metadata", {}).get("index", 0))

            # Create list -> item relationships
            for item in items:
                item_id = item["element_id"]

                # Create relationship
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": list_id,
                    "relationship_type": "contains_item",
                    "target_reference": item_id,
                    "metadata": {
                        "confidence": 1.0,
                        "item_index": item.get("metadata", {}).get("index", 0),
                        "list_type": list_element.get("metadata", {}).get("list_type", "unordered")
                    }
                }

                relationships.append(relationship)

        # Create document -> element relationships for top-level elements
        root_elements = [element for element in elements if element.get("element_type") == "root"]
        if root_elements:
            root_id = root_elements[0]["element_id"]

            # Find all direct children of root
            root_children = [element["element_id"] for element in elements
                             if element.get("parent_id") == root_id and element.get("element_type") != "root"]

            for child_id in root_children:
                # Create relationship
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": root_id,
                    "relationship_type": "contains",
                    "target_reference": child_id,
                    "metadata": {
                        "confidence": 1.0,
                        "top_level": True
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
            if element.get("element_type") == "header":
                element_level = element.get("metadata", {}).get("level", 0)
                if element_level <= header_level:
                    break

            # Add non-header elements to section
            if element.get("element_type") != "header":
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
