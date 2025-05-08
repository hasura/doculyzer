import logging
import uuid
from typing import Dict, Any, List, Tuple, Union, TYPE_CHECKING

# Import types for type checking only - these won't be imported at runtime
if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    # Define type aliases for type checking
    VectorType = NDArray[np.float32]  # NumPy array type for vectors
else:
    # Runtime type aliases - use generic Python types
    VectorType = List[float]  # Generic list of floats for vectors

from .base import RelationshipDetector

logger = logging.getLogger(__name__)

# Define global flags for availability - these will be set at runtime
NUMPY_AVAILABLE = False

# Try to import NumPy conditionally at runtime
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("NumPy not available. Will use slower pure Python vector operations.")


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
            source_embedding = element_embeddings[source_id]

            # Only calculate for elements after this one (avoid duplicates)
            for target_id in element_ids[i + 1:]:
                target_embedding = element_embeddings[target_id]

                # Calculate cosine similarity
                similarity = self._cosine_similarity(source_embedding, target_embedding)

                # Add to results
                similarities.append(((source_id, target_id), similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Limit to max relationships
        return similarities[:self.max_relationships * len(element_ids)]

    def _cosine_similarity(self, vec1: Union[VectorType, 'np.ndarray'],
                           vec2: Union[VectorType, 'np.ndarray']) -> float:
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
    def _cosine_similarity_numpy(vec1: Union[VectorType, 'np.ndarray'],
                                 vec2: Union[VectorType, 'np.ndarray']) -> float:
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

        # Convert to numpy arrays if not already
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def _cosine_similarity_python(vec1: VectorType, vec2: VectorType) -> float:
        """
        Calculate cosine similarity between two vectors using pure Python.
        This is a fallback when NumPy is not available.

        Args:
            vec1: First vector (list of floats)
            vec2: Second vector (list of floats)

        Returns:
            Cosine similarity (between -1 and 1)
        """
        # Check vector lengths
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")

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

    @staticmethod
    def _generate_id(prefix: str = "") -> str:
        """Generate a unique ID."""
        return f"{prefix}{uuid.uuid4()}"
