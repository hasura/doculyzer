import hashlib
import logging
from typing import List, Dict, Any, Optional

from .base import EmbeddingGenerator
from ..config import Config

logger = logging.getLogger(__name__)

# Try to import FastEmbed library, but don't fail if not available
try:
    from fastembed import TextEmbedding

    FASTEMBED_AVAILABLE = True
except ImportError:
    TextEmbedding = None
    FASTEMBED_AVAILABLE = False
    logger.warning("FastEmbed not available. Install with: pip install fastembed")


class FastEmbedGenerator(EmbeddingGenerator):
    """Embedding generator using FastEmbed models."""

    # Default model dimensions (can be overridden in config)
    MODEL_DIMENSIONS = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "intfloat/e5-base-v2": 768
    }

    def __init__(self, _config: Config, model_name: str = "BAAI/bge-small-en-v1.5",
                 dimensions: Optional[int] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize the FastEmbed embedding generator.

        Args:
            model_name: Name of the FastEmbed model
            dimensions: Override dimensions (optional)
            cache_dir: Directory to cache models (optional)
        """
        super().__init__(_config)
        if not FASTEMBED_AVAILABLE:
            raise ImportError("FastEmbed library is required for FastEmbed embeddings")

        self.model_name = model_name
        self.configured_dimensions = dimensions
        self.cache_dir = cache_dir
        self.cache = {}  # Simple cache for embeddings
        self.model = None

        # Load the model
        self._load_model()

    def _load_model(self) -> None:
        """Load the FastEmbed model."""
        try:
            # Initialize the TextEmbedding model
            self.model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir
            )
            logger.info(f"Loaded FastEmbed model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading FastEmbed model {self.model_name}: {str(e)}")
            raise

    def generate(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not text:
            # Return zero vector for empty text
            return [0.0] * self.get_dimensions()

        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Generate embedding
        try:
            # Ensure model is loaded
            if self.model is None:
                self._load_model()

            # FastEmbed returns an iterator with a single embedding for a single text
            embeddings = list(self.model.embed([text]))
            if not embeddings:
                raise ValueError("No embedding generated")

            embedding = embeddings[0].tolist()

            # Cache embedding
            self.cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Error generating FastEmbed embedding: {str(e)}")
            # Return zero vector on error
            return [0.0] * self.get_dimensions()

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        # Check cache for each text
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if not text:
                # Empty text gets zero vector
                results.append([0.0] * self.get_dimensions())
            else:
                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    results.append(self.cache[cache_key])
                else:
                    # Mark for batch processing
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    # Add placeholder
                    results.append(None)

        if uncached_texts:
            try:
                # Ensure model is loaded
                if self.model is None:
                    self._load_model()

                # Generate embeddings - FastEmbed already handles batching internally
                embeddings = list(self.model.embed(uncached_texts))

                # Update results and cache
                for i, embedding in zip(uncached_indices, embeddings):
                    embedding_list = embedding.tolist()
                    results[i] = embedding_list
                    cache_key = self._get_cache_key(texts[i])
                    self.cache[cache_key] = embedding_list

            except Exception as e:
                logger.error(f"Error generating batch FastEmbed embeddings: {str(e)}")
                # Fill in with zero vectors for any that failed
                for i in uncached_indices:
                    if results[i] is None:
                        results[i] = [0.0] * self.get_dimensions()

        return results

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        # First check if dimensions were explicitly configured
        if self.configured_dimensions is not None:
            return self.configured_dimensions

        # Otherwise use model's default dimensions
        return self.MODEL_DIMENSIONS.get(self.model_name, 384)

    def get_model_name(self) -> str:
        """Get embedding model name."""
        return self.model_name

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache = {}

    @staticmethod
    def _get_cache_key(text: str) -> str:
        """
        Generate a cache key for text.

        Args:
            text: Input text

        Returns:
            Cache key string
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def generate_from_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Generate embeddings for document elements."""
        embeddings = {}

        for element in elements:
            element_pk = element["element_pk"]

            # Skip root element
            if element["element_type"] == "root":
                continue

            # Get content from preview
            content = element.get("content_preview", "")
            if not content:
                continue

            # Generate embedding directly from content
            embedding = self.generate(content)
            embeddings[element_pk] = embedding

        return embeddings
