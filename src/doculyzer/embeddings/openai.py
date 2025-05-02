import hashlib
import logging
import os
from typing import List, Dict, Any, Optional

from .base import EmbeddingGenerator
from ..config import Config

logger = logging.getLogger(__name__)

# Try to import OpenAI library, but don't fail if not available
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with: pip install openai")


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator using OpenAI's embedding models."""

    # Default model dimensions (can be overridden in config)
    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072
    }

    def __init__(self, _config: Config, model_name: str = "text-embedding-3-small",
                 api_key: Optional[str] = None,
                 dimensions: Optional[int] = None):
        """
        Initialize the OpenAI embedding generator.

        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key (optional, can also be set via OPENAI_API_KEY env var)
            dimensions: Override dimensions (optional)
        """
        super().__init__(_config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required for OpenAI embeddings")

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.configured_dimensions = dimensions
        self.cache = {}  # Simple cache for embeddings

        # Set up the client
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it in config or OPENAI_API_KEY env var.")

        # Initialize client
        openai.api_key = self.api_key

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
            # Make API request
            response = openai.embeddings.create(
                model=self.model_name,
                input=text
            )

            # Extract embedding from response
            embedding = response.data[0].embedding

            # Cache embedding
            self.cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {str(e)}")
            # Return zero vector on error
            return [0.0] * self.get_dimensions()

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        # Check cache for each text
        results = list()
        uncached_texts = list()
        uncached_indices = list()

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
                # Make batch API request
                response = openai.embeddings.create(
                    model=self.model_name,
                    input=uncached_texts
                )

                # Extract embeddings from response
                embeddings = [item.embedding for item in response.data]

                # Update results and cache
                for i, idx in enumerate(uncached_indices):
                    results[idx] = embeddings[i]
                    cache_key = self._get_cache_key(texts[idx])
                    self.cache[cache_key] = embeddings[i]

            except Exception as e:
                logger.error(f"Error generating batch OpenAI embeddings: {str(e)}")
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
        return self.MODEL_DIMENSIONS.get(self.model_name, 1536)

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
