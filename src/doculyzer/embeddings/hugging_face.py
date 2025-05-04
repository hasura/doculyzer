import hashlib
import logging
from typing import List, Dict, Any

from .base import EmbeddingGenerator
from ..config import Config

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator using Hugging Face Sentence Transformers."""

    def __init__(self, _config: Config, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the Hugging Face embedding generator.

        Args:
            model_name: Name of the Sentence Transformers model
        """
        super().__init__(_config)
        self.model_name = model_name
        self.model = None
        self.cache = {}  # Simple cache for embeddings

        # Load model
        self._load_model()

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
            import torch

            # Ensure model is loaded
            if self.model is None:
                self._load_model()

            # Generate embedding
            with torch.no_grad():
                embedding = self.model.encode(text, normalize_embeddings=True, show_progress_bar=True, )

            # Convert to list
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.tolist()
            elif isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], torch.Tensor):
                embedding = [tensor.tolist() for tensor in embedding]

            # Convert numpy arrays to lists if needed
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()

            # Cache embedding
            self.cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
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
                import torch

                # Ensure model is loaded
                if self.model is None:
                    self._load_model()

                # Generate embeddings
                with torch.no_grad():
                    embeddings = self.model.encode(uncached_texts, normalize_embeddings=True, show_progress_bar=True, )

                # Convert to lists
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.tolist()
                elif isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], torch.Tensor):
                    embeddings = [tensor.tolist() for tensor in embeddings]

                # Convert numpy arrays to lists if needed
                if hasattr(embeddings, 'tolist'):
                    embeddings = embeddings.tolist()

                # Update results and cache
                for i, embedding in zip(uncached_indices, embeddings):
                    results[i] = embedding
                    cache_key = self._get_cache_key(texts[i])
                    self.cache[cache_key] = embedding

            except Exception as e:
                logger.error(f"Error generating batch embeddings: {str(e)}")
                # Fill in with zero vectors for any that failed
                for i in uncached_indices:
                    if results[i] is None:
                        results[i] = [0.0] * self.get_dimensions()

        return results

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        if self.model is None:
            self._load_model()

        return self.model.get_sentence_embedding_dimension()

    def get_model_name(self) -> str:
        """Get embedding model name."""
        return self.model_name

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache = {}

    def _load_model(self) -> None:
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")

        except ImportError:
            logger.error("Error loading sentence-transformers. Please install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading embedding model {self.model_name}: {str(e)}")
            raise

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
