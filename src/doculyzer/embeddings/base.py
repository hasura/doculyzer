from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..config import Config


class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generators."""

    _config: Config

    def __init__(self, _config: Config):
        """
        Initialize the base class with a config.

        Args:
            _config: The Config
        """
        self._config = _config  # New field added to the base class

    @abstractmethod
    def generate(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Vector embedding as list of floats
        """
        pass

    @abstractmethod
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of vector embeddings
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """
        Get embedding dimensions.

        Returns:
            Number of dimensions in the embedding
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get embedding model name.

        Returns:
            Model name
        """
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """
        Clear the embedding cache.
        """
        pass

    @abstractmethod
    def generate_from_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Generate embeddings for document elements.

        Args:
            elements: List of document elements

        Returns:
            Dictionary mapping element_id to embedding
        """
        pass
