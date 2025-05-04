"""
Embedding Generator Module for the document pointer system.

This module generates vector embeddings for document elements,
with support for different embedding models and contextual embeddings.
"""

import logging

from .base import EmbeddingGenerator
from .contextual_embedding import ContextualEmbeddingGenerator
from .hugging_face import HuggingFaceEmbeddingGenerator
from .openai import OpenAIEmbeddingGenerator
from ..config import Config

logger = logging.getLogger(__name__)


def get_embedding_generator(config: Config) -> EmbeddingGenerator:
    """
    Factory function to create embedding generator from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        EmbeddingGenerator instance
    """
    embeddings = config.config.get("embedding", {})

    # Get provider (defaults to huggingface)
    provider = embeddings.get("provider", "huggingface")

    # Get optional dimensions configuration
    dimensions = embeddings.get("dimensions", None)

    # Create base generator based on provider
    if provider.lower() == "openai":
        # Get OpenAI-specific config
        model = embeddings.get("model", "text-embedding-3-small")
        api_key = embeddings.get("api_key", None)

        # Create OpenAI generator
        base_generator = OpenAIEmbeddingGenerator(config, model, api_key, dimensions)
        logger.info(f"Created OpenAI embedding generator with model {model}")
    else:
        # Default to Hugging Face
        model = embeddings.get("model", "sentence-transformers/all-MiniLM-L6-v2")

        # Create Hugging Face generator
        base_generator = HuggingFaceEmbeddingGenerator(config, model)
        logger.info(f"Created Hugging Face embedding generator with model {model}")

    # Add contextual embedding if configured
    if embeddings.get("contextual", False):
        window_size = embeddings.get("window_size", 3)
        overlap_size = embeddings.get("overlap_size", 1)
        predecessor_count = embeddings.get("predecessor_count", 1)
        successor_count = embeddings.get("successor_count", 1)
        ancestor_depth = embeddings.get("ancestor_depth", 1)

        contextual_generator = ContextualEmbeddingGenerator(
            config,
            base_generator,
            window_size,
            overlap_size,
            predecessor_count,
            successor_count,
            ancestor_depth
        )
        logger.info(f"Added contextual embedding wrapper")
        return contextual_generator

    return base_generator
