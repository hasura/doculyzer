"""
Embedding Generator Module for the document pointer system.
This module generates vector embeddings for document elements,
with support for different embedding models and contextual embeddings.
"""
import logging
from .base import EmbeddingGenerator
from .contextual_embedding import ContextualEmbeddingGenerator
from ..config import Config

logger = logging.getLogger(__name__)

# Conditionally import embedding providers
HUGGINGFACE_AVAILABLE = False
OPENAI_AVAILABLE = False
FASTEMBED_AVAILABLE = False

# Import HuggingFace provider if available
try:
    from .hugging_face import HuggingFaceEmbeddingGenerator
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    logger.warning("HuggingFace (sentence-transformers) not available. Install with: pip install sentence-transformers")

# Import OpenAI provider if available
try:
    from .openai import OpenAIEmbeddingGenerator
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI not available. Install with: pip install openai")

# Import FastEmbed provider if available
try:
    from .fastembed import FastEmbedGenerator
    FASTEMBED_AVAILABLE = True
except ImportError:
    logger.warning("FastEmbed not available. Install with: pip install fastembed")


def get_embedding_generator(config: Config) -> EmbeddingGenerator:
    """
    Factory function to create embedding generator from configuration.

    Args:
        config: Configuration object

    Returns:
        EmbeddingGenerator instance
    """
    embeddings = config.config.get("embedding", {})

    # Get provider (defaults to huggingface)
    provider = embeddings.get("provider", "huggingface").lower()

    # Get optional dimensions configuration
    dimensions = embeddings.get("dimensions", None)

    # Create base generator based on provider
    if provider == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required for OpenAI embeddings. Install with: pip install openai")

        # Get OpenAI-specific config
        model = embeddings.get("model", "text-embedding-3-small")
        api_key = embeddings.get("api_key", None)

        # Create OpenAI generator
        base_generator = OpenAIEmbeddingGenerator(config, model, api_key, dimensions)
        logger.info(f"Created OpenAI embedding generator with model {model}")

    elif provider == "fastembed":
        if not FASTEMBED_AVAILABLE:
            raise ImportError("FastEmbed library is required for FastEmbed embeddings. Install with: pip install fastembed")

        # Get FastEmbed-specific config
        model = embeddings.get("model", "BAAI/bge-small-en-v1.5")
        cache_dir = embeddings.get("cache_dir", None)

        # Create FastEmbed generator
        base_generator = FastEmbedGenerator(config, model, dimensions, cache_dir)
        logger.info(f"Created FastEmbed embedding generator with model {model}")

    else:
        # Default to Hugging Face
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("Sentence-Transformers library is required for HuggingFace embeddings. Install with: pip install sentence-transformers")

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
        child_count = embeddings.get("child_count", 1)

        contextual_generator = ContextualEmbeddingGenerator(
            config,
            base_generator,
            window_size,
            overlap_size,
            predecessor_count,
            successor_count,
            ancestor_depth,
            child_count
        )
        logger.info("Added contextual embedding wrapper")
        return contextual_generator

    return base_generator
