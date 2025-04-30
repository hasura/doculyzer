import logging
import os
from typing import List

import pytest
from dotenv import load_dotenv

from doculyzer.embeddings import EmbeddingGenerator, get_embedding_generator

# Load environment variables from .env file
load_dotenv()
from doculyzer import Config, search_with_content, SearchResult

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger('doculyzer_test')


@pytest.fixture
def config_emb() -> (Config, EmbeddingGenerator):
    """Load test configuration as a fixture."""
    _config = Config(os.environ.get('DOCULYZER_CONFIG_PATH', 'config.yaml'))
    _embedding_generator = None
    return _config, _embedding_generator


def test_document_ingestion(config_emb: (Config, EmbeddingGenerator)):
    """Test the full document ingestion process."""
    from doculyzer.main import ingest_documents
    from doculyzer.adapter import create_content_resolver

    _config, _embedding_generator = config_emb

    # Get database from config
    db = _config.get_document_database()

    # Initialize database
    logger.info("Initializing database")
    db.initialize()

    try:
        # Create content resolver
        content_resolver = create_content_resolver(_config)
        logger.info("Created content resolver")

        # Ingest documents
        logger.info("Starting document ingestion")
        stats = ingest_documents(_config)
        logger.info(f"Document ingestion completed: {stats}")

        # Assert documents were ingested
        # assert stats['documents'] > 0, "No documents were processed"
        # assert stats['elements'] > 0, "No elements were processed"

        # If embeddings enabled, test similarity search

        # Run a sample search
        logger.info("Running similarity search")
        query_text = "document management"
        logger.debug(f"Generating embedding for query: {query_text}")


        logger.debug("Searching for similar elements")
        results: List[SearchResult] = search_with_content(query_text)
        logger.info(f"Found {len(results)} similar elements")

        # Display a few results
        for i, result in enumerate(results[:10]):
            print(result)

        # Log summary
        logger.info(
            f"Processed {stats['documents']} documents with {stats['elements']} elements and {stats['relationships']} relationships")

        return stats
    finally:
        # Always close the database connection
        logger.info("Closing database connection")
        db.close()


if __name__ == "__main__":
    # This allows the test to be run directly as a script too
    config = Config('config.yaml')
    embedding_generator = get_embedding_generator(config)
    test_document_ingestion((config, embedding_generator))
