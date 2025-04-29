import logging
import os

import pytest
from dotenv import load_dotenv

from doculyzer.embeddings import EmbeddingGenerator, get_embedding_generator

# Load environment variables from .env file
load_dotenv()
from doculyzer import Config

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
    _embedding_generator = get_embedding_generator(_config)
    return _config, _embedding_generator


def test_document_ingestion(config_emb: (Config, EmbeddingGenerator)):
    """Test the full document ingestion process."""
    from doculyzer.main import ingest_documents
    from doculyzer.content_resolver import create_content_resolver

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
        if _embedding_generator:
            # Run a sample search
            logger.info("Running similarity search")
            query_text = "document management"
            logger.debug(f"Generating embedding for query: {query_text}")
            query_embedding = _embedding_generator.generate(query_text)

            logger.debug("Searching for similar elements")
            similar_elements = db.search_by_embedding(query_embedding)
            logger.info(f"Found {len(similar_elements)} similar elements")

            # Display a few results
            for i, (element_id, similarity) in enumerate(similar_elements[:10]):
                # Get the element
                element = db.get_element(element_id)
                if element:
                    logger.info(f"Similar element {i + 1}: {element.get('element_type')}, Similarity: {similarity}")

                    # Try to resolve content
                    content_location = element.get("content_location")
                    if content_location and content_resolver.supports_location(content_location):
                        try:
                            original_content = content_resolver.resolve_content(content_location)
                            logger.info(f"Content preview: {original_content[:100]}...")
                        except Exception as e:
                            logger.error(f"Error resolving content: {str(e)}")
                else:
                    logger.warning(f"Could not find element with ID: {element_id}")

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
