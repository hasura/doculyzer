#!/usr/bin/env python3
"""
Doculyzer Startup Script

This script runs a single document ingestion process using the configuration
specified in a config file.
"""

import argparse
import logging
import sys
from pathlib import Path


# Setup logging early
def setup_logging(log_level="INFO"):
    """Configure logging for the startup script."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("doculyzer_startup")


# Parse command line arguments
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Doculyzer Startup - Run a single document ingestion")
    parser.add_argument("--config", required=False, default="config.yaml",
                        help="Path to configuration file (default: config.yaml)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Set the logging level")
    return parser.parse_args()


def main():
    """Main function to run the document ingestion process once."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info(f"Starting Doculyzer with config file: {args.config}")

    try:
        # Initialize config
        from doculyzer import Config
        config_path = Path(args.config)

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)

        logger.info(f"Loading configuration from {config_path}")
        config = Config(str(config_path))
        config.get_document_database().initialize()

        # Import ingest_documents function
        from doculyzer import ingest_documents

        # Run ingestion
        logger.info("Starting document ingestion process")
        stats = ingest_documents(config)

        # Log results
        logger.info("Document ingestion completed successfully")
        logger.info(f"Processed {stats['documents']} documents with {stats['elements']} elements")
        logger.info(f"Created {stats['relationships']} relationships")
        if 'unchanged_documents' in stats:
            logger.info(f"Skipped {stats['unchanged_documents']} unchanged documents")
        if 'semantic_relationships' in stats:
            logger.info(f"Created {stats['semantic_relationships']} semantic relationships")

        return 0

    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure Doculyzer is installed correctly")
        return 1
    except Exception as e:
        logger.error(f"Error during document ingestion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
