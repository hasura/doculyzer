import logging
import os
from datetime import datetime

import time

from .config import Config
from .main import ingest_documents


def _setup_logger():
    logger = logging.getLogger("doculyzer_crawler")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _ingest(config_path, logger):
    logger.info(f"Starting crawl at {datetime.now()}")

    # Load configuration
    config = Config(config_path)

    # Initialize database
    config.initialize_database()

    # Ingest documents (only processes changed documents)
    try:
        stats = ingest_documents(config)
        logger.info(f"Crawl completed: {stats['documents']} documents processed, "
                    f"{stats['unchanged_documents']} unchanged, "
                    f"{stats['elements']} elements, "
                    f"{stats['relationships']} relationships")
    except Exception as e:
        logger.error(f"Error during crawl: {str(e)}")
    finally:
        config.close_database()


def crawl(config_path: str = None, interval: int = None):
    config = Config(config_path if config_path else os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
    interval = interval if interval is not None else int(os.environ.get("CRAWLER_INTERVAL", "86400"))

    logger = _setup_logger()
    logger.info(f"Crawler initialized with interval {interval} seconds")

    while True:
        _ingest(config, logger)
        logger.info(f"Sleeping for {interval} seconds")
        time.sleep(interval)
