import logging
import os


def configure_logging():
    # Get log level from an environment variable (default to INFO)
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),  # Set global log level
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
