"""
Content Source Module for the document pointer system.

This module contains adapters for different content sources such as:
- Markdown files
- Database blobs
- Web URLs
"""

import logging
from typing import Dict, Any

from .base import ContentSource
from .confluence import ConfluenceContentSource
from .database import DatabaseContentSource
from .file import FileContentSource
from .jira import JiraContentSource
from .mongodb import MongoDBContentSource
from .s3 import S3ContentSource
from .servicenow import ServiceNowContentSource
from .web import WebContentSource

logger = logging.getLogger(__name__)


def get_content_source(source_config: Dict[str, Any]) -> ContentSource:
    """
    Factory function to create appropriate content source from config.

    Args:
        source_config: Content source configuration

    Returns:
        ContentSource instance

    Raises:
        ValueError: If source type is not supported
    """
    source_type = source_config.get("type")

    if source_type == "file":
        return FileContentSource(source_config)
    elif source_type == "database":
        return DatabaseContentSource(source_config)
    elif source_type == "web":
        return WebContentSource(source_config)
    elif source_type == "confluence":
        return ConfluenceContentSource(source_config)
    elif source_type == "jira":
        return JiraContentSource(source_config)
    elif source_type == "s3":
        return S3ContentSource(source_config)
    elif source_type == "servicenow":
        return ServiceNowContentSource(source_config)
    elif source_type == "mongodb":  # Add the MongoDB option
        return MongoDBContentSource(source_config)
    else:
        raise ValueError(f"Unsupported content source type: {source_type}")
