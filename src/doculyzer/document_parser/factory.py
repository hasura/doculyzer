"""
Factory module for creating document parsers.

This module provides factory functions to create appropriate parsers
for different document types.
"""

import logging
from typing import Dict, Any, Optional

from .base import DocumentParser
from .html_parser import HtmlParser
from .markdown_parser import MarkdownParser
from .pdf import PdfParser
from .xlsx import XlsxParser

logger = logging.getLogger(__name__)


def create_parser(doc_type: str, config: Optional[Dict[str, Any]] = None) -> DocumentParser:
    """
    Factory function to create appropriate parser for document type.

    Args:
        doc_type: Document type ('markdown', 'html', 'text', 'xlsx', 'docx', 'pdf', 'pptx')
        config: Parser configuration

    Returns:
        DocumentParser instance

    Raises:
        ValueError: If parser type is not supported
    """
    config = config or {}

    if doc_type == "markdown":
        return MarkdownParser(config)
    elif doc_type == "html":
        return HtmlParser(config)
    elif doc_type == "xlsx":
        return XlsxParser(config)
    elif doc_type == "pdf":
        return PdfParser(config)  # Added PDF support
    elif doc_type == "text":
        # For plain text, we use a simplified version of the Markdown parser
        # that only handles paragraphs and extracts URLs
        text_config = config.copy()
        text_config["extract_front_matter"] = False
        return MarkdownParser(text_config)
    else:
        logger.warning(f"Unsupported document type: {doc_type}, falling back to text parser")
        text_config = config.copy()
        text_config["extract_front_matter"] = False
        return MarkdownParser(text_config)


def get_parser_for_content(content: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> DocumentParser:
    """
    Get appropriate parser for content based on metadata.

    Args:
        content: Document content with metadata
        config: Parser configuration

    Returns:
        DocumentParser instance
    """
    doc_type = content.get("doc_type")
    metadata = content.get("metadata", {})

    # If doc_type is not specified, check metadata
    if not doc_type:
        content_type = metadata.get("content_type", "")
        filename = metadata.get("filename", "")

        # Check file extension first
        if filename.lower().endswith('.pdf'):
            doc_type = "pdf"
        elif filename.lower().endswith(('.xlsx', '.xls')):
            doc_type = "xlsx"
        elif filename.lower().endswith(('.docx', '.doc')):
            doc_type = "docx"
        elif filename.lower().endswith(('.pptx', '.ppt')):
            doc_type = "pptx"
        # Check content type if extension didn't give us an answer
        elif "application/pdf" in content_type.lower():
            doc_type = "pdf"
        elif "markdown" in content_type.lower() or "md" in content_type.lower():
            doc_type = "markdown"
        elif "html" in content_type.lower() or "xhtml" in content_type.lower():
            doc_type = "html"
        elif "spreadsheet" in content_type.lower() or "excel" in content_type.lower():
            doc_type = "xlsx"
        else:
            # Default to text
            doc_type = "text"

    # Create and return parser
    return create_parser(doc_type, config)
