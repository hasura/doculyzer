"""
Factory module for creating document parsers.

This module provides factory functions to create appropriate parsers
for different document types.
"""

import logging
from typing import Dict, Any, Optional

from .base import DocumentParser
from .csv import CsvParser  # Import the new CSV parser
from .docx import DocxParser
from .html import HtmlParser
from .markdown import MarkdownParser
from .pdf import PdfParser
from .pptx import PptxParser
from .text import TextParser
from .xlsx import XlsxParser
from .xml import XmlParser

logger = logging.getLogger(__name__)


def create_parser(doc_type: str, config: Optional[Dict[str, Any]] = None) -> DocumentParser:
    """
    Factory function to create appropriate parser for document type.

    Args:
        doc_type: Document type ('markdown', 'html', 'text', 'xlsx', 'docx', 'pdf', 'pptx', 'xml', 'csv')
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
        return PdfParser(config)
    elif doc_type == "xml":
        return XmlParser(config)
    elif doc_type == "docx":
        return DocxParser(config)
    elif doc_type == "pptx":
        return PptxParser(config)
    elif doc_type == "csv":
        return CsvParser(config)  # Added CSV parser support
    elif doc_type == "text":
        return TextParser(config)
    else:
        logger.warning(f"Unsupported document type: {doc_type}, falling back to text parser")
        return TextParser(config)


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
    if doc_type == 'text' and metadata.get('content_type', '').lower().startswith('text/'):
        doc_type = metadata.get('content_type', '').lower().replace("text/", "")
    if doc_type == 'text' and metadata.get('content_type', '').lower().startswith('application/'):
        doc_type = metadata.get('content_type', '').lower().replace("application/", "")

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
        elif filename.lower().endswith(('.xml', '.xsd', '.rdf', '.rss', '.svg', '.wsdl', '.xslt')):
            doc_type = "xml"
        elif filename.lower().endswith(('.csv', '.tsv')):
            doc_type = "csv"  # Added CSV file extensions detection
        elif filename.lower().endswith(('.md', '.markdown', '.mdown')):
            doc_type = "markdown"
        elif filename.lower().endswith(('.txt', '.text')):
            doc_type = "text"
        elif filename.lower().endswith(('.htm', '.html', '.xhtml')):
            doc_type = "html"
        # Check content type if extension didn't give us an answer
        elif "application/pdf" in content_type.lower():
            doc_type = "pdf"
        elif "markdown" in content_type.lower() or "md" in content_type.lower():
            doc_type = "markdown"
        elif "html" in content_type.lower() or "xhtml" in content_type.lower():
            doc_type = "html"
        elif "spreadsheet" in content_type.lower() or "excel" in content_type.lower():
            doc_type = "xlsx"
        elif "xml" in content_type.lower() or "application/xml" in content_type.lower() or "text/xml" in content_type.lower():
            doc_type = "xml"
        elif "csv" in content_type.lower() or "text/csv" in content_type.lower() or "text/tab-separated-values" in content_type.lower():
            doc_type = "csv"  # Added CSV content type detection
        elif "msword" in content_type.lower() or "officedocument.wordprocessingml" in content_type.lower():
            doc_type = "docx"
        elif "officedocument.presentationml" in content_type.lower() or "powerpoint" in content_type.lower():
            doc_type = "pptx"
        elif "text/plain" in content_type.lower():
            doc_type = "text"
        else:
            # Default to text
            doc_type = "text"

    # Create and return parser
    return create_parser(doc_type, config)
