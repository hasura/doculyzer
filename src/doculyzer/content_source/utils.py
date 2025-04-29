"""
Utility functions for document parsing.

This module provides helper functions for detecting content types and routing documents
to the appropriate parser.
"""

import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)


def detect_content_type(content: str, metadata: Dict[str, Any] = None) -> str:
    """
    Detect content type from content and metadata.

    Args:
        content: Document content
        metadata: Optional metadata that might contain content type information

    Returns:
        Content type: 'markdown', 'html', or 'text'
    """
    metadata = metadata or {}

    # Check metadata first
    content_type = metadata.get("content_type", "")
    if content_type:
        if "markdown" in content_type.lower() or "md" in content_type.lower():
            return "markdown"
        elif "html" in content_type.lower() or "xhtml" in content_type.lower():
            return "html"

    # Check content patterns
    if content.strip().startswith(('<!DOCTYPE html>', '<html', '<?xml')):
        return "html"

    # Count markdown-specific patterns
    md_patterns = 0
    if re.search(r'^#{1,6}\s+', content, re.MULTILINE):  # Headers
        md_patterns += 1
    if re.search(r'^-\s+', content, re.MULTILINE):  # List items
        md_patterns += 1
    if re.search(r'\[.+?\]\(.+?\)', content):  # Links
        md_patterns += 1
    if re.search(r'^```', content, re.MULTILINE):  # Code blocks
        md_patterns += 1

    if md_patterns >= 2:
        return "markdown"

    # Count HTML-specific patterns
    html_patterns = 0
    if re.search(r'<[a-z]+[^>]*>', content):  # Opening tags
        html_patterns += 1
    if re.search(r'</[a-z]+>', content):  # Closing tags
        html_patterns += 1
    if re.search(r'<[a-z]+[^>]*/>', content):  # Self-closing tags
        html_patterns += 1

    if html_patterns >= 2:
        return "html"

    # Default to text
    return "text"


def extract_url_links(content: str, element_id: str) -> list:
    """
    Extract URL links from plain text.

    Args:
        content: Text content
        element_id: Source element ID

    Returns:
        List of link objects
    """
    links = []

    # Simple URL pattern for plain text
    url_pattern = r'https?://[^\s()<>]+(?:\([\w\d]+\)|(?:[^,.;:`!()\[\]{}<>"\'\s]|/))'

    matches = re.findall(url_pattern, content)

    for url in matches:
        links.append({
            "source_id": element_id,
            "link_text": url,  # Use the URL as the text for plain text links
            "link_target": url,
            "link_type": "url"
        })

    return links
