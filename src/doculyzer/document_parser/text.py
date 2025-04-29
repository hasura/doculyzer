"""
Plain text document parser module for the document pointer system.

This module parses plain text documents into structured paragraph elements.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional

from .base import DocumentParser

logger = logging.getLogger(__name__)


class TextParser(DocumentParser):
    """Parser for plain text documents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the text parser."""
        super().__init__(config)

        # Configuration options
        self.config = config or {}
        self.paragraph_separator = self.config.get("paragraph_separator", "\n\n")
        self.min_paragraph_length = self.config.get("min_paragraph_length", 1)
        self.extract_urls = self.config.get("extract_urls", True)
        self.extract_email_addresses = self.config.get("extract_email_addresses", True)
        self.strip_whitespace = self.config.get("strip_whitespace", True)
        self.normalize_whitespace = self.config.get("normalize_whitespace", True)

    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a text document into structured elements."""
        # Extract metadata from doc_content
        content = doc_content["content"]
        source_id = doc_content["id"]
        metadata = doc_content.get("metadata", {}).copy()  # Make a copy to avoid modifying original

        # Generate document ID if not present
        doc_id = metadata.get("doc_id", self._generate_id("doc_"))

        # Create document record with metadata
        document = {
            "doc_id": doc_id,
            "doc_type": "text",
            "source": source_id,
            "metadata": self._extract_document_metadata(content, metadata),
            "content_hash": doc_content.get("content_hash", self._generate_hash(content))
        }

        # Create root element
        elements = [self._create_root_element(doc_id, source_id)]
        root_id = elements[0]["element_id"]

        # Parse document content into paragraphs
        paragraphs = self._split_into_paragraphs(content)
        elements.extend(self._create_paragraph_elements(paragraphs, doc_id, root_id, source_id))

        # Extract links from content
        links = self._extract_links(content, root_id)

        # Return the parsed document
        return {
            "document": document,
            "elements": elements,
            "links": links,
            "relationships": []
        }

    def _extract_document_metadata(self, content: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from text document content.

        Args:
            content: Document content
            base_metadata: Base metadata from content source

        Returns:
            Dictionary of document metadata
        """
        # Combine base metadata with computed metadata
        metadata = base_metadata.copy()

        # Calculate basic text statistics
        char_count = len(content)
        word_count = len(re.findall(r'\b\w+\b', content))
        line_count = content.count('\n') + 1
        paragraphs = self._split_into_paragraphs(content)
        paragraph_count = len(paragraphs)

        # Add computed statistics to metadata
        metadata.update({
            "char_count": char_count,
            "word_count": word_count,
            "line_count": line_count,
            "paragraph_count": paragraph_count
        })

        # Try to find a title (first non-empty line)
        lines = content.split('\n')
        for line in lines:
            if line.strip():
                metadata["title"] = line.strip()[:100]  # Use first 100 chars max
                break

        # Try to extract language if not already present
        if "language" not in metadata:
            # This would be a more complex detection in a real implementation
            # For now, just assume English
            metadata["language"] = "en"

        return metadata

    def _split_into_paragraphs(self, content: str) -> List[str]:
        """
        Split text content into paragraphs.

        Args:
            content: Document content

        Returns:
            List of paragraph strings
        """
        # Normalize line endings
        normalized_content = content.replace('\r\n', '\n').replace('\r', '\n')

        # Normalize whitespace if configured
        if self.normalize_whitespace:
            normalized_content = re.sub(r'\s+', ' ', normalized_content)
            normalized_content = re.sub(r'\n\s+', '\n', normalized_content)

        # Split by the configured paragraph separator
        paragraphs = normalized_content.split(self.paragraph_separator)

        # Filter and clean paragraphs
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            if self.strip_whitespace:
                paragraph = paragraph.strip()

            # Skip empty paragraphs or those below minimum length
            if paragraph and len(paragraph) >= self.min_paragraph_length:
                cleaned_paragraphs.append(paragraph)

        return cleaned_paragraphs

    def _create_paragraph_elements(self, paragraphs: List[str], doc_id: str, parent_id: str, source_id: str) -> List[
        Dict[str, Any]]:
        """
        Create paragraph elements from text paragraphs.

        Args:
            paragraphs: List of paragraph strings
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier

        Returns:
            List of paragraph elements
        """
        elements = []

        for idx, paragraph in enumerate(paragraphs):
            # Skip empty paragraphs
            if not paragraph:
                continue

            # Generate element ID
            element_id = self._generate_id(f"para_{idx}_")

            # Create paragraph element
            para_element = {
                "element_id": element_id,
                "doc_id": doc_id,
                "element_type": "paragraph",
                "parent_id": parent_id,
                "content_preview": paragraph[:100] + ("..." if len(paragraph) > 100 else ""),
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "paragraph",
                    "index": idx
                }),
                "content_hash": self._generate_hash(paragraph),
                "metadata": {
                    "index": idx,
                    "length": len(paragraph),
                    "word_count": len(re.findall(r'\b\w+\b', paragraph)),
                    "has_urls": bool(re.search(r'https?://\S+', paragraph)) if self.extract_urls else False,
                    "has_emails": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                                                 paragraph)) if self.extract_email_addresses else False
                }
            }

            elements.append(para_element)

        return elements

    def _extract_links(self, content: str, element_id: str) -> List[Dict[str, Any]]:
        """
        Extract links from text content.

        Args:
            content: Document content
            element_id: Source element ID

        Returns:
            List of extracted links
        """
        links = []

        if self.extract_urls:
            # Extract URLs
            url_pattern = r'(https?://[^\s<>"\'\(\)]+)'
            urls = re.findall(url_pattern, content)

            for url in urls:
                links.append({
                    "source_id": element_id,
                    "link_text": url,
                    "link_target": url,
                    "link_type": "url"
                })

        if self.extract_email_addresses:
            # Extract email addresses
            email_pattern = r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b'
            emails = re.findall(email_pattern, content)

            for email in emails:
                links.append({
                    "source_id": element_id,
                    "link_text": email,
                    "link_target": f"mailto:{email}",
                    "link_type": "email"
                })

        # Look for file paths
        file_path_pattern = r'(?:^|\s)([a-zA-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*|(?:/[^/\s:*?"<>|\r\n]+)+)(?:$|\s)'
        file_paths = re.findall(file_path_pattern, content)

        for path in file_paths:
            if path.strip():
                links.append({
                    "source_id": element_id,
                    "link_text": path,
                    "link_target": f"file://{path}",
                    "link_type": "file_path"
                })

        return links
