"""
Plain text document parser module for the document pointer system.

This module parses plain text documents into structured paragraph elements.
"""

import json
import logging
import os
import re
from typing import Dict, Any, List, Optional, Union

from .base import DocumentParser
from ..storage import ElementType

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
        self.temp_dir = self.config.get("temp_dir", os.path.join(os.path.dirname(__file__), 'temp'))

    def _resolve_element_text(self, location_data: Dict[str, Any], source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve the plain text representation of a text document element.

        Args:
            location_data: Content location data
            source_content: Optional preloaded source content

        Returns:
            Plain text representation of the element
        """
        # For plain text documents, the content is already in text form
        # So we can leverage the existing _resolve_element_content method
        content = self._resolve_element_content(location_data, source_content)

        # For plain text, we don't need to do much transformation
        # The element is already in text format without markup
        element_type = location_data.get("type", "")

        # Handle different element types
        if element_type == ElementType.ROOT.value or not element_type:
            # For the full document, we might want to clean it up
            if self.normalize_whitespace:
                # Normalize line endings and reduce excessive whitespace
                content = re.sub(r'\r\n', '\n', content)
                content = re.sub(r'\r', '\n', content)
                content = re.sub(r'\n{3,}', '\n\n', content)  # Reduce excessive blank lines
            return content.strip() if self.strip_whitespace else content

        elif element_type == ElementType.PARAGRAPH.value:
            # For paragraphs, just return the text (already clean)
            return content.strip() if self.strip_whitespace else content

        elif element_type == ElementType.LINE.value:
            # For a single line, just return it
            return content.strip() if self.strip_whitespace else content

        elif element_type == ElementType.RANGE.value or element_type == ElementType.SUBSTRING.value:
            # For text ranges or substrings, return as is
            return content.strip() if self.strip_whitespace else content

        # Default case: return the content as is
        return content.strip() if self.strip_whitespace else content

    def _resolve_element_content(self, location_data: Dict[str, Any],
                                 source_content: Optional[Union[str, bytes]] = None) -> str:
        """
        Resolve content for specific text document element types.

        Args:
            location_data: Content location data
            source_content: Optional pre-loaded source content

        Returns:
            Resolved content string
        """
        source = location_data.get("source", "")
        element_type = location_data.get("type", "")

        # Load the source content if not provided
        content = source_content
        if content is None:
            # Check if source is a file path
            if os.path.exists(source):
                try:
                    with open(source, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # Try to read as binary if text fails
                    with open(source, 'rb') as f:
                        binary_content = f.read()
                        try:
                            content = binary_content.decode('utf-8')
                        except UnicodeDecodeError:
                            return f"(Binary content of {len(binary_content)} bytes, cannot be displayed as text)"
            else:
                raise ValueError(f"Source file not found: {source}")
        elif isinstance(content, bytes):
            # Convert binary content to string
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                return f"(Binary content of {len(content)} bytes, cannot be displayed as text)"

        # Handle different element types
        if element_type == ElementType.ROOT.value or not element_type:
            # Return the entire document content
            return content

        elif element_type == ElementType.PARAGRAPH.value:
            # Extract specific paragraph by index
            paragraph_index = location_data.get("index", 0)

            # Split content into paragraphs
            paragraphs = self._split_into_paragraphs(content)

            # Check if paragraph index is valid
            if 0 <= paragraph_index < len(paragraphs):
                return paragraphs[paragraph_index]
            else:
                return f"Paragraph index {paragraph_index} out of range. Document has {len(paragraphs)} paragraphs."

        elif element_type == ElementType.LINE.value:
            # Extract specific line by index
            line_index = location_data.get("line", 0)

            # Split content into lines
            lines = content.split('\n')

            # Check if line index is valid
            if 0 <= line_index < len(lines):
                return lines[line_index]
            else:
                return f"Line index {line_index} out of range. Document has {len(lines)} lines."

        elif element_type == ElementType.RANGE.value:
            # Extract a range of text
            start = location_data.get("start", 0)
            end = location_data.get("end", len(content))

            # Check if range is valid
            if 0 <= start < len(content) and start < end <= len(content):
                return content[start:end]
            else:
                return f"Invalid range: start={start}, end={end}. Content length is {len(content)}."

        elif element_type == ElementType.SUBSTRING.value:
            # Extract text by search string
            search_string = location_data.get("search", "")
            occurrence = location_data.get("occurrence", 0)  # 0-based index
            context_chars = location_data.get("context", 50)  # Characters before and after match

            if not search_string:
                return "No search string specified"

            # Find all occurrences
            positions = [match.start() for match in re.finditer(re.escape(search_string), content)]

            # Check if we found the occurrence
            if 0 <= occurrence < len(positions):
                start_pos = max(0, positions[occurrence] - context_chars)
                end_pos = min(len(content), positions[occurrence] + len(search_string) + context_chars)

                # Add ellipsis if we're not at the beginning or end
                prefix = "..." if start_pos > 0 else ""
                suffix = "..." if end_pos < len(content) else ""

                return f"{prefix}{content[start_pos:end_pos]}{suffix}"
            else:
                if len(positions) == 0:
                    return f"String '{search_string}' not found in content."
                else:
                    return f"Occurrence {occurrence} not found. String '{search_string}' appears {len(positions)} times."

        else:
            # Default: return the entire content for unknown element types
            return content

    def supports_location(self, content_location: str) -> bool:
        """
        Check if this parser supports resolving the given location.

        Args:
            content_location: Content location pointer

        Returns:
            True if supported, False otherwise
        """
        try:
            location_data = json.loads(content_location)
            source = location_data.get("source", "")

            # If source is a file, check if it exists and has a text extension
            if os.path.exists(source) and os.path.isfile(source):
                _, ext = os.path.splitext(source.lower())
                # Support common text file extensions
                return ext in ['.txt', '.text', '.md', '.markdown', '.log', '.csv', '.tsv', '.json', '.xml', '.html',
                               '.htm', '.css', '.js', '.py', '.sh', '.bat', '.ini', '.cfg', '']

            # For non-file sources, check if we have the appropriate element type
            element_type = location_data.get("type", "")
            return element_type in [
                ElementType.ROOT.value,
                ElementType.PARAGRAPH.value,
                ElementType.LINE.value,
                ElementType.RANGE.value,
                ElementType.SUBSTRING.value
            ]

        except (json.JSONDecodeError, TypeError):
            return False

    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a text document into structured elements.

        Args:
            doc_content: Document content and metadata

        Returns:
            Dictionary with document metadata, elements, relationships, and extracted links
        """
        # Extract metadata from doc_content
        source_id = doc_content["id"]
        metadata = doc_content.get("metadata", {}).copy()  # Make a copy to avoid modifying original

        # Get content from binary_path or direct content
        content = None
        if "binary_path" in doc_content and os.path.exists(doc_content["binary_path"]):
            try:
                with open(doc_content["binary_path"], 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try to read as binary if text fails
                with open(doc_content["binary_path"], 'rb') as f:
                    binary_content = f.read()
                    try:
                        content = binary_content.decode('utf-8')
                    except UnicodeDecodeError:
                        raise ValueError(f"Cannot decode content as text: {doc_content['binary_path']}")
        elif "content" in doc_content:
            content = doc_content["content"]
            if isinstance(content, bytes):
                try:
                    content = content.decode('utf-8')
                except UnicodeDecodeError:
                    raise ValueError("Cannot decode binary content as text")

        if content is None:
            raise ValueError("No content provided for text parsing")

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
                "element_type": ElementType.PARAGRAPH.value,
                "parent_id": parent_id,
                "content_preview": paragraph[:100] + ("..." if len(paragraph) > 100 else ""),
                "content_location": json.dumps({
                    "source": source_id,
                    "type": ElementType.PARAGRAPH.value,
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

    @staticmethod
    def _generate_hash(content: str) -> str:
        """
        Generate a hash of content for change detection.

        Args:
            content: Text content

        Returns:
            MD5 hash of content
        """
        import hashlib
        return hashlib.md5(content.encode('utf-8')).hexdigest()
