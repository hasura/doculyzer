"""
Document type detector module for the document pointer system.

This module provides utilities to detect document types based on file extension,
MIME type, or content inspection.
"""

import json
import logging
import mimetypes
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import python-magic, but don't fail if not available
try:
    # noinspection PyUnresolvedReferences
    import magic

    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logger.warning("python-magic not available. Install with 'pip install python-magic' for better content detection.")


class DocumentTypeDetector:
    """Detects document type from various inputs."""

    # Centralized MIME type to document type mapping
    MIME_TYPE_MAP = {
        'text/markdown': 'markdown',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'application/msword': 'docx',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
        'application/vnd.ms-excel': 'xlsx',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
        'application/vnd.ms-powerpoint': 'pptx',
        'application/pdf': 'pdf',
        'text/html': 'html',
        'application/xhtml+xml': 'html',
        'text/plain': 'text',
        'text/csv': 'csv',
        'text/tab-separated-values': 'csv',
        'application/csv': 'csv',
        'application/json': 'json',
        'application/xml': 'xml',
        'text/xml': 'xml',
        'application/x-yaml': 'yaml',
        'text/yaml': 'yaml',
        'application/yaml': 'yaml',
        'image/svg+xml': 'xml',
        'application/rdf+xml': 'xml',
        'application/rss+xml': 'xml',
        'application/xslt+xml': 'xml',
        'application/wsdl+xml': 'xml'
    }

    # Centralized file extension mapping
    EXTENSION_MAP = {
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.mdown': 'markdown',
        '.docx': 'docx',
        '.doc': 'docx',
        '.xlsx': 'xlsx',
        '.xls': 'xlsx',
        '.pptx': 'pptx',
        '.ppt': 'pptx',
        '.pdf': 'pdf',
        '.html': 'html',
        '.htm': 'html',
        '.xhtml': 'html',
        '.txt': 'text',
        '.text': 'text',
        '.csv': 'csv',
        '.tsv': 'csv',
        '.json': 'json',
        '.xml': 'xml',
        '.xsd': 'xml',
        '.rdf': 'xml',
        '.rss': 'xml',
        '.svg': 'xml',
        '.wsdl': 'xml',
        '.xslt': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml'
    }

    # Binary file signatures (magic numbers)
    BINARY_SIGNATURES = {
        b'%PDF': 'pdf',
        b'PK\x03\x04': 'zip',  # ZIP files (could be docx, xlsx, pptx)
        b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1': 'ms_compound',  # MS Compound File (older doc, xls, ppt)
    }

    @staticmethod
    def detect_from_path(path):
        """
        Detect document type from file path.

        Args:
            path: Path to the file

        Returns:
            String representing document type: 'markdown', 'docx', etc.
        """
        if not path:
            return None

        # Use file extension for detection
        extension = Path(path).suffix.lower()

        # Return matched type or fallback to MIME detection
        return DocumentTypeDetector.EXTENSION_MAP.get(extension, DocumentTypeDetector.detect_from_mime(path))

    @staticmethod
    def detect_from_mime(path):
        """
        Detect document type from MIME type.

        Args:
            path: Path to the file

        Returns:
            String representing document type
        """
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(path)

        # If MIME type detection failed and python-magic is available, try it
        if not mime_type and os.path.exists(path) and MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_file(path, mime=True)
            except Exception as e:
                logger.debug(f"Error detecting MIME type with python-magic: {str(e)}")

        # Return matched type or default to text
        return DocumentTypeDetector.MIME_TYPE_MAP.get(mime_type, 'text')

    @staticmethod
    def detect_from_content(content, metadata=None):
        """
        Detect document type by inspecting content.

        Args:
            content: File content (bytes or string)
            metadata: Optional metadata that might provide hints

        Returns:
            String representing document type
        """
        # Check metadata hints first if provided
        if metadata:
            # Check explicit content type hint
            content_type = metadata.get('content_type')
            if content_type and content_type in DocumentTypeDetector.MIME_TYPE_MAP:
                return DocumentTypeDetector.MIME_TYPE_MAP[content_type]

            # Check column name hint for database content
            content_column = metadata.get('content_column', '')
            if content_column:
                if content_column.endswith('_html'):
                    return 'html'
                elif content_column.endswith(('_md', '_markdown')):
                    return 'markdown'
                elif content_column.endswith('_json'):
                    return 'json'
                elif content_column.endswith('_xml'):
                    return 'xml'
                elif content_column.endswith('_csv'):
                    return 'csv'

        # Ensure we have bytes for binary detection and string for text detection
        # content_bytes = None
        content_str = None

        if isinstance(content, str):
            content_str = content
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content
            try:
                content_str = content.decode('utf-8')
            except UnicodeDecodeError:
                # Content is definitely binary if it can't be decoded as UTF-8
                pass

        # Check binary signatures for binary content
        if content_bytes:
            for signature, doc_type in DocumentTypeDetector.BINARY_SIGNATURES.items():
                if content_bytes.startswith(signature):
                    # Special handling for ZIP-based Office formats
                    if doc_type == 'zip':
                        # Look for Office XML signatures in the first 4000 bytes
                        content_start = content_bytes[:4000]
                        if b'word/' in content_start:
                            return 'docx'
                        elif b'xl/' in content_start:
                            return 'xlsx'
                        elif b'ppt/' in content_start:
                            return 'pptx'
                    # Return the detected binary type
                    return doc_type

        # Use python-magic if available
        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_buffer(content_bytes, mime=True)
                doc_type = DocumentTypeDetector.MIME_TYPE_MAP.get(mime_type)
                if doc_type and doc_type != 'text':
                    return doc_type
            except Exception as e:
                logger.debug(f"Error detecting content type with python-magic: {str(e)}")

        # Fallback to text analysis if we have a string
        if content_str:
            # Check for CSV format
            if DocumentTypeDetector._is_likely_csv(content_str):
                return 'csv'

            # Check for markdown headers
            if re.search(r'^#{1,6}\s+', content_str, re.MULTILINE):
                return 'markdown'

            # Check for HTML
            if re.search(r'<!DOCTYPE html>|<html|<body|<div|<span|<p>', content_str, re.IGNORECASE):
                return 'html'

            # Check for JSON
            if content_str.strip().startswith('{') and content_str.strip().endswith('}'):
                try:
                    json.loads(content_str)
                    return 'json'
                except json.JSONDecodeError:
                    pass

            # Check for XML
            if content_str.strip().startswith('<') and content_str.strip().endswith('>'):
                if re.search(r'<\?xml|<[a-zA-Z]+>[^<>]*</[a-zA-Z]+>', content_str):
                    return 'xml'

            # Default to text for string content
            return 'text'

        # If all detection methods fail, default to binary
        return 'binary'

    @staticmethod
    def _is_likely_csv(text):
        """
        Detect if a text string is likely a CSV file.

        Args:
            text: Text content to check

        Returns:
            Boolean indicating if text is likely CSV format
        """
        # Quick check for empty content
        if not text or not text.strip():
            return False

        # Get first few lines
        lines = text.splitlines()[:5]
        if not lines:
            return False

        # Check if consistent delimiters exist
        potential_delimiters = [',', '\t', ';', '|']

        # Count delimiters in each line
        delimiter_counts = {}
        for delimiter in potential_delimiters:
            counts = [line.count(delimiter) for line in lines]
            # If delimiter appears consistently and at least once per line
            if all(count > 0 for count in counts) and max(counts) - min(counts) <= 1:
                delimiter_counts[delimiter] = sum(counts)

        # If we found consistent delimiters
        if delimiter_counts:
            # Choose the most frequent delimiter
            most_frequent = max(delimiter_counts, key=delimiter_counts.get)
            # Verify most lines have approximately same number of fields
            fields_per_line = [len(line.split(most_frequent)) for line in lines]
            avg_fields = sum(fields_per_line) / len(fields_per_line)
            # Check if field count is consistent (within 1 of average)
            if all(abs(fields - avg_fields) <= 1 for fields in fields_per_line):
                return True

        # Check for fixed-width format (harder to detect)
        # TODO: Add fixed-width detection if needed

        return False

    @staticmethod
    def detect(path=None, content=None, metadata=None):
        """
        Detect document type using all available methods.

        Args:
            path: Optional file path
            content: Optional file content
            metadata: Optional metadata hints

        Returns:
            String representing document type
        """
        # Try path-based detection first
        if path:
            doc_type = DocumentTypeDetector.detect_from_path(path)
            if doc_type:
                return doc_type

        # Then try content-based detection with metadata hints
        if content:
            doc_type = DocumentTypeDetector.detect_from_content(content, metadata)
            if doc_type:
                return doc_type

        # Default to text
        return 'text'
