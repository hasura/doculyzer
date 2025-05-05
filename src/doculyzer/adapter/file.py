"""
File adapter module for the document pointer system.

This module provides an adapter to retrieve content from file system sources.
"""

import logging
import mimetypes
import os
from typing import Dict, Any, Optional

from .base import ContentSourceAdapter
from ..document_parser.document_type_detector import DocumentTypeDetector

logger = logging.getLogger(__name__)


class FileAdapter(ContentSourceAdapter):
    """Adapter for file-based content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the file adapter."""
        super().__init__(config)
        self.base_path = self.config.get("base_path", "")
        self.follow_symlinks = self.config.get("follow_symlinks", True)
        self.encoding_fallbacks = self.config.get(
            "encoding_fallbacks", ['utf-8', 'latin-1', 'cp1252']
        )

    def get_content(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get content from a file.

        Args:
            location_data: Location data with file path

        Returns:
            Dictionary with content and metadata

        Raises:
            ValueError: If file cannot be read
        """
        source = location_data.get("source", "")

        # Ensure file exists
        if not os.path.exists(source):
            raise ValueError(f"File not found: {source}")

        if not os.path.isfile(source):
            raise ValueError(f"Not a file: {source}")

        # Check if we should follow symlinks
        if os.path.islink(source) and not self.follow_symlinks:
            raise ValueError(f"File is a symlink and following symlinks is disabled: {source}")

        # Get file stats for metadata
        file_stats = os.stat(source)

        # Determine if file is binary or text
        is_binary = self._is_binary_file(source)

        # Read file content
        try:
            if is_binary:
                # Read as binary
                with open(source, 'rb') as f:
                    content = f.read()
            else:
                # Try different encodings for text files
                for encoding in self.encoding_fallbacks:
                    try:
                        with open(source, 'r', encoding=encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        if encoding == self.encoding_fallbacks[-1]:
                            # If all encodings fail, read as binary
                            with open(source, 'rb') as f:
                                content = f.read()
                            is_binary = True
                        continue
        except Exception as e:
            raise ValueError(f"Error reading file {source}: {str(e)}")

        # Determine content type
        content_type = DocumentTypeDetector.detect(
            path=source,
            content=content,
            metadata={"binary": is_binary}
        )

        # Get file metadata
        metadata = {
            "filename": os.path.basename(source),
            "path": os.path.abspath(source),
            "size": file_stats.st_size,
            "modified": file_stats.st_mtime,
            "created": file_stats.st_ctime,
            "extension": os.path.splitext(source)[1].lower(),
            "is_binary": is_binary
        }

        # Add mime type if available
        mime_type, encoding = mimetypes.guess_type(source)
        if mime_type:
            metadata["mime_type"] = mime_type
        if encoding:
            metadata["encoding"] = encoding

        return {
            "content": content,
            "content_type": content_type,
            "metadata": metadata
        }

    def supports_location(self, location_data: Dict[str, Any]) -> bool:
        """
        Check if this adapter supports the location.

        Args:
            location_data: Location data

        Returns:
            True if supported, False otherwise
        """
        source = location_data.get("source", "")

        # Check if source is a valid file path
        if not source:
            return False

        # Check if it's a file:// URI
        if source.startswith("file://"):
            # Strip file:// prefix and normalize path
            file_path = source[7:]
            # Handle Windows network paths
            if file_path.startswith("//"):
                return True
            # Handle local paths
            return os.path.exists(file_path)

        # Check if it's an absolute path
        if os.path.isabs(source):
            return os.path.exists(source)

        # Check if it's a relative path from base_path
        if self.base_path:
            combined_path = os.path.join(self.base_path, source)
            return os.path.exists(combined_path)

        # Check if it's a relative path from current directory
        return os.path.exists(source)

    def get_binary_content(self, location_data: Dict[str, Any]) -> bytes:
        """
        Get the file as binary data.

        Args:
            location_data: Location data

        Returns:
            Binary content

        Raises:
            ValueError: If file cannot be read
        """
        source = location_data.get("source", "")

        # Ensure file exists
        if not os.path.exists(source):
            raise ValueError(f"File not found: {source}")

        if not os.path.isfile(source):
            raise ValueError(f"Not a file: {source}")

        # Read file as binary
        try:
            with open(source, 'rb') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error reading file {source}: {str(e)}")

    def get_metadata(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get metadata about the file without retrieving the full content.

        Args:
            location_data: Location data

        Returns:
            Dictionary with metadata

        Raises:
            ValueError: If file metadata cannot be retrieved
        """
        source = location_data.get("source", "")

        # Ensure file exists
        if not os.path.exists(source):
            raise ValueError(f"File not found: {source}")

        if not os.path.isfile(source):
            raise ValueError(f"Not a file: {source}")

        # Get file stats
        file_stats = os.stat(source)

        # Determine if file is binary or text without reading content
        is_binary = self._is_binary_file(source)

        # Create metadata
        metadata = {
            "filename": os.path.basename(source),
            "path": os.path.abspath(source),
            "size": file_stats.st_size,
            "modified": file_stats.st_mtime,
            "created": file_stats.st_ctime,
            "extension": os.path.splitext(source)[1].lower(),
            "is_binary": is_binary
        }

        # Add mime type if available
        mime_type, encoding = mimetypes.guess_type(source)
        if mime_type:
            metadata["mime_type"] = mime_type
        if encoding:
            metadata["encoding"] = encoding

        return metadata

    def resolve_uri(self, uri: str) -> Dict[str, Any]:
        """
        Parse a file URI into location data.

        Args:
            uri: File URI string (file:// or absolute/relative path)

        Returns:
            Dictionary with parsed location data

        Raises:
            ValueError: If URI cannot be parsed
        """
        # Handle file:// URIs
        if uri.startswith("file://"):
            path = uri[7:]
            # Handle Windows network paths
            if os.name == 'nt' and path.startswith("/"):
                path = path.replace("/", "\\", 1)
            # Normalize path
            path = os.path.normpath(path)
        else:
            # Handle absolute or relative paths
            path = uri

        # Resolve relative paths against base_path if provided
        if not os.path.isabs(path) and self.base_path:
            path = os.path.join(self.base_path, path)
            path = os.path.abspath(path)

        return {"source": path}

    @staticmethod
    def _is_binary_file(file_path: str) -> bool:
        """
        Check if a file is binary.

        Args:
            file_path: Path to the file

        Returns:
            True if binary, False if text
        """
        # Check extension first
        extension = os.path.splitext(file_path)[1].lower()
        text_extensions = [
            '.txt', '.md', '.markdown', '.html', '.htm', '.csv',
            '.json', '.xml', '.yaml', '.yml', '.css', '.js', '.py',
            '.java', '.c', '.cpp', '.h', '.cs', '.php', '.rb', '.pl',
            '.sh', '.bat', '.cfg', '.ini', '.conf', '.properties'
        ]

        if extension in text_extensions:
            return False

        binary_extensions = [
            '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt',
            '.zip', '.gz', '.tar', '.rar', '.7z', '.exe', '.dll', '.so',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.mp3',
            '.mp4', '.avi', '.mov', '.wmv', '.flv', '.wav', '.ogg'
        ]

        if extension in binary_extensions:
            return True

        # For unknown extensions, check file content
        try:
            with open(file_path, 'rb') as f:
                # Read first 8KB
                chunk = f.read(8192)

            # Check for null bytes (standard heuristic for binary files)
            if b'\x00' in chunk:
                return True

            # Check for high proportion of non-ASCII characters
            non_ascii = len([b for b in chunk if b > 127])
            if non_ascii > len(chunk) * 0.3:  # More than 30% non-ASCII
                return True

            return False

        except Exception:
            # Default to binary if we can't check
            return True
