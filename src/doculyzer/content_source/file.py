"""
File Content Source for the document pointer system.

This module provides access to documents from the local file system,
supporting various file formats that can be handled by document parsers.
"""

import logging
import mimetypes
import os
import re
from typing import Dict, Any, List, Optional, Set

import wcmatch.glob as glob

from .base import ContentSource
from ..document_parser.factory import get_parser_for_content

logger = logging.getLogger(__name__)


class FileContentSource(ContentSource):
    """Content source for file system documents."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the file content source.

        Args:
            config: Configuration dictionary, which may include:
                - base_path: Base directory for files (default: current directory)
                - file_pattern: Glob pattern for files (default: "**/*")
                - include_extensions: List of file extensions to include (default: all)
                - exclude_extensions: List of file extensions to exclude (default: none)
                - watch_for_changes: Whether to check for file changes (default: True)
                - recursive: Whether to recurse into subdirectories (default: True)
                - link_pattern: Regex pattern for links (default: format-specific)
                - max_link_depth: Maximum depth to follow links (default: 1)
        """
        super().__init__(config)
        self.base_path = os.path.abspath(os.path.expanduser(config.get("base_path", ".")))
        self.file_pattern = config.get("file_pattern", "**/*")
        self.include_extensions = config.get("include_extensions", [])
        self.exclude_extensions = config.get("exclude_extensions", [])
        self.watch_for_changes = config.get("watch_for_changes", True)
        self.recursive = config.get("recursive", True)
        self.max_link_depth = config.get("max_link_depth", 1)
        # Remove the link_pattern param - it's no longer used

        # Patterns for link extraction by file type - these are format facts, not configurable
        self.link_patterns = {
            # Markdown patterns
            "markdown": [
                r'\[\[(.*?)\]\]',  # Wiki-style links [[Page]]
                r'\[([^\]]+)\]\(([^)]+)\)'  # Markdown links [text](url)
            ],
            # HTML patterns
            "html": [
                r'<a\s+href=["\'](.*?)["\'][^>]*>(.*?)</a>'  # HTML links
            ],
            # Plain text (might contain URLs)
            "text": [
                r'https?://[^\s<>)"]+',  # Plain URLs
                r'file://[^\s<>)"]+',  # file:// URLs
            ],
            # XML/SGML formats
            "xml": [
                r'<link\s+.*?href=["\'](.*?)["\'].*?>',  # XML link elements
                r'xlink:href=["\'](.*?)["\']'  # XLink references
            ],
            # For formats without defined patterns, rely on parsers
            "default": []
        }

        # Initialize mimetypes
        mimetypes.init()

        logger.debug(f"Initialized file content source with base path: {self.base_path}")

    def fetch_document(self, source_id: str) -> Dict[str, Any]:
        """
        Fetch document content from file.

        Args:
            source_id: Document identifier (file path, can be absolute or relative to base_path)

        Returns:
            Dictionary containing document content and metadata

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        # Determine if source_id is already a full path or needs to be combined with base_path
        if os.path.isabs(source_id):
            file_path = source_id
        else:
            # Relative path - combine with base path
            file_path = os.path.abspath(os.path.join(self.base_path, source_id))

        logger.debug(f"Fetching document content from file: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file details
        file_stat = os.stat(file_path)
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()[1:]  # Extension without dot

        # Determine document type and read mode based on extension
        doc_type, read_mode = self._get_doc_type_and_mode(file_ext)

        # Read file content
        try:
            if read_mode == 'text':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                binary_path = None
            else:  # Binary mode
                with open(file_path, 'rb') as f:
                    binary_content = f.read()

                # For binary files, content will be empty and binary_path will be the file path
                content = ""
                binary_path = file_path

            logger.debug(f"Successfully read file {file_path} (size: {file_stat.st_size} bytes)")

            # Create basic metadata
            metadata = {
                "filename": file_name,
                "last_modified": file_stat.st_mtime,
                "size": file_stat.st_size,
                "full_path": file_path,
                "relative_path": os.path.relpath(file_path, self.base_path) if self.base_path else file_path,
                "extension": file_ext,
                "content_type": mimetypes.guess_type(file_path)[0] or f"application/{file_ext}"
            }

            # Generate content hash for change detection
            if read_mode == 'text':
                content_hash = self.get_content_hash(content)
            else:
                # For binary files, use a hash of the binary content
                content_hash = self.get_content_hash(str(binary_content))

            # Create response dictionary
            result = {
                "id": file_path,  # Use full path as ID
                "content": content if read_mode == 'text' else "",
                "metadata": metadata,
                "content_hash": content_hash,
                "doc_type": doc_type
            }

            # Add binary_path for binary documents
            if binary_path:
                result["binary_path"] = binary_path

            return result

        except Exception as e:
            logger.error(f"Error fetching document {source_id}: {str(e)}")
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List available documents in the file system.

        Returns:
            List of document identifiers and metadata
        """
        results = []

        # Determine glob pattern with recursion based on config
        if self.recursive:
            pattern = os.path.join(self.base_path, self.file_pattern)
            logger.debug(f"Searching for files with recursive pattern: {pattern}")
            flags = glob.BRACE | glob.GLOBSTAR
            files = glob.glob(pattern, flags=flags)
        else:
            pattern = os.path.join(self.base_path, self.file_pattern)
            flags = glob.BRACE
            logger.debug(f"Searching for files with non-recursive pattern: {pattern}")
            files = glob.glob(pattern, flags=flags)

        logger.debug(f"Found {len(files)} files initially")

        # Filter files based on extensions
        filtered_files = []
        for file_path in files:
            # Skip directories
            if os.path.isdir(file_path):
                continue

            file_ext = os.path.splitext(file_path)[1].lower()[1:]  # Extension without dot

            # Skip files with excluded extensions
            if self.exclude_extensions and file_ext in self.exclude_extensions:
                logger.debug(f"Skipping excluded extension: {file_path}")
                continue

            # Skip files not matching included extensions (if specified)
            if self.include_extensions and file_ext not in self.include_extensions:
                logger.debug(f"Skipping non-included extension: {file_path}")
                continue

            filtered_files.append(file_path)

        logger.debug(f"Found {len(filtered_files)} files after filtering")

        # Process filtered files
        for file_path in filtered_files:
            # Use absolute file path as source_id
            abs_path = os.path.abspath(file_path)
            rel_path = os.path.relpath(file_path, self.base_path)
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()[1:]

            # Get basic file stats
            file_stat = os.stat(file_path)

            # Determine doc type based on extension
            doc_type, _ = self._get_doc_type_and_mode(file_ext)

            # Create metadata
            metadata = {
                "filename": file_name,
                "last_modified": file_stat.st_mtime,
                "size": file_stat.st_size,
                "full_path": abs_path,
                "relative_path": rel_path,
                "extension": file_ext,
                "content_type": mimetypes.guess_type(file_path)[0] or f"application/{file_ext}"
            }

            results.append({
                "id": abs_path,  # Use full path as ID
                "metadata": metadata,
                "doc_type": doc_type
            })

        return results

    def has_changed(self, source_id: str, last_modified: Optional[float] = None) -> bool:
        """
        Check if file has changed based on modification time.

        Args:
            source_id: Identifier for the document (file path)
            last_modified: Timestamp of last known modification

        Returns:
            True if file has changed, False otherwise
        """
        # If watching for changes is disabled, always return True
        if not self.watch_for_changes:
            return True

        # source_id should already be an absolute path
        file_path = source_id
        logger.debug(f"Checking if file has changed: {file_path}")

        if not os.path.exists(file_path):
            logger.debug(f"File no longer exists: {file_path}")
            return False

        current_mtime = os.stat(file_path).st_mtime

        if last_modified is None:
            logger.debug(f"No previous modification time available for {file_path}")
            return True

        changed = current_mtime > last_modified
        logger.debug(f"File changed: {changed} - {file_path} (current: {current_mtime}, previous: {last_modified})")
        return changed

    def follow_links(self, content: str, source_id: str, current_depth: int = 0,
                     global_visited_docs: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract and follow links in file content with global visited tracking.

        Args:
            content: Document content
            source_id: Identifier for the source document (file path)
            current_depth: Current depth of link following
            global_visited_docs: Global set of all visited document IDs

        Returns:
            List of linked documents
        """
        if current_depth >= self.max_link_depth:
            logger.debug(f"Max link depth {self.max_link_depth} reached for {source_id}")
            return []

        # Initialize global visited set if not provided
        if global_visited_docs is None:
            global_visited_docs = set()

        # Add current document to global visited set
        global_visited_docs.add(source_id)

        logger.debug(f"Following links in file {source_id} at depth {current_depth}")

        # Get file extension to determine link extraction method
        file_ext = os.path.splitext(source_id)[1].lower()[1:]
        doc_type, _ = self._get_doc_type_and_mode(file_ext)

        # For binary files, we can't extract links from the content directly
        if not content and os.path.exists(source_id):
            # Try to use a parser to extract links
            doc_content = self.fetch_document(source_id)

            if "binary_path" in doc_content:
                # This is a binary document, try to use a parser
                try:
                    parser = get_parser_for_content(doc_content)
                    parsed_doc = parser.parse(doc_content)
                    links = parsed_doc.get('links', [])

                    return self._process_extracted_links(
                        links, source_id, current_depth, global_visited_docs
                    )
                except Exception as e:
                    logger.warning(f"Error parsing binary document for links: {str(e)}")
                    return []

            return []

        # Extract links based on document type
        links = self._extract_links_from_content(content, source_id, doc_type)
        return self._process_extracted_links(links, source_id, current_depth, global_visited_docs)

    def _extract_links_from_content(self, content: str, _source_id: str, doc_type: str) -> List[Dict[str, Any]]:
        """
        Extract links from document content based on document type.

        Args:
            content: Document content
            _source_id: Source document ID
            doc_type: Document type

        Returns:
            List of extracted link dictionaries
        """
        # Choose patterns based on document type
        if doc_type in self.link_patterns:
            patterns = self.link_patterns[doc_type]
        else:
            patterns = self.link_patterns["default"]

        links = []

        # Extract using regex patterns
        for pattern in patterns:
            matches = re.findall(pattern, content)

            for match in matches:
                if isinstance(match, tuple):
                    # Pattern with multiple capture groups, e.g., [text](url)
                    if len(match) >= 2:
                        link_text = match[0]
                        link_target = match[1]
                    else:
                        link_text = match[0]
                        link_target = match[0]
                else:
                    # Single capture group, e.g., [[page]]
                    link_text = match
                    link_target = match

                links.append({
                    "source_id": None,  # Will be filled in by the calling function
                    "link_text": link_text,
                    "link_target": link_target,
                    "link_type": doc_type
                })

        return links

    def _process_extracted_links(self, links: List[Dict[str, Any]], source_id: str,
                                 current_depth: int, global_visited_docs: Set[str]) -> List[Dict[str, Any]]:
        """
        Process and follow extracted links.

        Args:
            links: Extracted links
            source_id: Source document ID
            current_depth: Current depth
            global_visited_docs: Set of visited documents

        Returns:
            List of linked documents
        """
        # Source directory for resolving relative links
        base_dir = os.path.dirname(source_id)
        logger.debug(f"Base directory for resolving relative links: {base_dir}")

        linked_docs = []
        link_counter = 0

        for link in links:
            link_counter += 1
            link_target = link.get("link_target", "")

            # Skip empty links
            if not link_target:
                continue

            # Skip external links (http/https)
            if link_target.startswith(('http://', 'https://')):
                logger.debug(f"Skipping external link ({link_counter}/{len(links)}): {link_target}")
                continue

            # Resolve relative path to absolute path
            if not os.path.isabs(link_target):
                resolved_path = os.path.abspath(os.path.join(base_dir, link_target))
            else:
                resolved_path = link_target

            # Skip if the file doesn't exist
            if not os.path.exists(resolved_path):
                logger.debug(f"Skipping non-existent file ({link_counter}/{len(links)}): {resolved_path}")
                continue

            # Skip if globally visited
            if resolved_path in global_visited_docs:
                logger.debug(f"Skipping globally visited file ({link_counter}/{len(links)}): {resolved_path}")
                continue

            global_visited_docs.add(resolved_path)
            logger.debug(f"Following link {link_counter}/{len(links)}: {resolved_path}")

            try:
                # Fetch the linked document using the absolute path
                linked_doc = self.fetch_document(resolved_path)
                linked_docs.append(linked_doc)
                logger.debug(f"Successfully fetched linked document: {resolved_path}")

                # Recursively follow links if not at max depth
                if current_depth + 1 < self.max_link_depth:
                    logger.debug(f"Recursively following links from {resolved_path} at depth {current_depth + 1}")
                    content = linked_doc.get("content", "")
                    nested_docs = self.follow_links(
                        content,
                        resolved_path,
                        current_depth + 1,
                        global_visited_docs
                    )
                    logger.debug(f"Found {len(nested_docs)} nested documents from {resolved_path}")
                    linked_docs.extend(nested_docs)
            except Exception as e:
                logger.warning(f"Error following link {link_target} from {source_id}: {str(e)}")

        logger.debug(f"Completed following links from {source_id}: found {len(linked_docs)} linked documents")
        return linked_docs

    @staticmethod
    def _get_doc_type_and_mode(extension: str) -> tuple:
        """
        Determine document type and read mode based on file extension.

        Args:
            extension: File extension (without dot)

        Returns:
            Tuple of (doc_type, read_mode)
        """
        # Map extensions to document types and read modes
        extension_map = {
            # Text-based formats
            "md": ("markdown", "text"),
            "markdown": ("markdown", "text"),
            "txt": ("text", "text"),
            "html": ("html", "text"),
            "htm": ("html", "text"),
            "css": ("text", "text"),
            "js": ("text", "text"),
            "json": ("text", "text"),
            "xml": ("text", "text"),
            "yaml": ("text", "text"),
            "yml": ("text", "text"),

            # Binary formats
            "pdf": ("pdf", "binary"),
            "docx": ("docx", "binary"),
            "doc": ("doc", "binary"),
            "pptx": ("pptx", "binary"),
            "ppt": ("ppt", "binary"),
            "xlsx": ("xlsx", "binary"),
            "xls": ("xls", "binary"),
            "png": ("image", "binary"),
            "jpg": ("image", "binary"),
            "jpeg": ("image", "binary"),
            "gif": ("image", "binary"),
            "zip": ("binary", "binary"),
        }

        # Get document type and read mode from map, default to text
        return extension_map.get(extension.lower(), ("text", "text"))
