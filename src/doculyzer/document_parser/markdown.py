"""
Markdown document parser module with caching strategies for the document pointer system.

This module parses Markdown documents into structured elements and provides
semantic textual representations of the data with improved performance.
"""

import functools
import hashlib
import json
import logging
import os
import re
import uuid
from typing import Dict, List, Any, Tuple, Optional, Union

import markdown
import time
import yaml
from bs4 import BeautifulSoup

from .base import DocumentParser
from .lru_cache import LRUCache, ttl_cache
from ..relationships import RelationshipType
from ..storage import ElementType

logger = logging.getLogger(__name__)


class MarkdownParser(DocumentParser):
    """Parser for Markdown documents with caching for improved performance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Markdown parser with caching capabilities."""
        super().__init__(config)
        # Configuration options
        self.config = config or {}
        self.extract_front_matter = self.config.get("extract_front_matter", True)
        self.paragraph_threshold = self.config.get("paragraph_threshold", 1)  # Min lines to consider a paragraph
        self.max_content_preview = self.config.get("max_content_preview", 100)

        # Define Markdown-specific link patterns
        self.link_patterns = [
            r'\[\[(.*?)\]\]',  # Wiki-style links [[Page]]
            r'\[([^\]]+)\]\(([^)]+)\)'  # Markdown links [text](url)
        ]

        # Cache configurations
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # Default 1 hour TTL
        self.max_cache_size = self.config.get("max_cache_size", 128)  # Default max cache size
        self.enable_caching = self.config.get("enable_caching", True)

        # Performance monitoring
        self.enable_performance_monitoring = self.config.get("enable_performance_monitoring", False)
        self.performance_stats = {
            "parse_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_parse_time": 0.0,
            "total_element_processing_time": 0.0,
            "total_link_extraction_time": 0.0,
            "method_times": {}
        }

        # Initialize caches
        self.document_cache = LRUCache(max_size=self.max_cache_size, ttl=self.cache_ttl)
        self.html_cache = LRUCache(max_size=min(50, self.max_cache_size), ttl=self.cache_ttl)  # For converted HTML
        self.text_cache = LRUCache(max_size=self.max_cache_size * 2, ttl=self.cache_ttl)

    def _generate_hash(self, content):
        """Generate a hash for content, always returning a string."""
        try:
            if isinstance(content, str):
                return hashlib.md5(content.encode('utf-8')).hexdigest()
            elif isinstance(content, bytes):
                return hashlib.md5(content).hexdigest()
            else:
                # Convert any other type to string first
                return hashlib.md5(str(content).encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f"Error generating hash: {str(e)}")
            # Return a UUID as string for fallback
            return str(uuid.uuid4())

    def _generate_id(self, prefix: str = "") -> str:
        """
        Generate a unique ID with an optional prefix.

        Args:
            prefix: Optional ID prefix

        Returns:
            Unique ID string
        """
        return f"{prefix}{uuid.uuid4()}"

    def _create_root_element(self, doc_id: str, source_id: str) -> Dict[str, Any]:
        """
        Create a document root element.

        Args:
            doc_id: Document ID
            source_id: Source identifier

        Returns:
            Document root element dictionary
        """
        root_id = self._generate_id("doc_root_")
        root_element = {
            "element_id": root_id,
            "doc_id": doc_id,
            "element_type": "document_root",
            "parent_id": None,
            "content_preview": f"Document: {source_id}",
            "content_location": json.dumps({
                "source": source_id,
                "type": "root"
            }),
            "content_hash": self._generate_hash(source_id),
            "metadata": {
                "source_id": source_id,
                "path": "/"
            }
        }
        return root_element

    @staticmethod
    def _load_source_content(source_path: str) -> Tuple[Union[str, bytes], Optional[str]]:
        """
        Load content from a source file with proper error handling.

        Args:
            source_path: Path to the source file

        Returns:
            Tuple of (content, error_message)
            - content: The file content as string or bytes
            - error_message: Error message if loading failed, None otherwise
        """
        if not os.path.exists(source_path):
            error_msg = f"Error: Source file not found: {source_path}"
            logger.error(error_msg)
            return None, error_msg

        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    with open(source_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        return content, None
                except UnicodeDecodeError:
                    if encoding == 'cp1252':  # Last attempt
                        raise
                    continue
        except Exception as e:
            error_msg = f"Error: Cannot read content from {source_path}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def _resolve_element_text(self, location_data: Dict[str, Any], source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve the plain text representation of a Markdown element.

        Args:
            location_data: Content location data
            source_content: Optional preloaded source content

        Returns:
            Plain text representation of the element
        """
        source = location_data.get("source", "")
        element_type = location_data.get("type", "")
        element_id = location_data.get("element_id", "")

        # Generate cache key
        cache_key = f"text_{source}_{element_type}_{element_id}"
        if self.enable_caching:
            cached_text = self.text_cache.get(cache_key)
            if cached_text is not None:
                logger.debug(f"Cache hit for element text: {cache_key}")
                if self.enable_performance_monitoring:
                    self.performance_stats["cache_hits"] += 1
                return cached_text

        logger.debug(f"Cache miss for element text: {cache_key}")
        if self.enable_performance_monitoring:
            self.performance_stats["cache_misses"] += 1

        # First, get the content of the element using the content resolver
        raw_content = self._resolve_element_content(location_data, source_content)
        if not raw_content:
            return ""

        result = ""

        # Handle specific element types
        if element_type == ElementType.HEADER.value:
            # For headers, remove markdown formatting (#) and return text
            result = re.sub(r'^#+\s*', '', raw_content).strip()

        elif element_type == ElementType.PARAGRAPH.value:
            # For paragraphs, just return the text (already clean in Markdown)
            result = raw_content.strip()

        elif element_type == ElementType.LIST.value:
            # For lists, format each item on a new line with appropriate marker
            list_type = location_data.get("list_type", "unordered")

            if list_type == "ordered":
                # Extract items with their numbers
                items = re.findall(r'^\s*(\d+\.\s+.*)$', raw_content, re.MULTILINE)
            else:
                # Extract unordered items and normalize markers
                items = re.findall(r'^\s*[\*\-\+]\s+(.*)$', raw_content, re.MULTILINE)
                items = [f"• {item.strip()}" for item in items]

            result = "\n".join(items)

        elif element_type == ElementType.LIST_ITEM.value:
            # For a single list item, remove the marker and return text
            result = re.sub(r'^\s*[\*\-\+\d+\.]\s+', '', raw_content).strip()

        elif element_type in [
            ElementType.TABLE.value,
            ElementType.TABLE_ROW.value,
            ElementType.TABLE_CELL.value,
            ElementType.TABLE_HEADER.value]:
            if element_type == ElementType.TABLE.value:
                # For a table, convert to a readable text format
                rows = raw_content.strip().split('\n')

                # Remove separator row if present (contains only dashes and pipes)
                rows = [r for r in rows if not re.match(r'^\s*\|[\s\-\|]+\|\s*$', r)]

                # Process each row
                result_rows = []
                for row in rows:
                    # Split by pipe and clean cells
                    cells = re.findall(r'\|(.*?)(?=\||$)', row)
                    cells = [cell.strip() for cell in cells if cell.strip()]

                    # Join cells with proper spacing
                    result_rows.append(" | ".join(cells))

                result = "\n".join(result_rows)

            elif element_type == ElementType.TABLE_ROW.value:
                # For a table row, return cells separated by |
                cells = re.findall(r'\|(.*?)(?=\||$)', raw_content)
                cells = [cell.strip() for cell in cells if cell.strip()]
                result = " | ".join(cells)

            elif element_type in [ElementType.TABLE_CELL.value, ElementType.TABLE_HEADER.value]:
                # For a single cell, just return the text
                result = raw_content.strip().strip('|')

        elif element_type == ElementType.CODE_BLOCK.value:
            # For code blocks, remove backticks and language identifier
            cleaned = re.sub(r'^```\w*\s*\n', '', raw_content)
            cleaned = re.sub(r'\n```$', '', cleaned)

            language = location_data.get("language", "")
            if language:
                result = f"Code ({language}):\n{cleaned}"
            else:
                result = f"Code:\n{cleaned}"

        elif element_type == ElementType.BLOCKQUOTE.value:
            # For blockquotes, remove > markers
            lines = raw_content.split('\n')
            cleaned_lines = [re.sub(r'^\s*>\s?', '', line) for line in lines]
            result = "\n".join(cleaned_lines).strip()
        else:
            # Default case: return content as is (already text in Markdown)
            result = raw_content.strip()

        # Cache the result before returning
        if self.enable_caching:
            self.text_cache.set(cache_key, result)

        return result

    def _resolve_element_content(self, location_data: Dict[str, Any],
                                 source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve content for specific markdown element types.

        Args:
            location_data: Content location data
            source_content: Optional preloaded source content

        Returns:
            Resolved content string
        """
        source = location_data.get("source", "")
        element_type = location_data.get("type", "")
        element_id = location_data.get("element_id", "")

        # Generate cache key
        if self.enable_caching:
            cache_key = f"content_{source}_{element_type}_{element_id}"
            cached_content = self.text_cache.get(cache_key)
            if cached_content is not None:
                logger.debug(f"Content cache hit for {cache_key}")
                if self.enable_performance_monitoring:
                    self.performance_stats["cache_hits"] += 1
                return cached_content

        logger.debug(f"Content cache miss for {element_id}")
        if self.enable_performance_monitoring:
            self.performance_stats["cache_misses"] += 1

        # Load content if not provided
        content = source_content
        if content is None:
            content, error = self._load_source_content(source)
            if error:
                logger.error(f"Error loading content: {error}")
                return ""

        # Ensure content is string (not bytes)
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                logger.error("Cannot decode binary content as markdown")
                return ""

        # Extract front matter if enabled
        if self.extract_front_matter:
            content, _ = self._extract_front_matter(content)

        result = ""
        # Resolve based on element type
        if element_type == ElementType.HEADER.value:
            # Extract header by text or level
            header_text = location_data.get("text", "")
            header_level = location_data.get("level")

            result = self._extract_header(content, header_text, header_level)

        elif element_type == ElementType.PARAGRAPH.value:
            # Extract paragraph by text
            para_text = location_data.get("text", "")
            result = self._extract_paragraph(content, para_text)

        elif element_type in (ElementType.LIST.value, ElementType.LIST_ITEM.value):
            # Extract list or list item
            index = location_data.get("index", 0)
            list_type = location_data.get("list_type", "")
            result = self._extract_list_item(content, index, list_type)

        elif element_type == ElementType.CODE_BLOCK.value:
            # Extract code block
            language = location_data.get("language", "")
            result = self._extract_code_block(content, language)

        elif element_type == ElementType.BLOCKQUOTE.value:
            # Extract blockquote
            result = self._extract_blockquote(content)

        elif element_type in [ElementType.TABLE_CELL.value, ElementType.TABLE_HEADER.value]:
            # Extract table element
            row = location_data.get("row", 0)
            col = location_data.get("col", 0)
            result = self._extract_table_element(content, element_type, row, col)

        else:
            # Unknown element type, return full content
            result = content

        # Cache the result
        if self.enable_caching:
            self.text_cache.set(cache_key, result)

        return result

    @staticmethod
    def _extract_front_matter(content: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract YAML front matter from Markdown content.

        Args:
            content: Markdown content

        Returns:
            Tuple of (content without front matter, front matter dict)
        """
        front_matter = {}
        content_without_front_matter = content

        # Check for YAML front matter (---\n...\n---)
        front_matter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if front_matter_match:
            try:
                front_matter = yaml.safe_load(front_matter_match.group(1))
                if front_matter and isinstance(front_matter, dict):
                    content_without_front_matter = content[front_matter_match.end():]
            except Exception as e:
                logger.warning(f"Error parsing front matter: {str(e)}")

        return content_without_front_matter, front_matter

    def _extract_document_metadata(self, content: Union[str, bytes], base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from Markdown document.

        Args:
            content: Markdown content
            base_metadata: Base metadata from content source

        Returns:
            Dictionary of document metadata
        """
        # Generate cache key
        if self.enable_caching:
            content_hash = self._generate_hash(content)
            cache_key = f"metadata_{content_hash}"

            # Check metadata cache
            cached_metadata = self.text_cache.get(cache_key)
            if cached_metadata is not None:
                logger.debug(f"Metadata cache hit for {cache_key}")
                if self.enable_performance_monitoring:
                    self.performance_stats["cache_hits"] += 1
                return cached_metadata

        logger.debug(f"Metadata cache miss")
        if self.enable_performance_monitoring:
            self.performance_stats["cache_misses"] += 1

        metadata = base_metadata.copy()

        try:
            # Ensure content is string
            if isinstance(content, bytes):
                try:
                    content_str = content.decode('utf-8')
                except UnicodeDecodeError:
                    logger.warning("Cannot decode binary content as markdown for metadata extraction")
                    return metadata
            else:
                content_str = content

            # Extract front matter if enabled
            if self.extract_front_matter:
                content_without_fm, front_matter = self._extract_front_matter(content_str)
                if front_matter:
                    metadata["front_matter"] = front_matter
                content_str = content_without_fm

            # Extract headers and structure
            headers = re.findall(r'^(#+)\s+(.+)$', content_str, re.MULTILINE)
            if headers:
                metadata["header_count"] = len(headers)

                # Get title (first h1)
                h1_headers = [h[1] for h in headers if len(h[0]) == 1]
                if h1_headers:
                    metadata["title"] = h1_headers[0]

            # Extract basic document statistics
            metadata["paragraph_count"] = len(re.findall(r'\n\s*\n', content_str))
            metadata["code_block_count"] = len(re.findall(r'```\w*\n[\s\S]*?\n```', content_str))
            metadata["list_count"] = len(
                re.findall(r'(?:^\s*[\*\-\+]\s+.*(?:\n\s*[\*\-\+]\s+.*)*)|(?:^\s*\d+\.\s+.*(?:\n\s*\d+\.\s+.*)*)',
                           content_str, re.MULTILINE))

            # Extract links
            md_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content_str)
            wiki_links = re.findall(r'\[\[(.*?)\]\]', content_str)

            if md_links or wiki_links:
                metadata["link_count"] = len(md_links) + len(wiki_links)

                # Store unique link targets
                link_targets = [link[1] for link in md_links]
                link_targets.extend(wiki_links)
                unique_targets = set(link_targets)

                if unique_targets:
                    metadata["link_targets"] = list(unique_targets)

            # Extract languages used in code blocks
            code_langs = re.findall(r'```(\w+)', content_str)
            if code_langs:
                langs = [lang for lang in code_langs if lang]
                if langs:
                    metadata["code_languages"] = list(set(langs))

        except Exception as e:
            logger.warning(f"Error extracting document metadata: {str(e)}")

        # Cache the metadata
        if self.enable_caching:
            self.text_cache.set(cache_key, metadata)

        return metadata

    def _get_or_create_html(self, content: str) -> str:
        """
        Get cached HTML or convert Markdown to HTML if not cached.

        Args:
            content: Markdown content

        Returns:
            HTML content
        """
        # Generate a key for the HTML cache
        html_cache_key = self._generate_hash(content)

        # Try to get cached HTML
        html_content = None
        if self.enable_caching and html_cache_key:
            html_content = self.html_cache.get(html_cache_key)
            if html_content is not None:
                if self.enable_performance_monitoring:
                    self.performance_stats["cache_hits"] += 1
                logger.debug("HTML cache hit")
                return html_content

        if self.enable_performance_monitoring:
            self.performance_stats["cache_misses"] += 1

        # Convert markdown to HTML for easier parsing if not cached
        start_time = time.time()
        html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])

        # Wrap in complete HTML structure if needed
        if not html_content.startswith('<!DOCTYPE html>') and not html_content.startswith('<html'):
            html_content = f"<html><body>{html_content}</body></html>"

        parse_time = time.time() - start_time
        logger.debug(f"Markdown to HTML conversion time: {parse_time:.4f} seconds")

        # Cache the HTML
        if self.enable_caching and html_cache_key:
            self.html_cache.set(html_cache_key, html_content)

        return html_content

    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a Markdown document into structured elements with caching."""
        content = doc_content["content"]
        source_id = doc_content["id"]  # Should already be a fully qualified path
        metadata = doc_content.get("metadata", {}).copy()  # Make a copy to avoid modifying original

        # Generate document cache key
        if self.enable_caching:
            content_hash = self._generate_hash(content)
            doc_cache_key = f"{source_id}_{content_hash}"

            # Check document cache
            cached_doc = self.document_cache.get(doc_cache_key)
            if cached_doc is not None:
                logger.info(f"Document cache hit for {source_id}")
                if self.enable_performance_monitoring:
                    self.performance_stats["cache_hits"] += 1
                return cached_doc

        logger.info(f"Document cache miss for {source_id}")
        if self.enable_performance_monitoring:
            self.performance_stats["cache_misses"] += 1
            self.performance_stats["parse_count"] += 1
            start_time = time.time()

        # Make sure source_id is an absolute path if it's a file
        if os.path.exists(source_id):
            source_id = os.path.abspath(source_id)

        # Generate document ID if not present
        doc_id = metadata.get("doc_id", self._generate_id("doc_"))

        # Extract front matter if enabled
        if self.extract_front_matter:
            content, front_matter = self._extract_front_matter(content)
            metadata.update(front_matter)

        # Create document record
        document = {
            "doc_id": doc_id,
            "doc_type": "markdown",
            "source": source_id,  # This is now a fully qualified path
            "metadata": self._extract_document_metadata(content, metadata),
            "content_hash": doc_content.get("content_hash", self._generate_hash(content))
        }

        # Create root element
        elements = [self._create_root_element(doc_id, source_id)]
        root_id = elements[0]["element_id"]

        # Initialize relationships list
        relationships = []

        # Extract links directly from Markdown content
        start_link_time = time.time()
        direct_links = self._extract_markdown_links(content, root_id)
        if self.enable_performance_monitoring:
            self.performance_stats["total_link_extraction_time"] += time.time() - start_link_time

        # Get or convert HTML content
        html_content = self._get_or_create_html(content)

        # Parse HTML to extract elements and create relationships
        start_element_time = time.time()
        html_elements, html_links, element_relationships = self._parse_html_elements(html_content, doc_id, root_id,
                                                                                     source_id)
        if self.enable_performance_monitoring:
            self.performance_stats["total_element_processing_time"] += time.time() - start_element_time

        elements.extend(html_elements)
        relationships.extend(element_relationships)

        # Combine links from both sources
        extracted_links = direct_links + html_links

        # Prepare result
        result = {
            "document": document,
            "elements": elements,
            "links": extracted_links,
            "relationships": relationships
        }

        # Add performance metrics if enabled
        if self.enable_performance_monitoring:
            total_time = time.time() - start_time
            self.performance_stats["total_parse_time"] += total_time
            result["performance"] = self.get_performance_stats()

        # Cache the document
        if self.enable_caching:
            self.document_cache.set(doc_cache_key, result)

        return result

    def _extract_markdown_links(self, content: str, element_id: str) -> List[Dict[str, Any]]:
        """
        Extract links directly from Markdown content.

        Args:
            content: Markdown content
            element_id: ID of the element containing the links

        Returns:
            List of extracted link dictionaries
        """
        # Generate cache key
        if self.enable_caching:
            cache_key = f"links_{element_id}_{self._generate_hash(content)}"
            cached_links = self.text_cache.get(cache_key)
            if cached_links is not None:
                logger.debug(f"Links cache hit for {cache_key}")
                if self.enable_performance_monitoring:
                    self.performance_stats["cache_hits"] += 1
                return cached_links

        logger.debug(f"Links cache miss for {element_id}")
        if self.enable_performance_monitoring:
            self.performance_stats["cache_misses"] += 1

        links = []

        for pattern in self.link_patterns:
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
                    "source_id": element_id,
                    "link_text": link_text,
                    "link_target": link_target,
                    "link_type": "markdown"
                })

        # Cache the links
        if self.enable_caching:
            self.text_cache.set(cache_key, links)

        return links

    def _parse_html_elements(self, html_content: str, doc_id: str, root_id: str, source_id: str) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Parse HTML content into structured elements and create relationships.

        Args:
            html_content: HTML content converted from Markdown
            doc_id: Document ID
            root_id: Root element ID
            source_id: Source identifier (fully qualified path)

        Returns:
            Tuple of (list of elements, list of links, list of relationships)
        """
        # Generate cache key
        if self.enable_caching:
            cache_key = f"html_elements_{doc_id}_{self._generate_hash(html_content)}"
            cached_result = self.document_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"HTML elements cache hit for {cache_key}")
                if self.enable_performance_monitoring:
                    self.performance_stats["cache_hits"] += 1
                return cached_result

        logger.debug(f"HTML elements cache miss for {doc_id}")
        if self.enable_performance_monitoring:
            self.performance_stats["cache_misses"] += 1

        elements = []
        links = []
        relationships = []

        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Keep track of current parent and section level
        current_parent = root_id
        section_stack = [{"id": root_id, "level": 0}]

        # Process each element in order
        for tag in soup.body.children if soup.body else []:
            # Skip empty elements
            if tag.name is None:
                continue

            # Process element based on type
            if tag.name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                # Header element
                level = int(tag.name[1])

                # Find the appropriate parent based on header level
                while section_stack[-1]["level"] >= level:
                    section_stack.pop()

                current_parent = section_stack[-1]["id"]

                # Create header element
                element_id = self._generate_id(f"header{level}_")
                header_text = tag.get_text().strip()

                header_element = {
                    "element_id": element_id,
                    "doc_id": doc_id,
                    "element_type": ElementType.HEADER.value,
                    "parent_id": current_parent,
                    "content_preview": header_text[:self.max_content_preview] + (
                        "..." if len(header_text) > self.max_content_preview else ""),
                    "content_location": json.dumps({
                        "source": source_id,  # Now using fully qualified path
                        "type": ElementType.HEADER.value,
                        "text": header_text,
                        "level": level,
                        "element_id": element_id
                    }),
                    "content_hash": self._generate_hash(header_text),
                    "metadata": {
                        "level": level,
                        "text": header_text,
                        "full_path": source_id  # Store the full path in metadata
                    }
                }

                elements.append(header_element)

                # Create relationship from parent to header
                contains_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": current_parent,
                    "target_id": element_id,
                    "relationship_type": RelationshipType.CONTAINS.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contains_relationship)

                # Create inverse relationship from header to parent
                contained_by_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": element_id,
                    "target_id": current_parent,
                    "relationship_type": RelationshipType.CONTAINED_BY.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contained_by_relationship)

                # Update section stack
                section_stack.append({"id": element_id, "level": level})
                current_parent = element_id

                # Extract links from header
                for a in tag.find_all('a', href=True):
                    links.append({
                        "source_id": element_id,
                        "link_text": a.get_text().strip(),
                        "link_target": a['href'],
                        "link_type": "html"
                    })

            elif tag.name == 'p':
                # Paragraph element
                para_text = tag.get_text().strip()

                # Skip if too short
                if para_text.count('\n') < self.paragraph_threshold and len(para_text) < 10:
                    continue

                element_id = self._generate_id("para_")

                para_element = {
                    "element_id": element_id,
                    "doc_id": doc_id,
                    "element_type": ElementType.PARAGRAPH.value,
                    "parent_id": current_parent,
                    "content_preview": para_text[:self.max_content_preview] + (
                        "..." if len(para_text) > self.max_content_preview else ""),
                    "content_location": json.dumps({
                        "source": source_id,  # Now using fully qualified path
                        "type": ElementType.PARAGRAPH.value,
                        "text": para_text[:20],  # Enough to identify but not full content
                        "element_id": element_id
                    }),
                    "content_hash": self._generate_hash(para_text),
                    "metadata": {
                        "length": len(para_text),
                        "full_path": source_id  # Store the full path in metadata
                    }
                }

                elements.append(para_element)

                # Create relationship from parent to paragraph
                contains_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": current_parent,
                    "target_id": element_id,
                    "relationship_type": RelationshipType.CONTAINS_TEXT.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contains_relationship)

                # Create inverse relationship
                contained_by_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": element_id,
                    "target_id": current_parent,
                    "relationship_type": RelationshipType.CONTAINED_BY.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contained_by_relationship)

                # Extract links from paragraph
                for a in tag.find_all('a', href=True):
                    links.append({
                        "source_id": element_id,
                        "link_text": a.get_text().strip(),
                        "link_target": a['href'],
                        "link_type": "html"
                    })

            elif tag.name == 'ul' or tag.name == 'ol':
                # List element
                list_id = self._generate_id("list_")
                list_type = 'ordered' if tag.name == 'ol' else 'unordered'

                list_element = {
                    "element_id": list_id,
                    "doc_id": doc_id,
                    "element_type": ElementType.LIST.value,
                    "parent_id": current_parent,
                    "content_preview": f"{list_type.capitalize()} list",
                    "content_location": json.dumps({
                        "source": source_id,  # Now using fully qualified path
                        "type": ElementType.LIST.value,
                        "list_type": list_type,
                        "element_id": list_id
                    }),
                    "content_hash": self._generate_hash(tag.get_text()),
                    "metadata": {
                        "list_type": list_type,
                        "full_path": source_id  # Store the full path in metadata
                    }
                }

                elements.append(list_element)

                # Create relationship from parent to list
                contains_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": current_parent,
                    "target_id": list_id,
                    "relationship_type": RelationshipType.CONTAINS.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contains_relationship)

                # Create inverse relationship
                contained_by_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": list_id,
                    "target_id": current_parent,
                    "relationship_type": RelationshipType.CONTAINED_BY.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contained_by_relationship)

                # Process list items
                for i, item in enumerate(tag.find_all('li', recursive=False)):
                    item_text = item.get_text().strip()
                    item_id = self._generate_id("item_")

                    item_element = {
                        "element_id": item_id,
                        "doc_id": doc_id,
                        "element_type": ElementType.LIST_ITEM.value,
                        "parent_id": list_id,
                        "content_preview": item_text[:self.max_content_preview] + (
                            "..." if len(item_text) > self.max_content_preview else ""),
                        "content_location": json.dumps({
                            "source": source_id,  # Now using fully qualified path
                            "type": ElementType.LIST_ITEM.value,
                            "list_type": list_type,
                            "index": i,
                            "element_id": item_id
                        }),
                        "content_hash": self._generate_hash(item_text),
                        "metadata": {
                            "index": i,
                            "full_path": source_id  # Store the full path in metadata
                        }
                    }

                    elements.append(item_element)

                    # Create relationship from list to item
                    contains_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": list_id,
                        "target_id": item_id,
                        "relationship_type": RelationshipType.CONTAINS_LIST_ITEM.value,
                        "metadata": {
                            "confidence": 1.0,
                            "index": i
                        }
                    }
                    relationships.append(contains_relationship)

                    # Create inverse relationship
                    contained_by_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": item_id,
                        "target_id": list_id,
                        "relationship_type": RelationshipType.CONTAINED_BY.value,
                        "metadata": {
                            "confidence": 1.0
                        }
                    }
                    relationships.append(contained_by_relationship)

                    # Extract links from list item
                    for a in item.find_all('a', href=True):
                        links.append({
                            "source_id": item_id,
                            "link_text": a.get_text().strip(),
                            "link_target": a['href'],
                            "link_type": "html"
                        })

            elif tag.name == 'pre':
                # Code block
                code_tag = tag.find('code')
                code_text = code_tag.get_text() if code_tag else tag.get_text()

                # Try to get language
                language = ""
                if code_tag and code_tag.has_attr('class'):
                    for cls in code_tag['class']:
                        if cls.startswith('language-'):
                            language = cls[9:]
                            break

                element_id = self._generate_id("code_")

                code_element = {
                    "element_id": element_id,
                    "doc_id": doc_id,
                    "element_type": ElementType.CODE_BLOCK.value,
                    "parent_id": current_parent,
                    "content_preview": code_text[:self.max_content_preview] + (
                        "..." if len(code_text) > self.max_content_preview else ""),
                    "content_location": json.dumps({
                        "source": source_id,  # Now using fully qualified path
                        "type": ElementType.CODE_BLOCK.value,
                        "language": language,
                        "element_id": element_id
                    }),
                    "content_hash": self._generate_hash(code_text),
                    "metadata": {
                        "language": language,
                        "full_path": source_id  # Store the full path in metadata
                    }
                }

                elements.append(code_element)

                # Create relationship from parent to code block
                contains_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": current_parent,
                    "target_id": element_id,
                    "relationship_type": RelationshipType.CONTAINS.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contains_relationship)

                # Create inverse relationship
                contained_by_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": element_id,
                    "target_id": current_parent,
                    "relationship_type": RelationshipType.CONTAINED_BY.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contained_by_relationship)

            elif tag.name == 'blockquote':
                # Blockquote element
                quote_text = tag.get_text().strip()
                element_id = self._generate_id("quote_")

                quote_element = {
                    "element_id": element_id,
                    "doc_id": doc_id,
                    "element_type": ElementType.BLOCKQUOTE.value,
                    "parent_id": current_parent,
                    "content_preview": quote_text[:self.max_content_preview] + (
                        "..." if len(quote_text) > self.max_content_preview else ""),
                    "content_location": json.dumps({
                        "source": source_id,  # Now using fully qualified path
                        "type": ElementType.BLOCKQUOTE.value,
                        "element_id": element_id
                    }),
                    "content_hash": self._generate_hash(quote_text),
                    "metadata": {
                        "full_path": source_id  # Store the full path in metadata
                    }
                }

                elements.append(quote_element)

                # Create relationship from parent to blockquote
                contains_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": current_parent,
                    "target_id": element_id,
                    "relationship_type": RelationshipType.CONTAINS.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contains_relationship)

                # Create inverse relationship
                contained_by_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": element_id,
                    "target_id": current_parent,
                    "relationship_type": RelationshipType.CONTAINED_BY.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contained_by_relationship)

                # Extract links from blockquote
                for a in tag.find_all('a', href=True):
                    links.append({
                        "source_id": element_id,
                        "link_text": a.get_text().strip(),
                        "link_target": a['href'],
                        "link_type": "html"
                    })

            elif tag.name == 'table':
                # Table element
                table_id = self._generate_id("table_")
                table_html = str(tag)

                table_element = {
                    "element_id": table_id,
                    "doc_id": doc_id,
                    "element_type": ElementType.TABLE.value,
                    "parent_id": current_parent,
                    "content_preview": "Table",
                    "content_location": json.dumps({
                        "source": source_id,  # Now using fully qualified path
                        "type": ElementType.TABLE.value,
                        "element_id": table_id
                    }),
                    "content_hash": self._generate_hash(table_html),
                    "metadata": {
                        "rows": len(tag.find_all('tr')),
                        "has_header": bool(tag.find('thead') or tag.find('th')),
                        "full_path": source_id  # Store the full path in metadata
                    }
                }

                elements.append(table_element)

                # Create relationship from parent to table
                contains_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": current_parent,
                    "target_id": table_id,
                    "relationship_type": RelationshipType.CONTAINS.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contains_relationship)

                # Create inverse relationship
                contained_by_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": table_id,
                    "target_id": current_parent,
                    "relationship_type": RelationshipType.CONTAINED_BY.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contained_by_relationship)

                # Process headers
                header_row = tag.find('thead')
                if header_row:
                    header_cells = header_row.find_all('th')
                    for i, cell in enumerate(header_cells):
                        cell_text = cell.get_text().strip()
                        cell_id = self._generate_id("th_")

                        cell_element = {
                            "element_id": cell_id,
                            "doc_id": doc_id,
                            "element_type": ElementType.TABLE_HEADER.value,
                            "parent_id": table_id,
                            "content_preview": cell_text[:self.max_content_preview] + (
                                "..." if len(cell_text) > self.max_content_preview else ""),
                            "content_location": json.dumps({
                                "source": source_id,  # Now using fully qualified path
                                "type": ElementType.TABLE_HEADER.value,
                                "col": i,
                                "element_id": cell_id
                            }),
                            "content_hash": self._generate_hash(cell_text),
                            "metadata": {
                                "col": i,
                                "full_path": source_id  # Store the full path in metadata
                            }
                        }

                        elements.append(cell_element)

                        # Create relationship from table to header cell
                        contains_relationship = {
                            "relationship_id": self._generate_id("rel_"),
                            "source_id": table_id,
                            "target_id": cell_id,
                            "relationship_type": RelationshipType.CONTAINS_TABLE_HEADER.value,
                            "metadata": {
                                "confidence": 1.0,
                                "col": i
                            }
                        }
                        relationships.append(contains_relationship)

                        # Create inverse relationship
                        contained_by_relationship = {
                            "relationship_id": self._generate_id("rel_"),
                            "source_id": cell_id,
                            "target_id": table_id,
                            "relationship_type": RelationshipType.CONTAINED_BY.value,
                            "metadata": {
                                "confidence": 1.0
                            }
                        }
                        relationships.append(contained_by_relationship)

                        # Extract links from header cell
                        for a in cell.find_all('a', href=True):
                            links.append({
                                "source_id": cell_id,
                                "link_text": a.get_text().strip(),
                                "link_target": a['href'],
                                "link_type": "html"
                            })

                # Process rows
                tbody = tag.find('tbody') or tag
                for i, row in enumerate(tbody.find_all('tr')):
                    row_id = self._generate_id("tr_")

                    row_element = {
                        "element_id": row_id,
                        "doc_id": doc_id,
                        "element_type": ElementType.TABLE_ROW.value,
                        "parent_id": table_id,
                        "content_preview": f"Row {i + 1}",
                        "content_location": json.dumps({
                            "source": source_id,  # Now using fully qualified path
                            "type": ElementType.TABLE_ROW.value,
                            "row": i,
                            "element_id": row_id
                        }),
                        "content_hash": self._generate_hash(str(row)),
                        "metadata": {
                            "row": i,
                            "full_path": source_id
                        }
                    }

                    elements.append(row_element)

                    # Create relationship from table to row
                    contains_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": table_id,
                        "target_id": row_id,
                        "relationship_type": RelationshipType.CONTAINS_TABLE_ROW.value,
                        "metadata": {
                            "confidence": 1.0,
                            "row": i
                        }
                    }
                    relationships.append(contains_relationship)

                    # Create inverse relationship
                    contained_by_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": row_id,
                        "target_id": table_id,
                        "relationship_type": RelationshipType.CONTAINED_BY.value,
                        "metadata": {
                            "confidence": 1.0
                        }
                    }
                    relationships.append(contained_by_relationship)

                    # Process cells
                    for j, cell in enumerate(row.find_all(['td', 'th'])):
                        cell_text = cell.get_text().strip()
                        cell_id = self._generate_id("td_")

                        cell_element = {
                            "element_id": cell_id,
                            "doc_id": doc_id,
                            "element_type": ElementType.TABLE_CELL.value,
                            "parent_id": row_id,
                            "content_preview": cell_text[:self.max_content_preview] + (
                                "..." if len(cell_text) > self.max_content_preview else ""),
                            "content_location": json.dumps({
                                "source": source_id,
                                "type": ElementType.TABLE_CELL.value,
                                "row": i,
                                "col": j,
                                "element_id": cell_id
                            }),
                            "content_hash": self._generate_hash(cell_text),
                            "metadata": {
                                "row": i,
                                "col": j,
                                "full_path": source_id
                            }
                        }

                        elements.append(cell_element)

                        # Create relationship from row to cell
                        contains_relationship = {
                            "relationship_id": self._generate_id("rel_"),
                            "source_id": row_id,
                            "target_id": cell_id,
                            "relationship_type": RelationshipType.CONTAINS_TABLE_CELL.value,
                            "metadata": {
                                "confidence": 1.0,
                                "col": j
                            }
                        }
                        relationships.append(contains_relationship)

                        # Create inverse relationship
                        contained_by_relationship = {
                            "relationship_id": self._generate_id("rel_"),
                            "source_id": cell_id,
                            "target_id": row_id,
                            "relationship_type": RelationshipType.CONTAINED_BY.value,
                            "metadata": {
                                "confidence": 1.0
                            }
                        }
                        relationships.append(contained_by_relationship)

                        # Extract links from cell
                        for a in cell.find_all('a', href=True):
                            links.append({
                                "source_id": cell_id,
                                "link_text": a.get_text().strip(),
                                "link_target": a['href'],
                                "link_type": "html"
                            })

        result = (elements, links, relationships)

        # Cache the result
        if self.enable_caching:
            self.document_cache.set(cache_key, result)

        return result

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

            # Check if source exists and is a file
            if not os.path.exists(source) or not os.path.isfile(source):
                return False

            # Check file extension for markdown
            _, ext = os.path.splitext(source.lower())
            return ext in ['.md', '.markdown']

        except (json.JSONDecodeError, TypeError):
            return False

    @ttl_cache(maxsize=256, ttl=3600)
    def _extract_header(self, content: str, header_text: str, header_level: Optional[int] = None) -> str:
        """
        Extract header by text and/or level.

        Args:
            content: Markdown content
            header_text: Header text to match
            header_level: Optional header level (1-6)

        Returns:
            Extracted header or empty string if not found
        """
        if not header_text and header_level is None:
            return ""

        # Pattern for headers with specific level if provided
        level_pattern = f"^{'#' * header_level}\\s+" if header_level else r'^#{1,6}\s+'

        if header_text:
            # Look for exact header with specific level
            header_pattern = re.compile(level_pattern + re.escape(header_text) + r'$', re.MULTILINE)
            match = header_pattern.search(content)

            if match:
                return match.group(0)

            # Not found, look for approximate match
            lines = content.split('\n')
            for line in lines:
                if re.match(level_pattern, line) and header_text in line:
                    return line

            return ""
        else:
            # No specific text, just find header by level
            header_pattern = re.compile(level_pattern + r'(.+)$', re.MULTILINE)
            match = header_pattern.search(content)

            if match:
                return match.group(0)

            return ""

    @ttl_cache(maxsize=256, ttl=3600)
    def _extract_paragraph(self, content: str, para_text: str) -> str:
        """
        Extract paragraph by text.

        Args:
            content: Markdown content
            para_text: Paragraph text snippet to find

        Returns:
            Extracted paragraph or empty string if not found
        """
        if not para_text:
            return ""

        # Split content into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)

        # Look for paragraph containing the text
        for para in paragraphs:
            if para_text in para:
                return para.strip()

        return ""

    @ttl_cache(maxsize=256, ttl=3600)
    def _extract_list_item(self, content: str, index: int, list_type: str) -> str:
        """
        Extract list item by index and type.

        Args:
            content: Markdown content
            index: Item index
            list_type: List type (ordered or unordered)

        Returns:
            Extracted list item or empty string if not found
        """
        # Define patterns based on list type
        if list_type == "ordered":
            item_pattern = r'^\s*\d+\.\s+(.+)$'
        else:  # unordered
            item_pattern = r'^\s*[\*\-\+]\s+(.+)$'

        # Find list items
        items = re.findall(item_pattern, content, re.MULTILINE)

        # Return item at specified index
        if 0 <= index < len(items):
            return items[index]

        return ""

    @ttl_cache(maxsize=256, ttl=3600)
    def _extract_code_block(self, content: str, language: str) -> str:
        """
        Extract code block by language.

        Args:
            content: Markdown content
            language: Programming language of code block

        Returns:
            Extracted code block or empty string if not found
        """
        # Look for code blocks with specified language
        pattern = r'```' + language + r'\s*\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)

        # Return first match
        if matches:
            return matches[0]

        # If language-specific code block not found, look for any code block
        pattern = r'```.*?\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)

        # Return first match
        if matches:
            return matches[0]

        return ""

    @ttl_cache(maxsize=256, ttl=3600)
    def _extract_blockquote(self, content: str) -> str:
        """
        Extract blockquote from markdown content.

        Args:
            content: Markdown content

        Returns:
            Extracted blockquote or empty string if not found
        """
        # Look for blockquote (lines starting with >)
        pattern = r'((?:^\s*>.*$\n?)+)'
        matches = re.findall(pattern, content, re.MULTILINE)

        # Return first match
        if matches:
            # Remove > prefix from each line
            lines = matches[0].split('\n')
            cleaned_lines = [re.sub(r'^\s*>\s?', '', line) for line in lines if line.strip()]
            return '\n'.join(cleaned_lines)

        return ""

    @ttl_cache(maxsize=256, ttl=3600)
    def _extract_table_element(self, content: str, element_type: str, row: int, col: int) -> str:
        """
        Extract table element by type, row, and column.

        Args:
            content: Markdown content
            element_type: Element type (table, table_row, table_cell, table_header)
            row: Row index
            col: Column index

        Returns:
            Extracted table element or empty string if not found
        """
        # Extract table from markdown
        tables = re.findall(r'(\|.*\|(?:\n\|.*\|)+)', content)

        if not tables:
            return ""

        # Use first table found
        table_str = tables[0]

        # Split into rows
        rows = table_str.strip().split('\n')

        # Remove separator row if present (contains only dashes and pipes)
        rows = [r for r in rows if not re.match(r'^\s*\|[\s\-\|]+\|\s*$', r)]

        if element_type == ElementType.TABLE.value:
            # Return entire table
            return table_str
        elif element_type in (ElementType.TABLE_ROW.value, ElementType.TABLE_HEADER.value):
            # Return specific row
            if 0 <= row < len(rows):
                return rows[row]
            return ""
        elif element_type in (ElementType.TABLE_CELL.value, ElementType.TABLE_HEADER.value):
            # Return specific cell
            if 0 <= row < len(rows):
                # Split row into cells
                cells = re.findall(r'\|(.*?)(?=\||$)', rows[row])

                # Remove empty cells
                cells = [cell.strip() for cell in cells if cell.strip()]

                if 0 <= col < len(cells):
                    return cells[col]

            return ""
        else:
            return ""

    def clear_caches(self):
        """Clear all caches."""
        self.document_cache.clear()
        self.html_cache.clear()
        self.text_cache.clear()
        logger.info("All caches cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the caches.

        Returns:
            Dictionary with cache statistics
        """
        if not self.enable_caching:
            return {"caching_enabled": False}

        return {
            "caching_enabled": True,
            "document_cache": {
                "size": len(self.document_cache.cache),
                "max_size": self.document_cache.max_size,
                "ttl": self.document_cache.ttl
            },
            "html_cache": {
                "size": len(self.html_cache.cache),
                "max_size": self.html_cache.max_size,
                "ttl": self.html_cache.ttl
            },
            "text_cache": {
                "size": len(self.text_cache.cache),
                "max_size": self.text_cache.max_size,
                "ttl": self.text_cache.ttl
            }
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance statistics
        """
        if not self.enable_performance_monitoring:
            return {"performance_monitoring_enabled": False}

        stats = self.performance_stats.copy()

        # Add derived metrics
        if stats["parse_count"] > 0:
            stats["avg_parse_time"] = stats["total_parse_time"] / stats["parse_count"]

        # Add cache efficiency
        total_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_requests > 0:
            stats["cache_hit_ratio"] = stats["cache_hits"] / total_requests

        return {
            "performance_monitoring_enabled": True,
            **stats
        }

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            "parse_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_parse_time": 0.0,
            "total_element_processing_time": 0.0,
            "total_link_extraction_time": 0.0,
            "method_times": {}
        }
        logger.info("Performance statistics reset")

    def performance_monitor(self, method_name):
        """
        Decorator for monitoring method performance.

        Args:
            method_name: Name of the method to monitor
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_performance_monitoring:
                    return func(*args, **kwargs)

                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time

                # Update statistics
                if method_name not in self.performance_stats["method_times"]:
                    self.performance_stats["method_times"][method_name] = {
                        "calls": 0,
                        "total_time": 0,
                        "min_time": float('inf'),
                        "max_time": 0
                    }

                stats = self.performance_stats["method_times"][method_name]
                stats["calls"] += 1
                stats["total_time"] += elapsed_time
                stats["min_time"] = min(stats["min_time"], elapsed_time)
                stats["max_time"] = max(stats["max_time"], elapsed_time)

                # Log if this is a particularly slow operation
                if elapsed_time > 1.0:  # Log operations taking more than 1 second
                    logger.warning(f"Slow operation: {method_name} took {elapsed_time:.4f} seconds")

                return result

            return wrapper

        return decorator
