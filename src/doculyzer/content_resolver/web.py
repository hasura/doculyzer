"""
Web Content Resolver implementation for the document pointer system.

This module resolves web-based content pointers to actual content.
"""
import json
import logging
from typing import Dict, Any

import requests
from bs4 import BeautifulSoup

from .base import ContentResolver

logger = logging.getLogger(__name__)


class WebContentResolver(ContentResolver):
    """Resolver for web content."""

    def __init__(self, web_config: Dict[str, Any] = None):
        """
        Initialize the web content resolver.

        Args:
            web_config: Web content source configuration
        """
        self.binary_cache = None
        self.config = web_config or {}
        self.session = requests.Session()

        # Configure session
        self.headers = self.config.get("headers", {})
        self.session.headers.update(self.headers)

        # Configure authentication if provided
        auth_config = self.config.get("authentication", {})
        auth_type = auth_config.get("type")

        if auth_type == "basic":
            self.session.auth = (
                auth_config.get("username", ""),
                auth_config.get("password", "")
            )
        elif auth_type == "bearer":
            self.session.headers.update({
                "Authorization": f"Bearer {auth_config.get('token', '')}"
            })

        # Cache for retrieved content
        self.cache = {}

    def resolve_content(self, content_location: str) -> str:
        """
        Resolve web content.

        Args:
            content_location: JSON-formatted content location pointer

        Returns:
            Resolved content as string
        """
        location_data = json.loads(content_location)

        source = location_data.get("source", "")
        element_type = location_data.get("type", "")

        # For web URLs, source should already be a fully qualified URL
        url = source

        # Fetch content if not in cache
        if url not in self.cache:
            try:
                response = self.session.get(url)
                response.raise_for_status()
                self.cache[url] = response.text
            except Exception as e:
                logger.error(f"Error fetching URL {url}: {str(e)}")
                raise

        content = self.cache[url]

        # Extract specific content based on element type and location
        if element_type == "root":
            # Return full content
            return content
        else:
            # Extract based on selector if available
            selector = location_data.get("selector", "")
            if selector:
                return self._extract_by_selector(content, selector)

            # Fall back to extraction based on element type
            return self._extract_by_type(content, location_data)

    def supports_location(self, content_location: str) -> bool:
        """
        Check if this resolver supports the location.

        Args:
            content_location: Content location pointer

        Returns:
            True if supported, False otherwise
        """
        try:
            location_data = json.loads(content_location)
            source = location_data.get("source", "")
            # Source must be a URL
            return source.startswith(('http://', 'https://'))
        except (json.JSONDecodeError, TypeError):
            return False

    def get_document_binary(self, content_location: str) -> bytes:
        """
        Get the containing document as a binary blob.

        Args:
            content_location: Content location pointer

        Returns:
            Document binary content

        Raises:
            ValueError: If document cannot be retrieved
        """
        location_data = json.loads(content_location)
        source = location_data.get("source", "")

        # Ensure source is a valid URL
        if not source.startswith(('http://', 'https://')):
            raise ValueError(f"Not a valid URL: {source}")

        # Check binary content cache
        binary_cache_key = f"binary_{source}"
        if hasattr(self, 'binary_cache') and binary_cache_key in self.binary_cache:
            return self.binary_cache[binary_cache_key]

        # Initialize binary cache if needed
        if not hasattr(self, 'binary_cache'):
            self.binary_cache = {}

        # Fetch binary content
        try:
            response = self.session.get(source, stream=True)
            response.raise_for_status()

            # Get binary content
            binary_content = response.content

            # Cache binary content
            self.binary_cache[binary_cache_key] = binary_content

            return binary_content
        except Exception as e:
            logger.error(f"Error fetching URL {source}: {str(e)}")
            raise ValueError(f"Failed to retrieve document: {str(e)}")

    @staticmethod
    def _extract_by_selector(content: str, selector: str) -> str:
        """
        Extract content using CSS selector.

        Args:
            content: HTML content
            selector: CSS selector

        Returns:
            Extracted content
        """
        soup = BeautifulSoup(content, 'html.parser')
        element = soup.select_one(selector)

        if element:
            return element.get_text()

        return ""

    @staticmethod
    def _extract_by_type(content: str, location_data: Dict[str, Any]) -> str:
        """
        Extract content based on element type and location data.

        Args:
            content: HTML content
            location_data: Content location data

        Returns:
            Extracted content
        """
        element_type = location_data.get("type", "")

        # Parse HTML
        soup = BeautifulSoup(content, 'html.parser')

        # Extract based on element type
        if element_type == "header":
            header_level = location_data.get("level", 1)
            headers = soup.find_all(f'h{header_level}')

            header_text = location_data.get("text", "")
            if header_text:
                # Find header by text
                for header in headers:
                    if header_text in header.get_text():
                        return header.get_text()

            # Return first header if text not specified
            if headers:
                return headers[0].get_text()

        elif element_type == "paragraph":
            paragraphs = soup.find_all('p')

            para_text = location_data.get("text", "")
            if para_text:
                # Find paragraph by text
                for para in paragraphs:
                    if para_text in para.get_text():
                        return para.get_text()

            # Return paragraph by index
            para_index = location_data.get("index", 0)
            if 0 <= para_index < len(paragraphs):
                return paragraphs[para_index].get_text()

        elif element_type in ("list", "list_item"):
            list_type = location_data.get("list_type", "")
            list_tag = 'ol' if list_type == "ordered" else 'ul'

            lists = soup.find_all(list_tag)
            list_index = location_data.get("list_index", 0)

            if 0 <= list_index < len(lists):
                list_element = lists[list_index]

                if element_type == "list":
                    return list_element.get_text()
                else:
                    # Extract specific list item
                    items = list_element.find_all('li')
                    item_index = location_data.get("index", 0)

                    if 0 <= item_index < len(items):
                        return items[item_index].get_text()

        elif element_type in ("table", "table_row", "table_cell", "table_header"):
            tables = soup.find_all('table')
            table_index = location_data.get("table_index", 0)

            if 0 <= table_index < len(tables):
                table = tables[table_index]

                if element_type == "table":
                    return table.get_text()
                else:
                    # Extract row, cell, etc.
                    row_index = location_data.get("row", 0)
                    col_index = location_data.get("col", 0)

                    rows = table.find_all('tr')
                    if 0 <= row_index < len(rows):
                        row = rows[row_index]

                        if element_type == "table_row":
                            return row.get_text()
                        else:
                            # Extract cell
                            cells = row.find_all(['td', 'th'])
                            if 0 <= col_index < len(cells):
                                return cells[col_index].get_text()

        # Default: return full content
        return content
