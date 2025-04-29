"""
Confluence Content Resolver implementation for the document pointer system.

This module resolves Confluence content pointers to actual content.
"""
import json
import logging
import re

import requests

from .base import ContentResolver

logger = logging.getLogger(__name__)


class ConfluenceContentResolver(ContentResolver):
    """Resolver for Confluence content."""

    def __init__(self):
        """Initialize the Confluence content resolver."""
        self.sessions = {}  # Cache for API sessions
        self.content_cache = {}  # Cache for retrieved content

    def resolve_content(self, content_location: str) -> str:
        """
        Resolve Confluence content pointer to actual content.

        Args:
            content_location: JSON-formatted content location pointer

        Returns:
            Resolved content as string
        """
        location_data = json.loads(content_location)

        source = location_data.get("source", "")
        if not source.startswith("confluence://"):
            raise ValueError(f"Invalid Confluence source: {source}")

        # Extract info from source identifier
        # Format: confluence://base_url/space_key/content_id
        match = re.match(r'confluence://([^/]+)/([^/]+)/(\d+)', source)
        if not match:
            raise ValueError(f"Invalid Confluence source format: {source}")

        base_url, space_key, content_id = match.groups()

        # Ensure base_url has a scheme
        if not base_url.startswith(('http://', 'https://')):
            base_url = 'https://' + base_url

        # Try to get API credentials from session cache
        session = self._get_session(base_url)

        # Determine what part of the content to return based on element type
        element_type = location_data.get("type", "")

        # Construct API URL
        api_url = f"{base_url}/rest/api/content/{content_id}"
        params = {
            "expand": "body.storage,metadata,version"
        }

        # Check cache first
        cache_key = f"{base_url}:{content_id}:{element_type}"
        if cache_key in self.content_cache:
            logger.debug(f"Using cached content for: {cache_key}")
            return self.content_cache[cache_key]

        try:
            # Make API request
            response = session.get(api_url, params=params)
            response.raise_for_status()
            content_data = response.json()

            # Extract content based on element type
            if element_type == "root" or element_type == "body":
                # Return full content
                html_content = content_data.get("body", {}).get("storage", {}).get("value", "")
                resolved_content = html_content
            elif element_type == "header":
                # Extract header by text or level
                header_text = location_data.get("text", "")
                level = location_data.get("level", 1)

                html_content = content_data.get("body", {}).get("storage", {}).get("value", "")
                resolved_content = self._extract_header(html_content, header_text, level)
            elif element_type == "paragraph":
                # Extract paragraph by text or index
                para_text = location_data.get("text", "")
                para_index = location_data.get("index", 0)

                html_content = content_data.get("body", {}).get("storage", {}).get("value", "")
                resolved_content = self._extract_paragraph(html_content, para_text, para_index)
            elif element_type == "table":
                # Extract table
                table_index = location_data.get("index", 0)

                html_content = content_data.get("body", {}).get("storage", {}).get("value", "")
                resolved_content = self._extract_table(html_content, table_index)
            elif element_type == "list" or element_type == "list_item":
                # Extract list or list item
                list_type = location_data.get("list_type", "")
                item_index = location_data.get("index", 0)

                html_content = content_data.get("body", {}).get("storage", {}).get("value", "")
                resolved_content = self._extract_list(html_content, list_type, item_index, element_type == "list_item")
            else:
                # Default: return full content
                html_content = content_data.get("body", {}).get("storage", {}).get("value", "")
                resolved_content = html_content

            # Cache the result
            self.content_cache[cache_key] = resolved_content

            return resolved_content

        except Exception as e:
            logger.error(f"Error resolving Confluence content: {str(e)}")
            raise

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
            # Source must be a Confluence URI
            return source.startswith("confluence://")
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
        # For Confluence, we'll return the HTML content as bytes
        content = self.resolve_content(content_location)
        return content.encode('utf-8')

    def _get_session(self, base_url: str) -> requests.Session:
        """
        Get or create a session for the given base URL.

        Args:
            base_url: Confluence base URL

        Returns:
            Requests session with authentication
        """
        if base_url in self.sessions:
            return self.sessions[base_url]

        # Create a new session
        session = requests.Session()

        # Try to find credentials for this base URL
        # In a real implementation, this would use a secure credential store
        # or configuration service. For now, we'll just return the unauthenticated session.

        # Add any required headers
        session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json"
        })

        # Cache the session
        self.sessions[base_url] = session

        return session

    @staticmethod
    def _extract_header(html_content: str, header_text: str, level: int) -> str:
        """
        Extract header from HTML content.

        Args:
            html_content: HTML content
            header_text: Header text to find
            level: Header level

        Returns:
            Header HTML or empty string if not found
        """
        from bs4 import BeautifulSoup

        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Look for specific header
        headers = soup.find_all(f'h{level}')

        if header_text:
            # Find by text
            for header in headers:
                if header_text in header.get_text():
                    return str(header)

            # Not found at specified level, try any level
            for i in range(1, 7):
                headers = soup.find_all(f'h{i}')
                for header in headers:
                    if header_text in header.get_text():
                        return str(header)
        elif headers:
            # Return first header at specified level
            return str(headers[0])

        # Header not found
        return ""

    @staticmethod
    def _extract_paragraph(html_content: str, para_text: str, para_index: int) -> str:
        """
        Extract paragraph from HTML content.

        Args:
            html_content: HTML content
            para_text: Paragraph text to find
            para_index: Paragraph index

        Returns:
            Paragraph HTML or empty string if not found
        """
        from bs4 import BeautifulSoup

        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all paragraphs
        paragraphs = soup.find_all('p')

        if para_text:
            # Find by text
            for paragraph in paragraphs:
                if para_text in paragraph.get_text():
                    return str(paragraph)
        elif 0 <= para_index < len(paragraphs):
            # Find by index
            return str(paragraphs[para_index])

        # Paragraph not found
        return ""

    @staticmethod
    def _extract_table(html_content: str, table_index: int) -> str:
        """
        Extract table from HTML content.

        Args:
            html_content: HTML content
            table_index: Table index

        Returns:
            Table HTML or empty string if not found
        """
        from bs4 import BeautifulSoup

        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all tables
        tables = soup.find_all('table')

        if 0 <= table_index < len(tables):
            return str(tables[table_index])

        # Table not found
        return ""

    @staticmethod
    def _extract_list(html_content: str, list_type: str, item_index: int, extract_item: bool) -> str:
        """
        Extract list or list item from HTML content.

        Args:
            html_content: HTML content
            list_type: List type ('ordered' or 'unordered')
            item_index: Item index
            extract_item: True to extract a specific item, False to extract the whole list

        Returns:
            List or list item HTML or empty string if not found
        """
        from bs4 import BeautifulSoup

        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Determine list tag based on type
        list_tag = 'ol' if list_type == 'ordered' else 'ul'

        # Find all lists of specified type
        lists = soup.find_all(list_tag)

        if not lists:
            return ""

        # Default to first list
        target_list = lists[0]

        if extract_item:
            # Extract specific list item
            items = target_list.find_all('li')

            if 0 <= item_index < len(items):
                return str(items[item_index])

            # Item not found
            return ""
        else:
            # Extract entire list
            return str(target_list)
