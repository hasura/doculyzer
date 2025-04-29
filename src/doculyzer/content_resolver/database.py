"""
Database Content Resolver implementation for the document pointer system.

This module resolves database-based content pointers to actual content.
"""
import json
import logging
import re
from typing import Dict, Any, Optional

from .base import ContentResolver

logger = logging.getLogger(__name__)


class DatabaseContentResolver(ContentResolver):
    """Resolver for database blob content."""

    def __init__(self):
        """Initialize the database content resolver."""
        self.connections = {}  # Cache for database connections

    def resolve_content(self, content_location: str) -> str:
        """
        Resolve database content.

        Args:
            content_location: JSON-formatted content location pointer

        Returns:
            Resolved content as string

        Raises:
            ValueError: If source is invalid or record not found
        """
        location_data = json.loads(content_location)

        source = location_data.get("source", "")
        element_type = location_data.get("type", "")

        # Parse database connection info from source
        db_info = self._parse_db_source(source)

        if not db_info:
            raise ValueError(f"Invalid database source: {source}")

        # Get database connection
        conn = self._get_connection(db_info)

        # Extract content based on location data
        if element_type == "root":
            # Return full content of the specified record
            return self._fetch_record(conn, db_info)
        else:
            # Extract specific part of content
            content = self._fetch_record(conn, db_info)
            return self._extract_content(content, location_data)

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
            # Source must be a database URI
            return source.startswith("db://")
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

        # Parse database connection info from source
        db_info = self._parse_db_source(source)

        if not db_info:
            raise ValueError(f"Invalid database source: {source}")

        # Get database connection
        conn = self._get_connection(db_info)

        # Build query - modify to fetch binary content if available
        table = db_info["table"]
        pk_column = db_info["pk_column"]
        pk_value = db_info["pk_value"]
        content_column = db_info["content_column"]

        query = f"SELECT {content_column} FROM {table} WHERE {pk_column} = ?"

        cursor = conn.execute(query, (pk_value,))
        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"Record not found: {pk_value}")

        content = row[content_column]

        # If content is already bytes, return directly
        if isinstance(content, bytes):
            return content

        # Otherwise convert string to bytes
        return content.encode('utf-8')

    @staticmethod
    def _parse_db_source(source) -> Optional[Dict[str, str]]:
        """
        Parse database source URI.

        Format: db://<connection_id>/<table>/<pk_column>/<pk_value>/<content_column>

        Returns:
            Dictionary with connection info or None if invalid
        """
        if not source.startswith("db://"):
            return None

        # Remove 'db://' prefix
        path = source[5:]

        # Split path
        parts = path.split('/')

        if len(parts) < 5:
            return None

        return {
            "connection_id": parts[0],
            "table": parts[1],
            "pk_column": parts[2],
            "pk_value": parts[3],
            "content_column": parts[4]
        }

    def _get_connection(self, db_info: Dict[str, str]) -> Any:
        """
        Get database connection.

        Args:
            db_info: Database connection info

        Returns:
            Database connection
        """
        import sqlite3

        connection_id = db_info["connection_id"]

        # Check if connection already exists
        if connection_id in self.connections:
            return self.connections[connection_id]

        # Create new connection
        # This is a simplified example that assumes SQLite
        # In a real implementation, this would handle different database types
        conn = sqlite3.connect(f"{connection_id}.db")
        conn.row_factory = sqlite3.Row

        # Cache connection
        self.connections[connection_id] = conn

        return conn

    @staticmethod
    def _fetch_record(conn, db_info: Dict[str, str]) -> str:
        """
        Fetch content from database.

        Args:
            db_info: Database connection info

        Returns:
            Content as string
        """
        table = db_info["table"]
        pk_column = db_info["pk_column"]
        pk_value = db_info["pk_value"]
        content_column = db_info["content_column"]

        # Build query
        query = f"SELECT {content_column} FROM {table} WHERE {pk_column} = ?"

        try:
            cursor = conn.execute(query, (pk_value,))
            row = cursor.fetchone()

            if row is None:
                return ""

            content = row[content_column]

            # Handle binary data
            if isinstance(content, bytes):
                return content.decode('utf-8')

            return content

        except Exception as e:
            logger.error(f"Error fetching record: {str(e)}")
            raise

    def _extract_content(self, content: str, location_data: Dict[str, Any]) -> str:
        """
        Extract specific content based on location data.

        Args:
            content: Full content string
            location_data: Content location data

        Returns:
            Extracted content
        """
        element_type = location_data.get("type", "")

        # For database blobs, the content might be in various formats
        # We need to determine the format and extract accordingly

        # Check if content is HTML
        if content.strip().startswith(("<html", "<!DOCTYPE html")):
            # Use HTML extraction
            return self._extract_html_content(content, location_data)

        # Check if content is Markdown
        elif re.search(r'^#{1,6}\s+', content, re.MULTILINE) or re.search(r'\[.+?\]\(.+?\)', content):
            # Use Markdown extraction (reuse file resolver methods)
            # We'll implement simplified versions of these here to avoid circular imports

            if element_type == "header":
                return self._extract_header(content, location_data.get("text", ""))
            elif element_type == "paragraph":
                return self._extract_paragraph(content, location_data.get("text", ""))
            elif element_type in ("list", "list_item"):
                return self._extract_list_item(
                    content, location_data.get("index", 0), location_data.get("list_type", ""))
            elif element_type == "code_block":
                return self._extract_code_block(content, location_data.get("language", ""))
            elif element_type == "blockquote":
                return self._extract_blockquote(content)
            elif element_type in ("table", "table_row", "table_cell", "table_header"):
                return self._extract_table_element(
                    content, element_type,
                    location_data.get("row", 0),
                    location_data.get("col", 0))

        # Default: return section by index
        section_index = location_data.get("index", 0)
        sections = re.split(r'\n\s*\n', content)

        if 0 <= section_index < len(sections):
            return sections[section_index].strip()

        return content

    @staticmethod
    def _extract_html_content(content: str, location_data: Dict[str, Any]) -> str:
        """
        Extract content from HTML.

        Args:
            content: HTML content
            location_data: Content location data

        Returns:
            Extracted content
        """
        from bs4 import BeautifulSoup

        element_type = location_data.get("type", "")
        selector = location_data.get("selector", "")

        # Parse HTML
        soup = BeautifulSoup(content, 'html.parser')

        # Use CSS selector if provided
        if selector:
            element = soup.select_one(selector)
            if element:
                return element.get_text()

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

    # Simplified markdown extraction methods to prevent circular imports

    @staticmethod
    def _extract_header(content: str, header_text: str) -> str:
        """Extract header by text."""
        if not header_text:
            return ""

        # Look for exact header
        header_pattern = re.compile(r'^(#{1,6}\s+' + re.escape(header_text) + r')$', re.MULTILINE)
        match = header_pattern.search(content)

        if match:
            return match.group(1)

        # Not found, look for approximate match
        lines = content.split('\n')
        for line in lines:
            if re.match(r'^#{1,6}\s+', line) and header_text in line:
                return line

        return header_text

    @staticmethod
    def _extract_paragraph(content: str, para_text: str) -> str:
        """Extract paragraph by text."""
        if not para_text:
            return ""

        # Split content into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)

        # Look for paragraph containing the text
        for para in paragraphs:
            if para_text in para:
                return para.strip()

        return ""

    @staticmethod
    def _extract_list_item(content: str, index: int, list_type: str) -> str:
        """Extract list item by index and type."""
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

    @staticmethod
    def _extract_code_block(content: str, language: str) -> str:
        """Extract code block by language."""
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

    @staticmethod
    def _extract_blockquote(content: str) -> str:
        """Extract blockquote."""
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

    @staticmethod
    def _extract_table_element(content: str, element_type: str, row: int, col: int) -> str:
        """Extract table element by type, row, and column."""
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

        if element_type == "table":
            # Return entire table
            return table_str
        elif element_type in ("table_row", "table_header"):
            # Return specific row
            if 0 <= row < len(rows):
                return rows[row]
            return ""
        elif element_type in ("table_cell", "table_header"):
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
