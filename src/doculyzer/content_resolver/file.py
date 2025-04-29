import json
import logging
import os
import re

from .base import ContentResolver

logger = logging.getLogger(__name__)


class FileContentResolver(ContentResolver):
    """Resolver for file-based content."""

    def resolve_content(self, content_location: str) -> str:
        """
        Resolve file-based content.

        Args:
            content_location: JSON-formatted content location pointer

        Returns:
            Resolved content as string

        Raises:
            FileNotFoundError: If source file doesn't exist
        """
        location_data = json.loads(content_location)

        source = location_data.get("source", "")
        element_type = location_data.get("type", "")

        # Since we're using fully qualified paths, no need to join with base path
        # Just ensure the file exists
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Read file content
        with open(source, 'r', encoding='utf-8') as f:
            file_content = f.read()

        # Extract specific content based on element type and location
        if element_type == "root":
            # Return full file content
            return file_content
        elif element_type == "header":
            # Extract header by text
            header_text = location_data.get("text", "")
            return self._extract_header(file_content, header_text)
        elif element_type == "paragraph":
            # Extract paragraph
            para_text = location_data.get("text", "")
            return self._extract_paragraph(file_content, para_text)
        elif element_type in ("list", "list_item"):
            # Extract list or list item
            index = location_data.get("index", 0)
            list_type = location_data.get("list_type", "")
            return self._extract_list_item(file_content, index, list_type)
        elif element_type == "code_block":
            # Extract code block
            language = location_data.get("language", "")
            return self._extract_code_block(file_content, language)
        elif element_type == "blockquote":
            # Extract blockquote
            return self._extract_blockquote(file_content)
        elif element_type in ("table", "table_row", "table_cell", "table_header"):
            # Extract table element
            row = location_data.get("row", 0)
            col = location_data.get("col", 0)
            return self._extract_table_element(file_content, element_type, row, col)
        else:
            # Unknown element type
            return ""

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

            # Source must be a file path and file must exist
            return os.path.exists(source)

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
            FileNotFoundError: If document cannot be found
        """
        location_data = json.loads(content_location)
        source = location_data.get("source", "")

        # Ensure source is a valid file path
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Read file as binary
        with open(source, 'rb') as f:
            return f.read()

    @staticmethod
    def _extract_header(content: str, header_text: str) -> str:
        """
        Extract header by text.

        Args:
            content: File content
            header_text: Header text to find

        Returns:
            Extracted header or empty string if not found
        """
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
        """
        Extract paragraph by text.

        Args:
            content: File content
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

    @staticmethod
    def _extract_list_item(content: str, index: int, list_type: str) -> str:
        """
        Extract list item by index and type.

        Args:
            content: File content
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

    @staticmethod
    def _extract_code_block(content: str, language: str) -> str:
        """
        Extract code block by language.

        Args:
            content: File content
            language: Code block language

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

    @staticmethod
    def _extract_blockquote(content: str) -> str:
        """
        Extract blockquote.

        Args:
            content: File content

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

    @staticmethod
    def _extract_table_element(content: str, element_type: str, row: int, col: int) -> str:
        """
        Extract table element by type, row, and column.

        Args:
            content: File content
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
