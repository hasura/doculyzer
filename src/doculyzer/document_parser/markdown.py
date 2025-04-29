"""
Markdown document parser module for the document pointer system.

This module parses Markdown documents into structured elements.
"""

import json
import logging
import os
import re
from typing import Dict, List, Any, Tuple, Optional, Union

import markdown
import yaml
from bs4 import BeautifulSoup

from .base import DocumentParser

logger = logging.getLogger(__name__)


class MarkdownParser(DocumentParser):
    """Parser for Markdown documents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Markdown parser."""
        super().__init__(config)
        self.extract_front_matter = self.config.get("extract_front_matter", True)
        self.paragraph_threshold = self.config.get("paragraph_threshold", 1)  # Min lines to consider a paragraph

        # Define Markdown-specific link patterns
        self.link_patterns = [
            r'\[\[(.*?)\]\]',  # Wiki-style links [[Page]]
            r'\[([^\]]+)\]\(([^)]+)\)'  # Markdown links [text](url)
        ]

    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a Markdown document into structured elements."""
        content = doc_content["content"]
        source_id = doc_content["id"]  # Should already be a fully qualified path
        metadata = doc_content.get("metadata", {}).copy()  # Make a copy to avoid modifying original

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
            "metadata": metadata,
            "content_hash": doc_content.get("content_hash", self._generate_hash(content))
        }

        # Create root element
        elements = [self._create_root_element(doc_id, source_id)]
        root_id = elements[0]["element_id"]

        # Extract links directly from Markdown content
        direct_links = self._extract_markdown_links(content, root_id)

        # Convert markdown to HTML for easier parsing
        html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])

        # Wrap in complete HTML structure if needed
        if not html_content.startswith('<!DOCTYPE html>') and not html_content.startswith('<html'):
            html_content = f"<html><body>{html_content}</body></html>"

        # Parse HTML to extract elements
        html_elements, html_links = self._parse_html_elements(html_content, doc_id, root_id, source_id)
        elements.extend(html_elements)

        # Combine links from both sources
        extracted_links = direct_links + html_links

        # Return the parsed document with extracted links
        return {
            "document": document,
            "elements": elements,
            "links": extracted_links,
            "relationships": []
        }

    def _resolve_element_content(self, location_data: Dict[str, Any],
                                 source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve content for specific markdown element types.

        Args:
            location_data: Content location data
            source_content: Optional pre-loaded source content

        Returns:
            Resolved content string
        """
        source = location_data.get("source", "")
        element_type = location_data.get("type", "")

        # Load content if not provided
        content = source_content
        if content is None:
            if os.path.exists(source):
                try:
                    # Try different encodings
                    for encoding in ['utf-8', 'latin1', 'cp1252']:
                        try:
                            with open(source, 'r', encoding=encoding) as f:
                                content = f.read()
                            break
                        except UnicodeDecodeError:
                            if encoding == 'cp1252':  # Last attempt
                                raise
                            continue
                except Exception as e:
                    raise ValueError(f"Error reading markdown file: {str(e)}")
            else:
                raise ValueError(f"Source file not found: {source}")

        # Ensure content is string (not bytes)
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                raise ValueError("Cannot decode binary content as markdown")

        # Extract front matter if enabled
        if self.extract_front_matter:
            content, _ = self._extract_front_matter(content)

        # Resolve based on element type
        if element_type == "header":
            # Extract header by text or level
            header_text = location_data.get("text", "")
            header_level = location_data.get("level")

            return self._extract_header(content, header_text, header_level)

        elif element_type == "paragraph":
            # Extract paragraph by text
            para_text = location_data.get("text", "")
            return self._extract_paragraph(content, para_text)

        elif element_type in ("list", "list_item"):
            # Extract list or list item
            index = location_data.get("index", 0)
            list_type = location_data.get("list_type", "")
            return self._extract_list_item(content, index, list_type)

        elif element_type == "code_block":
            # Extract code block
            language = location_data.get("language", "")
            return self._extract_code_block(content, language)

        elif element_type == "blockquote":
            # Extract blockquote
            return self._extract_blockquote(content)

        elif element_type in ("table", "table_row", "table_cell", "table_header"):
            # Extract table element
            row = location_data.get("row", 0)
            col = location_data.get("col", 0)
            return self._extract_table_element(content, element_type, row, col)

        else:
            # Unknown element type, return full content
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

            # Check if source exists and is a file
            if not os.path.exists(source) or not os.path.isfile(source):
                return False

            # Check file extension for markdown
            _, ext = os.path.splitext(source.lower())
            return ext in ['.md', '.markdown']

        except (json.JSONDecodeError, TypeError):
            return False

    @staticmethod
    def _extract_header(content: str, header_text: str, header_level: Optional[int] = None) -> str:
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

    # Rest of your implementation...
    # [Your existing methods for extract_paragraph, extract_list_item, etc.]

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

    def _extract_markdown_links(self, content: str, element_id: str) -> List[Dict[str, Any]]:
        """
        Extract links directly from Markdown content.

        Args:
            content: Markdown content
            element_id: ID of the element containing the links

        Returns:
            List of extracted link dictionaries
        """
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

        return links

    def _parse_html_elements(self, html_content: str, doc_id: str, root_id: str, source_id: str) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Parse HTML content into structured elements.

        Args:
            html_content: HTML content converted from Markdown
            doc_id: Document ID
            root_id: Root element ID
            source_id: Source identifier (fully qualified path)

        Returns:
            Tuple of (list of elements, list of links)
        """
        elements = []
        links = []

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
                    "element_type": "header",
                    "parent_id": current_parent,
                    "content_preview": header_text,
                    "content_location": json.dumps({
                        "source": source_id,  # Now using fully qualified path
                        "type": "header",
                        "text": header_text
                    }),
                    "content_hash": self._generate_hash(header_text),
                    "metadata": {
                        "level": level,
                        "text": header_text,
                        "full_path": source_id  # Store the full path in metadata
                    }
                }

                elements.append(header_element)

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
                    "element_type": "paragraph",
                    "parent_id": current_parent,
                    "content_preview": para_text,
                    "content_location": json.dumps({
                        "source": source_id,  # Now using fully qualified path
                        "type": "paragraph",
                        "text": para_text[:20]  # Enough to identify but not full content
                    }),
                    "content_hash": self._generate_hash(para_text),
                    "metadata": {
                        "length": len(para_text),
                        "full_path": source_id  # Store the full path in metadata
                    }
                }

                elements.append(para_element)

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
                    "element_type": "list",
                    "parent_id": current_parent,
                    "content_preview": f"{list_type.capitalize()} list",
                    "content_location": json.dumps({
                        "source": source_id,  # Now using fully qualified path
                        "type": "list",
                        "list_type": list_type
                    }),
                    "content_hash": self._generate_hash(tag.get_text()),
                    "metadata": {
                        "list_type": list_type,
                        "full_path": source_id  # Store the full path in metadata
                    }
                }

                elements.append(list_element)

                # Process list items
                for i, item in enumerate(tag.find_all('li', recursive=False)):
                    item_text = item.get_text().strip()
                    item_id = self._generate_id("item_")

                    item_element = {
                        "element_id": item_id,
                        "doc_id": doc_id,
                        "element_type": "list_item",
                        "parent_id": list_id,
                        "content_preview": item_text,
                        "content_location": json.dumps({
                            "source": source_id,  # Now using fully qualified path
                            "type": "list_item",
                            "list_type": list_type,
                            "index": i
                        }),
                        "content_hash": self._generate_hash(item_text),
                        "metadata": {
                            "index": i,
                            "full_path": source_id  # Store the full path in metadata
                        }
                    }

                    elements.append(item_element)

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
                    "element_type": "code_block",
                    "parent_id": current_parent,
                    "content_preview": code_text,
                    "content_location": json.dumps({
                        "source": source_id,  # Now using fully qualified path
                        "type": "code_block",
                        "language": language
                    }),
                    "content_hash": self._generate_hash(code_text),
                    "metadata": {
                        "language": language,
                        "full_path": source_id  # Store the full path in metadata
                    }
                }

                elements.append(code_element)

            elif tag.name == 'blockquote':
                # Blockquote element
                quote_text = tag.get_text().strip()
                element_id = self._generate_id("quote_")

                quote_element = {
                    "element_id": element_id,
                    "doc_id": doc_id,
                    "element_type": "blockquote",
                    "parent_id": current_parent,
                    "content_preview": quote_text,
                    "content_location": json.dumps({
                        "source": source_id,  # Now using fully qualified path
                        "type": "blockquote"
                    }),
                    "content_hash": self._generate_hash(quote_text),
                    "metadata": {
                        "full_path": source_id  # Store the full path in metadata
                    }
                }

                elements.append(quote_element)

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
                    "element_type": "table",
                    "parent_id": current_parent,
                    "content_preview": "Table",
                    "content_location": json.dumps({
                        "source": source_id,  # Now using fully qualified path
                        "type": "table"
                    }),
                    "content_hash": self._generate_hash(table_html),
                    "metadata": {
                        "rows": len(tag.find_all('tr')),
                        "has_header": bool(tag.find('thead') or tag.find('th')),
                        "full_path": source_id  # Store the full path in metadata
                    }
                }

                elements.append(table_element)

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
                            "element_type": "table_header",
                            "parent_id": table_id,
                            "content_preview": cell_text,
                            "content_location": json.dumps({
                                "source": source_id,  # Now using fully qualified path
                                "type": "table_header",
                                "col": i
                            }),
                            "content_hash": self._generate_hash(cell_text),
                            "metadata": {
                                "col": i,
                                "full_path": source_id  # Store the full path in metadata
                            }
                        }

                        elements.append(cell_element)

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
                        "element_type": "table_row",
                        "parent_id": table_id,
                        "content_preview": f"Row {i + 1}",
                        "content_location": json.dumps({
                            "source": source_id,  # Now using fully qualified path
                            "type": "table_row",
                            "row": i
                        }),
                        "content_hash": self._generate_hash(str(row)),
                        "metadata": {
                            "row": i,
                            "full_path": source_id
                        }
                    }

                    elements.append(row_element)

                    # Process cells
                    for j, cell in enumerate(row.find_all(['td', 'th'])):
                        cell_text = cell.get_text().strip()
                        cell_id = self._generate_id("td_")

                        cell_element = {
                            "element_id": cell_id,
                            "doc_id": doc_id,
                            "element_type": "table_cell",
                            "parent_id": row_id,
                            "content_preview": cell_text,
                            "content_location": json.dumps({
                                "source": source_id,
                                "type": "table_cell",
                                "row": i,
                                "col": j
                            }),
                            "content_hash": self._generate_hash(cell_text),
                            "metadata": {
                                "row": i,
                                "col": j,
                                "full_path": source_id
                            }
                        }

                        elements.append(cell_element)

                        # Extract links from cell
                        for a in cell.find_all('a', href=True):
                            links.append({
                                "source_id": cell_id,
                                "link_text": a.get_text().strip(),
                                "link_target": a['href'],
                                "link_type": "html"
                            })

        return elements, links

    def _extract_links(self, content: str, element_id: str) -> List[Dict[str, Any]]:
        """Override base method to extract Markdown links."""
        return self._extract_markdown_links(content, element_id)

    @staticmethod
    def _extract_paragraph(content: str, para_text: str) -> str:
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

    @staticmethod
    def _extract_list_item(content: str, index: int, list_type: str) -> str:
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

    @staticmethod
    def _extract_code_block(content: str, language: str) -> str:
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

    @staticmethod
    def _extract_blockquote(content: str) -> str:
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

    @staticmethod
    def _extract_table_element(content: str, element_type: str, row: int, col: int) -> str:
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
