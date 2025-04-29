"""
HTML document parser module for the document pointer system.

This module parses HTML documents into structured elements.
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List, Union

from bs4 import BeautifulSoup

from .base import DocumentParser

logger = logging.getLogger(__name__)


class HtmlParser(DocumentParser):
    """Parser for HTML documents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the HTML parser."""
        super().__init__(config)
        # Define HTML-specific link patterns
        self.link_patterns = [
            r'<a\s+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>'  # HTML links
        ]
        self.max_content_preview = self.config.get("max_content_preview", 100)

    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse an HTML document into structured elements."""
        content = doc_content["content"]
        source_id = doc_content["id"]  # Should already be a fully qualified path
        metadata = doc_content.get("metadata", {}).copy()  # Make a copy to avoid modifying original

        # Generate document ID if not present
        doc_id = metadata.get("doc_id", self._generate_id("doc_"))

        # Create document record
        document = {
            "doc_id": doc_id,
            "doc_type": "html",
            "source": source_id,
            "metadata": metadata,
            "content_hash": doc_content.get("content_hash", self._generate_hash(content))
        }

        # Create root element
        elements: List = [self._create_root_element(doc_id, source_id)]
        root_id = elements[0]["element_id"]

        # Parse HTML
        soup = BeautifulSoup(content, 'html.parser')

        # Add CSS selectors to elements for better location tracking
        self._add_selectors(soup)

        # Extract links directly from HTML
        extracted_links = []
        for a in soup.find_all('a', href=True):
            extracted_links.append({
                "source_id": root_id,  # Initially assign to root, will update later
                "link_text": a.get_text().strip(),
                "link_target": a['href'],
                "link_type": "html"
            })

        # Parse HTML elements
        parsed_elements, element_links = self._parse_document(soup, doc_id, root_id, source_id)
        elements.extend(parsed_elements)

        # Update link source_ids with the correct element IDs
        self._update_link_sources(extracted_links, parsed_elements)
        extracted_links.extend(element_links)

        # Return the parsed document with extracted links
        return {
            "document": document,
            "elements": elements,
            "links": extracted_links,
            "relationships": []
        }

    def _parse_document(self, soup, doc_id, parent_id, source_id):
        """Parse the entire document in a unified way."""
        elements = []
        links = []

        # Create a map to track element IDs by tag reference
        element_id_map = {}

        # Start with the body if it exists
        if soup.body:
            # Process the body element first
            body_element = self._create_element_for_tag(soup.body, doc_id, parent_id, source_id)
            if body_element:
                elements.append(body_element)
                element_id_map[soup.body] = body_element["element_id"]
                body_id = body_element["element_id"]
            else:
                body_id = parent_id

            # Use a breadth-first approach to process children
            self._process_tag_children(soup.body, doc_id, body_id, source_id, elements, links, element_id_map)

        return elements, links

    def _process_tag_children(self, parent_tag, doc_id, parent_id, source_id, elements, links, element_id_map):
        """Process all children of a tag."""
        # Get direct children
        for child in parent_tag.children:
            # Skip text nodes and other non-element nodes
            if not hasattr(child, 'name') or not child.name:
                continue

            # Create an element for this tag
            if child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol',
                              'pre', 'code', 'blockquote', 'table', 'img', 'div',
                              'article', 'section', 'nav', 'aside', 'figure']:

                # Create an element
                element = self._create_element_for_tag(child, doc_id, parent_id, source_id)

                if element:
                    elements.append(element)
                    element_id = element["element_id"]
                    element_id_map[child] = element_id

                    # Extract links from this element
                    for a in child.find_all('a', href=True):
                        links.append({
                            "source_id": element_id,
                            "link_text": a.get_text().strip(),
                            "link_target": a['href'],
                            "link_type": "html"
                        })

                    # Special handling for specific element types
                    if child.name == 'table':
                        table_elements, table_links = self._process_table(child, doc_id, element_id, source_id)
                        elements.extend(table_elements)
                        links.extend(table_links)
                    elif child.name in ['ul', 'ol']:
                        list_elements, list_links = self._process_list(child, doc_id, element_id, source_id)
                        elements.extend(list_elements)
                        links.extend(list_links)

                    # Process this tag's children recursively
                    self._process_tag_children(child, doc_id, element_id, source_id, elements, links, element_id_map)
                else:
                    # If no element was created, still process children with parent_id
                    self._process_tag_children(child, doc_id, parent_id, source_id, elements, links, element_id_map)
            else:
                # For non-content tags, just process their children with the same parent_id
                self._process_tag_children(child, doc_id, parent_id, source_id, elements, links, element_id_map)

    def _create_element_for_tag(self, tag, doc_id, parent_id, source_id):
        """Create an appropriate element based on tag type."""
        element_type = self._get_element_type(tag.name)
        content_text = tag.get_text().strip()

        # Skip empty elements
        if not content_text and tag.name not in ['img', 'table']:
            return None

        element_id = self._generate_id(f"{element_type}_")

        # Create content preview
        if len(content_text) > self.max_content_preview:
            content_preview = content_text[:self.max_content_preview] + "..."
        else:
            content_preview = content_text

        # Create element with common fields
        element = {
            "element_id": element_id,
            "doc_id": doc_id,
            "element_type": element_type,
            "parent_id": parent_id,
            "content_preview": content_preview,
            "content_location": json.dumps({
                "source": source_id,
                "type": element_type,
                "selector": tag.get('_selector', '')
            }),
            "content_hash": self._generate_hash(str(tag)),
            "metadata": {
                "id": tag.get('id', ''),
                "class": tag.get('class', ''),
                "full_path": source_id
            }
        }

        # Add element-specific metadata
        if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            element["metadata"]["level"] = int(tag.name[1])
            element["metadata"]["text"] = content_text
            element["content_location"] = json.dumps({
                "source": source_id,
                "type": element_type,
                "selector": tag.get('_selector', ''),
                "level": int(tag.name[1]),
                "text": content_text[:50] if len(content_text) > 50 else content_text
            })
        elif tag.name == 'img':
            element["metadata"]["src"] = tag.get('src', '')
            element["metadata"]["alt"] = tag.get('alt', '')
            element["metadata"]["width"] = tag.get('width', '')
            element["metadata"]["height"] = tag.get('height', '')
            element["content_preview"] = tag.get('alt', 'Image')
            element["content_location"] = json.dumps({
                "source": source_id,
                "type": element_type,
                "selector": tag.get('_selector', ''),
                "src": tag.get('src', '')
            })
        elif tag.name == 'pre' or tag.name == 'code':
            language = ""
            if tag.name == 'code' and tag.has_attr('class'):
                for cls in tag['class']:
                    if cls.startswith('language-'):
                        language = cls[9:]
                        break
            element["metadata"]["language"] = language
            element["content_location"] = json.dumps({
                "source": source_id,
                "type": element_type,
                "selector": tag.get('_selector', ''),
                "language": language
            })

        # Store the element ID on the tag for reference
        tag._element_id = element_id

        return element

    @staticmethod
    def _get_element_type(tag_name):
        """Map HTML tag names to element types."""
        if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            return "header"
        elif tag_name == 'p':
            return "paragraph"
        elif tag_name in ['ul', 'ol']:
            return "list"
        elif tag_name == 'li':
            return "list_item"
        elif tag_name == 'table':
            return "table"
        elif tag_name == 'tr':
            return "table_row"
        elif tag_name == 'th':
            return "table_header"
        elif tag_name == 'td':
            return "table_cell"
        elif tag_name == 'img':
            return "image"
        elif tag_name in ['pre', 'code']:
            return "code_block"
        elif tag_name == 'blockquote':
            return "blockquote"
        else:
            # For container elements
            return tag_name  # Use the tag name as the element type (div, article, etc.)

    def _update_link_sources(self, links, elements):
        """Update link source IDs based on their position in the document."""
        # This would be a more sophisticated implementation that uses the
        # selectors or positions to determine which element contains each link
        # For now, we'll keep it simple and leave links assigned to the root
        pass

    def _process_list(self, tag, doc_id, parent_id, source_id):
        """Process a list element."""
        elements = []
        links = []

        list_type = 'ordered' if tag.name == 'ol' else 'unordered'
        list_id = self._generate_id("list_")

        list_element = {
            "element_id": list_id,
            "doc_id": doc_id,
            "element_type": "list",
            "parent_id": parent_id,
            "content_preview": f"{list_type.capitalize()} list",
            "content_location": json.dumps({
                "source": source_id,
                "type": "list",
                "list_type": list_type,
                "selector": tag.get('_selector', '')
            }),
            "content_hash": self._generate_hash(tag.get_text()),
            "metadata": {
                "list_type": list_type,
                "class": tag.get('class', ''),
                "full_path": source_id
            }
        }

        elements.append(list_element)
        tag._element_id = list_id

        # Process list items
        for i, item in enumerate(tag.find_all('li', recursive=False)):
            item_text = item.get_text().strip()
            if not item_text:
                continue

            item_id = self._generate_id("item_")

            item_element = {
                "element_id": item_id,
                "doc_id": doc_id,
                "element_type": "list_item",
                "parent_id": list_id,
                "content_preview": item_text[:self.max_content_preview] + (
                    "..." if len(item_text) > self.max_content_preview else ""),
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "list_item",
                    "list_type": list_type,
                    "index": i,
                    "selector": item.get('_selector', '')
                }),
                "content_hash": self._generate_hash(item_text),
                "metadata": {
                    "index": i,
                    "full_path": source_id
                }
            }

            elements.append(item_element)
            item._element_id = item_id

            # Extract links from list item
            for a in item.find_all('a', href=True):
                links.append({
                    "source_id": item_id,
                    "link_text": a.get_text().strip(),
                    "link_target": a['href'],
                    "link_type": "html"
                })

        return elements, links

    def _process_table(self, tag, doc_id, parent_id, source_id):
        """Process a table element."""
        elements = []
        links = []

        table_id = self._generate_id("table_")
        table_html = str(tag)

        table_element = {
            "element_id": table_id,
            "doc_id": doc_id,
            "element_type": "table",
            "parent_id": parent_id,
            "content_preview": "Table",
            "content_location": json.dumps({
                "source": source_id,
                "type": "table",
                "selector": tag.get('_selector', '')
            }),
            "content_hash": self._generate_hash(table_html),
            "metadata": {
                "rows": len(tag.find_all('tr')),
                "has_header": bool(tag.find('thead') or tag.find('th')),
                "id": tag.get('id', ''),
                "class": tag.get('class', ''),
                "full_path": source_id
            }
        }

        elements.append(table_element)
        tag._element_id = table_id

        # Process headers
        header_row = tag.find('thead')
        if header_row:
            header_cells = header_row.find_all('th')
            for i, cell in enumerate(header_cells):
                cell_text = cell.get_text().strip()
                if not cell_text:
                    continue

                cell_id = self._generate_id("th_")

                cell_element = {
                    "element_id": cell_id,
                    "doc_id": doc_id,
                    "element_type": "table_header",
                    "parent_id": table_id,
                    "content_preview": cell_text[:self.max_content_preview] + (
                        "..." if len(cell_text) > self.max_content_preview else ""),
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "table_header",
                        "col": i,
                        "selector": cell.get('_selector', '')
                    }),
                    "content_hash": self._generate_hash(cell_text),
                    "metadata": {
                        "col": i,
                        "full_path": source_id
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
                    "source": source_id,
                    "type": "table_row",
                    "row": i,
                    "selector": row.get('_selector', '')
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
                if not cell_text:
                    continue

                cell_id = self._generate_id("td_")

                cell_element = {
                    "element_id": cell_id,
                    "doc_id": doc_id,
                    "element_type": "table_cell",
                    "parent_id": row_id,
                    "content_preview": cell_text[:self.max_content_preview] + (
                        "..." if len(cell_text) > self.max_content_preview else ""),
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "table_cell",
                        "row": i,
                        "col": j,
                        "selector": cell.get('_selector', '')
                    }),
                    "content_hash": self._generate_hash(cell_text),
                    "metadata": {
                        "row": i,
                        "col": j,
                        "colspan": cell.get('colspan', '1'),
                        "rowspan": cell.get('rowspan', '1'),
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

    def _add_selectors(self, element, parent_selector=""):
        """
        Add CSS selectors to elements for location.

        Args:
            element: BeautifulSoup element
            parent_selector: Parent's selector
        """
        if not hasattr(element, 'name') or not element.name:
            return

        # Build selector for this element
        if element.name == 'body':
            selector = 'body'
        else:
            tag_selector = element.name

            # Add ID if present
            if element.get('id'):
                id_selector = f"#{element.get('id')}"
                tag_selector = f"{tag_selector}{id_selector}"

            # Add first class if present
            elif element.get('class'):
                class_selector = f".{element.get('class')[0]}"
                tag_selector = f"{tag_selector}{class_selector}"

            # Combine with parent selector
            if parent_selector:
                selector = f"{parent_selector} > {tag_selector}"
            else:
                selector = tag_selector

        # Store selector on element
        element['_selector'] = selector

        # Process children
        for child in element.children:
            if hasattr(child, 'name') and child.name:
                self._add_selectors(child, selector)

    def _resolve_element_content(self, location_data: Dict[str, Any],
                                 source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve content for specific HTML element types.

        Args:
            location_data: Content location data
            source_content: Optional pre-loaded source content

        Returns:
            Resolved content string (HTML format)
        """
        source = location_data.get("source", "")
        element_type = location_data.get("type", "")
        selector = location_data.get("selector", "")

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
                    raise ValueError(f"Error reading HTML file: {str(e)}")
            else:
                raise ValueError(f"Source file not found: {source}")

        # Ensure content is string (not bytes)
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    content = content.decode('latin1')
                except UnicodeDecodeError:
                    raise ValueError("Cannot decode binary content as HTML")

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')

        # If a CSS selector is provided, use it
        if selector:
            elements = soup.select(selector)
            if elements:
                # Always return HTML structure
                return str(elements[0])
            return ""

        # Handle element type-specific content
        if element_type == "header":
            # Extract header by level and/or text
            header_level = location_data.get("level")
            header_text = location_data.get("text", "")

            # Find header by level and text
            if header_level:
                headers = soup.find_all(f'h{header_level}')

                if header_text:
                    # Find header with matching text
                    for header in headers:
                        if header_text in header.get_text():
                            return str(header)

                # If no text match but we have headers at this level, return the first one
                if headers:
                    return str(headers[0])

            # If no level specified, search all header levels
            if header_text:
                for level in range(1, 7):
                    headers = soup.find_all(f'h{level}')
                    for header in headers:
                        if header_text in header.get_text():
                            return str(header)

            return ""

        elif element_type == "paragraph":
            # Extract paragraph by text or index
            para_text = location_data.get("text", "")
            para_index = location_data.get("index", 0)

            paragraphs = soup.find_all('p')

            if para_text:
                # Find paragraph with matching text
                for para in paragraphs:
                    if para_text in para.get_text():
                        return str(para)

            # Return paragraph by index
            if 0 <= para_index < len(paragraphs):
                return str(paragraphs[para_index])

            return ""

        elif element_type == "list" or element_type == "list_item":
            # Extract list or list item
            list_type = location_data.get("list_type", "")
            list_tag = 'ol' if list_type == 'ordered' else 'ul'
            index = location_data.get("index", 0)

            lists = soup.find_all(list_tag)

            if lists:
                if element_type == "list":
                    # Return the whole list
                    return str(lists[0])
                else:
                    # Return specific list item
                    items = lists[0].find_all('li')
                    if 0 <= index < len(items):
                        return str(items[index])

            return ""

        elif element_type in ["table", "table_row", "table_cell", "table_header"]:
            # Extract table element
            table_index = location_data.get("table_index", 0)
            row = location_data.get("row", 0)
            col = location_data.get("col", 0)

            tables = soup.find_all('table')

            if not tables or table_index >= len(tables):
                return ""

            table = tables[table_index]

            if element_type == "table":
                # Return the whole table
                return str(table)

            # Get rows
            rows = table.find_all('tr')
            if row >= len(rows):
                return ""

            if element_type == "table_row":
                # Return the whole row
                return str(rows[row])

            # Get cells
            cells = rows[row].find_all(['td', 'th'])
            if col >= len(cells):
                return ""

            # Return specific cell
            return str(cells[col])

        elif element_type == "image":
            # Extract image information
            src = location_data.get("src", "")

            images = soup.find_all('img')
            for img in images:
                if src and src == img.get('src'):
                    # Return the img tag as string
                    return str(img)

            return ""

        elif element_type == "code_block":
            # Extract code block
            language = location_data.get("language", "")

            code_blocks = soup.find_all('pre')
            for block in code_blocks:
                code_tag = block.find('code')
                if code_tag and language:
                    # Check for language in class
                    classes = code_tag.get('class', [])
                    for cls in classes:
                        if cls.startswith('language-') and cls[9:] == language:
                            return str(code_tag)
                elif code_tag:
                    # Return first code block if no language specified
                    return str(code_tag)
                else:
                    # Return pre tag content if no code tag inside
                    return str(block)

            return ""

        elif element_type == "blockquote":
            # Extract blockquote
            blockquotes = soup.find_all('blockquote')
            if blockquotes:
                return str(blockquotes[0])

            return ""

        else:
            # For other element types or if no specific handler,
            # return full document HTML
            return str(soup)

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

            # Check file extension for HTML
            _, ext = os.path.splitext(source.lower())
            return ext in ['.html', '.htm', '.xhtml']

        except (json.JSONDecodeError, TypeError):
            return False

    def _extract_links(self, content: str, element_id: str) -> List[Dict[str, Any]]:
        """
        Extract links from HTML content.

        Args:
            content: HTML content
            element_id: ID of the element containing the links

        Returns:
            List of extracted links
        """
        links = []

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')

        # Extract all links
        for a in soup.find_all('a', href=True):
            links.append({
                "source_id": element_id,
                "link_text": a.get_text().strip(),
                "link_target": a['href'],
                "link_type": "html"
            })

        return links
