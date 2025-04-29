"""
HTML document parser module for the document pointer system.

This module parses HTML documents into structured elements.
"""

import json
import logging
from typing import Dict, Any, Optional, List

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

        # Create element with common fields
        element = {
            "element_id": element_id,
            "doc_id": doc_id,
            "element_type": element_type,
            "parent_id": parent_id,
            "content_preview": content_text,
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
        elif tag.name == 'img':
            element["metadata"]["src"] = tag.get('src', '')
            element["metadata"]["alt"] = tag.get('alt', '')
            element["metadata"]["width"] = tag.get('width', '')
            element["metadata"]["height"] = tag.get('height', '')
            element["content_preview"] = tag.get('alt', 'Image')
        elif tag.name == 'pre' or tag.name == 'code':
            language = ""
            if tag.name == 'code' and tag.has_attr('class'):
                for cls in tag['class']:
                    if cls.startswith('language-'):
                        language = cls[9:]
                        break
            element["metadata"]["language"] = language

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

    def _parse_element(self, tag, doc_id, parent_id, source_id):
        """Process a single HTML element based on its type."""
        elements = []
        links = []

        # Get the text content for logging
        content_preview = tag.get_text().strip()[:50]
        logger.info(f"Processing {tag.name} element with content: {content_preview}...")

        try:
            if tag.name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                # Header processing
                level = int(tag.name[1])
                header_text = tag.get_text().strip()

                element_id = self._generate_id(f"header{level}_")

                header_element = {
                    "element_id": element_id,
                    "doc_id": doc_id,
                    "element_type": "header",
                    "parent_id": parent_id,
                    "content_preview": header_text,
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "header",
                        "text": header_text,
                        "selector": tag.get('_selector', '')
                    }),
                    "content_hash": self._generate_hash(header_text),
                    "metadata": {
                        "level": level,
                        "text": header_text,
                        "id": tag.get('id', ''),
                        "full_path": source_id
                    }
                }

                elements.append(header_element)
                tag._element_id = element_id

                # Extract links from header
                for a in tag.find_all('a', href=True):
                    links.append({
                        "source_id": element_id,
                        "link_text": a.get_text().strip(),
                        "link_target": a['href'],
                        "link_type": "html"
                    })

            elif tag.name == 'p':
                # Paragraph processing
                para_text = tag.get_text().strip()
                if not para_text:
                    return elements, links

                element_id = self._generate_id("para_")

                para_element = {
                    "element_id": element_id,
                    "doc_id": doc_id,
                    "element_type": "paragraph",
                    "parent_id": parent_id,
                    "content_preview": para_text,
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "paragraph",
                        "selector": tag.get('_selector', '')
                    }),
                    "content_hash": self._generate_hash(para_text),
                    "metadata": {
                        "length": len(para_text),
                        "class": tag.get('class', ''),
                        "full_path": source_id
                    }
                }

                elements.append(para_element)
                tag._element_id = element_id

                # Extract links from paragraph
                for a in tag.find_all('a', href=True):
                    links.append({
                        "source_id": element_id,
                        "link_text": a.get_text().strip(),
                        "link_target": a['href'],
                        "link_type": "html"
                    })

            elif tag.name in ('ul', 'ol'):
                # List processing
                list_type = 'ordered' if tag.name == 'ol' else 'unordered'
                list_id = self._generate_id("list_")

                list_text = tag.get_text().strip()
                if not list_text:
                    return elements, links

                logger.info(f"Processing {list_type} list")

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
                    "content_hash": self._generate_hash(list_text),
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
                        "content_preview": item_text,
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

            elif tag.name == 'table':
                # Table processing
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

                # Process the rest of the table (rows, cells, etc.)
                # ... [table processing code similar to previous implementation]

            elif tag.name == 'img':
                # Image processing
                img_id = self._generate_id("img_")
                alt_text = tag.get('alt', '')
                src = tag.get('src', '')

                img_element = {
                    "element_id": img_id,
                    "doc_id": doc_id,
                    "element_type": "image",
                    "parent_id": parent_id,
                    "content_preview": alt_text or "Image",
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "image",
                        "src": src,
                        "selector": tag.get('_selector', '')
                    }),
                    "content_hash": self._generate_hash(src + alt_text),
                    "metadata": {
                        "alt": alt_text,
                        "src": src,
                        "width": tag.get('width', ''),
                        "height": tag.get('height', ''),
                        "full_path": source_id
                    }
                }

                elements.append(img_element)
                tag._element_id = img_id

            elif tag.name == 'pre' or tag.name == 'code':
                # Code block processing
                code_text = tag.get_text().strip()
                language = ""

                if tag.name == 'code' and tag.has_attr('class'):
                    for cls in tag['class']:
                        if cls.startswith('language-'):
                            language = cls[9:]
                            break

                element_id = self._generate_id("code_")

                code_element = {
                    "element_id": element_id,
                    "doc_id": doc_id,
                    "element_type": "code_block",
                    "parent_id": parent_id,
                    "content_preview": code_text,
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "code_block",
                        "language": language,
                        "selector": tag.get('_selector', '')
                    }),
                    "content_hash": self._generate_hash(code_text),
                    "metadata": {
                        "language": language,
                        "full_path": source_id
                    }
                }

                elements.append(code_element)
                tag._element_id = element_id

            elif tag.name == 'blockquote':
                # Blockquote processing
                quote_text = tag.get_text().strip()
                element_id = self._generate_id("quote_")

                quote_element = {
                    "element_id": element_id,
                    "doc_id": doc_id,
                    "element_type": "blockquote",
                    "parent_id": parent_id,
                    "content_preview": quote_text,
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "blockquote",
                        "selector": tag.get('_selector', '')
                    }),
                    "content_hash": self._generate_hash(quote_text),
                    "metadata": {
                        "full_path": source_id
                    }
                }

                elements.append(quote_element)
                tag._element_id = element_id

                # Extract links from blockquote
                for a in tag.find_all('a', href=True):
                    links.append({
                        "source_id": element_id,
                        "link_text": a.get_text().strip(),
                        "link_target": a['href'],
                        "link_type": "html"
                    })

            logger.info(f"Successfully processed {tag.name} element, created {len(elements)} elements")
        except Exception as e:
            logger.error(f"Error processing {tag.name} element: {str(e)}")

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
                "content_preview": item_text,
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
                    "content_preview": cell_text,
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
                    "content_preview": cell_text,
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
