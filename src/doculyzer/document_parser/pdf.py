"""
PDF document parser module for the document pointer system.

This module parses PDF documents into structured elements.
"""

import hashlib
import json
import logging
import os
import re
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union

from ..relationships import RelationshipType

try:
    # noinspection PyPackageRequirements
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. Install with 'pip install pymupdf' to use PDF parser")

from .base import DocumentParser

logger = logging.getLogger(__name__)


class PdfParser(DocumentParser):
    """Parser for PDF documents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the PDF parser."""
        super().__init__(config)

        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required for PDF parsing")

        # Configuration options
        self.config = config or {}
        self.extract_images = self.config.get("extract_images", False)
        self.extract_annotations = self.config.get("extract_annotations", True)
        self.extract_links = self.config.get("extract_links", True)
        self.extract_tables = self.config.get("extract_tables", True)
        self.detect_headers = self.config.get("detect_headers", True)
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.max_pages = self.config.get("max_pages", 1000)  # Limit for large documents
        self.min_header_font_size = self.config.get("min_header_font_size", 12)  # Minimum font size for headers
        self.temp_dir = self.config.get("temp_dir", os.path.join(os.path.dirname(__file__), 'temp'))
        self.max_content_preview = self.config.get("max_content_preview", 100)

        # Table detection parameters
        self.table_detection_method = self.config.get("table_detection_method",
                                                      "heuristic")  # Options: "heuristic", "ml"
        self.min_table_rows = self.config.get("min_table_rows", 2)
        self.min_table_cols = self.config.get("min_table_cols", 2)

    def _resolve_element_text(self, location_data: Dict[str, Any], source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve the plain text representation of a PDF element.

        Args:
            location_data: Content location data
            source_content: Optional preloaded source content

        Returns:
            Plain text representation of the element
        """
        # Get the content using the existing method
        content = self._resolve_element_content(location_data, source_content)
        element_type = location_data.get("type", "")

        # Handle specific element types
        if element_type == "content":
            # For the document content, just return a simple summary
            return content.strip()

        elif element_type == "page":
            # For full pages, the content is already just text
            return content.strip()

        elif element_type == "paragraph" or element_type == "text_block":
            # For paragraphs, just return the text content directly
            return content.strip()

        elif element_type == "header":
            # For headers, return the text without any formatting
            return content.strip()

        elif element_type == "table":
            # For tables, preserve the tabular structure but remove any PDF-specific formatting
            # The _resolve_element_content method already returns tabular text
            return content.strip()

        elif element_type == "annotation":
            # For annotations, return just the annotation text without metadata
            return content.strip()

        elif element_type == "image":
            # For images, extract any descriptive info if available
            if "Alt text:" in content:
                return content.split("Alt text:", 1)[1].strip()
            return "Image"

        elif element_type == "section":
            # For sections, return the section text
            return content.strip()

        # Default: return the content as is
        return content.strip()

    def _resolve_element_content(self, location_data: Dict[str, Any],
                                 source_content: Optional[Union[str, bytes]] = None) -> str:
        """
        Resolve content for specific PDF element types.

        Args:
            location_data: Content location data
            source_content: Optional preloaded source content

        Returns:
            Resolved content string
        """
        source = location_data.get("source", "")
        element_type = location_data.get("type", "")
        page_num = location_data.get("page", 1)

        # Load the document if source content is not provided
        doc = None
        temp_file = None
        try:
            if source_content is None:
                # Check if source is a file path
                if os.path.exists(source):
                    try:
                        doc = fitz.open(source)
                    except Exception as e:
                        raise ValueError(f"Error loading PDF document: {str(e)}")
                else:
                    raise ValueError(f"Source file not found: {source}")
            else:
                # Save content to a temporary file
                if not os.path.exists(self.temp_dir):
                    os.makedirs(self.temp_dir, exist_ok=True)

                import uuid
                temp_file = os.path.join(self.temp_dir, f"temp_{uuid.uuid4().hex}.pdf")
                with open(temp_file, 'wb') as f:
                    if isinstance(source_content, str):
                        f.write(source_content.encode('utf-8'))
                    else:
                        f.write(source_content)

                # Load the document
                try:
                    doc = fitz.open(temp_file)
                except Exception as e:
                    raise ValueError(f"Error loading PDF document: {str(e)}")

            # Check if page number is valid
            if 1 <= page_num <= len(doc):
                page = doc[page_num - 1]
            else:
                return f"Invalid page number: {page_num}. Document has {len(doc)} pages."

            # Handle different element types
            if element_type == "content":
                # Return basic document information
                return f"PDF document with {len(doc)} pages"

            elif element_type == "page":
                # Return the entire page text
                return page.get_text()

            elif element_type in ["paragraph", "header", "text_block"]:
                # Extract text from a specific bounding box
                bbox = location_data.get("bbox")
                if not bbox:
                    return "No bounding box specified"

                # Extract text from the bounding box
                text = page.get_text("text", clip=fitz.Rect(bbox))
                return text

            elif element_type == "table":
                # Extract table content
                bbox = location_data.get("bbox")
                if not bbox:
                    return "No bounding box specified for table"

                # Extract text and attempt to format as tabular data
                table_text = page.get_text("text", clip=fitz.Rect(bbox))

                # Simple formatting to preserve table structure (basic approach)
                return table_text

            elif element_type == "annotation":
                # Extract annotation content
                bbox = location_data.get("bbox")
                if not bbox:
                    return "No bounding box specified for annotation"

                # Look for annotations in the bounding box
                for annot in page.annots():
                    if self._rectangles_overlap(annot.rect, bbox):
                        return annot.info.get("content", "No annotation content")

                return "No annotation found in the specified area"

            elif element_type == "image":
                # Return information about the image
                xref = location_data.get("xref")
                if not xref:
                    return "No image reference specified"

                # Get basic image info
                for img in page.get_images(full=True):
                    if img[0] == xref:
                        width = img[2]
                        height = img[3]
                        return f"Image (xref: {xref}, dimensions: {width}x{height})"

                return f"Image with reference {xref} not found"

            elif element_type == "section":
                # Return content of a section
                bbox = location_data.get("bbox")
                if not bbox:
                    return "No bounding box specified for section"

                # Extract text from the section's bounding box
                section_text = page.get_text("text", clip=fitz.Rect(bbox))
                return section_text

            else:
                # For other element types or if no specific handler,
                # return page text
                return page.get_text()

        finally:
            # Clean up resources
            if doc:
                doc.close()

            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file}: {str(e)}")

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

            # Check file extension for PDF
            _, ext = os.path.splitext(source.lower())
            return ext == '.pdf'

        except (json.JSONDecodeError, TypeError):
            return False

    def _extract_links(self, content: str, element_id: str) -> List[Dict[str, Any]]:
        """
        Base method for extracting links from content.

        Args:
            content: Text content
            element_id: ID of the element containing the links

        Returns:
            List of extracted link dictionaries
        """
        links = []

        # Extract URLs using a regular expression
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        urls = re.findall(url_pattern, content)

        # Create link entries for each URL found
        for url in urls:
            links.append({
                "source_id": element_id,
                "link_text": url,
                "link_target": url,
                "link_type": "uri"
            })

        return links

    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a PDF document into structured elements.

        Args:
            doc_content: Document content and metadata

        Returns:
            Dictionary with document metadata, elements, relationships, and extracted links
        """
        # Extract metadata from doc_content
        source_id = doc_content["id"]
        metadata = doc_content.get("metadata", {}).copy()  # Make a copy to avoid modifying original

        # Check if we have binary content or a path to a file
        binary_path = doc_content.get("binary_path")
        if not binary_path:
            # If we have binary content but no path, we need to save it to a temp file
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir, exist_ok=True)

            binary_content = doc_content.get("content", b"")
            if isinstance(binary_content, str):
                logger.warning("Expected binary content for PDF but got string. Attempting to process anyway.")

            temp_file_path = os.path.join(self.temp_dir, f"temp_{uuid.uuid4().hex}.pdf")
            with open(temp_file_path, 'wb') as f:
                if isinstance(binary_content, str):
                    f.write(binary_content.encode('utf-8'))
                else:
                    f.write(binary_content)

            binary_path = temp_file_path
            logger.debug(f"Saved binary content to temporary file: {binary_path}")

        # Generate document ID if not present
        doc_id = metadata.get("doc_id", self._generate_id("doc_"))

        # Load PDF document
        try:
            doc = fitz.open(binary_path)
        except Exception as e:
            logger.error(f"Error loading PDF document: {str(e)}")
            raise

        # Create document record with metadata
        document = {
            "doc_id": doc_id,
            "doc_type": "pdf",
            "source": source_id,
            "metadata": self._extract_document_metadata(doc, metadata),
            "content_hash": doc_content.get("content_hash", "")
        }

        # Create root element
        elements = [self._create_root_element(doc_id, source_id)]
        root_id = elements[0]["element_id"]

        # Initialize relationships list
        relationships = []

        # Parse document elements and create relationships
        page_elements, page_relationships = self._parse_document(doc, doc_id, root_id, source_id)
        elements.extend(page_elements)
        relationships.extend(page_relationships)

        # Extract links from the document using the helper method
        links = self._extract_document_links(doc, elements)

        # Clean up temporary file if needed
        if binary_path != doc_content.get("binary_path") and os.path.exists(binary_path):
            try:
                doc.close()
                os.remove(binary_path)
                logger.debug(f"Deleted temporary file: {binary_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {binary_path}: {str(e)}")

        # Return the parsed document with extracted links and relationships
        return {
            "document": document,
            "elements": elements,
            "links": links,
            "relationships": relationships
        }

    def _extract_document_metadata(self, doc: fitz.Document, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from PDF document.

        Args:
            doc: The PDF document
            base_metadata: Base metadata from content source

        Returns:
            Dictionary of document metadata
        """
        # Combine base metadata with document properties
        metadata = base_metadata.copy()

        try:
            if self.extract_metadata:
                # Get document metadata
                pdf_metadata = doc.metadata

                # Add core properties to metadata
                if pdf_metadata.get("title"):
                    metadata["title"] = pdf_metadata["title"]
                if pdf_metadata.get("author"):
                    metadata["author"] = pdf_metadata["author"]
                if pdf_metadata.get("subject"):
                    metadata["subject"] = pdf_metadata["subject"]
                if pdf_metadata.get("keywords"):
                    metadata["keywords"] = pdf_metadata["keywords"]
                if pdf_metadata.get("creator"):
                    metadata["creator"] = pdf_metadata["creator"]
                if pdf_metadata.get("producer"):
                    metadata["producer"] = pdf_metadata["producer"]
                if pdf_metadata.get("creationDate"):
                    metadata["creation_date"] = pdf_metadata["creationDate"]
                if pdf_metadata.get("modDate"):
                    metadata["modification_date"] = pdf_metadata["modDate"]

            # Add document statistics
            page_count = min(len(doc), self.max_pages)
            metadata["page_count"] = page_count

            # Add basic layout information
            if page_count > 0:
                first_page = doc[0]
                metadata["page_width"] = first_page.rect.width
                metadata["page_height"] = first_page.rect.height
                metadata["is_encrypted"] = doc.is_encrypted
                metadata["permissions"] = doc.permissions if hasattr(doc, "permissions") else None

        except Exception as e:
            logger.warning(f"Error extracting document metadata: {str(e)}")

        return metadata

    def _parse_document(self, doc: fitz.Document, doc_id: str, parent_id: str, source_id: str) -> tuple[
        List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Parse PDF document into structured elements.

        Args:
            doc: The PDF document
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier

        Returns:
            Tuple of (list of elements, list of relationships)
        """
        elements = []
        relationships = []

        # Create and add a content element to hold all pages
        content_id = self._generate_id("content_")
        content_element = {
            "element_id": content_id,
            "doc_id": doc_id,
            "element_type": "content",
            "parent_id": parent_id,
            "content_preview": f"PDF content with {min(len(doc), self.max_pages)} pages",
            "content_location": json.dumps({
                "source": source_id,
                "type": "content"
            }),
            "content_hash": "",
            "metadata": {
                "page_count": min(len(doc), self.max_pages)
            }
        }
        elements.append(content_element)

        # Create relationship from root to content
        contains_relationship = {
            "relationship_id": self._generate_id("rel_"),
            "source_id": parent_id,
            "target_id": content_id,
            "relationship_type": RelationshipType.CONTAINS.value,
            "metadata": {
                "confidence": 1.0
            }
        }
        relationships.append(contains_relationship)

        # Create inverse relationship
        contained_by_relationship = {
            "relationship_id": self._generate_id("rel_"),
            "source_id": content_id,
            "target_id": parent_id,
            "relationship_type": RelationshipType.CONTAINED_BY.value,
            "metadata": {
                "confidence": 1.0
            }
        }
        relationships.append(contained_by_relationship)

        # Process each page up to max_pages
        for page_idx, page in enumerate(doc):
            if page_idx >= self.max_pages:
                break

            # Process the page
            page_elements, page_relationships = self._process_page(doc, page, page_idx, doc_id, content_id, source_id)
            elements.extend(page_elements)
            relationships.extend(page_relationships)

        return elements, relationships

    def _process_page(self, _doc: fitz.Document, page: fitz.Page, page_idx: int, doc_id: str, parent_id: str,
                      source_id: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process a single PDF page.

        Args:
            _doc: The PDF document
            page: The page to process
            page_idx: Page index
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier

        Returns:
            Tuple of (list of page elements, list of relationships)
        """
        elements = []
        relationships = []

        # Create page element
        page_id = self._generate_id(f"page_{page_idx + 1}_")

        # Extract basic page information
        page_text = page.get_text()
        page_number = page_idx + 1
        width, height = page.rect.width, page.rect.height

        # Create page preview
        preview = f"Page {page_number} ({int(width)}x{int(height)} pts)"
        if len(page_text) > 0:
            # Add a snippet of text to the preview
            text_preview = ' '.join(page_text.split()[:20])
            if text_preview:
                if len(text_preview) > 100:
                    text_preview = text_preview[:97] + "..."
                preview += f": {text_preview}"

        # Create page element
        page_element = {
            "element_id": page_id,
            "doc_id": doc_id,
            "element_type": "page",
            "parent_id": parent_id,
            "content_preview": preview,
            "content_location": json.dumps({
                "source": source_id,
                "type": "page",
                "page": page_number
            }),
            "content_hash": self._generate_hash(page_text[:1000]),  # Hash first 1000 chars for performance
            "metadata": {
                "page_number": page_number,
                "width": width,
                "height": height,
                "rotation": page.rotation
            }
        }

        elements.append(page_element)

        # Create relationship from content to page
        contains_relationship = {
            "relationship_id": self._generate_id("rel_"),
            "source_id": parent_id,
            "target_id": page_id,
            "relationship_type": RelationshipType.CONTAINS.value,
            "metadata": {
                "confidence": 1.0,
                "index": page_number
            }
        }
        relationships.append(contains_relationship)

        # Create inverse relationship
        contained_by_relationship = {
            "relationship_id": self._generate_id("rel_"),
            "source_id": page_id,
            "target_id": parent_id,
            "relationship_type": RelationshipType.CONTAINED_BY.value,
            "metadata": {
                "confidence": 1.0
            }
        }
        relationships.append(contained_by_relationship)

        # Extract text blocks and create relationships
        text_elements, text_relationships = self._extract_text_blocks(page, doc_id, page_id, source_id)
        elements.extend(text_elements)
        relationships.extend(text_relationships)

        # Extract tables if enabled
        if self.extract_tables:
            table_elements, table_relationships = self._extract_tables(page, doc_id, page_id, source_id)
            elements.extend(table_elements)
            relationships.extend(table_relationships)

        # Extract annotations if enabled
        if self.extract_annotations:
            annotation_elements, annotation_relationships = self._extract_annotations(page, doc_id, page_id, source_id)
            elements.extend(annotation_elements)
            relationships.extend(annotation_relationships)

        # Extract images if enabled
        if self.extract_images:
            image_elements, image_relationships = self._extract_images(page, doc_id, page_id, source_id)
            elements.extend(image_elements)
            relationships.extend(image_relationships)

        return elements, relationships

    def _extract_text_blocks(self, page: fitz.Page, doc_id: str, page_id: str, source_id: str) -> tuple[
        List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract text blocks from a PDF page.

        Args:
            page: The PDF page
            doc_id: Document ID
            page_id: Page element ID
            source_id: Source identifier

        Returns:
            Tuple of (list of text block elements, list of relationships)
        """
        elements = []
        relationships = []

        # Extract text blocks using PyMuPDF's built-in text extraction
        blocks = page.get_text("blocks")

        # Track headers for section structuring
        section_stack = [{"id": page_id, "level": 0}]
        current_section = None

        # Sort blocks by their vertical position (top to bottom)
        blocks.sort(key=lambda b: b[1])  # Sort by y0 coordinate

        for i, block in enumerate(blocks):
            # Block format: (x0, y0, x1, y1, text, block_type, block_no)
            x0, y0, x1, y1, text, block_type, block_no = block

            # Skip empty blocks
            if not text.strip():
                continue

            # Detect if this might be a header based on formatting
            is_header = False

            if self.detect_headers:
                # Try to get font information for this block
                try:
                    # Get span information to check for formatting
                    spans = page.get_textpage().extractDICT()["blocks"]
                    for span_block in spans:
                        if not span_block.get("lines"):
                            continue

                        # Check if this span contains our text
                        block_spans = []
                        for line in span_block["lines"]:
                            for span in line.get("spans", []):
                                block_spans.append(span)

                        # Check if any span has large font size or is bold
                        for span in block_spans:
                            span_text = span.get("text", "")
                            if not span_text or span_text not in text:
                                continue

                            font_size = span.get("size", 0)
                            font_flags = span.get("flags", 0)

                            # Check for header-like properties
                            if (font_size >= self.min_header_font_size or
                                    (font_flags & 16) != 0):  # Check if bold (font flag 16)
                                is_header = True
                                break

                except Exception as e:
                    logger.debug(f"Error analyzing text formatting: {str(e)}")

                # Fallback header detection based on text properties
                if not is_header and len(text.strip()) <= 100:
                    # Check if it's a short line without punctuation at the end
                    text_stripped = text.strip()
                    if text_stripped and text_stripped[-1] not in ".,:;?!":
                        # Check if it's all caps or ends with a number (like "SECTION 1" or "Chapter 2")
                        if (text_stripped.upper() == text_stripped or
                                re.search(r'\b(SECTION|Section|Chapter|CHAPTER|PART|Part)\s+\d+', text_stripped)):
                            is_header = True

            # Generate element ID based on type
            if is_header:
                element_id = self._generate_id(f"header_{i}_")
                element_type = "header"

                # Determine header level based on font size or other heuristics
                header_level = 1  # Default
                try:
                    # Check font size and position for hints about header level
                    spans = page.get_textpage().extractDICT()["blocks"]
                    for span_block in spans:
                        if not span_block.get("lines"):
                            continue

                        for line in span_block["lines"]:
                            for span in line.get("spans", []):
                                span_text = span.get("text", "")
                                if not span_text or span_text not in text:
                                    continue

                                font_size = span.get("size", 0)

                                # Assign header level based on font size
                                if font_size > 18:
                                    header_level = 1
                                elif font_size > 16:
                                    header_level = 2
                                elif font_size > 14:
                                    header_level = 3
                                elif font_size > 12:
                                    header_level = 4
                                else:
                                    header_level = 5
                except Exception as e:
                    logger.debug(f"Error determining header level: {str(e)}")

                # Update section stack based on header level
                while section_stack[-1]["level"] >= header_level:
                    section_stack.pop()

                # Create section element
                section_id = self._generate_id(f"section_{i}_")
                section_element = {
                    "element_id": section_id,
                    "doc_id": doc_id,
                    "element_type": "section",
                    "parent_id": section_stack[-1]["id"],
                    "content_preview": f"Section: {text[:50]}{'...' if len(text) > 50 else ''}",
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "section",
                        "page": page.number + 1,
                        "bbox": [x0, y0, x1, y1]
                    }),
                    "content_hash": self._generate_hash(text),
                    "metadata": {
                        "page_number": page.number + 1,
                        "bbox": [x0, y0, x1, y1],
                        "level": header_level
                    }
                }

                elements.append(section_element)

                # Create relationship from parent to section
                section_parent_id = section_stack[-1]["id"]
                contains_section_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": section_parent_id,
                    "target_id": section_id,
                    "relationship_type": RelationshipType.CONTAINS.value,
                    "metadata": {
                        "confidence": 1.0,
                        "index": i
                    }
                }
                relationships.append(contains_section_relationship)

                # Create inverse relationship
                section_contained_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": section_id,
                    "target_id": section_parent_id,
                    "relationship_type": RelationshipType.CONTAINED_BY.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(section_contained_relationship)

                # Add to section stack
                section_stack.append({"id": section_id, "level": header_level})
                current_section = section_id

                # Set this header's parent to the section
                parent_id_for_block = section_id
            else:
                element_id = self._generate_id(f"textblock_{i}_")
                element_type = "paragraph"

                # Set parent to current section if we have one, otherwise to the page
                parent_id_for_block = section_stack[-1]["id"]

            # Create text block element
            content_preview = text.replace('\n', ' ')
            if len(content_preview) > self.max_content_preview:
                content_preview = content_preview[:self.max_content_preview - 3] + "..."

            block_element = {
                "element_id": element_id,
                "doc_id": doc_id,
                "element_type": element_type,
                "parent_id": parent_id_for_block,
                "content_preview": content_preview,
                "content_location": json.dumps({
                    "source": source_id,
                    "type": element_type,
                    "page": page.number + 1,
                    "bbox": [x0, y0, x1, y1]
                }),
                "content_hash": self._generate_hash(text),
                "metadata": {
                    "page_number": page.number + 1,
                    "bbox": [x0, y0, x1, y1],
                    "block_type": block_type,
                    "block_number": block_no
                }
            }

            if is_header:
                # Add header-specific metadata
                block_element["metadata"]["level"] = header_level

            elements.append(block_element)

            # Create relationship from parent to text block
            relationship_type = RelationshipType.CONTAINS.value
            if element_type == "paragraph":
                relationship_type = RelationshipType.CONTAINS_TEXT.value

            contains_block_relationship = {
                "relationship_id": self._generate_id("rel_"),
                "source_id": parent_id_for_block,
                "target_id": element_id,
                "relationship_type": relationship_type,
                "metadata": {
                    "confidence": 1.0,
                    "index": i
                }
            }
            relationships.append(contains_block_relationship)

            # Create inverse relationship
            block_contained_relationship = {
                "relationship_id": self._generate_id("rel_"),
                "source_id": element_id,
                "target_id": parent_id_for_block,
                "relationship_type": RelationshipType.CONTAINED_BY.value,
                "metadata": {
                    "confidence": 1.0
                }
            }
            relationships.append(block_contained_relationship)

        return elements, relationships

    def _extract_tables(self, page: fitz.Page, doc_id: str, page_id: str, source_id: str) -> tuple[
        List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract tables from a PDF page.

        Args:
            page: The PDF page
            doc_id: Document ID
            page_id: Page element ID
            source_id: Source identifier

        Returns:
            Tuple of (list of table elements, list of relationships)
        """
        elements = []
        relationships = []

        # Use heuristic-based table detection as PyMuPDF doesn't have built-in table detection
        if self.table_detection_method == "heuristic":
            # Detect tables using heuristics based on text blocks and positioning
            tables = self._detect_tables_heuristic(page)

            for table_idx, table in enumerate(tables):
                table_id = self._generate_id(f"table_{table_idx}_")

                # Extract basic table information
                rows, cols, cells, bbox = table

                # Create table content preview
                content_preview = f"Table with {rows} rows and {cols} columns"

                # Extract table text
                table_text = []
                for row in sorted(cells.keys()):
                    row_text = []
                    for col in sorted(cells[row].keys()):
                        cell_text = cells[row][col].strip()
                        row_text.append(cell_text if cell_text else "")
                    table_text.append("\t".join(row_text))

                table_text_str = "\n".join(table_text)

                # Create table element
                table_element = {
                    "element_id": table_id,
                    "doc_id": doc_id,
                    "element_type": "table",
                    "parent_id": page_id,
                    "content_preview": content_preview,
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "table",
                        "page": page.number + 1,
                        "bbox": bbox
                    }),
                    "content_hash": self._generate_hash(table_text_str),
                    "metadata": {
                        "page_number": page.number + 1,
                        "rows": rows,
                        "columns": cols,
                        "bbox": bbox,
                        "table_content": table_text_str
                    }
                }

                elements.append(table_element)

                # Create relationship from page to table
                contains_table_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": page_id,
                    "target_id": table_id,
                    "relationship_type": RelationshipType.CONTAINS.value,
                    "metadata": {
                        "confidence": 1.0,
                        "index": table_idx
                    }
                }
                relationships.append(contains_table_relationship)

                # Create inverse relationship
                table_contained_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": table_id,
                    "target_id": page_id,
                    "relationship_type": RelationshipType.CONTAINED_BY.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(table_contained_relationship)

                # Add the header row if one exists
                if rows > 0:
                    header_row_id = self._generate_id(f"table_header_{table_idx}_")
                    header_texts = []

                    for col in sorted(cells[0].keys()):
                        header_cell = cells[0][col].strip()
                        header_texts.append(header_cell if header_cell else "")

                    header_text = "\t".join(header_texts)

                    # Create header row element
                    header_row_element = {
                        "element_id": header_row_id,
                        "doc_id": doc_id,
                        "element_type": "table_header_row",
                        "parent_id": table_id,
                        "content_preview": header_text[:self.max_content_preview] + (
                            "..." if len(header_text) > self.max_content_preview else ""),
                        "content_location": json.dumps({
                            "source": source_id,
                            "type": "table_header_row",
                            "page": page.number + 1,
                            "table": table_idx
                        }),
                        "content_hash": self._generate_hash(header_text),
                        "metadata": {
                            "page_number": page.number + 1,
                            "row": 0,
                            "values": header_texts
                        }
                    }

                    elements.append(header_row_element)

                    # Create relationship from table to header row
                    contains_header_row_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": table_id,
                        "target_id": header_row_id,
                        "relationship_type": RelationshipType.CONTAINS_TABLE_ROW.value,
                        "metadata": {
                            "confidence": 1.0,
                            "row_index": 0
                        }
                    }
                    relationships.append(contains_header_row_relationship)

                    # Create inverse relationship
                    header_row_contained_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": header_row_id,
                        "target_id": table_id,
                        "relationship_type": RelationshipType.CONTAINED_BY.value,
                        "metadata": {
                            "confidence": 1.0
                        }
                    }
                    relationships.append(header_row_contained_relationship)

                    # Process header cells
                    for col_idx, col in enumerate(sorted(cells[0].keys())):
                        cell_id = self._generate_id(f"table_header_cell_{table_idx}_{col_idx}_")
                        cell_text = cells[0][col].strip()

                        # Create header cell element
                        header_cell_element = {
                            "element_id": cell_id,
                            "doc_id": doc_id,
                            "element_type": "table_header",
                            "parent_id": header_row_id,
                            "content_preview": cell_text[:self.max_content_preview] + (
                                "..." if len(cell_text) > self.max_content_preview else ""),
                            "content_location": json.dumps({
                                "source": source_id,
                                "type": "table_header",
                                "page": page.number + 1,
                                "table": table_idx,
                                "row": 0,
                                "col": col_idx
                            }),
                            "content_hash": self._generate_hash(cell_text),
                            "metadata": {
                                "page_number": page.number + 1,
                                "row": 0,
                                "col": col_idx,
                                "text": cell_text
                            }
                        }

                        elements.append(header_cell_element)

                        # Create relationship from header row to header cell
                        contains_header_cell_relationship = {
                            "relationship_id": self._generate_id("rel_"),
                            "source_id": header_row_id,
                            "target_id": cell_id,
                            "relationship_type": RelationshipType.CONTAINS_TABLE_HEADER.value,
                            "metadata": {
                                "confidence": 1.0,
                                "col_index": col_idx
                            }
                        }
                        relationships.append(contains_header_cell_relationship)

                        # Create inverse relationship
                        header_cell_contained_relationship = {
                            "relationship_id": self._generate_id("rel_"),
                            "source_id": cell_id,
                            "target_id": header_row_id,
                            "relationship_type": RelationshipType.CONTAINED_BY.value,
                            "metadata": {
                                "confidence": 1.0
                            }
                        }
                        relationships.append(header_cell_contained_relationship)

                # Process data rows
                for row_idx in range(1, rows):  # Skip header row (0)
                    row_id = self._generate_id(f"table_row_{table_idx}_{row_idx}_")

                    # Create row element
                    row_text = []
                    for col in sorted(cells[row_idx].keys()):
                        cell_text = cells[row_idx][col].strip()
                        row_text.append(cell_text if cell_text else "")

                    row_text_str = "\t".join(row_text)

                    row_element = {
                        "element_id": row_id,
                        "doc_id": doc_id,
                        "element_type": "table_row",
                        "parent_id": table_id,
                        "content_preview": f"Row {row_idx + 1}",
                        "content_location": json.dumps({
                            "source": source_id,
                            "type": "table_row",
                            "page": page.number + 1,
                            "table": table_idx,
                            "row": row_idx
                        }),
                        "content_hash": self._generate_hash(row_text_str),
                        "metadata": {
                            "page_number": page.number + 1,
                            "row": row_idx
                        }
                    }

                    elements.append(row_element)

                    # Create relationship from table to row
                    contains_row_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": table_id,
                        "target_id": row_id,
                        "relationship_type": RelationshipType.CONTAINS_TABLE_ROW.value,
                        "metadata": {
                            "confidence": 1.0,
                            "row_index": row_idx
                        }
                    }
                    relationships.append(contains_row_relationship)

                    # Create inverse relationship
                    row_contained_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": row_id,
                        "target_id": table_id,
                        "relationship_type": RelationshipType.CONTAINED_BY.value,
                        "metadata": {
                            "confidence": 1.0
                        }
                    }
                    relationships.append(row_contained_relationship)

                    # Process cells in the row
                    for col_idx, col in enumerate(sorted(cells[row_idx].keys())):
                        cell_id = self._generate_id(f"table_cell_{table_idx}_{row_idx}_{col_idx}_")
                        cell_text = cells[row_idx][col].strip()

                        # Create cell element
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
                                "page": page.number + 1,
                                "table": table_idx,
                                "row": row_idx,
                                "col": col_idx
                            }),
                            "content_hash": self._generate_hash(cell_text),
                            "metadata": {
                                "page_number": page.number + 1,
                                "row": row_idx,
                                "col": col_idx,
                                "text": cell_text
                            }
                        }

                        elements.append(cell_element)

                        # Create relationship from row to cell
                        contains_cell_relationship = {
                            "relationship_id": self._generate_id("rel_"),
                            "source_id": row_id,
                            "target_id": cell_id,
                            "relationship_type": RelationshipType.CONTAINS_TABLE_CELL.value,
                            "metadata": {
                                "confidence": 1.0,
                                "col_index": col_idx
                            }
                        }
                        relationships.append(contains_cell_relationship)

                        # Create inverse relationship
                        cell_contained_relationship = {
                            "relationship_id": self._generate_id("rel_"),
                            "source_id": cell_id,
                            "target_id": row_id,
                            "relationship_type": RelationshipType.CONTAINED_BY.value,
                            "metadata": {
                                "confidence": 1.0
                            }
                        }
                        relationships.append(cell_contained_relationship)

        return elements, relationships

    def _detect_tables_heuristic(self, page: fitz.Page) -> List[Tuple]:
        """
        Detect tables using heuristic methods.

        Args:
            page: The PDF page

        Returns:
            List of detected tables as (rows, cols, cells_dict, bbox) tuples
        """
        tables = []

        try:
            # Get word level text to analyze positioning
            words = page.get_text("words")

            # Sort words by y-coordinate to group into potential rows
            words.sort(key=lambda w: w[3])  # Sort by y1 (bottom of word)

            # Group words into rows based on y-coordinate proximity
            row_tolerance = 5  # Max vertical gap between words in the same row
            rows = []
            current_row = []
            last_y = None

            for word in words:
                # Word format: (x0, y0, x1, y1, text, block_no, line_no, word_no)
                if not word[4].strip():  # Skip empty words
                    continue

                if last_y is None or abs(word[3] - last_y) <= row_tolerance:
                    current_row.append(word)
                    last_y = word[3]
                else:
                    if current_row:
                        rows.append(sorted(current_row, key=lambda w: w[0]))  # Sort by x0
                    current_row = [word]
                    last_y = word[3]

            if current_row:
                rows.append(sorted(current_row, key=lambda w: w[0]))

            # Identify potential tables: consecutive rows with similar column alignment
            min_rows = self.min_table_rows
            min_cols = self.min_table_cols
            min_alignment = 0.5  # Minimum proportion of columns that must align

            current_table_start = 0
            in_table = False

            for i in range(1, len(rows)):
                alignment_score = self._calculate_column_alignment(rows[i - 1], rows[i])

                if alignment_score >= min_alignment:
                    if not in_table:
                        current_table_start = i - 1
                        in_table = True
                else:
                    if in_table:
                        # End of a potential table
                        table_rows = rows[current_table_start:i]

                        if len(table_rows) >= min_rows:
                            # Analyze column structure
                            cols, cells = self._extract_table_structure(table_rows)

                            if cols >= min_cols:
                                # Calculate table bounding box
                                x0 = min(word[0] for row in table_rows for word in row)
                                y0 = min(word[1] for row in table_rows for word in row)
                                x1 = max(word[2] for row in table_rows for word in row)
                                y1 = max(word[3] for row in table_rows for word in row)

                                tables.append((len(table_rows), cols, cells, [x0, y0, x1, y1]))

                        in_table = False

            # Check if the last rows form a table
            if in_table and len(rows) - current_table_start >= min_rows:
                table_rows = rows[current_table_start:]
                cols, cells = self._extract_table_structure(table_rows)

                if cols >= min_cols:
                    x0 = min(word[0] for row in table_rows for word in row)
                    y0 = min(word[1] for row in table_rows for word in row)
                    x1 = max(word[2] for row in table_rows for word in row)
                    y1 = max(word[3] for row in table_rows for word in row)

                    tables.append((len(table_rows), cols, cells, [x0, y0, x1, y1]))

        except Exception as e:
            logger.warning(f"Error during table detection: {str(e)}")

        return tables

    @staticmethod
    def _calculate_column_alignment(row1: List, row2: List) -> float:
        """
        Calculate how well the columns align between two rows.

        Args:
            row1: First row of words
            row2: Second row of words

        Returns:
            Alignment score (0-1)
        """
        # Simple alignment calculation based on x-coordinate overlap
        alignment_count = 0
        total_checks = 0

        # Define tolerance for x-coordinate matching
        x_tolerance = 10

        for word1 in row1:
            for word2 in row2:
                if abs(word1[0] - word2[0]) < x_tolerance:  # Start positions align
                    alignment_count += 1
                    break
            total_checks += 1

        if total_checks == 0:
            return 0

        return alignment_count / total_checks

    @staticmethod
    def _extract_table_structure(rows: List[List]) -> Tuple[int, Dict]:
        """
        Extract the column structure from table rows.

        Args:
            rows: List of rows, where each row is a list of words

        Returns:
            Tuple of (number of columns, cell dictionary)
        """
        # Identify column boundaries based on word positions
        all_positions = []
        for row in rows:
            for word in row:
                all_positions.append(word[0])  # Start x
                all_positions.append(word[2])  # End x

        all_positions.sort()

        # Find clusters of positions to determine column boundaries
        tolerance = 10  # Max gap within a cluster
        clusters = []
        current_cluster = [all_positions[0]]

        for pos in all_positions[1:]:
            if pos - current_cluster[-1] <= tolerance:
                current_cluster.append(pos)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [pos]

        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))

        # Reduce clusters with very small gaps
        min_col_gap = 20
        filtered_clusters = [clusters[0]]
        for i in range(1, len(clusters)):
            if clusters[i] - filtered_clusters[-1] >= min_col_gap:
                filtered_clusters.append(clusters[i])

        # Use filtered clusters as column boundaries
        boundaries = filtered_clusters
        num_cols = len(boundaries) - 1 if len(boundaries) > 1 else 1

        # Now assign words to cells based on boundaries
        cells = {}

        for row_idx, row in enumerate(rows):
            cells[row_idx] = {}

            for word in row:
                # Find which column this word belongs to
                col_idx = 0
                word_center = (word[0] + word[2]) / 2

                while col_idx < len(boundaries) - 1 and word_center > boundaries[col_idx]:
                    col_idx += 1

                # Append to cell content
                if col_idx not in cells[row_idx]:
                    cells[row_idx][col_idx] = word[4]
                else:
                    cells[row_idx][col_idx] += " " + word[4]

        return num_cols, cells

    def _extract_annotations(self, page: fitz.Page, doc_id: str, page_id: str, source_id: str) -> tuple[
        List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract annotations from a PDF page.

        Args:
            page: The PDF page
            doc_id: Document ID
            page_id: Page element ID
            source_id: Source identifier

        Returns:
            Tuple of (list of annotation elements, list of relationships)
        """
        elements = []
        relationships = []

        # Create annotations container
        annotations_id = None
        if page.annots():
            annotations_id = self._generate_id("annotations_")
            annotations_element = {
                "element_id": annotations_id,
                "doc_id": doc_id,
                "element_type": "annotations",
                "parent_id": page_id,
                "content_preview": "Page annotations",
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "annotations",
                    "page": page.number + 1
                }),
                "content_hash": "",
                "metadata": {
                    "page_number": page.number + 1
                }
            }
            elements.append(annotations_element)

            # Create relationship from page to annotations container
            contains_annotations_relationship = {
                "relationship_id": self._generate_id("rel_"),
                "source_id": page_id,
                "target_id": annotations_id,
                "relationship_type": RelationshipType.CONTAINS.value,
                "metadata": {
                    "confidence": 1.0
                }
            }
            relationships.append(contains_annotations_relationship)

            # Create inverse relationship
            annotations_contained_relationship = {
                "relationship_id": self._generate_id("rel_"),
                "source_id": annotations_id,
                "target_id": page_id,
                "relationship_type": RelationshipType.CONTAINED_BY.value,
                "metadata": {
                    "confidence": 1.0
                }
            }
            relationships.append(annotations_contained_relationship)

            # Get all annotations on the page
            for annot_idx, annot in enumerate(page.annots()):
                # Generate element ID
                annot_id = self._generate_id(f"annot_{annot_idx}_")

                # Get annotation details
                annot_type = annot.type[1]  # Get type name without number
                rect = annot.rect
                content = annot.info.get("content", "")
                author = annot.info.get("title", "")
                created = annot.info.get("creationDate", "")
                modified = annot.info.get("modDate", "")

                # Create content preview
                if content:
                    content_preview = f"{annot_type} annotation: {content}"
                    if len(content_preview) > self.max_content_preview:
                        content_preview = content_preview[:self.max_content_preview - 3] + "..."
                else:
                    content_preview = f"{annot_type} annotation"

                # Determine element type based on annotation type
                if annot_type == "Highlight":
                    element_type = "highlight"
                elif annot_type == "Text":
                    element_type = "comment"
                elif annot_type == "FreeText":
                    element_type = "text_annotation"
                elif annot_type == "Ink":
                    element_type = "ink_annotation"
                elif annot_type == "Underline":
                    element_type = "underline"
                else:
                    element_type = "annotation"

                # Create annotation element
                annot_element = {
                    "element_id": annot_id,
                    "doc_id": doc_id,
                    "element_type": element_type,
                    "parent_id": annotations_id,
                    "content_preview": content_preview,
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": element_type,
                        "page": page.number + 1,
                        "bbox": [rect.x0, rect.y0, rect.x1, rect.y1]
                    }),
                    "content_hash": self._generate_hash(content),
                    "metadata": {
                        "page_number": page.number + 1,
                        "annotation_type": annot_type,
                        "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                        "author": author,
                        "created": created,
                        "modified": modified,
                        "flags": annot.flags
                    }
                }

                elements.append(annot_element)

                # Create relationship from annotations container to annotation
                contains_annotation_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": annotations_id,
                    "target_id": annot_id,
                    "relationship_type": RelationshipType.CONTAINS.value,
                    "metadata": {
                        "confidence": 1.0,
                        "index": annot_idx
                    }
                }
                relationships.append(contains_annotation_relationship)

                # Create inverse relationship
                annotation_contained_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": annot_id,
                    "target_id": annotations_id,
                    "relationship_type": RelationshipType.CONTAINED_BY.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(annotation_contained_relationship)

        return elements, relationships

    def _extract_images(self, page: fitz.Page, doc_id: str, page_id: str, source_id: str) -> tuple[
        List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract images from a PDF page.

        Args:
            page: The PDF page
            doc_id: Document ID
            page_id: Page element ID
            source_id: Source identifier

        Returns:
            Tuple of (list of image elements, list of relationships)
        """
        elements = []
        relationships = []

        try:
            # Get image blocks
            image_list = page.get_images(full=True)

            # Create images container if we have images
            images_id = None

            if image_list:
                images_id = self._generate_id("images_")
                images_element = {
                    "element_id": images_id,
                    "doc_id": doc_id,
                    "element_type": "images",
                    "parent_id": page_id,
                    "content_preview": f"Page images ({len(image_list)})",
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "images",
                        "page": page.number + 1
                    }),
                    "content_hash": "",
                    "metadata": {
                        "page_number": page.number + 1,
                        "image_count": len(image_list)
                    }
                }
                elements.append(images_element)

                # Create relationship from page to images container
                contains_images_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": page_id,
                    "target_id": images_id,
                    "relationship_type": RelationshipType.CONTAINS.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(contains_images_relationship)

                # Create inverse relationship
                images_contained_relationship = {
                    "relationship_id": self._generate_id("rel_"),
                    "source_id": images_id,
                    "target_id": page_id,
                    "relationship_type": RelationshipType.CONTAINED_BY.value,
                    "metadata": {
                        "confidence": 1.0
                    }
                }
                relationships.append(images_contained_relationship)

                for img_idx, img_info in enumerate(image_list):
                    # Generate element ID
                    img_id = self._generate_id(f"img_{img_idx}_")

                    # Extract image information
                    xref = img_info[0]  # Cross-reference number
                    smask = img_info[1]  # SMask number (transparency)
                    width = img_info[2]  # Width
                    height = img_info[3]  # Height
                    bpc = img_info[4]  # Bits per component
                    colorspace = img_info[5]  # Colorspace
                    name = img_info[7]  # Image name

                    # Try to determine image position on the page
                    img_bbox = None
                    img_rects = []

                    # Search for image references in the page's content streams
                    try:
                        # Use a heuristic approach to find image positions
                        for item in page.get_drawings():
                            if item["type"] == "image" and item.get("xref") == xref:
                                rect = item["rect"]
                                img_bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
                                img_rects.append(img_bbox)
                    except Exception as e:
                        logger.debug(f"Error locating image on page: {str(e)}")

                    # Use the largest rectangle if multiple were found
                    if img_rects:
                        img_bbox = max(img_rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))

                    # Create content preview
                    content_preview = f"Image ({width}x{height})"

                    # Create image element
                    img_element = {
                        "element_id": img_id,
                        "doc_id": doc_id,
                        "element_type": "image",
                        "parent_id": images_id,
                        "content_preview": content_preview,
                        "content_location": json.dumps({
                            "source": source_id,
                            "type": "image",
                            "page": page.number + 1,
                            "xref": xref,
                            "bbox": img_bbox
                        }),
                        "content_hash": f"img_{xref}_{smask}",  # Use xref and smask as content hash
                        "metadata": {
                            "page_number": page.number + 1,
                            "xref": xref,
                            "width": width,
                            "height": height,
                            "bbox": img_bbox,
                            "bits_per_component": bpc,
                            "colorspace": colorspace,
                            "name": name if name else None
                        }
                    }

                    elements.append(img_element)

                    # Create relationship from images container to image
                    contains_image_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": images_id,
                        "target_id": img_id,
                        "relationship_type": RelationshipType.CONTAINS.value,
                        "metadata": {
                            "confidence": 1.0,
                            "index": img_idx
                        }
                    }
                    relationships.append(contains_image_relationship)

                    # Create inverse relationship
                    image_contained_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": img_id,
                        "target_id": images_id,
                        "relationship_type": RelationshipType.CONTAINED_BY.value,
                        "metadata": {
                            "confidence": 1.0
                        }
                    }
                    relationships.append(image_contained_relationship)

        except Exception as e:
            logger.warning(f"Error extracting images: {str(e)}")

        return elements, relationships

    def _extract_document_links(self, doc: fitz.Document, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Helper method to extract hyperlinks from PDF document.
        This is called during the parsing phase.

        Args:
            doc: The PDF document
            elements: List of extracted elements

        Returns:
            List of hyperlink dictionaries
        """
        links = []

        if not self.extract_links:
            return links

        # Build a mapping from page/bbox to element_id
        element_map = {}
        for element in elements:
            element_type = element.get("element_type", "")

            if element_type in ["paragraph", "header", "table_cell", "table_header"]:
                content_location = element.get("content_location", "{}")
                try:
                    location = json.loads(content_location)
                    page_num = location.get("page")
                    bbox = tuple(location.get("bbox", [0, 0, 0, 0]))

                    key = (page_num, bbox)
                    element_map[key] = element.get("element_id")
                except (json.JSONDecodeError, TypeError):
                    pass

        # Extract links from each page
        for page_idx in range(min(len(doc), self.max_pages)):
            page = doc[page_idx]

            # Get all links on the page
            for link in page.get_links():
                link_type = link.get("kind")

                if link_type == fitz.LINK_URI:
                    # External URL link
                    uri = link.get("uri", "")
                    if not uri:
                        continue

                    # Get the link rectangle
                    rect = link.get("from")
                    if not rect:
                        continue

                    # Find the closest text element containing or overlapping with the link
                    source_id = None
                    min_distance = float('inf')

                    for key, elem_id in element_map.items():
                        page_num, bbox = key

                        if page_num != page_idx + 1:
                            continue

                        # Calculate overlap or distance
                        if self._rectangles_overlap(rect, bbox):
                            # Found an overlapping element
                            source_id = elem_id
                            break

                        # Calculate distance between rectangles
                        distance = self._rectangle_distance(rect, bbox)
                        if distance < min_distance:
                            min_distance = distance
                            source_id = elem_id

                    # If we found a source element, create the link
                    if source_id:
                        # Try to get the link text from the element
                        link_text = ""
                        for element in elements:
                            if element.get("element_id") == source_id:
                                link_text = element.get("content_preview", "")
                                break

                        links.append({
                            "source_id": source_id,
                            "link_text": link_text,
                            "link_target": uri,
                            "link_type": "uri"
                        })

                elif link_type == fitz.LINK_GOTO:
                    # Internal link to another page
                    dest = link.get("page", -1)

                    if 0 <= dest < len(doc):
                        # Get the link rectangle
                        rect = link.get("from")
                        if not rect:
                            continue

                        # Find the source element as above
                        source_id = None
                        min_distance = float('inf')

                        for key, elem_id in element_map.items():
                            page_num, bbox = key

                            if page_num != page_idx + 1:
                                continue

                            if self._rectangles_overlap(rect, bbox):
                                source_id = elem_id
                                break

                            distance = self._rectangle_distance(rect, bbox)
                            if distance < min_distance:
                                min_distance = distance
                                source_id = elem_id

                        if source_id:
                            # Try to get the link text
                            link_text = ""
                            for element in elements:
                                if element.get("element_id") == source_id:
                                    link_text = element.get("content_preview", "")
                                    break

                            # Find the target element (first element on the destination page)
                            target_id = None
                            for element in elements:
                                if element.get("element_type") == "page":
                                    content_location = element.get("content_location", "{}")
                                    try:
                                        location = json.loads(content_location)
                                        if location.get("page") == dest + 1:
                                            target_id = element.get("element_id")
                                            break
                                    except (json.JSONDecodeError, TypeError):
                                        pass

                            if target_id:
                                links.append({
                                    "source_id": source_id,
                                    "link_text": link_text,
                                    "link_target": target_id,
                                    "link_type": "internal"
                                })

        return links

    @staticmethod
    def _rectangles_overlap(rect1, rect2):
        """Check if two rectangles overlap."""
        if isinstance(rect1, fitz.Rect):
            x0_1, y0_1, x1_1, y1_1 = rect1.x0, rect1.y0, rect1.x1, rect1.y1
        else:
            x0_1, y0_1, x1_1, y1_1 = rect1

        if isinstance(rect2, tuple) or isinstance(rect2, list):
            x0_2, y0_2, x1_2, y1_2 = rect2
        else:
            x0_2, y0_2, x1_2, y1_2 = rect2.x0, rect2.y0, rect2.x1, rect2.y1

        return not (x0_1 > x1_2 or x1_1 < x0_2 or y0_1 > y1_2 or y1_1 < y0_2)

    @staticmethod
    def _rectangle_distance(rect1, rect2):
        """Calculate the minimum distance between two rectangles."""
        if isinstance(rect1, fitz.Rect):
            x0_1, y0_1, x1_1, y1_1 = rect1.x0, rect1.y0, rect1.x1, rect1.y1
        else:
            x0_1, y0_1, x1_1, y1_1 = rect1

        if isinstance(rect2, tuple) or isinstance(rect2, list):
            x0_2, y0_2, x1_2, y1_2 = rect2
        else:
            x0_2, y0_2, x1_2, y1_2 = rect2.x0, rect2.y0, rect2.x1, rect2.y1

        dx = max(0, max(x0_1, x0_2) - min(x1_1, x1_2))
        dy = max(0, max(y0_1, y0_2) - min(y1_1, y1_2))

        return (dx ** 2 + dy ** 2) ** 0.5

    @staticmethod
    def _generate_hash(content: str) -> str:
        """
        Generate a hash of content for change detection.

        Args:
            content: Text content

        Returns:
            MD5 hash of content
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()
