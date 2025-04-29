"""
PPTX document parser module for the document pointer system.

This module parses PowerPoint (PPTX) documents into structured elements.
"""

import json
import logging
import os
import re
from typing import Dict, Any, List, Optional

try:
    import pptx
    # noinspection PyUnresolvedReferences
    from pptx import Presentation
    from pptx.shapes.autoshape import Shape
    # noinspection PyUnresolvedReferences
    from pptx.shapes.group import GroupShape
    # noinspection PyUnresolvedReferences
    from pptx.shapes.picture import Picture
    # noinspection PyUnresolvedReferences
    from pptx.slide import Slide, SlideLayout
    # noinspection PyUnresolvedReferences
    from pptx.text.text import TextFrame

    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False
    logging.warning("python-pptx not available. Install with 'pip install python-pptx' to use PPTX parser")

from .base import DocumentParser

logger = logging.getLogger(__name__)


class PptxParser(DocumentParser):
    """Parser for PowerPoint (PPTX) documents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the PPTX parser."""
        super().__init__(config)

        if not PPTX_AVAILABLE:
            raise ImportError("python-pptx is required for PPTX parsing")

        # Configuration options
        self.config = config or {}
        self.extract_notes = self.config.get("extract_notes", True)
        self.extract_hidden_slides = self.config.get("extract_hidden_slides", False)
        self.extract_comments = self.config.get("extract_comments", True)
        self.extract_shapes = self.config.get("extract_shapes", True)
        self.extract_images = self.config.get("extract_images", True)
        self.extract_tables = self.config.get("extract_tables", True)
        self.extract_text_boxes = self.config.get("extract_text_boxes", True)
        self.extract_charts = self.config.get("extract_charts", True)
        self.extract_masters = self.config.get("extract_masters", False)
        self.extract_templates = self.config.get("extract_templates", False)
        self.temp_dir = self.config.get("temp_dir", os.path.join(os.path.dirname(__file__), 'temp'))

    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a PPTX document into structured elements."""
        # Extract metadata from doc_content
        source_id = doc_content["id"]
        metadata = doc_content.get("metadata", {}).copy()  # Make a copy to avoid modifying original

        # Check if we have a binary path
        binary_path = doc_content.get("binary_path")
        if not binary_path:
            raise ValueError("PPTX parser requires a binary_path to process the presentation")

        # Generate document ID if not present
        doc_id = metadata.get("doc_id", self._generate_id("doc_"))

        # Load PPTX document
        try:
            presentation = Presentation(binary_path)
        except Exception as e:
            logger.error(f"Error loading PPTX document: {str(e)}")
            raise

        # Create document record with metadata
        document = {
            "doc_id": doc_id,
            "doc_type": "pptx",
            "source": source_id,
            "metadata": self._extract_document_metadata(presentation, metadata),
            "content_hash": doc_content.get("content_hash", "")
        }

        # Create root element
        elements = [self._create_root_element(doc_id, source_id)]
        root_id = elements[0]["element_id"]

        # Parse document elements
        elements.extend(self._parse_presentation(presentation, doc_id, root_id, source_id))

        # Extract links from the document
        links = self._extract_links(presentation, elements)

        # Return the parsed document with extracted links
        return {
            "document": document,
            "elements": elements,
            "links": links,
            "relationships": []
        }

    @staticmethod
    def _extract_document_metadata(presentation: Presentation, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from PPTX document.

        Args:
            presentation: The PPTX presentation
            base_metadata: Base metadata from content source

        Returns:
            Dictionary of document metadata
        """
        # Combine base metadata with document properties
        metadata = base_metadata.copy()

        try:
            # Add core properties
            core_props = presentation.core_properties
            if core_props:
                # Add core properties to metadata
                if core_props.title:
                    metadata["title"] = core_props.title
                if core_props.author:
                    metadata["author"] = core_props.author
                if core_props.created:
                    metadata["created"] = core_props.created.timestamp() if hasattr(core_props.created,
                                                                                    'timestamp') else str(
                        core_props.created)
                if core_props.modified:
                    metadata["modified"] = core_props.modified.timestamp() if hasattr(core_props.modified,
                                                                                      'timestamp') else str(
                        core_props.modified)
                if core_props.last_modified_by:
                    metadata["last_modified_by"] = core_props.last_modified_by
                if core_props.keywords:
                    metadata["keywords"] = core_props.keywords
                if core_props.subject:
                    metadata["subject"] = core_props.subject
                if core_props.comments:
                    metadata["comments"] = core_props.comments
                if core_props.category:
                    metadata["category"] = core_props.category

            # Add presentation statistics
            metadata["slide_count"] = len(presentation.slides)
            metadata["slide_width"] = presentation.slide_width
            metadata["slide_height"] = presentation.slide_height

        except Exception as e:
            logger.warning(f"Error extracting document metadata: {str(e)}")

        return metadata

    def _parse_presentation(self, presentation: Presentation, doc_id: str, parent_id: str, source_id: str) -> List[
        Dict[str, Any]]:
        """
        Parse PowerPoint presentation into structured elements.

        Args:
            presentation: The PPTX presentation
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier

        Returns:
            List of parsed elements
        """
        elements: List = []

        # Create presentation body element
        body_id = self._generate_id("body_")
        body_element = {
            "element_id": body_id,
            "doc_id": doc_id,
            "element_type": "presentation_body",
            "parent_id": parent_id,
            "content_preview": "Presentation body",
            "content_location": json.dumps({
                "source": source_id,
                "type": "presentation_body"
            }),
            "content_hash": "",
            "metadata": {
                "slide_count": len(presentation.slides)
            }
        }
        elements.append(body_element)

        # Process slides
        for slide_idx, slide in enumerate(presentation.slides):
            # Skip hidden slides if not configured to extract them
            if not self.extract_hidden_slides and hasattr(slide, 'hidden') and slide.hidden:
                continue

            # Process this slide
            slide_elements = self._process_slide(slide, slide_idx, doc_id, body_id, source_id)
            elements.extend(slide_elements)

        # Extract masters if configured
        if self.extract_masters and hasattr(presentation, 'slide_masters'):
            master_elements = self._extract_slide_masters(presentation, doc_id, parent_id, source_id)
            elements.extend(master_elements)

        # Extract templates if configured
        if self.extract_templates and hasattr(presentation, 'slide_layouts'):
            template_elements = self._extract_slide_templates(presentation, doc_id, parent_id, source_id)
            elements.extend(template_elements)

        return elements

    def _process_slide(self, slide: Slide, slide_idx: int, doc_id: str, parent_id: str, source_id: str) -> List[
        Dict[str, Any]]:
        """
        Process a PowerPoint slide into structured elements.

        Args:
            slide: The PPTX slide
            slide_idx: Slide index (0-based)
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier

        Returns:
            List of slide-related elements
        """
        elements = []

        # Generate slide ID
        slide_id = self._generate_id(f"slide_{slide_idx}_")

        # Get slide title if available
        slide_title = self._get_slide_title(slide)

        # Create slide element
        slide_element = {
            "element_id": slide_id,
            "doc_id": doc_id,
            "element_type": "slide",
            "parent_id": parent_id,
            "content_preview": f"Slide {slide_idx + 1}: {slide_title}" if slide_title else f"Slide {slide_idx + 1}",
            "content_location": json.dumps({
                "source": source_id,
                "type": "slide",
                "index": slide_idx
            }),
            "content_hash": self._generate_hash(f"slide_{slide_idx}"),
            "metadata": {
                "index": slide_idx,
                "number": slide_idx + 1,
                "title": slide_title,
                "layout": self._get_slide_layout_name(slide),
                "has_notes": bool(slide.notes_slide and slide.notes_slide.notes_text_frame.text),
                "shape_count": len(slide.shapes)
            }
        }
        elements.append(slide_element)

        # Process slide shapes
        if self.extract_shapes:
            shape_elements = self._process_shapes(slide.shapes, doc_id, slide_id, source_id, slide_idx)
            elements.extend(shape_elements)

        # Process slide notes
        if self.extract_notes and slide.notes_slide and slide.notes_slide.notes_text_frame.text:
            notes_elements = self._process_notes(slide.notes_slide, slide_idx, doc_id, slide_id, source_id)
            elements.extend(notes_elements)

        # Process slide comments if available and configured
        if self.extract_comments and hasattr(slide, 'comments'):
            comment_elements = self._process_comments(slide, slide_idx, doc_id, slide_id, source_id)
            elements.extend(comment_elements)

        return elements

    def _process_shapes(self, shapes, doc_id: str, parent_id: str, source_id: str, slide_idx: int,
                        shape_path: str = "") -> List[Dict[str, Any]]:
        """
        Process PowerPoint shapes into structured elements.

        Args:
            shapes: Collection of shapes
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier
            slide_idx: Slide index
            shape_path: Path to the shape (for nested shapes)

        Returns:
            List of shape-related elements
        """
        elements = []

        for shape_idx, shape in enumerate(shapes):
            # Generate current shape path
            current_shape_path = f"{shape_path}/{shape_idx}" if shape_path else f"{shape_idx}"

            # Process shape based on type
            if isinstance(shape, GroupShape):
                # Process group shape and its children
                group_id = self._generate_id(f"group_{current_shape_path}_")

                # Create group element
                group_element = {
                    "element_id": group_id,
                    "doc_id": doc_id,
                    "element_type": "shape_group",
                    "parent_id": parent_id,
                    "content_preview": f"Shape Group {current_shape_path}",
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "shape_group",
                        "slide_index": slide_idx,
                        "shape_path": current_shape_path
                    }),
                    "content_hash": self._generate_hash(f"group_{current_shape_path}"),
                    "metadata": {
                        "slide_index": slide_idx,
                        "shape_index": shape_idx,
                        "shape_path": current_shape_path,
                        "shape_type": "group",
                        "shape_count": len(shape.shapes) if hasattr(shape, 'shapes') else 0
                    }
                }
                elements.append(group_element)

                # Process child shapes
                child_elements = self._process_shapes(shape.shapes, doc_id, group_id, source_id, slide_idx,
                                                      current_shape_path)
                elements.extend(child_elements)

            elif hasattr(shape, 'has_table') and shape.has_table and self.extract_tables:
                # Process table shape
                table_elements = self._process_table_shape(shape, doc_id, parent_id, source_id, slide_idx,
                                                           current_shape_path)
                elements.extend(table_elements)

            elif hasattr(shape, 'has_chart') and shape.has_chart and self.extract_charts:
                # Process chart shape
                chart_elements = self._process_chart_shape(shape, doc_id, parent_id, source_id, slide_idx,
                                                           current_shape_path)
                elements.extend(chart_elements)

            elif isinstance(shape, Picture) and self.extract_images:
                # Process picture shape
                picture_elements = self._process_picture_shape(shape, doc_id, parent_id, source_id, slide_idx,
                                                               current_shape_path)
                elements.extend(picture_elements)

            elif hasattr(shape, 'has_text_frame') and shape.has_text_frame and self.extract_text_boxes:
                # Process text shape
                text_elements = self._process_text_shape(shape, doc_id, parent_id, source_id, slide_idx,
                                                         current_shape_path)
                elements.extend(text_elements)

            elif self.extract_shapes:
                # Process generic shape
                shape_id = self._generate_id(f"shape_{current_shape_path}_")

                # Get shape name and type info
                shape_name = shape.name if hasattr(shape, 'name') else ""
                shape_type = self._get_shape_type(shape)

                # Create shape element
                shape_element = {
                    "element_id": shape_id,
                    "doc_id": doc_id,
                    "element_type": "shape",
                    "parent_id": parent_id,
                    "content_preview": f"Shape: {shape_name}" if shape_name else f"Shape {current_shape_path}",
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "shape",
                        "slide_index": slide_idx,
                        "shape_path": current_shape_path
                    }),
                    "content_hash": self._generate_hash(f"shape_{current_shape_path}"),
                    "metadata": {
                        "slide_index": slide_idx,
                        "shape_index": shape_idx,
                        "shape_path": current_shape_path,
                        "shape_type": shape_type,
                        "shape_name": shape_name
                    }
                }
                elements.append(shape_element)

        return elements

    def _process_text_shape(self, shape, doc_id: str, parent_id: str, source_id: str,
                            slide_idx: int, shape_path: str) -> List[Dict[str, Any]]:
        """
        Process a text shape into structured elements.

        Args:
            shape: The text shape
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier
            slide_idx: Slide index
            shape_path: Path to the shape

        Returns:
            List of text-related elements
        """
        elements = []

        try:
            if not shape.has_text_frame or not hasattr(shape, 'text_frame'):
                return elements

            text_frame = shape.text_frame
            text = text_frame.text

            if not text:
                return elements

            # Generate text element ID
            text_id = self._generate_id(f"text_{shape_path}_")

            # Get shape name if available
            shape_name = shape.name if hasattr(shape, 'name') else ""

            # Create text element
            text_element = {
                "element_id": text_id,
                "doc_id": doc_id,
                "element_type": "text_box",
                "parent_id": parent_id,
                "content_preview": text[:100] + ("..." if len(text) > 100 else ""),
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "text_box",
                    "slide_index": slide_idx,
                    "shape_path": shape_path
                }),
                "content_hash": self._generate_hash(text),
                "metadata": {
                    "slide_index": slide_idx,
                    "shape_path": shape_path,
                    "shape_name": shape_name,
                    "text": text,
                    "is_title": self._is_title_shape(shape),
                    "level": self._get_paragraph_level(text_frame) if hasattr(text_frame, 'paragraphs') else 0
                }
            }
            elements.append(text_element)

            # Process paragraphs if detailed paragraph extraction is desired
            if hasattr(text_frame, 'paragraphs') and len(text_frame.paragraphs) > 1:
                for para_idx, paragraph in enumerate(text_frame.paragraphs):
                    para_text = paragraph.text
                    if not para_text:
                        continue

                    para_id = self._generate_id(f"para_{shape_path}_{para_idx}_")

                    para_element = {
                        "element_id": para_id,
                        "doc_id": doc_id,
                        "element_type": "paragraph",
                        "parent_id": text_id,
                        "content_preview": para_text[:100] + ("..." if len(para_text) > 100 else ""),
                        "content_location": json.dumps({
                            "source": source_id,
                            "type": "paragraph",
                            "slide_index": slide_idx,
                            "shape_path": shape_path,
                            "paragraph_index": para_idx
                        }),
                        "content_hash": self._generate_hash(para_text),
                        "metadata": {
                            "slide_index": slide_idx,
                            "shape_path": shape_path,
                            "paragraph_index": para_idx,
                            "text": para_text,
                            "level": paragraph.level if hasattr(paragraph, 'level') else 0
                        }
                    }
                    elements.append(para_element)

        except Exception as e:
            logger.warning(f"Error processing text shape: {str(e)}")

        return elements

    def _process_picture_shape(self, shape, doc_id: str, parent_id: str, source_id: str,
                               slide_idx: int, shape_path: str) -> List[Dict[str, Any]]:
        """
        Process a picture shape into structured elements.

        Args:
            shape: The picture shape
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier
            slide_idx: Slide index
            shape_path: Path to the shape

        Returns:
            List of picture-related elements
        """
        elements = []

        try:
            # Generate image element ID
            image_id = self._generate_id(f"image_{shape_path}_")

            # Get shape name and image info
            shape_name = shape.name if hasattr(shape, 'name') else ""
            image_name = ""

            if hasattr(shape, 'image') and hasattr(shape.image, 'filename'):
                image_name = shape.image.filename

            # Create image element
            image_element = {
                "element_id": image_id,
                "doc_id": doc_id,
                "element_type": "image",
                "parent_id": parent_id,
                "content_preview": f"Image: {image_name}" if image_name else f"Image {shape_path}",
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "image",
                    "slide_index": slide_idx,
                    "shape_path": shape_path
                }),
                "content_hash": self._generate_hash(f"image_{shape_path}"),
                "metadata": {
                    "slide_index": slide_idx,
                    "shape_path": shape_path,
                    "shape_name": shape_name,
                    "image_name": image_name,
                    "alt_text": shape.alt_text if hasattr(shape, 'alt_text') else ""
                }
            }
            elements.append(image_element)

            # Process image caption (text) if available
            if hasattr(shape, 'text_frame') and shape.text_frame.text:
                caption_elements = self._process_text_shape(shape, doc_id, image_id, source_id, slide_idx,
                                                            f"{shape_path}_caption")
                elements.extend(caption_elements)

        except Exception as e:
            logger.warning(f"Error processing picture shape: {str(e)}")

        return elements

    def _process_table_shape(self, shape, doc_id: str, parent_id: str, source_id: str,
                             slide_idx: int, shape_path: str) -> List[Dict[str, Any]]:
        """
        Process a table shape into structured elements.

        Args:
            shape: The table shape
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier
            slide_idx: Slide index
            shape_path: Path to the shape

        Returns:
            List of table-related elements
        """
        elements = []

        try:
            if not shape.has_table or not hasattr(shape, 'table'):
                return elements

            table = shape.table
            shape_name = shape.name if hasattr(shape, 'name') else ""

            # Generate table element ID
            table_id = self._generate_id(f"table_{shape_path}_")

            # Extract basic table content for preview
            table_text = ""
            for row in table.rows:
                for cell in row.cells:
                    if cell.text_frame.text:
                        table_text += cell.text_frame.text + " | "
                table_text += "\n"

            # Create table element
            table_element = {
                "element_id": table_id,
                "doc_id": doc_id,
                "element_type": "table",
                "parent_id": parent_id,
                "content_preview": f"Table: {table_text[:100]}" + ("..." if len(table_text) > 100 else ""),
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "table",
                    "slide_index": slide_idx,
                    "shape_path": shape_path
                }),
                "content_hash": self._generate_hash(table_text),
                "metadata": {
                    "slide_index": slide_idx,
                    "shape_path": shape_path,
                    "shape_name": shape_name,
                    "rows": len(table.rows),
                    "columns": len(table.columns)
                }
            }
            elements.append(table_element)

            # Process table rows and cells
            for row_idx, row in enumerate(table.rows):
                # Generate row element ID
                row_id = self._generate_id(f"row_{shape_path}_{row_idx}_")

                # Create row element
                row_element = {
                    "element_id": row_id,
                    "doc_id": doc_id,
                    "element_type": "table_row",
                    "parent_id": table_id,
                    "content_preview": f"Row {row_idx + 1}",
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "table_row",
                        "slide_index": slide_idx,
                        "shape_path": shape_path,
                        "row": row_idx
                    }),
                    "content_hash": self._generate_hash(f"row_{shape_path}_{row_idx}"),
                    "metadata": {
                        "slide_index": slide_idx,
                        "shape_path": shape_path,
                        "row": row_idx
                    }
                }
                elements.append(row_element)

                # Process cells in this row
                for col_idx, cell in enumerate(row.cells):
                    cell_text = cell.text_frame.text if hasattr(cell, 'text_frame') else ""

                    if not cell_text:
                        continue

                    # Generate cell element ID
                    cell_id = self._generate_id(f"cell_{shape_path}_{row_idx}_{col_idx}_")

                    # Create cell element
                    cell_element = {
                        "element_id": cell_id,
                        "doc_id": doc_id,
                        "element_type": "table_cell",
                        "parent_id": row_id,
                        "content_preview": cell_text[:100] + ("..." if len(cell_text) > 100 else ""),
                        "content_location": json.dumps({
                            "source": source_id,
                            "type": "table_cell",
                            "slide_index": slide_idx,
                            "shape_path": shape_path,
                            "row": row_idx,
                            "col": col_idx
                        }),
                        "content_hash": self._generate_hash(cell_text),
                        "metadata": {
                            "slide_index": slide_idx,
                            "shape_path": shape_path,
                            "row": row_idx,
                            "col": col_idx,
                            "text": cell_text
                        }
                    }
                    elements.append(cell_element)

        except Exception as e:
            logger.warning(f"Error processing table shape: {str(e)}")

        return elements

    def _process_chart_shape(self, shape, doc_id: str, parent_id: str, source_id: str,
                             slide_idx: int, shape_path: str) -> List[Dict[str, Any]]:
        """
        Process a chart shape into structured elements.

        Args:
            shape: The chart shape
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier
            slide_idx: Slide index
            shape_path: Path to the shape

        Returns:
            List of chart-related elements
        """
        elements = []

        try:
            if not shape.has_chart or not hasattr(shape, 'chart'):
                return elements

            chart = shape.chart
            shape_name = shape.name if hasattr(shape, 'name') else ""

            # Generate chart element ID
            chart_id = self._generate_id(f"chart_{shape_path}_")

            # Get chart type
            chart_type = "unknown"
            if hasattr(chart, 'chart_type'):
                chart_type = str(chart.chart_type)

            # Get chart title
            chart_title = ""
            if hasattr(chart, 'chart_title') and hasattr(chart.chart_title, 'text_frame'):
                chart_title = chart.chart_title.text_frame.text

            # Create chart element
            chart_element = {
                "element_id": chart_id,
                "doc_id": doc_id,
                "element_type": "chart",
                "parent_id": parent_id,
                "content_preview": f"Chart: {chart_title}" if chart_title else f"Chart {shape_path}",
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "chart",
                    "slide_index": slide_idx,
                    "shape_path": shape_path
                }),
                "content_hash": self._generate_hash(f"chart_{shape_path}"),
                "metadata": {
                    "slide_index": slide_idx,
                    "shape_path": shape_path,
                    "shape_name": shape_name,
                    "chart_type": chart_type,
                    "chart_title": chart_title
                }
            }
            elements.append(chart_element)

            # Extract chart category and series names if available
            if hasattr(chart, 'plots') and chart.plots:
                plot = chart.plots[0]

                if hasattr(plot, 'categories'):
                    categories = []
                    for category in plot.categories:
                        if category:
                            categories.append(str(category))

                    if categories:
                        chart_element["metadata"]["categories"] = categories

                if hasattr(plot, 'series'):
                    series_names = []
                    for series in plot.series:
                        if hasattr(series, 'name') and series.name:
                            series_names.append(str(series.name))

                    if series_names:
                        chart_element["metadata"]["series"] = series_names

        except Exception as e:
            logger.warning(f"Error processing chart shape: {str(e)}")

        return elements

    def _process_notes(self, notes_slide, slide_idx: int, doc_id: str, parent_id: str, source_id: str) -> List[
        Dict[str, Any]]:
        """
        Process slide notes into structured elements.

        Args:
            notes_slide: The notes slide
            slide_idx: Slide index
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier

        Returns:
            List of notes-related elements
        """
        elements = []

        try:
            if not notes_slide or not hasattr(notes_slide, 'notes_text_frame'):
                return elements

            notes_text = notes_slide.notes_text_frame.text

            if not notes_text:
                return elements

            # Generate notes element ID
            notes_id = self._generate_id(f"notes_{slide_idx}_")

            # Create notes element
            notes_element = {
                "element_id": notes_id,
                "doc_id": doc_id,
                "element_type": "slide_notes",
                "parent_id": parent_id,
                "content_preview": notes_text[:100] + ("..." if len(notes_text) > 100 else ""),
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "slide_notes",
                    "slide_index": slide_idx
                }),
                "content_hash": self._generate_hash(notes_text),
                "metadata": {
                    "slide_index": slide_idx,
                    "text": notes_text
                }
            }
            elements.append(notes_element)

        except Exception as e:
            logger.warning(f"Error processing slide notes: {str(e)}")

        return elements

    def _process_comments(self, slide, slide_idx: int, doc_id: str, parent_id: str, source_id: str) -> List[
        Dict[str, Any]]:
        """
        Process slide comments into structured elements.

        Args:
            slide: The slide
            slide_idx: Slide index
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier

        Returns:
            List of comment-related elements
        """
        elements = []

        try:
            # Check if slide has comments
            if not hasattr(slide, 'comments') or not slide.comments:
                return elements

            # Create comments container element
            comments_id = self._generate_id(f"comments_{slide_idx}_")

            comments_element = {
                "element_id": comments_id,
                "doc_id": doc_id,
                "element_type": "comments_container",
                "parent_id": parent_id,
                "content_preview": f"Comments for Slide {slide_idx + 1}",
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "comments_container",
                    "slide_index": slide_idx
                }),
                "content_hash": self._generate_hash(f"comments_{slide_idx}"),
                "metadata": {
                    "slide_index": slide_idx,
                    "comment_count": len(slide.comments)
                }
            }
            elements.append(comments_element)

            # Process individual comments
            for comment_idx, comment in enumerate(slide.comments):
                comment_text = comment.text if hasattr(comment, 'text') else ""

                if not comment_text:
                    continue

                # Generate comment element ID
                comment_id = self._generate_id(f"comment_{slide_idx}_{comment_idx}_")

                # Get comment author and date if available
                author = comment.author if hasattr(comment, 'author') else "Unknown"
                date = comment.date if hasattr(comment, 'date') else None

                # Create comment element
                comment_element = {
                    "element_id": comment_id,
                    "doc_id": doc_id,
                    "element_type": "comment",
                    "parent_id": comments_id,
                    "content_preview": comment_text[:100] + ("..." if len(comment_text) > 100 else ""),
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "comment",
                        "slide_index": slide_idx,
                        "comment_index": comment_idx
                    }),
                    "content_hash": self._generate_hash(comment_text),
                    "metadata": {
                        "slide_index": slide_idx,
                        "comment_index": comment_idx,
                        "author": author,
                        "date": date.timestamp() if date and hasattr(date, 'timestamp') else None,
                        "text": comment_text
                    }
                }
                elements.append(comment_element)

        except Exception as e:
            logger.warning(f"Error processing slide comments: {str(e)}")

        return elements

    def _extract_slide_masters(self, presentation, doc_id: str, parent_id: str, source_id: str) -> List[Dict[str, Any]]:
        """
        Extract slide masters from presentation.

        Args:
            presentation: The PPTX presentation
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier

        Returns:
            List of slide master elements
        """
        elements = []

        try:
            if not hasattr(presentation, 'slide_masters'):
                return elements

            # Create masters container element
            masters_id = self._generate_id("masters_")

            masters_element = {
                "element_id": masters_id,
                "doc_id": doc_id,
                "element_type": "slide_masters",
                "parent_id": parent_id,
                "content_preview": "Slide Masters",
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "slide_masters"
                }),
                "content_hash": "",
                "metadata": {
                    "master_count": len(presentation.slide_masters)
                }
            }
            elements.append(masters_element)

            # Process individual masters
            for master_idx, master in enumerate(presentation.slide_masters):
                # Generate master ID
                master_id = self._generate_id(f"master_{master_idx}_")

                # Create master element
                master_element = {
                    "element_id": master_id,
                    "doc_id": doc_id,
                    "element_type": "slide_master",
                    "parent_id": masters_id,
                    "content_preview": f"Slide Master {master_idx + 1}",
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "slide_master",
                        "index": master_idx
                    }),
                    "content_hash": self._generate_hash(f"master_{master_idx}"),
                    "metadata": {
                        "index": master_idx,
                        "layout_count": len(master.slide_layouts) if hasattr(master, 'slide_layouts') else 0
                    }
                }
                elements.append(master_element)

                # Process master shapes if desired
                if self.extract_shapes and hasattr(master, 'shapes'):
                    shape_elements = self._process_shapes(master.shapes, doc_id, master_id, source_id, -1,
                                                          f"master_{master_idx}")
                    elements.extend(shape_elements)

        except Exception as e:
            logger.warning(f"Error extracting slide masters: {str(e)}")

        return elements

    def _extract_slide_templates(self, presentation, doc_id: str, parent_id: str, source_id: str) -> List[
        Dict[str, Any]]:
        """
        Extract slide templates (layouts) from presentation.

        Args:
            presentation: The PPTX presentation
            doc_id: Document ID
            parent_id: Parent element ID
            source_id: Source identifier

        Returns:
            List of slide template elements
        """
        elements = []

        try:
            # Collect all slide layouts from all masters
            layouts = []
            layout_names = set()

            if hasattr(presentation, 'slide_masters'):
                for master in presentation.slide_masters:
                    if hasattr(master, 'slide_layouts'):
                        for layout in master.slide_layouts:
                            # Avoid duplicates by name
                            layout_name = layout.name if hasattr(layout, 'name') else ""
                            if layout_name not in layout_names:
                                layouts.append(layout)
                                layout_names.add(layout_name)

            if not layouts:
                return elements

            # Create templates container element
            templates_id = self._generate_id("templates_")

            templates_element = {
                "element_id": templates_id,
                "doc_id": doc_id,
                "element_type": "slide_templates",
                "parent_id": parent_id,
                "content_preview": "Slide Templates",
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "slide_templates"
                }),
                "content_hash": "",
                "metadata": {
                    "template_count": len(layouts)
                }
            }
            elements.append(templates_element)

            # Process individual templates
            for layout_idx, layout in enumerate(layouts):
                # Generate layout ID
                layout_id = self._generate_id(f"layout_{layout_idx}_")

                # Get layout name
                layout_name = layout.name if hasattr(layout, 'name') else f"Layout {layout_idx + 1}"

                # Create layout element
                layout_element = {
                    "element_id": layout_id,
                    "doc_id": doc_id,
                    "element_type": "slide_layout",
                    "parent_id": templates_id,
                    "content_preview": layout_name,
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "slide_layout",
                        "index": layout_idx
                    }),
                    "content_hash": self._generate_hash(f"layout_{layout_idx}"),
                    "metadata": {
                        "index": layout_idx,
                        "name": layout_name
                    }
                }
                elements.append(layout_element)

                # Process layout shapes if desired
                if self.extract_shapes and hasattr(layout, 'shapes'):
                    shape_elements = self._process_shapes(layout.shapes, doc_id, layout_id, source_id, -1,
                                                          f"layout_{layout_idx}")
                    elements.extend(shape_elements)

        except Exception as e:
            logger.warning(f"Error extracting slide templates: {str(e)}")

        return elements

    def _extract_links(self, presentation, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract links from PowerPoint presentation.

        Args:
            presentation: The PPTX presentation
            elements: Document elements

        Returns:
            List of extracted links
        """
        links = []

        # Map element IDs to elements for quick lookup
        # element_map = {elem["element_id"]: elem for elem in elements}

        # Extract hyperlinks from text shapes
        for element in elements:
            if element["element_type"] in ["text_box", "paragraph", "table_cell"]:
                element_id = element["element_id"]
                text = element.get("metadata", {}).get("text", "")

                # Look for hyperlink patterns in text
                url_pattern = r'https?://[^\s<>)"\']+|www\.[^\s<>)"\']+|ftp://[^\s<>)"\']+|file://[^\s<>)"\']+|mailto:[^\s<>)"\']+|[^\s<>)"\']+\.(?:com|org|net|edu|gov|io|ai|app)[^\s<>)"\']*'
                urls = re.findall(url_pattern, text)

                for url in urls:
                    # Clean up URL
                    if url.startswith('www.'):
                        url = 'http://' + url

                    # Add link
                    links.append({
                        "source_id": element_id,
                        "link_text": url,
                        "link_target": url,
                        "link_type": "url"
                    })

                # Look for slide references (e.g., "See slide 5")
                slide_refs = re.findall(r'slide\s+(\d+)', text, re.IGNORECASE)

                for ref in slide_refs:
                    try:
                        slide_num = int(ref)
                        # Adjust for 0-based indexing
                        # slide_idx = slide_num - 1

                        # Find target slide element
                        target_slide = None
                        for slide_elem in elements:
                            if (slide_elem["element_type"] == "slide" and
                                    slide_elem.get("metadata", {}).get("number") == slide_num):
                                target_slide = slide_elem
                                break

                        if target_slide:
                            # Add link
                            links.append({
                                "source_id": element_id,
                                "link_text": f"Slide {slide_num}",
                                "link_target": target_slide["element_id"],
                                "link_type": "slide_reference"
                            })
                    except (ValueError, IndexError):
                        pass

        return links

    @staticmethod
    def _get_slide_title(slide: Slide) -> str:
        """Get slide title text."""
        try:
            # Look for a shape with placeholder type as title
            for shape in slide.shapes:
                if hasattr(shape, 'is_placeholder') and shape.is_placeholder:
                    if hasattr(shape, 'placeholder_format') and shape.placeholder_format.type == 1:  # 1 = TITLE
                        if hasattr(shape, 'text_frame') and shape.text_frame.text:
                            return shape.text_frame.text

            # If no title placeholder, look for first shape with text
            for shape in slide.shapes:
                if hasattr(shape, 'has_text_frame') and shape.has_text_frame:
                    if shape.text_frame.text:
                        return shape.text_frame.text

        except Exception as e:
            logger.debug(f"Error getting slide title: {str(e)}")

        return ""

    @staticmethod
    def _get_slide_layout_name(slide: Slide) -> str:
        """Get slide layout name."""
        try:
            if hasattr(slide, 'slide_layout') and hasattr(slide.slide_layout, 'name'):
                return slide.slide_layout.name
        except Exception as e:
            logger.debug(f"Error getting slide layout name: {str(e)}")

        return "Unknown Layout"

    @staticmethod
    def _is_title_shape(shape) -> bool:
        """Check if shape is a title placeholder."""
        try:
            if hasattr(shape, 'is_placeholder') and shape.is_placeholder:
                if hasattr(shape, 'placeholder_format') and shape.placeholder_format.type in [1,
                                                                                              2]:  # 1 = TITLE, 2 = CENTERED_TITLE
                    return True
        except Exception:
            pass

        return False

    @staticmethod
    def _get_paragraph_level(text_frame: TextFrame) -> int:
        """Get the outline level of text frame."""
        try:
            if hasattr(text_frame, 'paragraphs') and text_frame.paragraphs:
                return text_frame.paragraphs[0].level if hasattr(text_frame.paragraphs[0], 'level') else 0
        except Exception:
            pass

        return 0

    @staticmethod
    def _get_shape_type(shape) -> str:
        """Get the type of shape."""
        try:
            if hasattr(shape, 'shape_type'):
                return str(shape.shape_type)

            if isinstance(shape, Picture):
                return "picture"

            if hasattr(shape, 'has_table') and shape.has_table:
                return "table"

            if hasattr(shape, 'has_chart') and shape.has_chart:
                return "chart"

            if hasattr(shape, 'has_text_frame') and shape.has_text_frame:
                return "text"

            if isinstance(shape, GroupShape):
                return "group"

        except Exception:
            pass

        return "shape"
