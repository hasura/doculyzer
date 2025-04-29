"""
CSV document parser module for the document pointer system.

This module parses CSV documents into structured elements.
"""

import csv
import io
import json
import logging
import os
import re
from typing import Dict, Any, Optional, List, Union, Tuple

from .base import DocumentParser

logger = logging.getLogger(__name__)


class CsvParser(DocumentParser):
    """Parser for CSV documents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the CSV parser."""
        super().__init__(config)
        # Configuration options
        self.config = config or {}
        self.max_content_preview = self.config.get("max_content_preview", 100)
        self.extract_header = self.config.get("extract_header", True)
        self.delimiter = self.config.get("delimiter", ",")
        self.quotechar = self.config.get("quotechar", '"')
        self.encoding = self.config.get("encoding", "utf-8")
        self.max_rows = self.config.get("max_rows", 1000)  # Limit for large files
        self.max_preview_columns = self.config.get("max_preview_columns", 5)
        self.detect_dialect = self.config.get("detect_dialect", True)
        self.strip_whitespace = self.config.get("strip_whitespace", True)

    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a CSV document into structured elements."""
        content = doc_content["content"]
        source_id = doc_content["id"]  # Should already be a fully qualified path
        metadata = doc_content.get("metadata", {}).copy()  # Make a copy to avoid modifying original

        # Generate document ID if not present
        doc_id = metadata.get("doc_id", self._generate_id("doc_"))

        # Parse the CSV content
        csv_data, dialect = self._parse_csv_content(content)

        # Update metadata with dialect information
        csv_metadata = self._extract_document_metadata(csv_data, dialect, metadata)

        # Create document record
        document = {
            "doc_id": doc_id,
            "doc_type": "csv",
            "source": source_id,
            "metadata": csv_metadata,
            "content_hash": doc_content.get("content_hash", self._generate_hash(content))
        }

        # Create root element
        elements: List = [self._create_root_element(doc_id, source_id)]
        root_id = elements[0]["element_id"]

        # Create table container element
        table_id = self._generate_id("csv_table_")
        table_element = {
            "element_id": table_id,
            "doc_id": doc_id,
            "element_type": "table",
            "parent_id": root_id,
            "content_preview": f"CSV table with {len(csv_data)} rows",
            "content_location": json.dumps({
                "source": source_id,
                "type": "table"
            }),
            "content_hash": self._generate_hash("csv_table"),
            "metadata": {
                "rows": len(csv_data),
                "columns": len(csv_data[0]) if csv_data else 0,
                "has_header": self.extract_header,
                "dialect": {
                    "delimiter": dialect.delimiter,
                    "quotechar": dialect.quotechar,
                    "doublequote": dialect.doublequote,
                    "escapechar": dialect.escapechar or "",
                    "lineterminator": dialect.lineterminator.replace("\r", "\\r").replace("\n", "\\n")
                }
            }
        }
        elements.append(table_element)

        # Process header row if present
        header_row = None
        if self.extract_header and csv_data:
            header_row = csv_data[0]
            header_id = self._generate_id("header_row_")

            # Create header preview
            header_preview = ", ".join(header_row[:self.max_preview_columns])
            if len(header_row) > self.max_preview_columns:
                header_preview += "..."

            header_element = {
                "element_id": header_id,
                "doc_id": doc_id,
                "element_type": "table_header_row",
                "parent_id": table_id,
                "content_preview": header_preview,
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "table_header_row",
                    "row": 0
                }),
                "content_hash": self._generate_hash(",".join(header_row)),
                "metadata": {
                    "row": 0,
                    "values": header_row,
                    "column_count": len(header_row)
                }
            }
            elements.append(header_element)

        # Process data rows
        data_start_idx = 1 if self.extract_header and csv_data else 0

        for row_idx, row in enumerate(csv_data[data_start_idx:data_start_idx + self.max_rows]):
            abs_row_idx = row_idx + data_start_idx
            row_id = self._generate_id(f"row_{abs_row_idx}_")

            # Create row preview
            row_preview = ", ".join(str(val) for val in row[:self.max_preview_columns])
            if len(row) > self.max_preview_columns:
                row_preview += "..."

            row_element = {
                "element_id": row_id,
                "doc_id": doc_id,
                "element_type": "table_row",
                "parent_id": table_id,
                "content_preview": row_preview,
                "content_location": json.dumps({
                    "source": source_id,
                    "type": "table_row",
                    "row": abs_row_idx
                }),
                "content_hash": self._generate_hash(",".join(str(val) for val in row)),
                "metadata": {
                    "row": abs_row_idx,
                    "values": row,
                    "column_count": len(row)
                }
            }
            elements.append(row_element)

            # Process cells in this row
            for col_idx, cell_value in enumerate(row):
                cell_id = self._generate_id(f"cell_{abs_row_idx}_{col_idx}_")

                # Get header name for this column if available
                header_name = header_row[col_idx] if header_row and col_idx < len(
                    header_row) else f"Column {col_idx + 1}"

                # Create cell preview
                cell_preview = str(cell_value)
                if len(cell_preview) > self.max_content_preview:
                    cell_preview = cell_preview[:self.max_content_preview] + "..."

                cell_element = {
                    "element_id": cell_id,
                    "doc_id": doc_id,
                    "element_type": "table_cell",
                    "parent_id": row_id,
                    "content_preview": cell_preview,
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": "table_cell",
                        "row": abs_row_idx,
                        "col": col_idx
                    }),
                    "content_hash": self._generate_hash(str(cell_value)),
                    "metadata": {
                        "row": abs_row_idx,
                        "col": col_idx,
                        "header": header_name,
                        "value": cell_value
                    }
                }
                elements.append(cell_element)

        # Extract any relationships (like column-to-column relationships)
        relationships = self._extract_relationships(csv_data, header_row, doc_id)

        # Return the parsed document
        return {
            "document": document,
            "elements": elements,
            "links": [],  # CSV typically doesn't have links, but we could extract URLs from cells
            "relationships": relationships
        }

    def _parse_csv_content(self, content: Union[str, bytes]) -> Tuple[List[List[str]], csv.Dialect]:
        """
        Parse CSV content into a list of rows and detect dialect.

        Args:
            content: CSV content as string or bytes

        Returns:
            Tuple of (list of rows, dialect)
        """
        # Ensure content is string
        if isinstance(content, bytes):
            try:
                content = content.decode(self.encoding)
            except UnicodeDecodeError:
                # Try different encodings
                encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        content = content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode CSV content with any known encoding")

        # Detect dialect if requested
        # dialect = None
        if self.detect_dialect:
            try:
                # Create a sample for dialect detection
                sample = content[:min(len(content), 8192)]  # Use first 8kb max for detection
                dialect = csv.Sniffer().sniff(sample)
                has_header = csv.Sniffer().has_header(sample)
                self.extract_header = has_header
            except Exception as e:
                logger.warning(f"Error detecting CSV dialect: {str(e)}. Using default.")
                dialect = csv.excel  # Use excel dialect as fallback
        else:
            # Create custom dialect with configured parameters
            class CustomDialect(csv.Dialect):
                delimiter = self.delimiter
                quotechar = self.quotechar
                escapechar = None
                doublequote = True
                skipinitialspace = True
                lineterminator = '\r\n'
                quoting = csv.QUOTE_MINIMAL

            dialect = CustomDialect

        # Parse CSV data
        csv_data = []
        try:
            csv_file = io.StringIO(content)
            reader = csv.reader(csv_file, dialect=dialect)

            # Read rows
            for row in reader:
                if self.strip_whitespace:
                    row = [cell.strip() if isinstance(cell, str) else cell for cell in row]
                csv_data.append(row)
        except Exception as e:
            logger.error(f"Error parsing CSV content: {str(e)}")
            raise ValueError(f"Error parsing CSV: {str(e)}")

        return csv_data, dialect

    def _extract_document_metadata(self, csv_data: List[List[str]], dialect: csv.Dialect,
                                   base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from CSV document.

        Args:
            csv_data: Parsed CSV data
            dialect: CSV dialect
            base_metadata: Base metadata from content source

        Returns:
            Dictionary of document metadata
        """
        metadata = base_metadata.copy()

        # Add CSV specific metadata
        metadata.update({
            "row_count": len(csv_data),
            "column_count": len(csv_data[0]) if csv_data else 0,
            "has_header": self.extract_header,
            "dialect": {
                "delimiter": dialect.delimiter,
                "quotechar": dialect.quotechar,
                "doublequote": dialect.doublequote,
                "escapechar": dialect.escapechar or "",
                "lineterminator": dialect.lineterminator.replace("\r", "\\r").replace("\n", "\\n")
            }
        })

        # Add header information if available
        if self.extract_header and csv_data:
            metadata["headers"] = csv_data[0]

            # Analyze data types for each column
            if len(csv_data) > 1:
                column_types = []
                for col_idx in range(len(csv_data[0])):
                    col_values = [row[col_idx] for row in csv_data[1:] if col_idx < len(row)]
                    col_type = self._detect_column_type(col_values)
                    column_types.append(col_type)
                metadata["column_types"] = column_types

        return metadata

    @staticmethod
    def _detect_column_type(values: List[str]) -> str:
        """
        Detect the data type of a column.

        Args:
            values: List of values in the column

        Returns:
            Detected data type ("integer", "float", "date", "boolean", "string")
        """
        # Skip empty values for type detection
        non_empty_values = [val for val in values if val]

        if not non_empty_values:
            return "string"

        # Check if all values are integers
        try:
            all(int(val) for val in non_empty_values)
            return "integer"
        except (ValueError, TypeError):
            pass

        # Check if all values are floats
        try:
            all(float(val) for val in non_empty_values)
            return "float"
        except (ValueError, TypeError):
            pass

        # Check if all values are booleans
        boolean_values = {"true", "false", "yes", "no", "1", "0", "y", "n"}
        if all(val.lower() in boolean_values for val in non_empty_values):
            return "boolean"

        # Check if all values match a date pattern (simplified)
        date_pattern = r'^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}$'
        if all(re.match(date_pattern, val) for val in non_empty_values):
            return "date"

        # Default to string
        return "string"

    @staticmethod
    def _extract_relationships(csv_data: List[List[str]], header_row: Optional[List[str]],
                               _doc_id: str) -> List[Dict[str, Any]]:
        """
        Extract relationships between columns in CSV data.

        Args:
            csv_data: Parsed CSV data
            header_row: CSV header row or None
            _doc_id: Document ID

        Returns:
            List of relationship dictionaries
        """
        relationships = []

        # Skip if no header or not enough data
        if not header_row or len(csv_data) < 2:
            return relationships

        # TODO: Implement more sophisticated relationship detection,
        # such as foreign key relationships, correlations, etc.

        return relationships

    def _resolve_element_content(self, location_data: Dict[str, Any],
                                 source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve content for specific CSV element types.

        Args:
            location_data: Content location data
            source_content: Optional pre-loaded source content

        Returns:
            Resolved content string
        """
        source = location_data.get("source", "")
        element_type = location_data.get("type", "")
        row = location_data.get("row")
        col = location_data.get("col")

        # Load content if not provided
        content = source_content
        if content is None:
            if os.path.exists(source):
                try:
                    with open(source, 'r', encoding=self.encoding) as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # Try different encodings
                    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
                    for encoding in encodings:
                        try:
                            with open(source, 'r', encoding=encoding) as f:
                                content = f.read()
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        with open(source, 'rb') as f:
                            content = f.read()
            else:
                raise ValueError(f"Source file not found: {source}")

        # Parse CSV
        csv_data, _ = self._parse_csv_content(content)

        # Handle different element types
        if element_type == "table":
            # Return the entire CSV as formatted text
            return "\n".join(",".join(row) for row in csv_data)

        elif element_type == "table_header_row" and row is not None:
            # Return the header row
            if row < len(csv_data):
                return ",".join(csv_data[row])
            return ""

        elif element_type == "table_row" and row is not None:
            # Return a specific row
            if row < len(csv_data):
                return ",".join(str(val) for val in csv_data[row])
            return ""

        elif element_type == "table_cell" and row is not None and col is not None:
            # Return a specific cell
            if row < len(csv_data) and col < len(csv_data[row]):
                return str(csv_data[row][col])
            return ""

        else:
            # Default: return full content
            return "\n".join(",".join(str(val) for val in row) for row in csv_data)

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
            element_type = location_data.get("type", "")

            # Check if source exists and is a file
            if not os.path.exists(source) or not os.path.isfile(source):
                return False

            # Check if element type is one we handle
            if element_type not in ["root", "table", "table_header_row", "table_row", "table_cell"]:
                return False

            # Check file extension for CSV
            _, ext = os.path.splitext(source.lower())
            return ext in ['.csv', '.tsv', '.txt']

        except (json.JSONDecodeError, TypeError):
            return False

    def _extract_links(self, content: str, element_id: str) -> List[Dict[str, Any]]:
        """
        Extract links from CSV content.

        Args:
            content: CSV content
            element_id: ID of the element containing the links

        Returns:
            List of extracted links
        """
        import re
        links = []

        # URL pattern for detection
        url_pattern = r'https?://[^\s,"\']+'

        # Parse CSV
        try:
            csv_data, _ = self._parse_csv_content(content)

            # Look for URLs in cells
            for row_idx, row in enumerate(csv_data):
                for col_idx, cell in enumerate(row):
                    if not isinstance(cell, str):
                        continue

                    # Find URLs in cell
                    urls = re.findall(url_pattern, cell)
                    for url in urls:
                        links.append({
                            "source_id": element_id,
                            "link_text": url,
                            "link_target": url,
                            "link_type": "url",
                            "metadata": {
                                "row": row_idx,
                                "col": col_idx
                            }
                        })
        except Exception as e:
            logger.warning(f"Error extracting links from CSV: {str(e)}")

        return links
