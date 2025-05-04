"""
CSV document parser module for the document pointer system.

This module parses CSV documents into structured elements with temporal semantics support.
"""

import csv
import io
import json
import logging
import os
import re
from typing import Dict, Any, Optional, List, Union, Tuple

from .base import DocumentParser
from .temporal_semantics import detect_temporal_type, TemporalType, create_semantic_temporal_expression
from ..relationships import RelationshipType
from ..storage import ElementType

logger = logging.getLogger(__name__)


class CsvParser(DocumentParser):
    """Parser for CSV documents with temporal semantics support."""

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
        self.enable_temporal_detection = self.config.get("enable_temporal_detection", True)

    @staticmethod
    def _is_identity_column(column_name: str) -> bool:
        """
        Determine if a column likely represents an identity or entity.

        Args:
            column_name: The name of the column

        Returns:
            True if it appears to be an identity column, False otherwise
        """
        # Common entity/identity fields
        common_identities = [
            # General identifiers
            "id", "identifier", "uuid", "guid", "key", "name", "title", "label",
            # People and organizations
            "person", "author", "publisher", "company", "organization", "user", "owner",
            "creator", "editor", "manager", "developer", "provider", "customer",
            # Places
            "country", "city", "state", "province", "location", "address", "region",
            # Descriptors
            "type", "category", "class", "genre", "style", "format", "model", "brand", "version"
        ]

        # Simple text matching approach
        column_lower = column_name.lower()

        # Check if it's in our list of common entities
        if column_lower in common_identities:
            return True
        # Check for possessive forms that suggest identity
        elif column_lower.endswith("'s"):
            base_word = column_lower[:-2]
            return base_word in common_identities
        # Check for compound words containing identity terms (e.g., userId, productName)
        else:
            return any(identity in column_lower and identity != column_lower for identity in common_identities)

    def _resolve_element_text(self, location_data: Dict[str, Any], source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve the plain text representation of a CSV element with temporal semantics.

        Args:
            location_data: Content location data
            source_content: Optional preloaded source content

        Returns:
            Plain text representation of the element
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
                return f"Source file not found: {source}"

        # Parse CSV
        csv_data, _ = self._parse_csv_content(content)

        # Get header row if available
        header_row = None
        if self.extract_header and csv_data:
            header_row = csv_data[0]

        # Handle different element types
        if element_type == ElementType.TABLE_CELL.value and row is not None and col is not None:
            # Return a specific cell with potential temporal semantics
            if row < len(csv_data) and col < len(csv_data[row]):
                cell_value = str(csv_data[row][col])

                # Apply temporal semantics for string values
                if self.enable_temporal_detection and isinstance(cell_value, str):
                    temporal_type = detect_temporal_type(cell_value)
                    if temporal_type is not TemporalType.NONE:
                        semantic_value = create_semantic_temporal_expression(cell_value)

                        # Format based on whether this is a header or data cell
                        if row == 0 and self.extract_header:
                            # For header cells, just return the original text
                            return cell_value
                        else:
                            # For data cells, include column name if available
                            if header_row and col < len(header_row):
                                column_name = header_row[col]

                                # Format based on column semantics
                                if self._is_identity_column(column_name):
                                    return f"{column_name} is {semantic_value}"
                                else:
                                    return f"{column_name}: {semantic_value}"
                            else:
                                return semantic_value

                # Non-temporal value or temporal detection disabled
                if row > 0 and header_row and col < len(header_row):
                    column_name = header_row[col]
                    if self._is_identity_column(column_name):
                        return f"{column_name} is \"{cell_value}\""
                    else:
                        return f"{column_name}: {cell_value}"
                else:
                    return cell_value
            return ""

        elif element_type == ElementType.TABLE_ROW.value and row is not None:
            # Return a specific row with potential temporal semantics
            if row < len(csv_data):
                result = []
                for col_idx, cell_value in enumerate(csv_data[row]):
                    cell_str = str(cell_value)

                    # Apply temporal semantics if this is a data row and not a header
                    if row > 0 and self.extract_header and self.enable_temporal_detection:
                        temporal_type = detect_temporal_type(cell_str)
                        if temporal_type is not TemporalType.NONE:
                            semantic_value = create_semantic_temporal_expression(cell_str)

                            # Include column name if available
                            if header_row and col_idx < len(header_row):
                                column_name = header_row[col_idx]

                                # Format based on column semantics
                                if self._is_identity_column(column_name):
                                    cell_str = f"{column_name} is {semantic_value}"
                                else:
                                    cell_str = f"{column_name}: {semantic_value}"
                            else:
                                cell_str = semantic_value
                        elif header_row and col_idx < len(header_row):
                            # For non-temporal values, include column name if available
                            column_name = header_row[col_idx]
                            if self._is_identity_column(column_name):
                                cell_str = f"{column_name} is \"{cell_str}\""
                            else:
                                cell_str = f"{column_name}: {cell_str}"

                    result.append(cell_str)

                return ", ".join(result)
            return ""

        elif element_type == ElementType.TABLE_HEADER_ROW.value and row is not None:
            # Return the header row (no temporal semantics needed for headers)
            if row < len(csv_data):
                return ", ".join(csv_data[row])
            return ""

        elif element_type == ElementType.TABLE.value:
            # For entire table, return a preview with first few rows
            result = []
            max_preview_rows = min(5, len(csv_data))

            if self.extract_header and csv_data:
                result.append(f"Headers: {', '.join(csv_data[0])}")

            result.append(f"Table with {len(csv_data)} rows and {len(csv_data[0]) if csv_data else 0} columns")

            if csv_data and max_preview_rows > (1 if self.extract_header else 0):
                result.append("Sample data:")
                start_idx = 1 if self.extract_header else 0
                for i in range(start_idx, min(start_idx + max_preview_rows, len(csv_data))):
                    result.append(f"Row {i}: {', '.join(str(val) for val in csv_data[i])}")

            return "\n".join(result)

        else:
            # Default: handle as entire document
            return self._resolve_element_content(location_data, source_content)

    def parse(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a CSV document into structured elements with temporal semantics."""
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

        # Initialize relationships list
        relationships = []

        # Create table container element
        table_id = self._generate_id("csv_table_")
        table_element = {
            "element_id": table_id,
            "doc_id": doc_id,
            "element_type": ElementType.TABLE.value,
            "parent_id": root_id,
            "content_preview": f"CSV table with {len(csv_data)} rows",
            "content_location": json.dumps({
                "source": source_id,
                "type": ElementType.TABLE.value
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

        # Create relationship from root to table
        contains_relationship = {
            "relationship_id": self._generate_id("rel_"),
            "source_id": root_id,
            "target_id": table_id,
            "relationship_type": RelationshipType.CONTAINS.value,
            "metadata": {
                "confidence": 1.0
            }
        }
        relationships.append(contains_relationship)

        # Create inverse relationship from table to root
        contained_by_relationship = {
            "relationship_id": self._generate_id("rel_"),
            "source_id": table_id,
            "target_id": root_id,
            "relationship_type": RelationshipType.CONTAINED_BY.value,
            "metadata": {
                "confidence": 1.0
            }
        }
        relationships.append(contained_by_relationship)

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
                "element_type": ElementType.TABLE_HEADER_ROW.value,
                "parent_id": table_id,
                "content_preview": header_preview,
                "content_location": json.dumps({
                    "source": source_id,
                    "type": ElementType.TABLE_HEADER_ROW.value,
                    "row": 0
                }),
                "content_hash": self._generate_hash(",".join(header_row)),
                "metadata": {
                    "row": 0,
                    "values": header_row,
                    "column_count": len(header_row),
                    "identity_columns": [i for i, name in enumerate(header_row) if self._is_identity_column(name)]
                }
            }
            elements.append(header_element)

            # Create relationship from table to header row
            contains_header_relationship = {
                "relationship_id": self._generate_id("rel_"),
                "source_id": table_id,
                "target_id": header_id,
                "relationship_type": RelationshipType.CONTAINS_TABLE_HEADER.value,
                "metadata": {
                    "confidence": 1.0,
                    "row": 0
                }
            }
            relationships.append(contains_header_relationship)

            # Create inverse relationship
            header_contained_relationship = {
                "relationship_id": self._generate_id("rel_"),
                "source_id": header_id,
                "target_id": table_id,
                "relationship_type": RelationshipType.CONTAINED_BY.value,
                "metadata": {
                    "confidence": 1.0
                }
            }
            relationships.append(header_contained_relationship)

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
                "element_type": ElementType.TABLE_ROW.value,
                "parent_id": table_id,
                "content_preview": row_preview,
                "content_location": json.dumps({
                    "source": source_id,
                    "type": ElementType.TABLE_ROW.value,
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

            # Create relationship from table to row
            contains_row_relationship = {
                "relationship_id": self._generate_id("rel_"),
                "source_id": table_id,
                "target_id": row_id,
                "relationship_type": RelationshipType.CONTAINS_TABLE_ROW.value,
                "metadata": {
                    "confidence": 1.0,
                    "row_index": abs_row_idx
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

                # Check for temporal data
                cell_str = str(cell_value)
                temporal_metadata = {}

                if self.enable_temporal_detection and isinstance(cell_str, str):
                    temporal_type = detect_temporal_type(cell_str)
                    if temporal_type is not TemporalType.NONE:
                        semantic_value = create_semantic_temporal_expression(cell_str)
                        temporal_metadata = {
                            "temporal_type": temporal_type.name,
                            "semantic_value": semantic_value
                        }

                        # Add indicator for temporal values in preview
                        cell_preview = f"[TIME] {cell_preview}"

                cell_element = {
                    "element_id": cell_id,
                    "doc_id": doc_id,
                    "element_type": ElementType.TABLE_CELL.value,
                    "parent_id": row_id,
                    "content_preview": cell_preview,
                    "content_location": json.dumps({
                        "source": source_id,
                        "type": ElementType.TABLE_CELL.value,
                        "row": abs_row_idx,
                        "col": col_idx
                    }),
                    "content_hash": self._generate_hash(str(cell_value)),
                    "metadata": {
                        "row": abs_row_idx,
                        "col": col_idx,
                        "header": header_name,
                        "value": cell_value,
                        "is_identity_column": self._is_identity_column(header_name) if header_name else False,
                        **temporal_metadata
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

                # Create column relationships if this is a data row
                if header_row and abs_row_idx > 0:
                    # Find the header cell for this column
                    header_cell_id = None
                    for element in elements:
                        if (element["element_type"] == ElementType.TABLE_CELL.value and
                                element.get("metadata", {}).get("row") == 0 and
                                element.get("metadata", {}).get("col") == col_idx):
                            header_cell_id = element["element_id"]
                            break

                    if header_cell_id:
                        # Create relationship from header cell to data cell
                        header_to_data_relationship = {
                            "relationship_id": self._generate_id("rel_"),
                            "source_id": header_cell_id,
                            "target_id": cell_id,
                            "relationship_type": RelationshipType.DESCRIBES.value,
                            "metadata": {
                                "confidence": 1.0,
                                "header_name": header_name
                            }
                        }
                        relationships.append(header_to_data_relationship)

                        # Create inverse relationship
                        data_to_header_relationship = {
                            "relationship_id": self._generate_id("rel_"),
                            "source_id": cell_id,
                            "target_id": header_cell_id,
                            "relationship_type": RelationshipType.DESCRIBED_BY.value,
                            "metadata": {
                                "confidence": 1.0,
                                "header_name": header_name
                            }
                        }
                        relationships.append(data_to_header_relationship)

        # Add additional column-based relationships if needed
        column_relationships = self._extract_column_relationships(csv_data, header_row, elements, doc_id)
        relationships.extend(column_relationships)

        # Return the parsed document
        return {
            "document": document,
            "elements": elements,
            "links": self._extract_links(content, root_id),
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
        Extract metadata from CSV document with temporal semantics information.

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

            # Identify identity columns
            metadata["identity_columns"] = [
                i for i, name in enumerate(csv_data[0]) if self._is_identity_column(name)
            ]

            # Analyze data types for each column
            if len(csv_data) > 1:
                column_types = []
                temporal_columns = []

                for col_idx in range(len(csv_data[0])):
                    col_values = [row[col_idx] for row in csv_data[1:] if col_idx < len(row)]
                    col_type = self._detect_column_type(col_values)
                    column_types.append(col_type)

                    # Detect if this is likely a temporal column
                    if self.enable_temporal_detection and col_type in ["date", "string"]:
                        # Check sample of values for temporal data
                        sample_size = min(10, len(col_values))
                        temporal_count = 0

                        for val in col_values[:sample_size]:
                            if isinstance(val, str) and detect_temporal_type(val) is not TemporalType.NONE:
                                temporal_count += 1

                        # If more than half of samples are temporal, consider it a temporal column
                        if temporal_count > sample_size / 2:
                            temporal_columns.append(col_idx)

                metadata["column_types"] = column_types
                if temporal_columns:
                    metadata["temporal_columns"] = temporal_columns

        return metadata

    def _detect_column_type(self, values: List[str]) -> str:
        """
        Detect the data type of column with enhanced temporal detection.

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

        # Enhanced temporal detection
        if self.enable_temporal_detection:
            # Sample the first few values to check for temporal data
            sample_size = min(5, len(non_empty_values))
            temporal_count = 0

            for val in non_empty_values[:sample_size]:
                if detect_temporal_type(val) is not TemporalType.NONE:
                    temporal_count += 1

            # If most sample values are temporal, classify as date
            if temporal_count >= sample_size * 0.6:
                return "date"

        # Check date pattern (simple date pattern for traditional date detection)
        date_pattern = r'^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}$'
        if all(re.match(date_pattern, val) for val in non_empty_values):
            return "date"

        # Default to string
        return "string"

    def _extract_column_relationships(self, csv_data: List[List[str]], header_row: Optional[List[str]],
                                      elements: List[Dict[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
        """
        Extract relationships between columns in CSV data, including temporal relationships.

        Args:
            csv_data: Parsed CSV data
            header_row: CSV header row or None
            elements: List of elements already created
            doc_id: Document ID

        Returns:
            List of column relationship dictionaries
        """
        relationships = []

        # Skip if no header or not enough data
        if not header_row or len(csv_data) < 2:
            return relationships

        # Create a map of column indices to potential "key" columns
        # Heuristic: columns with names containing 'id', 'key', 'code' are potential keys
        potential_keys = []
        for col_idx, header in enumerate(header_row):
            if any(key_term in header.lower() for key_term in ['id', 'key', 'code', 'num', 'number']):
                potential_keys.append(col_idx)

        # Identify temporal columns
        temporal_columns = []
        if self.enable_temporal_detection:
            for col_idx, header in enumerate(header_row):
                # Skip first row (header)
                col_values = [row[col_idx] for row in csv_data[1:] if col_idx < len(row)]

                # Check if this column contains temporal data
                sample_size = min(10, len(col_values))
                temporal_count = 0

                for val in col_values[:sample_size]:
                    if isinstance(val, str) and detect_temporal_type(val) is not TemporalType.NONE:
                        temporal_count += 1

                # If sufficient proportion of values are temporal, consider it a temporal column
                if temporal_count > sample_size * 0.6:
                    temporal_columns.append(col_idx)

                    # Add "date" or "time" related terms for relationship detection
                    if any(date_term in header.lower() for date_term in ['date', 'time', 'day', 'month', 'year']):
                        # This is explicitly a date/time column
                        logger.debug(f"Identified explicit temporal column: {header}")

        # For each potential key column, check if there are other columns that might be related
        for key_col in potential_keys:
            # Find the element ID for this column's header
            key_header_id = None
            key_header_name = header_row[key_col]

            for element in elements:
                if (element["element_type"] == ElementType.TABLE_CELL.value and
                        element.get("metadata", {}).get("row") == 0 and
                        element.get("metadata", {}).get("col") == key_col):
                    key_header_id = element["element_id"]
                    break

            if not key_header_id:
                continue

            # Look for columns that might be related to this key column
            for col_idx, header in enumerate(header_row):
                if col_idx == key_col:
                    continue

                # Check if this column appears to be related to the key column
                # Examples: 'customer_id' and 'customer_name', 'product_code' and 'product_description'
                key_terms = key_header_name.replace('_', ' ').replace('-', ' ').split()
                header_terms = header.replace('_', ' ').replace('-', ' ').split()

                # If they share some terms (but not the key term itself), they might be related
                common_terms = set(key_terms) & set(header_terms)
                key_related_terms = [term for term in common_terms if
                                     term.lower() not in ['id', 'key', 'code', 'num', 'number']]

                if key_related_terms:
                    # Find the element ID for this potentially related column's header
                    related_header_id = None
                    for element in elements:
                        if (element["element_type"] == ElementType.TABLE_CELL.value and
                                element.get("metadata", {}).get("row") == 0 and
                                element.get("metadata", {}).get("col") == col_idx):
                            related_header_id = element["element_id"]
                            break

                    if not related_header_id:
                        continue

                    # Create a relationship between these two columns
                    relationship_type = RelationshipType.RELATED_TO.value
                    confidence = 0.7  # Default confidence

                    # Special case for temporal relationships
                    if col_idx in temporal_columns and any(time_term in header.lower() for time_term in
                                                           ['date', 'time', 'created', 'updated', 'modified']):
                        relationship_type = RelationshipType.TEMPORAL_RELATIONSHIP.value
                        confidence = 0.85  # Higher confidence for temporal relationships

                    key_to_related_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": key_header_id,
                        "target_id": related_header_id,
                        "relationship_type": relationship_type,
                        "metadata": {
                            "confidence": confidence,
                            "common_terms": key_related_terms,
                            "key_column": True,
                            "is_temporal": col_idx in temporal_columns
                        }
                    }
                    relationships.append(key_to_related_relationship)

                    # Create inverse relationship
                    inverse_relationship_type = relationship_type
                    if relationship_type == RelationshipType.TEMPORAL_RELATIONSHIP.value:
                        # For temporal relationships, specify direction
                        inverse_relationship_type = RelationshipType.HAS_TEMPORAL_INFORMATION.value

                    related_to_key_relationship = {
                        "relationship_id": self._generate_id("rel_"),
                        "source_id": related_header_id,
                        "target_id": key_header_id,
                        "relationship_type": inverse_relationship_type,
                        "metadata": {
                            "confidence": confidence,
                            "common_terms": key_related_terms,
                            "key_column": False,
                            "is_temporal": col_idx in temporal_columns
                        }
                    }
                    relationships.append(related_to_key_relationship)

        # Add relationships between temporal columns (if multiple exist)
        if len(temporal_columns) > 1:
            # Create relationships between temporal columns
            for i, col1 in enumerate(temporal_columns):
                for col2 in temporal_columns[i + 1:]:
                    # Find the element IDs for these columns' headers
                    col1_header_id = None
                    col2_header_id = None

                    for element in elements:
                        if element["element_type"] == ElementType.TABLE_CELL.value and element.get("metadata", {}).get(
                                "row") == 0:
                            if element.get("metadata", {}).get("col") == col1:
                                col1_header_id = element["element_id"]
                            elif element.get("metadata", {}).get("col") == col2:
                                col2_header_id = element["element_id"]

                    if col1_header_id and col2_header_id:
                        # Create relationship between temporal columns
                        temp_relationship = {
                            "relationship_id": self._generate_id("rel_"),
                            "source_id": col1_header_id,
                            "target_id": col2_header_id,
                            "relationship_type": RelationshipType.TEMPORAL_RELATIONSHIP.value,
                            "metadata": {
                                "confidence": 0.75,
                                "related_temporal_columns": True
                            }
                        }
                        relationships.append(temp_relationship)

                        # Create inverse relationship
                        inverse_temp_relationship = {
                            "relationship_id": self._generate_id("rel_"),
                            "source_id": col2_header_id,
                            "target_id": col1_header_id,
                            "relationship_type": RelationshipType.TEMPORAL_RELATIONSHIP.value,
                            "metadata": {
                                "confidence": 0.75,
                                "related_temporal_columns": True
                            }
                        }
                        relationships.append(inverse_temp_relationship)

        return relationships

    def _resolve_element_content(self, location_data: Dict[str, Any],
                                 source_content: Optional[Union[str, bytes]]) -> str:
        """
        Resolve content for specific CSV element types with temporal semantics.

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

        # Get header row if available
        header_row = None
        if self.extract_header and csv_data:
            header_row = csv_data[0]

        # Handle different element types
        if element_type == ElementType.TABLE.value:
            # For table, provide a more structured representation
            result = {
                "type": "table",
                "headers": csv_data[0] if self.extract_header and csv_data else [],
                "rows": csv_data[1:] if self.extract_header and csv_data else csv_data,
                "row_count": len(csv_data),
                "column_count": len(csv_data[0]) if csv_data else 0
            }

            # If temporal semantics is enabled, add enriched data
            if self.enable_temporal_detection and header_row:
                # Identify temporal columns
                temporal_columns = []
                for col_idx, header in enumerate(header_row):
                    # Check sample of data rows
                    sample_size = min(5, len(csv_data) - 1)
                    temporal_count = 0

                    for row_idx in range(1, 1 + sample_size):
                        if row_idx < len(csv_data) and col_idx < len(csv_data[row_idx]):
                            val = csv_data[row_idx][col_idx]
                            if isinstance(val, str) and detect_temporal_type(val) is not TemporalType.NONE:
                                temporal_count += 1

                    if temporal_count > sample_size * 0.6:
                        temporal_columns.append(col_idx)

                if temporal_columns:
                    result["temporal_columns"] = temporal_columns

                    # Add semantic representations for these columns
                    temporal_data = {}
                    for col_idx in temporal_columns:
                        header_name = header_row[col_idx]
                        semantic_values = []

                        for row_idx in range(1, len(csv_data)):
                            if col_idx < len(csv_data[row_idx]):
                                val = csv_data[row_idx][col_idx]
                                if isinstance(val, str):
                                    temporal_type = detect_temporal_type(val)
                                    if temporal_type is not TemporalType.NONE:
                                        semantic_values.append({
                                            "original": val,
                                            "semantic": create_semantic_temporal_expression(val),
                                            "temporal_type": temporal_type.name
                                        })
                                    else:
                                        semantic_values.append({"original": val})

                        temporal_data[header_name] = semantic_values

                    result["temporal_data"] = temporal_data

            return json.dumps(result, indent=2)

        elif element_type == ElementType.TABLE_HEADER_ROW.value and row is not None:
            # Return the header row (no temporal semantics needed for headers)
            if row < len(csv_data):
                result = {
                    "type": "header_row",
                    "values": csv_data[row]
                }
                return json.dumps(result, indent=2)
            return json.dumps({"error": "Row index out of range"})

        elif element_type == ElementType.TABLE_ROW.value and row is not None:
            # Return a specific row with potential temporal semantics
            if row < len(csv_data):
                result = {
                    "type": "row",
                    "row_index": row,
                    "values": csv_data[row]
                }

                # If this is a data row and temporal semantics are enabled
                if self.enable_temporal_detection and row > 0 and header_row:
                    temporal_values = {}

                    for col_idx, cell_value in enumerate(csv_data[row]):
                        if col_idx < len(header_row):
                            header_name = header_row[col_idx]

                            if isinstance(cell_value, str):
                                temporal_type = detect_temporal_type(cell_value)
                                if temporal_type is not TemporalType.NONE:
                                    temporal_values[header_name] = {
                                        "value": cell_value,
                                        "semantic": create_semantic_temporal_expression(cell_value),
                                        "temporal_type": temporal_type.name
                                    }

                    if temporal_values:
                        result["temporal_values"] = temporal_values

                return json.dumps(result, indent=2)
            return json.dumps({"error": "Row index out of range"})

        elif element_type == ElementType.TABLE_CELL.value and row is not None and col is not None:
            # Return a specific cell with potential temporal semantics
            if row < len(csv_data) and col < len(csv_data[row]):
                cell_value = csv_data[row][col]
                result = {
                    "type": "cell",
                    "row": row,
                    "col": col,
                    "value": cell_value
                }

                # Add header information if available
                if header_row and row > 0 and col < len(header_row):
                    result["header"] = header_row[col]
                    result["is_identity_column"] = self._is_identity_column(header_row[col])

                # Add temporal semantics if applicable
                if self.enable_temporal_detection and isinstance(cell_value, str):
                    temporal_type = detect_temporal_type(cell_value)
                    if temporal_type is not TemporalType.NONE:
                        result["temporal"] = {
                            "type": temporal_type.name,
                            "semantic": create_semantic_temporal_expression(cell_value)
                        }

                return json.dumps(result, indent=2)
            return json.dumps({"error": "Cell coordinates out of range"})

        else:
            # Default: return the raw content
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
            element_type = location_data.get("type", "")

            # Check if source exists and is a file
            if not os.path.exists(source) or not os.path.isfile(source):
                return False

            # Check if element type is one we handle
            if element_type not in ["root", ElementType.TABLE.value, ElementType.TABLE_HEADER_ROW.value,
                                    ElementType.TABLE_ROW.value, ElementType.TABLE_CELL.value]:
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

    def _create_root_element(self, doc_id: str, source_id: str) -> Dict[str, Any]:
        """
        Create a document root element.

        Args:
            doc_id: Document ID
            source_id: Source identifier

        Returns:
            Document root element dictionary
        """
        root_id = self._generate_id("doc_root_")
        root_element = {
            "element_id": root_id,
            "doc_id": doc_id,
            "element_type": ElementType.ROOT.value,
            "parent_id": None,
            "content_preview": f"Document: {source_id}",
            "content_location": json.dumps({
                "source": source_id,
                "type": ElementType.ROOT.value
            }),
            "content_hash": self._generate_hash(source_id),
            "metadata": {
                "source_id": source_id,
                "path": "/"
            }
        }
        return root_element

    @staticmethod
    def _generate_id(prefix: str = "") -> str:
        """Generate a unique ID with optional prefix."""
        import uuid
        return f"{prefix}{uuid.uuid4()}"

    @staticmethod
    def _generate_hash(content: Union[str, bytes]) -> str:
        """Generate a hash of content for change detection."""
        import hashlib
        if isinstance(content, str):
            return hashlib.md5(content.encode('utf-8')).hexdigest()
        elif isinstance(content, bytes):
            return hashlib.md5(content).hexdigest()
        else:
            # Convert any other type to string first
            return hashlib.md5(str(content).encode('utf-8')).hexdigest()
