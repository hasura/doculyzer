import uuid
from typing import Dict, Any, List

from .base import RelationshipDetector


class StructuralRelationshipDetector(RelationshipDetector):
    """Detector for structural relationships between elements."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the structural relationship detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    def detect_relationships(self, document: Dict[str, Any],
                             elements: List[Dict[str, Any]],
                             links: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Detect structural relationships between elements."""
        relationships = []
        doc_id = document["doc_id"]

        # Create element ID to element mapping for easier lookup
        # element_map = {element["element_id"]: element for element in elements}

        # Create parent-child mapping
        parent_children = {}
        for element in elements:
            parent_id = element.get("parent_id")
            element_id = element["element_id"]

            if parent_id:
                if parent_id not in parent_children:
                    parent_children[parent_id] = []

                parent_children[parent_id].append(element_id)

        # Create sibling relationships
        for parent_id, children in parent_children.items():
            # Skip if only one child
            if len(children) <= 1:
                continue

            # Create relationships between consecutive siblings
            for i in range(len(children) - 1):
                prev_id = children[i]
                next_id = children[i + 1]

                # Create relationships
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": prev_id,
                    "relationship_type": "next_sibling",
                    "target_reference": next_id,
                    "metadata": {
                        "confidence": 1.0
                    }
                }

                relationships.append(relationship)

                # Create reverse relationship
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": next_id,
                    "relationship_type": "previous_sibling",
                    "target_reference": prev_id,
                    "metadata": {
                        "confidence": 1.0
                    }
                }

                relationships.append(relationship)

        # Create section relationships (header -> content elements)
        # Find all headers
        headers = [element for element in elements if element.get("element_type") == "header"]

        # Sort headers by their level (h1, h2, etc.)
        headers.sort(key=lambda h: h.get("metadata", {}).get("level", 0))

        # Process each header
        for header in headers:
            header_id = header["element_id"]
            header_level = header.get("metadata", {}).get("level", 0)

            # Find elements that should be in this header's section
            section_elements = self._get_section_elements(header, headers, elements)

            # Create relationships from header to section elements
            for element_id in section_elements:
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": header_id,
                    "relationship_type": "contains",
                    "target_reference": element_id,
                    "metadata": {
                        "confidence": 1.0,
                        "section_level": header_level
                    }
                }

                relationships.append(relationship)

                # Create reverse relationship
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": element_id,
                    "relationship_type": "contained_by",
                    "target_reference": header_id,
                    "metadata": {
                        "confidence": 1.0,
                        "section_level": header_level
                    }
                }

                relationships.append(relationship)

        # Create table relationships (table -> rows -> cells)
        tables = [element for element in elements if element.get("element_type") == "table"]

        for table in tables:
            table_id = table["element_id"]

            # Find table rows
            rows = [element for element in elements
                    if element.get("element_type") == "table_row" and element.get("parent_id") == table_id]

            # Sort rows by index
            rows.sort(key=lambda r: r.get("metadata", {}).get("row", 0))

            # Create table -> row relationships
            for row in rows:
                row_id = row["element_id"]

                # Create relationship
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": table_id,
                    "relationship_type": "contains_row",
                    "target_reference": row_id,
                    "metadata": {
                        "confidence": 1.0,
                        "row_index": row.get("metadata", {}).get("row", 0)
                    }
                }

                relationships.append(relationship)

                # Find cells in this row
                cells = [element for element in elements
                         if element.get("element_type") in ("table_cell", "table_header")
                         and element.get("parent_id") == row_id]

                # Sort cells by column
                cells.sort(key=lambda c: c.get("metadata", {}).get("col", 0))

                # Create row -> cell relationships
                for cell in cells:
                    cell_id = cell["element_id"]

                    # Create relationship
                    relationship_id = self._generate_id("rel_")

                    relationship = {
                        "relationship_id": relationship_id,
                        "doc_id": doc_id,
                        "source_id": row_id,
                        "relationship_type": "contains_cell",
                        "target_reference": cell_id,
                        "metadata": {
                            "confidence": 1.0,
                            "col_index": cell.get("metadata", {}).get("col", 0)
                        }
                    }

                    relationships.append(relationship)

        # Create list relationships (list -> list items)
        lists = [element for element in elements if element.get("element_type") == "list"]

        for list_element in lists:
            list_id = list_element["element_id"]

            # Find list items
            items = [element for element in elements
                     if element.get("element_type") == "list_item" and element.get("parent_id") == list_id]

            # Sort items by index
            items.sort(key=lambda ii: ii.get("metadata", {}).get("index", 0))

            # Create list -> item relationships
            for item in items:
                item_id = item["element_id"]

                # Create relationship
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": list_id,
                    "relationship_type": "contains_item",
                    "target_reference": item_id,
                    "metadata": {
                        "confidence": 1.0,
                        "item_index": item.get("metadata", {}).get("index", 0),
                        "list_type": list_element.get("metadata", {}).get("list_type", "unordered")
                    }
                }

                relationships.append(relationship)

        # Create document -> element relationships for top-level elements
        root_elements = [element for element in elements if element.get("element_type") == "root"]
        if root_elements:
            root_id = root_elements[0]["element_id"]

            # Find all direct children of root
            root_children = [element["element_id"] for element in elements
                             if element.get("parent_id") == root_id and element.get("element_type") != "root"]

            for child_id in root_children:
                # Create relationship
                relationship_id = self._generate_id("rel_")

                relationship = {
                    "relationship_id": relationship_id,
                    "doc_id": doc_id,
                    "source_id": root_id,
                    "relationship_type": "contains",
                    "target_reference": child_id,
                    "metadata": {
                        "confidence": 1.0,
                        "top_level": True
                    }
                }

                relationships.append(relationship)

        return relationships

    @staticmethod
    def _get_section_elements(header: Dict[str, Any], _all_headers: List[Dict[str, Any]],
                              all_elements: List[Dict[str, Any]]) -> List[str]:
        """
        Get elements that belong to a header's section.

        A section includes all elements that:
        1. Come after this header
        2. Come before the next header of equal or higher level
        3. Are not headers themselves

        Args:
            header: Header element
            _all_headers: List of all headers
            all_elements: List of all elements

        Returns:
            List of element IDs in the section
        """
        header_id = header["element_id"]
        header_level = header.get("metadata", {}).get("level", 0)

        # Create list of elements in document order
        # This assumes elements are provided in document order
        element_ids = [e["element_id"] for e in all_elements]

        # Find index of this header
        try:
            header_index = element_ids.index(header_id)
        except ValueError:
            return []

        section_element_ids = []

        # Iterate through elements after this header
        for i in range(header_index + 1, len(element_ids)):
            element_id = element_ids[i]
            element = next((e for e in all_elements if e["element_id"] == element_id), None)

            if not element:
                continue

            # Stop at next header of equal or higher level
            if element.get("element_type") == "header":
                element_level = element.get("metadata", {}).get("level", 0)
                if element_level <= header_level:
                    break

            # Add non-header elements to section
            if element.get("element_type") != "header":
                section_element_ids.append(element_id)

        return section_element_ids

    @staticmethod
    def _generate_id(prefix: str = "") -> str:
        """
        Generate a unique ID.

        Args:
            prefix: Optional prefix for the ID

        Returns:
            Unique ID string
        """
        return f"{prefix}{uuid.uuid4()}"
