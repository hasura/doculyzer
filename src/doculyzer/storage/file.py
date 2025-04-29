import glob
import json
import logging
import os
import time
from typing import Optional, Dict, Any, List, Tuple

from .base import DocumentDatabase

logger = logging.getLogger(__name__)


class FileDocumentDatabase(DocumentDatabase):
    """File-based implementation of document database."""

    def __init__(self, storage_path: str):
        """
        Initialize file-based document database.

        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = storage_path
        self.documents = {}
        self.elements = {}
        self.element_pks = {}  # Map element_id to element_pk
        self.next_element_pk = 1  # Starting auto-increment value
        self.relationships = {}
        self.embeddings = {}
        self.processing_history = {}  # New dictionary to track processing history

    def initialize(self) -> None:
        """Initialize the database by loading existing data."""
        os.makedirs(self.storage_path, exist_ok=True)

        # Create subdirectories
        os.makedirs(os.path.join(self.storage_path, 'documents'), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, 'elements'), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, 'relationships'), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, 'embeddings'), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, 'processing_history'), exist_ok=True)  # New directory

        # Load existing data if available
        self._load_documents()
        self._load_elements()
        self._load_relationships()
        self._load_embeddings()
        self._load_processing_history()  # Load processing history

        logger.info(f"Loaded {len(self.documents)} documents, "
                    f"{len(self.elements)} elements, "
                    f"{len(self.relationships)} relationships, "
                    f"{len(self.embeddings)} embeddings, "
                    f"{len(self.processing_history)} processing history records")

    def close(self) -> None:
        """Close the database (no-op for file-based database)."""
        pass

    def get_last_processed_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about when a document was last processed."""
        # Convert source_id to a safe filename
        safe_id = self._get_safe_filename(source_id)

        # Look up in processing history
        return self.processing_history.get(safe_id)

    def update_processing_history(self, source_id: str, content_hash: str) -> None:
        """Update the processing history for a document."""
        # Convert source_id to a safe filename
        safe_id = self._get_safe_filename(source_id)

        # Create history record
        history_record = {
            "source_id": source_id,
            "content_hash": content_hash,
            "last_modified": time.time(),
            "processing_count": self.processing_history.get(safe_id, {}).get("processing_count", 0) + 1
        }

        # Store in memory and on disk
        self.processing_history[safe_id] = history_record
        self._save_processing_history(safe_id)

        logger.debug(f"Updated processing history for {source_id}")

    def store_document(self, document: Dict[str, Any], elements: List[Dict[str, Any]],
                       relationships: List[Dict[str, Any]]) -> None:
        """Store a document with its elements and relationships."""
        doc_id = document["doc_id"]
        source_id = document.get("source", "")
        content_hash = document.get("content_hash", "")

        # Store document
        self.documents[doc_id] = document
        self._save_document(doc_id)

        # Store elements
        for element in elements:
            element_id = element["element_id"]

            # Assign an auto-increment element_pk
            element_pk = self.next_element_pk
            self.next_element_pk += 1
            element["element_pk"] = element_pk

            # Store mapping
            self.element_pks[element_id] = element_pk

            # Store element
            self.elements[element_id] = element
            self._save_element(element_id)

        # Store relationships
        for relationship in relationships:
            relationship_id = relationship["relationship_id"]
            self.relationships[relationship_id] = relationship
            self._save_relationship(relationship_id)

        # Update processing history
        if source_id:
            self.update_processing_history(source_id, content_hash)

    def update_document(self, doc_id: str, document: Dict[str, Any],
                        elements: List[Dict[str, Any]],
                        relationships: List[Dict[str, Any]]) -> None:
        """Update an existing document."""
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        source_id = document.get("source", "")
        content_hash = document.get("content_hash", "")

        # Update document
        self.documents[doc_id] = document
        self._save_document(doc_id)

        # Get existing elements and relationships for this document
        existing_elements = {e["element_id"]: e for e in self.get_document_elements(doc_id)}
        existing_relationships = {r["relationship_id"]: r
                                  for r in self.get_document_relationships(doc_id)}

        # Update elements - retain unchanged ones
        updated_element_ids = set()
        for element in elements:
            element_id = element["element_id"]
            updated_element_ids.add(element_id)

            if element_id in existing_elements:
                # Check if element has changed
                if self._has_element_changed(element, existing_elements[element_id]):
                    # Preserve the element_pk from the existing element
                    element["element_pk"] = existing_elements[element_id].get("element_pk")
                    self.elements[element_id] = element
                    self._save_element(element_id)
            else:
                # New element - assign an auto-increment element_pk
                element_pk = self.next_element_pk
                self.next_element_pk += 1
                element["element_pk"] = element_pk

                # Store mapping
                self.element_pks[element_id] = element_pk

                # Store element
                self.elements[element_id] = element
                self._save_element(element_id)

        # Remove elements that no longer exist
        for element_id in existing_elements:
            if element_id not in updated_element_ids:
                if element_id in self.elements:
                    # Get the element_pk for embedding deletion
                    element_pk = self.elements[element_id].get("element_pk")

                    # Delete element
                    del self.elements[element_id]
                    self._delete_element_file(element_id)

                    # Remove from element_pks mapping
                    if element_id in self.element_pks:
                        del self.element_pks[element_id]

                    # Also remove any embeddings
                    if element_pk in self.embeddings:
                        del self.embeddings[element_pk]
                        self._delete_embedding_file(element_pk)

        # Update relationships - similar approach
        updated_relationship_ids = set()
        for relationship in relationships:
            relationship_id = relationship["relationship_id"]
            updated_relationship_ids.add(relationship_id)

            if relationship_id in existing_relationships:
                # Check if relationship has changed
                if self._has_relationship_changed(relationship, existing_relationships[relationship_id]):
                    self.relationships[relationship_id] = relationship
                    self._save_relationship(relationship_id)
            else:
                # New relationship
                self.relationships[relationship_id] = relationship
                self._save_relationship(relationship_id)

        # Remove relationships that no longer exist
        for relationship_id in existing_relationships:
            if relationship_id not in updated_relationship_ids:
                if relationship_id in self.relationships:
                    del self.relationships[relationship_id]
                    self._delete_relationship_file(relationship_id)

        # Update processing history
        if source_id:
            self.update_processing_history(source_id, content_hash)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        return self.documents.get(doc_id)

    def get_document_elements(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get elements for a document."""
        return [element for element in self.elements.values()
                if element.get("doc_id") == doc_id]

    def get_document_relationships(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get relationships for a document."""
        # First get all element IDs for the document
        element_ids = {element["element_id"] for element in self.get_document_elements(doc_id)}

        # Find relationships involving these elements
        return [relationship for relationship in self.relationships.values()
                if relationship.get("source_id") in element_ids]

    def get_element(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Get element by ID."""
        return self.elements.get(element_id)

    def find_documents(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find documents matching query."""
        if query is None:
            query = {}

        results = []

        for doc in self.documents.values():
            match = True

            for key, value in query.items():
                if key == "metadata":
                    # Check metadata fields
                    for meta_key, meta_value in value.items():
                        if meta_key not in doc.get("metadata", {}) or doc["metadata"][meta_key] != meta_value:
                            match = False
                            break
                elif key not in doc or doc[key] != value:
                    match = False
                    break

            if match:
                results.append(doc)
                if len(results) >= limit:
                    break

        return results

    def find_elements(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find elements matching query."""
        if query is None:
            query = {}

        results = []

        for element in self.elements.values():
            match = True

            for key, value in query.items():
                if key == "metadata":
                    # Check metadata fields
                    for meta_key, meta_value in value.items():
                        if meta_key not in element.get("metadata", {}) or element["metadata"][meta_key] != meta_value:
                            match = False
                            break
                elif key not in element or element[key] != value:
                    match = False
                    break

            if match:
                results.append(element)
                if len(results) >= limit:
                    break

        return results

    def search_elements_by_content(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search elements by content preview."""
        results = []
        search_text_lower = search_text.lower()

        for element in self.elements.values():
            content_preview = element.get("content_preview", "").lower()

            if search_text_lower in content_preview:
                results.append(element)

                if len(results) >= limit:
                    break

        return results

    def store_embedding(self, element_pk: int, embedding: List[float]) -> None:
        """Store embedding for an element."""
        # Verify element pk exists in some element
        found = False
        for element in self.elements.values():
            if element.get("element_pk") == element_pk:
                found = True
                break

        if not found:
            raise ValueError(f"Element pk not found: {element_pk}")

        self.embeddings[element_pk] = embedding
        self._save_embedding(element_pk)

    def get_embedding(self, element_pk: int) -> Optional[List[float]]:
        """Get embedding for an element."""
        return self.embeddings.get(element_pk)

    def search_by_embedding(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[str, float]]:
        """Search elements by embedding similarity."""
        import numpy as np

        query_embedding_np = np.array(query_embedding)
        results = []

        for element_pk, embedding in self.embeddings.items():
            embedding_np = np.array(embedding)

            # Calculate cosine similarity
            similarity = np.dot(query_embedding_np, embedding_np) / (
                    np.linalg.norm(query_embedding_np) * np.linalg.norm(embedding_np))

            # Find the element_id for this element_pk
            element_id = None
            for eid, epk in self.element_pks.items():
                if epk == element_pk:
                    element_id = eid
                    break

            if element_id:
                results.append((element_id, float(similarity)))

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all associated elements and relationships."""
        if doc_id not in self.documents:
            return False

        # Get elements to delete
        elements_to_delete = self.get_document_elements(doc_id)
        element_ids = {element["element_id"] for element in elements_to_delete}
        element_pks = {element.get("element_pk") for element in elements_to_delete if
                       element.get("element_pk") is not None}

        # Get relationships to delete
        relationships_to_delete = self.get_document_relationships(doc_id)
        relationship_ids = {rel["relationship_id"] for rel in relationships_to_delete}

        # Delete elements
        for element_id in element_ids:
            if element_id in self.elements:
                # Remove element_pk mapping
                if element_id in self.element_pks:
                    del self.element_pks[element_id]

                # Delete element
                del self.elements[element_id]
                self._delete_element_file(element_id)

        # Delete embeddings
        for element_pk in element_pks:
            if element_pk in self.embeddings:
                del self.embeddings[element_pk]
                self._delete_embedding_file(element_pk)

        # Delete relationships
        for relationship_id in relationship_ids:
            if relationship_id in self.relationships:
                del self.relationships[relationship_id]
                self._delete_relationship_file(relationship_id)

        # Delete document
        del self.documents[doc_id]
        self._delete_document_file(doc_id)

        return True

    def _load_processing_history(self) -> None:
        """Load processing history from files."""
        history_files = glob.glob(os.path.join(self.storage_path, 'processing_history', '*.json'))

        for file_path in history_files:
            try:
                with open(file_path, 'r') as f:
                    history = json.load(f)

                # Use the safe filename as the key
                filename = os.path.basename(file_path)
                safe_id = os.path.splitext(filename)[0]
                self.processing_history[safe_id] = history

            except Exception as e:
                logger.error(f"Error loading processing history from {file_path}: {str(e)}")

    def _save_processing_history(self, safe_id: str) -> None:
        """Save processing history to file."""
        if safe_id not in self.processing_history:
            return

        file_path = os.path.join(self.storage_path, 'processing_history', f"{safe_id}.json")

        try:
            with open(file_path, 'w') as f:
                json.dump(self.processing_history[safe_id], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving processing history to {file_path}: {str(e)}")

    @staticmethod
    def _get_safe_filename(source_id: str) -> str:
        """Convert source_id to a safe filename by hashing it."""
        import hashlib
        return hashlib.md5(source_id.encode('utf-8')).hexdigest()

    def _load_documents(self) -> None:
        """Load documents from files."""
        document_files = glob.glob(os.path.join(self.storage_path, 'documents', '*.json'))

        for file_path in document_files:
            try:
                with open(file_path, 'r') as f:
                    document = json.load(f)

                if "doc_id" in document:
                    self.documents[document["doc_id"]] = document
            except Exception as e:
                logger.error(f"Error loading document from {file_path}: {str(e)}")

    def _load_elements(self) -> None:
        """Load elements from files."""
        element_files = glob.glob(os.path.join(self.storage_path, 'elements', '*.json'))
        max_pk = 0  # Track the highest element_pk

        for file_path in element_files:
            try:
                with open(file_path, 'r') as f:
                    element = json.load(f)

                if "element_id" in element:
                    # Ensure element has element_pk
                    if "element_pk" not in element:
                        element["element_pk"] = self.next_element_pk
                        self.next_element_pk += 1

                    # Track the highest element_pk
                    max_pk = max(max_pk, element["element_pk"])

                    # Store element and mapping
                    self.elements[element["element_id"]] = element
                    self.element_pks[element["element_id"]] = element["element_pk"]
            except Exception as e:
                logger.error(f"Error loading element from {file_path}: {str(e)}")

        # Update next_element_pk to be one more than the highest seen
        self.next_element_pk = max_pk + 1

    def _load_relationships(self) -> None:
        """Load relationships from files."""
        relationship_files = glob.glob(os.path.join(self.storage_path, 'relationships', '*.json'))

        for file_path in relationship_files:
            try:
                with open(file_path, 'r') as f:
                    relationship = json.load(f)

                if "relationship_id" in relationship:
                    self.relationships[relationship["relationship_id"]] = relationship
            except Exception as e:
                logger.error(f"Error loading relationship from {file_path}: {str(e)}")

    def _load_embeddings(self) -> None:
        """Load embeddings from files."""
        import numpy as np

        embedding_files = glob.glob(os.path.join(self.storage_path, 'embeddings', '*.npy'))

        for file_path in embedding_files:
            try:
                # Extract element_pk from filename
                filename = os.path.basename(file_path)
                element_pk = int(os.path.splitext(filename)[0])

                # Load embedding
                embedding = np.load(file_path).tolist()
                self.embeddings[element_pk] = embedding
            except Exception as e:
                logger.error(f"Error loading embedding from {file_path}: {str(e)}")

    def _save_document(self, doc_id: str) -> None:
        """Save document to file."""
        if doc_id not in self.documents:
            return

        file_path = os.path.join(self.storage_path, 'documents', f"{doc_id}.json")

        try:
            with open(file_path, 'w') as f:
                json.dump(self.documents[doc_id], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving document to {file_path}: {str(e)}")

    def _save_element(self, element_id: str) -> None:
        """Save element to file."""
        if element_id not in self.elements:
            return

        file_path = os.path.join(self.storage_path, 'elements', f"{element_id}.json")

        try:
            with open(file_path, 'w') as f:
                json.dump(self.elements[element_id], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving element to {file_path}: {str(e)}")

    def _save_relationship(self, relationship_id: str) -> None:
        """Save relationship to file."""
        if relationship_id not in self.relationships:
            return

        file_path = os.path.join(self.storage_path, 'relationships', f"{relationship_id}.json")

        try:
            with open(file_path, 'w') as f:
                json.dump(self.relationships[relationship_id], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving relationship to {file_path}: {str(e)}")

    def _save_embedding(self, element_pk: int) -> None:
        """Save embedding to file."""
        if element_pk not in self.embeddings:
            return

        import numpy as np

        file_path = os.path.join(self.storage_path, 'embeddings', f"{element_pk}.npy")

        try:
            np.save(file_path, np.array(self.embeddings[element_pk], dtype=np.float32))
        except Exception as e:
            logger.error(f"Error saving embedding to {file_path}: {str(e)}")

    def _delete_document_file(self, doc_id: str) -> None:
        """Delete document file."""
        file_path = os.path.join(self.storage_path, 'documents', f"{doc_id}.json")

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting document file {file_path}: {str(e)}")

    def _delete_element_file(self, element_id: str) -> None:
        """Delete element file."""
        file_path = os.path.join(self.storage_path, 'elements', f"{element_id}.json")

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting element file {file_path}: {str(e)}")

    def _delete_relationship_file(self, relationship_id: str) -> None:
        """Delete relationship file."""
        file_path = os.path.join(self.storage_path, 'relationships', f"{relationship_id}.json")

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting relationship file {file_path}: {str(e)}")

    def _delete_embedding_file(self, element_pk: int) -> None:
        """Delete embedding file."""
        file_path = os.path.join(self.storage_path, 'embeddings', f"{element_pk}.npy")

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting embedding file {file_path}: {str(e)}")

    @staticmethod
    def _has_element_changed(new_element: Dict[str, Any],
                             old_element: Dict[str, Any]) -> bool:
        """Check if element has changed."""
        # Check content hash first
        if new_element.get("content_hash") != old_element.get("content_hash"):
            return True

        # Check other fields
        for field in ["element_type", "parent_id", "content_preview", "content_location"]:
            if new_element.get(field) != old_element.get(field):
                return True

        # Check metadata
        new_metadata = new_element.get("metadata", {})
        old_metadata = old_element.get("metadata", {})

        if set(new_metadata.keys()) != set(old_metadata.keys()):
            return True

        for key, value in new_metadata.items():
            if old_metadata.get(key) != value:
                return True

        return False

    @staticmethod
    def _has_relationship_changed(new_rel: Dict[str, Any],
                                  old_rel: Dict[str, Any]) -> bool:
        """Check if relationship has changed."""
        for field in ["source_id", "relationship_type", "target_reference"]:
            if new_rel.get(field) != old_rel.get(field):
                return True

        # Check metadata
        new_metadata = new_rel.get("metadata", {})
        old_metadata = old_rel.get("metadata", {})

        if set(new_metadata.keys()) != set(old_metadata.keys()):
            return True

        for key, value in new_metadata.items():
            if old_metadata.get(key) != value:
                return True

        return False

    def search_by_text(self, search_text: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Search elements by semantic similarity to the provided text.

        This method combines text-to-embedding conversion and embedding search
        into a single convenient operation.

        Args:
            search_text: Text to search for semantically
            limit: Maximum number of results

        Returns:
            List of (element_id, similarity_score) tuples
        """
        # Import necessary modules
        from doculyzer.embeddings import get_embedding_generator

        try:
            # Get config
            try:
                from doculyzer.config import Config
                config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
            except Exception as e:
                logger.warning(f"Error loading config: {str(e)}. Using default config.")
                config = Config()

            # Get the embedding generator
            embedding_generator = get_embedding_generator(config)

            # Generate embedding for the search text
            query_embedding = embedding_generator.generate(search_text)

            # Use the embedding to search
            return self.search_by_embedding(query_embedding, limit)

        except Exception as e:
            logger.error(f"Error in semantic search by text: {str(e)}")
            # Return empty list on error
            return []
