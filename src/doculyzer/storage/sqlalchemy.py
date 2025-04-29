"""
SQLAlchemy implementation of document database for the Doculyzer system.

This module provides a SQLAlchemy-based storage backend for the document pointer system,
allowing for flexible database support and ORM capabilities.
"""

import json
import logging
import os
import time
from typing import Optional, Dict, Any, List, Tuple, Union

import numpy as np
from sqlalchemy import (
    create_engine, Column, ForeignKey, String, Integer, Float, Text, LargeBinary, func, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session

from .base import DocumentDatabase
from ..config import Config

logger = logging.getLogger(__name__)

# Create declarative base
Base = declarative_base()

config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))


# Define ORM models
class Document(Base):
    """Document model for SQLAlchemy ORM."""
    __tablename__ = 'documents'

    doc_id = Column(String(255), primary_key=True)
    doc_type = Column(String(50))
    source = Column(String(1024))
    content_hash = Column(String(255))
    metadata = Column(Text)
    created_at = Column(Float)
    updated_at = Column(Float)

    # Relationships
    elements = relationship("Element", back_populates="document", cascade="all, delete-orphan")


class Element(Base):
    """Element model for SQLAlchemy ORM."""
    __tablename__ = 'elements'

    element_pk = Column(Integer, primary_key=True, autoincrement=True)
    element_id = Column(String(255), unique=True, nullable=False)
    doc_id = Column(String(255), ForeignKey('documents.doc_id', ondelete='CASCADE'))
    element_type = Column(String(50))
    parent_id = Column(String(255), ForeignKey('elements.element_id'))
    content_preview = Column(Text)
    content_location = Column(Text)
    content_hash = Column(String(255))
    metadata = Column(Text)

    # Relationships
    document = relationship("Document", back_populates="elements")
    embedding = relationship("Embedding", uselist=False, back_populates="element", cascade="all, delete-orphan")
    relationships_as_source = relationship("Relationship", foreign_keys="Relationship.source_id",
                                           cascade="all, delete-orphan")
    children = relationship("Element",
                            backref="parent",
                            remote_side=[element_id])


class Relationship(Base):
    """Relationship model for SQLAlchemy ORM."""
    __tablename__ = 'relationships'

    relationship_id = Column(String(255), primary_key=True)
    source_id = Column(String(255), ForeignKey('elements.element_id', ondelete='CASCADE'))
    relationship_type = Column(String(50))
    target_reference = Column(String(255))
    metadata = Column(Text)


class Embedding(Base):
    """Embedding model for SQLAlchemy ORM."""
    __tablename__ = 'embeddings'

    element_pk = Column(Integer, ForeignKey('elements.element_pk', ondelete='CASCADE'), primary_key=True)
    embedding = Column(LargeBinary)
    dimensions = Column(Integer)
    created_at = Column(Float)

    # Relationships
    element = relationship("Element", back_populates="embedding")


class ProcessingHistory(Base):
    """Processing history model for SQLAlchemy ORM."""
    __tablename__ = 'processing_history'

    source_id = Column(String(1024), primary_key=True)
    content_hash = Column(String(255))
    last_modified = Column(Float)
    processing_count = Column(Integer, default=1)


class SQLAlchemyDocumentDatabase(DocumentDatabase):
    """SQLAlchemy implementation of document database."""

    def __init__(self, db_uri: str, echo: bool = False):
        """
        Initialize SQLAlchemy document database.

        Args:
            db_uri: Database URI (e.g. 'sqlite:///path/to/database.db',
                                 'postgresql://user:pass@localhost/dbname')
            echo: Whether to echo SQL statements
        """
        self.config = None
        self.db_uri = db_uri
        self.echo = echo
        self.engine = None
        self.Session = None
        self.session = None
        self._vector_extension = None
        self._vector_dimension = config.config.get('embedding', {}).get('dimensions', 384)

    def initialize(self) -> None:
        """Initialize the database by creating tables if they don't exist."""
        # Create directory if it's a sqlite file
        if self.db_uri.startswith('sqlite:///'):
            db_path = self.db_uri.replace('sqlite:///', '')
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        # Create engine
        self.engine = create_engine(self.db_uri, echo=self.echo)

        # Create session factory
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        self.session = self.Session()

        # Create tables
        Base.metadata.create_all(self.engine)

        # Check for vector extension support
        self._check_vector_extension()

        logger.info(f"Initialized SQLAlchemy database with URI: {self.db_uri}")

    def _check_vector_extension(self) -> None:
        """Check for vector extension support in the database."""
        if self.db_uri.startswith('postgresql'):
            try:
                # Check for pgvector
                result = self.session.execute(
                    text("SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'vector')"))
                pgvector_available = result.scalar()

                if pgvector_available:
                    # Check if installed
                    result = self.session.execute(
                        text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"))
                    pgvector_installed = result.scalar()

                    if not pgvector_installed:
                        try:
                            # Try to install
                            self.session.execute(text("CREATE EXTENSION vector"))
                            self.session.commit()
                            self._vector_extension = "pgvector"
                            logger.info("Installed pgvector extension")
                        except Exception as e:
                            logger.warning(f"Failed to install pgvector extension: {str(e)}")
                    else:
                        self._vector_extension = "pgvector"
                        logger.info("Using pgvector extension")
            except Exception as e:
                logger.warning(f"Error checking for vector extension: {str(e)}")
        elif self.db_uri.startswith('sqlite'):
            # Check for sqlite vector extensions
            # This is a simplified check - in a real implementation you'd want to
            # actually test if the extension works as expected
            try:
                import sqlite_vec
                self._vector_extension = "vec0"
                logger.info("Using sqlite-vec extension")
                return
            except ImportError:
                pass

            try:
                import sqlite_vss
                self._vector_extension = "vss0"
                logger.info("Using sqlite-vss extension")
                return
            except ImportError:
                pass

            logger.info("No vector extensions found, using native implementation")

    def close(self) -> None:
        """Close the database connection."""
        if self.session:
            self.session.close()
            self.session = None

        if self.engine:
            self.engine.dispose()
            self.engine = None

    def get_last_processed_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about when a document was last processed."""
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            history = self.session.query(ProcessingHistory).filter_by(source_id=source_id).first()

            if not history:
                return None

            return {
                "source_id": history.source_id,
                "content_hash": history.content_hash,
                "last_modified": history.last_modified,
                "processing_count": history.processing_count
            }
        except Exception as e:
            logger.error(f"Error getting processing history for {source_id}: {str(e)}")
            return None

    def update_processing_history(self, source_id: str, content_hash: str) -> None:
        """Update the processing history for a document."""
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            # Check if record exists
            history = self.session.query(ProcessingHistory).filter_by(source_id=source_id).first()

            if history:
                # Update existing record
                history.content_hash = content_hash
                history.last_modified = time.time()
                history.processing_count += 1
            else:
                # Create new record
                history = ProcessingHistory(
                    source_id=source_id,
                    content_hash=content_hash,
                    last_modified=time.time(),
                    processing_count=1
                )
                self.session.add(history)

            self.session.commit()
            logger.debug(f"Updated processing history for {source_id}")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating processing history for {source_id}: {str(e)}")

    def store_document(self, document: Dict[str, Any], elements: List[Dict[str, Any]],
                       relationships: List[Dict[str, Any]]) -> None:
        """
        Store a document with its elements and relationships.
        If a document with the same source already exists, update it instead.

        Args:
            document: Document metadata
            elements: Document elements
            relationships: Element relationships
        """
        if not self.session:
            raise ValueError("Database not initialized")

        source = document.get("source", "")
        content_hash = document.get("content_hash", "")

        # Check if document already exists with this source
        existing_doc = None
        if source:
            existing_doc = self.session.query(Document).filter_by(source=source).first()

        if existing_doc:
            # Document exists with same source, update it
            doc_id = existing_doc.doc_id
            document["doc_id"] = doc_id  # Use existing doc_id

            # Update all elements to use the existing doc_id
            for element in elements:
                element["doc_id"] = doc_id

            self.update_document(doc_id, document, elements, relationships)
            return

        try:
            # Start a transaction
            self.session.begin()

            # Create document record
            doc_id = document["doc_id"]
            doc_record = Document(
                doc_id=doc_id,
                doc_type=document.get("doc_type", ""),
                source=source,
                content_hash=content_hash,
                metadata=json.dumps(document.get("metadata", {})),
                created_at=document.get("created_at", time.time()),
                updated_at=document.get("updated_at", time.time())
            )
            self.session.add(doc_record)
            self.session.flush()  # Flush to get doc_id if it's generated

            # Store elements
            element_records = {}
            for element in elements:
                element_id = element["element_id"]
                element_record = Element(
                    element_id=element_id,
                    doc_id=element.get("doc_id", doc_id),
                    element_type=element.get("element_type", ""),
                    parent_id=element.get("parent_id"),
                    content_preview=element.get("content_preview", ""),
                    content_location=element.get("content_location", ""),
                    content_hash=element.get("content_hash", ""),
                    metadata=json.dumps(element.get("metadata", {}))
                )
                self.session.add(element_record)
                element_records[element_id] = element_record

            # Flush to get element PKs
            self.session.flush()

            # Update the original elements with their PKs
            for element in elements:
                element_id = element["element_id"]
                if element_id in element_records:
                    element["element_pk"] = element_records[element_id].element_pk

            # Store relationships
            for relationship in relationships:
                relationship_id = relationship["relationship_id"]
                relationship_record = Relationship(
                    relationship_id=relationship_id,
                    source_id=relationship.get("source_id", ""),
                    relationship_type=relationship.get("relationship_type", ""),
                    target_reference=relationship.get("target_reference", ""),
                    metadata=json.dumps(relationship.get("metadata", {}))
                )
                self.session.add(relationship_record)

            # Commit the transaction
            self.session.commit()

            # Update processing history
            if source:
                self.update_processing_history(source, content_hash)

            logger.info(
                f"Stored document {doc_id} with {len(elements)} elements and {len(relationships)} relationships")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error storing document {document.get('doc_id')}: {str(e)}")
            raise

    def update_document(self, doc_id: str, document: Dict[str, Any],
                        elements: List[Dict[str, Any]],
                        relationships: List[Dict[str, Any]]) -> None:
        """
        Update an existing document.

        Args:
            doc_id: Document ID
            document: Document metadata
            elements: Document elements
            relationships: Element relationships
        """
        if not self.session:
            raise ValueError("Database not initialized")

        # Check if document exists
        existing_doc = self.session.query(Document).filter_by(doc_id=doc_id).first()
        if not existing_doc:
            raise ValueError(f"Document not found: {doc_id}")

        source = document.get("source", "")
        content_hash = document.get("content_hash", "")

        try:
            # Start a transaction
            self.session.begin()

            # Delete existing relationships
            self.session.query(Relationship).filter(
                Relationship.source_id.in_(
                    self.session.query(Element.element_id).filter_by(doc_id=doc_id)
                )
            ).delete(synchronize_session=False)

            # Delete existing embeddings
            self.session.query(Embedding).filter(
                Embedding.element_pk.in_(
                    self.session.query(Element.element_pk).filter_by(doc_id=doc_id)
                )
            ).delete(synchronize_session=False)

            # Delete existing elements
            self.session.query(Element).filter_by(doc_id=doc_id).delete(synchronize_session=False)

            # Update document record
            existing_doc.doc_type = document.get("doc_type", existing_doc.doc_type)
            existing_doc.source = source
            existing_doc.content_hash = content_hash
            existing_doc.metadata = json.dumps(document.get("metadata", {}))
            existing_doc.updated_at = time.time()

            # Store elements
            element_records = {}
            for element in elements:
                element_id = element["element_id"]
                element_record = Element(
                    element_id=element_id,
                    doc_id=doc_id,
                    element_type=element.get("element_type", ""),
                    parent_id=element.get("parent_id"),
                    content_preview=element.get("content_preview", ""),
                    content_location=element.get("content_location", ""),
                    content_hash=element.get("content_hash", ""),
                    metadata=json.dumps(element.get("metadata", {}))
                )
                self.session.add(element_record)
                element_records[element_id] = element_record

            # Flush to get element PKs
            self.session.flush()

            # Update the original elements with their PKs
            for element in elements:
                element_id = element["element_id"]
                if element_id in element_records:
                    element["element_pk"] = element_records[element_id].element_pk

            # Store relationships
            for relationship in relationships:
                relationship_id = relationship["relationship_id"]
                relationship_record = Relationship(
                    relationship_id=relationship_id,
                    source_id=relationship.get("source_id", ""),
                    relationship_type=relationship.get("relationship_type", ""),
                    target_reference=relationship.get("target_reference", ""),
                    metadata=json.dumps(relationship.get("metadata", {}))
                )
                self.session.add(relationship_record)

            # Commit the transaction
            self.session.commit()

            # Update processing history
            if source:
                self.update_processing_history(source, content_hash)

            logger.info(
                f"Updated document {doc_id} with {len(elements)} elements and {len(relationships)} relationships")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating document {doc_id}: {str(e)}")
            raise

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        if not self.session:
            raise ValueError("Database not initialized")

        document = self.session.query(Document).filter_by(doc_id=doc_id).first()
        if not document:
            return None

        # Convert to dictionary
        result = {
            "doc_id": document.doc_id,
            "doc_type": document.doc_type,
            "source": document.source,
            "content_hash": document.content_hash,
            "created_at": document.created_at,
            "updated_at": document.updated_at
        }

        # Parse metadata JSON
        try:
            result["metadata"] = json.loads(document.metadata)
        except (json.JSONDecodeError, TypeError):
            result["metadata"] = {}

        return result

    def get_document_elements(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get elements for a document."""
        if not self.session:
            raise ValueError("Database not initialized")

        elements = self.session.query(Element).filter_by(doc_id=doc_id).order_by(Element.element_id).all()

        result = []
        for element in elements:
            # Convert to dictionary
            element_dict = {
                "element_id": element.element_id,
                "element_pk": element.element_pk,
                "doc_id": element.doc_id,
                "element_type": element.element_type,
                "parent_id": element.parent_id,
                "content_preview": element.content_preview,
                "content_location": element.content_location,
                "content_hash": element.content_hash
            }

            # Parse metadata JSON
            try:
                element_dict["metadata"] = json.loads(element.metadata)
            except (json.JSONDecodeError, TypeError):
                element_dict["metadata"] = {}

            result.append(element_dict)

        return result

    def get_document_relationships(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get relationships for a document."""
        if not self.session:
            raise ValueError("Database not initialized")

        # Get all element IDs for this document
        element_ids = [row[0] for row in
                       self.session.query(Element.element_id).filter_by(doc_id=doc_id).all()]

        if not element_ids:
            return []

        # Get relationships involving these elements
        relationships = self.session.query(Relationship).filter(
            Relationship.source_id.in_(element_ids)
        ).all()

        result = []
        for relationship in relationships:
            # Convert to dictionary
            rel_dict = {
                "relationship_id": relationship.relationship_id,
                "source_id": relationship.source_id,
                "relationship_type": relationship.relationship_type,
                "target_reference": relationship.target_reference
            }

            # Parse metadata JSON
            try:
                rel_dict["metadata"] = json.loads(relationship.metadata)
            except (json.JSONDecodeError, TypeError):
                rel_dict["metadata"] = {}

            result.append(rel_dict)

        return result

    def get_element(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Get element by ID."""
        if not self.session:
            raise ValueError("Database not initialized")

        element = self.session.query(Element).filter_by(element_id=element_id).first()
        if not element:
            return None

        # Convert to dictionary
        result = {
            "element_id": element.element_id,
            "element_pk": element.element_pk,
            "doc_id": element.doc_id,
            "element_type": element.element_type,
            "parent_id": element.parent_id,
            "content_preview": element.content_preview,
            "content_location": element.content_location,
            "content_hash": element.content_hash
        }

        # Parse metadata JSON
        try:
            result["metadata"] = json.loads(element.metadata)
        except (json.JSONDecodeError, TypeError):
            result["metadata"] = {}

        return result

    def find_documents(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find documents matching query."""
        if not self.session:
            raise ValueError("Database not initialized")

        if query is None:
            query = {}

        # Build query
        db_query = self.session.query(Document)

        # Apply filters
        for key, value in query.items():
            if key == "metadata":
                # Skip metadata for now, will handle below
                continue
            elif hasattr(Document, key):
                db_query = db_query.filter(getattr(Document, key) == value)

        # Apply metadata filters if any
        if "metadata" in query:
            for meta_key, meta_value in query["metadata"].items():
                # This is a simplification - proper JSON field querying depends on the database
                json_filter = func.json_extract(Document.metadata, f'$.{meta_key}') == json.dumps(meta_value)
                db_query = db_query.filter(json_filter)

        # Apply limit
        db_query = db_query.limit(limit)

        # Execute query
        documents = db_query.all()

        # Convert to dictionaries
        result = []
        for document in documents:
            doc_dict = {
                "doc_id": document.doc_id,
                "doc_type": document.doc_type,
                "source": document.source,
                "content_hash": document.content_hash,
                "created_at": document.created_at,
                "updated_at": document.updated_at
            }

            # Parse metadata JSON
            try:
                doc_dict["metadata"] = json.loads(document.metadata)
            except (json.JSONDecodeError, TypeError):
                doc_dict["metadata"] = {}

            result.append(doc_dict)

        return result

    def find_elements(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find elements matching query."""
        if not self.session:
            raise ValueError("Database not initialized")

        if query is None:
            query = {}

        # Build query
        db_query = self.session.query(Element)

        # Apply filters
        for key, value in query.items():
            if key == "metadata":
                # Skip metadata for now, will handle below
                continue
            elif hasattr(Element, key):
                db_query = db_query.filter(getattr(Element, key) == value)

        # Apply metadata filters if any
        if "metadata" in query:
            for meta_key, meta_value in query["metadata"].items():
                # This is a simplification - proper JSON field querying depends on the database
                json_filter = func.json_extract(Element.metadata, f'$.{meta_key}') == json.dumps(meta_value)
                db_query = db_query.filter(json_filter)

        # Apply limit
        db_query = db_query.limit(limit)

        # Execute query
        elements = db_query.all()

        # Convert to dictionaries
        result = []
        for element in elements:
            element_dict = {
                "element_id": element.element_id,
                "element_pk": element.element_pk,
                "doc_id": element.doc_id,
                "element_type": element.element_type,
                "parent_id": element.parent_id,
                "content_preview": element.content_preview,
                "content_location": element.content_location,
                "content_hash": element.content_hash
            }

            # Parse metadata JSON
            try:
                element_dict["metadata"] = json.loads(element.metadata)
            except (json.JSONDecodeError, TypeError):
                element_dict["metadata"] = {}

            result.append(element_dict)

        return result

    def search_elements_by_content(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search elements by content preview."""
        if not self.session:
            raise ValueError("Database not initialized")

        # Use LIKE operator for text search
        elements = self.session.query(Element).filter(
            Element.content_preview.like(f"%{search_text}%")
        ).limit(limit).all()

        # Convert to dictionaries
        result = []
        for element in elements:
            element_dict = {
                "element_id": element.element_id,
                "element_pk": element.element_pk,
                "doc_id": element.doc_id,
                "element_type": element.element_type,
                "parent_id": element.parent_id,
                "content_preview": element.content_preview,
                "content_location": element.content_location,
                "content_hash": element.content_hash
            }

            # Parse metadata JSON
            try:
                element_dict["metadata"] = json.loads(element.metadata)
            except (json.JSONDecodeError, TypeError):
                element_dict["metadata"] = {}

            result.append(element_dict)

        return result

    def store_embedding(self, element_pk: Union[int, str], embedding: List[float]) -> None:
        """Store embedding for an element."""
        if not self.session:
            raise ValueError("Database not initialized")

        # Verify element exists
        element = self.session.query(Element).filter_by(element_pk=element_pk).first()
        if not element:
            raise ValueError(f"Element not found: {element_pk}")

        # Update vector dimension
        self._vector_dimension = max(self._vector_dimension, len(embedding))

        try:
            # Encode embedding as binary
            embedding_blob = self._encode_embedding(embedding)

            # Check if embedding already exists
            existing = self.session.query(Embedding).filter_by(element_pk=element_pk).first()

            if existing:
                # Update existing embedding
                existing.embedding = embedding_blob
                existing.dimensions = len(embedding)
                existing.created_at = time.time()
            else:
                # Create new embedding
                new_embedding = Embedding(
                    element_pk=element_pk,
                    embedding=embedding_blob,
                    dimensions=len(embedding),
                    created_at=time.time()
                )
                self.session.add(new_embedding)

            # Commit changes
            self.session.commit()

            # Handle vector extension specific storage
            if self._vector_extension == "pgvector" and self.db_uri.startswith('postgresql'):
                self._store_pgvector_embedding(element_pk, embedding)

            logger.debug(f"Stored embedding for element {element_pk} with {len(embedding)} dimensions")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error storing embedding for {element_pk}: {str(e)}")
            raise

    def _store_pgvector_embedding(self, element_pk: Union[int, str], embedding: List[float]) -> None:
        """Store embedding using pgvector extension."""
        try:
            # Convert embedding to string for pgvector
            embedding_str = json.dumps(embedding)

            # Execute raw SQL to update the vector column
            self.session.execute(text(
                f"UPDATE embeddings SET vector_embedding = :embedding::vector WHERE element_pk = :pk"
            ), {"embedding": embedding_str, "pk": element_pk})

            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error storing pgvector embedding: {str(e)}")

    def get_embedding(self, element_pk: Union[int, str]) -> Optional[List[float]]:
        """Get embedding for an element."""
        if not self.session:
            raise ValueError("Database not initialized")

        embedding_record = self.session.query(Embedding).filter_by(element_pk=element_pk).first()
        if not embedding_record:
            return None

        # Decode binary embedding
        return self._decode_embedding(embedding_record.embedding)

    def search_by_embedding(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[str, float]]:
        """Search elements by embedding similarity."""
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            if self._vector_extension == "pgvector" and self.db_uri.startswith('postgresql'):
                return self._search_by_pgvector(query_embedding, limit)
            else:
                # Use native implementation
                return self._search_by_embedding_native(query_embedding, limit)
        except Exception as e:
            logger.error(f"Error searching by embedding: {str(e)}")
            # Fall back to native implementation
            try:
                return self._search_by_embedding_native(query_embedding, limit)
            except Exception as e2:
                logger.error(f"Error in fallback search: {str(e2)}")
                return []

    def _search_by_pgvector(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[str, float]]:
        """Use pgvector for similarity search."""
        # Convert embedding to JSON string for pgvector
        embedding_json = json.dumps(query_embedding)

        try:
            # Use the <=> operator (cosine distance)
            result = self.session.execute(text(
                """
                SELECT e.element_id, 1 - (em.vector_embedding <=> :query::vector) as similarity
                FROM embeddings em
                JOIN elements e ON e.element_pk = em.element_pk
                ORDER BY em.vector_embedding <=> :query::vector
                LIMIT :limit
                """
            ), {"query": embedding_json, "limit": limit})

            return [(row[0], row[1]) for row in result]
        except Exception as e:
            logger.error(f"Error using pgvector search: {str(e)}")
            raise

    def _search_by_embedding_native(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[str, float]]:
        """Use native Python implementation for similarity search."""
        # Get all embeddings
        embedding_records = self.session.query(Embedding, Element.element_id).join(
            Element, Embedding.element_pk == Element.element_pk
        ).all()

        # Calculate similarities
        similarities = []
        query_np = np.array(query_embedding)

        for record, element_id in embedding_records:
            embedding = self._decode_embedding(record.embedding)
            if len(embedding) != len(query_embedding):
                # Skip if dimensions don't match
                continue

            embedding_np = np.array(embedding)
            similarity = self._cosine_similarity(query_np, embedding_np)
            similarities.append((element_id, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        return similarities[:limit]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all associated elements and relationships."""
        if not self.session:
            raise ValueError("Database not initialized")

        # Check if document exists
        document = self.session.query(Document).filter_by(doc_id=doc_id).first()
        if not document:
            return False

        try:
            # Start transaction
            self.session.begin()

            # Delete the document (cascading delete will handle elements,
            # relationships, and embeddings due to our relationship configurations)
            self.session.delete(document)

            # Commit changes
            self.session.commit()

            logger.info(f"Deleted document {doc_id}")
            return True

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False

    @staticmethod
    def _encode_embedding(embedding: List[float]) -> bytes:
        """Encode embedding as binary blob."""
        # Convert to numpy array and then to bytes
        return np.array(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def _decode_embedding(blob: bytes) -> List[float]:
        """Decode embedding from binary blob."""
        # Convert from bytes to numpy array and then to list
        return np.frombuffer(blob, dtype=np.float32).tolist()

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        # Calculate dot product
        dot_product = np.dot(vec1, vec2)

        # Calculate magnitudes
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        return float(dot_product / (norm1 * norm2))

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
        if not self.session:
            raise ValueError("Database not initialized")

        try:
            # Import necessary modules
            from doculyzer.embeddings import get_embedding_generator

            # Get config from the connection parameters or load from path
            config = self.config
            if not config:
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
