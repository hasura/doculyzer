import datetime
import json
import logging
import os
from typing import Optional, Dict, Any, List, Tuple

import time

from .base import DocumentDatabase
from .element_relationship import ElementRelationship

# Try to import neo4j
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
except ImportError:
    raise ImportError("Neo4j driver not installed. Please install with: pip install neo4j")

# Setup logger
logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()  # Convert date/datetime to ISO 8601 string
        return super().default(obj)


class Neo4jDocumentDatabase(DocumentDatabase):
    """Neo4j implementation of document database."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j document database.

        Args:
            uri: Neo4j connection URI (e.g., 'bolt://localhost:7687')
            user: Neo4j username
            password: Neo4j password
            database: Neo4j database name (default is 'neo4j')
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.embedding_generator = None

    def initialize(self) -> None:
        """Initialize the database by creating constraints and indexes."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )

            # Test the connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")

            # Create constraints and indexes
            self._create_constraints_and_indexes()
            logger.info(f"Successfully connected to Neo4j at {self.uri}")

        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            self.driver = None

    def _create_constraints_and_indexes(self) -> None:
        """Create necessary constraints and indexes for optimal performance."""
        with self.driver.session(database=self.database) as session:
            # Create constraints for unique IDs
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS ON (d:Document) ASSERT d.doc_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS ON (e:Element) ASSERT e.element_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS ON (d:Document) ASSERT d.source IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS ON (h:ProcessingHistory) ASSERT h.source_id IS UNIQUE"
            ]

            # Create indexes for faster lookups
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (e:Element) ON (e.doc_id)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Element) ON (e.element_type)",
                "CREATE INDEX IF NOT EXISTS FOR (r:RELATIONSHIP) ON (r.relationship_type)"
            ]

            # Execute all constraints and indexes
            for query in constraints + indexes:
                try:
                    session.run(query)
                except Exception as e:
                    logger.warning(f"Error creating constraint or index: {str(e)}")

    def get_last_processed_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about when a document was last processed."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (h:ProcessingHistory {source_id: $source_id})
                RETURN h.source_id AS source_id, 
                       h.content_hash AS content_hash,
                       h.last_modified AS last_modified,
                       h.processing_count AS processing_count
                """,
                source_id=source_id
            )

            record = result.single()
            if not record:
                return None

            return {
                "source_id": record["source_id"],
                "content_hash": record["content_hash"],
                "last_modified": record["last_modified"],
                "processing_count": record["processing_count"]
            }

    def update_processing_history(self, source_id: str, content_hash: str) -> None:
        """Update the processing history for a document."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # Check if record exists and get processing count
            result = session.run(
                """
                MATCH (h:ProcessingHistory {source_id: $source_id})
                RETURN h.processing_count AS processing_count
                """,
                source_id=source_id
            )

            record = result.single()
            processing_count = 1  # Default for new records

            if record:
                processing_count = record["processing_count"] + 1

                # Update existing record
                session.run(
                    """
                    MATCH (h:ProcessingHistory {source_id: $source_id})
                    SET h.content_hash = $content_hash,
                        h.last_modified = $timestamp,
                        h.processing_count = $processing_count
                    """,
                    source_id=source_id,
                    content_hash=content_hash,
                    timestamp=time.time(),
                    processing_count=processing_count
                )
            else:
                # Create new record
                session.run(
                    """
                    CREATE (h:ProcessingHistory {
                        source_id: $source_id,
                        content_hash: $content_hash,
                        last_modified: $timestamp,
                        processing_count: $processing_count
                    })
                    """,
                    source_id=source_id,
                    content_hash=content_hash,
                    timestamp=time.time(),
                    processing_count=processing_count
                )

            logger.debug(f"Updated processing history for {source_id}")

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
        if not self.driver:
            raise ValueError("Database not initialized")

        source = document.get("source", "")
        content_hash = document.get("content_hash", "")

        with self.driver.session(database=self.database) as session:
            # Check if document already exists
            result = session.run(
                """
                MATCH (d:Document {source: $source})
                RETURN d.doc_id AS doc_id
                """,
                source=source
            )

            record = result.single()
            if record:
                # Document exists, update it
                doc_id = record["doc_id"]
                document["doc_id"] = doc_id  # Use existing doc_id

                # Update all elements to use the existing doc_id
                for element in elements:
                    element["doc_id"] = doc_id

                self.update_document(doc_id, document, elements, relationships)
                return

            # New document, proceed with creation
            doc_id = document["doc_id"]

            # Store document
            metadata_json = json.dumps(document.get("metadata", {}), cls=DateTimeEncoder)

            session.run(
                """
                CREATE (d:Document {
                    doc_id: $doc_id,
                    doc_type: $doc_type,
                    source: $source,
                    content_hash: $content_hash,
                    metadata: $metadata,
                    created_at: $created_at,
                    updated_at: $updated_at
                })
                """,
                doc_id=doc_id,
                doc_type=document.get("doc_type", ""),
                source=source,
                content_hash=content_hash,
                metadata=metadata_json,
                created_at=document.get("created_at", time.time()),
                updated_at=document.get("updated_at", time.time())
            )

            # Store elements and create relationships to document
            element_pk_map = {}  # Maps element_id to Neo4j node id

            for element in elements:
                element_id = element["element_id"]
                metadata_json = json.dumps(element.get("metadata", {}))
                content_preview = element.get("content_preview", "")

                if len(content_preview) > 100:
                    content_preview = content_preview[:100] + "..."

                # Create the element node and link to document
                result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    CREATE (e:Element {
                        element_id: $element_id,
                        doc_id: $doc_id,
                        element_type: $element_type,
                        parent_id: $parent_id,
                        content_preview: $content_preview,
                        content_location: $content_location,
                        content_hash: $content_hash,
                        metadata: $metadata
                    })
                    CREATE (e)-[:BELONGS_TO]->(d)
                    RETURN id(e) AS node_id
                    """,
                    doc_id=element.get("doc_id", ""),
                    element_id=element_id,
                    element_type=element.get("element_type", ""),
                    parent_id=element.get("parent_id", ""),
                    content_preview=content_preview,
                    content_location=element.get("content_location", ""),
                    content_hash=element.get("content_hash", ""),
                    metadata=metadata_json
                )

                record = result.single()
                if record:
                    element_pk_map[element_id] = record["node_id"]

            # Create parent-child relationships between elements
            for element in elements:
                if element.get("parent_id"):
                    session.run(
                        """
                        MATCH (child:Element {element_id: $element_id})
                        MATCH (parent:Element {element_id: $parent_id})
                        CREATE (child)-[:CHILD_OF]->(parent)
                        """,
                        element_id=element["element_id"],
                        parent_id=element["parent_id"]
                    )

            # Store custom relationships
            for relationship in relationships:
                relationship_id = relationship["relationship_id"]
                metadata_json = json.dumps(relationship.get("metadata", {}))
                source_id = relationship.get("source_id", "")
                target_reference = relationship.get("target_reference", "")
                relationship_type = relationship.get("relationship_type", "")

                # Create the relationship between elements
                session.run(
                    """
                    MATCH (source:Element {element_id: $source_id})
                    MATCH (target:Element {element_id: $target_reference})
                    CREATE (source)-[r:RELATES_TO {
                        relationship_id: $relationship_id,
                        relationship_type: $relationship_type,
                        metadata: $metadata
                    }]->(target)
                    """,
                    source_id=source_id,
                    target_reference=target_reference,
                    relationship_id=relationship_id,
                    relationship_type=relationship_type,
                    metadata=metadata_json
                )

            # Update processing history
            if source:
                self.update_processing_history(source, content_hash)

    def update_document(self, doc_id: str, document: Dict[str, Any],
                        elements: List[Dict[str, Any]],
                        relationships: List[Dict[str, Any]]) -> None:
        """
        Update an existing document by removing it and then reinserting.
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # Check if document exists
            result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                RETURN d.doc_id AS doc_id
                """,
                doc_id=doc_id
            )

            if not result.single():
                raise ValueError(f"Document not found: {doc_id}")

            # Delete the document, which will cascade to elements and relationships
            self.delete_document(doc_id)

            # Use store_document to insert everything
            self.store_document(document, elements, relationships)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (d:Document)
                WHERE d.doc_id = $doc_id OR d.source = $doc_id
                RETURN d
                """,
                doc_id=doc_id
            )

            record = result.single()
            if not record:
                return None

            document = dict(record["d"])

            # Convert metadata from JSON
            try:
                document["metadata"] = json.loads(document["metadata"])
            except (json.JSONDecodeError, TypeError):
                document["metadata"] = {}

            return document

    def get_document_elements(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get elements for a document."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (d:Document)
                WHERE d.doc_id = $doc_id OR d.source = $doc_id
                MATCH (e:Element)-[:BELONGS_TO]->(d)
                RETURN e, id(e) AS element_pk
                """,
                doc_id=doc_id
            )

            elements = []
            for record in result:
                element = dict(record["e"])
                element["element_pk"] = record["element_pk"]

                # Convert metadata from JSON
                try:
                    element["metadata"] = json.loads(element["metadata"])
                except (json.JSONDecodeError, TypeError):
                    element["metadata"] = {}

                elements.append(element)

            return elements

    def get_document_relationships(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get relationships for a document."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                MATCH (e:Element)-[:BELONGS_TO]->(d)
                MATCH (e)-[r:RELATES_TO]->(target:Element)
                RETURN r.relationship_id AS relationship_id,
                       e.element_id AS source_id,
                       r.relationship_type AS relationship_type,
                       target.element_id AS target_reference,
                       r.metadata AS metadata
                """,
                doc_id=doc_id
            )

            relationships = []
            for record in result:
                relationship = {
                    "relationship_id": record["relationship_id"],
                    "source_id": record["source_id"],
                    "relationship_type": record["relationship_type"],
                    "target_reference": record["target_reference"],
                }

                # Convert metadata from JSON
                try:
                    relationship["metadata"] = json.loads(record["metadata"])
                except (json.JSONDecodeError, TypeError):
                    relationship["metadata"] = {}

                relationships.append(relationship)

            return relationships

    def get_element(self, element_pk: int | str) -> Optional[Dict[str, Any]]:
        """Get element by ID."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # If element_pk is numeric, treat as a Neo4j node ID
            if str(element_pk).isdigit():
                result = session.run(
                    """
                    MATCH (e:Element)
                    WHERE id(e) = $element_pk
                    RETURN e, id(e) AS element_pk
                    """,
                    element_pk=int(element_pk)
                )
            else:
                # Treat as element_id string
                result = session.run(
                    """
                    MATCH (e:Element {element_id: $element_id})
                    RETURN e, id(e) AS element_pk
                    """,
                    element_id=str(element_pk)
                )

            record = result.single()
            if not record:
                return None

            element = dict(record["e"])
            element["element_pk"] = record["element_pk"]

            # Convert metadata from JSON
            try:
                element["metadata"] = json.loads(element["metadata"])
            except (json.JSONDecodeError, TypeError):
                element["metadata"] = {}

            return element

    def get_outgoing_relationships(self, element_pk: int) -> List[ElementRelationship]:
        """
        Find all relationships where the specified element_pk is the source.

        Args:
            element_pk: The primary key of the element

        Returns:
            List of ElementRelationship objects where the specified element is the source
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        relationships = []

        with self.driver.session(database=self.database) as session:
            # Get the element to find its element_id and type
            if str(element_pk).isdigit():
                element_result = session.run(
                    """
                    MATCH (e:Element)
                    WHERE id(e) = $element_pk
                    RETURN e.element_id AS element_id, e.element_type AS element_type
                    """,
                    element_pk=int(element_pk)
                )
            else:
                element_result = session.run(
                    """
                    MATCH (e:Element {element_id: $element_id})
                    RETURN e.element_id AS element_id, e.element_type AS element_type, id(e) AS element_pk
                    """,
                    element_id=str(element_pk)
                )

            element_record = element_result.single()
            if not element_record:
                logger.warning(f"Element with PK {element_pk} not found")
                return []

            element_id = element_record["element_id"]
            element_type = element_record["element_type"]
            element_pk = element_record.get("element_pk", element_pk)

            # Find relationships with target element information
            result = session.run(
                """
                MATCH (source:Element {element_id: $element_id})-[r:RELATES_TO]->(target:Element)
                RETURN r.relationship_id AS relationship_id,
                       r.relationship_type AS relationship_type,
                       r.metadata AS metadata,
                       source.element_id AS source_id,
                       target.element_id AS target_reference,
                       target.element_type AS target_element_type,
                       target.content_preview AS target_content_preview,
                       id(target) AS target_element_pk,
                       source.doc_id AS doc_id
                """,
                element_id=element_id
            )

            for record in result:
                # Convert metadata from JSON
                try:
                    metadata = json.loads(record["metadata"]) if record["metadata"] else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                # Create relationship object
                relationship = ElementRelationship(
                    relationship_id=record["relationship_id"],
                    source_id=element_id,
                    source_element_pk=element_pk,
                    source_element_type=element_type,
                    relationship_type=record["relationship_type"],
                    target_reference=record["target_reference"],
                    target_element_pk=record["target_element_pk"],
                    target_element_type=record["target_element_type"],
                    target_content_preview=record["target_content_preview"],
                    doc_id=record["doc_id"],
                    metadata=metadata,
                    is_source=True
                )

                relationships.append(relationship)

        return relationships

    def find_documents(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find documents matching query."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # Start with base query
            cypher_query = "MATCH (d:Document)"
            params = {}

            # Apply filters if provided
            if query:
                conditions = []

                for key, value in query.items():
                    if key == "metadata":
                        # Metadata filters require special handling
                        for meta_key, meta_value in value.items():
                            # For simplicity, we'll check if the JSON contains this key/value
                            # Note: This is a simplistic approach and may need refinement for production
                            conditions.append(f"d.metadata CONTAINS $meta_{meta_key}")
                            params[f"meta_{meta_key}"] = f'"{meta_key}":"{meta_value}"'
                    else:
                        conditions.append(f"d.{key} = ${key}")
                        params[key] = value

                if conditions:
                    cypher_query += " WHERE " + " AND ".join(conditions)

            # Add return and limit
            cypher_query += f" RETURN d LIMIT {limit}"

            # Execute query
            result = session.run(cypher_query, params)

            documents = []
            for record in result:
                doc = dict(record["d"])

                # Convert metadata from JSON
                try:
                    doc["metadata"] = json.loads(doc["metadata"])
                except (json.JSONDecodeError, TypeError):
                    doc["metadata"] = {}

                documents.append(doc)

            return documents

    def find_elements(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find elements matching query."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # Start with base query
            cypher_query = "MATCH (e:Element)"
            params = {}

            # Apply filters if provided
            if query:
                conditions = []

                for key, value in query.items():
                    if key == "metadata":
                        # Metadata filters require special handling
                        for meta_key, meta_value in value.items():
                            # For simplicity, we'll check if the JSON contains this key/value
                            conditions.append(f"e.metadata CONTAINS $meta_{meta_key}")
                            params[f"meta_{meta_key}"] = f'"{meta_key}":"{meta_value}"'
                    else:
                        conditions.append(f"e.{key} = ${key}")
                        params[key] = value

                if conditions:
                    cypher_query += " WHERE " + " AND ".join(conditions)

            # Add return and limit
            cypher_query += f" RETURN e, id(e) AS element_pk LIMIT {limit}"

            # Execute query
            result = session.run(cypher_query, params)

            elements = []
            for record in result:
                element = dict(record["e"])
                element["element_pk"] = record["element_pk"]

                # Convert metadata from JSON
                try:
                    element["metadata"] = json.loads(element["metadata"])
                except (json.JSONDecodeError, TypeError):
                    element["metadata"] = {}

                elements.append(element)

            return elements

    def search_elements_by_content(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search elements by content preview."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (e:Element)
                WHERE e.content_preview CONTAINS $search_text
                RETURN e, id(e) AS element_pk
                LIMIT $limit
                """,
                search_text=search_text,
                limit=limit
            )

            elements = []
            for record in result:
                element = dict(record["e"])
                element["element_pk"] = record["element_pk"]

                # Convert metadata from JSON
                try:
                    element["metadata"] = json.loads(element["metadata"])
                except (json.JSONDecodeError, TypeError):
                    element["metadata"] = {}

                elements.append(element)

            return elements

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all associated elements and relationships."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # Check if document exists
            result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                RETURN d.doc_id AS doc_id
                """,
                doc_id=doc_id
            )

            if not result.single():
                return False

            # Delete the document and all its elements and relationships
            session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                OPTIONAL MATCH (e:Element)-[:BELONGS_TO]->(d)
                OPTIONAL MATCH (e)-[r:RELATES_TO]->()
                OPTIONAL MATCH ()-[r2:RELATES_TO]->(e)
                OPTIONAL MATCH (e)-[r3:CHILD_OF]->()
                OPTIONAL MATCH ()-[r4:CHILD_OF]->(e)
                OPTIONAL MATCH (e)-[r5:BELONGS_TO]->()
                DELETE r, r2, r3, r4, r5, e, d
                """,
                doc_id=doc_id
            )

            # Delete embeddings in a separate query
            session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                OPTIONAL MATCH (e:Element)-[:BELONGS_TO]->(d)
                OPTIONAL MATCH (emb:Embedding)-[:EMBEDDING_OF]->(e)
                DELETE emb
                """,
                doc_id=doc_id
            )

            return True

    def store_relationship(self, relationship: Dict[str, Any]) -> None:
        """
        Store a relationship between elements.

        Args:
            relationship: Relationship data
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # Convert metadata to JSON
            metadata_json = json.dumps(relationship.get("metadata", {}))

            # Create the relationship
            session.run(
                """
                MATCH (source:Element {element_id: $source_id})
                MATCH (target:Element {element_id: $target_reference})
                MERGE (source)-[r:RELATES_TO {relationship_id: $relationship_id}]->(target)
                SET r.relationship_type = $relationship_type,
                    r.metadata = $metadata
                """,
                relationship_id=relationship["relationship_id"],
                source_id=relationship.get("source_id", ""),
                target_reference=relationship.get("target_reference", ""),
                relationship_type=relationship.get("relationship_type", ""),
                metadata=metadata_json
            )

    def delete_relationships_for_element(self, element_id: str, relationship_type: str = None) -> None:
        """
        Delete relationships for an element.

        Args:
            element_id: Element ID
            relationship_type: Optional relationship type to filter by
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # Delete relationships where element is the source
            if relationship_type:
                session.run(
                    """
                    MATCH (source:Element {element_id: $element_id})-[r:RELATES_TO]->(target)
                    WHERE r.relationship_type = $relationship_type
                    DELETE r
                    """,
                    element_id=element_id,
                    relationship_type=relationship_type
                )

                # Delete relationships where element is the target
                session.run(
                    """
                    MATCH (source)-[r:RELATES_TO]->(target:Element {element_id: $element_id})
                    WHERE r.relationship_type = $relationship_type
                    DELETE r
                    """,
                    element_id=element_id,
                    relationship_type=relationship_type
                )
            else:
                # Delete all relationships regardless of type
                session.run(
                    """
                    MATCH (source:Element {element_id: $element_id})-[r:RELATES_TO]->()
                    DELETE r
                    """,
                    element_id=element_id
                )

                session.run(
                    """
                    MATCH ()-[r:RELATES_TO]->(target:Element {element_id: $element_id})
                    DELETE r
                    """,
                    element_id=element_id
                )

    # Embedding functions
    def store_embedding(self, element_pk: str, embedding: List[float]) -> None:
        """Store embedding for an element."""
        if not self.driver:
            raise ValueError("Database not initialized")

        # Convert embedding to a JSON string for storage
        embedding_json = json.dumps(embedding)

        with self.driver.session(database=self.database) as session:
            if str(element_pk).isdigit():
                # Using Neo4j internal ID
                session.run(
                    """
                    MATCH (e:Element)
                    WHERE id(e) = $element_pk
                    MERGE (emb:Embedding {element_pk: $element_pk})
                    SET emb.embedding = $embedding,
                        emb.dimensions = $dimensions,
                        emb.created_at = $created_at
                    MERGE (emb)-[:EMBEDDING_OF]->(e)
                    """,
                    element_pk=int(element_pk),
                    embedding=embedding_json,
                    dimensions=len(embedding),
                    created_at=time.time()
                )
            else:
                # Using element_id string
                session.run(
                    """
                    MATCH (e:Element {element_id: $element_id})
                    WITH e, id(e) AS element_pk
                    MERGE (emb:Embedding {element_pk: element_pk})
                    SET emb.embedding = $embedding,
                        emb.dimensions = $dimensions,
                        emb.created_at = $created_at
                    MERGE (emb)-[:EMBEDDING_OF]->(e)
                    """,
                    element_id=str(element_pk),
                    embedding=embedding_json,
                    dimensions=len(embedding),
                    created_at=time.time()
                )

    def get_embedding(self, element_pk: int) -> Optional[List[float]]:
        """Get embedding for an element."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            if str(element_pk).isdigit():
                result = session.run(
                    """
                    MATCH (emb:Embedding {element_pk: $element_pk})
                    RETURN emb.embedding AS embedding
                    """,
                    element_pk=int(element_pk)
                )
            else:
                result = session.run(
                    """
                    MATCH (e:Element {element_id: $element_id})
                    WITH id(e) AS element_pk
                    MATCH (emb:Embedding {element_pk: element_pk})
                    RETURN emb.embedding AS embedding
                    """,
                    element_id=str(element_pk)
                )

            record = result.single()
            if not record:
                return None

            # Convert from JSON string back to list
            try:
                return json.loads(record["embedding"])
            except (json.JSONDecodeError, TypeError):
                return None

    def search_by_embedding(self, query_embedding: List[float], limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        """
        Search elements by embedding similarity.

        Note: This is a simplified implementation. Neo4j supports vector search plugins
        like neo4j-vector-search for more efficient similarity search.
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        # Convert query embedding to JSON
        query_embedding_json = json.dumps(query_embedding)

        with self.driver.session(database=self.database) as session:
            # Start building query
            cypher_query = """
            // Load custom procedures for vector similarity
            CALL apoc.custom.declareFunction(
                'vector.cosineSimilarity',
                'RETURN apoc.algo.cosineSimilarity($embedding1, $embedding2) AS similarity',
                'DOUBLE',
                [['embedding1', 'LIST'], ['embedding2', 'LIST']]
            )
            
            // Main query
            MATCH (emb:Embedding)-[:EMBEDDING_OF]->(e:Element)-[:BELONGS_TO]->(d:Document)
            WHERE emb.dimensions = $dimensions
            """

            params = {
                "query_embedding": query_embedding_json,
                "dimensions": len(query_embedding)
            }

            # Add filter criteria if provided
            if filter_criteria:
                for key, value in filter_criteria.items():
                    if key == "element_type" and isinstance(value, list):
                        # Handle list of allowed element types
                        cypher_query += " AND e.element_type IN $element_types"
                        params["element_types"] = value
                    elif key == "doc_id" and isinstance(value, list):
                        # Handle list of document IDs to include
                        cypher_query += " AND e.doc_id IN $doc_ids"
                        params["doc_ids"] = value
                    elif key == "exclude_doc_id" and isinstance(value, list):
                        # Handle list of document IDs to exclude
                        cypher_query += " AND NOT e.doc_id IN $exclude_doc_ids"
                        params["exclude_doc_ids"] = value
                    elif key == "exclude_doc_source" and isinstance(value, list):
                        # Handle list of document sources to exclude
                        cypher_query += " AND NOT d.source IN $exclude_sources"
                        params["exclude_sources"] = value
                    else:
                        # Simple equality filter
                        cypher_query += f" AND e.{key} = ${key}"
                        params[key] = value

            # Complete the query
            cypher_query += """
            WITH e, id(e) AS element_pk, emb, 
                 apoc.custom.vector.cosineSimilarity(
                    $query_embedding, 
                    json_extract_as_list(emb.embedding)
                 ) AS similarity
            RETURN element_pk, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """

            params["limit"] = limit

            try:
                # Execute query
                result = session.run(cypher_query, params)

                # Process results
                return [(record["element_pk"], record["similarity"]) for record in result]
            except Exception as e:
                logger.error(f"Error in vector search: {str(e)}")

                # Fall back to a simpler implementation if the vector functions fail
                return self._fallback_embedding_search(query_embedding, limit, filter_criteria)

    def _fallback_embedding_search(self, query_embedding: List[float], limit: int = 10,
                                   filter_criteria: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        """
        Fallback implementation for embedding search if vector extensions fail.
        This is much slower as it processes embeddings in Python instead of in the database.
        """
        import numpy as np

        # Convert query embedding to numpy array
        query_np = np.array(query_embedding)

        with self.driver.session(database=self.database) as session:
            # Build query to fetch embeddings
            cypher_query = """
            MATCH (emb:Embedding)-[:EMBEDDING_OF]->(e:Element)-[:BELONGS_TO]->(d:Document)
            WHERE emb.dimensions = $dimensions
            """

            params = {
                "dimensions": len(query_embedding)
            }

            # Add filter criteria if provided
            if filter_criteria:
                for key, value in filter_criteria.items():
                    if key == "element_type" and isinstance(value, list):
                        cypher_query += " AND e.element_type IN $element_types"
                        params["element_types"] = value
                    elif key == "doc_id" and isinstance(value, list):
                        cypher_query += " AND e.doc_id IN $doc_ids"
                        params["doc_ids"] = value
                    elif key == "exclude_doc_id" and isinstance(value, list):
                        cypher_query += " AND NOT e.doc_id IN $exclude_doc_ids"
                        params["exclude_doc_ids"] = value
                    elif key == "exclude_doc_source" and isinstance(value, list):
                        cypher_query += " AND NOT d.source IN $exclude_sources"
                        params["exclude_sources"] = value
                    else:
                        cypher_query += f" AND e.{key} = ${key}"
                        params[key] = value

            # Complete the query
            cypher_query += """
            RETURN id(e) AS element_pk, emb.embedding AS embedding
            """

            # Execute query
            result = session.run(cypher_query, params)

            # Process results in Python
            similarities = []
            for record in result:
                element_pk = record["element_pk"]
                embedding_json = record["embedding"]

                try:
                    # Parse embedding
                    embedding = json.loads(embedding_json)
                    embedding_np = np.array(embedding)

                    # Calculate cosine similarity
                    dot_product = np.dot(query_np, embedding_np)
                    norm1 = np.linalg.norm(query_np)
                    norm2 = np.linalg.norm(embedding_np)

                    if norm1 == 0 or norm2 == 0:
                        similarity = 0.0
                    else:
                        similarity = float(dot_product / (norm1 * norm2))

                    similarities.append((element_pk, similarity))
                except Exception as e:
                    logger.warning(f"Error processing embedding: {str(e)}")

            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]

    def search_by_text(self, search_text: str, limit: int = 10,
                       filter_criteria: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        """
        Search elements by semantic similarity to the provided text.

        Args:
            search_text: Text to search for semantically
            limit: Maximum number of results
            filter_criteria: Optional dictionary with criteria to filter results

        Returns:
            List of (element_id, similarity_score) tuples
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        try:
            # Initialize embedding generator if not already done
            if self.embedding_generator is None:
                from ..embeddings import get_embedding_generator
                from ..config import Config

                config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
                self.embedding_generator = get_embedding_generator(config)

            # Generate embedding for the search text
            query_embedding = self.embedding_generator.generate(search_text)

            # Use the embedding to search
            return self.search_by_embedding(query_embedding, limit, filter_criteria)

        except Exception as e:
            logger.error(f"Error in semantic search by text: {str(e)}")
            return []
