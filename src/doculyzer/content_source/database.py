import json
import logging
from typing import Dict, Any, List, Optional

import sqlalchemy

from .base import ContentSource

logger = logging.getLogger(__name__)


class DatabaseContentSource(ContentSource):
    """Content source for database blob columns or JSON-structured columns."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the database content source."""
        super().__init__(config)
        self.connection_string = config.get("connection_string")
        self.query = config.get("query")
        self.id_column = config.get("id_column", "id")
        self.content_column = config.get("content_column", "content")
        self.metadata_columns = config.get("metadata_columns", [])
        self.timestamp_column = config.get("timestamp_column")

        # New configuration options for JSON document mode
        self.json_mode = config.get("json_mode", False)
        self.json_columns = config.get("json_columns", [])
        self.json_include_metadata = config.get("json_include_metadata", True)

        # Initialize database connection
        self.engine = None
        if self.connection_string:
            self.engine = sqlalchemy.create_engine(self.connection_string)

    def fetch_document(self, source_id: str) -> Dict[str, Any]:
        """Fetch document content from database."""
        if not self.engine:
            raise ValueError("Database not configured")

        # Extract the actual ID from the fully qualified source identifier
        # Format: db://<connection>/<query>/<id_column>/<id_value>/<content_column>
        parts = source_id.split('/')
        if len(parts) >= 5 and parts[0] == 'db:':
            actual_id = parts[-2]  # Second to last part is the ID value
        else:
            actual_id = source_id

        try:
            if self.json_mode:
                return self._fetch_json_document(actual_id)
            else:
                return self._fetch_blob_document(actual_id)
        except Exception as e:
            logger.error(f"Error fetching document {source_id} from database: {str(e)}")
            raise

    def _fetch_blob_document(self, source_id: str) -> Dict[str, Any]:
        """Fetch document as a blob from a single column."""
        # Build query to fetch a specific document
        query = f"""
        SELECT {self.id_column}, {self.content_column}
        {', ' + ', '.join(self.metadata_columns) if self.metadata_columns else ''}
        {', ' + self.timestamp_column if self.timestamp_column else ''}
        FROM ({self.query}) as subquery
        WHERE {self.id_column} = :id
        """

        with self.engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(query), {"id": source_id})
            row = result.fetchone()

            if not row:
                raise ValueError(f"Document not found: {source_id}")

            # Extract content and metadata
            content = row[self.content_column]

            # If content is bytes, decode to string
            if isinstance(content, bytes):
                content = content.decode('utf-8')

            metadata = {}
            for col in self.metadata_columns:
                metadata[col] = row[col]

            if self.timestamp_column:
                metadata["last_modified"] = row[self.timestamp_column]

            # Create a fully qualified source identifier for database content
            db_source = f"db://{self.connection_string.split('://')[1]}/{self.query}/{self.id_column}/{source_id}/{self.content_column}"

            return {
                "id": db_source,  # Use a fully qualified database identifier
                "content": content,
                "metadata": metadata,
                "content_hash": self.get_content_hash(content)
            }

    def _fetch_json_document(self, source_id: str) -> Dict[str, Any]:
        """Fetch document as a JSON structure from multiple columns."""
        # If no JSON columns specified, use all non-ID columns
        columns_to_fetch = self.json_columns or []

        # Build query to fetch a specific document with all needed columns
        needed_columns = [self.id_column]

        # If no specific JSON columns provided, fetch all columns except ID
        if not columns_to_fetch:
            # We'll need a query to get column names first
            table_name = self.query
            if table_name.strip().lower().startswith("select"):
                # It's a complex query, we'll need to wrap it
                column_query = f"SELECT * FROM ({self.query}) as subquery LIMIT 1"
            else:
                # It's a simple table name
                column_query = f"SELECT * FROM {table_name} LIMIT 1"

            with self.engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(column_query))
                # Get all column names from result
                all_columns = result.keys()
                # Filter out the ID column
                columns_to_fetch = [col for col in all_columns if col != self.id_column]

        # Add all columns we need to fetch
        needed_columns.extend(columns_to_fetch)

        # Add metadata columns if not already included and if we should include them
        if self.json_include_metadata:
            for col in self.metadata_columns:
                if col not in needed_columns:
                    needed_columns.append(col)

            if self.timestamp_column and self.timestamp_column not in needed_columns:
                needed_columns.append(self.timestamp_column)

        # Build and execute the query
        query = f"""
        SELECT {', '.join(needed_columns)}
        FROM ({self.query}) as subquery
        WHERE {self.id_column} = :id
        """

        with self.engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(query), {"id": source_id})
            row = result.fetchone()

            if not row:
                raise ValueError(f"Document not found: {source_id}")

            # Create a dictionary with all the column data
            json_data = {}
            for col in columns_to_fetch:
                value = row[col]
                # Handle special types that need conversion
                if isinstance(value, bytes):
                    try:
                        # Try to decode as UTF-8 string
                        value = value.decode('utf-8')
                    except UnicodeDecodeError:
                        # If it's not valid UTF-8, encode as base64
                        import base64
                        value = base64.b64encode(value).decode('ascii')

                # Handle dates and other types that need JSON serialization
                try:
                    # Test if value is JSON serializable
                    json.dumps({col: value})
                    json_data[col] = value
                except (TypeError, OverflowError):
                    # Convert to string if not serializable
                    json_data[col] = str(value)

            # Convert to JSON string
            content = json.dumps(json_data, ensure_ascii=False, indent=2)

            # Extract metadata (if not included in the JSON content)
            metadata = {}
            if not self.json_include_metadata:
                for col in self.metadata_columns:
                    metadata[col] = row[col]

                if self.timestamp_column:
                    metadata["last_modified"] = row[self.timestamp_column]
            else:
                # If metadata is included in JSON, still add timestamp to metadata dict
                if self.timestamp_column:
                    metadata["last_modified"] = row[self.timestamp_column]

            # Create a fully qualified source identifier for database content
            columns_part = "_".join(columns_to_fetch[:3]) + (
                f"_plus_{len(columns_to_fetch) - 3}_more" if len(columns_to_fetch) > 3 else "")
            db_source = f"db://{self.connection_string.split('://')[1]}/{self.query}/{self.id_column}/{source_id}/{columns_part}/json"

            return {
                "id": db_source,  # Use a fully qualified database identifier
                "content": content,
                "metadata": metadata,
                "content_hash": self.get_content_hash(content),
                "content_type": "application/json"  # Specify content type as JSON
            }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List available documents in database."""
        if not self.engine:
            raise ValueError("Database not configured")

        # Build query to list documents
        columns = [self.id_column]
        columns.extend(self.metadata_columns)
        if self.timestamp_column:
            columns.append(self.timestamp_column)

        query = f"""
        SELECT {', '.join(columns)}
        FROM ({self.query}) as subquery
        """

        try:
            results = []
            with self.engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(query))

                for row in result:
                    metadata = {}
                    for col in self.metadata_columns:
                        metadata[col] = row[col]

                    if self.timestamp_column:
                        metadata["last_modified"] = row[self.timestamp_column]

                    # Create a fully qualified source identifier
                    if self.json_mode:
                        columns_part = "_".join(self.json_columns[:3]) + (
                            f"_plus_{len(self.json_columns) - 3}_more" if len(self.json_columns) > 3 else "")
                        db_source = f"db://{self.connection_string.split('://')[1]}/{self.query}/{self.id_column}/{row[self.id_column]}/{columns_part}/json"
                    else:
                        db_source = f"db://{self.connection_string.split('://')[1]}/{self.query}/{self.id_column}/{row[self.id_column]}/{self.content_column}"

                    results.append({
                        "id": db_source,  # Use fully qualified path
                        "metadata": metadata
                    })

            return results
        except Exception as e:
            logger.error(f"Error listing documents from database: {str(e)}")
            raise

    def has_changed(self, source_id: str, last_modified: Optional[float] = None) -> bool:
        """Check if document has changed based on timestamp column."""
        if not self.engine or not self.timestamp_column:
            # Can't determine changes without timestamp
            return True

        # Extract the actual ID from the fully qualified source identifier
        # Format: db://<connection>/<query>/<id_column>/<id_value>/<content_column>
        parts = source_id.split('/')
        if len(parts) >= 5 and parts[0] == 'db:':
            actual_id = parts[-2]  # Second to last part is the ID value
        else:
            actual_id = source_id

        query = f"""
        SELECT {self.timestamp_column}
        FROM ({self.query}) as subquery
        WHERE {self.id_column} = :id
        """

        try:
            with self.engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(query), {"id": actual_id})
                row = result.fetchone()

                if not row:
                    return False

                current_timestamp = row[self.timestamp_column]

                if last_modified is None:
                    return True

                # Compare timestamps
                return current_timestamp > last_modified
        except Exception as e:
            logger.error(f"Error checking changes for {source_id}: {str(e)}")
            return True


'''
storage:
  backend: neo4j  # Can be neo4j, sqlite, etc.
  path: "./data"  
  
  # Neo4j-specific configuration
  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "password"
    database: "doculyzer"

embedding:
  enabled: true
  model: "sentence-transformers/all-MiniLM-L6-v2"

content_sources:
  # Example of a blob-based database content source
  - name: "database-blobs"
    type: "database"
    connection_string: "postgresql://user:password@localhost:5432/mydatabase"
    query: "SELECT * FROM documents"
    id_column: "doc_id"
    content_column: "content_blob" 
    metadata_columns: ["title", "author", "created_date"]
    timestamp_column: "updated_at"
    
  # Example of a JSON-structured database content source
  - name: "database-json"
    type: "database"
    connection_string: "mysql://user:password@localhost:3306/customer_db"
    query: "SELECT * FROM customers"
    id_column: "customer_id"
    json_mode: true
    json_columns: ["first_name", "last_name", "email", "address", "phone_number", "signup_date"]
    metadata_columns: ["account_status", "customer_type"]
    timestamp_column: "last_modified"
    json_include_metadata: true  # Include metadata columns in the JSON document
    
  # Example of automatic column discovery (all columns except ID will be in JSON)
  - name: "database-json-auto"
    type: "database"
    connection_string: "sqlite:///local_database.db"
    query: "products"  # Simple table name
    id_column: "product_id"
    json_mode: true
    # No json_columns specified - will automatically use all non-ID columns
    metadata_columns: ["category", "supplier"]
    timestamp_column: "updated_at"
    
  # Example with a complex query
  - name: "database-complex-query"
    type: "database"
    connection_string: "mssql+pyodbc://user:password@server/database"
    query: "SELECT o.order_id, c.customer_name, o.order_date, p.product_name, oi.quantity 
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN products p ON oi.product_id = p.product_id"
    id_column: "order_id"
    json_mode: true
    json_columns: ["customer_name", "order_date", "product_name", "quantity"]
    timestamp_column: "order_date"

relationship_detection:
  enabled: true

logging:
  level: "INFO"
  file: "./logs/doculyzer.log"
'''
