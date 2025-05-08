import base64
import json
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

# Import types for type checking only - these won't be imported at runtime
if TYPE_CHECKING:
    import sqlalchemy
    from sqlalchemy import text
    from sqlalchemy.engine import Engine, Connection, Result

    # Define type aliases for type checking
    SQLAlchemyEngineType = Engine
    SQLAlchemyConnectionType = Connection
    SQLAlchemyResultType = Result
    SQLAlchemyTextType = text
else:
    # Runtime type aliases - use generic Python types
    SQLAlchemyEngineType = Any
    SQLAlchemyConnectionType = Any
    SQLAlchemyResultType = Any
    SQLAlchemyTextType = Any

from .base import ContentSource

logger = logging.getLogger(__name__)

# Define global flags for availability - these will be set at runtime
SQLALCHEMY_AVAILABLE = False

# Try to import SQLAlchemy conditionally
try:
    import sqlalchemy
    from sqlalchemy import text

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    logger.warning("SQLAlchemy not available. Install with 'pip install sqlalchemy'.")


class DatabaseContentSource(ContentSource):
    """Content source for database blob columns or JSON-structured columns."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the database content source.

        Args:
            config: Configuration dictionary with connection details and query parameters
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for DatabaseContentSource but not available")

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
        self.engine: Optional[SQLAlchemyEngineType] = None
        if self.connection_string:
            try:
                self.engine = sqlalchemy.create_engine(self.connection_string)
                logger.debug(f"Successfully created SQLAlchemy engine for {self.get_safe_connection_string()}")
            except Exception as e:
                logger.error(f"Error creating SQLAlchemy engine: {str(e)}")
                raise

    def get_safe_connection_string(self) -> str:
        """Return a safe version of the connection string with password masked."""
        if not self.connection_string:
            return "<no connection string>"

        try:
            parts = self.connection_string.split("://")
            if len(parts) != 2:
                return "<malformed connection string>"

            protocol = parts[0]
            connection_parts = parts[1].split("@")

            if len(connection_parts) == 2:
                # Connection string with authentication
                auth_parts = connection_parts[0].split(":")
                if len(auth_parts) == 2:
                    username = auth_parts[0]
                    masked_conn = f"{protocol}://{username}:****@{connection_parts[1]}"
                    return masked_conn

            # If we can't parse properly, return a generic masked string
            return f"{protocol}://*****"
        except Exception:
            return "<connection string parsing error>"

    def fetch_document(self, source_id: str) -> Dict[str, Any]:
        """
        Fetch document content from database.

        Args:
            source_id: Document identifier (can be fully qualified or simple ID)

        Returns:
            Dictionary with document content and metadata

        Raises:
            ValueError: If database is not configured or document not found
        """
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
        except ValueError as e:
            # Re-raise ValueError for not found or configuration issues
            raise
        except Exception as e:
            logger.error(f"Error fetching document {source_id} from database: {str(e)}")
            raise

    def _fetch_blob_document(self, source_id: str) -> Dict[str, Any]:
        """
        Fetch document as a blob from a single column.

        Args:
            source_id: Document identifier

        Returns:
            Dictionary with document content and metadata
        """
        if not self.engine:
            raise ValueError("Database not configured")

        # Build query to fetch a specific document
        columns_clause = f"{self.id_column}, {self.content_column}"

        # Add metadata columns if specified
        if self.metadata_columns:
            columns_clause += f", {', '.join(self.metadata_columns)}"

        # Add timestamp column if specified
        if self.timestamp_column:
            columns_clause += f", {self.timestamp_column}"

        query = f"""
        SELECT {columns_clause}
        FROM ({self.query}) as subquery
        WHERE {self.id_column} = :id
        """

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"id": source_id})
                row = result.fetchone()

                if not row:
                    raise ValueError(f"Document not found: {source_id}")

                # Extract content and metadata
                content = row[self.content_column]

                # If content is bytes, decode to string
                if isinstance(content, bytes):
                    try:
                        content = content.decode('utf-8')
                    except UnicodeDecodeError:
                        # If it's not valid UTF-8, encode as base64
                        content = f"<binary data: {len(content)} bytes>"
                        logger.warning(f"Could not decode binary content from {source_id} as UTF-8")

                # Extract metadata
                metadata = {}
                for col in self.metadata_columns:
                    if col in row._mapping:
                        metadata[col] = row[col]

                if self.timestamp_column and self.timestamp_column in row._mapping:
                    metadata["last_modified"] = row[self.timestamp_column]

                # Create a fully qualified source identifier for database content
                conn_str_safe = self.connection_string.split('://')[1] if '://' in self.connection_string else 'unknown'
                db_source = f"db://{conn_str_safe}/{self.query}/{self.id_column}/{source_id}/{self.content_column}"

                return {
                    "id": db_source,  # Use a fully qualified database identifier
                    "content": content,
                    "metadata": metadata,
                    "content_hash": self.get_content_hash(content)
                }
        except ValueError:
            # Re-raise ValueError for not found
            raise
        except Exception as e:
            logger.error(f"Error fetching blob document {source_id}: {str(e)}")
            raise

    def _fetch_json_document(self, source_id: str) -> Dict[str, Any]:
        """
        Fetch document as a JSON structure from multiple columns.

        Args:
            source_id: Document identifier

        Returns:
            Dictionary with document content as JSON and metadata
        """
        if not self.engine:
            raise ValueError("Database not configured")

        # If no JSON columns specified, use all non-ID columns
        columns_to_fetch = self.json_columns or []

        # Build query to fetch a specific document with all needed columns
        needed_columns = [self.id_column]

        # If no specific JSON columns provided, fetch all columns except ID
        if not columns_to_fetch:
            try:
                # We'll need a query to get column names first
                table_name = self.query
                if table_name.strip().lower().startswith("select"):
                    # It's a complex query, we'll need to wrap it
                    column_query = f"SELECT * FROM ({self.query}) as subquery LIMIT 1"
                else:
                    # It's a simple table name
                    column_query = f"SELECT * FROM {table_name} LIMIT 1"

                with self.engine.connect() as conn:
                    result = conn.execute(text(column_query))
                    # Get all column names from result
                    all_columns = result.keys()
                    # Filter out the ID column
                    columns_to_fetch = [col for col in all_columns if col != self.id_column]
            except Exception as e:
                logger.error(f"Error discovering columns for {source_id}: {str(e)}")
                # Default to empty list if discovery fails
                columns_to_fetch = []

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

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"id": source_id})
                row = result.fetchone()

                if not row:
                    raise ValueError(f"Document not found: {source_id}")

                # Create a dictionary with all the column data
                json_data = {}
                for col in columns_to_fetch:
                    if col in row._mapping:
                        value = row[col]
                        # Handle special types that need conversion
                        if isinstance(value, bytes):
                            try:
                                # Try to decode as UTF-8 string
                                value = value.decode('utf-8')
                            except UnicodeDecodeError:
                                # If it's not valid UTF-8, encode as base64
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
                        if col in row._mapping:
                            metadata[col] = row[col]

                    if self.timestamp_column and self.timestamp_column in row._mapping:
                        metadata["last_modified"] = row[self.timestamp_column]
                else:
                    # If metadata is included in JSON, still add timestamp to metadata dict
                    if self.timestamp_column and self.timestamp_column in row._mapping:
                        metadata["last_modified"] = row[self.timestamp_column]

                # Create a fully qualified source identifier for database content
                conn_str_safe = self.connection_string.split('://')[1] if '://' in self.connection_string else 'unknown'
                columns_part = "_".join(columns_to_fetch[:3]) + (
                    f"_plus_{len(columns_to_fetch) - 3}_more" if len(columns_to_fetch) > 3 else "")
                db_source = f"db://{conn_str_safe}/{self.query}/{self.id_column}/{source_id}/{columns_part}/json"

                return {
                    "id": db_source,  # Use a fully qualified database identifier
                    "content": content,
                    "metadata": metadata,
                    "content_hash": self.get_content_hash(content),
                    "content_type": "application/json"  # Specify content type as JSON
                }
        except ValueError:
            # Re-raise ValueError for not found
            raise
        except Exception as e:
            logger.error(f"Error fetching JSON document {source_id}: {str(e)}")
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List available documents in database.

        Returns:
            List of document metadata dictionaries
        """
        if not self.engine:
            raise ValueError("Database not configured")

        # Build query to list documents
        columns = [self.id_column]
        columns.extend([col for col in self.metadata_columns if col != self.id_column])
        if self.timestamp_column and self.timestamp_column not in columns:
            columns.append(self.timestamp_column)

        query = f"""
        SELECT {', '.join(columns)}
        FROM ({self.query}) as subquery
        """

        try:
            results = []
            with self.engine.connect() as conn:
                result = conn.execute(text(query))

                for row in result:
                    metadata = {}
                    for col in self.metadata_columns:
                        if col in row._mapping:
                            metadata[col] = row[col]

                    if self.timestamp_column and self.timestamp_column in row._mapping:
                        metadata["last_modified"] = row[self.timestamp_column]

                    # Create a fully qualified source identifier
                    conn_str_safe = self.connection_string.split('://')[
                        1] if '://' in self.connection_string else 'unknown'

                    if self.json_mode:
                        columns_part = "_".join(self.json_columns[:3]) + (
                            f"_plus_{len(self.json_columns) - 3}_more" if len(self.json_columns) > 3 else "")
                        db_source = f"db://{conn_str_safe}/{self.query}/{self.id_column}/{row[self.id_column]}/{columns_part}/json"
                    else:
                        db_source = f"db://{conn_str_safe}/{self.query}/{self.id_column}/{row[self.id_column]}/{self.content_column}"

                    results.append({
                        "id": db_source,  # Use fully qualified path
                        "metadata": metadata
                    })

            return results
        except Exception as e:
            logger.error(f"Error listing documents from database: {str(e)}")
            raise

    def has_changed(self, source_id: str, last_modified: Optional[float] = None) -> bool:
        """
        Check if document has changed based on timestamp column.

        Args:
            source_id: Document identifier
            last_modified: Previous last modified timestamp for comparison

        Returns:
            True if document has changed or we can't determine, False otherwise
        """
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
                result = conn.execute(text(query), {"id": actual_id})
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
            return True  # Assume changed if there's an error
