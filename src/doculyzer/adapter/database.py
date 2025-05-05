"""
Database adapter module for the document pointer system.

This module provides an adapter to retrieve content from database sources.
"""

import logging
import re
import sqlite3
from typing import Dict, Any, Optional, Union

from .base import ContentSourceAdapter
from ..document_parser.document_type_detector import DocumentTypeDetector

logger = logging.getLogger(__name__)


class DatabaseAdapter(ContentSourceAdapter):
    """Adapter for database blob content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the database adapter."""
        super().__init__(config)
        self.config = config or {}
        self.connections = {}  # Cache for database connections

    def get_content(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get content from a database.

        Args:
            location_data: Location data with database connection info

        Returns:
            Dictionary with content and metadata

        Raises:
            ValueError: If source is invalid or record not found
        """
        source = location_data.get("source", "")

        # Parse database connection info from source
        db_info = self._parse_db_source(source)
        if not db_info:
            raise ValueError(f"Invalid database source: {source}")

        # Get database connection
        conn = self._get_connection(db_info)

        # Extract content based on location data
        content = self._fetch_record(conn, db_info)
        if content is None:
            raise ValueError(f"Content not found for {source}")

        # Determine the content type
        content_type = DocumentTypeDetector.detect_from_content(
            content,
            metadata={
                "content_column": db_info.get("content_column", ""),
                "content_type": db_info.get("content_type", "")
            }
        )

        # Return content with metadata
        return {
            "content": content,
            "content_type": content_type,
            "metadata": {
                "database": db_info["connection_id"],
                "table": db_info["table"],
                "record_id": db_info["pk_value"],
                "content_column": db_info["content_column"]
            }
        }

    def supports_location(self, location_data: Dict[str, Any]) -> bool:
        """
        Check if this adapter supports the location.

        Args:
            location_data: Content location data

        Returns:
            True if supported, False otherwise
        """
        source = location_data.get("source", "")
        # Source must be a database URI
        return source.startswith("db://")

    def get_binary_content(self, location_data: Dict[str, Any]) -> bytes:
        """
        Get the containing document as a binary blob.

        Args:
            location_data: Content location data

        Returns:
            Document binary content

        Raises:
            ValueError: If document cannot be retrieved
        """
        source = location_data.get("source", "")

        # Parse database connection info from source
        db_info = self._parse_db_source(source)
        if not db_info:
            raise ValueError(f"Invalid database source: {source}")

        # Get database connection
        conn = self._get_connection(db_info)

        # Build query - modify to fetch binary content if available
        table = db_info["table"]
        pk_column = db_info["pk_column"]
        pk_value = db_info["pk_value"]
        content_column = db_info["content_column"]

        query = f"SELECT {content_column} FROM {table} WHERE {pk_column} = ?"

        cursor = conn.execute(query, (pk_value,))
        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"Record not found: {pk_value}")

        content = row[0]

        # If content is already bytes, return directly
        if isinstance(content, bytes):
            return content

        # Otherwise convert string to bytes
        return content.encode('utf-8')

    @staticmethod
    def _parse_db_source(source: str) -> Optional[Dict[str, str]]:
        """
        Parse database source URI.

        Format: db://<connection_id>/<table>/<pk_column>/<pk_value>/<content_column>[/<content_type>]

        Returns:
            Dictionary with connection info or None if invalid
        """
        if not source.startswith("db://"):
            return None

        # Remove 'db://' prefix
        path = source[5:]

        # Split path
        parts = path.split('/')

        if len(parts) < 5:
            return None

        # Extract content_type if provided
        content_type = None
        if len(parts) >= 6:
            content_type = parts[5]

        return {
            "connection_id": parts[0],
            "table": parts[1],
            "pk_column": parts[2],
            "pk_value": parts[3],
            "content_column": parts[4],
            "content_type": content_type
        }

    def _get_connection(self, db_info: Dict[str, str]) -> Any:
        """
        Get database connection.

        Args:
            db_info: Database connection info

        Returns:
            Database connection

        Raises:
            ValueError: If connection cannot be established
        """
        connection_id = db_info["connection_id"]

        # Check if connection already exists in cache
        if connection_id in self.connections:
            return self.connections[connection_id]

        # Handle different database types based on connection_id
        if connection_id.endswith('.db') or connection_id.endswith('.sqlite'):
            # Assume SQLite database
            try:
                conn = sqlite3.connect(connection_id)
                conn.row_factory = sqlite3.Row

                # Cache connection
                self.connections[connection_id] = conn
                return conn
            except Exception as e:
                raise ValueError(f"Error connecting to SQLite database {connection_id}: {str(e)}")
        elif connection_id.startswith('postgres://') or connection_id.startswith('postgresql://'):
            # PostgreSQL connection
            try:
                import psycopg2
                import psycopg2.extras

                conn = psycopg2.connect(connection_id)
                conn.cursor_factory = psycopg2.extras.DictCursor

                # Cache connection
                self.connections[connection_id] = conn
                return conn
            except ImportError:
                raise ValueError("psycopg2 is required for PostgreSQL connections")
            except Exception as e:
                raise ValueError(f"Error connecting to PostgreSQL database: {str(e)}")
        elif connection_id.startswith('mysql://'):
            # MySQL connection
            try:
                import mysql.connector

                # Parse connection string
                # Format: mysql://user:password@host:port/database
                conn_parts = re.match(r'mysql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', connection_id)
                if not conn_parts:
                    raise ValueError(f"Invalid MySQL connection string: {connection_id}")

                user, password, host, port, database = conn_parts.groups()

                conn = mysql.connector.connect(
                    user=user,
                    password=password,
                    host=host,
                    port=int(port),
                    database=database
                )

                # Cache connection
                self.connections[connection_id] = conn
                return conn
            except ImportError:
                raise ValueError("mysql-connector-python is required for MySQL connections")
            except Exception as e:
                raise ValueError(f"Error connecting to MySQL database: {str(e)}")
        else:
            # Unknown database type
            raise ValueError(f"Unsupported database type for connection: {connection_id}")

    @staticmethod
    def _fetch_record(conn: Any, db_info: Dict[str, str]) -> Union[str, bytes]:
        """
        Fetch content from database.

        Args:
            conn: Database connection
            db_info: Database connection info

        Returns:
            Content as string or bytes

        Raises:
            ValueError: If record cannot be fetched
        """
        table = db_info["table"]
        pk_column = db_info["pk_column"]
        pk_value = db_info["pk_value"]
        content_column = db_info["content_column"]

        # Build query based on database type
        if isinstance(conn, sqlite3.Connection):
            # SQLite query
            query = f"SELECT {content_column} FROM {table} WHERE {pk_column} = ?"
            params = (pk_value,)
        else:
            # Generic SQL query with placeholder
            # This works for PostgreSQL, MySQL, etc.
            query = f"SELECT {content_column} FROM {table} WHERE {pk_column} = %s"
            params = (pk_value,)

        try:
            if isinstance(conn, sqlite3.Connection):
                cursor = conn.execute(query, params)
                row = cursor.fetchone()
            else:
                # For other database types
                cursor = conn.cursor()
                cursor.execute(query, params)
                row = cursor.fetchone()
                cursor.close()

            if row is None:
                return None

            # Extract content from row
            if isinstance(conn, sqlite3.Connection):
                content = row[content_column]
            else:
                # For dict-like cursors
                if isinstance(row, dict) or hasattr(row, 'keys'):
                    content = row[content_column]
                else:
                    # For tuple-like cursors, get column index
                    cursor = conn.cursor()
                    columns = [desc[0] for desc in cursor.description]
                    cursor.close()

                    col_idx = columns.index(content_column)
                    content = row[col_idx]

            # Handle binary data
            if isinstance(content, bytes):
                # Try to decode as text if appropriate
                try:
                    # Check if this might be a text blob (e.g., HTML, markdown)
                    if content.startswith(b'<') or content.startswith(b'#'):
                        return content.decode('utf-8')
                    # Otherwise return as binary
                    return content
                except UnicodeDecodeError:
                    # Definitely binary data
                    return content

            return content

        except Exception as e:
            logger.error(f"Error fetching record: {str(e)}")
            raise ValueError(f"Error fetching record: {str(e)}")

    def cleanup(self):
        """Clean up database connections."""
        for conn_id, conn in self.connections.items():
            try:
                conn.close()
                logger.debug(f"Closed database connection: {conn_id}")
            except Exception as e:
                logger.warning(f"Error closing database connection {conn_id}: {str(e)}")

        # Clear connection cache
        self.connections = {}
