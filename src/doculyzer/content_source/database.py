import logging
from typing import Dict, Any, List, Optional

import sqlalchemy

from .base import ContentSource

logger = logging.getLogger(__name__)


class DatabaseContentSource(ContentSource):
    """Content source for database blob columns."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the database content source."""
        super().__init__(config)
        self.connection_string = config.get("connection_string")
        self.query = config.get("query")
        self.id_column = config.get("id_column", "id")
        self.content_column = config.get("content_column", "content")
        self.metadata_columns = config.get("metadata_columns", [])
        self.timestamp_column = config.get("timestamp_column")

        # Initialize database connection
        self.engine = None
        if self.connection_string:
            self.engine = sqlalchemy.create_engine(self.connection_string)

    def fetch_document(self, source_id: str) -> Dict[str, Any]:
        """Fetch document content from database."""
        if not self.engine:
            raise ValueError("Database not configured")

        # Build query to fetch a specific document
        query = f"""
        SELECT {self.id_column}, {self.content_column}
        {', ' + ', '.join(self.metadata_columns) if self.metadata_columns else ''}
        {', ' + self.timestamp_column if self.timestamp_column else ''}
        FROM ({self.query}) as subquery
        WHERE {self.id_column} = :id
        """

        try:
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
        except Exception as e:
            logger.error(f"Error fetching document {source_id} from database: {str(e)}")
            raise

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
