"""
Document Database Module for the document pointer system.
This module stores document metadata, elements, and relationships,
while maintaining pointers to original content.
"""
import logging
from typing import Dict, Any

from .postgres import PostgreSQLDocumentDatabase
from .base import DocumentDatabase
from .file import FileDocumentDatabase
from .mongodb import MongoDBDocumentDatabase
from .neo4j import Neo4jDocumentDatabase
from .sqlite import SQLiteDocumentDatabase

logger = logging.getLogger(__name__)


def get_document_database(config: Dict[str, Any]) -> DocumentDatabase:
    """
    Factory function to create document database from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        DocumentDatabase instance

    Raises:
        ValueError: If database type is not supported
    """
    storage_path = config.get("path", "./data")
    backend_type = config.get("backend", "file")

    if backend_type == "file":
        return FileDocumentDatabase(storage_path)
    elif backend_type == "sqlite":
        return SQLiteDocumentDatabase(storage_path)
    elif backend_type.startswith("postgres"):
        return PostgreSQLDocumentDatabase(config.get("postgres", config.get("postgresql", {})))
    elif backend_type == "mongodb":
        # Extract MongoDB connection parameters from config
        conn_params = config.get("mongodb", {})
        if not conn_params:
            # Default connection parameters
            conn_params = {
                "host": "localhost",
                "port": 27017,
                "db_name": "doculyzer"
            }
        return MongoDBDocumentDatabase(conn_params)
    elif backend_type == "neo4j":
        # Extract Neo4j connection parameters from config
        neo4j_params = config.get("neo4j", {})
        if not neo4j_params:
            # Default connection parameters
            neo4j_params = {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password",
                "database": "neo4j"
            }
        return Neo4jDocumentDatabase(
            uri=neo4j_params.get("uri", "bolt://localhost:7687"),
            user=neo4j_params.get("username", "neo4j"),
            password=neo4j_params.get("password", "password"),
            database=neo4j_params.get("database", "neo4j")
        )
    else:
        raise ValueError(f"Unsupported database backend: {backend_type}")
