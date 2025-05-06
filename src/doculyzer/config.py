import json
import logging
import os
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
load_dotenv()

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the document pointer system."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from file or default settings.

        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        self.config = self._get_default_config()
        self._db_instance = None  # Add a database instance cache

        logger.debug(f"Initializing config, working directory: {os.getcwd()}")

        if config_path and os.path.exists(config_path):
            logger.debug(f"Loading config from: {config_path}")
            self._load_config(config_path)
        else:
            logger.debug("No config path provided or file not found, using defaults")

        self._validate_config()

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            "storage": {
                "path": "./data",
                "backend": "file"  # Options: file, sqlite, duckdb
            },
            "embedding": {
                "enabled": False,
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": 512,
                "overlap": 128
            },
            "content_sources": [],
            "relationship_detection": {
                "enabled": True,
                "link_pattern": r"\[\[(.*?)\]\]|href=[\"\'](.*?)[\"\']"
            },
            "logging": {
                "level": "INFO",
                "file": "./logs/docpointer.log"
            }
        }

    def _replace_env_vars(self, value: Any) -> Any:
        """
        Replace environment variables in string values.

        Args:
            value: The value to process

        Returns:
            The processed value with environment variables replaced
        """
        if isinstance(value, str):
            # Match ${VAR} or $VAR patterns
            pattern = r'\${([^}]+)}|\$([a-zA-Z0-9_]+)'

            def replace_match(match):
                env_var = match.group(1) or match.group(2)
                default_value = None

                # Handle default values with ${VAR:-default} syntax
                if ':-' in env_var:
                    env_var, default_value = env_var.split(':-', 1)

                env_value = os.environ.get(env_var)
                if env_value is None:
                    if default_value is not None:
                        logger.debug(f"Environment variable {env_var} not found, using default: {default_value}")
                        return default_value
                    logger.warning(f"Environment variable {env_var} not found and no default provided")
                    return match.group(0)  # Return the original placeholder if no value found

                logger.debug(f"Replaced environment variable {env_var}")
                return env_value

            return re.sub(pattern, replace_match, value)
        elif isinstance(value, dict):
            return {k: self._replace_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._replace_env_vars(item) for item in value]
        return value

    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    loaded_config = json.load(f)
                    logger.debug("Loaded JSON config")
                elif config_path.endswith(('.yaml', '.yml')):
                    loaded_config = yaml.safe_load(f)
                    logger.debug("Loaded YAML config")
                else:
                    message = f"Unsupported config file format: {config_path}"
                    logger.error(message)
                    raise ValueError(message)

            # Replace environment variables in the loaded config
            loaded_config = self._replace_env_vars(loaded_config)
            logger.debug("Replaced environment variables in config")

            # Merge with default config (deep merge)
            self._deep_merge(self.config, loaded_config)
            logger.debug("Config merged with defaults")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            raise

    def _deep_merge(self, dest: Dict[str, Any], src: Dict[str, Any]) -> None:
        """
        Deep merge two dictionaries.

        Args:
            dest: Destination dictionary (modified in-place)
            src: Source dictionary
        """
        for key, value in src.items():
            if key in dest and isinstance(dest[key], dict) and isinstance(value, dict):
                self._deep_merge(dest[key], value)
            else:
                dest[key] = value

    def _validate_config(self) -> None:
        """Validate configuration settings."""
        # Ensure storage path exists or create it
        storage_path = self.get_storage_path()
        logger.debug(f"Ensuring storage path exists: {storage_path}")
        os.makedirs(storage_path, exist_ok=True)

        # Ensure logs directory exists
        log_dir = os.path.dirname(self.config["logging"]["file"])
        if log_dir:
            logger.debug(f"Ensuring log directory exists: {log_dir}")
            os.makedirs(log_dir, exist_ok=True)

        # Validate each content source has required fields
        for idx, source in enumerate(self.config.get("content_sources", [])):
            if "type" not in source:
                message = f"Content source at index {idx} is missing 'type' field"
                logger.error(message)
                raise ValueError(message)
            if "name" not in source:
                message = f"Content source at index {idx} is missing 'name' field"
                logger.error(message)
                raise ValueError(message)

        logger.debug("Config validation complete")

    def get_storage_path(self) -> str:
        """Get the storage path for document database."""
        path = self.config.get("storage", {}).get("path", "./data")
        return os.path.expanduser(path)

    def get_storage_backend(self) -> str:
        """Get the storage backend type."""
        return self.config["storage"]["backend"]

    def is_embedding_enabled(self) -> bool:
        """Check if embeddings are enabled."""
        return self.config["embedding"]["enabled"]

    def get_embedding_model(self) -> str:
        """Get the embedding model name."""
        return self.config["embedding"]["model"]

    def get_embedding_params(self) -> Dict[str, Any]:
        """Get embedding parameters."""
        return self.config["embedding"]

    def get_content_sources(self) -> List[Dict[str, Any]]:
        """Get configured content sources."""
        return self.config["content_sources"]

    def get_relationship_detection_config(self) -> Dict[str, Any]:
        """Get relationship detection configuration."""
        return self.config["relationship_detection"]

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config["logging"]

    def get_document_database(self):
        """
        Get the document database singleton instance.

        Returns:
            DocumentDatabase instance
        """
        if self._db_instance is None:
            logger.debug("Creating new database instance")
            from .storage import get_document_database
            self._db_instance = get_document_database(self.config.get('storage', {}), )
        else:
            logger.debug("Using existing database instance")

        return self._db_instance

    def initialize_database(self):
        """
        Initialize the document database.

        Returns:
            The initialized database instance
        """
        db = self.get_document_database()
        logger.debug("Initializing database")
        db.initialize()
        return db

    def close_database(self):
        """
        Close the document database connection if it exists.
        """
        if self._db_instance is not None:
            logger.debug("Closing database connection")
            self._db_instance.close()
            self._db_instance = None
        else:
            logger.debug("No database instance to close")

    def add_content_source(self, source_config: Dict[str, Any]) -> None:
        """
        Add a new content source to the configuration.

        Args:
            source_config: Content source configuration
        """
        if "type" not in source_config:
            raise ValueError("Content source is missing 'type' field")
        if "name" not in source_config:
            raise ValueError("Content source is missing 'name' field")

        self.config["content_sources"].append(source_config)
        logger.debug(f"Added content source: {source_config['name']} ({source_config['type']})")

    def save(self, path: str) -> None:
        """
        Save current configuration to file.

        Args:
            path: Path to save configuration file
        """
        logger.debug(f"Saving config to: {path}")
        try:
            with open(path, 'w') as f:
                if path.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                    logger.debug("Saved config as JSON")
                elif path.endswith(('.yaml', '.yml')):
                    yaml.dump(self.config, f, default_flow_style=False)
                    logger.debug("Saved config as YAML")
                else:
                    message = f"Unsupported config file format: {path}"
                    logger.error(message)
                    raise ValueError(message)
        except Exception as e:
            logger.error(f"Error saving config to {path}: {str(e)}")
            raise
