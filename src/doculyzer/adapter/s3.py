"""
S3 adapter module for the document pointer system.

This module provides an adapter to retrieve content from Amazon S3 sources.
"""

import logging
import os
import tempfile
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from .base import ContentSourceAdapter
from ..document_parser.document_type_detector import DocumentTypeDetector

logger = logging.getLogger(__name__)

# Try to import boto3, but don't fail if not available
try:
    import boto3
    # noinspection PyPackageRequirements
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None
    ClientError = None
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available. Install with 'pip install boto3' to use S3 adapter")


class S3Adapter(ContentSourceAdapter):
    """Adapter for Amazon S3 content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the S3 adapter."""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 adapter")

        super().__init__(config)

        # Configuration options
        self.default_region = self.config.get("region")
        self.use_credentials = self.config.get("use_credentials", True)
        self.temp_dir = self.config.get("temp_dir", tempfile.gettempdir())
        self.encoding_fallbacks = self.config.get(
            "encoding_fallbacks", ['utf-8', 'latin-1', 'cp1252']
        )

        # Client cache
        self.clients = {}

        # Load AWS credentials
        if self.use_credentials:
            self.credentials = {
                "aws_access_key_id": self.config.get("aws_access_key_id") or os.environ.get("AWS_ACCESS_KEY_ID"),
                "aws_secret_access_key": self.config.get("aws_secret_access_key") or os.environ.get(
                    "AWS_SECRET_ACCESS_KEY"),
                "aws_session_token": self.config.get("aws_session_token") or os.environ.get("AWS_SESSION_TOKEN"),
                "region_name": self.default_region or os.environ.get("AWS_REGION")
            }
        else:
            # Use default credentials chain
            self.credentials = {}

    def get_content(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get content from an S3 location.

        Args:
            location_data: Location data with S3 information

        Returns:
            Dictionary with content and metadata

        Raises:
            ValueError: If object cannot be retrieved
        """
        source = location_data.get("source", "")

        # Check if location is valid S3 URI
        if not source.startswith("s3://"):
            raise ValueError(f"Invalid S3 location: {source}")

        # Parse S3 URI
        bucket, key, region = self._parse_s3_uri(source)

        # Get S3 client
        s3_client = self._get_s3_client(region)

        try:
            # Get object metadata first
            try:
                head_response = s3_client.head_object(Bucket=bucket, Key=key)
                content_type = head_response.get('ContentType', '')
                size = head_response.get('ContentLength', 0)
                last_modified = head_response.get('LastModified')
                etag = head_response.get('ETag', '').strip('"')
                storage_class = head_response.get('StorageClass')
                user_metadata = {k.lower()[11:]: v for k, v in head_response.get("Metadata", {}).items()}
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                if error_code == "NoSuchKey":
                    raise ValueError(f"Object not found: s3://{bucket}/{key}")
                elif error_code == "AccessDenied":
                    raise ValueError(f"Access denied to s3://{bucket}/{key}")
                else:
                    logger.error(f"Error retrieving S3 object info: {str(e)}")
                    raise ValueError(f"Error retrieving S3 object info: {str(e)}")

            # Get the object content
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read()

            # Determine if content is binary or text
            is_binary = not self._is_text_content(content_type)

            # If text content, try to decode
            if not is_binary:
                # Try different encodings for text content
                for encoding in self.encoding_fallbacks:
                    try:
                        content = content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        if encoding == self.encoding_fallbacks[-1]:
                            # If all encodings fail, leave as binary
                            is_binary = True
                        continue

            # Determine document type
            doc_type = DocumentTypeDetector.detect(
                path=key,
                content=content,
                metadata={
                    "content_type": content_type,
                    "binary": is_binary
                }
            )

            # Create metadata
            metadata = {
                "bucket": bucket,
                "key": key,
                "region": region,
                "size": size,
                "last_modified": last_modified,
                "etag": etag,
                "storage_class": storage_class,
                "content_type": content_type,
                "is_binary": is_binary,
                "user_metadata": user_metadata
            }

            # Add filename if available from the key
            if '/' in key:
                filename = key.split('/')[-1]
                if filename:
                    metadata["filename"] = filename

            return {
                "content": content,
                "content_type": doc_type,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Error retrieving S3 content: {str(e)}")
            raise ValueError(f"Error retrieving S3 content: {str(e)}")

    def supports_location(self, location_data: Dict[str, Any]) -> bool:
        """
        Check if this adapter supports the location.

        Args:
            location_data: Location data

        Returns:
            True if supported, False otherwise
        """
        source = location_data.get("source", "")

        # Check if source is an S3 URI
        return source.startswith("s3://")

    def get_binary_content(self, location_data: Dict[str, Any]) -> bytes:
        """
        Get the S3 object as binary data.

        Args:
            location_data: Location data

        Returns:
            Binary content

        Raises:
            ValueError: If object cannot be retrieved
        """
        source = location_data.get("source", "")

        # Check if location is valid S3 URI
        if not source.startswith("s3://"):
            raise ValueError(f"Invalid S3 location: {source}")

        # Parse S3 URI
        bucket, key, region = self._parse_s3_uri(source)

        # Get S3 client
        s3_client = self._get_s3_client(region)

        try:
            # Get the object content
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                raise ValueError(f"Object not found: s3://{bucket}/{key}")
            elif error_code == "AccessDenied":
                raise ValueError(f"Access denied to s3://{bucket}/{key}")
            else:
                logger.error(f"Error retrieving S3 binary content: {str(e)}")
                raise ValueError(f"Error retrieving S3 binary content: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error retrieving S3 binary content: {str(e)}")
            raise ValueError(f"Error retrieving S3 binary content: {str(e)}")

    def get_metadata(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get metadata about the S3 object without retrieving the full content.

        Args:
            location_data: Location data

        Returns:
            Dictionary with metadata

        Raises:
            ValueError: If metadata cannot be retrieved
        """
        source = location_data.get("source", "")

        # Check if location is valid S3 URI
        if not source.startswith("s3://"):
            raise ValueError(f"Invalid S3 location: {source}")

        # Parse S3 URI
        bucket, key, region = self._parse_s3_uri(source)

        # Get S3 client
        s3_client = self._get_s3_client(region)

        try:
            # Get object metadata
            head_response = s3_client.head_object(Bucket=bucket, Key=key)
            content_type = head_response.get('ContentType', '')
            size = head_response.get('ContentLength', 0)
            last_modified = head_response.get('LastModified')
            etag = head_response.get('ETag', '').strip('"')
            storage_class = head_response.get('StorageClass')
            user_metadata = {k.lower()[11:]: v for k, v in head_response.get("Metadata", {}).items()}

            # Determine if binary or text
            is_binary = not self._is_text_content(content_type)

            # Determine document type without content
            doc_type = DocumentTypeDetector.detect_from_mime(key) or "binary"

            # Create metadata
            metadata = {
                "bucket": bucket,
                "key": key,
                "region": region,
                "size": size,
                "last_modified": last_modified,
                "etag": etag,
                "storage_class": storage_class,
                "content_type": content_type,
                "doc_type": doc_type,
                "is_binary": is_binary,
                "user_metadata": user_metadata
            }

            # Add filename if available from the key
            if '/' in key:
                filename = key.split('/')[-1]
                if filename:
                    metadata["filename"] = filename

            return metadata

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                raise ValueError(f"Object not found: s3://{bucket}/{key}")
            elif error_code == "AccessDenied":
                raise ValueError(f"Access denied to s3://{bucket}/{key}")
            else:
                logger.error(f"Error retrieving S3 metadata: {str(e)}")
                raise ValueError(f"Error retrieving S3 metadata: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error retrieving S3 metadata: {str(e)}")
            raise ValueError(f"Error retrieving S3 metadata: {str(e)}")

    def resolve_uri(self, uri: str) -> Dict[str, Any]:
        """
        Parse an S3 URI into location data.

        Args:
            uri: S3 URI string (s3://bucket/key)

        Returns:
            Dictionary with parsed location data

        Raises:
            ValueError: If URI cannot be parsed
        """
        if not uri.startswith("s3://"):
            raise ValueError(f"Not an S3 URI: {uri}")

        # Just use the URI as source
        return {"source": uri}

    def _parse_s3_uri(self, uri: str) -> tuple:
        """
        Parse an S3 URI into bucket, key, and region.

        Args:
            uri: S3 URI (s3://bucket/key)

        Returns:
            Tuple of (bucket, key, region)

        Raises:
            ValueError: If URI format is invalid
        """
        # Check for region in extended format: s3://region.bucket/key
        region = None
        if uri.startswith("s3://") and "." in uri[5:]:
            parts = uri[5:].split(".", 1)
            if parts[0] in self._get_valid_regions():
                region = parts[0]
                uri = f"s3://{parts[1]}"

        # Parse standard S3 URI
        parsed = urlparse(uri)
        if parsed.scheme != "s3":
            raise ValueError(f"Not an S3 URI: {uri}")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        if not bucket:
            raise ValueError(f"Missing bucket in S3 URI: {uri}")

        return bucket, key, region or self.default_region

    def _get_s3_client(self, region: Optional[str] = None) -> Any:
        """
        Get or create an S3 client for the given region.

        Args:
            region: AWS region

        Returns:
            Boto3 S3 client
        """
        # Use default region if none specified
        region = region or self.default_region or self.credentials.get("region_name")

        # Check if we already have a client for this region
        if region in self.clients:
            return self.clients[region]

        # Create client configuration
        client_kwargs = {}
        if self.use_credentials:
            if self.credentials.get("aws_access_key_id"):
                client_kwargs["aws_access_key_id"] = self.credentials["aws_access_key_id"]
            if self.credentials.get("aws_secret_access_key"):
                client_kwargs["aws_secret_access_key"] = self.credentials["aws_secret_access_key"]
            if self.credentials.get("aws_session_token"):
                client_kwargs["aws_session_token"] = self.credentials["aws_session_token"]

        if region:
            client_kwargs["region_name"] = region

        # Create a new client
        try:
            s3_client = boto3.client('s3', **client_kwargs)

            # Cache the client
            self.clients[region or "default"] = s3_client

            return s3_client
        except Exception as e:
            logger.error(f"Error creating S3 client: {str(e)}")
            raise ValueError(f"Error creating S3 client: {str(e)}")

    @staticmethod
    def _get_valid_regions() -> list:
        """Get list of valid AWS regions."""
        return [
            "us-east-1", "us-east-2", "us-west-1", "us-west-2",
            "ca-central-1", "eu-west-1", "eu-west-2", "eu-west-3",
            "eu-central-1", "eu-north-1", "ap-northeast-1",
            "ap-northeast-2", "ap-northeast-3", "ap-southeast-1",
            "ap-southeast-2", "ap-south-1", "sa-east-1",
            "us-gov-east-1", "us-gov-west-1", "cn-north-1", "cn-northwest-1"
        ]

    @staticmethod
    def _is_text_content(content_type: str) -> bool:
        """
        Determine if a content type represents text-based content.

        Args:
            content_type: MIME content type

        Returns:
            True if text-based, False otherwise
        """
        # Text MIME types
        text_types = [
            'text/',
            'application/json',
            'application/xml',
            'application/yaml',
            'application/x-yaml',
            'application/javascript',
            'application/typescript',
            'application/csv',
            'application/x-csv',
            'application/markdown',
            'application/x-markdown'
        ]

        # Check if content type starts with any of the text types
        for text_type in text_types:
            if content_type.startswith(text_type):
                return True

        return False
