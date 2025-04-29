"""
S3 Content Resolver implementation for the document pointer system.

This module resolves S3 content pointers to actual content.
"""
import json
import logging
import os
import tempfile
from typing import Any, Optional
from urllib.parse import urlparse

from .base import ContentResolver

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
    logger.warning("boto3 not available. Install with 'pip install boto3' to use S3 resolver")


class S3ContentResolver(ContentResolver):
    """Resolver for Amazon S3 content."""

    def __init__(self):
        """Initialize the S3 content resolver."""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 content resolver")

        self.clients = {}  # Cache for S3 clients
        self.content_cache = {}  # Cache for retrieved content
        self.temp_dir = tempfile.gettempdir()

        # Try to load credentials from environment
        self.default_credentials = {
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "aws_session_token": os.environ.get("AWS_SESSION_TOKEN"),
            "region_name": os.environ.get("AWS_REGION")
        }

    def resolve_content(self, content_location: str) -> str:
        """
        Resolve S3 content pointer to actual content.

        Args:
            content_location: JSON-formatted content location pointer

        Returns:
            Resolved content as string
        """
        location_data = json.loads(content_location)

        source = location_data.get("source", "")
        if not source.startswith("s3://"):
            raise ValueError(f"Invalid S3 source: {source}")

        # Extract bucket and key from S3 URI
        parsed = urlparse(source)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')

        if not bucket or not key:
            raise ValueError(f"Invalid S3 URI format: {source}")

        # Extract region from additional metadata if available
        region = location_data.get("region")

        # Get S3 client
        s3_client = self._get_s3_client(region)

        # Determine what part of the content to return based on element type
        element_type = location_data.get("type", "")

        # Check cache first
        cache_key = f"{bucket}:{key}:{element_type}"
        if cache_key in self.content_cache:
            logger.debug(f"Using cached content for: {cache_key}")
            return self.content_cache[cache_key]

        try:
            # Get object information first to determine content type
            try:
                head_response = s3_client.head_object(Bucket=bucket, Key=key)
                content_type = head_response.get('ContentType', '')
                # size = head_response.get('ContentLength', 0)
            except ClientError as e:
                logger.error(f"Error retrieving object info for {source}: {str(e)}")
                raise

            # Check if the content is binary or text-based
            is_text = self._is_text_content(content_type)

            # Get the content
            try:
                response = s3_client.get_object(Bucket=bucket, Key=key)
                content = response['Body'].read()

                # For text content, decode to string
                if is_text:
                    try:
                        content = content.decode('utf-8')
                    except UnicodeDecodeError:
                        # Try another common encoding
                        try:
                            content = content.decode('latin-1')
                        except UnicodeDecodeError:
                            # If still can't decode, treat as binary
                            is_text = False

                # For binary content, save to temp file and return file path
                if not is_text:
                    # We can't return the actual binary content, so we'll save it to a temp file
                    file_name = os.path.basename(key)
                    temp_file_path = os.path.join(self.temp_dir, f"s3_{bucket}_{file_name}")

                    with open(temp_file_path, 'wb') as f:
                        if isinstance(content, str):
                            f.write(content.encode('utf-8'))
                        else:
                            f.write(content)

                    # Return a message with the temp file path
                    return f"Binary content saved to temporary file: {temp_file_path}"

            except ClientError as e:
                logger.error(f"Error downloading object {source}: {str(e)}")
                raise

            # Extract specific content based on element type if needed
            resolved_content = content

            if element_type == "text_range":
                # Extract a specific range of text
                start = location_data.get("start", 0)
                end = location_data.get("end")

                if end is not None:
                    resolved_content = content[start:end]
                else:
                    resolved_content = content[start:]

            elif element_type == "line_range":
                # Extract specific lines
                start_line = location_data.get("start_line", 0)
                end_line = location_data.get("end_line")

                lines = content.splitlines()

                if end_line is not None:
                    resolved_content = "\n".join(lines[start_line:end_line])
                else:
                    resolved_content = "\n".join(lines[start_line:])

            elif element_type == "json_path":
                # Extract content using JSON path
                try:
                    import jsonpath_ng.ext as jsonpath

                    path = location_data.get("path", "$")
                    json_data = json.loads(content)

                    # Parse and find using JSONPath
                    jsonpath_expr = jsonpath.parse(path)
                    matches = [match.value for match in jsonpath_expr.find(json_data)]

                    if matches:
                        resolved_content = json.dumps(matches, indent=2)
                    else:
                        resolved_content = "No matches found for JSONPath"

                except (ImportError, json.JSONDecodeError) as e:
                    logger.error(f"Error processing JSONPath: {str(e)}")
                    resolved_content = f"Error processing JSONPath: {str(e)}"

            elif element_type == "xpath":
                # Extract content using XPath (for XML content)
                try:
                    # noinspection PyPackageRequirements
                    from lxml import etree

                    path = location_data.get("path", "//")

                    # Parse XML
                    xml_root = etree.fromstring(content.encode('utf-8') if isinstance(content, str) else content)

                    # Find elements using XPath
                    matches = xml_root.xpath(path)

                    if matches:
                        # Convert matches to strings
                        result = []
                        for match in matches:
                            if isinstance(match, etree._Element):
                                result.append(etree.tostring(match, encoding='utf-8').decode('utf-8'))
                            else:
                                result.append(str(match))

                        resolved_content = "\n".join(result)
                    else:
                        resolved_content = "No matches found for XPath"

                except (ImportError, etree.XMLSyntaxError) as e:
                    logger.error(f"Error processing XPath: {str(e)}")
                    resolved_content = f"Error processing XPath: {str(e)}"

            elif element_type == "css_selector":
                # Extract content using CSS selector (for HTML content)
                try:
                    from bs4 import BeautifulSoup

                    selector = location_data.get("selector", "")

                    # Parse HTML
                    soup = BeautifulSoup(content, 'html.parser')

                    # Find elements using CSS selector
                    matches = soup.select(selector)

                    if matches:
                        # Convert matches to strings
                        result = []
                        for match in matches:
                            result.append(str(match))

                        resolved_content = "\n".join(result)
                    else:
                        resolved_content = f"No matches found for CSS selector: {selector}"

                except (ImportError, Exception) as e:
                    logger.error(f"Error processing CSS selector: {str(e)}")
                    resolved_content = f"Error processing CSS selector: {str(e)}"

            # Cache the result
            self.content_cache[cache_key] = resolved_content

            return resolved_content

        except Exception as e:
            logger.error(f"Error resolving S3 content: {str(e)}")
            raise

    def supports_location(self, content_location: str) -> bool:
        """
        Check if this resolver supports the location.

        Args:
            content_location: Content location pointer

        Returns:
            True if supported, False otherwise
        """
        try:
            location_data = json.loads(content_location)
            source = location_data.get("source", "")
            # Source must be an S3 URI
            return source.startswith("s3://")
        except (json.JSONDecodeError, TypeError):
            return False

    def get_document_binary(self, content_location: str) -> bytes:
        """
        Get the containing document as a binary blob.

        Args:
            content_location: Content location pointer

        Returns:
            Document binary content

        Raises:
            ValueError: If document cannot be retrieved
        """
        location_data = json.loads(content_location)
        source = location_data.get("source", "")

        # Parse S3 URI
        parsed = urlparse(source)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')

        if not bucket or not key:
            raise ValueError(f"Invalid S3 URI format: {source}")

        # Extract region from additional metadata if available
        region = location_data.get("region")

        # Get S3 client
        s3_client = self._get_s3_client(region)

        try:
            # Get the object
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"Error retrieving binary content from {source}: {str(e)}")
            raise ValueError(f"Error retrieving binary content: {str(e)}")

    def _get_s3_client(self, region: Optional[str] = None) -> Any:
        """
        Get or create an S3 client for the given region.

        Args:
            region: AWS region

        Returns:
            Boto3 S3 client
        """
        # Use default region if none specified
        if not region:
            region = self.default_credentials.get("region_name")

        # Check if we already have a client for this region
        if region in self.clients:
            return self.clients[region]

        # Create a new client
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.default_credentials.get("aws_access_key_id"),
                aws_secret_access_key=self.default_credentials.get("aws_secret_access_key"),
                aws_session_token=self.default_credentials.get("aws_session_token"),
                region_name=region
            )

            # Cache the client
            self.clients[region] = s3_client

            return s3_client
        except Exception as e:
            logger.error(f"Error creating S3 client: {str(e)}")
            raise

    @staticmethod
    def _is_text_content(content_type: str) -> bool:
        """
        Determine if the content type represents text-based content.

        Args:
            content_type: MIME content type

        Returns:
            True if text-based, False otherwise
        """
        # List of common text-based MIME types
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

        # Some special cases to handle
        if content_type in [
            'application/octet-stream',  # This can be anything, but worth trying as text
            'binary/octet-stream'
        ]:
            # For ambiguous types, we'll try to parse as text
            return True

        return False
