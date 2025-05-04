"""
S3 Content Source for the document pointer system.

This module provides integration with Amazon S3 to ingest documents stored in buckets.
"""

import logging
import os
import re
import tempfile
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

import time

from .base import ContentSource
from .utils import detect_content_type
from ..document_parser.factory import get_parser_for_content

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
    logger.warning("boto3 not available. Install with 'pip install boto3' to use S3 content source.")


class S3ContentSource(ContentSource):
    """Content source for Amazon S3."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the S3 content source.

        Args:
            config: Configuration dictionary containing S3 connection details
        """
        super().__init__(config)

        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 content source")

        # S3 connection details
        self.bucket_name = config.get("bucket_name", "")
        self.prefix = config.get("prefix", "")
        self.region_name = config.get("region_name", None)
        self.aws_access_key_id = config.get("aws_access_key_id", None)
        self.aws_secret_access_key = config.get("aws_secret_access_key", None)
        self.aws_session_token = config.get("aws_session_token", None)
        self.assume_role_arn = config.get("assume_role_arn", None)
        self.endpoint_url = config.get("endpoint_url", None)  # For S3-compatible storage

        # Content filtering
        self.include_extensions = config.get("include_extensions", [])
        self.exclude_extensions = config.get("exclude_extensions", [])
        self.include_prefixes = config.get("include_prefixes", [])
        self.exclude_prefixes = config.get("exclude_prefixes", [])
        self.include_patterns = config.get("include_patterns", [])
        self.exclude_patterns = config.get("exclude_patterns", [])
        self.recursive = config.get("recursive", True)
        self.max_depth = config.get("max_depth", 10)
        self.follow_symlinks = config.get("follow_symlinks", False)

        # Content processing
        self.detect_mimetype = config.get("detect_mimetype", True)
        self.temp_dir = config.get("temp_dir", tempfile.gettempdir())
        self.delete_after_processing = config.get("delete_after_processing", True)

        # Link following configuration
        self.local_link_mode = config.get("local_link_mode", "relative")  # relative, absolute, or none

        # Initialize S3 client
        self.s3_client = self._initialize_s3_client()

        # Cache for content
        self.content_cache = {}

    def fetch_document(self, source_id: str) -> Dict[str, Any]:
        """
        Fetch document content from S3.

        Args:
            source_id: Identifier for the S3 object (s3://bucket/key or just key)

        Returns:
            Dictionary containing document content and metadata
        """
        logger.debug(f"Fetching S3 object: {source_id}")

        try:
            # Extract bucket and key if source_id is a fully qualified S3 URI
            bucket, key = self._extract_bucket_and_key(source_id)

            # If no bucket specified, use configured bucket
            if not bucket:
                bucket = self.bucket_name
                key = source_id

            # Normalize key to remove leading slash
            key = key.lstrip('/')

            # Create fully qualified source identifier
            qualified_source = f"s3://{bucket}/{key}"

            # Check cache first
            cache_key = f"{bucket}/{key}"
            if cache_key in self.content_cache:
                cache_entry = self.content_cache[cache_key]
                logger.debug(f"Using cached content for: {qualified_source}")
                return cache_entry

            # Get object metadata first
            try:
                response = self.s3_client.head_object(Bucket=bucket, Key=key)
                metadata = response.get('Metadata', {})
                content_type = response.get('ContentType', '')
                last_modified = response.get('LastModified', None)
                size = response.get('ContentLength', 0)
                etag = response.get('ETag', '').strip('"')
            except ClientError as e:
                logger.error(f"Error retrieving metadata for {qualified_source}: {str(e)}")
                raise

            # Get object content
            try:
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                content = response['Body'].read()

                # Convert binary content to string if possible
                try:
                    content = content.decode('utf-8')
                    is_binary = False
                except UnicodeDecodeError:
                    # Keep binary content as is
                    is_binary = True
            except ClientError as e:
                logger.error(f"Error downloading {qualified_source}: {str(e)}")
                raise

            # Save to temp file if binary
            temp_file_path = None
            if is_binary:
                # Create temp file for binary content
                file_name = os.path.basename(key)
                temp_file_path = os.path.join(self.temp_dir, f"s3_{etag}_{file_name}")

                with open(temp_file_path, 'wb') as f:
                    f.write(content)

                logger.debug(f"Saved binary content to temp file: {temp_file_path}")

            # Detect document type if not explicitly provided
            # doc_type = None

            if self.detect_mimetype:
                if is_binary:
                    # For binary content, use extension to guess type
                    if key.lower().endswith(('.md', '.markdown')):
                        doc_type = "markdown"
                    elif key.lower().endswith(('.html', '.htm')):
                        doc_type = "html"
                    elif key.lower().endswith('.pdf'):
                        doc_type = "pdf"
                    elif key.lower().endswith('.docx'):
                        doc_type = "docx"
                    elif key.lower().endswith('.pptx'):
                        doc_type = "pptx"
                    elif key.lower().endswith('.xlsx'):
                        doc_type = "xlsx"
                    else:
                        doc_type = "binary"
                else:
                    # For text content, detect type from content
                    doc_type = detect_content_type(content, {"content_type": content_type})
            else:
                # Use content_type from metadata
                if 'markdown' in content_type.lower() or 'md' in content_type.lower():
                    doc_type = "markdown"
                elif 'html' in content_type.lower():
                    doc_type = "html"
                elif 'text' in content_type.lower():
                    doc_type = "text"
                else:
                    doc_type = "binary"

            # Create metadata for document
            document_metadata = {
                "bucket": bucket,
                "key": key,
                "content_type": content_type,
                "last_modified": last_modified.timestamp() if last_modified else time.time(),
                "size": size,
                "etag": etag,
                "s3_metadata": metadata,
                "is_binary": is_binary,
                "temp_file_path": temp_file_path,
                "filename": os.path.basename(key),
                "extension": os.path.splitext(key)[1].lower()[1:] if '.' in os.path.basename(key) else '',
                "url": f"s3://{bucket}/{key}"
            }

            # Generate content hash for change detection
            content_hash = self.get_content_hash(str(content) if not is_binary else etag)

            # Create result
            result = {
                "id": qualified_source,
                "content": content if not is_binary else "",
                "binary_path": temp_file_path,
                "doc_type": doc_type,
                "metadata": document_metadata,
                "content_hash": content_hash
            }

            # Cache the content for faster access
            self.content_cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Error fetching S3 object {source_id}: {str(e)}")
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List available documents in S3.

        Returns:
            List of document identifiers and metadata
        """
        logger.debug(f"Listing S3 objects in bucket: {self.bucket_name}, prefix: {self.prefix}")

        results = []
        try:
            # Set up paginator for listing objects
            paginator = self.s3_client.get_paginator('list_objects_v2')

            # Configure paginator
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=self.prefix,
                PaginationConfig={'MaxItems': 1000}
            )

            # Process each page
            for page in page_iterator:
                # Process each object
                for obj in page.get('Contents', []):
                    key = obj.get('Key', '')

                    # Skip if it's a directory-like object
                    if key.endswith('/'):
                        continue

                    # Apply filters
                    if not self._should_include_object(key):
                        logger.debug(f"Skipping excluded object: {key}")
                        continue

                    # Get metadata
                    size = obj.get('Size', 0)
                    last_modified = obj.get('LastModified', None)
                    etag = obj.get('ETag', '').strip('"')

                    # Create fully qualified source identifier
                    qualified_source = f"s3://{self.bucket_name}/{key}"

                    # Extract file extension
                    extension = os.path.splitext(key)[1].lower()[1:] if '.' in os.path.basename(key) else ''

                    # Create metadata for document
                    metadata = {
                        "bucket": self.bucket_name,
                        "key": key,
                        "last_modified": last_modified.timestamp() if last_modified else None,
                        "size": size,
                        "etag": etag,
                        "filename": os.path.basename(key),
                        "extension": extension,
                        "url": qualified_source
                    }

                    # Guess document type based on extension
                    doc_type = None
                    if extension in ['md', 'markdown']:
                        doc_type = "markdown"
                    elif extension in ['html', 'htm']:
                        doc_type = "html"
                    elif extension in ['txt']:
                        doc_type = "text"

                    results.append({
                        "id": qualified_source,
                        "metadata": metadata,
                        "doc_type": doc_type
                    })

            logger.info(f"Found {len(results)} S3 objects")
            return results

        except Exception as e:
            logger.error(f"Error listing S3 objects: {str(e)}")
            raise

    def has_changed(self, source_id: str, last_modified: Optional[float] = None) -> bool:
        """
        Check if an S3 object has changed since last processing.

        Args:
            source_id: Identifier for the S3 object
            last_modified: Timestamp of last known modification

        Returns:
            True if object has changed, False otherwise
        """
        logger.debug(f"Checking if S3 object has changed: {source_id}")

        try:
            # Extract bucket and key
            bucket, key = self._extract_bucket_and_key(source_id)

            # If no bucket specified, use configured bucket
            if not bucket:
                bucket = self.bucket_name
                key = source_id

            # Normalize key to remove leading slash
            key = key.lstrip('/')

            # Check cache first
            cache_key = f"{bucket}/{key}"
            if cache_key in self.content_cache:
                cache_entry = self.content_cache[cache_key]
                cache_modified = cache_entry["metadata"].get("last_modified")

                if cache_modified and last_modified and cache_modified <= last_modified:
                    logger.debug(f"Object {source_id} unchanged according to cache")
                    return False

            # Make API request to check object metadata
            try:
                response = self.s3_client.head_object(Bucket=bucket, Key=key)
                current_modified = response.get('LastModified')
                # current_etag = response.get('ETag', '').strip('"')

                # Convert to timestamp for comparison
                if current_modified:
                    current_timestamp = current_modified.timestamp()

                    if last_modified:
                        changed = current_timestamp > last_modified
                        logger.debug(f"Object {source_id} changed: {changed}")
                        return changed
            except ClientError as e:
                logger.error(f"Error checking metadata for {source_id}: {str(e)}")
                return True

            # If we can't determine based on timestamp, consider it changed
            return True

        except Exception as e:
            logger.error(f"Error checking changes for {source_id}: {str(e)}")
            return True

    def follow_links(self, content: str, source_id: str, current_depth: int = 0, global_visited_docs=None) -> List[
        Dict[str, Any]]:
        """
        Extract and follow links in S3 content.

        Args:
            content: Document content
            source_id: Identifier for the source document
            current_depth: Current depth of link following
            global_visited_docs: Global set of all visited document IDs

        Returns:
            List of linked documents
        """
        if current_depth >= self.max_link_depth:
            logger.debug(f"Max link depth {self.max_link_depth} reached for {source_id}")
            return []

        # Initialize global visited set if not provided
        if global_visited_docs is None:
            global_visited_docs = set()

        # Add current document to global visited set
        global_visited_docs.add(source_id)

        # If local link mode is none, skip link following
        if self.local_link_mode == "none":
            return []

        logger.debug(f"Following links in S3 object {source_id} at depth {current_depth}")

        linked_docs = []

        # Extract bucket and key from source_id
        source_bucket, source_key = self._extract_bucket_and_key(source_id)

        # If no bucket specified, use configured bucket
        if not source_bucket:
            source_bucket = self.bucket_name
            source_key = source_id

        # Skip if content is empty or binary
        if not content:
            return []

        # Fetch document metadata
        try:
            metadata = None

            # Try cache first
            cache_key = f"{source_bucket}/{source_key}"
            if cache_key in self.content_cache:
                metadata = self.content_cache[cache_key]["metadata"]

            if not metadata:
                # Get metadata from S3
                response = self.s3_client.head_object(Bucket=source_bucket, Key=source_key)
                content_type = response.get('ContentType', '')
            else:
                content_type = metadata.get("content_type", "")

            # Create parser for the document to extract links correctly
            doc_content = {
                "content": content,
                "metadata": {"content_type": content_type}
            }

            parser = get_parser_for_content(doc_content)

            # Extract links based on document type
            links = parser._extract_links(content, "root")

            # Process each link
            for link in links:
                link_target = link.get("link_target", "")

                # Skip empty links
                if not link_target:
                    continue

                # Handle different link types

                # Skip external links (http/https)
                if link_target.startswith(('http://', 'https://')):
                    continue

                # Absolute S3 link
                if link_target.startswith('s3://'):
                    target_bucket, target_key = self._extract_bucket_and_key(link_target)
                    qualified_target = link_target

                # Relative link
                else:
                    # Resolve based on local link mode
                    target_bucket = source_bucket

                    if self.local_link_mode == "relative":
                        # Resolve relative to source key's directory
                        source_dir = os.path.dirname(source_key)
                        target_key = os.path.normpath(os.path.join(source_dir, link_target))
                    else:  # absolute
                        # Link is relative to bucket root
                        target_key = link_target.lstrip('/')

                    qualified_target = f"s3://{target_bucket}/{target_key}"

                # Skip if globally visited
                if qualified_target in global_visited_docs:
                    logger.debug(f"Skipping globally visited link: {qualified_target}")
                    continue

                # Mark as visited
                global_visited_docs.add(qualified_target)

                # Check if the object exists
                try:
                    self.s3_client.head_object(Bucket=target_bucket, Key=target_key)
                except ClientError:
                    logger.debug(f"Linked object not found: {qualified_target}")
                    continue

                try:
                    # Fetch the linked document
                    linked_doc = self.fetch_document(qualified_target)
                    linked_docs.append(linked_doc)
                    logger.debug(f"Successfully fetched linked document: {qualified_target}")

                    # If this is a non-binary document, recursively follow links
                    if current_depth + 1 < self.max_link_depth and not linked_doc["metadata"].get("is_binary", False):
                        logger.debug(
                            f"Recursively following links from {qualified_target} at depth {current_depth + 1}")
                        nested_docs = self.follow_links(
                            linked_doc.get("content", ""),
                            qualified_target,
                            current_depth + 1,
                            global_visited_docs
                        )
                        linked_docs.extend(nested_docs)
                except Exception as e:
                    logger.warning(f"Error following link {qualified_target} from {source_id}: {str(e)}")

            logger.debug(f"Completed following links from {source_id}: found {len(linked_docs)} linked documents")
            return linked_docs

        except Exception as e:
            logger.error(f"Error following links from S3 object {source_id}: {str(e)}")
            return []

    def _initialize_s3_client(self):
        """
        Initialize S3 client with configured credentials.

        Returns:
            boto3 S3 client
        """
        try:
            # First, check if we need to assume a role
            if self.assume_role_arn:
                sts_client = boto3.client(
                    'sts',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    aws_session_token=self.aws_session_token,
                    region_name=self.region_name
                )

                # Assume role
                response = sts_client.assume_role(
                    RoleArn=self.assume_role_arn,
                    RoleSessionName='DocumentPointerSession'
                )

                # Extract temporary credentials
                credentials = response['Credentials']

                # Create S3 client with temporary credentials
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=credentials['AccessKeyId'],
                    aws_secret_access_key=credentials['SecretAccessKey'],
                    aws_session_token=credentials['SessionToken'],
                    region_name=self.region_name,
                    endpoint_url=self.endpoint_url
                )
            else:
                # Create S3 client with provided credentials
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    aws_session_token=self.aws_session_token,
                    region_name=self.region_name,
                    endpoint_url=self.endpoint_url
                )

            # Test connection by listing buckets
            s3_client.list_buckets()

            return s3_client

        except Exception as e:
            logger.error(f"Error initializing S3 client: {str(e)}")
            raise

    @staticmethod
    def _extract_bucket_and_key(s3_uri: str) -> tuple:
        """
        Extract bucket and key from S3 URI.

        Args:
            s3_uri: S3 URI (s3://bucket/key) or just key

        Returns:
            Tuple of (bucket, key)
        """
        # Check if it's a full S3 URI
        if s3_uri.startswith('s3://'):
            parsed = urlparse(s3_uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            return bucket, key
        else:
            # Not a full URI, treat as key only
            return None, s3_uri

    def _should_include_object(self, key: str) -> bool:
        """
        Check if S3 object should be included based on configured filters.

        Args:
            key: S3 object key

        Returns:
            True if object should be included, False otherwise
        """
        # Get file extension
        extension = os.path.splitext(key)[1].lower()[1:] if '.' in os.path.basename(key) else ''

        # Check exclude extensions
        if self.exclude_extensions and extension in self.exclude_extensions:
            return False

        # Check include extensions
        if self.include_extensions and extension not in self.include_extensions:
            return False

        # Check exclude prefixes
        for prefix in self.exclude_prefixes:
            if key.startswith(prefix):
                return False

        # Check include prefixes
        if self.include_prefixes:
            if not any(key.startswith(prefix) for prefix in self.include_prefixes):
                return False

        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if re.search(pattern, key):
                return False

        # Check include patterns
        if self.include_patterns:
            if not any(re.search(pattern, key) for pattern in self.include_patterns):
                return False

        return True

    def __del__(self):
        """Cleanup temporary files on deletion."""
        # Clean up any temp files
        for cache_key, cache_entry in self.content_cache.items():
            if self.delete_after_processing:
                temp_file_path = cache_entry.get("binary_path")
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        logger.debug(f"Deleted temporary file: {temp_file_path}")
                    except Exception as e:
                        logger.warning(f"Error deleting temporary file {temp_file_path}: {str(e)}")
