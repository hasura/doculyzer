"""
Web adapter module for the document pointer system.

This module provides an adapter to retrieve content from web sources.
"""

import logging
import mimetypes
import os
import re
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .base import ContentSourceAdapter
from ..document_parser.document_type_detector import DocumentTypeDetector

logger = logging.getLogger(__name__)


class WebAdapter(ContentSourceAdapter):
    """Adapter for web-based content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the web adapter."""
        super().__init__(config)
        self.config = config or {}

        # Initialize session with configured headers and auth
        self.session = requests.Session()
        self.headers = self.config.get("headers", {
            "User-Agent": "Doculyzer/1.0 Web Content Adapter"
        })
        self.session.headers.update(self.headers)

        # Configure authentication if provided
        auth_config = self.config.get("authentication", {})
        auth_type = auth_config.get("type")

        if auth_type == "basic":
            self.session.auth = (
                auth_config.get("username", ""),
                auth_config.get("password", "")
            )
        elif auth_type == "bearer":
            self.session.headers.update({
                "Authorization": f"Bearer {auth_config.get('token', '')}"
            })

        # Configure request parameters
        self.timeout = self.config.get("timeout", 30)
        self.max_redirects = self.config.get("max_redirects", 5)
        self.verify_ssl = self.config.get("verify_ssl", True)
        self.follow_links = self.config.get("follow_links", False)
        self.max_link_depth = self.config.get("max_link_depth", 1)

        # Content processing options
        self.download_assets = self.config.get("download_assets", False)
        self.temp_dir = self.config.get("temp_dir", tempfile.gettempdir())
        self.encoding_fallbacks = self.config.get(
            "encoding_fallbacks", ['utf-8', 'latin-1', 'cp1252']
        )

        # Content cache
        self.content_cache = {}
        self.binary_cache = {}
        self.metadata_cache = {}

        # Initialize mimetypes
        mimetypes.init()

    def get_content(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get content from a web source.

        Args:
            location_data: Location data with URL

        Returns:
            Dictionary with content and metadata

        Raises:
            ValueError: If URL cannot be accessed
        """
        source = location_data.get("source", "")

        # Validate URL
        if not source.startswith(("http://", "https://")):
            raise ValueError(f"Invalid web source: {source}")

        # Check cache first
        if source in self.content_cache:
            return self.content_cache[source]

        # Extract request parameters from location data
        params = location_data.get("params", {})
        headers = {**self.headers, **location_data.get("headers", {})}
        auth = None

        # Extract custom auth if provided
        auth_data = location_data.get("auth", {})
        if auth_data:
            auth_type = auth_data.get("type")
            if auth_type == "basic":
                auth = (auth_data.get("username", ""), auth_data.get("password", ""))

        try:
            # Fetch content
            response = self.session.get(
                source,
                params=params,
                headers=headers,
                auth=auth,
                timeout=self.timeout,
                allow_redirects=True,
                verify=self.verify_ssl
            )

            # Raise for HTTP errors
            response.raise_for_status()

            # Get content type from response headers
            content_type = response.headers.get("Content-Type", "").split(';')[0].strip()

            # Determine if content is binary based on content type
            is_binary = not self._is_text_content(content_type)

            # Get content
            if is_binary:
                content = response.content
            else:
                # Try to decode using response encoding or fallbacks
                content = response.text

            # Determine document type
            doc_type = DocumentTypeDetector.detect(
                path=source,
                content=content,
                metadata={
                    "content_type": content_type,
                    "binary": is_binary
                }
            )

            # Extract metadata
            metadata = self._extract_metadata(response, source, is_binary)

            # Add additional metadata for HTML content
            if doc_type == "html" and not is_binary:
                html_metadata = self._extract_html_metadata(content, source)
                metadata.update(html_metadata)

            # Create result
            result = {
                "content": content,
                "content_type": doc_type,
                "metadata": metadata
            }

            # Cache the result
            self.content_cache[source] = result
            self.metadata_cache[source] = metadata

            return result

        except requests.RequestException as e:
            logger.error(f"Error fetching URL {source}: {str(e)}")
            raise ValueError(f"Error fetching URL: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error processing URL {source}: {str(e)}")
            raise ValueError(f"Error processing URL: {str(e)}")

    def supports_location(self, location_data: Dict[str, Any]) -> bool:
        """
        Check if this adapter supports the location.

        Args:
            location_data: Content location data

        Returns:
            True if supported, False otherwise
        """
        source = location_data.get("source", "")
        # Check if source is a valid HTTP/HTTPS URL
        return source.startswith(("http://", "https://"))

    def get_binary_content(self, location_data: Dict[str, Any]) -> bytes:
        """
        Get the web content as binary data.

        Args:
            location_data: Location data

        Returns:
            Binary content

        Raises:
            ValueError: If content cannot be retrieved as binary
        """
        source = location_data.get("source", "")

        # Validate URL
        if not source.startswith(("http://", "https://")):
            raise ValueError(f"Invalid web source: {source}")

        # Check cache first
        if source in self.binary_cache:
            return self.binary_cache[source]

        try:
            # Fetch content with streaming enabled
            response = self.session.get(
                source,
                timeout=self.timeout,
                allow_redirects=True,
                verify=self.verify_ssl,
                stream=True
            )

            # Raise for HTTP errors
            response.raise_for_status()

            # Get binary content
            binary_content = response.content

            # Cache the result
            self.binary_cache[source] = binary_content

            return binary_content

        except requests.RequestException as e:
            logger.error(f"Error fetching binary content from {source}: {str(e)}")
            raise ValueError(f"Error fetching binary content: {str(e)}")

    def get_metadata(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get metadata about the web content without retrieving the full content.

        Args:
            location_data: Location data

        Returns:
            Dictionary with metadata

        Raises:
            ValueError: If metadata cannot be retrieved
        """
        source = location_data.get("source", "")

        # Validate URL
        if not source.startswith(("http://", "https://")):
            raise ValueError(f"Invalid web source: {source}")

        # Check cache first
        if source in self.metadata_cache:
            return self.metadata_cache[source]

        try:
            # Fetch headers only
            response = self.session.head(
                source,
                timeout=self.timeout,
                allow_redirects=True,
                verify=self.verify_ssl
            )

            # Raise for HTTP errors
            response.raise_for_status()

            # Get content type
            content_type = response.headers.get("Content-Type", "").split(';')[0].strip()

            # Determine if content is binary based on content type
            is_binary = not self._is_text_content(content_type)

            # Extract metadata
            metadata = self._extract_metadata(response, source, is_binary)

            # Cache the metadata
            self.metadata_cache[source] = metadata

            return metadata

        except requests.RequestException as e:
            logger.error(f"Error fetching metadata from {source}: {str(e)}")
            raise ValueError(f"Error fetching metadata: {str(e)}")

    def resolve_uri(self, uri: str) -> Dict[str, Any]:
        """
        Parse a web URI into location data.

        Args:
            uri: Web URI string (http:// or https://)

        Returns:
            Dictionary with parsed location data

        Raises:
            ValueError: If URI cannot be parsed
        """
        if not uri.startswith(("http://", "https://")):
            raise ValueError(f"Not a web URI: {uri}")

        # Parse URL parts
        parsed = urlparse(uri)

        # Build location data
        location_data = {
            "source": uri,
            "scheme": parsed.scheme,
            "netloc": parsed.netloc,
            "path": parsed.path,
            "params": parsed.params,
            "query": parsed.query,
            "fragment": parsed.fragment
        }

        return location_data

    def cleanup(self):
        """
        Clean up resources used by this adapter.

        This method should be called when the adapter is no longer needed.
        """
        # Close session
        if hasattr(self, 'session') and self.session:
            try:
                self.session.close()
                logger.debug("Closed web adapter session")
            except Exception as e:
                logger.warning(f"Error closing web adapter session: {str(e)}")

        # Clear caches
        self.content_cache = {}
        self.binary_cache = {}
        self.metadata_cache = {}

    @staticmethod
    def _extract_metadata(response: requests.Response, url: str, is_binary: bool) -> Dict[str, Any]:
        """
        Extract metadata from response.

        Args:
            response: HTTP response
            url: Source URL
            is_binary: Whether content is binary

        Returns:
            Dictionary with metadata
        """
        # Parse URL
        parsed_url = urlparse(url)

        # Extract headers
        last_modified = None
        if "Last-Modified" in response.headers:
            try:
                last_modified = datetime.strptime(
                    response.headers["Last-Modified"],
                    "%a, %d %b %Y %H:%M:%S %Z"
                )
            except (ValueError, TypeError):
                pass

        # Build metadata
        metadata = {
            "url": url,
            "domain": parsed_url.netloc,
            "path": parsed_url.path,
            "status_code": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "content_length": int(response.headers.get("Content-Length", 0)),
            "last_modified": last_modified,
            "etag": response.headers.get("ETag"),
            "is_binary": is_binary,
            "encoding": response.encoding,
            "final_url": response.url,  # In case of redirects
            "headers": dict(response.headers)
        }

        # Extract filename from URL path or Content-Disposition header
        filename = os.path.basename(parsed_url.path)
        content_disposition = response.headers.get("Content-Disposition", "")
        if content_disposition:
            filename_match = re.search(r'filename="?([^";]+)"?', content_disposition)
            if filename_match:
                filename = filename_match.group(1)

        if filename:
            metadata["filename"] = filename

        return metadata

    def _extract_html_metadata(self, content: str, url: str) -> Dict[str, Any]:
        """
        Extract metadata from HTML content.

        Args:
            content: HTML content
            url: Source URL

        Returns:
            Dictionary with HTML metadata
        """
        metadata = {}

        try:
            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')

            # Extract title
            title = soup.title.string if soup.title else None
            if title:
                metadata["title"] = title.strip()

            # Extract meta tags
            meta_tags = {}
            for meta in soup.find_all('meta'):
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                if name and content:
                    meta_tags[name] = content

            if meta_tags:
                metadata["meta_tags"] = meta_tags

            # Extract links
            if self.follow_links:
                links = [a['href'] for a in soup.find_all('a', href=True)]
                metadata["links"] = links

            # Count elements
            metadata["element_counts"] = {
                "headings": len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
                "paragraphs": len(soup.find_all('p')),
                "tables": len(soup.find_all('table')),
                "images": len(soup.find_all('img')),
                "links": len(soup.find_all('a', href=True))
            }

        except Exception as e:
            logger.warning(f"Error extracting HTML metadata from {url}: {str(e)}")

        return metadata

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
            'application/x-markdown',
            'application/rss+xml',
            'application/atom+xml',
            'application/xhtml+xml'
        ]

        # Check if content type starts with any of the text types
        for text_type in text_types:
            if content_type.startswith(text_type):
                return True

        return False
