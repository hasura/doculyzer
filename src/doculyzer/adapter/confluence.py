"""
Confluence adapter module for the document pointer system.

This module provides an adapter to retrieve content from Atlassian Confluence sources.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from .base import ContentSourceAdapter
from ..document_parser.document_type_detector import DocumentTypeDetector

logger = logging.getLogger(__name__)

# Try to import requests, but don't fail if not available
try:
    import requests
    from bs4 import BeautifulSoup

    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    BeautifulSoup = None
    REQUESTS_AVAILABLE = False
    logger.warning("requests is required for Confluence adapter. Install with 'pip install requests beautifulsoup4'")


class ConfluenceAdapter(ContentSourceAdapter):
    """Adapter for Atlassian Confluence content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Confluence adapter."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests and beautifulsoup4 are required for Confluence adapter")

        super().__init__(config)
        self.config = config or {}

        # Confluence connection settings
        self.base_url = self.config.get("base_url")
        self.api_path = "/rest/api"
        self.timeout = self.config.get("timeout", 30)
        self.verify_ssl = self.config.get("verify_ssl", True)

        # Authentication settings
        self.auth_type = self.config.get("auth_type", "basic")
        self.username = self.config.get("username")
        self.password = self.config.get("password")
        self.api_token = self.config.get("api_token")

        # Optional settings
        self.expand_macros = self.config.get("expand_macros", True)
        self.include_attachments = self.config.get("include_attachments", False)
        self.include_children = self.config.get("include_children", False)
        self.max_results = self.config.get("max_results", 100)

        # Initialize session
        self.sessions = {}

        # Caches
        self.content_cache = {}
        self.binary_cache = {}
        self.metadata_cache = {}
        self.space_cache = {}
        self.page_cache = {}

    def get_content(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get content from Confluence.

        Args:
            location_data: Location data with Confluence information

        Returns:
            Dictionary with content and metadata

        Raises:
            ValueError: If Confluence content cannot be retrieved
        """
        source = location_data.get("source", "")
        if not source.startswith("confluence://"):
            raise ValueError(f"Invalid Confluence source: {source}")

        # Check cache first
        if source in self.content_cache:
            return self.content_cache[source]

        # Parse Confluence URI
        parsed_data = self._parse_confluence_uri(source)

        # Get session for this base URL
        session = self._get_session(parsed_data["base_url"])

        try:
            # Determine the type of resource we need to fetch
            content_type = parsed_data.get("content_type", "page")

            # Handle different content types
            if content_type == "page":
                return self._get_page_content(session, parsed_data, location_data)
            elif content_type == "blog":
                return self._get_blog_content(session, parsed_data, location_data)
            elif content_type == "attachment":
                return self._get_attachment_content(session, parsed_data, location_data)
            elif content_type == "space":
                return self._get_space_content(session, parsed_data, location_data)
            elif content_type == "comment":
                return self._get_comment_content(session, parsed_data, location_data)
            else:
                raise ValueError(f"Unsupported Confluence content type: {content_type}")

        except Exception as e:
            logger.error(f"Error retrieving Confluence content: {str(e)}")
            raise ValueError(f"Error retrieving Confluence content: {str(e)}")

    def supports_location(self, location_data: Dict[str, Any]) -> bool:
        """
        Check if this adapter supports the location.

        Args:
            location_data: Content location data

        Returns:
            True if supported, False otherwise
        """
        if not REQUESTS_AVAILABLE:
            return False

        source = location_data.get("source", "")
        # Source must be a Confluence URI
        return source.startswith("confluence://")

    def get_binary_content(self, location_data: Dict[str, Any]) -> bytes:
        """
        Get the Confluence content as binary data.

        Args:
            location_data: Location data

        Returns:
            Binary content

        Raises:
            ValueError: If content cannot be retrieved as binary
        """
        source = location_data.get("source", "")

        # Check if it's specifically an attachment
        parsed_data = self._parse_confluence_uri(source)

        # If it's an attachment, get the binary content directly
        if parsed_data.get("content_type") == "attachment":
            # Check cache first
            if source in self.binary_cache:
                return self.binary_cache[source]

            # Get session
            session = self._get_session(parsed_data["base_url"])

            try:
                # Construct URL
                attachment_url = f"{parsed_data['base_url']}/download/attachments/{parsed_data['content_id']}/{parsed_data['attachment_id']}"

                # Fetch attachment
                response = session.get(
                    attachment_url,
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )
                response.raise_for_status()

                # Get binary content
                binary_data = response.content

                # Cache the result
                self.binary_cache[source] = binary_data

                return binary_data

            except Exception as e:
                logger.error(f"Error retrieving attachment binary: {str(e)}")
                raise ValueError(f"Error retrieving attachment binary: {str(e)}")

        # For non-attachments, convert the content to bytes
        content_info = self.get_content(location_data)
        content = content_info.get("content", "")

        # If content is already bytes, return it
        if isinstance(content, (bytes, bytearray)):
            return content

        # Convert string content to bytes
        if isinstance(content, str):
            return content.encode('utf-8')

        # For other types, convert to JSON and then to bytes
        if isinstance(content, (dict, list)):
            return json.dumps(content).encode('utf-8')

        # As a last resort, convert to string
        return str(content).encode('utf-8')

    def get_metadata(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get metadata about the Confluence content without retrieving the full content.

        Args:
            location_data: Location data

        Returns:
            Dictionary with metadata

        Raises:
            ValueError: If metadata cannot be retrieved
        """
        source = location_data.get("source", "")

        # Check cache first
        if source in self.metadata_cache:
            return self.metadata_cache[source]

        # Parse Confluence URI
        parsed_data = self._parse_confluence_uri(source)

        # Get session for this base URL
        session = self._get_session(parsed_data["base_url"])

        try:
            # Determine content type
            content_type = parsed_data.get("content_type", "page")

            # Base API URL
            api_url = f"{parsed_data['base_url']}{self.api_path}"

            # Prepare metadata dictionary
            metadata = {
                "base_url": parsed_data["base_url"],
                "content_type": content_type,
                "retrieved_at": datetime.now().isoformat()
            }

            # Add space key if available
            if "space_key" in parsed_data:
                metadata["space_key"] = parsed_data["space_key"]

            # Handle different content types
            if content_type in ["page", "blog"]:
                # Fetch content metadata using content ID
                content_id = parsed_data.get("content_id")

                if content_id:
                    # Construct API URL
                    content_url = f"{api_url}/content/{content_id}"
                    params = {
                        "expand": "version,metadata.labels,history,space",
                    }

                    # Fetch metadata
                    response = session.get(
                        content_url,
                        params=params,
                        timeout=self.timeout,
                        verify=self.verify_ssl
                    )
                    response.raise_for_status()

                    # Parse response
                    content_data = response.json()

                    # Extract metadata
                    metadata.update({
                        "id": content_data.get("id"),
                        "title": content_data.get("title"),
                        "type": content_data.get("type"),
                        "status": content_data.get("status"),
                        "created_at": content_data.get("history", {}).get("createdDate"),
                        "updated_at": content_data.get("history", {}).get("lastUpdated", {}).get("when"),
                        "version": content_data.get("version", {}).get("number"),
                        "space_name": content_data.get("space", {}).get("name")
                    })

                    # Add labels if available
                    labels = content_data.get("metadata", {}).get("labels", {}).get("results", [])
                    if labels:
                        metadata["labels"] = [label.get("name") for label in labels]

            elif content_type == "attachment":
                # Fetch attachment metadata
                content_id = parsed_data.get("content_id")
                attachment_id = parsed_data.get("attachment_id")

                if content_id and attachment_id:
                    # Construct API URL
                    attachment_url = f"{api_url}/content/{content_id}/child/attachment/{attachment_id}"

                    # Fetch metadata
                    response = session.get(
                        attachment_url,
                        timeout=self.timeout,
                        verify=self.verify_ssl
                    )
                    response.raise_for_status()

                    # Parse response
                    attachment_data = response.json()

                    # Extract metadata
                    metadata.update({
                        "id": attachment_data.get("id"),
                        "title": attachment_data.get("title"),
                        "filename": attachment_data.get("title"),
                        "filesize": attachment_data.get("extensions", {}).get("fileSize"),
                        "media_type": attachment_data.get("metadata", {}).get("mediaType"),
                        "comment": attachment_data.get("metadata", {}).get("comment"),
                        "created_at": attachment_data.get("history", {}).get("createdDate"),
                        "updated_at": attachment_data.get("version", {}).get("when")
                    })

            elif content_type == "space":
                # Fetch space metadata
                space_key = parsed_data.get("space_key")

                if space_key:
                    # Construct API URL
                    space_url = f"{api_url}/space/{space_key}"
                    params = {
                        "expand": "description,metadata"
                    }

                    # Fetch metadata
                    response = session.get(
                        space_url,
                        params=params,
                        timeout=self.timeout,
                        verify=self.verify_ssl
                    )
                    response.raise_for_status()

                    # Parse response
                    space_data = response.json()

                    # Extract metadata
                    metadata.update({
                        "id": space_data.get("id"),
                        "key": space_data.get("key"),
                        "name": space_data.get("name"),
                        "type": space_data.get("type"),
                        "status": space_data.get("status"),
                        "description": space_data.get("description", {}).get("plain", {}).get("value", ""),
                        "created_at": space_data.get("created")
                    })

            elif content_type == "comment":
                # Fetch comment metadata
                content_id = parsed_data.get("content_id")
                comment_id = parsed_data.get("comment_id")

                if content_id and comment_id:
                    # Construct API URL
                    comment_url = f"{api_url}/content/{comment_id}"
                    params = {
                        "expand": "version,history,metadata"
                    }

                    # Fetch metadata
                    response = session.get(
                        comment_url,
                        params=params,
                        timeout=self.timeout,
                        verify=self.verify_ssl
                    )
                    response.raise_for_status()

                    # Parse response
                    comment_data = response.json()

                    # Extract metadata
                    metadata.update({
                        "id": comment_data.get("id"),
                        "title": comment_data.get("title"),
                        "type": comment_data.get("type"),
                        "status": comment_data.get("status"),
                        "created_at": comment_data.get("history", {}).get("createdDate"),
                        "created_by": comment_data.get("history", {}).get("createdBy", {}).get("displayName"),
                        "updated_at": comment_data.get("version", {}).get("when"),
                        "parent_id": content_id
                    })

            # Cache the metadata
            self.metadata_cache[source] = metadata

            return metadata

        except Exception as e:
            logger.error(f"Error retrieving Confluence metadata: {str(e)}")
            raise ValueError(f"Error retrieving Confluence metadata: {str(e)}")

    def resolve_uri(self, uri: str) -> Dict[str, Any]:
        """
        Parse a Confluence URI into location data.

        Args:
            uri: Confluence URI string

        Returns:
            Dictionary with parsed location data

        Raises:
            ValueError: If URI cannot be parsed
        """
        if not uri.startswith("confluence://"):
            raise ValueError(f"Not a Confluence URI: {uri}")

        # Parse URI into components
        parsed_data = self._parse_confluence_uri(uri)

        # Build location data
        location_data = {
            "source": uri
        }

        # Add all components to the location data
        location_data.update(parsed_data)

        return location_data

    def cleanup(self):
        """
        Clean up resources used by this adapter.

        This method should be called when the adapter is no longer needed.
        """
        # Close sessions
        for base_url, session in self.sessions.items():
            try:
                session.close()
                logger.debug(f"Closed Confluence session for: {base_url}")
            except Exception as e:
                logger.warning(f"Error closing Confluence session for {base_url}: {str(e)}")

        # Clear caches and session
        self.sessions = {}
        self.content_cache = {}
        self.binary_cache = {}
        self.metadata_cache = {}
        self.space_cache = {}
        self.page_cache = {}

    @staticmethod
    def _parse_confluence_uri(uri: str) -> Dict[str, Any]:
        """
        Parse Confluence URI into components.

        Expected formats:
        - confluence://domain/space/{space_key}
        - confluence://domain/page/{space_key}/{page_id}
        - confluence://domain/page/{space_key}/{page_id}/{element_type}/{element_id}
        - confluence://domain/blog/{space_key}/{blog_post_id}
        - confluence://domain/attachment/{content_id}/{attachment_id}
        - confluence://domain/comment/{content_id}/{comment_id}

        Args:
            uri: Confluence URI string

        Returns:
            Dictionary with parsed components
        """
        if not uri.startswith("confluence://"):
            raise ValueError(f"Invalid Confluence URI: {uri}")

        # Remove the confluence:// prefix
        path = uri[13:]

        # Split path
        parts = path.split('/')

        # We need at least a domain
        if len(parts) < 1:
            raise ValueError(f"Invalid Confluence URI format: {uri}")

        # Extract domain and build base URL
        domain = parts[0]
        if not domain.startswith(('http://', 'https://')):
            base_url = f"https://{domain}"
        else:
            base_url = domain

        # Create result with base URL
        result = {
            "base_url": base_url
        }

        # If we only have the domain, assume we're referring to the full instance
        if len(parts) == 1:
            return result

        # Check for content type
        if len(parts) >= 2:
            content_type = parts[1]
            result["content_type"] = content_type

            # Handle different content types
            if content_type == "space" and len(parts) >= 3:
                # Space URI
                result["space_key"] = parts[2]

            elif content_type == "page" and len(parts) >= 4:
                # Page URI
                result["space_key"] = parts[2]
                result["content_id"] = parts[3]

                # Check for element type and ID
                if len(parts) >= 6:
                    result["element_type"] = parts[4]
                    result["element_id"] = parts[5]

            elif content_type == "blog" and len(parts) >= 4:
                # Blog post URI
                result["space_key"] = parts[2]
                result["content_id"] = parts[3]

            elif content_type == "attachment" and len(parts) >= 4:
                # Attachment URI
                result["content_id"] = parts[2]
                result["attachment_id"] = parts[3]

            elif content_type == "comment" and len(parts) >= 4:
                # Comment URI
                result["content_id"] = parts[2]
                result["comment_id"] = parts[3]

        return result

    def _get_session(self, base_url: str) -> requests.Session:
        """
        Get or create a session for the given base URL.

        Args:
            base_url: Confluence base URL

        Returns:
            Requests session with authentication
        """
        # Check if session exists in cache
        if base_url in self.sessions:
            return self.sessions[base_url]

        # Create new session
        session = requests.Session()

        # Add authentication
        if self.auth_type == "basic":
            if self.username and (self.password or self.api_token):
                # Use API token if available, otherwise use password
                auth_password = self.api_token or self.password
                session.auth = (self.username, auth_password)
        elif self.auth_type == "bearer":
            if self.api_token:
                session.headers.update({
                    "Authorization": f"Bearer {self.api_token}"
                })

        # Add common headers
        session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Atlassian-Token": "no-check"
        })

        # Cache session
        self.sessions[base_url] = session

        return session

    def _get_page_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                          _location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get Confluence page content.

        Args:
            session: Requests session
            parsed_data: Parsed Confluence URI data
            _location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        content_id = parsed_data.get("content_id")
        element_type = parsed_data.get("element_type")
        element_id = parsed_data.get("element_id")

        # Construct API URL
        api_url = f"{base_url}{self.api_path}/content/{content_id}"

        # Determine what to expand
        expand = ["body.storage", "body.view", "version", "metadata.labels", "space"]

        if element_type == "history":
            expand.append("history")
        elif element_type == "children":
            expand.append("children")
        elif element_type == "descendants":
            expand.append("descendants")

        params = {
            "expand": ",".join(expand)
        }

        # Fetch page
        response = session.get(
            api_url,
            params=params,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Parse response
        page_data = response.json()

        # Determine what content to return based on element_type
        content = None
        content_type = "html"

        if element_type == "storage":
            # Return storage format (wiki markup or other source format)
            content = page_data.get("body", {}).get("storage", {}).get("value", "")
            content_type = page_data.get("body", {}).get("storage", {}).get("representation", "storage")
        elif element_type == "view":
            # Return view format (rendered HTML)
            content = page_data.get("body", {}).get("view", {}).get("value", "")
        elif element_type == "history":
            # Return history
            content = page_data.get("history", {})
            content_type = "json"
        elif element_type == "children":
            # Return children
            content = page_data.get("children", {})
            content_type = "json"
        elif element_type == "descendants":
            # Return descendants
            content = page_data.get("descendants", {})
            content_type = "json"
        elif element_type == "labels":
            # Return labels
            content = page_data.get("metadata", {}).get("labels", {})
            content_type = "json"
        elif element_type == "space":
            # Return space info
            content = page_data.get("space", {})
            content_type = "json"
        elif element_type and element_id:
            # Extract specific element by ID/selector
            storage_content = page_data.get("body", {}).get("storage", {}).get("value", "")

            # Parse HTML
            soup = BeautifulSoup(storage_content, 'html.parser')

            # Find element by ID or class
            if element_type == "id":
                element = soup.find(id=element_id)
                if element:
                    content = str(element)
            elif element_type == "class":
                element = soup.find(class_=element_id)
                if element:
                    content = str(element)
            elif element_type == "tag":
                elements = soup.find_all(element_id)
                if elements:
                    content = "\n".join(str(el) for el in elements)
            elif element_type == "selector":
                elements = soup.select(element_id)
                if elements:
                    content = "\n".join(str(el) for el in elements)
        else:
            # Return full content (default to view format)
            content = page_data.get("body", {}).get("view", {}).get("value", "")

        # If no content was found, but we have the page data
        if content is None:
            # Fall back to the complete page data
            content = page_data
            content_type = "json"

        # Extract metadata
        metadata = {
            "id": page_data.get("id"),
            "title": page_data.get("title"),
            "type": page_data.get("type"),
            "status": page_data.get("status"),
            "created_at": page_data.get("history", {}).get("createdDate"),
            "created_by": page_data.get("history", {}).get("createdBy", {}).get("displayName"),
            "updated_at": page_data.get("version", {}).get("when"),
            "updated_by": page_data.get("version", {}).get("by", {}).get("displayName"),
            "version": page_data.get("version", {}).get("number"),
            "space_key": page_data.get("space", {}).get("key"),
            "space_name": page_data.get("space", {}).get("name"),
            "base_url": base_url,
            "content_type": "page"
        }

        # Add labels if available
        labels = page_data.get("metadata", {}).get("labels", {}).get("results", [])
        if labels:
            metadata["labels"] = [label.get("name") for label in labels]

        # Create result
        result = {
            "content": content,
            "content_type": content_type,
            "metadata": metadata
        }

        # Cache the result
        self.content_cache[parsed_data["source"]] = result
        self.metadata_cache[parsed_data["source"]] = metadata

        return result

    def _get_blog_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                          location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get Confluence blog post content.

        Args:
            session: Requests session
            parsed_data: Parsed Confluence URI data
            location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        # Blog posts are handled the same way as pages in the Confluence API
        return self._get_page_content(session, parsed_data, location_data)

    def _get_attachment_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                                _location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get Confluence attachment content.

        Args:
            session: Requests session
            parsed_data: Parsed Confluence URI data
            _location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        content_id = parsed_data.get("content_id")
        attachment_id = parsed_data.get("attachment_id")

        # Construct API URL for attachment metadata
        api_url = f"{base_url}{self.api_path}/content/{content_id}/child/attachment/{attachment_id}"

        # Fetch attachment metadata
        response = session.get(
            api_url,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Parse response
        attachment_data = response.json()

        # Get download URL
        download_url = f"{base_url}/download/attachments/{content_id}/{attachment_id}"

        # Fetch attachment content
        response = session.get(
            download_url,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Get content
        content = response.content

        # Determine content type
        media_type = attachment_data.get("metadata", {}).get("mediaType", "")
        filename = attachment_data.get("title", "")

        # Detect content type
        if media_type:
            content_type = self._media_type_to_content_type(media_type, filename)
        else:
            content_type = DocumentTypeDetector.detect(
                path=filename,
                content=content,
                metadata={"binary": True}
            )

        # Try to convert to text if it's a text-based format
        if self._is_text_content(media_type):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                # Keep as binary if decoding fails
                pass

        # Extract metadata
        metadata = {
            "id": attachment_data.get("id"),
            "title": attachment_data.get("title"),
            "type": attachment_data.get("type"),
            "status": attachment_data.get("status"),
            "filename": attachment_data.get("title"),
            "media_type": media_type,
            "comment": attachment_data.get("metadata", {}).get("comment"),
            "filesize": attachment_data.get("extensions", {}).get("fileSize"),
            "created_at": attachment_data.get("history", {}).get(
                "createdDate") if "history" in attachment_data else None,
            "created_by": attachment_data.get("history", {}).get("createdBy", {}).get(
                "displayName") if "history" in attachment_data else None,
            "updated_at": attachment_data.get("version", {}).get("when") if "version" in attachment_data else None,
            "updated_by": attachment_data.get("version", {}).get("by", {}).get(
                "displayName") if "version" in attachment_data else None,
            "version": attachment_data.get("version", {}).get("number") if "version" in attachment_data else None,
            "download_url": download_url,
            "base_url": base_url,
            "content_type": "attachment",
            "parent_id": content_id
        }

        # Create result
        result = {
            "content": content,
            "content_type": content_type,
            "metadata": metadata
        }

        # Cache the result
        self.content_cache[parsed_data["source"]] = result
        self.metadata_cache[parsed_data["source"]] = metadata

        return result

    def _get_space_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                           _location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get Confluence space content.

        Args:
            session: Requests session
            parsed_data: Parsed Confluence URI data
            _location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        space_key = parsed_data.get("space_key")

        # Construct API URL
        api_url = f"{base_url}{self.api_path}/space/{space_key}"

        # Determine what to expand
        params = {
            "expand": "description,metadata,icon,homepage"
        }

        # Fetch space
        response = session.get(
            api_url,
            params=params,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Parse response
        space_data = response.json()

        # Get space description
        description = space_data.get("description", {}).get("view", {}).get("value", "")

        # Extract metadata
        metadata = {
            "id": space_data.get("id"),
            "key": space_data.get("key"),
            "name": space_data.get("name"),
            "type": space_data.get("type"),
            "status": space_data.get("status"),
            "created_at": space_data.get("created"),
            "icon": space_data.get("icon", {}).get("path"),
            "homepage_id": space_data.get("homepage", {}).get("id") if "homepage" in space_data else None,
            "base_url": base_url,
            "content_type": "space"
        }

        # Add homepage information if available
        if "homepage" in space_data:
            metadata["homepage_title"] = space_data["homepage"].get("title")

        # Create result
        result = {
            "content": description,
            "content_type": "html",
            "metadata": metadata
        }

        # Cache the result
        self.content_cache[parsed_data["source"]] = result
        self.metadata_cache[parsed_data["source"]] = metadata

        return result

    def _get_comment_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                             _location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get Confluence comment content.

        Args:
            session: Requests session
            parsed_data: Parsed Confluence URI data
            _location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        content_id = parsed_data.get("content_id")
        comment_id = parsed_data.get("comment_id")

        # Construct API URL
        api_url = f"{base_url}{self.api_path}/content/{comment_id}"

        # Determine what to expand
        params = {
            "expand": "body.view,version,history"
        }

        # Fetch comment
        response = session.get(
            api_url,
            params=params,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Parse response
        comment_data = response.json()

        # Get comment content
        content = comment_data.get("body", {}).get("view", {}).get("value", "")

        # Extract metadata
        metadata = {
            "id": comment_data.get("id"),
            "title": comment_data.get("title"),
            "type": comment_data.get("type"),
            "status": comment_data.get("status"),
            "created_at": comment_data.get("history", {}).get("createdDate"),
            "created_by": comment_data.get("history", {}).get("createdBy", {}).get("displayName"),
            "updated_at": comment_data.get("version", {}).get("when"),
            "updated_by": comment_data.get("version", {}).get("by", {}).get("displayName"),
            "version": comment_data.get("version", {}).get("number"),
            "base_url": base_url,
            "content_type": "comment",
            "parent_id": content_id
        }

        # Create result
        result = {
            "content": content,
            "content_type": "html",
            "metadata": metadata
        }

        # Cache the result
        self.content_cache[parsed_data["source"]] = result
        self.metadata_cache[parsed_data["source"]] = metadata

        return result

    @staticmethod
    def _media_type_to_content_type(media_type: str, filename: str) -> str:
        """
        Convert MIME media type to content type.

        Args:
            media_type: MIME media type
            filename: Filename for additional context

        Returns:
            Content type string
        """
        # Extract extension from filename
        extension = ""
        if filename and "." in filename:
            extension = filename.split(".")[-1].lower()

        # Handle common document types
        if media_type.startswith("text/html") or extension in ["html", "htm"]:
            return "html"
        elif media_type.startswith("text/markdown") or extension in ["md", "markdown"]:
            return "markdown"
        elif media_type.startswith("application/pdf") or extension == "pdf":
            return "pdf"
        elif media_type.startswith(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document") or extension == "docx":
            return "docx"
        elif media_type.startswith(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") or extension == "xlsx":
            return "xlsx"
        elif media_type.startswith(
                "application/vnd.openxmlformats-officedocument.presentationml.presentation") or extension == "pptx":
            return "pptx"
        elif media_type.startswith("application/json") or extension == "json":
            return "json"
        elif media_type.startswith("application/xml") or extension in ["xml", "svg"]:
            return "xml"
        elif media_type.startswith("text/plain") or extension == "txt":
            return "text"
        elif media_type.startswith("text/csv") or extension == "csv":
            return "csv"
        elif media_type.startswith("image/"):
            return "image"
        elif media_type.startswith("video/"):
            return "video"
        elif media_type.startswith("audio/"):
            return "audio"

        # Default to binary for unknown types
        return "binary"

    @staticmethod
    def _is_text_content(media_type: str) -> bool:
        """
        Determine if a media type represents text-based content.

        Args:
            media_type: MIME media type

        Returns:
            True if text-based, False otherwise
        """
        if not media_type:
            return False

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
            if media_type.startswith(text_type):
                return True

        return False
