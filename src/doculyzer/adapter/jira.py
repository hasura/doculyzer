"""
JIRA adapter module for the document pointer system.

This module provides an adapter to retrieve content from Atlassian JIRA sources.
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
    logger.warning("requests is required for JIRA adapter. Install with 'pip install requests beautifulsoup4'")


class JiraAdapter(ContentSourceAdapter):
    """Adapter for Atlassian JIRA content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the JIRA adapter."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests and beautifulsoup4 are required for JIRA adapter")

        super().__init__(config)
        self.config = config or {}

        # JIRA connection settings
        self.base_url = self.config.get("base_url")
        self.api_path = "/rest/api/2"
        self.timeout = self.config.get("timeout", 30)
        self.verify_ssl = self.config.get("verify_ssl", True)

        # Authentication settings
        self.auth_type = self.config.get("auth_type", "basic")
        self.username = self.config.get("username")
        self.password = self.config.get("password")
        self.api_token = self.config.get("api_token")

        # Optional settings
        self.include_attachments = self.config.get("include_attachments", False)
        self.include_comments = self.config.get("include_comments", True)
        self.include_changelog = self.config.get("include_changelog", False)
        self.max_results = self.config.get("max_results", 100)
        self.fields = self.config.get("fields", "*all")

        # Initialize session
        self.sessions = {}

        # Caches
        self.content_cache = {}
        self.binary_cache = {}
        self.metadata_cache = {}
        self.field_cache = {}
        self.project_cache = {}

    def get_content(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get content from JIRA.

        Args:
            location_data: Location data with JIRA information

        Returns:
            Dictionary with content and metadata

        Raises:
            ValueError: If JIRA content cannot be retrieved
        """
        source = location_data.get("source", "")
        if not source.startswith("jira://"):
            raise ValueError(f"Invalid JIRA source: {source}")

        # Check cache first
        if source in self.content_cache:
            return self.content_cache[source]

        # Parse JIRA URI
        parsed_data = self._parse_jira_uri(source)

        # Get session for this base URL
        session = self._get_session(parsed_data["base_url"])

        try:
            # Determine the type of resource we need to fetch
            content_type = parsed_data.get("content_type", "issue")

            # Handle different content types
            if content_type == "issue":
                return self._get_issue_content(session, parsed_data, location_data)
            elif content_type == "attachment":
                return self._get_attachment_content(session, parsed_data, location_data)
            elif content_type == "comment":
                return self._get_comment_content(session, parsed_data, location_data)
            elif content_type == "project":
                return self._get_project_content(session, parsed_data, location_data)
            elif content_type == "search":
                return self._get_search_content(session, parsed_data, location_data)
            elif content_type == "field":
                return self._get_field_content(session, parsed_data, location_data)
            else:
                raise ValueError(f"Unsupported JIRA content type: {content_type}")

        except Exception as e:
            logger.error(f"Error retrieving JIRA content: {str(e)}")
            raise ValueError(f"Error retrieving JIRA content: {str(e)}")

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
        # Source must be a JIRA URI
        return source.startswith("jira://")

    def get_binary_content(self, location_data: Dict[str, Any]) -> bytes:
        """
        Get the JIRA content as binary data.

        Args:
            location_data: Location data

        Returns:
            Binary content

        Raises:
            ValueError: If content cannot be retrieved as binary
        """
        source = location_data.get("source", "")

        # Check if it's specifically an attachment
        parsed_data = self._parse_jira_uri(source)

        # If it's an attachment, get the binary content directly
        if parsed_data.get("content_type") == "attachment":
            # Check cache first
            if source in self.binary_cache:
                return self.binary_cache[source]

            # Get session
            session = self._get_session(parsed_data["base_url"])

            try:
                # Construct URL
                attachment_id = parsed_data.get("attachment_id")
                # issue_key = parsed_data.get("issue_key")

                # Fetch attachment metadata first to get the content URL
                metadata_url = f"{parsed_data['base_url']}{self.api_path}/attachment/{attachment_id}"

                response = session.get(
                    metadata_url,
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )
                response.raise_for_status()

                # Parse response
                attachment_data = response.json()

                # Get download URL
                download_url = attachment_data.get("content")

                if not download_url:
                    raise ValueError(f"Could not find download URL for attachment {attachment_id}")

                # Fetch attachment
                response = session.get(
                    download_url,
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
        Get metadata about the JIRA content without retrieving the full content.

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

        # Parse JIRA URI
        parsed_data = self._parse_jira_uri(source)

        # Get session for this base URL
        session = self._get_session(parsed_data["base_url"])

        try:
            # Determine content type
            content_type = parsed_data.get("content_type", "issue")

            # Base API URL
            api_url = f"{parsed_data['base_url']}{self.api_path}"

            # Prepare metadata dictionary
            metadata = {
                "base_url": parsed_data["base_url"],
                "content_type": content_type,
                "retrieved_at": datetime.now().isoformat()
            }

            # Handle different content types
            if content_type == "issue":
                # Fetch issue metadata
                issue_key = parsed_data.get("issue_key")

                if issue_key:
                    # Construct API URL - just get fields, not content
                    issue_url = f"{api_url}/issue/{issue_key}"
                    params = {
                        "fields": "summary,issuetype,status,created,updated,project,creator,reporter"
                    }

                    # Fetch issue
                    response = session.get(
                        issue_url,
                        params=params,
                        timeout=self.timeout,
                        verify=self.verify_ssl
                    )
                    response.raise_for_status()

                    # Parse response
                    issue_data = response.json()
                    fields = issue_data.get("fields", {})

                    # Extract metadata
                    metadata.update({
                        "id": issue_data.get("id"),
                        "key": issue_data.get("key"),
                        "summary": fields.get("summary"),
                        "issue_type": fields.get("issuetype", {}).get("name"),
                        "status": fields.get("status", {}).get("name"),
                        "created_at": fields.get("created"),
                        "updated_at": fields.get("updated"),
                        "project_key": fields.get("project", {}).get("key"),
                        "project_name": fields.get("project", {}).get("name"),
                        "creator": fields.get("creator", {}).get("displayName"),
                        "reporter": fields.get("reporter", {}).get("displayName")
                    })

            elif content_type == "attachment":
                # Fetch attachment metadata
                attachment_id = parsed_data.get("attachment_id")

                if attachment_id:
                    # Construct API URL
                    attachment_url = f"{api_url}/attachment/{attachment_id}"

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
                        "filename": attachment_data.get("filename"),
                        "created_at": attachment_data.get("created"),
                        "author": attachment_data.get("author", {}).get("displayName"),
                        "size": attachment_data.get("size"),
                        "mime_type": attachment_data.get("mimeType"),
                        "issue_id": attachment_data.get("self", "").split("/")[-3],
                        "issue_key": parsed_data.get("issue_key")
                    })

            elif content_type == "comment":
                # Fetch comment metadata
                comment_id = parsed_data.get("comment_id")
                issue_key = parsed_data.get("issue_key")

                if comment_id and issue_key:
                    # Construct API URL
                    comment_url = f"{api_url}/issue/{issue_key}/comment/{comment_id}"

                    # Fetch comment
                    response = session.get(
                        comment_url,
                        timeout=self.timeout,
                        verify=self.verify_ssl
                    )
                    response.raise_for_status()

                    # Parse response
                    comment_data = response.json()

                    # Extract metadata
                    metadata.update({
                        "id": comment_data.get("id"),
                        "created_at": comment_data.get("created"),
                        "updated_at": comment_data.get("updated"),
                        "author": comment_data.get("author", {}).get("displayName"),
                        "issue_key": issue_key
                    })

            elif content_type == "project":
                # Fetch project metadata
                project_key = parsed_data.get("project_key")

                if project_key:
                    # Construct API URL
                    project_url = f"{api_url}/project/{project_key}"

                    # Fetch project
                    response = session.get(
                        project_url,
                        timeout=self.timeout,
                        verify=self.verify_ssl
                    )
                    response.raise_for_status()

                    # Parse response
                    project_data = response.json()

                    # Extract metadata
                    metadata.update({
                        "id": project_data.get("id"),
                        "key": project_data.get("key"),
                        "name": project_data.get("name"),
                        "project_type": project_data.get("projectTypeKey"),
                        "lead": project_data.get("lead", {}).get("displayName")
                    })

            # Cache the metadata
            self.metadata_cache[source] = metadata

            return metadata

        except Exception as e:
            logger.error(f"Error retrieving JIRA metadata: {str(e)}")
            raise ValueError(f"Error retrieving JIRA metadata: {str(e)}")

    def resolve_uri(self, uri: str) -> Dict[str, Any]:
        """
        Parse a JIRA URI into location data.

        Args:
            uri: JIRA URI string

        Returns:
            Dictionary with parsed location data

        Raises:
            ValueError: If URI cannot be parsed
        """
        if not uri.startswith("jira://"):
            raise ValueError(f"Not a JIRA URI: {uri}")

        # Parse URI into components
        parsed_data = self._parse_jira_uri(uri)

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
                logger.debug(f"Closed JIRA session for: {base_url}")
            except Exception as e:
                logger.warning(f"Error closing JIRA session for {base_url}: {str(e)}")

        # Clear caches and session
        self.sessions = {}
        self.content_cache = {}
        self.binary_cache = {}
        self.metadata_cache = {}
        self.field_cache = {}
        self.project_cache = {}

    @staticmethod
    def _parse_jira_uri(uri: str) -> Dict[str, Any]:
        """
        Parse JIRA URI into components.

        Expected formats:
        - jira://domain/issue/{issue_key}
        - jira://domain/issue/{issue_key}/{field_id}
        - jira://domain/issue/{issue_key}/attachment/{attachment_id}
        - jira://domain/issue/{issue_key}/comment/{comment_id}
        - jira://domain/project/{project_key}
        - jira://domain/search/{jql_query}
        - jira://domain/field/{field_id}

        Args:
            uri: JIRA URI string

        Returns:
            Dictionary with parsed components
        """
        if not uri.startswith("jira://"):
            raise ValueError(f"Invalid JIRA URI: {uri}")

        # Remove the jira:// prefix
        path = uri[7:]

        # Split path
        parts = path.split('/')

        # We need at least a domain
        if len(parts) < 1:
            raise ValueError(f"Invalid JIRA URI format: {uri}")

        # Extract domain and build base URL
        domain = parts[0]
        if not domain.startswith(('http://', 'https://')):
            base_url = f"https://{domain}"
        else:
            base_url = domain

        # Create result with base URL
        result = {
            "base_url": base_url,
            "source": uri
        }

        # If we only have the domain, assume default content type
        if len(parts) == 1:
            result["content_type"] = "issue"
            return result

        # Check for content type
        if len(parts) >= 2:
            content_type = parts[1]
            result["content_type"] = content_type

            # Handle different content types
            if content_type == "issue" and len(parts) >= 3:
                # Issue URI - always include issue key
                result["issue_key"] = parts[2]

                # Check for additional components
                if len(parts) >= 5 and parts[3] == "attachment":
                    # Attachment URI
                    result["content_type"] = "attachment"
                    result["attachment_id"] = parts[4]
                elif len(parts) >= 5 and parts[3] == "comment":
                    # Comment URI
                    result["content_type"] = "comment"
                    result["comment_id"] = parts[4]
                elif len(parts) >= 4:
                    # Field URI
                    result["field_id"] = parts[3]

            elif content_type == "project" and len(parts) >= 3:
                # Project URI
                result["project_key"] = parts[2]

            elif content_type == "search" and len(parts) >= 3:
                # Search URI - extract JQL
                result["jql"] = "/".join(parts[2:])

            elif content_type == "field" and len(parts) >= 3:
                # Field URI
                result["field_id"] = parts[2]

        return result

    def _get_session(self, base_url: str) -> requests.Session:
        """
        Get or create a session for the given base URL.

        Args:
            base_url: JIRA base URL

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

    def _get_issue_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                           _location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get JIRA issue content.

        Args:
            session: Requests session
            parsed_data: Parsed JIRA URI data
            _location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        issue_key = parsed_data.get("issue_key")
        field_id = parsed_data.get("field_id")

        if not issue_key:
            raise ValueError("Issue key is required for issue content")

        # Construct API URL
        api_url = f"{base_url}{self.api_path}/issue/{issue_key}"

        # Determine what to expand
        expand = []

        if self.include_attachments:
            expand.append("attachment")

        if self.include_comments:
            expand.append("comment")

        if self.include_changelog:
            expand.append("changelog")

        params = {
            "fields": self.fields
        }

        if expand:
            params["expand"] = ",".join(expand)

        # Fetch issue
        response = session.get(
            api_url,
            params=params,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Parse response
        issue_data = response.json()

        # Determine what content to return based on field_id
        # content = None
        content_type = "json"

        if field_id:
            # Get specific field
            fields = issue_data.get("fields", {})

            # Check if this is a custom field ID
            if field_id.startswith("customfield_"):
                # Direct access to custom field
                content = fields.get(field_id)
            else:
                # Map field ID to field name
                field_map = self._get_field_map(session, base_url)
                actual_field_id = field_map.get(field_id, field_id)
                content = fields.get(actual_field_id)

            # If field value is complex, return as JSON
            if isinstance(content, (dict, list)):
                content_type = "json"
                # Convert to string for consistency
                content = json.dumps(content, indent=2)
            else:
                # For simple values, treat as text
                content_type = "text"
                content = str(content) if content is not None else ""

        else:
            # Return entire issue as JSON
            content = json.dumps(issue_data, indent=2)

        # Extract metadata
        fields = issue_data.get("fields", {})
        metadata = {
            "id": issue_data.get("id"),
            "key": issue_data.get("key"),
            "url": f"{base_url}/browse/{issue_data.get('key')}",
            "summary": fields.get("summary"),
            "issue_type": fields.get("issuetype", {}).get("name"),
            "status": fields.get("status", {}).get("name"),
            "created_at": fields.get("created"),
            "updated_at": fields.get("updated"),
            "project_key": fields.get("project", {}).get("key"),
            "project_name": fields.get("project", {}).get("name"),
            "creator": fields.get("creator", {}).get("displayName"),
            "reporter": fields.get("reporter", {}).get("displayName"),
            "base_url": base_url,
            "content_type": "issue"
        }

        # Add attachment info if available
        attachments = fields.get("attachment", [])
        if attachments:
            metadata["attachment_count"] = len(attachments)
            metadata["attachments"] = [{
                "id": att.get("id"),
                "filename": att.get("filename"),
                "size": att.get("size")
            } for att in attachments]

        # Add comment info if available
        comments = fields.get("comment", {}).get("comments", [])
        if comments:
            metadata["comment_count"] = len(comments)

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

    def _get_attachment_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                                _location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get JIRA attachment content.

        Args:
            session: Requests session
            parsed_data: Parsed JIRA URI data
            _location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        issue_key = parsed_data.get("issue_key")
        attachment_id = parsed_data.get("attachment_id")

        if not attachment_id:
            raise ValueError("Attachment ID is required for attachment content")

        # Construct API URL for attachment metadata
        api_url = f"{base_url}{self.api_path}/attachment/{attachment_id}"

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
        download_url = attachment_data.get("content")

        if not download_url:
            raise ValueError(f"Could not find download URL for attachment {attachment_id}")

        # Fetch attachment content
        response = session.get(
            download_url,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Get content
        content = response.content

        # Determine content type from MIME type
        mime_type = attachment_data.get("mimeType", "")
        filename = attachment_data.get("filename", "")

        # Detect content type
        if mime_type:
            content_type = self._mime_type_to_content_type(mime_type, filename)
        else:
            content_type = DocumentTypeDetector.detect(
                path=filename,
                content=content,
                metadata={"binary": True}
            )

        # Try to convert to text if it's a text-based format
        if self._is_text_content(mime_type):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                # Keep as binary if decoding fails
                pass

        # Extract metadata
        metadata = {
            "id": attachment_data.get("id"),
            "filename": attachment_data.get("filename"),
            "created_at": attachment_data.get("created"),
            "author": attachment_data.get("author", {}).get("displayName"),
            "size": attachment_data.get("size"),
            "mime_type": attachment_data.get("mimeType"),
            "download_url": download_url,
            "base_url": base_url,
            "content_type": "attachment",
            "issue_key": issue_key,
            "issue_id": attachment_data.get("self", "").split("/")[-3]
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

    def _get_comment_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                             _location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get JIRA comment content.

        Args:
            session: Requests session
            parsed_data: Parsed JIRA URI data
            _location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        issue_key = parsed_data.get("issue_key")
        comment_id = parsed_data.get("comment_id")

        if not comment_id or not issue_key:
            raise ValueError("Issue key and comment ID are required for comment content")

        # Construct API URL
        api_url = f"{base_url}{self.api_path}/issue/{issue_key}/comment/{comment_id}"

        # Fetch comment
        response = session.get(
            api_url,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Parse response
        comment_data = response.json()

        # Get comment body in different formats
        content = comment_data.get("body", "")

        # Try to extract rendered content if available
        rendered_content = comment_data.get("renderedBody", "")
        if rendered_content:
            content = rendered_content
            content_type = "html"
        else:
            # Use plain text
            content_type = "text"

        # Extract metadata
        metadata = {
            "id": comment_data.get("id"),
            "created_at": comment_data.get("created"),
            "updated_at": comment_data.get("updated"),
            "author": comment_data.get("author", {}).get("displayName"),
            "base_url": base_url,
            "content_type": "comment",
            "issue_key": issue_key
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

    def _get_project_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                             _location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get JIRA project content.

        Args:
            session: Requests session
            parsed_data: Parsed JIRA URI data
            _location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        project_key = parsed_data.get("project_key")

        if not project_key:
            raise ValueError("Project key is required for project content")

        # Construct API URL
        api_url = f"{base_url}{self.api_path}/project/{project_key}"

        # Fetch project
        response = session.get(
            api_url,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Parse response
        project_data = response.json()

        # Get project content as JSON
        content = json.dumps(project_data, indent=2)
        content_type = "json"

        # Extract metadata
        metadata = {
            "id": project_data.get("id"),
            "key": project_data.get("key"),
            "name": project_data.get("name"),
            "url": f"{base_url}/projects/{project_data.get('key')}",
            "project_type": project_data.get("projectTypeKey"),
            "lead": project_data.get("lead", {}).get("displayName"),
            "base_url": base_url,
            "content_type": "project"
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
        self.project_cache[project_key] = project_data

        return result

    def _get_search_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                            _location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get JIRA search content.

        Args:
            session: Requests session
            parsed_data: Parsed JIRA URI data
            _location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        jql = parsed_data.get("jql", "")

        if not jql:
            raise ValueError("JQL query is required for search content")

        # Construct API URL
        api_url = f"{base_url}{self.api_path}/search"

        # Prepare params
        params = {
            "jql": jql,
            "maxResults": self.max_results,
            "fields": self.fields
        }

        # Fetch search results
        response = session.get(
            api_url,
            params=params,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Parse response
        search_data = response.json()

        # Get search content as JSON
        content = json.dumps(search_data, indent=2)
        content_type = "json"

        # Extract metadata
        metadata = {
            "jql": jql,
            "total": search_data.get("total", 0),
            "max_results": self.max_results,
            "start_at": search_data.get("startAt", 0),
            "base_url": base_url,
            "content_type": "search"
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

    def _get_field_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                           _location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get JIRA field content.

        Args:
            session: Requests session
            parsed_data: Parsed JIRA URI data
            _location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        field_id = parsed_data.get("field_id")

        if not field_id:
            # Get all fields
            api_url = f"{base_url}{self.api_path}/field"

            # Fetch fields
            response = session.get(
                api_url,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            response.raise_for_status()

            # Parse response
            fields_data = response.json()

            # Get fields content as JSON
            content = json.dumps(fields_data, indent=2)
            content_type = "json"

            # Extract metadata
            metadata = {
                "field_count": len(fields_data),
                "base_url": base_url,
                "content_type": "field"
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
        else:
            # Get specific field
            field_map = self._get_field_map(session, base_url)

            # Find field by ID or name
            field_data = None

            # Check if field exists in the map
            if field_id in field_map:
                field_data = field_map[field_id]
            else:
                # Get all fields
                api_url = f"{base_url}{self.api_path}/field"

                # Fetch fields
                response = session.get(
                    api_url,
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )
                response.raise_for_status()

                # Parse response
                fields_data = response.json()

                # Find field
                for field in fields_data:
                    if field.get("id") == field_id or field.get("name") == field_id:
                        field_data = field
                        break

            if not field_data:
                raise ValueError(f"Field not found: {field_id}")

            # Get field content as JSON
            content = json.dumps(field_data, indent=2)
            content_type = "json"

            # Extract metadata
            metadata = {
                "id": field_data.get("id"),
                "name": field_data.get("name"),
                "custom": field_data.get("custom", False),
                "schema": field_data.get("schema"),
                "base_url": base_url,
                "content_type": "field"
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

    def _get_field_map(self, session: requests.Session, base_url: str) -> Dict[str, Any]:
        """
        Get field name to ID mapping.

        Args:
            session: Requests session
            base_url: JIRA base URL

        Returns:
            Dictionary mapping field names to field IDs
        """
        # Check cache first
        cache_key = f"field_map_{base_url}"
        if cache_key in self.field_cache:
            return self.field_cache[cache_key]

        # Construct API URL
        api_url = f"{base_url}{self.api_path}/field"

        try:
            # Fetch fields
            response = session.get(
                api_url,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            response.raise_for_status()

            # Parse response
            fields_data = response.json()

            # Build mapping
            field_map = {}

            for field in fields_data:
                field_id = field.get("id")
                field_name = field.get("name")
                if field_id and field_name:
                    field_map[field_name] = field_id
                    field_map[field_id] = field

            # Cache the mapping
            self.field_cache[cache_key] = field_map

            return field_map

        except Exception as e:
            logger.error(f"Error fetching JIRA fields: {str(e)}")
            return {}

    @staticmethod
    def _mime_type_to_content_type(mime_type: str, filename: str) -> str:
        """
        Convert MIME type to content type.

        Args:
            mime_type: MIME type
            filename: Filename for additional context

        Returns:
            Content type string
        """
        # Extract extension from filename
        extension = ""
        if filename and "." in filename:
            extension = filename.split(".")[-1].lower()

        # Handle common document types
        if mime_type.startswith("text/html") or extension in ["html", "htm"]:
            return "html"
        elif mime_type.startswith("text/markdown") or extension in ["md", "markdown"]:
            return "markdown"
        elif mime_type.startswith("application/pdf") or extension == "pdf":
            return "pdf"
        elif mime_type.startswith(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document") or extension == "docx":
            return "docx"
        elif mime_type.startswith(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") or extension == "xlsx":
            return "xlsx"
        elif mime_type.startswith(
                "application/vnd.openxmlformats-officedocument.presentationml.presentation") or extension == "pptx":
            return "pptx"
        elif mime_type.startswith("application/json") or extension == "json":
            return "json"
        elif mime_type.startswith("application/xml") or extension in ["xml", "svg"]:
            return "xml"
        elif mime_type.startswith("text/plain") or extension == "txt":
            return "text"
        elif mime_type.startswith("text/csv") or extension == "csv":
            return "csv"
        elif mime_type.startswith("image/"):
            return "image"
        elif mime_type.startswith("video/"):
            return "video"
        elif mime_type.startswith("audio/"):
            return "audio"

        # Default to binary for unknown types
        return "binary"

    @staticmethod
    def _is_text_content(mime_type: str) -> bool:
        """
        Determine if a MIME type represents text-based content.

        Args:
            mime_type: MIME type

        Returns:
            True if text-based, False otherwise
        """
        if not mime_type:
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

        # Check if MIME type starts with any of the text types
        for text_type in text_types:
            if mime_type.startswith(text_type):
                return True

        return False
