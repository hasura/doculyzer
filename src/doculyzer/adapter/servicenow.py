"""
ServiceNow adapter module for the document pointer system.

This module provides an adapter to retrieve content from ServiceNow sources.
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
    logger.warning("requests is required for ServiceNow adapter. Install with 'pip install requests beautifulsoup4'")


class ServiceNowAdapter(ContentSourceAdapter):
    """Adapter for ServiceNow content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ServiceNow adapter."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests and beautifulsoup4 are required for ServiceNow adapter")

        super().__init__(config)
        self.config = config or {}

        # ServiceNow connection settings
        self.base_url = self.config.get("base_url")
        self.table_api_path = "/api/now/table"
        self.knowledge_api_path = "/api/sn_km_api/knowledge"
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
        self.include_work_notes = self.config.get("include_work_notes", False)
        self.max_results = self.config.get("max_results", 100)

        # Initialize session
        self.sessions = {}

        # Caches
        self.content_cache = {}
        self.binary_cache = {}
        self.metadata_cache = {}

    def get_content(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get content from ServiceNow.

        Args:
            location_data: Location data with ServiceNow information

        Returns:
            Dictionary with content and metadata

        Raises:
            ValueError: If ServiceNow content cannot be retrieved
        """
        source = location_data.get("source", "")
        if not source.startswith("servicenow://"):
            raise ValueError(f"Invalid ServiceNow source: {source}")

        # Check cache first
        if source in self.content_cache:
            return self.content_cache[source]

        # Parse ServiceNow URI
        parsed_data = self._parse_servicenow_uri(source)

        # Get session for this base URL
        session = self._get_session(parsed_data["base_url"])

        try:
            # Determine the type of resource we need to fetch
            content_type = parsed_data.get("content_type", "incident")

            # Handle different content types
            if content_type == "knowledge":
                return self._get_knowledge_content(session, parsed_data, location_data)
            elif content_type == "incident":
                return self._get_incident_content(session, parsed_data, location_data)
            elif content_type == "catalog_item":
                return self._get_catalog_item_content(session, parsed_data, location_data)
            elif content_type == "attachment":
                return self._get_attachment_content(session, parsed_data, location_data)
            elif content_type == "cmdb":
                return self._get_cmdb_content(session, parsed_data, location_data)
            else:
                raise ValueError(f"Unsupported ServiceNow content type: {content_type}")

        except Exception as e:
            logger.error(f"Error retrieving ServiceNow content: {str(e)}")
            raise ValueError(f"Error retrieving ServiceNow content: {str(e)}")

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
        # Source must be a ServiceNow URI
        return source.startswith("servicenow://")

    def get_binary_content(self, location_data: Dict[str, Any]) -> bytes:
        """
        Get the ServiceNow content as binary data.

        Args:
            location_data: Location data

        Returns:
            Binary content

        Raises:
            ValueError: If content cannot be retrieved as binary
        """
        source = location_data.get("source", "")

        # Check if it's specifically an attachment
        parsed_data = self._parse_servicenow_uri(source)

        # If it's an attachment, get the binary content directly
        if parsed_data.get("content_type") == "attachment":
            # Check cache first
            if source in self.binary_cache:
                return self.binary_cache[source]

            # Get session
            session = self._get_session(parsed_data["base_url"])

            try:
                # Get attachment information first
                attachment_id = parsed_data.get("attachment_id")
                attachment_url = f"{parsed_data['base_url']}{self.table_api_path}/sys_attachment/{attachment_id}"

                # Get attachment metadata
                response = session.get(
                    attachment_url,
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )
                response.raise_for_status()

                # Get attachment download URL
                download_url = f"{parsed_data['base_url']}/sys_attachment.do?sys_id={attachment_id}"

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
        Get metadata about the ServiceNow content without retrieving the full content.

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

        # Parse ServiceNow URI
        parsed_data = self._parse_servicenow_uri(source)

        # Get session for this base URL
        session = self._get_session(parsed_data["base_url"])

        try:
            # Determine content type
            content_type = parsed_data.get("content_type", "incident")

            # Base API URL
            if content_type == "knowledge":
                api_url = f"{parsed_data['base_url']}{self.knowledge_api_path}"
            else:
                api_url = f"{parsed_data['base_url']}{self.table_api_path}"

            # Prepare metadata dictionary
            metadata = {
                "base_url": parsed_data["base_url"],
                "content_type": content_type,
                "retrieved_at": datetime.now().isoformat()
            }

            # Handle different content types
            if content_type == "knowledge":
                # Fetch knowledge article metadata
                article_id = parsed_data.get("article_id")

                if article_id:
                    # Construct API URL
                    content_url = f"{api_url}/articles/{article_id}"

                    # Fetch metadata
                    response = session.get(
                        content_url,
                        timeout=self.timeout,
                        verify=self.verify_ssl
                    )
                    response.raise_for_status()

                    # Parse response
                    article_data = response.json().get("result", {})

                    # Extract metadata
                    metadata.update({
                        "id": article_id,
                        "number": article_data.get("number"),
                        "short_description": article_data.get("short_description"),
                        "sys_created_on": article_data.get("sys_created_on"),
                        "sys_updated_on": article_data.get("sys_updated_on"),
                        "sys_created_by": article_data.get("sys_created_by"),
                        "sys_updated_by": article_data.get("sys_updated_by"),
                        "url": f"{parsed_data['base_url']}/kb_view.do?sys_kb_id={article_id}"
                    })

            elif content_type == "incident":
                # Fetch incident metadata
                incident_id = parsed_data.get("incident_id")

                if incident_id:
                    # Construct API URL
                    content_url = f"{api_url}/incident/{incident_id}"

                    # Fetch metadata
                    response = session.get(
                        content_url,
                        timeout=self.timeout,
                        verify=self.verify_ssl
                    )
                    response.raise_for_status()

                    # Parse response
                    incident_data = response.json().get("result", {})

                    # Extract metadata
                    metadata.update({
                        "id": incident_id,
                        "number": incident_data.get("number"),
                        "short_description": incident_data.get("short_description"),
                        "state": incident_data.get("state"),
                        "priority": incident_data.get("priority"),
                        "sys_created_on": incident_data.get("sys_created_on"),
                        "sys_updated_on": incident_data.get("sys_updated_on"),
                        "sys_created_by": incident_data.get("sys_created_by"),
                        "sys_updated_by": incident_data.get("sys_updated_by"),
                        "url": f"{parsed_data['base_url']}/nav_to.do?uri=incident.do?sys_id={incident_id}"
                    })

            elif content_type == "catalog_item":
                # Fetch catalog item metadata
                item_id = parsed_data.get("item_id")

                if item_id:
                    # Construct API URL
                    content_url = f"{api_url}/sc_cat_item/{item_id}"

                    # Fetch metadata
                    response = session.get(
                        content_url,
                        timeout=self.timeout,
                        verify=self.verify_ssl
                    )
                    response.raise_for_status()

                    # Parse response
                    item_data = response.json().get("result", {})

                    # Extract metadata
                    metadata.update({
                        "id": item_id,
                        "name": item_data.get("name"),
                        "short_description": item_data.get("short_description"),
                        "category": item_data.get("category"),
                        "sys_created_on": item_data.get("sys_created_on"),
                        "sys_updated_on": item_data.get("sys_updated_on"),
                        "sys_created_by": item_data.get("sys_created_by"),
                        "sys_updated_by": item_data.get("sys_updated_by"),
                        "url": f"{parsed_data['base_url']}/nav_to.do?uri=sc_cat_item.do?sys_id={item_id}"
                    })

            elif content_type == "attachment":
                # Fetch attachment metadata
                attachment_id = parsed_data.get("attachment_id")

                if attachment_id:
                    # Construct API URL
                    content_url = f"{api_url}/sys_attachment/{attachment_id}"

                    # Fetch metadata
                    response = session.get(
                        content_url,
                        timeout=self.timeout,
                        verify=self.verify_ssl
                    )
                    response.raise_for_status()

                    # Parse response
                    attachment_data = response.json().get("result", {})

                    # Extract metadata
                    metadata.update({
                        "id": attachment_id,
                        "file_name": attachment_data.get("file_name"),
                        "content_type": attachment_data.get("content_type"),
                        "size_bytes": attachment_data.get("size_bytes"),
                        "table_name": attachment_data.get("table_name"),
                        "table_sys_id": attachment_data.get("table_sys_id"),
                        "sys_created_on": attachment_data.get("sys_created_on"),
                        "sys_updated_on": attachment_data.get("sys_updated_on"),
                        "sys_created_by": attachment_data.get("sys_created_by"),
                        "sys_updated_by": attachment_data.get("sys_updated_by"),
                        "url": f"{parsed_data['base_url']}/sys_attachment.do?sys_id={attachment_id}"
                    })

            elif content_type == "cmdb":
                # Fetch CMDB CI metadata
                ci_id = parsed_data.get("ci_id")

                if ci_id:
                    # Construct API URL
                    content_url = f"{api_url}/cmdb_ci/{ci_id}"

                    # Fetch metadata
                    response = session.get(
                        content_url,
                        timeout=self.timeout,
                        verify=self.verify_ssl
                    )
                    response.raise_for_status()

                    # Parse response
                    ci_data = response.json().get("result", {})

                    # Extract metadata
                    metadata.update({
                        "id": ci_id,
                        "name": ci_data.get("name"),
                        "sys_class_name": ci_data.get("sys_class_name"),
                        "sys_created_on": ci_data.get("sys_created_on"),
                        "sys_updated_on": ci_data.get("sys_updated_on"),
                        "sys_created_by": ci_data.get("sys_created_by"),
                        "sys_updated_by": ci_data.get("sys_updated_by"),
                        "url": f"{parsed_data['base_url']}/nav_to.do?uri=cmdb_ci.do?sys_id={ci_id}"
                    })

            # Cache the metadata
            self.metadata_cache[source] = metadata

            return metadata

        except Exception as e:
            logger.error(f"Error retrieving ServiceNow metadata: {str(e)}")
            raise ValueError(f"Error retrieving ServiceNow metadata: {str(e)}")

    def resolve_uri(self, uri: str) -> Dict[str, Any]:
        """
        Parse a ServiceNow URI into location data.

        Args:
            uri: ServiceNow URI string

        Returns:
            Dictionary with parsed location data

        Raises:
            ValueError: If URI cannot be parsed
        """
        if not uri.startswith("servicenow://"):
            raise ValueError(f"Not a ServiceNow URI: {uri}")

        # Parse URI into components
        parsed_data = self._parse_servicenow_uri(uri)

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
                logger.debug(f"Closed ServiceNow session for: {base_url}")
            except Exception as e:
                logger.warning(f"Error closing ServiceNow session for {base_url}: {str(e)}")

        # Clear caches and session
        self.sessions = {}
        self.content_cache = {}
        self.binary_cache = {}
        self.metadata_cache = {}

    @staticmethod
    def _parse_servicenow_uri(uri: str) -> Dict[str, Any]:
        """
        Parse ServiceNow URI into components.

        Expected formats:
        - servicenow://instance/knowledge/{article_id}
        - servicenow://instance/incident/{incident_id}
        - servicenow://instance/catalog_item/{item_id}
        - servicenow://instance/attachment/{attachment_id}
        - servicenow://instance/cmdb/{ci_id}

        Args:
            uri: ServiceNow URI string

        Returns:
            Dictionary with parsed components
        """
        if not uri.startswith("servicenow://"):
            raise ValueError(f"Invalid ServiceNow URI: {uri}")

        # Remove the servicenow:// prefix
        path = uri[13:]

        # Split path
        parts = path.split('/')

        # We need at least a domain
        if len(parts) < 1:
            raise ValueError(f"Invalid ServiceNow URI format: {uri}")

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

        # If we only have the domain, assume default content type
        if len(parts) == 1:
            result["content_type"] = "incident"
            return result

        # Check for content type
        if len(parts) >= 2:
            content_type = parts[1]
            result["content_type"] = content_type

            # Handle different content types
            if content_type == "knowledge" and len(parts) >= 3:
                # Knowledge article URI
                result["article_id"] = parts[2]

            elif content_type == "incident" and len(parts) >= 3:
                # Incident URI
                result["incident_id"] = parts[2]

            elif content_type == "catalog_item" and len(parts) >= 3:
                # Catalog item URI
                result["item_id"] = parts[2]

            elif content_type == "attachment" and len(parts) >= 3:
                # Attachment URI
                result["attachment_id"] = parts[2]
                # Add table name and record ID if available
                if len(parts) >= 5:
                    result["table_name"] = parts[3]
                    result["table_sys_id"] = parts[4]

            elif content_type == "cmdb" and len(parts) >= 3:
                # CMDB CI URI
                result["ci_id"] = parts[2]

        return result

    def _get_session(self, base_url: str) -> requests.Session:
        """
        Get or create a session for the given base URL.

        Args:
            base_url: ServiceNow base URL

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
            "Content-Type": "application/json"
        })

        # Cache session
        self.sessions[base_url] = session

        return session

    def _get_knowledge_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                               location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get ServiceNow knowledge article content.

        Args:
            session: Requests session
            parsed_data: Parsed ServiceNow URI data
            location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        article_id = parsed_data.get("article_id")

        if not article_id:
            raise ValueError("Article ID is required for knowledge content")

        # Construct API URL
        api_url = f"{base_url}{self.knowledge_api_path}/articles/{article_id}"

        # Fetch article
        response = session.get(
            api_url,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Parse response
        article_data = response.json().get("result", {})

        # Get article content
        content = article_data.get("content", "")

        # Clean up HTML content if needed
        if content and isinstance(content, str):
            try:
                # Parse HTML to strip scripts or other unwanted elements
                soup = BeautifulSoup(content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()

                # Get clean HTML
                content = str(soup)
            except Exception as e:
                logger.warning(f"Error cleaning HTML content: {str(e)}")

        # Extract metadata
        metadata = {
            "id": article_id,
            "number": article_data.get("number"),
            "short_description": article_data.get("short_description"),
            "sys_created_on": article_data.get("sys_created_on"),
            "sys_updated_on": article_data.get("sys_updated_on"),
            "sys_created_by": article_data.get("sys_created_by"),
            "sys_updated_by": article_data.get("sys_updated_by"),
            "base_url": base_url,
            "content_type": "knowledge",
            "url": f"{base_url}/kb_view.do?sys_kb_id={article_id}"
        }

        # Create result
        result = {
            "content": content,
            "content_type": "html",
            "metadata": metadata
        }

        # Cache the result
        source = location_data.get("source", "")
        self.content_cache[source] = result
        self.metadata_cache[source] = metadata

        return result

    def _get_incident_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                              location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get ServiceNow incident content.

        Args:
            session: Requests session
            parsed_data: Parsed ServiceNow URI data
            location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        incident_id = parsed_data.get("incident_id")

        if not incident_id:
            raise ValueError("Incident ID is required for incident content")

        # Construct API URL for incident
        api_url = f"{base_url}{self.table_api_path}/incident/{incident_id}"

        # Fetch incident
        response = session.get(
            api_url,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Parse response
        incident_data = response.json().get("result", {})

        # Should we include comments and work notes?
        if self.include_comments or self.include_work_notes:
            try:
                # Get comments/work notes from journal entries
                journal_url = f"{base_url}{self.table_api_path}/sys_journal_field"
                params = {
                    "sysparm_query": f"element_id={incident_id}^ORDERBYsys_created_on",
                    "sysparm_fields": "sys_created_on,sys_created_by,value,element"
                }

                journal_response = session.get(
                    journal_url,
                    params=params,
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )
                journal_response.raise_for_status()

                # Extract journal entries and add to incident data
                entries = journal_response.json().get("result", [])

                # Filter based on settings
                filtered_entries = []
                for entry in entries:
                    element = entry.get("element", "")
                    if element == "comments" and self.include_comments:
                        filtered_entries.append(entry)
                    elif element == "work_notes" and self.include_work_notes:
                        filtered_entries.append(entry)

                incident_data["journal_entries"] = filtered_entries

            except Exception as e:
                logger.warning(f"Error fetching journal entries: {str(e)}")

        # Extract metadata
        metadata = {
            "id": incident_id,
            "number": incident_data.get("number"),
            "short_description": incident_data.get("short_description"),
            "state": incident_data.get("state"),
            "priority": incident_data.get("priority"),
            "sys_created_on": incident_data.get("sys_created_on"),
            "sys_updated_on": incident_data.get("sys_updated_on"),
            "sys_created_by": incident_data.get("sys_created_by"),
            "sys_updated_by": incident_data.get("sys_updated_by"),
            "assigned_to": incident_data.get("assigned_to", {}).get("display_value"),
            "base_url": base_url,
            "content_type": "incident",
            "url": f"{base_url}/nav_to.do?uri=incident.do?sys_id={incident_id}"
        }

        # Create result
        result = {
            "content": incident_data,
            "content_type": "json",
            "metadata": metadata
        }

        # Cache the result
        source = location_data.get("source", "")
        self.content_cache[source] = result
        self.metadata_cache[source] = metadata

        return result

    def _get_catalog_item_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                                  location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get ServiceNow catalog item content.

        Args:
            session: Requests session
            parsed_data: Parsed ServiceNow URI data
            location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        item_id = parsed_data.get("item_id")

        if not item_id:
            raise ValueError("Item ID is required for catalog item content")

        # Construct API URL
        api_url = f"{base_url}{self.table_api_path}/sc_cat_item/{item_id}"

        # Fetch catalog item
        response = session.get(
            api_url,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Parse response
        item_data = response.json().get("result", {})

        # Get variables if any
        variables_url = f"{base_url}{self.table_api_path}/item_option_new"
        params = {
            "sysparm_query": f"cat_item={item_id}",
            "sysparm_fields": "name,question_text,type,default_value,mandatory"
        }

        try:
            variables_response = session.get(
                variables_url,
                params=params,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            variables_response.raise_for_status()

            # Add variables to item data
            item_data["variables"] = variables_response.json().get("result", [])
        except Exception as e:
            logger.warning(f"Error fetching catalog item variables: {str(e)}")

        # Extract metadata
        metadata = {
            "id": item_id,
            "name": item_data.get("name"),
            "short_description": item_data.get("short_description"),
            "category": item_data.get("category"),
            "sys_created_on": item_data.get("sys_created_on"),
            "sys_updated_on": item_data.get("sys_updated_on"),
            "sys_created_by": item_data.get("sys_created_by"),
            "sys_updated_by": item_data.get("sys_updated_by"),
            "base_url": base_url,
            "content_type": "catalog_item",
            "url": f"{base_url}/nav_to.do?uri=sc_cat_item.do?sys_id={item_id}"
        }

        # Create result
        result = {
            "content": item_data,
            "content_type": "json",
            "metadata": metadata
        }

        # Cache the result
        source = location_data.get("source", "")
        self.content_cache[source] = result
        self.metadata_cache[source] = metadata

        return result

    def _get_attachment_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                                location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get ServiceNow attachment content.

        Args:
            session: Requests session
            parsed_data: Parsed ServiceNow URI data
            location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        attachment_id = parsed_data.get("attachment_id")

        if not attachment_id:
            raise ValueError("Attachment ID is required for attachment content")

        # Construct API URL for attachment metadata
        api_url = f"{base_url}{self.table_api_path}/sys_attachment/{attachment_id}"

        # Fetch attachment metadata
        response = session.get(
            api_url,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Parse response
        attachment_data = response.json().get("result", {})

        # Get download URL
        download_url = f"{base_url}/sys_attachment.do?sys_id={attachment_id}"

        # Fetch attachment content
        response = session.get(
            download_url,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Get content
        content = response.content

        # Get content type
        content_type_str = attachment_data.get("content_type", "")
        file_name = attachment_data.get("file_name", "")

        # Determine document type based on content type or file name
        doc_type = DocumentTypeDetector.detect(
            path=file_name,
            content=content,
            metadata={"binary": True, "content_type": content_type_str}
        )

        # Try to convert to text if appropriate
        if self._is_text_content(content_type_str):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                # Keep as binary if decoding fails
                pass

        # Extract metadata
        metadata = {
            "id": attachment_id,
            "file_name": attachment_data.get("file_name"),
            "content_type": attachment_data.get("content_type"),
            "size_bytes": attachment_data.get("size_bytes"),
            "table_name": attachment_data.get("table_name"),
            "table_sys_id": attachment_data.get("table_sys_id"),
            "sys_created_on": attachment_data.get("sys_created_on"),
            "sys_updated_on": attachment_data.get("sys_updated_on"),
            "sys_created_by": attachment_data.get("sys_created_by"),
            "sys_updated_by": attachment_data.get("sys_updated_by"),
            "base_url": base_url,
            "download_url": download_url
        }

        # Create result
        result = {
            "content": content,
            "content_type": doc_type,
            "metadata": metadata
        }

        # Cache the result
        source = location_data.get("source", "")
        self.content_cache[source] = result
        self.metadata_cache[source] = metadata

        return result

    def _get_cmdb_content(self, session: requests.Session, parsed_data: Dict[str, Any],
                          location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get ServiceNow CMDB CI content.

        Args:
            session: Requests session
            parsed_data: Parsed ServiceNow URI data
            location_data: Original location data

        Returns:
            Dictionary with content and metadata
        """
        base_url = parsed_data["base_url"]
        ci_id = parsed_data.get("ci_id")

        if not ci_id:
            raise ValueError("CI ID is required for CMDB content")

        # Construct API URL
        api_url = f"{base_url}{self.table_api_path}/cmdb_ci/{ci_id}"

        # Fetch CMDB CI
        response = session.get(
            api_url,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()

        # Parse response
        ci_data = response.json().get("result", {})

        # Get relationships if available
        try:
            # Query for relationships
            rel_url = f"{base_url}{self.table_api_path}/cmdb_rel_ci"
            params = {
                "sysparm_query": f"parent={ci_id}^ORchild={ci_id}",
                "sysparm_fields": "parent,child,type,relation_type"
            }

            rel_response = session.get(
                rel_url,
                params=params,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            rel_response.raise_for_status()

            # Add relationships to CI data
            ci_data["relationships"] = rel_response.json().get("result", [])
        except Exception as e:
            logger.warning(f"Error fetching CI relationships: {str(e)}")

        # Extract metadata
        metadata = {
            "id": ci_id,
            "name": ci_data.get("name"),
            "sys_class_name": ci_data.get("sys_class_name"),
            "sys_created_on": ci_data.get("sys_created_on"),
            "sys_updated_on": ci_data.get("sys_updated_on"),
            "sys_created_by": ci_data.get("sys_created_by"),
            "sys_updated_by": ci_data.get("sys_updated_by"),
            "base_url": base_url,
            "content_type": "cmdb",
            "url": f"{base_url}/nav_to.do?uri=cmdb_ci.do?sys_id={ci_id}"
        }

        # Create result
        result = {
            "content": ci_data,
            "content_type": "json",
            "metadata": metadata
        }

        # Cache the result
        source = location_data.get("source", "")
        self.content_cache[source] = result
        self.metadata_cache[source] = metadata

        return result

    @staticmethod
    def _is_text_content(content_type: str) -> bool:
        """
        Determine if a content type represents text-based content.

        Args:
            content_type: MIME content type

        Returns:
            True if text-based, False otherwise
        """
        if not content_type:
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
            if content_type.startswith(text_type):
                return True

        return False
