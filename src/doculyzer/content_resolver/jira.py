"""
JIRA Content Resolver implementation for the document pointer system.

This module resolves JIRA content pointers to actual content by utilizing
the pre-rendered HTML directly from JIRA's API.
"""
import json
import logging
import re
from typing import Dict, Any, Optional

import requests

from .base import ContentResolver

logger = logging.getLogger(__name__)


class JiraContentResolver(ContentResolver):
    """Resolver for JIRA content."""

    def __init__(self):
        """Initialize the JIRA content resolver."""
        self.sessions = {}  # Cache for API sessions
        self.content_cache = {}  # Cache for retrieved content

    def resolve_content(self, content_location: str) -> str:
        """
        Resolve JIRA content pointer to actual content.

        Args:
            content_location: JSON-formatted content location pointer

        Returns:
            Resolved content as string
        """
        location_data = json.loads(content_location)

        source = location_data.get("source", "")
        if not source.startswith("jira://"):
            raise ValueError(f"Invalid JIRA source: {source}")

        # Extract info from source identifier
        # Format: jira://base_url/ISSUE-123
        match = re.match(r'jira://([^/]+)/([A-Z][A-Z0-9_]+-\d+)', source)
        if not match:
            raise ValueError(f"Invalid JIRA source format: {source}")

        base_url, issue_key = match.groups()

        # Ensure base_url has a scheme
        if not base_url.startswith(('http://', 'https://')):
            base_url = 'https://' + base_url

        # Try to get API credentials from session cache
        session = self._get_session(base_url)

        # Determine what part of the content to return based on element type
        element_type = location_data.get("type", "")

        # Check cache first
        cache_key = f"{base_url}:{issue_key}:{element_type}"
        if cache_key in self.content_cache:
            logger.debug(f"Using cached content for: {cache_key}")
            return self.content_cache[cache_key]

        try:
            # Construct API URL
            api_url = f"{base_url}/rest/api/2/issue/{issue_key}"

            # Set up parameters based on what we need to retrieve
            expand_params = ["renderedFields"]

            # Include comments if needed
            if element_type == "comment" or element_type == "comments":
                expand_params.append("comment")

            # Include change history if needed
            if element_type == "changelog" or element_type == "history":
                expand_params.append("changelog")

            # Set up parameters
            params = {
                "expand": ",".join(expand_params)
            }

            # Make API request
            response = session.get(api_url, params=params)
            response.raise_for_status()
            issue_data = response.json()

            # Extract content based on element type
            if element_type == "root" or element_type == "issue":
                # Format full issue content as HTML
                resolved_content = self._format_full_issue(issue_data)
            elif element_type == "description":
                # Return description field directly from renderedFields
                resolved_content = issue_data.get("renderedFields", {}).get("description", "")
            elif element_type == "comment" or element_type == "comments":
                # Extract specific comment(s)
                comment_id = location_data.get("comment_id")
                comment_index = location_data.get("index")

                comments_data = issue_data.get("comment", {}).get("comments", [])
                resolved_content = self._extract_comments(comments_data, comment_id, comment_index)
            elif element_type == "field":
                # Extract specific field
                field_id = location_data.get("field_id", "")
                field_name = location_data.get("field_name", "")

                resolved_content = self._extract_field(issue_data, field_id, field_name)
            elif element_type == "changelog" or element_type == "history":
                # Extract changelog
                history_id = location_data.get("history_id")

                changelog = issue_data.get("changelog", {}).get("histories", [])
                resolved_content = self._format_changelog(changelog, history_id)
            else:
                # Default: return full issue content
                resolved_content = self._format_full_issue(issue_data)

            # Cache the result
            self.content_cache[cache_key] = resolved_content

            return resolved_content

        except Exception as e:
            logger.error(f"Error resolving JIRA content: {str(e)}")
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
            # Source must be a JIRA URI
            return source.startswith("jira://")
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
        # For JIRA, we'll return the HTML content as bytes
        content = self.resolve_content(content_location)
        return content.encode('utf-8')

    def _get_session(self, base_url: str) -> requests.Session:
        """
        Get or create a session for the given base URL.

        Args:
            base_url: JIRA base URL

        Returns:
            Requests session with authentication
        """
        if base_url in self.sessions:
            return self.sessions[base_url]

        # Create a new session
        session = requests.Session()

        # Try to find credentials for this base URL
        # In a real implementation, this would use a secure credential store
        # or configuration service. For now, we'll just return the unauthenticated session.

        # Add any required headers
        session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json"
        })

        # Cache the session
        self.sessions[base_url] = session

        return session

    @staticmethod
    def _format_full_issue(issue_data: Dict[str, Any]) -> str:
        """
        Format complete issue as HTML, using rendered fields from JIRA API.

        Args:
            issue_data: JIRA issue data

        Returns:
            Formatted HTML content
        """
        # Extract basic issue information
        key = issue_data.get("key", "")
        fields = issue_data.get("fields", {})
        rendered_fields = issue_data.get("renderedFields", {})

        summary = fields.get("summary", "")
        status = fields.get("status", {}).get("name", "")
        issue_type = fields.get("issuetype", {}).get("name", "")

        # Get description (use rendered version for HTML)
        description_html = rendered_fields.get("description", "")

        # Build minimal wrapper HTML
        html = f"<div class='jira-issue'>\n"
        html += f"<h1>{key}: {summary}</h1>\n"
        html += f"<div class='jira-meta'>\n"
        html += f"<p><strong>Type:</strong> {issue_type} | <strong>Status:</strong> {status}</p>\n"
        html += f"</div>\n"

        # Add description (using the pre-rendered HTML directly)
        if description_html:
            html += f"<h2>Description</h2>\n"
            html += f"{description_html}\n"

        # Add comments if available (using pre-rendered HTML directly)
        comments = issue_data.get("comment", {}).get("comments", [])
        if comments:
            html += f"<h2>Comments</h2>\n"
            html += f"<div class='jira-comments'>\n"

            for comment in comments:
                author = comment.get("author", {}).get("displayName", "Unknown")
                created = comment.get("created", "")

                # Use the pre-rendered HTML from JIRA
                body_html = comment.get("renderedBody", comment.get("body", ""))

                html += f"<div class='jira-comment'>\n"
                html += f"<p class='jira-comment-meta'><strong>{author}</strong> - {created}</p>\n"
                html += f"{body_html}\n"
                html += f"</div>\n"

            html += f"</div>\n"

        html += f"</div>\n"
        return html

    @staticmethod
    def _extract_comments(comments_data: list, comment_id: Optional[str] = None,
                          comment_index: Optional[int] = None) -> str:
        """
        Extract comment(s) from JIRA issue, using pre-rendered HTML.

        Args:
            comments_data: List of comments
            comment_id: Specific comment ID to extract
            comment_index: Specific comment index to extract

        Returns:
            HTML content with comment(s)
        """
        if not comments_data:
            return "<p>No comments found.</p>"

        if comment_id:
            # Extract specific comment by ID
            for comment in comments_data:
                if comment.get("id") == comment_id:
                    # Use the pre-rendered HTML directly
                    return comment.get("renderedBody", comment.get("body", ""))

            # Comment not found
            return f"<p>Comment with ID {comment_id} not found.</p>"

        elif comment_index is not None:
            # Extract specific comment by index
            if 0 <= comment_index < len(comments_data):
                comment = comments_data[comment_index]
                # Use the pre-rendered HTML directly
                return comment.get("renderedBody", comment.get("body", ""))

            # Index out of range
            return f"<p>Comment at index {comment_index} not found.</p>"

        else:
            # Return all comments with minimal wrapper
            html = f"<div class='jira-comments'>\n"

            for comment in comments_data:
                author = comment.get("author", {}).get("displayName", "Unknown")
                created = comment.get("created", "")

                # Use the pre-rendered HTML directly
                body_html = comment.get("renderedBody", comment.get("body", ""))

                html += f"<div class='jira-comment'>\n"
                html += f"<p class='jira-comment-meta'><strong>{author}</strong> - {created}</p>\n"
                html += f"{body_html}\n"
                html += f"</div>\n"

            html += f"</div>\n"
            return html

    @staticmethod
    def _extract_field(issue_data: Dict[str, Any], field_id: str, field_name: str) -> str:
        """
        Extract specific field from JIRA issue, using pre-rendered HTML when available.

        Args:
            issue_data: JIRA issue data
            field_id: Field ID (e.g., "customfield_10010")
            field_name: Field name

        Returns:
            HTML content with field value
        """
        fields = issue_data.get("fields", {})
        rendered_fields = issue_data.get("renderedFields", {})
        names = issue_data.get("names", {})

        # Determine field ID if only name provided
        if not field_id and field_name:
            for f_id, f_name in names.items():
                if f_name.lower() == field_name.lower():
                    field_id = f_id
                    break

        # Try to get rendered value first (preferred)
        if field_id and field_id in rendered_fields and rendered_fields[field_id]:
            display_name = names.get(field_id, field_id)

            # Return the pre-rendered HTML with minimal wrapper
            html = f"<div class='jira-field'>\n"
            html += f"<h3>{display_name}</h3>\n"
            html += f"{rendered_fields[field_id]}\n"
            html += f"</div>\n"
            return html

        # If no rendered value or field ID not found, try fields
        if field_id and field_id in fields and fields[field_id]:
            value = fields[field_id]
            display_name = names.get(field_id, field_id)

            # Format value based on type (only needed for non-rendered fields)
            # formatted_value = ""

            if isinstance(value, dict):
                # Handle complex field types (e.g., user, option)
                if "displayName" in value:
                    formatted_value = value["displayName"]
                elif "name" in value:
                    formatted_value = value["name"]
                elif "value" in value:
                    formatted_value = value["value"]
                else:
                    formatted_value = str(value)
            elif isinstance(value, list):
                # Handle array fields
                if all(isinstance(item, dict) for item in value):
                    # List of objects
                    if all("displayName" in item for item in value):
                        formatted_value = ", ".join(item["displayName"] for item in value)
                    elif all("name" in item for item in value):
                        formatted_value = ", ".join(item["name"] for item in value)
                    elif all("value" in item for item in value):
                        formatted_value = ", ".join(item["value"] for item in value)
                    else:
                        formatted_value = ", ".join(str(item) for item in value)
                else:
                    # Simple array
                    formatted_value = ", ".join(str(item) for item in value)
            else:
                # Simple value
                formatted_value = str(value)

            # Return with minimal wrapper
            html = f"<div class='jira-field'>\n"
            html += f"<h3>{display_name}</h3>\n"
            html += f"<div class='jira-field-value'>{formatted_value}</div>\n"
            html += f"</div>\n"
            return html

        # Field not found
        html = f"<div class='jira-field-error'>\n"
        html += f"<p>Field not found: {field_name or field_id}</p>\n"
        html += f"</div>\n"
        return html

    @staticmethod
    def _format_changelog(changelog_data: list, history_id: Optional[str] = None) -> str:
        """
        Format changelog as HTML.

        Args:
            changelog_data: List of changelog histories
            history_id: Specific history ID to extract

        Returns:
            HTML content with changelog
        """
        if not changelog_data:
            return "<p>No change history found.</p>"

        if history_id:
            # Extract specific history entry
            for history in changelog_data:
                if history.get("id") == history_id:
                    author = history.get("author", {}).get("displayName", "Unknown")
                    created = history.get("created", "")
                    items = history.get("items", [])

                    # Minimal HTML wrapper for a single history entry
                    html = f"<div class='jira-history'>\n"
                    html += f"<p class='jira-history-meta'><strong>{author}</strong> - {created}</p>\n"

                    if items:
                        html += f"<ul class='jira-history-items'>\n"

                        for item in items:
                            field = item.get("field", "")
                            from_string = item.get("fromString", "")
                            to_string = item.get("toString", "")

                            html += f"<li><strong>{field}</strong>: {from_string} → {to_string}</li>\n"

                        html += f"</ul>\n"

                    html += f"</div>\n"
                    return html

            # History not found
            return f"<p>History with ID {history_id} not found.</p>"

        else:
            # Return all history entries with minimal HTML wrapper
            html = f"<div class='jira-changelog'>\n"
            html += f"<h2>Change History</h2>\n"

            for history in changelog_data:
                author = history.get("author", {}).get("displayName", "Unknown")
                created = history.get("created", "")
                items = history.get("items", [])

                html += f"<div class='jira-history'>\n"
                html += f"<p class='jira-history-meta'><strong>{author}</strong> - {created}</p>\n"

                if items:
                    html += f"<ul class='jira-history-items'>\n"

                    for item in items:
                        field = item.get("field", "")
                        from_string = item.get("fromString", "")
                        to_string = item.get("toString", "")

                        html += f"<li><strong>{field}</strong>: {from_string} → {to_string}</li>\n"

                    html += f"</ul>\n"

                html += f"</div>\n"

            html += f"</div>\n"
            return html
