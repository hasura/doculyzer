"""
JIRA Content Source for the document pointer system.

This module provides integration with Atlassian JIRA via its REST API.
"""

import logging
import re
from typing import Dict, Any, List, Optional

import requests
import time

from .base import ContentSource

logger = logging.getLogger(__name__)


class JiraContentSource(ContentSource):
    """Content source for Atlassian JIRA."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the JIRA content source.

        Args:
            config: Configuration dictionary containing JIRA connection details
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "").rstrip('/')
        self.username = config.get("username", "")
        self.api_token = config.get("api_token", "")
        self.password = config.get("password", "")

        # JQL configuration
        self.jql_query = config.get("jql_query", "")
        self.projects = config.get("projects", [])
        self.issue_types = config.get("issue_types", [])
        self.statuses = config.get("statuses", [])
        self.include_closed = config.get("include_closed", False)
        self.max_results = config.get("max_results", 100)

        # Content configuration
        self.include_description = config.get("include_description", True)
        self.include_comments = config.get("include_comments", True)
        self.include_attachments = config.get("include_attachments", False)
        self.include_subtasks = config.get("include_subtasks", True)
        self.include_linked_issues = config.get("include_linked_issues", False)
        self.include_custom_fields = config.get("include_custom_fields", [])

        # Link following configuration
        self.max_link_depth = config.get("max_link_depth", 1)

        # Initialize session
        self.session = requests.Session()

        # Set up authentication
        if self.username:
            if self.api_token:
                # Use API token authentication
                self.session.auth = (self.username, self.api_token)
            elif self.password:
                # Use basic authentication
                self.session.auth = (self.username, self.password)

        # Cache for content
        self.content_cache = {}

    def fetch_document(self, source_id: str) -> Dict[str, Any]:
        """
        Fetch document content from JIRA.

        Args:
            source_id: Identifier for the issue (usually issue key or ID)

        Returns:
            Dictionary containing document content and metadata
        """
        logger.debug(f"Fetching JIRA issue: {source_id}")

        try:
            # Extract issue key if source_id is a URL or complex identifier
            issue_key = self._extract_issue_key(source_id)

            # Construct API URL
            api_url = f"{self.base_url}/rest/api/2/issue/{issue_key}"

            # Set up parameters for the API request
            params = {
                "expand": "renderedFields,names,schema,operations,editmeta,changelog,versionedRepresentations"
            }

            # Make API request
            response = self.session.get(api_url, params=params)
            response.raise_for_status()
            issue_data = response.json()

            # Extract issue details
            issue_key = issue_data.get("key", "")
            issue_id = issue_data.get("id", "")
            fields = issue_data.get("fields", {})
            rendered_fields = issue_data.get("renderedFields", {})

            # Extract basic issue information
            summary = fields.get("summary", "")
            issue_type = fields.get("issuetype", {}).get("name", "")
            status = fields.get("status", {}).get("name", "")
            project_key = fields.get("project", {}).get("key", "")
            project_name = fields.get("project", {}).get("name", "")

            # Get HTML content from rendered fields
            description_html = rendered_fields.get("description", "")

            # Build HTML content for the entire issue
            html_content = f"<h1>{issue_key}: {summary}</h1>\n"
            html_content += f"<p><strong>Type:</strong> {issue_type} | <strong>Status:</strong> {status}</p>\n"

            # Add description
            if description_html and self.include_description:
                html_content += f"<h2>Description</h2>\n{description_html}\n"

            # Fetch and add comments if configured
            if self.include_comments:
                comments_html = self._fetch_comments(issue_key)
                if comments_html:
                    html_content += f"<h2>Comments</h2>\n{comments_html}\n"

            # Add custom fields if configured
            if self.include_custom_fields:
                custom_fields_html = self._fetch_custom_fields(issue_data, rendered_fields)
                if custom_fields_html:
                    html_content += f"<h2>Additional Details</h2>\n{custom_fields_html}\n"

            # Create fully qualified source identifier
            qualified_source = f"jira://{self.base_url}/{issue_key}"

            # Construct metadata
            metadata = {
                "issue_key": issue_key,
                "issue_id": issue_id,
                "summary": summary,
                "project": project_key,
                "project_name": project_name,
                "issue_type": issue_type,
                "status": status,
                "url": f"{self.base_url}/browse/{issue_key}",
                "api_url": api_url,
                "content_type": "html"  # Explicitly mark as HTML content
            }

            # Add some additional fields to metadata
            if "created" in fields:
                metadata["created"] = fields["created"]
            if "updated" in fields:
                metadata["updated"] = fields["updated"]
            if "creator" in fields:
                metadata["creator"] = fields["creator"].get("displayName", "")
            if "assignee" in fields and fields["assignee"]:
                metadata["assignee"] = fields["assignee"].get("displayName", "")

            # Generate content hash for change detection
            content_hash = self.get_content_hash(html_content)

            # Cache the content for faster access
            self.content_cache[issue_key] = {
                "content": html_content,
                "metadata": metadata,
                "hash": content_hash,
                "last_accessed": time.time()
            }

            return {
                "id": qualified_source,
                "content": html_content,
                "doc_type": "html",  # Explicitly mark as HTML document type
                "metadata": metadata,
                "content_hash": content_hash
            }

        except Exception as e:
            logger.error(f"Error fetching JIRA issue {source_id}: {str(e)}")
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List available issues in JIRA.

        Returns:
            List of issue identifiers and metadata
        """
        logger.debug("Listing JIRA issues")
        results = []

        try:
            # Build JQL query if not explicitly provided
            jql = self._build_jql_query()
            logger.debug(f"Using JQL query: {jql}")

            # Set up API request
            api_url = f"{self.base_url}/rest/api/2/search"

            # JIRA API limits max results per request, so we'll need to paginate
            start_at = 0
            max_per_request = min(100, self.max_results)  # JIRA API typically limits to 100 per request
            total_fetched = 0

            while total_fetched < self.max_results:
                # Set up parameters for this request
                params = {
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": max_per_request,
                    "fields": "summary,issuetype,status,project,updated,created"
                }

                # Make API request
                response = self.session.get(api_url, params=params)
                response.raise_for_status()
                search_data = response.json()

                # Process results
                issues = search_data.get("issues", [])
                if not issues:
                    break  # No more issues to fetch

                for issue in issues:
                    issue_key = issue.get("key", "")
                    fields = issue.get("fields", {})

                    # Extract basic issue information
                    summary = fields.get("summary", "")
                    issue_type = fields.get("issuetype", {}).get("name", "")
                    status = fields.get("status", {}).get("name", "")
                    project_key = fields.get("project", {}).get("key", "")

                    # Create fully qualified source identifier
                    qualified_source = f"jira://{self.base_url}/{issue_key}"

                    # Create metadata
                    metadata = {
                        "issue_key": issue_key,
                        "summary": summary,
                        "project": project_key,
                        "issue_type": issue_type,
                        "status": status,
                        "url": f"{self.base_url}/browse/{issue_key}",
                        "content_type": "html"  # Explicitly mark as HTML content
                    }

                    # Add timestamps if available
                    if "created" in fields:
                        metadata["created"] = fields["created"]
                    if "updated" in fields:
                        metadata["updated"] = fields["updated"]

                    results.append({
                        "id": qualified_source,
                        "metadata": metadata,
                        "doc_type": "html"  # Also set document type for consistency
                    })

                # Update counters for next page
                total_fetched += len(issues)
                start_at += len(issues)

                # Check if we've reached the end
                if total_fetched >= search_data.get("total", 0):
                    break

            logger.info(f"Found {len(results)} JIRA issues")
            return results

        except Exception as e:
            logger.error(f"Error listing JIRA issues: {str(e)}")
            raise

    def has_changed(self, source_id: str, last_modified: Optional[float] = None) -> bool:
        """
        Check if a JIRA issue has changed since last processing.

        Args:
            source_id: Identifier for the issue
            last_modified: Timestamp of last known modification

        Returns:
            True if issue has changed, False otherwise
        """
        logger.debug(f"Checking if JIRA issue has changed: {source_id}")

        try:
            # Extract issue key
            issue_key = self._extract_issue_key(source_id)

            # If we have it in cache, check cache first
            if issue_key in self.content_cache:
                cache_entry = self.content_cache[issue_key]
                cache_metadata = cache_entry.get("metadata", {})

                if "updated" in cache_metadata:
                    cache_modified = self._parse_jira_timestamp(cache_metadata["updated"])

                    if cache_modified and last_modified and cache_modified <= last_modified:
                        logger.debug(f"Issue {issue_key} unchanged according to cache")
                        return False

            # Make API request to check updated timestamp
            api_url = f"{self.base_url}/rest/api/2/issue/{issue_key}"
            params = {"fields": "updated"}

            response = self.session.get(api_url, params=params)
            response.raise_for_status()
            issue_data = response.json()

            # Get current updated timestamp
            updated = issue_data.get("fields", {}).get("updated", "")
            current_timestamp = self._parse_jira_timestamp(updated)

            # Convert JIRA timestamp to epoch time for comparison
            if current_timestamp and last_modified:
                changed = current_timestamp > last_modified
                logger.debug(f"Issue {issue_key} changed: {changed}")
                return changed

            # If we can't determine based on timestamp, consider it changed
            return True

        except Exception as e:
            logger.error(f"Error checking changes for {source_id}: {str(e)}")
            return True

    def follow_links(self, content: str, source_id: str, current_depth: int = 0, global_visited_docs=None) -> List[
        Dict[str, Any]]:
        """
        Extract and follow links to other JIRA issues.

        Args:
            content: Document content (HTML format)
            source_id: Identifier for the source issue
            current_depth: Current depth of link following
            global_visited_docs: Global set of all visited issue IDs

        Returns:
            List of linked issues
        """
        if current_depth >= self.max_link_depth:
            logger.debug(f"Max link depth {self.max_link_depth} reached for {source_id}")
            return []

        # Initialize global visited set if not provided
        if global_visited_docs is None:
            global_visited_docs = set()

        # Add current issue to global visited set
        global_visited_docs.add(source_id)

        logger.debug(f"Following links in JIRA issue {source_id} at depth {current_depth}")

        linked_docs = []

        # Extract issue key from source_id
        issue_key = self._extract_issue_key(source_id)

        try:
            # First, fetch linked issues from JIRA API (more reliable than parsing HTML)
            linked_issues = self._fetch_linked_issues(issue_key)

            # Also try to parse HTML content for issue links using a simple HTML parser
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')

                # Find all anchor tags with href attributes
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']

                    # Look for JIRA issue keys in link text or href
                    # JIRA issue keys typically follow pattern: PROJECT-123
                    issue_key_match = re.search(r'([A-Z][A-Z0-9_]+)-\d+', href)
                    if not issue_key_match:
                        issue_key_match = re.search(r'([A-Z][A-Z0-9_]+)-\d+', a_tag.get_text())

                    if issue_key_match:
                        linked_key = issue_key_match.group(0)
                        linked_issues.add(linked_key)

            except Exception as e:
                logger.warning(f"Error parsing HTML for JIRA issue links: {str(e)}")

            # Process each linked issue
            for linked_key in linked_issues:
                # Skip if globally visited
                qualified_id = f"jira://{self.base_url}/{linked_key}"

                if qualified_id in global_visited_docs or linked_key in global_visited_docs:
                    logger.debug(f"Skipping globally visited issue: {linked_key}")
                    continue

                global_visited_docs.add(qualified_id)
                global_visited_docs.add(linked_key)

                try:
                    # Fetch the linked issue
                    linked_doc = self.fetch_document(linked_key)
                    linked_docs.append(linked_doc)
                    logger.debug(f"Successfully fetched linked issue: {linked_key}")

                    # Recursively follow links if not at max depth
                    if current_depth + 1 < self.max_link_depth:
                        logger.debug(f"Recursively following links from {linked_key} at depth {current_depth + 1}")
                        nested_docs = self.follow_links(
                            linked_doc["content"],
                            linked_doc["id"],
                            current_depth + 1,
                            global_visited_docs
                        )
                        linked_docs.extend(nested_docs)
                except Exception as e:
                    logger.warning(f"Error following link {linked_key} from {source_id}: {str(e)}")

            logger.debug(f"Completed following links from {source_id}: found {len(linked_docs)} linked issues")
            return linked_docs

        except Exception as e:
            logger.error(f"Error following links from JIRA issue {source_id}: {str(e)}")
            return []

    def _build_jql_query(self) -> str:
        """
        Build JQL query from configuration.

        Returns:
            JQL query string
        """
        # If JQL query is explicitly provided, use it
        if self.jql_query:
            return self.jql_query

        # Otherwise, build JQL from other configuration
        jql_parts = []

        # Add project filter
        if self.projects:
            project_clause = " OR ".join([f'"{p}"' for p in self.projects])
            jql_parts.append(f"project IN ({project_clause})")

        # Add issue type filter
        if self.issue_types:
            type_clause = " OR ".join([f'"{t}"' for t in self.issue_types])
            jql_parts.append(f"issuetype IN ({type_clause})")

        # Add status filter
        if self.statuses:
            status_clause = " OR ".join([f'"{s}"' for s in self.statuses])
            jql_parts.append(f"status IN ({status_clause})")
        elif not self.include_closed:
            jql_parts.append("status != Closed")

        # Combine all parts with AND
        jql = " AND ".join(jql_parts) if jql_parts else ""

        # Add ordering - newest first for consistency
        if jql:
            jql += " ORDER BY updated DESC"
        else:
            jql = "ORDER BY updated DESC"

        return jql

    def _fetch_comments(self, issue_key: str) -> str:
        """
        Fetch comments for a JIRA issue.

        Args:
            issue_key: JIRA issue key

        Returns:
            HTML content with comments
        """
        try:
            # Construct API URL
            api_url = f"{self.base_url}/rest/api/2/issue/{issue_key}/comment"

            # Make API request
            response = self.session.get(api_url)
            response.raise_for_status()
            comments_data = response.json()

            comments = comments_data.get("comments", [])
            if not comments:
                return ""

            # Build HTML content for comments
            html_content = "<div class='comments'>\n"

            for comment in comments:
                author = comment.get("author", {}).get("displayName", "Unknown")
                created = comment.get("created", "")
                comment_body = comment.get("body", "")

                # Convert created timestamp to readable format
                created_date = self._format_timestamp(created)

                # Add comment to HTML
                html_content += f"<div class='comment'>\n"
                html_content += f"<p class='comment-meta'><strong>{author}</strong> - {created_date}</p>\n"
                html_content += f"<div class='comment-body'>{comment_body}</div>\n"
                html_content += f"</div>\n"

            html_content += "</div>\n"
            return html_content

        except Exception as e:
            logger.warning(f"Error fetching comments for issue {issue_key}: {str(e)}")
            return ""

    def _fetch_custom_fields(self, issue_data: Dict[str, Any], rendered_fields: Dict[str, Any]) -> str:
        """
        Fetch specified custom fields from issue data.

        Args:
            issue_data: JIRA issue data
            rendered_fields: Rendered fields data

        Returns:
            HTML content with custom fields
        """
        if not self.include_custom_fields:
            return ""

        fields = issue_data.get("fields", {})
        names = issue_data.get("names", {})

        html_content = "<dl>\n"

        for field_id in self.include_custom_fields:
            # Skip if field not found
            if field_id not in fields:
                continue

            # Get field name and value
            field_name = names.get(field_id, field_id)
            field_value = fields.get(field_id)

            # Skip empty values
            if field_value is None or field_value == "":
                continue

            # Try to get rendered value for rich text fields
            rendered_value = rendered_fields.get(field_id)

            # Format value based on type
            if rendered_value:
                formatted_value = rendered_value
            elif isinstance(field_value, dict):
                # Handle complex field types (e.g., user, option)
                if "displayName" in field_value:
                    formatted_value = field_value["displayName"]
                elif "value" in field_value:
                    formatted_value = field_value["value"]
                else:
                    formatted_value = str(field_value)
            elif isinstance(field_value, list):
                # Handle array fields
                if all(isinstance(item, dict) for item in field_value):
                    # List of objects
                    if all("displayName" in item for item in field_value):
                        formatted_value = ", ".join(item["displayName"] for item in field_value)
                    elif all("value" in item for item in field_value):
                        formatted_value = ", ".join(item["value"] for item in field_value)
                    else:
                        formatted_value = ", ".join(str(item) for item in field_value)
                else:
                    # Simple array
                    formatted_value = ", ".join(str(item) for item in field_value)
            else:
                # Simple value
                formatted_value = str(field_value)

            # Add field to HTML
            html_content += f"<dt>{field_name}</dt>\n"
            html_content += f"<dd>{formatted_value}</dd>\n"

        html_content += "</dl>\n"

        # Return empty string if no fields were added
        if html_content == "<dl>\n</dl>\n":
            return ""

        return html_content

    def _fetch_linked_issues(self, issue_key: str) -> set:
        """
        Fetch linked issues for a JIRA issue.

        Args:
            issue_key: JIRA issue key

        Returns:
            Set of linked issue keys
        """
        linked_issues = set()

        try:
            # Construct API URL
            api_url = f"{self.base_url}/rest/api/2/issue/{issue_key}"
            params = {"fields": "issuelinks,subtasks"}

            # Make API request
            response = self.session.get(api_url, params=params)
            response.raise_for_status()
            issue_data = response.json()

            fields = issue_data.get("fields", {})

            # Process issue links
            if self.include_linked_issues:
                issue_links = fields.get("issuelinks", [])

                for link in issue_links:
                    # Check for inward links
                    if "inwardIssue" in link:
                        linked_key = link["inwardIssue"]["key"]
                        linked_issues.add(linked_key)

                    # Check for outward links
                    if "outwardIssue" in link:
                        linked_key = link["outwardIssue"]["key"]
                        linked_issues.add(linked_key)

            # Process subtasks
            if self.include_subtasks:
                subtasks = fields.get("subtasks", [])

                for subtask in subtasks:
                    subtask_key = subtask.get("key")
                    if subtask_key:
                        linked_issues.add(subtask_key)

            return linked_issues

        except Exception as e:
            logger.warning(f"Error fetching linked issues for {issue_key}: {str(e)}")
            return linked_issues

    @staticmethod
    def _extract_issue_key(source_id: str) -> str:
        """
        Extract JIRA issue key from source ID.

        Args:
            source_id: Source identifier

        Returns:
            JIRA issue key
        """
        # If source_id is already a valid issue key, return it directly
        if re.match(r'^[A-Z][A-Z0-9_]+-\d+$', source_id):
            return source_id

        # Try to extract from fully qualified source identifier
        # Pattern: jira://base_url/PROJ-123
        qualified_match = re.search(r'jira://[^/]+/([A-Z][A-Z0-9_]+-\d+)', source_id)
        if qualified_match:
            return qualified_match.group(1)

        # Try to extract from JIRA web URL
        # Pattern: /browse/PROJ-123
        url_match = re.search(r'/browse/([A-Z][A-Z0-9_]+-\d+)', source_id)
        if url_match:
            return url_match.group(1)

        # If no patterns match, return the original ID as is
        return source_id

    @staticmethod
    def _parse_jira_timestamp(timestamp: str) -> Optional[float]:
        """
        Parse JIRA timestamp into epoch time.

        Args:
            timestamp: JIRA timestamp string

        Returns:
            Timestamp as epoch time or None if parsing fails
        """
        if not timestamp:
            return None

        try:
            # JIRA uses ISO 8601 format timestamps
            # Example: 2023-05-01T12:34:56.789+0000
            from datetime import datetime
            import dateutil.parser

            dt = dateutil.parser.parse(timestamp)
            return dt.timestamp()
        except Exception:
            return None

    @staticmethod
    def _format_timestamp(timestamp: str) -> str:
        """
        Format timestamp into readable format.

        Args:
            timestamp: Timestamp string

        Returns:
            Formatted timestamp
        """
        if not timestamp:
            return ""

        try:
            from datetime import datetime
            import dateutil.parser

            dt = dateutil.parser.parse(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return timestamp
