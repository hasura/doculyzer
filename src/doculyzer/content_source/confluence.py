"""
Confluence Content Source for the document pointer system.

This module provides integration with Atlassian Confluence via its REST API.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin

import requests
import time

from .base import ContentSource

logger = logging.getLogger(__name__)


class ConfluenceContentSource(ContentSource):
    """Content source for Atlassian Confluence."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Confluence content source.

        Args:
            config: Configuration dictionary containing Confluence connection details
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "").rstrip('/')
        self.username = config.get("username", "")
        self.api_token = config.get("api_token", "")
        self.password = config.get("password", "")
        self.spaces = config.get("spaces", [])
        self.include_pages = config.get("include_pages", True)
        self.include_blogs = config.get("include_blogs", False)
        self.include_comments = config.get("include_comments", False)
        self.include_attachments = config.get("include_attachments", False)
        self.exclude_personal_spaces = config.get("exclude_personal_spaces", True)
        self.include_patterns = config.get("include_patterns", [])
        self.exclude_patterns = config.get("exclude_patterns", [])
        self.expand_macros = config.get("expand_macros", True)
        self.limit = config.get("limit", 100)
        self.link_pattern = config.get("link_pattern", r'/wiki/spaces/[^/]+/pages/(\d+)')

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
        Fetch document content from Confluence.

        Args:
            source_id: Identifier for the content (usually content ID)

        Returns:
            Dictionary containing document content and metadata
        """
        logger.debug(f"Fetching Confluence content: {source_id}")

        try:
            # Extract content ID if source_id is a URL or complex identifier
            content_id = self._extract_content_id(source_id)

            # Construct API URL
            api_url = f"{self.base_url}/rest/api/content/{content_id}"
            params = {
                "expand": "body.storage,version,metadata,history,space"
            }

            # Make API request
            response = self.session.get(api_url, params=params)
            response.raise_for_status()
            content_data = response.json()

            # Extract content details
            title = content_data.get("title", "")
            space_key = content_data.get("space", {}).get("key", "")
            content_type = content_data.get("type", "")
            version = content_data.get("version", {}).get("number", 1)
            last_modified = content_data.get("version", {}).get("when", "")

            # Get HTML content
            html_content = content_data.get("body", {}).get("storage", {}).get("value", "")

            # Create fully qualified source identifier
            qualified_source = f"confluence://{self.base_url}/{space_key}/{content_id}"

            # Construct metadata
            metadata = {
                "title": title,
                "space": space_key,
                "type": content_type,
                "version": version,
                "last_modified": last_modified,
                "url": f"{self.base_url}/wiki/spaces/{space_key}/pages/{content_id}",
                "api_url": api_url,
                "content_type": "html"  # Explicitly mark as HTML content
            }

            # Get the content's web URL
            # web_url = f"{self.base_url}/wiki/spaces/{space_key}/pages/{content_id}"
            # if "webui" in content_data.get("_links", {}):
            #     web_url = urljoin(self.base_url, content_data["_links"]["webui"])

            # Generate content hash for change detection
            content_hash = self.get_content_hash(html_content)

            # Cache the content for faster access
            self.content_cache[content_id] = {
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
            logger.error(f"Error fetching Confluence content {source_id}: {str(e)}")
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List available documents in Confluence.

        Returns:
            List of document identifiers and metadata
        """
        logger.debug("Listing Confluence content")
        results = []

        try:
            # Process specific spaces if configured
            spaces_to_process = self._get_spaces_to_process()

            for space in spaces_to_process:
                logger.debug(f"Processing space: {space}")

                # Get content in this space
                space_content = self._get_content_in_space(space)
                results.extend(space_content)

                # Apply limits if needed
                if len(results) >= self.limit:
                    logger.debug(f"Reached limit of {self.limit} documents")
                    results = results[:self.limit]
                    break

            logger.info(f"Found {len(results)} Confluence documents")
            return results

        except Exception as e:
            logger.error(f"Error listing Confluence documents: {str(e)}")
            raise

    def has_changed(self, source_id: str, last_modified: Optional[float] = None) -> bool:
        """
        Check if a document has changed since last processing.

        Args:
            source_id: Identifier for the document
            last_modified: Timestamp of last known modification

        Returns:
            True if document has changed, False otherwise
        """
        logger.debug(f"Checking if Confluence content has changed: {source_id}")

        try:
            # Extract content ID
            content_id = self._extract_content_id(source_id)

            # If we have it in cache, check cache first
            if content_id in self.content_cache:
                cache_entry = self.content_cache[content_id]
                cache_modified = self._parse_confluence_timestamp(cache_entry["metadata"].get("last_modified", ""))

                if cache_modified and last_modified and cache_modified <= last_modified:
                    logger.debug(f"Content {content_id} unchanged according to cache")
                    return False

            # Make API request to check version
            api_url = f"{self.base_url}/rest/api/content/{content_id}"
            params = {"expand": "version"}

            response = self.session.get(api_url, params=params)
            response.raise_for_status()
            content_data = response.json()

            # Get current version information
            # current_version = content_data.get("version", {}).get("number", 0)
            current_modified = content_data.get("version", {}).get("when", "")
            current_timestamp = self._parse_confluence_timestamp(current_modified)

            # Convert Confluence timestamp to epoch time for comparison
            if current_timestamp and last_modified:
                changed = current_timestamp > last_modified
                logger.debug(f"Content {content_id} changed: {changed}")
                return changed

            # If we can't determine based on timestamp, consider it changed
            return True

        except Exception as e:
            logger.error(f"Error checking changes for {source_id}: {str(e)}")
            return True

    def follow_links(self, content: str, source_id: str, current_depth: int = 0, global_visited_docs=None) -> List[
        Dict[str, Any]]:
        """
        Extract and follow links in Confluence content.

        Args:
            content: Document content (HTML format)
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

        logger.debug(f"Following links in Confluence content {source_id} at depth {current_depth}")

        linked_docs = []

        # Extract content ID and space key from source_id
        content_id = self._extract_content_id(source_id)
        space_key = None

        # If we have cache entry, get space from metadata
        if content_id in self.content_cache:
            space_key = self.content_cache[content_id]["metadata"].get("space")
        else:
            # Try to extract space from source_id
            match = re.search(r'confluence://[^/]+/([^/]+)/', source_id)
            if match:
                space_key = match.group(1)

        # Use both regex-based extraction and HTML parsing for links
        # First, try to parse HTML content for links using a simple HTML parser
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')

            # Find all anchor tags with href attributes
            found_content_ids = set()
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']

                # Try to extract Confluence page ID from the href
                match = re.search(r'/pages/(\d+)', href)
                if match:
                    content_link_id = match.group(1)
                    found_content_ids.add(content_link_id)

                # Also check for pageId parameter
                match = re.search(r'pageId=(\d+)', href)
                if match:
                    content_link_id = match.group(1)
                    found_content_ids.add(content_link_id)

        except Exception as e:
            logger.warning(f"Error parsing HTML for links: {str(e)}, falling back to regex patterns")
            # found_content_ids = set()

            # Find all Confluence page links in content using regex
            # Look for various patterns that could indicate Confluence links
            patterns = [
                # HTML links to Confluence pages
                r'href="(?:https?://[^/]+)?/wiki/spaces/[^/]+/pages/(\d+)[^"]*"',
                # Plain URLs to Confluence pages
                r'(?:https?://[^/]+)?/wiki/spaces/[^/]+/pages/(\d+)',
                # Relative page links
                r'pages/(\d+)',
                # Page IDs in various formats
                r'pageId=(\d+)'
            ]

        found_content_ids = set()

        for pattern in patterns:
            matches = re.findall(pattern, content)

            for match in matches:
                if isinstance(match, tuple):
                    # Some patterns might have multiple groups
                    content_link_id = match[-1]  # Last group should be the ID
                else:
                    content_link_id = match

                # Add to unique set of found IDs
                if content_link_id and content_link_id.isdigit():
                    found_content_ids.add(content_link_id)

        # Process each unique linked document
        for linked_id in found_content_ids:
            # Skip if globally visited
            qualified_id = f"confluence://{self.base_url}/{space_key}/{linked_id}" if space_key else linked_id

            if qualified_id in global_visited_docs or linked_id in global_visited_docs:
                logger.debug(f"Skipping globally visited link: {linked_id}")
                continue

            global_visited_docs.add(qualified_id)
            global_visited_docs.add(linked_id)

            try:
                # Fetch the linked document
                linked_doc = self.fetch_document(linked_id)
                linked_docs.append(linked_doc)
                logger.debug(f"Successfully fetched linked document: {linked_id}")

                # Recursively follow links if not at max depth
                if current_depth + 1 < self.max_link_depth:
                    logger.debug(f"Recursively following links from {linked_id} at depth {current_depth + 1}")
                    nested_docs = self.follow_links(
                        linked_doc["content"],
                        linked_doc["id"],
                        current_depth + 1,
                        global_visited_docs
                    )
                    linked_docs.extend(nested_docs)
            except Exception as e:
                logger.warning(f"Error following link {linked_id} from {source_id}: {str(e)}")

        logger.debug(f"Completed following links from {source_id}: found {len(linked_docs)} linked documents")
        return linked_docs

    def _get_spaces_to_process(self) -> List[str]:
        """
        Get list of spaces to process.

        Returns:
            List of space keys
        """
        # If spaces are explicitly configured, use those
        if self.spaces:
            return self.spaces

        # Otherwise, fetch spaces from Confluence
        spaces = []

        try:
            logger.debug("Fetching available spaces from Confluence")
            api_url = f"{self.base_url}/rest/api/space"

            # Set up parameters
            params = {
                "limit": 100,
                "expand": "description.plain,metadata"
            }

            # Make API request
            response = self.session.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Process results
            for space in data.get("results", []):
                space_key = space.get("key", "")
                space_type = space.get("type", "")

                # Skip personal spaces if configured to do so
                if self.exclude_personal_spaces and space_type == "personal":
                    logger.debug(f"Skipping personal space: {space_key}")
                    continue

                spaces.append(space_key)

            logger.debug(f"Found {len(spaces)} spaces to process")
            return spaces

        except Exception as e:
            logger.error(f"Error fetching Confluence spaces: {str(e)}")
            # Return empty list on error
            return []

    def _get_content_in_space(self, space_key: str) -> List[Dict[str, Any]]:
        """
        Get content in a specific Confluence space.

        Args:
            space_key: Space key to fetch content from

        Returns:
            List of content items
        """
        logger.debug(f"Fetching content in space: {space_key}")
        content_list = []

        try:
            # Determine content types to fetch
            content_types = []
            if self.include_pages:
                content_types.append("page")
            if self.include_blogs:
                content_types.append("blogpost")

            if not content_types:
                logger.debug(f"No content types configured for space {space_key}")
                return []

            # Process each content type
            for content_type in content_types:
                logger.debug(f"Fetching {content_type} content in space {space_key}")

                # Set up API request
                api_url = f"{self.base_url}/rest/api/content"
                params = {
                    "spaceKey": space_key,
                    "type": content_type,
                    "status": "current",
                    "expand": "version",
                    "limit": 50  # Fetch in batches
                }

                # Make initial request
                response = self.session.get(api_url, params=params)
                response.raise_for_status()
                data = response.json()

                # Process results and handle pagination
                while True:
                    # Add results to list
                    for item in data.get("results", []):
                        content_id = item.get("id", "")
                        title = item.get("title", "")

                        # Apply pattern filters if configured
                        if not self._should_include_content(title, content_id):
                            logger.debug(f"Skipping excluded content: {title} ({content_id})")
                            continue

                        # Create fully qualified source identifier
                        qualified_source = f"confluence://{self.base_url}/{space_key}/{content_id}"

                        # Get basic metadata
                        version = item.get("version", {}).get("number", 1)
                        last_modified = item.get("version", {}).get("when", "")

                        content_list.append({
                            "id": qualified_source,
                            "metadata": {
                                "title": title,
                                "space": space_key,
                                "type": content_type,
                                "version": version,
                                "last_modified": last_modified,
                                "url": f"{self.base_url}/wiki/spaces/{space_key}/pages/{content_id}",
                                "content_type": "html"  # Explicitly mark as HTML content
                            },
                            "doc_type": "html"  # Also set document type for consistency
                        })

                    # Check if there are more results
                    next_page_url = data.get("_links", {}).get("next")
                    if not next_page_url:
                        break

                    # Make next request
                    next_url = urljoin(self.base_url, next_page_url)
                    response = self.session.get(next_url)
                    response.raise_for_status()
                    data = response.json()

            logger.debug(f"Found {len(content_list)} content items in space {space_key}")
            return content_list

        except Exception as e:
            logger.error(f"Error fetching content in space {space_key}: {str(e)}")
            return []

    def _should_include_content(self, title: str, content_id: str) -> bool:
        """
        Check if content should be included based on configured patterns.

        Args:
            title: Content title
            content_id: Content ID

        Returns:
            True if content should be included, False otherwise
        """
        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if re.search(pattern, title) or re.search(pattern, content_id):
                logger.debug(f"Content {content_id} excluded by pattern: {pattern}")
                return False

        # If no include patterns are configured, include everything not excluded
        if not self.include_patterns:
            return True

        # Check include patterns
        for pattern in self.include_patterns:
            if re.search(pattern, title) or re.search(pattern, content_id):
                logger.debug(f"Content {content_id} included by pattern: {pattern}")
                return True

        # If include patterns are configured and none matched, exclude
        return False

    @staticmethod
    def _extract_content_id(source_id: str) -> str:
        """
        Extract Confluence content ID from source ID.

        Args:
            source_id: Source identifier

        Returns:
            Confluence content ID
        """
        # If source_id is just a numeric ID, return it directly
        if source_id.isdigit():
            return source_id

        # Try to extract ID from fully qualified source identifier
        # Pattern: confluence://base_url/space_key/content_id
        qualified_match = re.search(r'confluence://[^/]+/[^/]+/(\d+)', source_id)
        if qualified_match:
            return qualified_match.group(1)

        # Try to extract ID from Confluence web URL
        # Pattern: /wiki/spaces/space_key/pages/content_id
        url_match = re.search(r'/wiki/spaces/[^/]+/pages/(\d+)', source_id)
        if url_match:
            return url_match.group(1)

        # Other patterns can be added as needed

        # If no pattern matches, return the original ID as is
        return source_id

    @staticmethod
    def _parse_confluence_timestamp(timestamp: str) -> Optional[float]:
        """
        Parse Confluence timestamp into epoch time.

        Args:
            timestamp: Confluence timestamp string

        Returns:
            Timestamp as epoch time or None if parsing fails
        """
        if not timestamp:
            return None

        try:
            # Confluence uses ISO 8601 format timestamps
            # Example: 2023-05-01T12:34:56.789Z
            from datetime import datetime
            import dateutil.parser

            dt = dateutil.parser.parse(timestamp)
            return dt.timestamp()
        except Exception:
            return None
