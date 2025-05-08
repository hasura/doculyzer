"""
SharePoint Content Source for the document pointer system.

This module provides integration with Microsoft SharePoint via the Office365-REST-Python-Client library.
"""

import logging
import re
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from urllib.parse import urljoin, urlparse

import time

from .base import ContentSource

# Import types for type checking only - these won't be imported at runtime
if TYPE_CHECKING:
    from office365.runtime.auth.user_credential import UserCredential
    from office365.runtime.auth.client_credential import ClientCredential
    from office365.runtime.auth.authentication_context import AuthenticationContext
    from office365.sharepoint.client_context import ClientContext
    from office365.sharepoint.files.file import File
    from office365.sharepoint.folders.folder import Folder
    from office365.sharepoint.lists.list import List
    from office365.sharepoint.lists.list_item import ListItem
    from office365.sharepoint.sites.site import Site
    from office365.sharepoint.listitems.listitem_collection import ListItemCollection
    from bs4 import BeautifulSoup
    import dateutil.parser
    from datetime import datetime

    # Define type aliases for type checking
    ClientContextType = ClientContext
    UserCredentialType = UserCredential
    ClientCredentialType = ClientCredential
    AuthenticationContextType = AuthenticationContext
    SharePointFileType = File
    SharePointFolderType = Folder
    SharePointListType = List
    SharePointListItemType = ListItem
    SharePointSiteType = Site
    SharePointListItemCollectionType = ListItemCollection
    BeautifulSoupType = BeautifulSoup
    DateUtilParserType = dateutil.parser
    DatetimeType = datetime
else:
    # Runtime type aliases - use generic Python types
    ClientContextType = Any
    UserCredentialType = Any
    ClientCredentialType = Any
    AuthenticationContextType = Any
    SharePointFileType = Any
    SharePointFolderType = Any
    SharePointListType = Any
    SharePointListItemType = Any
    SharePointSiteType = Any
    SharePointListItemCollectionType = Any
    BeautifulSoupType = Any
    DateUtilParserType = Any
    DatetimeType = Any

logger = logging.getLogger(__name__)

# Define global flags for availability - these will be set at runtime
OFFICE365_AVAILABLE = False
BS4_AVAILABLE = False
DATEUTIL_AVAILABLE = False

# Try to import Office365 REST Python Client conditionally
try:
    from office365.runtime.auth.user_credential import UserCredential
    from office365.runtime.auth.client_credential import ClientCredential
    from office365.runtime.auth.authentication_context import AuthenticationContext
    from office365.sharepoint.client_context import ClientContext

    OFFICE365_AVAILABLE = True
except ImportError:
    logger.warning(
        "Office365-REST-Python-Client not available. Install with 'pip install Office365-REST-Python-Client' to use SharePoint content source.")

# Try to import BeautifulSoup conditionally
try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    logger.warning("beautifulsoup4 not available. Install with 'pip install beautifulsoup4' for improved HTML parsing.")

# Try to import dateutil conditionally
try:
    import dateutil.parser
    from datetime import datetime

    DATEUTIL_AVAILABLE = True
except ImportError:
    logger.warning(
        "python-dateutil not available. Install with 'pip install python-dateutil' for improved date handling.")


class SharePointContentSource(ContentSource):
    """Content source for Microsoft SharePoint."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SharePoint content source.

        Args:
            config: Configuration dictionary containing SharePoint connection details
        """
        if not OFFICE365_AVAILABLE:
            raise ImportError("Office365-REST-Python-Client is required for SharePointContentSource but not available")

        super().__init__(config)

        # SharePoint connection settings
        self.site_url = config.get("site_url", "").rstrip('/')

        # Authentication settings
        self.auth_type = config.get("auth_type", "user_credentials")  # user_credentials or client_credentials
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.client_id = config.get("client_id", "")
        self.client_secret = config.get("client_secret", "")
        self.tenant = config.get("tenant", "")

        # Content configuration
        self.libraries = config.get("libraries", [])  # Document libraries to include
        self.lists = config.get("lists", [])  # Lists to include
        self.sites = config.get("sites", [])  # Subsites to include
        self.include_subfolders = config.get("include_subfolders", True)
        self.exclude_system_libraries = config.get("exclude_system_libraries", True)
        self.include_patterns = config.get("include_patterns", [])
        self.exclude_patterns = config.get("exclude_patterns", [])
        self.file_extensions = config.get("file_extensions", [])  # Extensions to include, empty means all
        self.max_items = config.get("max_items", 100)
        self.include_list_items = config.get("include_list_items", True)
        self.include_versioning = config.get("include_versioning", False)
        self.include_permissions = config.get("include_permissions", False)

        # Link following configuration
        self.max_link_depth = config.get("max_link_depth", 3)

        # Initialize SharePoint client context
        self.ctx: Optional[ClientContextType] = None
        try:
            self.ctx = self._initialize_client_context()
            logger.debug(f"Successfully initialized client context for SharePoint: {self.get_safe_connection_string()}")
        except Exception as e:
            logger.error(f"Error initializing SharePoint client context: {str(e)}")
            raise

        # Cache for content
        self.content_cache = {}

    def get_safe_connection_string(self) -> str:
        """Return a safe version of the connection string with credentials masked."""
        if not self.site_url:
            return "<no site URL>"

        # Only show the site URL without credentials
        return self.site_url

    def fetch_document(self, source_id: str) -> Dict[str, Any]:
        """
        Fetch document content from SharePoint.

        Args:
            source_id: Identifier for the content (usually file path or document ID)

        Returns:
            Dictionary containing document content and metadata

        Raises:
            ValueError: If SharePoint is not configured or document not found
        """
        if not self.ctx:
            raise ValueError("SharePoint not configured")

        logger.debug(f"Fetching SharePoint content: {source_id}")

        try:
            # Extract document info from source_id (could be a path, URL, or ID)
            doc_info = self._extract_document_info(source_id)
            doc_type = doc_info.get("type", "file")

            if doc_type == "file":
                return self._fetch_file(doc_info, source_id)
            elif doc_type == "list_item":
                return self._fetch_list_item(doc_info, source_id)
            else:
                raise ValueError(f"Unsupported document type: {doc_type}")

        except Exception as e:
            logger.error(f"Error fetching SharePoint content {source_id}: {str(e)}")
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List available documents in SharePoint.

        Returns:
            List of document identifiers and metadata

        Raises:
            ValueError: If SharePoint is not configured
        """
        if not self.ctx:
            raise ValueError("SharePoint not configured")

        logger.debug("Listing SharePoint content")
        results = []

        try:
            # Get documents from document libraries
            if self.libraries:
                for library in self.libraries:
                    logger.debug(f"Processing document library: {library}")
                    library_docs = self._list_library_documents(library)
                    results.extend(library_docs)

                    # Apply limits if needed
                    if len(results) >= self.max_items:
                        logger.debug(f"Reached limit of {self.max_items} documents")
                        results = results[:self.max_items]
                        return results

            # Get items from lists
            if self.include_list_items and self.lists:
                for list_name in self.lists:
                    logger.debug(f"Processing list: {list_name}")
                    list_items = self._list_list_items(list_name)
                    results.extend(list_items)

                    # Apply limits if needed
                    if len(results) >= self.max_items:
                        logger.debug(f"Reached limit of {self.max_items} documents")
                        results = results[:self.max_items]
                        return results

            # If no specific libraries or lists are configured, get all document libraries
            if not self.libraries and not self.lists:
                logger.debug("No specific libraries or lists configured, fetching all document libraries")
                all_docs = self._list_all_documents()
                results.extend(all_docs)

                # Apply limits if needed
                if len(results) >= self.max_items:
                    logger.debug(f"Reached limit of {self.max_items} documents")
                    results = results[:self.max_items]
                    return results

            logger.info(f"Found {len(results)} SharePoint documents")
            return results

        except Exception as e:
            logger.error(f"Error listing SharePoint documents: {str(e)}")
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
        if not self.ctx:
            # Can't determine changes without connection
            return True

        logger.debug(f"Checking if SharePoint content has changed: {source_id}")

        try:
            # Extract document info
            doc_info = self._extract_document_info(source_id)
            doc_type = doc_info.get("type", "file")

            # If we have it in cache, check cache first
            cache_key = source_id
            if cache_key in self.content_cache:
                cache_entry = self.content_cache[cache_key]
                cache_metadata = cache_entry.get("metadata", {})

                if "last_modified" in cache_metadata:
                    cache_modified = self._parse_sharepoint_timestamp(cache_metadata["last_modified"])

                    if cache_modified and last_modified and cache_modified <= last_modified:
                        logger.debug(f"Content {source_id} unchanged according to cache")
                        return False

            # Need to check directly with SharePoint
            if doc_type == "file":
                # Get the file path
                server_relative_url = doc_info.get("server_relative_url")
                if not server_relative_url:
                    return True

                # Get file properties
                file = self.ctx.web.get_file_by_server_relative_url(server_relative_url)
                file.get_property("TimeLastModified").execute_query()

                # Parse the timestamp
                current_modified = file.properties.get("TimeLastModified", "")
                current_timestamp = self._parse_sharepoint_timestamp(current_modified)

            elif doc_type == "list_item":
                # Get the list and item ID
                list_title = doc_info.get("list_title")
                item_id = doc_info.get("item_id")

                if not list_title or not item_id:
                    return True

                # Get list item properties
                item = self.ctx.web.lists.get_by_title(list_title).items.get_by_id(item_id)
                item.get_property("Modified").execute_query()

                # Parse the timestamp
                current_modified = item.properties.get("Modified", "")
                current_timestamp = self._parse_sharepoint_timestamp(current_modified)

            else:
                # Unknown document type, consider it changed
                return True

            # Compare timestamps
            if current_timestamp and last_modified:
                changed = current_timestamp > last_modified
                logger.debug(f"Content {source_id} changed: {changed}")
                return changed

            # If we can't determine based on timestamp, consider it changed
            return True

        except Exception as e:
            logger.error(f"Error checking changes for {source_id}: {str(e)}")
            return True  # Assume changed if there's an error

    def follow_links(self, content: str, source_id: str, current_depth: int = 0,
                     global_visited_docs=None) -> List[Dict[str, Any]]:
        """
        Extract and follow links in SharePoint content.

        Args:
            content: Document content (usually HTML or text)
            source_id: Identifier for the source document
            current_depth: Current depth of link following
            global_visited_docs: Global set of all visited document IDs

        Returns:
            List of linked documents

        Raises:
            ValueError: If SharePoint is not configured
        """
        if not self.ctx:
            raise ValueError("SharePoint not configured")

        if current_depth >= self.max_link_depth:
            logger.debug(f"Max link depth {self.max_link_depth} reached for {source_id}")
            return []

        # Initialize global visited set if not provided
        if global_visited_docs is None:
            global_visited_docs = set()

        # Add current document to global visited set
        global_visited_docs.add(source_id)

        logger.debug(f"Following links in SharePoint content {source_id} at depth {current_depth}")

        linked_docs = []
        found_links = set()

        # Extract document info from source_id
        doc_info = self._extract_document_info(source_id)
        site_url = self.site_url

        # Extract links from content (if it's HTML)
        if BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(content, 'html.parser')

                # Find all anchor tags with href attributes
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']

                    # Only process internal SharePoint links
                    if not self._is_sharepoint_link(href):
                        continue

                    # Normalize the URL
                    normalized_url = self._normalize_sharepoint_url(href, site_url)
                    if normalized_url:
                        found_links.add(normalized_url)

            except Exception as e:
                logger.warning(f"Error parsing HTML for SharePoint links: {str(e)}")

        # Use regex to find SharePoint links as a fallback or supplement
        sharepoint_patterns = [
            # Document links
            r'\/([^\/]+\/)*([^\/]+\.(docx|xlsx|pptx|pdf|txt|html|aspx))(?:\?|#|$)',
            # List item links
            r'\/Lists\/([^\/]+)\/DispForm\.aspx\?ID=(\d+)',
            # Page links
            r'\/SitePages\/([^\/]+\.aspx)(?:\?|#|$)',
        ]

        for pattern in sharepoint_patterns:
            matches = re.findall(pattern, content)

            for match in matches:
                # Convert match to string if it's a tuple (from capturing groups)
                if isinstance(match, tuple):
                    path_component = ''.join(match)
                else:
                    path_component = match

                # Create a relative URL
                relative_url = f"/{path_component}" if not path_component.startswith('/') else path_component

                # Normalize the URL
                normalized_url = self._normalize_sharepoint_url(relative_url, site_url)
                if normalized_url:
                    found_links.add(normalized_url)

        # Process each unique linked document
        for link in found_links:
            # Skip if globally visited
            if link in global_visited_docs:
                logger.debug(f"Skipping globally visited link: {link}")
                continue

            global_visited_docs.add(link)

            try:
                # Fetch the linked document
                linked_doc = self.fetch_document(link)
                linked_docs.append(linked_doc)
                logger.debug(f"Successfully fetched linked document: {link}")

                # Recursively follow links if not at max depth
                if current_depth + 1 < self.max_link_depth:
                    logger.debug(f"Recursively following links from {link} at depth {current_depth + 1}")
                    nested_docs = self.follow_links(
                        linked_doc["content"],
                        linked_doc["id"],
                        current_depth + 1,
                        global_visited_docs
                    )
                    linked_docs.extend(nested_docs)
            except Exception as e:
                logger.warning(f"Error following link {link} from {source_id}: {str(e)}")

        logger.debug(f"Completed following links from {source_id}: found {len(linked_docs)} linked documents")
        return linked_docs

    def _initialize_client_context(self) -> ClientContextType:
        """
        Initialize the SharePoint client context with appropriate authentication.

        Returns:
            ClientContext object for SharePoint API access

        Raises:
            ValueError: If required authentication parameters are missing
        """
        if self.auth_type == "user_credentials":
            if not self.username or not self.password:
                raise ValueError("Username and password are required for user credentials authentication")

            # Create user credentials
            user_credentials = UserCredential(self.username, self.password)

            # Create client context
            return ClientContext(self.site_url).with_credentials(user_credentials)

        elif self.auth_type == "client_credentials":
            if not self.client_id or not self.client_secret or not self.tenant:
                raise ValueError(
                    "Client ID, Client Secret, and Tenant are required for client credentials authentication")

            # Create authentication context
            auth_ctx = AuthenticationContext(self.tenant)

            # Acquire token
            client_credentials = ClientCredential(self.client_id, self.client_secret)
            auth_ctx.acquire_token_for_app(client_id=self.client_id, client_secret=self.client_secret)

            # Create client context
            return ClientContext(self.site_url).with_credentials(auth_ctx)

        else:
            raise ValueError(f"Unsupported authentication type: {self.auth_type}")

    def _fetch_file(self, doc_info: Dict[str, Any], source_id: str) -> Dict[str, Any]:
        """
        Fetch a file document from SharePoint.

        Args:
            doc_info: Document information dictionary
            source_id: Original source ID

        Returns:
            Dictionary with document content and metadata

        Raises:
            ValueError: If file not found
        """
        # Get the file path
        server_relative_url = doc_info.get("server_relative_url")
        if not server_relative_url:
            raise ValueError(f"Invalid file path for document: {source_id}")

        # Get the file content
        try:
            file = self.ctx.web.get_file_by_server_relative_url(server_relative_url)
            file.get_property("Name")
            file.get_property("Title")
            file.get_property("TimeLastModified")
            file.get_property("Length")
            file.get_property("ServerRelativeUrl")
            file.get_property("UIVersionLabel")

            # Execute the query
            self.ctx.execute_query()

            # Get file content as text (try to decode if possible)
            response = file.read()
            self.ctx.execute_query()

            # Try to decode as UTF-8 text, but keep as bytes if not possible
            content = response.value
            is_binary = True

            try:
                content = content.decode('utf-8')
                is_binary = False
            except UnicodeDecodeError:
                pass

            # Determine content type based on file extension
            file_extension = doc_info.get("extension", "").lower()

            if file_extension in ['html', 'htm', 'aspx']:
                doc_type = "html"
            elif file_extension in ['docx', 'doc']:
                doc_type = "docx"
            elif file_extension in ['xlsx', 'xls']:
                doc_type = "xlsx"
            elif file_extension in ['pptx', 'ppt']:
                doc_type = "pptx"
            elif file_extension in ['pdf']:
                doc_type = "pdf"
            elif file_extension in ['txt', 'text', 'md', 'markdown']:
                doc_type = "text"
            else:
                doc_type = "binary"

            # Get file properties
            file_name = file.properties.get("Name", "")
            file_title = file.properties.get("Title", "")
            last_modified = file.properties.get("TimeLastModified", "")
            file_size = file.properties.get("Length", 0)
            version = file.properties.get("UIVersionLabel", "1.0")

            # Create a qualified source identifier
            qualified_source = f"sharepoint://{self.site_url}/{server_relative_url}"

            # Build file URL
            site_uri = urlparse(self.site_url)
            file_url = f"{site_uri.scheme}://{site_uri.netloc}{server_relative_url}"

            # Build metadata
            metadata = {
                "name": file_name,
                "title": file_title or file_name,
                "server_relative_url": server_relative_url,
                "url": file_url,
                "last_modified": last_modified,
                "size": file_size,
                "version": version,
                "extension": file_extension,
                "is_binary": is_binary,
                "content_type": "binary" if is_binary else "text/html" if doc_type == "html" else "text/plain"
            }

            # Generate content hash for change detection
            content_hash = self.get_content_hash(str(content) if not is_binary else str(file_size) + last_modified)

            # Cache the content
            self.content_cache[source_id] = {
                "content": content,
                "metadata": metadata,
                "hash": content_hash,
                "last_accessed": time.time()
            }

            return {
                "id": qualified_source,
                "content": content if not is_binary else "",
                "binary_content": content if is_binary else None,
                "doc_type": doc_type,
                "metadata": metadata,
                "content_hash": content_hash
            }

        except Exception as e:
            if "File Not Found" in str(e):
                raise ValueError(f"File not found: {server_relative_url}")
            raise

    def _fetch_list_item(self, doc_info: Dict[str, Any], source_id: str) -> Dict[str, Any]:
        """
        Fetch a list item from SharePoint.

        Args:
            doc_info: Document information dictionary
            source_id: Original source ID

        Returns:
            Dictionary with document content and metadata

        Raises:
            ValueError: If list item not found
        """
        # Get the list title and item ID
        list_title = doc_info.get("list_title")
        item_id = doc_info.get("item_id")

        if not list_title or not item_id:
            raise ValueError(f"Invalid list item information for document: {source_id}")

        # Get the list item
        try:
            # Get the SharePoint list
            sp_list = self.ctx.web.lists.get_by_title(list_title)

            # Get the list item by ID
            item = sp_list.items.get_by_id(item_id)
            item.expand(["FieldValuesAsText", "Modified", "Created", "Author", "Editor"])
            self.ctx.load(item)
            self.ctx.execute_query()

            # Get item data
            item_data = item.properties
            text_values = item.properties.get("FieldValuesAsText", {})

            # Build HTML content for the list item
            html_content = f"<h1>{text_values.get('Title', '')}</h1>\n<dl>\n"

            # Add all fields to the HTML content
            for field, value in text_values.items():
                if field not in ["Title", "ContentType", "Modified", "Created", "Author", "Editor", "ID", "GUID"]:
                    html_content += f"<dt>{field}</dt>\n<dd>{value}</dd>\n"

            html_content += "</dl>\n"

            # Add metadata fields
            html_content += "<h2>Item Information</h2>\n<dl>\n"
            html_content += f"<dt>Created</dt>\n<dd>{text_values.get('Created', '')}</dd>\n"
            html_content += f"<dt>Modified</dt>\n<dd>{text_values.get('Modified', '')}</dd>\n"
            html_content += f"<dt>Created By</dt>\n<dd>{text_values.get('Author', '')}</dd>\n"
            html_content += f"<dt>Modified By</dt>\n<dd>{text_values.get('Editor', '')}</dd>\n"
            html_content += "</dl>\n"

            # Build URL for the item
            site_uri = urlparse(self.site_url)
            item_url = f"{site_uri.scheme}://{site_uri.netloc}{site_uri.path}/Lists/{list_title}/DispForm.aspx?ID={item_id}"

            # Create a qualified source identifier
            qualified_source = f"sharepoint://{self.site_url}/Lists/{list_title}/Items/{item_id}"

            # Build metadata
            metadata = {
                "title": text_values.get("Title", f"List Item {item_id}"),
                "list": list_title,
                "id": item_id,
                "url": item_url,
                "last_modified": item_data.get("Modified", ""),
                "created": item_data.get("Created", ""),
                "content_type": "text/html"
            }

            # Convert item data to JSON string for content hash
            content_hash = self.get_content_hash(html_content)

            # Cache the content
            self.content_cache[source_id] = {
                "content": html_content,
                "metadata": metadata,
                "hash": content_hash,
                "last_accessed": time.time()
            }

            return {
                "id": qualified_source,
                "content": html_content,
                "doc_type": "html",
                "metadata": metadata,
                "content_hash": content_hash
            }

        except Exception as e:
            if "Item does not exist" in str(e):
                raise ValueError(f"List item not found: {list_title}/{item_id}")
            raise

    def _list_library_documents(self, library_name: str) -> List[Dict[str, Any]]:
        """
        List documents in a specific SharePoint document library.

        Args:
            library_name: Name of the document library

        Returns:
            List of document identifiers and metadata
        """
        results = []

        try:
            # Get the document library
            library = self.ctx.web.lists.get_by_title(library_name)
            root_folder = library.root_folder
            self.ctx.load(root_folder)
            self.ctx.execute_query()

            # Get the server relative URL of the root folder
            root_url = root_folder.properties.get("ServerRelativeUrl", "")

            # Process files and folders recursively
            files = self._process_folder(root_url, include_subfolders=self.include_subfolders)

            # Filter files if needed
            for file_info in files:
                file_name = file_info.get("Name", "")
                file_path = file_info.get("ServerRelativeUrl", "")

                # Apply pattern filters if configured
                if not self._should_include_document(file_name, file_path):
                    logger.debug(f"Skipping excluded file: {file_path}")
                    continue

                # Create qualified source identifier
                qualified_source = f"sharepoint://{self.site_url}/{file_path}"

                # Build file URL
                site_uri = urlparse(self.site_url)
                file_url = f"{site_uri.scheme}://{site_uri.netloc}{file_path}"

                # Get file extension
                file_extension = ""
                if "." in file_name:
                    file_extension = file_name.split(".")[-1].lower()

                # Filter by extension if configured
                if self.file_extensions and file_extension not in self.file_extensions:
                    logger.debug(f"Skipping file with excluded extension: {file_path}")
                    continue

                # Determine document type based on extension
                if file_extension in ['html', 'htm', 'aspx']:
                    doc_type = "html"
                elif file_extension in ['docx', 'doc']:
                    doc_type = "docx"
                elif file_extension in ['xlsx', 'xls']:
                    doc_type = "xlsx"
                elif file_extension in ['pptx', 'ppt']:
                    doc_type = "pptx"
                elif file_extension in ['pdf']:
                    doc_type = "pdf"
                elif file_extension in ['txt', 'text', 'md', 'markdown']:
                    doc_type = "text"
                else:
                    doc_type = "binary"

                results.append({
                    "id": qualified_source,
                    "metadata": {
                        "name": file_name,
                        "title": file_info.get("Title", file_name),
                        "server_relative_url": file_path,
                        "url": file_url,
                        "last_modified": file_info.get("TimeLastModified", ""),
                        "size": file_info.get("Length", 0),
                        "extension": file_extension,
                        "content_type": "text/html" if doc_type == "html" else "application/octet-stream"
                    },
                    "doc_type": doc_type
                })

            return results

        except Exception as e:
            logger.error(f"Error listing documents in library {library_name}: {str(e)}")
            return []

    def _list_list_items(self, list_name: str) -> List[Dict[str, Any]]:
        """
        List items in a specific SharePoint list.

        Args:
            list_name: Name of the list

        Returns:
            List of list item identifiers and metadata
        """
        results = []

        try:
            # Get the SharePoint list
            sp_list = self.ctx.web.lists.get_by_title(list_name)

            # Query list items
            items = sp_list.items.top(self.max_items)
            items.select(["ID", "Title", "Modified", "Created"])
            self.ctx.load(items)
            self.ctx.execute_query()

            # Build URL for the list
            site_uri = urlparse(self.site_url)
            list_url = f"{site_uri.scheme}://{site_uri.netloc}{site_uri.path}/Lists/{list_name}"

            # Process each item
            for item in items:
                item_id = item.properties.get("ID")
                title = item.properties.get("Title", f"Item {item_id}")

                # Apply pattern filters if configured
                if not self._should_include_document(title, f"Lists/{list_name}/{item_id}"):
                    logger.debug(f"Skipping excluded list item: {title} ({item_id})")
                    continue

                # Create qualified source identifier
                qualified_source = f"sharepoint://{self.site_url}/Lists/{list_name}/Items/{item_id}"

                # Build item URL
                item_url = f"{list_url}/DispForm.aspx?ID={item_id}"

                results.append({
                    "id": qualified_source,
                    "metadata": {
                        "title": title,
                        "list": list_name,
                        "id": item_id,
                        "url": item_url,
                        "last_modified": item.properties.get("Modified", ""),
                        "created": item.properties.get("Created", ""),
                        "content_type": "text/html"
                    },
                    "doc_type": "html"
                })

            return results

        except Exception as e:
            logger.error(f"Error listing items in list {list_name}: {str(e)}")
            return []

    def _list_all_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in SharePoint site.

        Returns:
            List of document identifiers and metadata
        """
        results = []

        try:
            # Get all lists in the site
            lists = self.ctx.web.lists
            lists.expand(["BaseTemplate", "Title"])
            self.ctx.load(lists)
            self.ctx.execute_query()

            # Process document libraries (BaseTemplate = 101) and lists (BaseTemplate = 100)
            for sp_list in lists:
                base_template = sp_list.properties.get("BaseTemplate", 0)
                title = sp_list.properties.get("Title", "")

                # Skip system libraries if configured
                if self.exclude_system_libraries and self._is_system_library(title):
                    logger.debug(f"Skipping system library: {title}")
                    continue

                if base_template == 101:  # Document library
                    logger.debug(f"Processing document library: {title}")
                    library_docs = self._list_library_documents(title)
                    results.extend(library_docs)

                elif base_template == 100 and self.include_list_items:  # Generic list
                    logger.debug(f"Processing list: {title}")
                    list_items = self._list_list_items(title)
                    results.extend(list_items)

                # Apply limits if needed
                if len(results) >= self.max_items:
                    logger.debug(f"Reached limit of {self.max_items} documents")
                    results = results[:self.max_items]
                    break

            return results

        except Exception as e:
            logger.error(f"Error listing all documents: {str(e)}")
            return []

    def _process_folder(self, folder_url: str, include_subfolders: bool = True) -> List[Dict[str, Any]]:
        """
        Process a folder to get all files.

        Args:
            folder_url: Server-relative URL of the folder
            include_subfolders: Whether to include subfolders recursively

        Returns:
            List of file information
        """
        results = []

        try:
            # Get the folder
            folder = self.ctx.web.get_folder_by_server_relative_url(folder_url)
            folder_content = folder.files
            folders = folder.folders if include_subfolders else None

            # Load files
            self.ctx.load(folder_content, ["ServerRelativeUrl", "Name", "Title", "TimeLastModified", "Length"])
            if include_subfolders:
                self.ctx.load(folders, ["ServerRelativeUrl", "Name"])

            self.ctx.execute_query()

            # Process files in this folder
            for file in folder_content:
                file_properties = {
                    "ServerRelativeUrl": file.properties.get("ServerRelativeUrl", ""),
                    "Name": file.properties.get("Name", ""),
                    "Title": file.properties.get("Title", ""),
                    "TimeLastModified": file.properties.get("TimeLastModified", ""),
                    "Length": file.properties.get("Length", 0)
                }
                results.append(file_properties)

            # Process subfolders if needed
            if include_subfolders:
                for subfolder in folders:
                    subfolder_url = subfolder.properties.get("ServerRelativeUrl", "")
                    subfolder_files = self._process_folder(subfolder_url, include_subfolders)
                    results.extend(subfolder_files)

            return results

        except Exception as e:
            logger.error(f"Error processing folder {folder_url}: {str(e)}")
            return []

    def _extract_document_info(self, source_id: str) -> Dict[str, Any]:
        """
        Extract document information from source ID.

        Args:
            source_id: Source identifier

        Returns:
            Dictionary with document information
        """
        # Check if it's a list item URL pattern
        list_item_match = re.search(r'(?:sharepoint://[^/]+)?/Lists/([^/]+)/(?:Items/)?(\d+)', source_id)
        if list_item_match:
            return {
                "type": "list_item",
                "list_title": list_item_match.group(1),
                "item_id": int(list_item_match.group(2))
            }

        # Check if it's a document library file
        file_path_match = re.search(r'(?:sharepoint://[^/]+)?(/.*)', source_id)
        if file_path_match:
            server_relative_url = file_path_match.group(1)

            # Extract file extension
            file_extension = ""
            if "." in server_relative_url:
                file_extension = server_relative_url.split(".")[-1].lower()

            return {
                "type": "file",
                "server_relative_url": server_relative_url,
                "extension": file_extension
            }

        # Default to treating it as a file path
        return {
            "type": "file",
            "server_relative_url": source_id
        }

    def _should_include_document(self, name: str, path: str) -> bool:
        """
        Check if document should be included based on configured patterns.

        Args:
            name: Document name
            path: Document path

        Returns:
            True if document should be included, False otherwise
        """
        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if re.search(pattern, name) or re.search(pattern, path):
                logger.debug(f"Document {path} excluded by pattern: {pattern}")
                return False

        # If no include patterns are configured, include everything not excluded
        if not self.include_patterns:
            return True

        # Check include patterns
        for pattern in self.include_patterns:
            if re.search(pattern, name) or re.search(pattern, path):
                logger.debug(f"Document {path} included by pattern: {pattern}")
                return True

        # If include patterns are configured and none matched, exclude
        return False

    def _is_system_library(self, library_name: str) -> bool:
        """
        Check if a library is a system library.

        Args:
            library_name: Library name

        Returns:
            True if it's a system library, False otherwise
        """
        system_libraries = [
            "Form Templates",
            "Style Library",
            "Site Assets",
            "Site Pages",
            "Composed Looks",
            "Master Page Gallery",
            "Web Part Gallery",
            "Site Collection Documents",
            "Site Collection Images",
            "Theme Gallery"
        ]

        return library_name in system_libraries

    def _is_sharepoint_link(self, url: str) -> bool:
        """
        Check if a URL is an internal SharePoint link.

        Args:
            url: URL to check

        Returns:
            True if it's a SharePoint link, False otherwise
        """
        # Skip empty links, anchors, and javascript links
        if not url or url.startswith('#') or url.startswith('javascript:'):
            return False

        # Check if it's an absolute URL to the same site
        if url.startswith(('http://', 'https://')):
            site_uri = urlparse(self.site_url)
            url_uri = urlparse(url)

            # Check if it's the same hostname
            if url_uri.netloc != site_uri.netloc:
                return False

        # Check for SharePoint-specific patterns
        sharepoint_patterns = [
            r'\.aspx(?:\?|$)',  # SharePoint page
            r'/Lists/',  # SharePoint list
            r'/Documents/',  # SharePoint document library
            r'/Shared%20Documents/',  # SharePoint document library (encoded)
            r'/SitePages/',  # SharePoint site pages
            r'/_layouts/'  # SharePoint layouts
        ]

        for pattern in sharepoint_patterns:
            if re.search(pattern, url):
                return True

        return True  # Default to treating it as a SharePoint link if it's not an external URL

    def _normalize_sharepoint_url(self, url: str, site_url: str) -> Optional[str]:
        """
        Normalize a SharePoint URL to a standard format.

        Args:
            url: URL to normalize
            site_url: Base site URL

        Returns:
            Normalized URL or None if invalid
        """
        # Handle empty or invalid URLs
        if not url:
            return None

        # Handle absolute URLs
        if url.startswith(('http://', 'https://')):
            url_uri = urlparse(url)
            site_uri = urlparse(site_url)

            # Check if it's the same host
            if url_uri.netloc != site_uri.netloc:
                return None  # External link

            # Extract server-relative path
            path = url_uri.path

            # Create qualified SharePoint URL
            return f"sharepoint://{site_url}{path}"

        # Handle server-relative URLs
        elif url.startswith('/'):
            return f"sharepoint://{site_url}{url}"

        # Handle relative URLs
        else:
            # Need to resolve relative to current page
            # This is an approximation since we don't have the current page context
            site_uri = urlparse(site_url)
            resolved_path = urljoin(site_uri.path + '/', url)

            return f"sharepoint://{site_url}{resolved_path}"

    @staticmethod
    def _parse_sharepoint_timestamp(timestamp: str) -> Optional[float]:
        """
        Parse SharePoint timestamp into epoch time.

        Args:
            timestamp: SharePoint timestamp string

        Returns:
            Timestamp as epoch time or None if parsing fails
        """
        if not timestamp:
            return None

        try:
            # SharePoint uses various formats, most commonly ISO 8601
            if DATEUTIL_AVAILABLE:
                dt = dateutil.parser.parse(timestamp)
                return dt.timestamp()
            else:
                # Fallback for when dateutil is not available
                from datetime import datetime

                # Try some common SharePoint formats
                formats = [
                    "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601 UTC
                    "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO 8601 UTC with milliseconds
                    "%Y-%m-%d %H:%M:%S",  # Simple format without timezone
                    "%m/%d/%Y %H:%M:%S"  # US date format
                ]

                for fmt in formats:
                    try:
                        dt = datetime.strptime(timestamp, fmt)
                        return dt.timestamp()
                    except ValueError:
                        continue

                # If none of the formats match, try a last-ditch approach
                return float(timestamp) if timestamp.isdigit() else None

        except Exception:
            logger.warning(f"Could not parse timestamp: {timestamp}")
            return None

    def __del__(self):
        """Clean up resources when object is deleted."""
        # There's no explicit cleanup needed for the Office365 client
        # but we'll log that we're closing the connection
        logger.debug("Closing SharePoint connection")
