"""
Google Drive Content Source for the document pointer system.

This module provides integration with Google Drive via the Google API Python Client.
"""

import io
import logging
import mimetypes
import os
import re
from typing import Dict, Any, List, Optional, TYPE_CHECKING, Tuple
from urllib.parse import urlparse, parse_qs

import time

from .base import ContentSource

# Import types for type checking only - these won't be imported at runtime
if TYPE_CHECKING:
    from googleapiclient.discovery import Resource
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2.credentials import Credentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from bs4 import BeautifulSoup
    import dateutil.parser
    from datetime import datetime

    # Define type aliases for type checking
    GoogleDriveServiceType = Resource
    GoogleCredentialsType = Credentials
    ServiceAccountCredentialsType = ServiceAccountCredentials
    InstalledAppFlowType = InstalledAppFlow
    GoogleAuthRequestType = GoogleAuthRequest
    MediaIoBaseDownloadType = MediaIoBaseDownload
    BeautifulSoupType = BeautifulSoup
    DateUtilParserType = dateutil.parser
    DatetimeType = datetime
else:
    # Runtime type aliases - use generic Python types
    GoogleDriveServiceType = Any
    GoogleCredentialsType = Any
    ServiceAccountCredentialsType = Any
    InstalledAppFlowType = Any
    GoogleAuthRequestType = Any
    MediaIoBaseDownloadType = Any
    BeautifulSoupType = Any
    DateUtilParserType = Any
    DatetimeType = Any

logger = logging.getLogger(__name__)

# Define global flags for availability - these will be set at runtime
GOOGLE_API_AVAILABLE = False
BS4_AVAILABLE = False
DATEUTIL_AVAILABLE = False

# Try to import Google API packages conditionally
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2.credentials import Credentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import pickle

    GOOGLE_API_AVAILABLE = True
except ImportError:
    logger.warning(
        "Google API packages not available. Install with 'pip install google-api-python-client google-auth-oauthlib google-auth-httplib2' to use Google Drive content source.")

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


class GoogleDriveContentSource(ContentSource):
    """Content source for Google Drive."""

    # Define Google Drive document MIME types and their corresponding formats
    GOOGLE_DOCUMENT_MIME_TYPES = {
        'application/vnd.google-apps.document': 'text/html',
        'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.google-apps.presentation': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'application/vnd.google-apps.drawing': 'image/png',
        'application/vnd.google-apps.script': 'application/json',
        'application/vnd.google-apps.form': 'text/html'
    }

    # File format extension map
    FILE_EXTENSIONS = {
        'application/vnd.google-apps.document': 'html',
        'application/vnd.google-apps.spreadsheet': 'xlsx',
        'application/vnd.google-apps.presentation': 'pptx',
        'application/vnd.google-apps.drawing': 'png',
        'application/vnd.google-apps.script': 'json',
        'application/vnd.google-apps.form': 'html',
        'text/html': 'html',
        'text/plain': 'txt',
        'application/pdf': 'pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
        'image/png': 'png',
        'image/jpeg': 'jpg',
        'application/json': 'json'
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Google Drive content source.

        Args:
            config: Configuration dictionary containing Google Drive connection details
        """
        if not GOOGLE_API_AVAILABLE:
            raise ImportError("Google API packages are required for GoogleDriveContentSource but not available")

        super().__init__(config)

        # Authentication settings
        self.auth_type = config.get("auth_type", "oauth")  # oauth or service_account
        self.token_path = config.get("token_path", "token.pickle")
        self.credentials_path = config.get("credentials_path", "credentials.json")
        self.service_account_file = config.get("service_account_file", "")
        self.impersonate_user = config.get("impersonate_user", "")

        # Google API settings
        self.scopes = config.get("scopes", ['https://www.googleapis.com/auth/drive.readonly'])

        # Content configuration
        self.include_shared = config.get("include_shared", True)
        self.include_trashed = config.get("include_trashed", False)
        self.folders = config.get("folders", [])
        self.file_types = config.get("file_types", [])
        self.include_patterns = config.get("include_patterns", [])
        self.exclude_patterns = config.get("exclude_patterns", [])
        self.include_google_docs = config.get("include_google_docs", True)
        self.max_results = config.get("max_results", 100)
        self.max_depth = config.get("max_depth", 5)

        # Link following configuration
        self.max_link_depth = config.get("max_link_depth", 3)

        # Allow export as different format
        self.export_format = config.get("export_format", "html")  # html, pdf, docx, etc.

        # Initialize Google Drive service
        self.drive_service: Optional[GoogleDriveServiceType] = None
        try:
            credentials = self._get_credentials()
            self.drive_service = build('drive', 'v3', credentials=credentials)
            logger.debug(f"Successfully initialized Google Drive service: {self.get_safe_connection_string()}")
        except Exception as e:
            logger.error(f"Error initializing Google Drive service: {str(e)}")
            raise

        # Cache for content
        self.content_cache = {}

    def get_safe_connection_string(self) -> str:
        """Return a safe version of the connection string with credentials masked."""
        if self.auth_type == "oauth":
            return f"Google Drive (OAuth) - {'with token' if os.path.exists(self.token_path) else 'no token'}"
        else:
            return f"Google Drive (Service Account) - {self.impersonate_user or 'no impersonation'}"

    def fetch_document(self, source_id: str) -> Dict[str, Any]:
        """
        Fetch document content from Google Drive.

        Args:
            source_id: Identifier for the content (file ID or URL)

        Returns:
            Dictionary containing document content and metadata

        Raises:
            ValueError: If Google Drive is not configured or document not found
        """
        if not self.drive_service:
            raise ValueError("Google Drive not configured")

        logger.debug(f"Fetching Google Drive content: {source_id}")

        try:
            # Extract file ID from source_id (could be a direct ID or URL)
            file_id = self._extract_file_id(source_id)

            # Get file metadata
            file_metadata = self.drive_service.files().get(
                fileId=file_id,
                fields="id,name,mimeType,description,createdTime,modifiedTime,size,webViewLink,parents,owners"
            ).execute()

            # Get file content based on MIME type
            mime_type = file_metadata.get('mimeType', '')
            content, is_binary = self._get_file_content(file_id, mime_type)

            # Determine document type
            doc_type = self._get_doc_type(mime_type)

            # Create a qualified source identifier
            qualified_source = f"gdrive://{file_id}"

            # Build metadata
            metadata = {
                "id": file_id,
                "name": file_metadata.get('name', ''),
                "mime_type": mime_type,
                "description": file_metadata.get('description', ''),
                "created_time": file_metadata.get('createdTime', ''),
                "modified_time": file_metadata.get('modifiedTime', ''),
                "size": file_metadata.get('size', '0'),
                "web_view_link": file_metadata.get('webViewLink', ''),
                "owners": [owner.get('displayName', '') for owner in file_metadata.get('owners', [])],
                "is_google_doc": mime_type.startswith('application/vnd.google-apps.'),
                "content_type": "binary" if is_binary else "text/html" if doc_type == "html" else "text/plain"
            }

            # Generate content hash for change detection
            content_hash = self.get_content_hash(str(content) if not is_binary else
                                                 str(file_metadata.get('size', '0')) + file_metadata.get('modifiedTime',
                                                                                                         ''))

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
            if "File not found" in str(e):
                raise ValueError(f"File not found: {source_id}")
            logger.error(f"Error fetching Google Drive content {source_id}: {str(e)}")
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List available documents in Google Drive.

        Returns:
            List of document identifiers and metadata

        Raises:
            ValueError: If Google Drive is not configured
        """
        if not self.drive_service:
            raise ValueError("Google Drive not configured")

        logger.debug("Listing Google Drive content")
        results = []

        try:
            # Build query to filter files
            query_parts = []

            # Set trashed filter
            query_parts.append(f"trashed = {str(self.include_trashed).lower()}")

            # Filter by file type if specified
            if self.file_types:
                mime_type_conditions = []
                for file_type in self.file_types:
                    if file_type.startswith('.'):  # Handle extension format
                        # Convert extension to MIME type
                        mime_type = mimetypes.guess_type(f"file{file_type}")[0]
                        if mime_type:
                            mime_type_conditions.append(f"mimeType = '{mime_type}'")
                    else:  # Assume it's already a MIME type
                        mime_type_conditions.append(f"mimeType = '{file_type}'")

                if mime_type_conditions:
                    query_parts.append(f"({' OR '.join(mime_type_conditions)})")

            # Include/exclude Google Docs
            if not self.include_google_docs:
                query_parts.append("not mimeType contains 'application/vnd.google-apps.'")

            # Filter by specific folders if provided
            if self.folders:
                folder_conditions = []
                for folder_id in self.folders:
                    # Clean up folder ID if it's a URL or qualified source ID
                    folder_id = self._extract_file_id(folder_id)
                    folder_conditions.append(f"'{folder_id}' in parents")

                if folder_conditions:
                    query_parts.append(f"({' OR '.join(folder_conditions)})")

            # Combine all query parts
            query = ' AND '.join(query_parts)
            logger.debug(f"Using query: {query}")

            # If no specific folders are provided and we're not filtering by type,
            # just get files from 'My Drive' (root) to avoid excessive results
            if not self.folders and not self.file_types and query == "trashed = false":
                query = "trashed = false AND 'root' in parents"

            # Fetch files
            files = []
            page_token = None

            while True:
                response = self.drive_service.files().list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType, description, createdTime, modifiedTime, size, webViewLink, parents, owners)",
                    pageToken=page_token,
                    pageSize=min(100, self.max_results)
                ).execute()

                files.extend(response.get('files', []))
                page_token = response.get('nextPageToken')

                if not page_token or len(files) >= self.max_results:
                    break

            # Apply include/exclude patterns
            for file in files:
                file_name = file.get('name', '')
                file_id = file.get('id', '')

                # Apply pattern filters if configured
                if not self._should_include_file(file_name, file_id):
                    logger.debug(f"Skipping excluded file: {file_name} ({file_id})")
                    continue

                # Create qualified source identifier
                qualified_source = f"gdrive://{file_id}"

                # Determine document type based on MIME type
                mime_type = file.get('mimeType', '')
                doc_type = self._get_doc_type(mime_type)

                results.append({
                    "id": qualified_source,
                    "metadata": {
                        "id": file_id,
                        "name": file_name,
                        "mime_type": mime_type,
                        "description": file.get('description', ''),
                        "created_time": file.get('createdTime', ''),
                        "modified_time": file.get('modifiedTime', ''),
                        "size": file.get('size', '0'),
                        "web_view_link": file.get('webViewLink', ''),
                        "owners": [owner.get('displayName', '') for owner in file.get('owners', [])],
                        "is_google_doc": mime_type.startswith('application/vnd.google-apps.'),
                        "content_type": "text/html" if doc_type == "html" else "application/octet-stream"
                    },
                    "doc_type": doc_type
                })

                # Apply limits if needed
                if len(results) >= self.max_results:
                    logger.debug(f"Reached limit of {self.max_results} documents")
                    break

            logger.info(f"Found {len(results)} Google Drive documents")
            return results

        except Exception as e:
            logger.error(f"Error listing Google Drive documents: {str(e)}")
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
        if not self.drive_service:
            # Can't determine changes without connection
            return True

        logger.debug(f"Checking if Google Drive content has changed: {source_id}")

        try:
            # Extract file ID
            file_id = self._extract_file_id(source_id)

            # If we have it in cache, check cache first
            if source_id in self.content_cache:
                cache_entry = self.content_cache[source_id]
                cache_metadata = cache_entry.get("metadata", {})

                if "modified_time" in cache_metadata:
                    cache_modified = self._parse_timestamp(cache_metadata["modified_time"])

                    if cache_modified and last_modified and cache_modified <= last_modified:
                        logger.debug(f"File {source_id} unchanged according to cache")
                        return False

            # Get file metadata from Google Drive
            file_metadata = self.drive_service.files().get(
                fileId=file_id,
                fields="modifiedTime"
            ).execute()

            # Get the current modification time
            current_modified_time = file_metadata.get("modifiedTime", "")
            current_timestamp = self._parse_timestamp(current_modified_time)

            # Compare timestamps
            if current_timestamp and last_modified:
                changed = current_timestamp > last_modified
                logger.debug(f"File {source_id} changed: {changed}")
                return changed

            # If we can't determine based on timestamp, consider it changed
            return True

        except Exception as e:
            logger.error(f"Error checking changes for {source_id}: {str(e)}")
            return True  # Assume changed if there's an error

    def follow_links(self, content: str, source_id: str, current_depth: int = 0,
                     global_visited_docs=None) -> List[Dict[str, Any]]:
        """
        Extract and follow links to other Google Drive documents.

        Args:
            content: Document content (HTML for Google Docs, text for other files)
            source_id: Identifier for the source document
            current_depth: Current depth of link following
            global_visited_docs: Global set of all visited document IDs

        Returns:
            List of linked documents

        Raises:
            ValueError: If Google Drive is not configured
        """
        if not self.drive_service:
            raise ValueError("Google Drive not configured")

        if current_depth >= self.max_link_depth:
            logger.debug(f"Max link depth {self.max_link_depth} reached for {source_id}")
            return []

        # Initialize global visited set if not provided
        if global_visited_docs is None:
            global_visited_docs = set()

        # Add current document to global visited set
        global_visited_docs.add(source_id)

        logger.debug(f"Following links in Google Drive content {source_id} at depth {current_depth}")

        linked_docs = []
        found_file_ids = set()

        # Extract links from content (if it's HTML)
        if BS4_AVAILABLE and content and isinstance(content, str):
            try:
                soup = BeautifulSoup(content, 'html.parser')

                # Find all anchor tags with href attributes
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']

                    # Try to extract Google Drive file ID from href
                    file_id = self._extract_file_id_from_url(href)
                    if file_id:
                        found_file_ids.add(file_id)

            except Exception as e:
                logger.warning(f"Error parsing HTML for links: {str(e)}")

        # Use regex to find Google Drive links as a fallback or supplement
        gdrive_patterns = [
            # Direct file ID in URL
            r'drive\.google\.com/(?:file/d/|open\?id=)([a-zA-Z0-9_-]{25,})',
            # Drive links
            r'drive\.google\.com/drive/(?:u/\d+/)?(?:folders|file|my-drive)/([a-zA-Z0-9_-]{25,})',
            # Docs links
            r'docs\.google\.com/(?:document|spreadsheet|presentation)/d/([a-zA-Z0-9_-]{25,})'
        ]

        if isinstance(content, str):
            for pattern in gdrive_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    found_file_ids.add(match)

        # Process each unique linked document
        for file_id in found_file_ids:
            # Create qualified source ID
            qualified_id = f"gdrive://{file_id}"

            # Skip if globally visited
            if qualified_id in global_visited_docs or file_id in global_visited_docs:
                logger.debug(f"Skipping globally visited link: {file_id}")
                continue

            global_visited_docs.add(qualified_id)
            global_visited_docs.add(file_id)

            try:
                # Fetch the linked document
                linked_doc = self.fetch_document(file_id)
                linked_docs.append(linked_doc)
                logger.debug(f"Successfully fetched linked document: {file_id}")

                # Recursively follow links if not at max depth
                if current_depth + 1 < self.max_link_depth:
                    logger.debug(f"Recursively following links from {file_id} at depth {current_depth + 1}")
                    nested_docs = self.follow_links(
                        linked_doc.get("content", ""),
                        linked_doc["id"],
                        current_depth + 1,
                        global_visited_docs
                    )
                    linked_docs.extend(nested_docs)
            except Exception as e:
                logger.warning(f"Error following link {file_id} from {source_id}: {str(e)}")

        logger.debug(f"Completed following links from {source_id}: found {len(linked_docs)} linked documents")
        return linked_docs

    def _get_credentials(self) -> GoogleCredentialsType:
        """
        Get Google API credentials based on configuration.

        Returns:
            Google API credentials

        Raises:
            ValueError: If credentials cannot be obtained
        """
        if self.auth_type == "oauth":
            return self._get_oauth_credentials()
        elif self.auth_type == "service_account":
            return self._get_service_account_credentials()
        else:
            raise ValueError(f"Unsupported authentication type: {self.auth_type}")

    def _get_oauth_credentials(self) -> GoogleCredentialsType:
        """
        Get OAuth credentials for Google API.

        Returns:
            OAuth credentials

        Raises:
            ValueError: If OAuth credentials cannot be obtained
        """
        credentials = None

        # Check if token exists
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                credentials = pickle.load(token)

        # Check if credentials are valid or need refreshing
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                # No valid credentials, need to create new ones
                if not os.path.exists(self.credentials_path):
                    raise ValueError(f"Credentials file not found: {self.credentials_path}")

                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.scopes)
                credentials = flow.run_local_server(port=0)

                # Save the credentials for future use
                with open(self.token_path, 'wb') as token:
                    pickle.dump(credentials, token)

        return credentials

    def _get_service_account_credentials(self) -> ServiceAccountCredentialsType:
        """
        Get service account credentials for Google API.

        Returns:
            Service account credentials

        Raises:
            ValueError: If service account credentials cannot be obtained
        """
        if not os.path.exists(self.service_account_file):
            raise ValueError(f"Service account file not found: {self.service_account_file}")

        # Create credentials from service account file
        credentials = ServiceAccountCredentials.from_service_account_file(
            self.service_account_file,
            scopes=self.scopes
        )

        # Impersonate a user if specified
        if self.impersonate_user:
            return credentials.with_subject(self.impersonate_user)

        return credentials

    def _get_file_content(self, file_id: str, mime_type: str) -> Tuple[Any, bool]:
        """
        Get file content from Google Drive.

        Args:
            file_id: Google Drive file ID
            mime_type: MIME type of the file

        Returns:
            Tuple of (content, is_binary)
        """
        # Handle Google Docs (Google's native formats)
        if mime_type in self.GOOGLE_DOCUMENT_MIME_TYPES:
            export_mime_type = self.GOOGLE_DOCUMENT_MIME_TYPES[mime_type]

            # Get the content as the specified format
            response = self.drive_service.files().export(
                fileId=file_id,
                mimeType=export_mime_type
            ).execute()

            # For HTML content, return as string
            if export_mime_type == 'text/html':
                return response.decode('utf-8'), False

            # For text formats, try to decode
            if export_mime_type in ['text/plain', 'application/json']:
                try:
                    return response.decode('utf-8'), False
                except UnicodeDecodeError:
                    return response, True

            # For binary formats
            return response, True

        # Handle regular files
        request = self.drive_service.files().get_media(fileId=file_id)
        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        file_content.seek(0)
        content = file_content.read()

        # Try to decode if it might be text
        if mime_type in ['text/plain', 'text/html', 'application/json', 'text/csv']:
            try:
                return content.decode('utf-8'), False
            except UnicodeDecodeError:
                return content, True

        return content, True

    def _get_doc_type(self, mime_type: str) -> str:
        """
        Determine document type based on MIME type.

        Args:
            mime_type: MIME type of the file

        Returns:
            Document type string
        """
        if mime_type == 'application/vnd.google-apps.document':
            return 'html'
        elif mime_type == 'application/vnd.google-apps.spreadsheet':
            return 'xlsx'
        elif mime_type == 'application/vnd.google-apps.presentation':
            return 'pptx'
        elif mime_type == 'application/vnd.google-apps.form':
            return 'html'
        elif mime_type == 'application/vnd.google-apps.script':
            return 'json'
        elif mime_type == 'text/html':
            return 'html'
        elif mime_type == 'text/plain':
            return 'text'
        elif mime_type == 'application/pdf':
            return 'pdf'
        elif mime_type.startswith('image/'):
            return mime_type.split('/')[1]  # Use image format (png, jpeg, etc.)
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return 'docx'
        elif mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            return 'xlsx'
        elif mime_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
            return 'pptx'
        else:
            # Default to binary for unknown types
            return 'binary'

    def _extract_file_id(self, source_id: str) -> str:
        """
        Extract Google Drive file ID from source ID.

        Args:
            source_id: Source identifier (could be ID, URL, or qualified source ID)

        Returns:
            Google Drive file ID
        """
        # If it's already a valid file ID format (25+ alphanumeric characters with hyphens and underscores)
        if re.match(r'^[a-zA-Z0-9_-]{25,}$', source_id):
            return source_id

        # If it's a qualified source ID
        qualified_match = re.match(r'^gdrive://([a-zA-Z0-9_-]{25,})$', source_id)
        if qualified_match:
            return qualified_match.group(1)

        # If it's a Google Drive URL
        file_id = self._extract_file_id_from_url(source_id)
        if file_id:
            return file_id

        # If no pattern matches, return the original ID as is (might be invalid)
        return source_id

    def _extract_file_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract Google Drive file ID from a URL.

        Args:
            url: Google Drive URL

        Returns:
            File ID or None if not found
        """
        # Handle direct file links
        file_match = re.search(r'drive\.google\.com/(?:file/d/|open\?id=)([a-zA-Z0-9_-]{25,})', url)
        if file_match:
            return file_match.group(1)

        # Handle folder links
        folder_match = re.search(r'drive\.google\.com/drive/(?:u/\d+/)?(?:folders|file|my-drive)/([a-zA-Z0-9_-]{25,})',
                                 url)
        if folder_match:
            return folder_match.group(1)

        # Handle Google Docs links
        docs_match = re.search(r'docs\.google\.com/(?:document|spreadsheet|presentation)/d/([a-zA-Z0-9_-]{25,})', url)
        if docs_match:
            return docs_match.group(1)

        # Handle URLs with id parameter
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        if 'id' in query_params:
            return query_params['id'][0]

        return None

    def _should_include_file(self, file_name: str, file_id: str) -> bool:
        """
        Check if file should be included based on configured patterns.

        Args:
            file_name: File name
            file_id: File ID

        Returns:
            True if file should be included, False otherwise
        """
        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if re.search(pattern, file_name) or re.search(pattern, file_id):
                logger.debug(f"File {file_id} excluded by pattern: {pattern}")
                return False

        # If no include patterns are configured, include everything not excluded
        if not self.include_patterns:
            return True

        # Check include patterns
        for pattern in self.include_patterns:
            if re.search(pattern, file_name) or re.search(pattern, file_id):
                logger.debug(f"File {file_id} included by pattern: {pattern}")
                return True

        # If include patterns are configured and none matched, exclude
        return False

    @staticmethod
    def _parse_timestamp(timestamp_str: str) -> Optional[float]:
        """
        Parse Google Drive timestamp into epoch time.

        Args:
            timestamp_str: Google Drive timestamp string (ISO 8601 format)

        Returns:
            Timestamp as epoch time or None if parsing fails
        """
        if not timestamp_str:
            return None

        try:
            # Google Drive uses ISO 8601 format timestamps
            if DATEUTIL_AVAILABLE:
                dt = dateutil.parser.parse(timestamp_str)
                return dt.timestamp()
            else:
                # Fallback for when dateutil is not available
                from datetime import datetime

                # Parse ISO 8601 format (might not handle all edge cases)
                if 'Z' in timestamp_str:
                    timestamp_str = timestamp_str.replace('Z', '+00:00')

                # Try to parse the timestamp
                dt = datetime.fromisoformat(timestamp_str)
                return dt.timestamp()
        except Exception:
            logger.warning(f"Could not parse timestamp: {timestamp_str}")
            return None

    def __del__(self):
        """Clean up resources when object is deleted."""
        logger.debug("Closing Google Drive service")
