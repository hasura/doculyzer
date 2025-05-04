"""
ServiceNow Content Source for the document pointer system.

This module provides integration with ServiceNow via its REST APIs.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional

import requests
import time

from .base import ContentSource

logger = logging.getLogger(__name__)


class ServiceNowContentSource(ContentSource):
    """Content source for ServiceNow."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ServiceNow content source.

        Args:
            config: Configuration dictionary containing ServiceNow connection details
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "").rstrip('/')
        self.username = config.get("username", "")
        self.api_token = config.get("api_token", "")
        self.password = config.get("password", "")

        # Content type settings
        self.include_knowledge = config.get("include_knowledge", True)
        self.include_incidents = config.get("include_incidents", False)
        self.include_service_catalog = config.get("include_service_catalog", False)
        self.include_cmdb = config.get("include_cmdb", False)

        # Filter settings
        self.knowledge_query = config.get("knowledge_query", "")
        self.incident_query = config.get("incident_query", "")
        self.service_catalog_query = config.get("service_catalog_query", "")
        self.cmdb_query = config.get("cmdb_query", "")
        self.include_patterns = config.get("include_patterns", [])
        self.exclude_patterns = config.get("exclude_patterns", [])
        self.limit = config.get("limit", 100)

        # API paths
        self.table_api_path = "/api/now/table"
        self.knowledge_api_path = "/api/sn_km_api/knowledge"

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

        # Add common headers
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json"
        })

        # Cache for content
        self.content_cache = {}

    def fetch_document(self, source_id: str) -> Dict[str, Any]:
        """
        Fetch document content from ServiceNow.

        Args:
            source_id: Identifier for the content

        Returns:
            Dictionary containing document content and metadata
        """
        logger.debug(f"Fetching ServiceNow content: {source_id}")

        try:
            # Parse the source ID to determine what type of content to fetch
            content_type, item_id = self._parse_source_id(source_id)

            # Handle different content types
            if content_type == "knowledge":
                return self._fetch_knowledge_article(item_id, source_id)
            elif content_type == "incident":
                return self._fetch_incident(item_id, source_id)
            elif content_type == "catalog_item":
                return self._fetch_catalog_item(item_id, source_id)
            elif content_type == "cmdb":
                return self._fetch_cmdb_item(item_id, source_id)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")

        except Exception as e:
            logger.error(f"Error fetching ServiceNow content {source_id}: {str(e)}")
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List available documents in ServiceNow.

        Returns:
            List of document identifiers and metadata
        """
        logger.debug("Listing ServiceNow content")
        results = []

        try:
            # Fetch knowledge articles if enabled
            if self.include_knowledge:
                logger.debug("Fetching knowledge articles")
                knowledge_articles = self._list_knowledge_articles()
                results.extend(knowledge_articles)

                # Apply limits if needed
                if len(results) >= self.limit:
                    logger.debug(f"Reached limit of {self.limit} documents")
                    results = results[:self.limit]
                    return results

            # Fetch incidents if enabled
            if self.include_incidents:
                logger.debug("Fetching incidents")
                incidents = self._list_incidents()
                results.extend(incidents)

                # Apply limits if needed
                if len(results) >= self.limit:
                    logger.debug(f"Reached limit of {self.limit} documents")
                    results = results[:self.limit]
                    return results

            # Fetch service catalog items if enabled
            if self.include_service_catalog:
                logger.debug("Fetching service catalog items")
                catalog_items = self._list_catalog_items()
                results.extend(catalog_items)

                # Apply limits if needed
                if len(results) >= self.limit:
                    logger.debug(f"Reached limit of {self.limit} documents")
                    results = results[:self.limit]
                    return results

            # Fetch CMDB items if enabled
            if self.include_cmdb:
                logger.debug("Fetching CMDB items")
                cmdb_items = self._list_cmdb_items()
                results.extend(cmdb_items)

                # Apply limits if needed
                if len(results) >= self.limit:
                    logger.debug(f"Reached limit of {self.limit} documents")
                    results = results[:self.limit]
                    return results

            logger.info(f"Found {len(results)} ServiceNow documents")
            return results

        except Exception as e:
            logger.error(f"Error listing ServiceNow documents: {str(e)}")
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
        logger.debug(f"Checking if ServiceNow content has changed: {source_id}")

        try:
            # Parse the source ID
            content_type, item_id = self._parse_source_id(source_id)

            # If we have it in cache, check cache first
            if item_id in self.content_cache:
                cache_entry = self.content_cache[item_id]
                cache_modified = self._parse_servicenow_timestamp(cache_entry.get("last_modified", ""))

                if cache_modified and last_modified and cache_modified <= last_modified:
                    logger.debug(f"Content {item_id} unchanged according to cache")
                    return False

            # Make API request to check last modified date
            if content_type == "knowledge":
                api_url = f"{self.base_url}{self.knowledge_api_path}/articles/{item_id}"
                field_name = "sys_updated_on"
            else:
                # For other types, use the table API
                table_name = {
                    "incident": "incident",
                    "catalog_item": "sc_cat_item",
                    "cmdb": "cmdb_ci"
                }.get(content_type)

                api_url = f"{self.base_url}{self.table_api_path}/{table_name}/{item_id}"
                field_name = "sys_updated_on"

            # Make the request
            response = self.session.get(api_url)
            response.raise_for_status()

            # Get the response data
            data = response.json()

            # Extract last modified timestamp
            if content_type == "knowledge":
                current_modified = data.get("result", {}).get(field_name)
            else:
                current_modified = data.get("result", {}).get(field_name)

            # Parse the timestamp
            current_timestamp = self._parse_servicenow_timestamp(current_modified)

            # Compare timestamps
            if current_timestamp and last_modified:
                changed = current_timestamp > last_modified
                logger.debug(f"Content {item_id} changed: {changed}")
                return changed

            # If we can't determine based on timestamp, consider it changed
            return True

        except Exception as e:
            logger.error(f"Error checking changes for {source_id}: {str(e)}")
            return True

    def follow_links(self, content: str, source_id: str, current_depth: int = 0, global_visited_docs=None) -> List[
        Dict[str, Any]]:
        """
        Extract and follow links in ServiceNow content.

        Args:
            content: Document content
            source_id: Identifier for the source document
            current_depth: Current depth of link following
            global_visited_docs: Global set of all visited document IDs

        Returns:
            List of linked documents
        """
        # Check if we've reached the maximum link depth
        if current_depth >= self.max_link_depth:
            logger.debug(f"Max link depth {self.max_link_depth} reached for {source_id}")
            return []

        # Initialize global visited set if not provided
        if global_visited_docs is None:
            global_visited_docs = set()

        # Add current document to global visited set
        global_visited_docs.add(source_id)

        logger.debug(f"Following links in ServiceNow content {source_id} at depth {current_depth}")

        linked_docs = []

        # For ServiceNow content, we need to handle the different content types differently
        content_type, item_id = self._parse_source_id(source_id)

        # For HTML content (like in knowledge articles), extract links
        if content_type == "knowledge" and isinstance(content, str):
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')

                # Find all anchor tags with href attributes
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']

                    # Try to extract ServiceNow KB article ID from the href
                    kb_match = re.search(r'kb_view\.do\?sys_kb_id=([a-f0-9]+)', href)
                    if kb_match:
                        kb_id = kb_match.group(1)
                        linked_id = f"servicenow://{self.base_url}/knowledge/{kb_id}"

                        if linked_id not in global_visited_docs:
                            global_visited_docs.add(linked_id)
                            try:
                                linked_doc = self.fetch_document(linked_id)
                                linked_docs.append(linked_doc)
                                logger.debug(f"Successfully fetched linked knowledge article: {kb_id}")

                                # Recursively follow links if not at max depth
                                if current_depth + 1 < self.max_link_depth:
                                    nested_docs = self.follow_links(
                                        linked_doc["content"],
                                        linked_doc["id"],
                                        current_depth + 1,
                                        global_visited_docs
                                    )
                                    linked_docs.extend(nested_docs)
                            except Exception as e:
                                logger.warning(f"Error following knowledge article link {kb_id}: {str(e)}")

                    # Try to extract ServiceNow incident ID from the href
                    inc_match = re.search(r'incident\.do\?sys_id=([a-f0-9]+)', href)
                    if inc_match:
                        inc_id = inc_match.group(1)
                        linked_id = f"servicenow://{self.base_url}/incident/{inc_id}"

                        if linked_id not in global_visited_docs:
                            global_visited_docs.add(linked_id)
                            try:
                                linked_doc = self.fetch_document(linked_id)
                                linked_docs.append(linked_doc)
                                logger.debug(f"Successfully fetched linked incident: {inc_id}")
                            except Exception as e:
                                logger.warning(f"Error following incident link {inc_id}: {str(e)}")

            except Exception as e:
                logger.warning(f"Error parsing HTML for links: {str(e)}")

        # For JSON content (like incidents, catalog items), extract related records
        elif isinstance(content, dict):
            # Extract related incident from CMDB
            if content_type == "cmdb" and self.include_incidents:
                try:
                    # Get related incidents from the cmdb_ci_incident table
                    api_url = f"{self.base_url}{self.table_api_path}/cmdb_ci_incident"
                    params = {
                        "sysparm_query": f"cmdb_ci={item_id}",
                        "sysparm_limit": 10
                    }

                    response = self.session.get(api_url, params=params)
                    response.raise_for_status()

                    related_incidents = response.json().get("result", [])

                    for incident in related_incidents:
                        inc_id = incident.get("incident")
                        if inc_id:
                            linked_id = f"servicenow://{self.base_url}/incident/{inc_id}"

                            if linked_id not in global_visited_docs:
                                global_visited_docs.add(linked_id)
                                try:
                                    linked_doc = self.fetch_document(linked_id)
                                    linked_docs.append(linked_doc)
                                    logger.debug(f"Successfully fetched related incident: {inc_id}")
                                except Exception as e:
                                    logger.warning(f"Error following related incident {inc_id}: {str(e)}")

                except Exception as e:
                    logger.warning(f"Error fetching related incidents for CMDB item {item_id}: {str(e)}")

            # Extract CMDB CI from incident
            elif content_type == "incident" and self.include_cmdb:
                try:
                    # Get CMDB CI from the incident
                    cmdb_ci = content.get("cmdb_ci")

                    if cmdb_ci:
                        linked_id = f"servicenow://{self.base_url}/cmdb/{cmdb_ci}"

                        if linked_id not in global_visited_docs:
                            global_visited_docs.add(linked_id)
                            try:
                                linked_doc = self.fetch_document(linked_id)
                                linked_docs.append(linked_doc)
                                logger.debug(f"Successfully fetched related CMDB item: {cmdb_ci}")
                            except Exception as e:
                                logger.warning(f"Error following related CMDB item {cmdb_ci}: {str(e)}")

                except Exception as e:
                    logger.warning(f"Error processing CMDB CI for incident {item_id}: {str(e)}")

        logger.debug(f"Completed following links from {source_id}: found {len(linked_docs)} linked documents")
        return linked_docs

    def _fetch_knowledge_article(self, article_id: str, source_id: str) -> Dict[str, Any]:
        """
        Fetch a knowledge article from ServiceNow.

        Args:
            article_id: Knowledge article ID
            source_id: Original source ID

        Returns:
            Dictionary with content and metadata
        """
        # Construct API URL for knowledge article
        api_url = f"{self.base_url}{self.knowledge_api_path}/articles/{article_id}"

        # Make API request
        response = self.session.get(api_url)
        response.raise_for_status()

        # Parse response
        article_data = response.json().get("result", {})

        # Extract content details
        title = article_data.get("short_description", "")
        article_number = article_data.get("number", "")
        html_content = article_data.get("content", "")
        sys_id = article_data.get("sys_id", "")
        created_on = article_data.get("sys_created_on", "")
        updated_on = article_data.get("sys_updated_on", "")

        # Clean up HTML content if available
        if html_content:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()

                # Get clean HTML
                html_content = str(soup)
            except Exception as e:
                logger.warning(f"Error cleaning HTML content: {str(e)}")

        # Construct metadata
        metadata = {
            "title": title,
            "number": article_number,
            "sys_id": sys_id,
            "created_on": created_on,
            "updated_on": updated_on,
            "type": "knowledge",
            "url": f"{self.base_url}/kb_view.do?sys_kb_id={sys_id}",
            "api_url": api_url,
            "content_type": "html"  # Explicitly mark as HTML content
        }

        # Generate content hash for change detection
        content_hash = self.get_content_hash(html_content)

        # Cache the content
        self.content_cache[article_id] = {
            "content": html_content,
            "metadata": metadata,
            "hash": content_hash,
            "last_modified": updated_on,
            "last_accessed": time.time()
        }

        return {
            "id": source_id,
            "content": html_content,
            "doc_type": "html",  # Explicitly mark as HTML document type
            "metadata": metadata,
            "content_hash": content_hash
        }

    def _fetch_incident(self, incident_id: str, source_id: str) -> Dict[str, Any]:
        """
        Fetch an incident from ServiceNow.

        Args:
            incident_id: Incident sys_id
            source_id: Original source ID

        Returns:
            Dictionary with content and metadata
        """
        # Construct API URL for incident
        api_url = f"{self.base_url}{self.table_api_path}/incident/{incident_id}"

        # Make API request
        response = self.session.get(api_url)
        response.raise_for_status()

        # Parse response
        incident_data = response.json().get("result", {})

        # Should we include journal/work notes?
        try:
            # Get journal entries (comments and work notes)
            journal_url = f"{self.base_url}{self.table_api_path}/sys_journal_field"
            journal_params = {
                "sysparm_query": f"element_id={incident_id}^ORDERBYDESCsys_created_on",
                "sysparm_limit": 50  # Limit the number of entries to avoid huge payloads
            }

            journal_response = self.session.get(journal_url, params=journal_params)
            journal_response.raise_for_status()

            # Add journal entries to incident data
            incident_data["journal_entries"] = journal_response.json().get("result", [])
        except Exception as e:
            logger.warning(f"Error fetching journal entries for incident {incident_id}: {str(e)}")

        # Extract content details
        number = incident_data.get("number", "")
        short_description = incident_data.get("short_description", "")
        # description = incident_data.get("description", "")
        state = incident_data.get("state", "")
        priority = incident_data.get("priority", "")
        created_on = incident_data.get("sys_created_on", "")
        updated_on = incident_data.get("sys_updated_on", "")

        # Construct metadata
        metadata = {
            "number": number,
            "short_description": short_description,
            "state": state,
            "priority": priority,
            "created_on": created_on,
            "updated_on": updated_on,
            "type": "incident",
            "url": f"{self.base_url}/nav_to.do?uri=incident.do?sys_id={incident_id}",
            "api_url": api_url,
            "content_type": "json"  # Explicitly mark as JSON content
        }

        # Generate content hash for change detection
        content_hash = self.get_content_hash(json.dumps(incident_data))

        # Cache the content
        self.content_cache[incident_id] = {
            "content": incident_data,
            "metadata": metadata,
            "hash": content_hash,
            "last_modified": updated_on,
            "last_accessed": time.time()
        }

        return {
            "id": source_id,
            "content": incident_data,
            "doc_type": "json",  # Explicitly mark as JSON document type
            "metadata": metadata,
            "content_hash": content_hash
        }

    def _fetch_catalog_item(self, item_id: str, source_id: str) -> Dict[str, Any]:
        """
        Fetch a catalog item from ServiceNow.

        Args:
            item_id: Catalog item sys_id
            source_id: Original source ID

        Returns:
            Dictionary with content and metadata
        """
        # Construct API URL for catalog item
        api_url = f"{self.base_url}{self.table_api_path}/sc_cat_item/{item_id}"

        # Make API request
        response = self.session.get(api_url)
        response.raise_for_status()

        # Parse response
        item_data = response.json().get("result", {})

        # Should we include variables?
        try:
            # Get variables for this catalog item
            variables_url = f"{self.base_url}{self.table_api_path}/item_option_new"
            variables_params = {
                "sysparm_query": f"cat_item={item_id}",
                "sysparm_limit": 100
            }

            variables_response = self.session.get(variables_url, params=variables_params)
            variables_response.raise_for_status()

            # Add variables to item data
            item_data["variables"] = variables_response.json().get("result", [])
        except Exception as e:
            logger.warning(f"Error fetching variables for catalog item {item_id}: {str(e)}")

        # Extract content details
        name = item_data.get("name", "")
        short_description = item_data.get("short_description", "")
        # description = item_data.get("description", "")
        category = item_data.get("category", "")
        created_on = item_data.get("sys_created_on", "")
        updated_on = item_data.get("sys_updated_on", "")

        # Construct metadata
        metadata = {
            "name": name,
            "short_description": short_description,
            "category": category,
            "created_on": created_on,
            "updated_on": updated_on,
            "type": "catalog_item",
            "url": f"{self.base_url}/nav_to.do?uri=sc_cat_item.do?sys_id={item_id}",
            "api_url": api_url,
            "content_type": "json"  # Explicitly mark as JSON content
        }

        # Generate content hash for change detection
        content_hash = self.get_content_hash(json.dumps(item_data))

        # Cache the content
        self.content_cache[item_id] = {
            "content": item_data,
            "metadata": metadata,
            "hash": content_hash,
            "last_modified": updated_on,
            "last_accessed": time.time()
        }

        return {
            "id": source_id,
            "content": item_data,
            "doc_type": "json",  # Explicitly mark as JSON document type
            "metadata": metadata,
            "content_hash": content_hash
        }

    def _fetch_cmdb_item(self, ci_id: str, source_id: str) -> Dict[str, Any]:
        """
        Fetch a CMDB CI from ServiceNow.

        Args:
            ci_id: CMDB CI sys_id
            source_id: Original source ID

        Returns:
            Dictionary with content and metadata
        """
        # Construct API URL for CMDB CI
        api_url = f"{self.base_url}{self.table_api_path}/cmdb_ci/{ci_id}"

        # Make API request
        response = self.session.get(api_url)
        response.raise_for_status()

        # Parse response
        ci_data = response.json().get("result", {})

        # Should we include relationships?
        try:
            # Get relationships for this CI
            relations_url = f"{self.base_url}{self.table_api_path}/cmdb_rel_ci"
            relations_params = {
                "sysparm_query": f"parent={ci_id}^ORchild={ci_id}",
                "sysparm_limit": 100
            }

            relations_response = self.session.get(relations_url, params=relations_params)
            relations_response.raise_for_status()

            # Add relationships to CI data
            ci_data["relationships"] = relations_response.json().get("result", [])
        except Exception as e:
            logger.warning(f"Error fetching relationships for CMDB CI {ci_id}: {str(e)}")

        # Extract content details
        name = ci_data.get("name", "")
        sys_class_name = ci_data.get("sys_class_name", "")
        short_description = ci_data.get("short_description", "")
        created_on = ci_data.get("sys_created_on", "")
        updated_on = ci_data.get("sys_updated_on", "")

        # Construct metadata
        metadata = {
            "name": name,
            "sys_class_name": sys_class_name,
            "short_description": short_description,
            "created_on": created_on,
            "updated_on": updated_on,
            "type": "cmdb",
            "url": f"{self.base_url}/nav_to.do?uri=cmdb_ci.do?sys_id={ci_id}",
            "api_url": api_url,
            "content_type": "json"  # Explicitly mark as JSON content
        }

        # Generate content hash for change detection
        content_hash = self.get_content_hash(json.dumps(ci_data))

        # Cache the content
        self.content_cache[ci_id] = {
            "content": ci_data,
            "metadata": metadata,
            "hash": content_hash,
            "last_modified": updated_on,
            "last_accessed": time.time()
        }

        return {
            "id": source_id,
            "content": ci_data,
            "doc_type": "json",  # Explicitly mark as JSON document type
            "metadata": metadata,
            "content_hash": content_hash
        }

    def _list_knowledge_articles(self) -> List[Dict[str, Any]]:
        """
        List knowledge articles in ServiceNow.

        Returns:
            List of knowledge article identifiers and metadata
        """
        articles = []

        try:
            # Construct API URL
            api_url = f"{self.base_url}{self.knowledge_api_path}/articles"

            # Set up parameters
            params = {
                "limit": self.limit
            }

            # Add query if specified
            if self.knowledge_query:
                params["query"] = self.knowledge_query

            # Make API request
            response = self.session.get(api_url, params=params)
            response.raise_for_status()

            # Parse response
            data = response.json()

            # Process results
            for article in data.get("result", []):
                article_id = article.get("sys_id", "")
                title = article.get("short_description", "")
                article_number = article.get("number", "")

                # Apply pattern filters if configured
                if not self._should_include_content(title, article_id):
                    logger.debug(f"Skipping excluded knowledge article: {title} ({article_id})")
                    continue

                # Create fully qualified source identifier
                qualified_source = f"servicenow://{self.base_url}/knowledge/{article_id}"

                # Get basic metadata
                created_on = article.get("sys_created_on", "")
                updated_on = article.get("sys_updated_on", "")

                articles.append({
                    "id": qualified_source,
                    "metadata": {
                        "title": title,
                        "number": article_number,
                        "sys_id": article_id,
                        "created_on": created_on,
                        "updated_on": updated_on,
                        "type": "knowledge",
                        "url": f"{self.base_url}/kb_view.do?sys_kb_id={article_id}",
                        "content_type": "html"  # Explicitly mark as HTML content
                    },
                    "doc_type": "html"  # Also set document type for consistency
                })

            logger.debug(f"Found {len(articles)} knowledge articles")
            return articles

        except Exception as e:
            logger.error(f"Error listing knowledge articles: {str(e)}")
            return []

    def _list_incidents(self) -> List[Dict[str, Any]]:
        """
        List incidents in ServiceNow.

        Returns:
            List of incident identifiers and metadata
        """
        incidents = []

        try:
            # Construct API URL
            api_url = f"{self.base_url}{self.table_api_path}/incident"

            # Set up parameters
            params = {
                "sysparm_limit": self.limit
            }

            # Add query if specified
            if self.incident_query:
                params["sysparm_query"] = self.incident_query

            # Make API request
            response = self.session.get(api_url, params=params)
            response.raise_for_status()

            # Parse response
            data = response.json()

            # Process results
            for incident in data.get("result", []):
                incident_id = incident.get("sys_id", "")
                number = incident.get("number", "")
                short_description = incident.get("short_description", "")

                # Apply pattern filters if configured
                if not self._should_include_content(short_description, incident_id):
                    logger.debug(f"Skipping excluded incident: {short_description} ({incident_id})")
                    continue

                # Create fully qualified source identifier
                qualified_source = f"servicenow://{self.base_url}/incident/{incident_id}"

                # Get basic metadata
                state = incident.get("state", "")
                priority = incident.get("priority", "")
                created_on = incident.get("sys_created_on", "")
                updated_on = incident.get("sys_updated_on", "")

                incidents.append({
                    "id": qualified_source,
                    "metadata": {
                        "number": number,
                        "short_description": short_description,
                        "state": state,
                        "priority": priority,
                        "created_on": created_on,
                        "updated_on": updated_on,
                        "type": "incident",
                        "url": f"{self.base_url}/nav_to.do?uri=incident.do?sys_id={incident_id}",
                        "content_type": "json"  # Explicitly mark as JSON content
                    },
                    "doc_type": "json"  # Also set document type for consistency
                })

            logger.debug(f"Found {len(incidents)} incidents")
            return incidents

        except Exception as e:
            logger.error(f"Error listing incidents: {str(e)}")
            return []

    def _list_catalog_items(self) -> List[Dict[str, Any]]:
        """
        List catalog items in ServiceNow.

        Returns:
            List of catalog item identifiers and metadata
        """
        catalog_items = []

        try:
            # Construct API URL
            api_url = f"{self.base_url}{self.table_api_path}/sc_cat_item"

            # Set up parameters
            params = {
                "sysparm_limit": self.limit
            }

            # Add query if specified
            if self.service_catalog_query:
                params["sysparm_query"] = self.service_catalog_query

            # Make API request
            response = self.session.get(api_url, params=params)
            response.raise_for_status()

            # Parse response
            data = response.json()

            # Process results
            for item in data.get("result", []):
                item_id = item.get("sys_id", "")
                name = item.get("name", "")
                short_description = item.get("short_description", "")

                # Apply pattern filters if configured
                if not self._should_include_content(name, item_id):
                    logger.debug(f"Skipping excluded catalog item: {name} ({item_id})")
                    continue

                # Create fully qualified source identifier
                qualified_source = f"servicenow://{self.base_url}/catalog_item/{item_id}"

                # Get basic metadata
                category = item.get("category", "")
                created_on = item.get("sys_created_on", "")
                updated_on = item.get("sys_updated_on", "")

                catalog_items.append({
                    "id": qualified_source,
                    "metadata": {
                        "name": name,
                        "short_description": short_description,
                        "category": category,
                        "created_on": created_on,
                        "updated_on": updated_on,
                        "type": "catalog_item",
                        "url": f"{self.base_url}/nav_to.do?uri=sc_cat_item.do?sys_id={item_id}",
                        "content_type": "json"  # Explicitly mark as JSON content
                    },
                    "doc_type": "json"  # Also set document type for consistency
                })

            logger.debug(f"Found {len(catalog_items)} catalog items")
            return catalog_items

        except Exception as e:
            logger.error(f"Error listing catalog items: {str(e)}")
            return []

    def _list_cmdb_items(self) -> List[Dict[str, Any]]:
        """
        List CMDB CIs in ServiceNow.

        Returns:
            List of CMDB CI identifiers and metadata
        """
        cmdb_items = []

        try:
            # Construct API URL
            api_url = f"{self.base_url}{self.table_api_path}/cmdb_ci"

            # Set up parameters
            params = {
                "sysparm_limit": self.limit
            }

            # Add query if specified
            if self.cmdb_query:
                params["sysparm_query"] = self.cmdb_query

            # Make API request
            response = self.session.get(api_url, params=params)
            response.raise_for_status()

            # Parse response
            data = response.json()

            # Process results
            for item in data.get("result", []):
                item_id = item.get("sys_id", "")
                name = item.get("name", "")
                sys_class_name = item.get("sys_class_name", "")

                # Apply pattern filters if configured
                if not self._should_include_content(name, item_id):
                    logger.debug(f"Skipping excluded CMDB CI: {name} ({item_id})")
                    continue

                # Create fully qualified source identifier
                qualified_source = f"servicenow://{self.base_url}/cmdb/{item_id}"

                # Get basic metadata
                short_description = item.get("short_description", "")
                created_on = item.get("sys_created_on", "")
                updated_on = item.get("sys_updated_on", "")

                cmdb_items.append({
                    "id": qualified_source,
                    "metadata": {
                        "name": name,
                        "sys_class_name": sys_class_name,
                        "short_description": short_description,
                        "created_on": created_on,
                        "updated_on": updated_on,
                        "type": "cmdb",
                        "url": f"{self.base_url}/nav_to.do?uri=cmdb_ci.do?sys_id={item_id}",
                        "content_type": "json"  # Explicitly mark as JSON content
                    },
                    "doc_type": "json"  # Also set document type for consistency
                })

            logger.debug(f"Found {len(cmdb_items)} CMDB CIs")
            return cmdb_items

        except Exception as e:
            logger.error(f"Error listing CMDB CIs: {str(e)}")
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
    def _parse_source_id(source_id: str) -> tuple:
        """
        Parse source ID to extract content type and item ID.

        Args:
            source_id: Source identifier

        Returns:
            Tuple of (content_type, item_id)
        """
        # Check if it's a fully qualified source ID
        # Pattern: servicenow://instance/content_type/item_id
        qualified_match = re.search(r'servicenow://([^/]+)/([^/]+)/([^/]+)', source_id)
        if qualified_match:
            # base_url = qualified_match.group(1)
            content_type = qualified_match.group(2)
            item_id = qualified_match.group(3)
            return content_type, item_id

        # If not fully qualified, try other formats

        # Check if it's just an ID (could be any content type)
        if re.match(r'^[a-f0-9]{32}$', source_id):
            # This is a ServiceNow sys_id, assume knowledge article by default
            return "knowledge", source_id

        # For knowledge articles in kb_view.do format
        kb_match = re.search(r'kb_view\.do\?sys_kb_id=([a-f0-9]+)', source_id)
        if kb_match:
            return "knowledge", kb_match.group(1)

        # For incidents in incident.do format
        inc_match = re.search(r'incident\.do\?sys_id=([a-f0-9]+)', source_id)
        if inc_match:
            return "incident", inc_match.group(1)

        # For catalog items in sc_cat_item.do format
        cat_match = re.search(r'sc_cat_item\.do\?sys_id=([a-f0-9]+)', source_id)
        if cat_match:
            return "catalog_item", cat_match.group(1)

        # For CMDB CIs in cmdb_ci.do format
        cmdb_match = re.search(r'cmdb_ci\.do\?sys_id=([a-f0-9]+)', source_id)
        if cmdb_match:
            return "cmdb", cmdb_match.group(1)

        # If no match, assume it's a knowledge article ID
        return "knowledge", source_id

    @staticmethod
    def _parse_servicenow_timestamp(timestamp: str) -> Optional[float]:
        """
        Parse ServiceNow timestamp into epoch time.

        Args:
            timestamp: ServiceNow timestamp string (format: yyyy-mm-dd hh:mm:ss)

        Returns:
            Timestamp as epoch time or None if parsing fails
        """
        if not timestamp:
            return None

        try:
            # ServiceNow uses a standard format: 2023-05-01 12:34:56
            from datetime import datetime

            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            return dt.timestamp()
        except Exception:
            try:
                # Try parsing with dateutil for more flexibility
                import dateutil.parser

                dt = dateutil.parser.parse(timestamp)
                return dt.timestamp()
            except Exception:
                return None
