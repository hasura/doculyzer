"""
Factory for creating content resolvers.
"""
import logging
from typing import Dict, Any

from .base import ContentResolver
from .base import ContentSourceAdapter
from .confluence import ConfluenceAdapter
from .database import DatabaseAdapter
from .enhanced_content import EnhancedContentResolver
from .file import FileAdapter
from .jira import JiraAdapter
from .mongodb import MongoDBAdapter
from .s3 import S3Adapter
from .servicenow import ServiceNowAdapter
from .web import WebAdapter
from ..config import Config
from ..document_parser.csv import CsvParser
from ..document_parser.docx import DocxParser
from ..document_parser.html import HtmlParser
from ..document_parser.json import JSONParser
from ..document_parser.markdown import MarkdownParser
from ..document_parser.pdf import PdfParser
from ..document_parser.pptx import PptxParser
from ..document_parser.text import TextParser
from ..document_parser.xlsx import XlsxParser
from ..document_parser.xml import XmlParser

logger = logging.getLogger(__name__)


class ContentResolverFactory:
    """Factory class for creating content resolvers and their components."""

    @staticmethod
    def create_adapters(content_sources: Dict[str, Any]) -> Dict[str, ContentSourceAdapter]:
        """
        Create content source adapters based on configuration.

        Args:
            content_sources: Content source configuration

        Returns:
            Dictionary of adapters by type
        """
        adapters = {'file': FileAdapter(content_sources.get('file', {}))}

        # Add database adapter if configured
        if 'database' in content_sources:
            adapters['database'] = DatabaseAdapter(content_sources.get('database', {}))

        # Add web adapter if configured
        if 'web' in content_sources or 'http' in content_sources or 'https' in content_sources:
            web_config = content_sources.get('web', {})
            if 'http' in content_sources:
                web_config.update(content_sources.get('http', {}))
            if 'https' in content_sources:
                web_config.update(content_sources.get('https', {}))
            adapters['web'] = WebAdapter(web_config)

        # Add MongoDB adapter if configured
        if 'mongodb' in content_sources:
            adapters['mongodb'] = MongoDBAdapter(content_sources.get('mongodb', {}))

        # Add ServiceNow adapter if configured
        if 'servicenow' in content_sources:
            adapters['servicenow'] = ServiceNowAdapter(content_sources.get('servicenow', {}))

        # Add S3 adapter if configured
        if 's3' in content_sources:
            adapters['s3'] = S3Adapter(content_sources.get('s3', {}))

        # Add Confluence adapter if configured
        if 'confluence' in content_sources:
            adapters['confluence'] = ConfluenceAdapter(content_sources.get('confluence', {}))

        # Add JIRA adapter if configured
        if 'jira' in content_sources:
            adapters['jira'] = JiraAdapter(content_sources.get('jira', {}))

        return adapters

    @staticmethod
    def create_parsers(content_sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create document parsers based on configuration.

        Args:
            content_sources: Content source configuration

        Returns:
            Dictionary of parsers by type
        """
        parser_config = content_sources.get('parser_config', {})

        parsers = {
            'markdown': MarkdownParser(parser_config.get('markdown', {})),
            'html': HtmlParser(parser_config.get('html', {})),
            'docx': DocxParser(parser_config.get('docx', {})),
            'xlsx': XlsxParser(parser_config.get('xlsx', {})),
            'pdf': PdfParser(parser_config.get('pdf', {})),
            'pptx': PptxParser(parser_config.get('pptx', {})),
            'text': TextParser(parser_config.get('text', {})),
            'json': JSONParser(parser_config.get('json', {})),
            'xml': XmlParser(parser_config.get('xml', {})),
            'csv': CsvParser(parser_config.get('csv', {})),
        }

        return parsers

    @classmethod
    def create_enhanced_resolver(cls, content_sources: Dict[str, Any], path_mappings: Dict[str, str] = None,
                                 resolver_config: Dict[str, Any] = None) -> EnhancedContentResolver:
        """
        Create an enhanced content resolver with all configured components.

        Args:
            content_sources: Content source configuration dictionary
            path_mappings: Optional path mappings for source remapping
            resolver_config: Optional resolver-specific configuration

        Returns:
            EnhancedContentResolver instance
        """
        adapters = cls.create_adapters(content_sources)
        parsers = cls.create_parsers(content_sources)

        resolver = EnhancedContentResolver(
            adapters=adapters,
            parsers=parsers,
            path_mappings=path_mappings or {},
            config=resolver_config or {"cache_enabled": True}
        )

        return resolver

    @classmethod
    def create_resolver_from_config(cls, config: Config) -> ContentResolver:
        """
        Create a content resolver from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            ContentResolver instance
        """
        # Extract content sources from config
        content_sources = config.config.get('content_sources', [])
        path_mappings = config.config.get('path_mappings', {})
        resolver_config = config.config.get('resolver_config', {})

        # Process content sources into dictionary if it's a list
        sources_dict = {}
        if isinstance(content_sources, list):
            for source_config in content_sources:
                source_type = source_config.get("type")
                if source_type:
                    sources_dict[source_type] = source_config
        else:
            sources_dict = content_sources

        # Create enhanced resolver
        resolver = cls.create_enhanced_resolver(
            content_sources=sources_dict,
            path_mappings=path_mappings,
            resolver_config=resolver_config
        )

        return resolver


def create_content_resolver(config: Config) -> ContentResolver:
    """
    Create content resolver from configuration.

    Args:
        config: Configuration object

    Returns:
        ContentResolver instance
    """
    return ContentResolverFactory.create_resolver_from_config(config)
