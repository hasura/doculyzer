"""Automatically generated __init__.py"""
__all__ = ['Config', 'SearchHelper', 'SearchResult', 'SearchResultItem', 'SearchResults', 'advanced_search_endpoint',
           'api_info', 'bad_request', 'check_api_key', 'config', 'configure_logging', 'crawl', 'crawler',
           'create_simple_search_query', 'create_topic_search_query', 'document_sources_endpoint',
           'extract_topic_parameters', 'get_document_sources', 'get_element_topics', 'get_topic_statistics',
           'get_vendor_path', 'health_check', 'ingest_documents', 'internal_error', 'load_openapi_spec', 'main',
           'not_found', 'openapi_spec', 'print_startup_info', 'root', 'search', 'search_by_text', 'search_endpoint',
           'search_simple_structured', 'search_structured', 'search_with_content', 'server',
           'simple_structured_search_endpoint', 'structured_search_endpoint', 'supports_topics', 'vendor']

from . import config
from . import crawler
from . import main
from . import search
from . import server
from . import vendor
from .config import Config
from .configure_logging import configure_logging
from .crawler import crawl
from .main import ingest_documents
from .search import SearchHelper
from .search import SearchResult
from .search import SearchResultItem
from .search import SearchResults
from .search import create_simple_search_query
from .search import create_topic_search_query
from .search import get_document_sources
from .search import get_element_topics
from .search import get_topic_statistics
from .search import search_by_text
from .search import search_simple_structured
from .search import search_structured
from .search import search_with_content
from .search import supports_topics
from .server import advanced_search_endpoint
from .server import api_info
from .server import bad_request
from .server import check_api_key
from .server import document_sources_endpoint
from .server import extract_topic_parameters
from .server import health_check
from .server import internal_error
from .server import load_openapi_spec
from .server import not_found
from .server import openapi_spec
from .server import print_startup_info
from .server import root
from .server import search_endpoint
from .server import simple_structured_search_endpoint
from .server import structured_search_endpoint
from .vendor import get_vendor_path

configure_logging()

__version__ = "0.41.0"
