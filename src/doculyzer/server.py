import json
import logging
import os
from datetime import datetime
from typing import List

import yaml
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, InternalServerError

from .adapter import create_content_resolver
from .config import Config
from .search import search_with_content, search_by_text, get_document_sources, SearchResult

# Configure logging
log_level = os.environ.get('LOG_LEVEL', 'INFO')
log_format = os.environ.get('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=getattr(logging, log_level), format=log_format)
logger = logging.getLogger(__name__)
_config = Config(os.environ.get('DOCULYZER_CONFIG_PATH', 'config.yaml'))
db = _config.get_document_database()
db.initialize()
resolver = create_content_resolver(_config)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
cors_origins = os.environ.get('CORS_ORIGINS', '*').split(',')
CORS(app, origins=cors_origins)

# Get the directory where server.py is located
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration from environment variables
CONFIG = {
    'HOST': os.environ.get('SERVER_HOST', '0.0.0.0'),
    'PORT': int(os.environ.get('SERVER_PORT', '5000')),
    'DEBUG': os.environ.get('DEBUG', 'False').lower() == 'true',
    'MAX_RESULTS': int(os.environ.get('MAX_RESULTS', '100')),
    'DEFAULT_RESULTS': int(os.environ.get('DEFAULT_RESULTS', '10')),
    'MIN_SCORE_THRESHOLD': float(os.environ.get('MIN_SCORE_THRESHOLD', '0.0')),
    'TIMEOUT': int(os.environ.get('REQUEST_TIMEOUT', '30')),
    'MAX_CONTENT_LENGTH': int(os.environ.get('MAX_CONTENT_LENGTH', '16777216')),  # 16MB
    'RATE_LIMIT': os.environ.get('RATE_LIMIT', '100 per minute'),
    'API_KEY': os.environ.get('API_KEY'),  # Optional API key for authentication
    'API_KEY_HEADER': os.environ.get('API_KEY_HEADER', 'X-API-Key'),
    'OPENAPI_SPEC_PATH': os.environ.get('OPENAPI_SPEC_PATH', os.path.join(SERVER_DIR, 'openapi.yaml')),
    'SWAGGER_UI_ENABLED': os.environ.get('SWAGGER_UI_ENABLED', 'True').lower() == 'true',
    'SWAGGER_UI_PATH': os.environ.get('SWAGGER_UI_PATH', '/docs'),
    'API_SPEC_PATH': os.environ.get('API_SPEC_PATH', '/api/spec'),
}

# Set Flask configuration
app.config['MAX_CONTENT_LENGTH'] = CONFIG['MAX_CONTENT_LENGTH']


# Load OpenAPI specification
def load_openapi_spec():
    """Load the OpenAPI specification from file."""
    try:
        spec_path = CONFIG['OPENAPI_SPEC_PATH']
        if not os.path.exists(spec_path):
            logger.warning(f"OpenAPI spec file not found at {spec_path}")
            return None

        with open(spec_path, 'r') as f:
            if spec_path.endswith('.yaml') or spec_path.endswith('.yml'):
                spec = yaml.safe_load(f)
            else:
                spec = json.load(f)

        # Update server URLs with current configuration
        if 'servers' not in spec:
            spec['servers'] = []

        # Add current server URL
        current_server = f"http://{CONFIG['HOST']}:{CONFIG['PORT']}"
        spec['servers'].insert(0, {
            'url': current_server,
            'description': 'Current server'
        })

        return spec
    except Exception as e:
        logger.error(f"Error loading OpenAPI spec: {str(e)}")
        return None


# Authentication middleware
def check_api_key():
    """Check API key if configured."""
    if CONFIG['API_KEY']:
        api_key = request.headers.get(CONFIG['API_KEY_HEADER'])
        if not api_key or api_key != CONFIG['API_KEY']:
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Invalid or missing API key'
            }), 401
    return None


# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad Request',
        'message': str(error.description)
    }), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not Found',
        'message': 'Resource not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500


# Root endpoint with API documentation links
@app.route('/', methods=['GET'])
def root():
    """Root endpoint with links to documentation."""
    response_data = {
        'name': 'Document Search API',
        'version': '1.0.0',
        'status': 'running',
        'links': {
            'api_documentation': CONFIG['SWAGGER_UI_PATH'] if CONFIG['SWAGGER_UI_ENABLED'] else None,
            'openapi_spec': CONFIG['API_SPEC_PATH'],
            'health': '/health',
            'api_info': '/api/info'
        }
    }

    return jsonify({k: v for k, v in response_data.items() if v is not None})


# OpenAPI specification endpoint
@app.route(CONFIG['API_SPEC_PATH'], methods=['GET'])
def openapi_spec():
    """Serve the OpenAPI specification."""
    spec = load_openapi_spec()
    if spec is None:
        return jsonify({
            'error': 'Not Found',
            'message': 'OpenAPI specification not available'
        }), 404

    return jsonify(spec)


# Swagger UI endpoint
if CONFIG['SWAGGER_UI_ENABLED']:
    # Swagger UI HTML template
    SWAGGER_UI_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Document Search API - Swagger UI</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.10.3/swagger-ui.css">
        <style>
            html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
            *, *:before, *:after { box-sizing: inherit; }
            body { margin:0; background: #fafafa; }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@5.10.3/swagger-ui-bundle.js"></script>
        <script src="https://unpkg.com/swagger-ui-dist@5.10.3/swagger-ui-standalone-preset.js"></script>
        <script>
            window.onload = function() {
                const ui = SwaggerUIBundle({
                    url: "{{ openapi_url }}",
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIStandalonePreset
                    ],
                    plugins: [
                        SwaggerUIBundle.plugins.DownloadUrl
                    ],
                    layout: "StandaloneLayout",
                    defaultModelsExpandDepth: 1,
                    defaultModelExampleFormat: "value",
                    tryItOutEnabled: true,
                    persistAuthorization: true
                });

                window.ui = ui;
            };
        </script>
    </body>
    </html>
    """


    @app.route(CONFIG['SWAGGER_UI_PATH'], methods=['GET'])
    def swagger_ui():
        """Serve Swagger UI."""
        openapi_url = f"{CONFIG['API_SPEC_PATH']}"
        return render_template_string(SWAGGER_UI_TEMPLATE, openapi_url=openapi_url)


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })


# API Info endpoint
@app.route('/api/info', methods=['GET'])
def api_info():
    """Get API information and available endpoints."""
    info_data = {
        'name': 'Document Search API',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'Health check',
            '/api/info': 'API information',
            '/api/search': 'Search for elements',
            '/api/search/advanced': 'Advanced search with full results',
            '/api/search/sources': 'Get document sources'
        },
        'configuration': {
            'max_results': CONFIG['MAX_RESULTS'],
            'default_results': CONFIG['DEFAULT_RESULTS'],
            'min_score_threshold': CONFIG['MIN_SCORE_THRESHOLD'],
            'timeout': CONFIG['TIMEOUT']
        }
    }

    if CONFIG['SWAGGER_UI_ENABLED']:
        info_data['documentation'] = CONFIG['SWAGGER_UI_PATH']
        info_data['openapi_spec'] = CONFIG['API_SPEC_PATH']

    return jsonify(info_data)


# Standard search endpoint
@app.route('/api/search', methods=['POST'])
def search_endpoint():
    """
    Search for elements and return basic results.

    Request body:
    {
        "query": "search text",
        "limit": 10,
        "min_score": 0.0,
        "filter_criteria": {},
        "text": false,
        "content": false,
        "flat": false,
        "include_parents": true
    }
    """
    # Check API key if required
    auth_response = check_api_key()
    if auth_response:
        return auth_response

    try:
        # Parse request JSON
        data = request.get_json()
        if not data:
            raise BadRequest("Request body must be valid JSON")

        # Extract parameters
        query_text = data.get('query')
        if not query_text:
            raise BadRequest("'query' parameter is required")

        limit = min(data.get('limit', CONFIG['DEFAULT_RESULTS']), CONFIG['MAX_RESULTS'])
        flat = data.get('flat', False)
        include_parents = data.get('include_parents', True)
        min_score = max(data.get('min_score', CONFIG['MIN_SCORE_THRESHOLD']), 0.0)
        filter_criteria = data.get('filter_criteria', {})
        text = data.get('text', False)
        content = data.get('content', False)

        # Perform search
        logger.info(f"Search request: query='{query_text}', limit={limit}, min_score={min_score}, flat={flat}, include_parents={include_parents}")
        results = search_by_text(
            query_text=query_text,
            limit=limit,
            filter_criteria=filter_criteria,
            min_score=min_score,
            text=text,
            content=content,
            include_parents=include_parents,
            flat=flat
        )

        # Use model_dump_json() to properly serialize all nested objects
        # Then convert back to dict for jsonify()
        json_str = results.model_dump_json()
        json_dict = json.loads(json_str)
        return jsonify(json_dict)

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise InternalServerError(f"Search operation failed: {str(e)}")


# Advanced search endpoint
@app.route('/api/search/advanced', methods=['POST'])
def advanced_search_endpoint():
    """
    Advanced search with full results including relationships.

    Request body:
    {
        "query": "search text",
        "limit": 10,
        "min_score": 0.0,
        "filter_criteria": {},
        "resolve_content": true,
        "include_relationships": true
    }
    """
    # Check API key if required
    auth_response = check_api_key()
    if auth_response:
        return auth_response

    try:
        # Parse request JSON
        data = request.get_json()
        if not data:
            raise BadRequest("Request body must be valid JSON")

        # Extract parameters
        query_text = data.get('query')
        if not query_text:
            raise BadRequest("'query' parameter is required")

        limit = min(data.get('limit', CONFIG['DEFAULT_RESULTS']), CONFIG['MAX_RESULTS'])
        min_score = max(data.get('min_score', CONFIG['MIN_SCORE_THRESHOLD']), 0.0)
        filter_criteria = data.get('filter_criteria', {})
        resolve_content = data.get('resolve_content', True)
        include_relationships = data.get('include_relationships', True)

        # Perform advanced search
        logger.info(f"Advanced search: query='{query_text}', limit={limit}, min_score={min_score}")
        results: List[SearchResult] = search_with_content(
            query_text=query_text,
            limit=limit,
            filter_criteria=filter_criteria,
            resolve_content=resolve_content,
            include_relationships=include_relationships,
            min_score=min_score
        )

        # Convert to serializable format
        response_data = {
            'query': query_text,
            'total_results': len(results),
            'min_score': min_score,
            'results': []
        }

        for result in results:
            result_dict = {
                'similarity': result.similarity,
                'element_pk': result.element_pk,
                'element_id': result.element_id,
                'element_type': result.element_type,
                'content_preview': result.content_preview,
                'content_location': result.content_location,
                'doc_id': result.doc_id,
                'doc_type': result.doc_type,
                'source': result.source,
                'resolved_content': result.resolved_content,
                'resolved_text': result.resolved_text,
                'resolution_error': result.resolution_error,
                'relationship_count': result.get_relationship_count()
            }

            # Add relationship information if requested
            if include_relationships:
                result_dict['relationships'] = {
                    'by_type': result.get_relationships_by_type(),
                    'contained_elements': [
                        {
                            'relationship_type': rel.relationship_type,
                            'target_element_pk': rel.target_element_pk,
                            'target_element_type': rel.target_element_type,
                            'target_reference': rel.target_reference
                        }
                        for rel in result.get_contained_elements()
                    ],
                    'linked_elements': [
                        {
                            'relationship_type': rel.relationship_type,
                            'target_element_pk': rel.target_element_pk,
                            'target_element_type': rel.target_element_type,
                            'target_reference': rel.target_reference
                        }
                        for rel in result.get_linked_elements()
                    ],
                    'semantic_relationships': [
                        {
                            'relationship_type': rel.relationship_type,
                            'target_element_pk': rel.target_element_pk,
                            'target_element_type': rel.target_element_type,
                            'target_reference': rel.target_reference
                        }
                        for rel in result.get_semantic_relationships()
                    ]
                }

            response_data['results'].append(result_dict)

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Advanced search error: {str(e)}")
        raise InternalServerError(f"Advanced search operation failed: {str(e)}")


# Document sources endpoint
@app.route('/api/search/sources', methods=['POST'])
def document_sources_endpoint():
    """
    Get document sources from search results.

    Request body:
    {
        "query": "search text",
        "limit": 10,
        "min_score": 0.0,
        "filter_criteria": {}
    }
    """
    # Check API key if required
    auth_response = check_api_key()
    if auth_response:
        return auth_response

    try:
        # Parse request JSON
        data = request.get_json()
        if not data:
            raise BadRequest("Request body must be valid JSON")

        # Extract parameters
        query_text = data.get('query')
        if not query_text:
            raise BadRequest("'query' parameter is required")

        limit = min(data.get('limit', CONFIG['DEFAULT_RESULTS']), CONFIG['MAX_RESULTS'])
        min_score = max(data.get('min_score', CONFIG['MIN_SCORE_THRESHOLD']), 0.0)
        filter_criteria = data.get('filter_criteria', {})

        # Perform search to get results
        search_results = search_by_text(
            query_text=query_text,
            limit=limit,
            filter_criteria=filter_criteria,
            min_score=min_score
        )

        # Get document sources
        document_sources = get_document_sources(search_results)

        return jsonify({
            'query': query_text,
            'total_results': search_results.total_results,
            'document_sources': document_sources
        })

    except Exception as e:
        logger.error(f"Document sources error: {str(e)}")
        raise InternalServerError(f"Document sources operation failed: {str(e)}")


# Optional: Rate limiting if using Flask-Limiter
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    # Initialize Limiter with first argument as the key_func (not a parameter name)
    limiter = Limiter(
        get_remote_address,  # First argument is key_func (no parameter name)
        app=app,  # Pass app as a keyword argument
        default_limits=[CONFIG['RATE_LIMIT']],
        storage_uri="memory://",
        strategy="fixed-window"
    )

    # Apply rate limiting to search endpoints
    limiter.limit(CONFIG['RATE_LIMIT'])(search_endpoint)
    limiter.limit(CONFIG['RATE_LIMIT'])(advanced_search_endpoint)
    limiter.limit(CONFIG['RATE_LIMIT'])(document_sources_endpoint)

    logger.info(f"Rate limiting enabled: {CONFIG['RATE_LIMIT']}")
except ImportError:
    logger.warning("Flask-Limiter not installed, rate limiting disabled")


# Startup message
def print_startup_info():
    """Print startup information."""
    logger.info("=" * 50)
    logger.info("Document Search API Server Starting")
    logger.info("=" * 50)
    logger.info(f"Server URL: http://{CONFIG['HOST']}:{CONFIG['PORT']}")
    logger.info(f"API Documentation: http://{CONFIG['HOST']}:{CONFIG['PORT']}{CONFIG['SWAGGER_UI_PATH']}")
    logger.info(f"OpenAPI Spec: http://{CONFIG['HOST']}:{CONFIG['PORT']}{CONFIG['API_SPEC_PATH']}")
    logger.info(f"Debug Mode: {CONFIG['DEBUG']}")
    logger.info(f"Authentication: {'Enabled' if CONFIG['API_KEY'] else 'Disabled'}")
    logger.info(f"Rate Limiting: {CONFIG['RATE_LIMIT']}")
    logger.info("=" * 50)


# Main entry point
if __name__ == '__main__':
    print_startup_info()
    app.run(
        host=CONFIG['HOST'],
        port=CONFIG['PORT'],
        debug=CONFIG['DEBUG'],
        threaded=True
    )
