# Doculyzer

## Universal, Searchable, Structured Document Manager

Doculyzer is a powerful document management system that creates a universal, structured representation of documents from various sources while maintaining pointers to the original content rather than duplicating it.

```
┌─────────────────┐     ┌─────────────────┐     ┌────────────────┐
│ Content Sources │     │Document Ingester│     │  Storage Layer │
└────────┬────────┘     └────────┬────────┘     └────────┬───────┘
         │                       │                       │
┌────────┼────────┐     ┌────────┼────────┐     ┌────────┼──────┐
│ Confluence API  │     │Parser Adapters  │     │SQLite Backend │
│ Markdown Files  │◄───►│Structure Extract│◄───►│MongoDB Backend│
│ HTML from URLs  │     │Embedding Gen    │     │Vector Database│
│ DOCX Documents  │     │Relationship Map │     │Elasticsearch  │
└─────────────────┘     └─────────────────┘     └───────────────┘
```

## Key Features

- **Universal Document Model**: Common representation across document types
- **Preservation of Structure**: Maintains hierarchical document structure
- **Content Resolution**: Resolves pointers back to original content when needed
- **Advanced Structured Search**: Powerful query language with logical operators, similarity thresholds, and backend capability detection
- **Enhanced Search Capabilities**: Advanced pattern matching, element type filtering, and metadata search with LIKE patterns and ElementType enum support
- **Contextual Semantic Search**: Uses advanced embedding techniques that incorporate document context (hierarchy, neighbors) for more accurate semantic search
- **Topic-Aware Organization**: Categorize and filter content by topics for enhanced organization and discovery
- **Element-Level Precision**: Maintains granular accuracy to specific document elements
- **Relationship Mapping**: Identifies connections between document elements
- **Configurable Vector Representations**: Support for different vector dimensions based on content needs, allowing larger vectors for technical content and smaller vectors for general content
- **Modular Dependencies**: Only install the components you need, with graceful fallbacks when optional dependencies are missing
- **Backend-Agnostic Architecture**: Pluggable storage backends with automatic capability detection and query optimization

## Supported Document Types

Doculyzer can ingest and process a variety of document formats:
- HTML pages
- Markdown files
- Plain text files
- PDF documents
- Microsoft Word documents (DOCX)
- Microsoft PowerPoint presentations (PPTX)
- Microsoft Excel spreadsheets (XLSX)
- CSV files
- XML files
- JSON files

## Content Sources

Doculyzer supports multiple content sources through a modular, pluggable architecture. Each content source has its own optional dependencies, which are only required if you use that specific source:

| Content Source | Description | Required Dependencies | Installation |
|---------------|-------------|----------------------|--------------|
| File System | Local, mounted, and network file systems | None (core) | Default install |
| HTTP/Web | Fetch content from URLs and websites | `requests` | Default install |
| Confluence | Atlassian Confluence wiki content | `atlassian-python-api` | `pip install "doculyzer[source-confluence]"` |
| JIRA | Atlassian JIRA issue tracking system | `atlassian-python-api` | `pip install "doculyzer[source-jira]"` |
| Amazon S3 | Cloud storage through S3 | `boto3` | `pip install "doculyzer[cloud-aws]"` |
| Databases | SQL and NoSQL database content | `sqlalchemy` | `pip install "doculyzer[source-database]"` |
| ServiceNow | ServiceNow platform content | `pysnow` | `pip install "doculyzer[source-servicenow]"` |
| MongoDB | MongoDB database content | `pymongo` | `pip install "doculyzer[source-mongodb]"` |
| SharePoint | Microsoft SharePoint content | `Office365-REST-Python-Client` | `pip install "doculyzer[source-sharepoint]"` |
| Google Drive | Google Drive content | `google-api-python-client` | `pip install "doculyzer[source-gdrive]"` |

### Content Source Graceful Fallbacks

Doculyzer's modular design handles missing dependencies gracefully. When attempting to use a content source without the required dependencies, Doculyzer provides helpful error messages and installation instructions:

```python
from doculyzer import Config, ingest_documents
from doculyzer.content_sources import DatabaseContentSource

try:
    # Create a database content source
    db_source = DatabaseContentSource({
        "connection_string": "postgresql://user:password@localhost:5432/mydatabase",
        "query": "SELECT * FROM documents",
        "id_column": "doc_id",
        "content_column": "content_blob"
    })
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("To use database content sources, install:")
    print("pip install 'doculyzer[source-database]'")
```

## Storage Backends

Doculyzer supports multiple storage backends through a modular, pluggable architecture. Each backend has its own optional dependencies, which are only required if you use that specific storage method:

| Storage Backend | Description | Topic Support | Vector Search | Required Dependencies | Installation |
|-----------------|-------------|---------------|---------------|----------------------|--------------|
| File-based | Simple storage using the file system | ✅ | ❌ | None (core) | Default install |
| SQLite | Lightweight, embedded database | ✅ | ❌ | None (core) | Default install |
| SQLite Enhanced | SQLite with vector extension support | ✅ | ✅ | `sqlean.py` | `pip install "doculyzer[db-core]"` |
| Neo4J | Graph database with native relationship support | ✅ | ✅ | `neo4j` | `pip install "doculyzer[db-neo4j]"` |
| PostgreSQL | Robust relational database for production | ✅ | ❌ | `psycopg2` | `pip install "doculyzer[db-postgresql]"` |
| PostgreSQL + pgvector | PostgreSQL with vector search | ✅ | ✅ | `psycopg2`, `pgvector` | `pip install "doculyzer[db-postgresql,db-vector]"` |
| MongoDB | Document-oriented database | ✅ | ✅ | `pymongo` | `pip install "doculyzer[db-mongodb]"` |
| MySQL/MariaDB | Popular open-source SQL database | ✅ | ❌ | `sqlalchemy`, `pymysql` | `pip install "doculyzer[db-mysql]"` |
| Oracle | Enterprise SQL database | ✅ | ❌ | `sqlalchemy`, `cx_Oracle` | `pip install "doculyzer[db-oracle]"` |
| Microsoft SQL Server | Enterprise SQL database | ✅ | ❌ | `sqlalchemy`, `pymssql` | `pip install "doculyzer[db-mssql]"` |
| **Elasticsearch** | **Distributed search and analytics** | ✅ | ✅ | `elasticsearch` | `pip install "doculyzer[db-elasticsearch]"` |

### Storage Backend Graceful Fallbacks

Doculyzer's modular design handles missing storage dependencies gracefully. When attempting to use a storage backend without the required dependencies, Doculyzer provides helpful error messages and installation instructions:

```python
from doculyzer import Config, initialize_database

# In your config file:
# storage:
#   backend: postgresql
#   postgresql:
#     host: localhost
#     port: 5432
#     database: doculyzer
#     user: postgres
#     password: postgres

try:
    config = Config("config.yaml")
    db = config.initialize_database()
    # Use the database...
except ImportError as e:
    print(f"Database backend not available: {e}")
    print("Please install the required package with:")
    print("pip install 'doculyzer[db-postgresql]'")
```

### Database Backend Selection

You can easily switch between different backend implementations by changing your configuration:

```yaml
# SQLite (default, no additional dependencies)
storage:
  backend: sqlite
  path: "./data/docs.db"
  topic_support: true  # Enable topic features

# Elasticsearch (requires elasticsearch Python client)
storage:
  backend: elasticsearch
  elasticsearch:
    hosts: ["localhost:9200"]
    username: "elastic"  # optional
    password: "password"  # optional
    index_prefix: "doculyzer"
    vector_dimension: 384
    store_full_text: true
    index_full_text: true
    compress_full_text: false

# Neo4j (requires neo4j Python driver)
storage:
  backend: neo4j
  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "password"
    database: "doculyzer"

# PostgreSQL (requires psycopg2)
storage:
  backend: postgresql
  topic_support: true  # Enable topic features
  postgresql:
    host: "localhost"
    port: 5432
    database: "doculyzer"
    user: "postgres"
    password: "postgres"
    
# MongoDB (requires pymongo)
storage:
  backend: mongodb
  mongodb:
    host: "localhost"
    port: 27017
    db_name: "doculyzer"
    username: "admin"  # optional
    password: "password"  # optional

# MySQL/MariaDB (requires sqlalchemy and pymysql)
storage:
  backend: sqlalchemy
  sqlalchemy:
    uri: "mysql+pymysql://user:password@localhost/doculyzer"
    
# Microsoft SQL Server (requires sqlalchemy and pymssql)
storage:
  backend: sqlalchemy
  sqlalchemy:
    uri: "mssql+pymssql://user:password@localhost/doculyzer"
    
# Oracle (requires sqlalchemy and cx_Oracle)
storage:
  backend: sqlalchemy
  sqlalchemy:
    uri: "oracle://user:password@localhost:1521/doculyzer"
```

### Using a Specific Database Backend

```python
from doculyzer.db import Neo4jDocumentDatabase, PostgreSQLDocumentDatabase, ElasticsearchDocumentDatabase

# Using Elasticsearch backend (requires elasticsearch)
try:
    es_db = ElasticsearchDocumentDatabase({
        "hosts": ["localhost:9200"],
        "username": "elastic",
        "password": "changeme",
        "index_prefix": "doculyzer",
        "vector_dimension": 384,
        "store_full_text": True,
        "index_full_text": True
    })
    es_db.initialize()
    
    # Store and retrieve documents with advanced search capabilities
    es_db.store_document(document, elements, relationships)
    retrieved_doc = es_db.get_document("doc123")
    
    # Use advanced vector search
    results = es_db.search_by_embedding(query_embedding, limit=10)
    
except ImportError as e:
    print(f"Could not initialize ElasticsearchDocumentDatabase: {e}")
    print("Install required dependencies with:")
    print("pip install 'doculyzer[db-elasticsearch]'")

# Using Neo4j backend (requires neo4j)
try:
    neo4j_db = Neo4jDocumentDatabase({
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "password",
        "database": "doculyzer"
    })
    neo4j_db.initialize()
    
    # Store and retrieve documents
    neo4j_db.store_document(document, elements, relationships)
    retrieved_doc = neo4j_db.get_document("doc123")
    
except ImportError as e:
    print(f"Could not initialize Neo4jDocumentDatabase: {e}")
    print("Install required dependencies with:")
    print("pip install 'doculyzer[db-neo4j]'")

# Using PostgreSQL backend (requires psycopg2)
try:
    pg_db = PostgreSQLDocumentDatabase({
        "host": "localhost",
        "port": 5432,
        "dbname": "doculyzer",
        "user": "postgres",
        "password": "postgres"
    })
    pg_db.initialize()
    
    # Perform vector search if pgvector is available
    try:
        results = pg_db.search_by_embedding(query_embedding)
        print(f"Found {len(results)} similar documents")
    except Exception as e:
        print(f"Vector search not available: {e}")
        print("For vector search support, install:")
        print("pip install 'doculyzer[db-vector]'")
    
except ImportError as e:
    print(f"Could not initialize PostgreSQLDocumentDatabase: {e}")
    print("Install required dependencies with:")
    print("pip install 'doculyzer[db-postgresql]'")
```

### Vector-Capable Storage

For semantic search, Doculyzer supports several vector-capable database backends:

| Storage Backend | Vector Technology | Topic Support | Required Dependencies | Installation |
|-----------------|------------------|---------------|----------------------|--------------|
| SQLite + sqlite-vec | SIMD-accelerated vector search | ✅ | `sqlean.py`, `sqlite-vec` | `pip install "doculyzer[db-core,db-vector]"` |
| PostgreSQL + pgvector | Postgres vector extension | ✅ | `psycopg2`, `pgvector` | `pip install "doculyzer[db-postgresql,db-vector]"` |
| MongoDB Atlas | Vector search capability | ✅ | `pymongo` | `pip install "doculyzer[db-mongodb]"` |
| Neo4j Vector Search | Graph + vector search | ✅ | `neo4j` | `pip install "doculyzer[db-neo4j]"` |
| **Elasticsearch** | **Dense vector + kNN search** | ✅ | `elasticsearch` | `pip install "doculyzer[db-elasticsearch]"` |

```python
# Configure vector-capable storage
from doculyzer import Config

config = Config({
    "storage": {
        "backend": "elasticsearch",
        "topic_support": True,  # Enable topic features
        "elasticsearch": {
            "hosts": ["localhost:9200"],
            "index_prefix": "doculyzer",
            "vector_dimension": 384,
            "store_full_text": True,
            "index_full_text": True,
            "compress_full_text": True  # Enable compression for large text
        }
    }
})

try:
    db = config.initialize_database()
    
    # Vector operations will use optimized search when available,
    # and automatically fall back to Python implementation otherwise
    results = db.search_by_embedding(query_embedding)
    
except ImportError as e:
    print(f"Vector-capable backend not available: {e}")
    print("Install required dependencies with:")
    print("pip install 'doculyzer[db-elasticsearch]'")
```

## Architecture

The system is built with a modular architecture:

1. **Content Sources**: Adapters for different content origins (with conditional dependencies)
2. **Document Parsers**: Transform content into structured elements (with format-specific dependencies)
3. **Document Database**: Stores metadata, elements, and relationships (with backend-specific dependencies)
4. **Content Resolver**: Retrieves original content when needed
5. **Embedding Generator**: Creates vector representations for semantic search (with model-specific dependencies)
6. **Relationship Detector**: Identifies connections between document elements
7. **Topic Manager**: Organizes content by topics for enhanced categorization and filtering
8. **Structured Search Engine**: Advanced query processing with backend capability detection

## Advanced Structured Search System

Doculyzer includes a powerful, backend-agnostic structured search system that provides sophisticated querying capabilities with automatic optimization based on backend capabilities.

### Search Capabilities and Backend Detection

The system automatically detects what search features each backend supports:

```python
from doculyzer.structured_search import SearchCapability

# Check backend capabilities
capabilities = db.get_backend_capabilities()

print("Supported capabilities:")
for capability in capabilities.supported:
    print(f"  - {capability.value}")

# Common capabilities include:
# - TEXT_SIMILARITY: Semantic text search
# - EMBEDDING_SIMILARITY: Vector-based similarity search  
# - DATE_FILTERING: Date range and temporal queries
# - TOPIC_FILTERING: Topic-based content filtering
# - METADATA_EXACT: Exact metadata matching
# - METADATA_LIKE: Pattern-based metadata search
# - LOGICAL_AND/OR/NOT: Boolean query composition
# - SIMILARITY_THRESHOLDS: Configurable similarity scoring
```

### Fluent Query Builder

Build complex search queries using the intuitive fluent interface:

```python
from doculyzer.structured_search import SearchQueryBuilder, LogicalOperator

# Simple text search
query = (SearchQueryBuilder()
         .text_search("machine learning algorithms", similarity_threshold=0.8)
         .limit(20)
         .build())

# Complex query with multiple criteria
complex_query = (SearchQueryBuilder()
                 .with_operator(LogicalOperator.AND)
                 .text_search("security policy", similarity_threshold=0.7)
                 .last_days(30)
                 .topics(include=["security%", "policy%"], exclude=["deprecated%"])
                 .element_types(["header", "paragraph"])
                 .metadata_exact(department="engineering", priority="high")
                 .content_length(min_length=100, max_length=5000)
                 .include_highlighting(True)
                 .build())

# Nested logical operations
nested_query = (SearchQueryBuilder()
                .begin_group(LogicalOperator.OR)
                .text_search("artificial intelligence")
                .topics(include=["ai%", "ml%"])
                .end_group()
                .begin_group(LogicalOperator.NOT)
                .topics(include=["deprecated%"])
                .end_group()
                .build())
```

### Advanced Date and Time Filtering

The structured search system provides sophisticated temporal filtering capabilities:

```python
from doculyzer.structured_search import DateRangeOperator
from datetime import datetime, timedelta

# Recent content
recent_query = (SearchQueryBuilder()
                .text_search("quarterly reports")
                .last_days(7)
                .build())

# Specific date ranges
date_range_query = (SearchQueryBuilder()
                    .text_search("financial data")
                    .date_range(
                        start_date=datetime(2024, 1, 1),
                        end_date=datetime(2024, 3, 31)
                    )
                    .build())

# Fiscal year filtering
fiscal_query = (SearchQueryBuilder()
                .fiscal_year(2024)
                .topics(include=["finance%", "budget%"])
                .build())

# Quarter-specific search
quarter_query = (SearchQueryBuilder()
                 .quarter(2024, 3)  # Q3 2024
                 .element_types(["header", "table"])
                 .build())
```

### Vector and Hybrid Search

Combine text search with vector similarity for optimal results:

```python
# Pure vector search
vector_query = (SearchQueryBuilder()
                .embedding_search(
                    embedding_vector=query_embedding,
                    similarity_threshold=0.75,
                    distance_metric="cosine"
                )
                .topics(include=["technical%"])
                .build())

# Hybrid search (text + vector)
hybrid_query = (SearchQueryBuilder()
                .text_search("database optimization", boost_factor=0.3)
                .embedding_search(query_embedding, boost_factor=0.7)
                .score_combination("weighted_avg", {
                    "text_similarity": 0.3,
                    "embedding_similarity": 0.7
                })
                .build())
```

### Executing Structured Queries

```python
# Execute query and get results
results = db.execute_structured_search(complex_query)

for result in results:
    print(f"Element: {result['element_id']}")
    print(f"Score: {result['final_score']:.3f}")
    print(f"Preview: {result['content_preview']}")
    
    if result.get('topics'):
        print(f"Topics: {result['topics']}")
    
    if result.get('extracted_dates'):
        print(f"Dates: {len(result['extracted_dates'])} found")
    
    print("---")
```

### Query Validation and Optimization

The system validates queries against backend capabilities and provides helpful error messages:

```python
from doculyzer.structured_search import validate_query_capabilities, UnsupportedSearchError

try:
    # Validate query before execution
    missing_capabilities = validate_query_capabilities(query, db.get_backend_capabilities())
    
    if missing_capabilities:
        print(f"Query requires unsupported capabilities: {[c.value for c in missing_capabilities]}")
        # Automatically fallback or modify query
    else:
        results = db.execute_structured_search(query)
        
except UnsupportedSearchError as e:
    print(f"Backend cannot execute this query: {e}")
    # Suggest alternative backends or query modifications
```

### Common Query Patterns

Access pre-built query patterns for common use cases:

```python
from doculyzer.structured_search import get_common_query_patterns

patterns = get_common_query_patterns()

# Use pre-built patterns
simple_text_query = patterns["simple_text"]
recent_content_query = patterns["recent_content"]
topic_search_query = patterns["topic_search"]
complex_logic_query = patterns["complex_logic"]

# Modify patterns as needed
custom_query = (SearchQueryBuilder()
                .text_search("custom search term")
                .last_days(14)  # Modified from pattern
                .element_types(["paragraph", "header"])
                .build())
```

## Enhanced Search Capabilities

Doculyzer provides powerful, flexible search capabilities across all database backends with support for pattern matching, element type filtering, and metadata search.

### Pattern Matching with LIKE Operators

Use `_like` and `_ilike` suffixes for pattern matching across any field:

```python
# Case-sensitive LIKE patterns
results = db.find_elements({
    "content_preview_like": "%important%",
    "element_type_like": "head%"
})

# Case-insensitive LIKE patterns (where supported)
results = db.find_elements({
    "content_preview_ilike": "%SUMMARY%",
    "element_type_ilike": "HEAD%"
})

# Document search with patterns
docs = db.find_documents({
    "source_like": "%reports%",
    "doc_type_ilike": "PDF"
})
```

### ElementType Enum Integration

Search using ElementType enums for type-safe, consistent queries:

```python
from doculyzer.db.element_element import ElementType

# Search with single ElementType
results = db.find_elements({
    "element_type": ElementType.HEADER
})

# Search with multiple ElementTypes
results = db.find_elements({
    "element_type": [ElementType.HEADER, ElementType.PARAGRAPH, ElementType.BLOCKQUOTE]
})

# Mix enum and string queries
results = db.find_elements({
    "element_type": [ElementType.HEADER, "custom_type"],
    "content_preview_like": "%important%"
})
```

### Category-Based Search

Use predefined element type categories for convenient grouping:

```python
# Search by predefined categories
text_elements = db.find_elements_by_category("text_elements")
table_elements = db.find_elements_by_category("table_elements")
media_elements = db.find_elements_by_category("media_elements")

# Combine category search with additional filters
results = db.find_elements_by_category(
    "text_elements",
    content_preview_like="%important%",
    doc_id=["doc1", "doc2"]
)

# Available categories
categories = db.get_element_types_by_category()
print("Available categories:", list(categories.keys()))
# Output: ['text_elements', 'structural_elements', 'list_elements', 
#          'table_elements', 'media_elements', 'code_elements', 
#          'presentation_elements', 'data_elements', 'xml_elements']
```

### Advanced Metadata Search

Search within JSON metadata using database-specific optimizations:

```python
# Exact metadata matching
results = db.find_elements({
    "metadata": {"section": "introduction", "priority": "high"}
})

# Metadata LIKE patterns
results = db.find_elements({
    "metadata_like": {"title": "%annual%", "author": "Smith%"}
})

# Case-insensitive metadata patterns (PostgreSQL)
results = db.find_elements({
    "metadata_ilike": {"title": "%ANNUAL%"}
})

# Document metadata search
docs = db.find_documents({
    "metadata_like": {"project": "%2024%"}
})
```

### Complex Query Examples

```python
# Complex element search combining multiple criteria
results = db.find_elements({
    "element_type": [ElementType.HEADER, ElementType.PARAGRAPH],
    "content_preview_like": "%security%",
    "doc_id": ["security_doc_1", "security_doc_2"],
    "metadata_like": {"classification": "%confidential%"}
}, limit=50)

# Search across different fields with patterns
results = db.find_documents({
    "doc_type": ["pdf", "docx"],
    "source_like": "%/reports/%",
    "metadata": {"status": "published"},
    "metadata_like": {"title": "%quarterly%"}
})

# Case-insensitive search for flexible matching
results = db.find_elements({
    "element_type_ilike": "head%",  # Matches "HEADER", "Header", "header"
    "content_preview_ilike": "%API%"  # Case-insensitive content search
})
```

### Database-Specific Optimizations

Doculyzer automatically uses database-specific features for optimal performance:

#### Elasticsearch Features
- Native vector search with kNN queries
- Configurable text storage and indexing options
- Bulk operations for high-performance ingestion
- Advanced text analysis and scoring
- Compressed storage for large text content

#### PostgreSQL Features
- Native `ILIKE` for case-insensitive patterns
- JSONB operators for efficient metadata queries
- GIN indexes for pattern matching performance
- pgvector integration for vector similarity search

#### SQLite Features  
- JSON1 extension for metadata queries
- FTS (Full Text Search) when available
- sqlite-vec or sqlite-vss for vector search
- COLLATE NOCASE for case-insensitive matching

#### Universal Features
- Automatic fallback to compatible implementations
- Consistent API across all database backends
- Pattern matching support in all storage layers
- ElementType enum support everywhere

### Search Capability Detection

Check backend capabilities at runtime:

```python
# Check what search features are supported
print(f"LIKE patterns: {db.supports_like_patterns()}")
print(f"Case-insensitive LIKE: {db.supports_case_insensitive_like()}")
print(f"ElementType enums: {db.supports_element_type_enums()}")
print(f"Topic support: {db.supports_topics()}")
print(f"Vector search: {db.supports_vector_search()}")

# Check structured search capabilities
capabilities = db.get_backend_capabilities()
print(f"Text similarity: {capabilities.supports(SearchCapability.TEXT_SIMILARITY)}")
print(f"Date filtering: {capabilities.supports(SearchCapability.DATE_FILTERING)}")
print(f"Nested queries: {capabilities.supports(SearchCapability.NESTED_QUERIES)}")

# Use case-insensitive search when available
if db.supports_case_insensitive_like():
    results = db.find_elements_ilike({"content_preview_ilike": "%IMPORTANT%"})
else:
    # Fallback to case-sensitive search
    results = db.find_elements({"content_preview_like": "%important%"})
```

## Content Monitoring and Updates

Doculyzer includes a robust system for monitoring content sources and handling updates:

### Change Detection

- **Efficient Monitoring**: Tracks content sources for changes using lightweight methods (timestamps, ETags, content hashes)
- **Selective Processing**: Only reprocesses documents that have changed since their last ingestion
- **Hash-Based Comparison**: Uses content hashes to avoid unnecessary processing when content hasn't changed
- **Source-Specific Strategies**: Each content source type implements its own optimal change detection mechanism

### Update Process

```python
# Schedule regular updates
from doculyzer import ingest_documents
import schedule
import time

def update_documents():
    # This will only process documents that have changed
    stats = ingest_documents(config)
    print(f"Updates: {stats['documents']} documents, {stats['unchanged_documents']} unchanged")

# Run updates every hour
schedule.every(1).hour.do(update_documents)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Update Status Tracking

- **Processing History**: Maintains a record of when each document was last processed
- **Content Hash Storage**: Stores content hashes to quickly identify changes
- **Update Statistics**: Provides metrics on documents processed, unchanged, and updated
- **Pointer-Based Architecture**: Since Doculyzer stores pointers to original content rather than copies, it efficiently handles updates without versioning complications

### Scheduled Crawling

For continuous monitoring of content sources, Doculyzer can be configured to run scheduled crawls:

```python
import argparse
import logging
import time
from doculyzer import crawl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doculyzer Crawler")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--interval", type=int, default=3600, help="Crawl interval in seconds")
    args = parser.parse_args()
    
    logger = logging.getLogger("Doculyzer Crawler")
    logger.info(f"Crawler initialized with interval {args.interval} seconds")
    
    while True:
        crawl(args.config, args.interval)
        logger.info(f"Sleeping for {args.interval} seconds")
        time.sleep(args.interval)
```

Run the crawler as a background process or service:

```bash
# Run crawler with 1-hour interval
python crawler.py --config config.yaml --interval 3600
```

For production environments, consider using a proper task scheduler like Celery or a cron job to manage the crawl process.

## Getting Started

### Flexible Installation

Doculyzer supports a modular installation system where you can choose which components to install based on your specific needs:

```bash
# Minimal installation (core functionality only)
pip install doculyzer

# Install with specific database backend
pip install "doculyzer[db-postgresql]"    # PostgreSQL support
pip install "doculyzer[db-mongodb]"       # MongoDB support
pip install "doculyzer[db-neo4j]"         # Neo4j support
pip install "doculyzer[db-mysql]"         # MySQL support
pip install "doculyzer[db-elasticsearch]" # Elasticsearch support
pip install "doculyzer[db-core]"          # SQLite extensions + SQLAlchemy

# Install with specific content sources
pip install "doculyzer[source-database]"     # Database content sources
pip install "doculyzer[source-confluence]"   # Confluence content sources
pip install "doculyzer[source-jira]"         # JIRA content sources
pip install "doculyzer[source-gdrive]"       # Google Drive content sources
pip install "doculyzer[source-sharepoint]"   # SharePoint content sources
pip install "doculyzer[source-servicenow]"   # ServiceNow content sources
pip install "doculyzer[source-mongodb]"      # MongoDB content sources

# Install with specific embedding provider
pip install "doculyzer[huggingface]"    # HuggingFace/PyTorch support
pip install "doculyzer[openai]"         # OpenAI API support
pip install "doculyzer[fastembed]"      # FastEmbed support (15x faster)

# Install with AWS S3 support
pip install "doculyzer[cloud-aws]"

# Install additional components
pip install "doculyzer[scientific]"     # NumPy and scientific libraries
pip install "doculyzer[document_parsing]"  # Additional document parsing utilities

# Install all database backends
pip install "doculyzer[db-all]"

# Install all content sources
pip install "doculyzer[source-all]"

# Install all embedding providers
pip install "doculyzer[embedding-all]"

# Install everything
pip install "doculyzer[all]"
```

You can also use requirements.txt with the desired components uncommented:

```txt
# REQUIRED DEPENDENCIES - Core functionality
lxml~=5.4.0
PyYAML~=6.0.2
beautifulsoup4~=4.13.4
Markdown~=3.8
requests~=2.32.3
python-dateutil~=2.9.0
jsonpath-ng~=1.7.0
python-dotenv~=1.1.0
wcmatch~=10.0

# Document parsers
python-docx~=1.1.2
openpyxl~=3.1.5
pymupdf~=1.25.5
python-pptx~=1.0.2

# Uncomment the components you need:
# SQLAlchemy (ORM framework)
# SQLAlchemy~=2.0.40

# SQLite extensions
# sqlean.py~=3.47.0; platform_system == 'Darwin'
# sqlean.py~=3.47.0; platform_system == 'Linux' and platform_machine == 'x86_64'

# Elasticsearch backend
# elasticsearch~=8.0.0

# Database content sources
# sqlalchemy~=2.0.40
# psycopg2-binary~=2.9.9; platform_system != 'Windows'
# psycopg2~=2.9.9; platform_system == 'Windows'
# pymssql~=2.2.10
# pymysql~=1.1.0

# Confluence/JIRA content sources
# atlassian-python-api~=3.41.9

# SharePoint content sources
# Office365-REST-Python-Client~=2.5.0

# NumPy (for vector operations)
# numpy~=2.0.2

# Embedding provider (choose one)
# torch==2.7.0
# sentence-transformers~=4.1.0
# openai~=1.76.0
# fastembed>=0.1.0
```

### Configuration

Create a configuration file `config.yaml`:

```yaml
storage:
  backend: elasticsearch  # Options: file, sqlite, mongodb, postgresql, elasticsearch, sqlalchemy
  topic_support: true  # Enable topic features
  
  # Elasticsearch-specific configuration
  elasticsearch:
    hosts: ["localhost:9200"]
    username: "elastic"  # optional
    password: "changeme"  # optional
    index_prefix: "doculyzer"
    vector_dimension: 384
    store_full_text: true
    index_full_text: true
    compress_full_text: false
    full_text_max_length: null  # No limit
  
  # MongoDB-specific configuration (if using MongoDB)
  mongodb:
    host: localhost
    port: 27017
    db_name: doculyzer
    username: myuser  # optional
    password: mypassword  # optional

embedding:
  enabled: true
  # Embedding provider: choose between "huggingface", "openai", or "fastembed"
  provider: "huggingface"
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimensions: 384  # Configurable based on content needs
  contextual: true  # Enable contextual embeddings
  
  # Contextual embedding configuration
  predecessor_count: 1
  successor_count: 1
  ancestor_depth: 1
  child_count: 1
  
  # Content-specific configurations
  content_types:
    technical:
      model: "sentence-transformers/all-mpnet-base-v2"
      dimensions: 768  # Larger vectors for technical content
    general:
      model: "sentence-transformers/all-MiniLM-L6-v2"
      dimensions: 384  # Smaller vectors for general content
  
  # OpenAI-specific configuration (if using OpenAI provider)
  openai:
    api_key: "your_api_key_here"
    model: "text-embedding-3-small"
    dimensions: 1536  # Embedding dimensions for OpenAI model
  
  # FastEmbed-specific configuration (if using FastEmbed provider)
  fastembed:
    model: "BAAI/bge-small-en-v1.5"  # Default FastEmbed model
    dimensions: 384  # Embedding dimensions for FastEmbed model
    cache_dir: "./model_cache"  # Optional: dir to cache models

content_sources:
  # Local file content source (core, no extra dependencies)
  - name: "documentation"
    type: "file"
    base_path: "./docs"
    file_pattern: "**/*.md"
    max_link_depth: 2
    topics: ["documentation", "user-guides"]  # Assign topics to this source
    
  # Example of a blob-based database content source (requires sqlalchemy)
  - name: "database-blobs"
    type: "database"
    connection_string: "postgresql://user:password@localhost:5432/mydatabase"
    query: "SELECT * FROM documents"
    id_column: "doc_id"
    content_column: "content_blob" 
    metadata_columns: ["title", "author", "created_date"]
    timestamp_column: "updated_at"
    topics: ["database", "technical-docs"]  # Assign topics to this source
    
  # Example of Confluence content source (requires atlassian-python-api)
  - name: "confluence-docs"
    type: "confluence"
    url: "https://company.atlassian.net/wiki"
    username: "${CONFLUENCE_USER}"  # Use environment variables securely
    password: "${CONFLUENCE_PASS}"
    space_keys: ["DEV", "PROD"]
    max_results: 1000
    topics: ["confluence", "team-docs", "development"]  # Assign topics to this source

  # Example of S3 bucket content source (requires boto3)
  - name: "aws-documents"
    type: "s3"
    bucket: "company-documents"
    prefix: "technical-docs/"
    region: "us-west-2"
    file_pattern: "*.{pdf,docx,xlsx}"
    topics: ["cloud", "aws", "infrastructure"]  # Assign topics to this source

relationship_detection:
  enabled: true
  link_pattern: r"\[\[(.*?)\]\]|href=[\"\'](.*?)[\"\']"

logging:
  level: "INFO"
  file: "./logs/docpointer.log"
```

### Basic Usage

```python
from doculyzer import Config, ingest_documents
from doculyzer.structured_search import SearchQueryBuilder, LogicalOperator

# Load configuration
config = Config("config.yaml")

# Initialize storage
db = config.initialize_database()

# Ingest documents
stats = ingest_documents(config)
print(f"Processed {stats['documents']} documents with {stats['elements']} elements")

# Enhanced search examples
from doculyzer.db.element_element import ElementType

# Search with pattern matching
results = db.find_elements({
    "content_preview_like": "%important%",
    "element_type": ElementType.HEADER
})

# Search by category
text_elements = db.find_elements_by_category(
    "text_elements", 
    content_preview_like="%summary%"
)

# Complex metadata search
docs = db.find_documents({
    "doc_type_like": "pdf",
    "metadata_like": {"title": "%annual%"},
    "source_like": "%reports%"
})

# Advanced structured search
structured_query = (SearchQueryBuilder()
                    .text_search("machine learning", similarity_threshold=0.8)
                    .last_days(30)
                    .topics(include=["tech%", "ai%"], exclude=["deprecated%"])
                    .element_types([ElementType.HEADER, ElementType.PARAGRAPH])
                    .metadata_exact(priority="high")
                    .include_highlighting(True)
                    .limit(20)
                    .build())

results = db.execute_structured_search(structured_query)
for result in results:
    print(f"Score: {result['final_score']:.3f} - {result['content_preview']}")

# Semantic search (if embeddings are enabled)
from doculyzer.embeddings import get_embedding_generator

# Get the configured embedding generator
embedding_generator = get_embedding_generator(config)
query_embedding = embedding_generator.generate("search query")
results = db.search_by_embedding(query_embedding)
for element_id, score in results:
    element = db.get_element(element_id)
    print(f"Semantic match ({score:.2f}): {element['content_preview']}")
```

### Using a Specific Content Source

```python
from doculyzer.content_sources import DatabaseContentSource, ConfluenceContentSource

# Using a database content source (requires sqlalchemy)
try:
    db_source = DatabaseContentSource({
        "connection_string": "postgresql://user:password@localhost:5432/mydatabase",
        "query": "SELECT * FROM documents",
        "id_column": "doc_id",
        "content_column": "content_blob",
        "json_mode": False  # Set to True for structured JSON output
    })
    
    # Fetch a specific document
    document = db_source.fetch_document("doc123")
    print(f"Document content: {document['content']}")
    
    # List all available documents
    documents = db_source.list_documents()
    print(f"Available documents: {len(documents)}")
    
except ImportError as e:
    print(f"Could not initialize DatabaseContentSource: {e}")
    print("Install required dependencies with:")
    print("pip install 'doculyzer[source-database]'")

# Using a Confluence content source (requires atlassian-python-api)
try:
    confluence_source = ConfluenceContentSource({
        "url": "https://company.atlassian.net/wiki",
        "username": "user",
        "password": "pass",
        "space_keys": ["DEV"]
    })
    
    # List available pages
    pages = confluence_source.list_documents()
    print(f"Found {len(pages)} pages in Confluence")
    
except ImportError as e:
    print(f"Could not initialize ConfluenceContentSource: {e}")
    print("Install required dependencies with:")
    print("pip install 'doculyzer[source-confluence]'")
```

## Advanced Features

### Topic-Aware Organization

Doculyzer includes a powerful topic system for organizing and categorizing content:

- **Source-Level Topics**: Assign topics to content sources during configuration
- **Topic-Aware Embeddings**: Store embeddings with associated topics for enhanced organization
- **Topic Filtering**: Search and filter content by topic patterns using LIKE syntax
- **Topic Analytics**: Get statistics on topic distribution across your document collection

#### Topic Configuration

Topics can be assigned to content sources in your configuration:

```yaml
content_sources:
  - name: "security-docs"
    type: "file"
    base_path: "./security"
    topics: ["security", "compliance", "policies"]
    
  - name: "dev-confluence"
    type: "confluence"
    space_keys: ["DEV"]
    topics: ["development", "engineering", "technical"]
```

#### Topic-Aware Search

```python
# Search with topic filtering (requires topic-enabled backend)
if db.supports_topics():
    # Search for security-related content
    results = db.search_by_text_and_topics(
        search_text="authentication policy",
        include_topics=["security%", "%.policy%"],  # LIKE patterns
        exclude_topics=["deprecated%"],
        min_confidence=0.8,
        limit=10
    )
    
    for result in results:
        print(f"Element {result['element_pk']}: {result['similarity']:.2f}")
        print(f"Topics: {result['topics']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
    # Get topic statistics
    stats = db.get_topic_statistics()
    for topic, info in stats.items():
        print(f"{topic}: {info['embedding_count']} embeddings, {info['document_count']} docs")

# Using structured search with topics
topic_query = (SearchQueryBuilder()
               .text_search("security protocols")
               .topics(
                   include=["security%", "protocol%"],
                   exclude=["deprecated%", "draft%"],
                   require_all=False,  # OR logic for includes
                   min_confidence=0.8
               )
               .last_months(6)
               .build())

results = db.execute_structured_search(topic_query)
```

#### Topic Filtering Logic

When using topic filtering, Doculyzer applies different logical operators for includes and excludes:

##### Include Topics (OR Logic)
Elements matching **any** of the include patterns will be included:

```python
# Example: Find elements with topics about security OR policies OR compliance
include_topics=["security%", "policy%", "compliance%"]

# This finds elements with topics like:
# - "security.authentication"  (matches "security%")
# - "policy.access"            (matches "policy%") 
# - "compliance.sox"           (matches "compliance%")
# - "security.policy.auth"     (matches both "security%" AND "policy%")
```

**Result:** An element is included if it has **ANY** topic matching **ANY** of the patterns.

##### Exclude Topics (AND Logic)
Elements matching **any** of the exclude patterns will be excluded:

```python
# Example: Exclude elements with deprecated OR draft topics
exclude_topics=["deprecated%", "draft%"]

# This excludes elements with topics like:
# - "deprecated.old"   (matches "deprecated%")
# - "draft.review"     (matches "draft%")
# - "deprecated.draft" (matches both patterns - still excluded)
```

**Result:** An element is excluded if it has **ANY** topic matching **ANY** of the excluded patterns.

##### Combined Example

```python
results = search_by_text_and_topics(
    search_text="authentication policy",
    include_topics=["security%", "policy%", "compliance%"],  # OR: security OR policy OR compliance
    exclude_topics=["deprecated%", "draft%"],               # AND: NOT deprecated AND NOT draft
    min_confidence=0.8
)
```

This finds elements that:
1. Match the text "authentication policy" **AND**
2. Have topics like `security.auth` **OR** `policy.access` **OR** `compliance.sox` **AND**
3. Do **NOT** have topics like `deprecated.old` **AND** do **NOT** have topics like `draft.review`

The OR logic for includes casts a wide net for relevant content, while the AND logic for excludes ensures comprehensive filtering of unwanted content.

#### Storing Topic-Aware Embeddings

```python
# Store embeddings with topics
topics = config.get_source_topics("security-docs")  # Get topics from config
embedding = embedding_generator.generate("security policy content")

db.store_embedding_with_topics(
    element_pk=123,
    embedding=embedding,
    topics=topics + ["authentication"],  # Combine source topics with content-specific topics
    confidence=0.95
)

# Get topics for a specific embedding
element_topics = db.get_embedding_topics(element_pk=123)
print(f"Element topics: {element_topics}")
```

### Relationship Detection

Doculyzer can detect various types of relationships between document elements:

- **Explicit Links**: Links explicitly defined in the document
- **Structural Relationships**: Parent-child, sibling, and section relationships
- **Semantic Relationships**: Connections based on content similarity

### Embedding Generation

Doculyzer uses advanced contextual embedding techniques to generate vector representations of document elements:

- **Pluggable Embedding Backends**: Choose from different embedding providers or implement your own
  - **HuggingFace Transformers**: Use transformer-based models like BERT, RoBERTa, or Sentence Transformers
  - **OpenAI Embeddings**: Leverage OpenAI's powerful embedding models
  - **FastEmbed**: Use the ultra-fast embedding library optimized for efficiency (15x faster than traditional models)
  - **Custom Embeddings**: Implement your own embedding generator with the provided interfaces
- **Contextual Embeddings**: Incorporates hierarchical relationships, predecessors, and successors into each element's embedding
- **Element-Level Precision**: Maintains accuracy to specific document elements rather than just document-level matching
- **Content-Optimized Vector Dimensions**: Flexibility to choose vector sizes based on content type
  - Larger vectors for highly technical content requiring more nuanced semantic representation
  - Smaller vectors for general content to optimize storage and query performance
  - Select the embedding provider and model that best suits your specific use case
- **Improved Relevance**: Context-aware embeddings produce more accurate similarity search results
- **Temporal Semantics**: Finds date references and expands them into a complete explanation of all date and time parts, improving ANN search.

#### Embedding Provider Comparison

| Provider | Speed | Quality | Dimension Options | Local/Remote | Installation |
|----------|-------|---------|-------------------|--------------|--------------|
| HuggingFace | Standard | High | 384-768 | Local | `pip install "doculyzer[huggingface]"` |
| OpenAI | Fast | Very High | 1536-3072 | Remote (API) | `pip install "doculyzer[openai]"` |
| FastEmbed | Very Fast (15x) | High | 384-1024 | Local | `pip install "doculyzer[fastembed]"` |

```python
from doculyzer.embeddings import get_embedding_generator
from doculyzer.embeddings.factory import create_embedding_generator

# Create embedding generator using configuration
embedding_generator = get_embedding_generator(config)

# Or manually create a specific embedding generator
huggingface_embedder = create_embedding_generator(
    provider="huggingface",
    model_name="sentence-transformers/all-mpnet-base-v2",
    dimensions=768,
    contextual=True
)

openai_embedder = create_embedding_generator(
    provider="openai",
    model_name="text-embedding-3-small",
    dimensions=1536,
    contextual=True,
    api_key="your-openai-api-key"
)

fastembed_embedder = create_embedding_generator(
    provider="fastembed",
    model_name="BAAI/bge-small-en-v1.5",
    dimensions=384,
    contextual=True,
    cache_dir="./model_cache"
)

# Generate embeddings for a document
elements = db.get_document_elements(doc_id)
embeddings = embedding_generator.generate_from_elements(elements)

# Store embeddings with topics
source_topics = config.get_source_topics(source_name)
for element_id, embedding in embeddings.items():
    element = db.get_element(element_id)
    if db.supports_topics():
        db.store_embedding_with_topics(element_id, embedding, source_topics)
    else:
        db.store_embedding(element_id, embedding)
```

#### Temporal Semantic Search Example

Doculyzer's temporal semantics feature provides lightweight consistency normalization to improve semantic search accuracy:

```python
from doculyzer import search_by_text

# Natural language date queries benefit from temporal consistency normalization
# When documents contain dates like "July 15, 2024", they get normalized to:
# "Q3 2024 July 2024 July 15, 2024"
# Time components like "14:30" become "14:30 14:30 2:30 PM business hours"

results = search_by_text(
    "Q3 strategic planning",
    include_topics=["planning%", "strategy%"],
    min_score=0.7,
    limit=10
)

print(f"Found {results.total_results} strategic planning documents")

for item in results.results:
    element = db.get_element(item.element_pk)
    print(f"Document: {element.get('title', 'Untitled')}")
    print(f"Similarity: {item.similarity:.3f}")
    print(f"Preview: {item.content_preview}")
    print("---")

# These searches now find semantically equivalent temporal references:
temporal_queries = [
    "Q3 planning",                    # Matches "July strategic planning", "third quarter review"
    "third quarter results",          # Matches "Q3 earnings", "July performance" 
    "July initiatives",              # Matches "Q3 projects", "third quarter goals"
    "summer strategy sessions",      # Matches "Q3 planning", "July meetings"
    "business hours meetings",       # Matches "14:30 conference", "2:30 PM calls"
]

for query in temporal_queries:
    results = search_by_text(query, limit=5)
    print(f"'{query}' found {results.total_results} results")
```

The lightweight temporal consistency works by:

1. **Temporal Detection**: Identifying date, time, datetime, and time range patterns in document content
2. **Canonical Normalization**: Converting temporal references to consistent canonical forms:
   - **Dates**: "July 15, 2024" → "Q3 2024 July 2024 July 15, 2024"
   - **Alternative dates**: "third quarter 2024" → "Q3 2024 July 2024 third quarter 2024"  
   - **Seasonal refs**: "summer planning" → "Q3 2024 July 2024 summer planning"
   - **Times**: "14:30" → "14:30 14:30 2:30 PM business hours"
3. **Consistent Matching**: Documents with equivalent temporal concepts now share canonical terms, improving semantic similarity

**Key Benefits:**
- **"Q3 planning"** searches find documents mentioning "July", "third quarter", or "summer" planning
- **"third quarter results"** searches find documents with "Q3", "July", or seasonal references
- **"business hours meetings"** searches find documents with various time formats (14:30, 2:30 PM, etc.)
- **Minimal overhead**: Only 2-3 canonical terms added per temporal reference
- **Natural language flow**: Maintains readable text that embedding models handle well
- **Consistent vector space**: Similar temporal concepts produce similar embeddings

### Handling Missing Dependencies

Doculyzer gracefully handles missing optional dependencies:

```python
# If you try to use an embedding provider without installing it:
from doculyzer.embeddings import get_embedding_generator

try:
    embedding_generator = get_embedding_generator(config)
    # Use the embedding generator...
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install the required package with:")
    print("pip install 'doculyzer[huggingface]'")  # or appropriate package
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Recommended Configurations

### Minimal Setup (File Content Sources Only)
```
pip install doculyzer
```

### Semantic Search with SQLite
```
pip install "doculyzer[db-core,fastembed]"
```

### High-Performance Search with Elasticsearch
```
pip install "doculyzer[db-elasticsearch,fastembed]"
```

### Production PostgreSQL with Database Content Sources
```
pip install "doculyzer[db-postgresql,source-database,fastembed]"
```

### Enterprise Configuration with All Content Sources
```
pip install "doculyzer[db-all,embedding-all,source-all,cloud-aws]"
```

### Topic-Aware Document Management
```
pip install "doculyzer[db-postgresql,fastembed]"
```

### Advanced Structured Search System
```
pip install "doculyzer[db-elasticsearch,huggingface]"
```

# Verified Compatibility

Tested and working with:
- SQLite storage (with and without vector search plugins)
- PostgreSQL storage (with and without pgvector extension)
- **Elasticsearch storage (with full-text search and vector capabilities)**
- Enhanced search capabilities (LIKE patterns, ElementType enums, metadata search)
- **Advanced structured search system with backend capability detection**
- Topic-aware embeddings and search
- Category-based element filtering
- Web Content Source
- File Content Source
- Database Content Source
- Content types: MD, HTML, XLSX, PDF, XML, CSV, DOCX, PPTX
- Embedding providers: HuggingFace, OpenAI, FastEmbed
- **Complex query composition with logical operators and similarity thresholds**
