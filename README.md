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
│ DOCX Documents  │     │Relationship Map │     │Graph Database │
└─────────────────┘     └─────────────────┘     └───────────────┘
```

## Key Features

- **Universal Document Model**: Common representation across document types
- **Preservation of Structure**: Maintains hierarchical document structure
- **Content Resolution**: Resolves pointers back to original content when needed
- **Contextual Semantic Search**: Uses advanced embedding techniques that incorporate document context (hierarchy, neighbors) for more accurate semantic search
- **Element-Level Precision**: Maintains granular accuracy to specific document elements
- **Relationship Mapping**: Identifies connections between document elements
- **Configurable Vector Representations**: Support for different vector dimensions based on content needs, allowing larger vectors for technical content and smaller vectors for general content
- **Modular Dependencies**: Only install the components you need, with graceful fallbacks when optional dependencies are missing

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

| Storage Backend | Description | Required Dependencies | Installation |
|-----------------|-------------|----------------------|--------------|
| File-based | Simple storage using the file system | None (core) | Default install |
| SQLite | Lightweight, embedded database | None (core) | Default install |
| SQLite Enhanced | SQLite with vector extension support | `sqlean.py` | `pip install "doculyzer[db-core]"` |
| Neo4J | Graph database with native relationship support | `neo4j` | `pip install "doculyzer[db-neo4j]"` |
| PostgreSQL | Robust relational database for production | `psycopg2` | `pip install "doculyzer[db-postgresql]"` |
| PostgreSQL + pgvector | PostgreSQL with vector search | `psycopg2`, `pgvector` | `pip install "doculyzer[db-postgresql,db-vector]"` |
| MongoDB | Document-oriented database | `pymongo` | `pip install "doculyzer[db-mongodb]"` |
| MySQL/MariaDB | Popular open-source SQL database | `sqlalchemy`, `pymysql` | `pip install "doculyzer[db-mysql]"` |
| Oracle | Enterprise SQL database | `sqlalchemy`, `cx_Oracle` | `pip install "doculyzer[db-oracle]"` |
| Microsoft SQL Server | Enterprise SQL database | `sqlalchemy`, `pymssql` | `pip install "doculyzer[db-mssql]"` |
| libSQL | SQLite-compatible distributed database | `libsql-client` | `pip install "doculyzer[db-libsql]"` |

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
    
# libSQL (requires libsql-client)
storage:
  backend: libsql
  libsql:
    url: "libsql://doculyzer.turso.io"
    auth_token: "your-auth-token"
```

### Using a Specific Database Backend

```python
from doculyzer.db import Neo4jDocumentDatabase, PostgreSQLDocumentDatabase

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

| Storage Backend | Vector Technology | Required Dependencies | Installation |
|-----------------|------------------|----------------------|--------------|
| SQLite + sqlite-vec | SIMD-accelerated vector search | `sqlean.py`, `sqlite-vec` | `pip install "doculyzer[db-core,db-vector]"` |
| PostgreSQL + pgvector | Postgres vector extension | `psycopg2`, `pgvector` | `pip install "doculyzer[db-postgresql,db-vector]"` |
| MongoDB Atlas | Vector search capability | `pymongo` | `pip install "doculyzer[db-mongodb]"` |
| Neo4j Vector Search | Graph + vector search | `neo4j` | `pip install "doculyzer[db-neo4j]"` |

```python
# Configure vector-capable storage
from doculyzer import Config

config = Config({
    "storage": {
        "backend": "postgresql",
        "postgresql": {
            "host": "localhost",
            "port": 5432,
            "database": "doculyzer",
            "user": "postgres",
            "password": "postgres",
            "vector_extension": "pgvector"  # Enable pgvector if available
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
    print("pip install 'doculyzer[db-postgresql,db-vector]'")
```

## Architecture

The system is built with a modular architecture:

1. **Content Sources**: Adapters for different content origins (with conditional dependencies)
2. **Document Parsers**: Transform content into structured elements (with format-specific dependencies)
3. **Document Database**: Stores metadata, elements, and relationships (with backend-specific dependencies)
4. **Content Resolver**: Retrieves original content when needed
5. **Embedding Generator**: Creates vector representations for semantic search (with model-specific dependencies)
6. **Relationship Detector**: Identifies connections between document elements

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
pip install "doculyzer[db-postgresql]"  # PostgreSQL support
pip install "doculyzer[db-mongodb]"     # MongoDB support
pip install "doculyzer[db-neo4j]"       # Neo4j support
pip install "doculyzer[db-mysql]"       # MySQL support
pip install "doculyzer[db-libsql]"      # libSQL support
pip install "doculyzer[db-core]"        # SQLite extensions + SQLAlchemy

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
  backend: sqlite  # Options: file, sqlite, mongodb, postgresql, sqlalchemy
  path: "./data"
  
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
    
  # Example of a blob-based database content source (requires sqlalchemy)
  - name: "database-blobs"
    type: "database"
    connection_string: "postgresql://user:password@localhost:5432/mydatabase"
    query: "SELECT * FROM documents"
    id_column: "doc_id"
    content_column: "content_blob" 
    metadata_columns: ["title", "author", "created_date"]
    timestamp_column: "updated_at"
    
  # Example of Confluence content source (requires atlassian-python-api)
  - name: "confluence-docs"
    type: "confluence"
    url: "https://company.atlassian.net/wiki"
    username: "${CONFLUENCE_USER}"  # Use environment variables securely
    password: "${CONFLUENCE_PASS}"
    space_keys: ["DEV", "PROD"]
    max_results: 1000

  # Example of S3 bucket content source (requires boto3)
  - name: "aws-documents"
    type: "s3"
    bucket: "company-documents"
    prefix: "technical-docs/"
    region: "us-west-2"
    file_pattern: "*.{pdf,docx,xlsx}"

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

# Load configuration
config = Config("config.yaml")

# Initialize storage
db = config.initialize_database()

# Ingest documents
stats = ingest_documents(config)
print(f"Processed {stats['documents']} documents with {stats['elements']} elements")

# Search documents
results = db.search_elements_by_content("search term")
for element in results:
    print(f"Found in {element['element_id']}: {element['content_preview']}")

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

# Store embeddings
for element_id, embedding in embeddings.items():
    db.store_embedding(element_id, embedding)
```

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

### Production PostgreSQL with Database Content Sources
```
pip install "doculyzer[db-postgresql,source-database,fastembed]"
```

### Enterprise Configuration with All Content Sources
```
pip install "doculyzer[db-all,embedding-all,source-all,cloud-aws]"
```

# Verified Compatibility

Tested and working with:
- SQLite storage (with and without vector search plugins)
- Web Content Source
- File Content Source
- Database Content Source
- Content types: MD, HTML, XLSX, PDF, XML, CSV, DOCX, PPTX
- Embedding providers: HuggingFace, OpenAI, FastEmbed
