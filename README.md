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

Doculyzer supports multiple content sources:
- File systems (local, mounted, and network shares)
- HTTP endpoints
- Confluence
- JIRA
- Amazon S3
- Relational Databases
- ServiceNow
- MongoDB

## Architecture

The system is built with a modular architecture:

1. **Content Sources**: Adapters for different content origins
2. **Document Parsers**: Transform content into structured elements
3. **Document Database**: Stores metadata, elements, and relationships
4. **Content Resolver**: Retrieves original content when needed
5. **Embedding Generator**: Creates vector representations for semantic search
6. **Relationship Detector**: Identifies connections between document elements

## Storage Backends

Doculyzer supports multiple storage backends:
- **File-based storage**: Simple storage using the file system
- **SQLite**: Lightweight, embedded database
- **Neo4J**: Graph datastore, all document elements, relationships and embeddings are stored
- **PostgreSQL**: Robust relational database for production deployments
- **MongoDB**: Document-oriented database for larger deployments
- **SQLAlchemy**: ORM layer supporting multiple relational databases:
  - MySQL/MariaDB
  - Oracle
  - Microsoft SQL Server
  - And other SQLAlchemy-compatible databases

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

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/doculyzer.git
cd doculyzer

# Install dependencies
pip install -r requirements.txt
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
  model: "sentence-transformers/all-MiniLM-L6-v2"
  backend: "huggingface"  # Options: huggingface, openai, custom
  chunk_size: 512
  overlap: 128
  contextual: true  # Enable contextual embeddings
  vector_size: 384  # Configurable based on content needs
  
  # Contextual embedding configuration
  predecessor_count: 1
  successor_count: 1
  ancestor_depth: 1
  
  # Content-specific configurations
  content_types:
    technical:
      model: "sentence-transformers/all-mpnet-base-v2"
      vector_size: 768  # Larger vectors for technical content
    general:
      model: "sentence-transformers/all-MiniLM-L6-v2"
      vector_size: 384  # Smaller vectors for general content
  
  # OpenAI-specific configuration (if using OpenAI backend)
  openai:
    api_key: "your_api_key_here"
    model: "text-embedding-ada-002"
    dimensions: 1536  # Configurable embedding dimensions

content_sources:
  - name: "documentation"
    type: "file"
    base_path: "./docs"
    file_pattern: "**/*.md"
    max_link_depth: 2

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
query_embedding = embedding_generator.generate("search query")
results = db.search_by_embedding(query_embedding)
for element_id, score in results:
    element = db.get_element(element_id)
    print(f"Semantic match ({score:.2f}): {element['content_preview']}")
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
  - **Custom Embeddings**: Implement your own embedding generator with the provided interfaces
- **Contextual Embeddings**: Incorporates hierarchical relationships, predecessors, and successors into each element's embedding
- **Element-Level Precision**: Maintains accuracy to specific document elements rather than just document-level matching
- **Content-Optimized Vector Dimensions**: Flexibility to choose vector sizes based on content type
  - Larger vectors for highly technical content requiring more nuanced semantic representation
  - Smaller vectors for general content to optimize storage and query performance
  - Select the embedding provider and model that best suits your specific use case
- **Improved Relevance**: Context-aware embeddings produce more accurate similarity search results
- **Temporal Semantics**: Finds date references and expands them into a complete explanation of all date and time parts, improving ANN search.

```python
from doculyzer.embeddings import get_embedding_generator

# Create contextual embedding generator with the configured backend
embedding_generator = get_embedding_generator(config)

# Use a specific embedding backend
from doculyzer.embeddings.factory import create_embedding_generator
from doculyzer.embeddings.hugging_face import HuggingFaceEmbedding

# Create a HuggingFace embedding generator with a specific model and vector size
embedding_generator = create_embedding_generator(
    backend="huggingface",
    model_name="sentence-transformers/all-mpnet-base-v2",
    vector_size=768,  # Larger vector size for technical content
    contextual=True
)

# Or choose a different model with smaller vectors for general content
general_content_embedder = create_embedding_generator(
    backend="huggingface",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    vector_size=384,  # Smaller vector size for general content
    contextual=True
)

# Generate embeddings for a document
elements = db.get_document_elements(doc_id)
embeddings = embedding_generator.generate_from_elements(elements)

# Store embeddings
for element_id, embedding in embeddings.items():
    db.store_embedding(element_id, embedding)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Sample Config

```yaml
storage:
  backend: neo4j  # Can be neo4j, sqlite, file, mongodb, postgresql, or sqlalchemy
  path: "./data"  
  
  # Neo4j-specific configuration
  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "password"
    database: "doculyzer"
    
  # File-based storage configuration (uncomment to use)
  # backend: file
  # path: "./data"
  # file:
  #   # Options for organizing file storage
  #   subdirectory_structure: "flat"  # Can be "flat" or "hierarchical"
  #   create_backups: true  # Whether to create backups before overwriting files
  #   backup_count: 3  # Number of backups to keep
  #   compression: false  # Whether to compress stored files
  #   index_in_memory: true  # Whether to keep indexes in memory for faster access
  
  # MongoDB-based storage configuration (uncomment to use)
  # backend: mongodb
  # mongodb:
  #   host: "localhost"
  #   port: 27017
  #   username: "admin"  # Optional
  #   password: "password"  # Optional
  #   db_name: "doculyzer"
  #   options:  # Optional connection options
  #     retryWrites: true
  #     w: "majority"
  #     connectTimeoutMS: 5000
  #   vector_search: true  # Whether to use vector search capabilities (requires MongoDB Atlas)
  #   create_vector_index: true  # Whether to create vector search index on startup
  
  # PostgreSQL-based storage configuration (uncomment to use)
  # backend: postgresql
  # postgresql:
  #   host: "localhost"
  #   port: 5432
  #   dbname: "doculyzer"
  #   user: "postgres"
  #   password: "password"
  #   # Optional SSL configuration
  #   sslmode: "prefer"  # Options: disable, prefer, require, verify-ca, verify-full
  #   # Vector search configuration using pgvector
  #   enable_vector: true  # Whether to try to enable pgvector extension
  #   create_vector_index: true  # Whether to create vector indexes automatically
  #   vector_index_type: "ivfflat"  # Options: ivfflat, hnsw
  
  # SQLAlchemy-based storage configuration (uncomment to use)
  # backend: sqlalchemy
  # sqlalchemy:
  #   # Database URI (examples for different database types)
  #   # SQLite:
  #   db_uri: "sqlite:///data/doculyzer.db"
  #   # PostgreSQL:
  #   # db_uri: "postgresql://user:password@localhost:5432/doculyzer"
  #   # MySQL:
  #   # db_uri: "mysql+pymysql://user:password@localhost:3306/doculyzer"
  #   # MS SQL Server:
  #   # db_uri: "mssql+pyodbc://user:password@server/database?driver=ODBC+Driver+17+for+SQL+Server"
  #   
  #   # Additional configuration options
  #   echo: false  # Whether to echo SQL statements for debugging
  #   pool_size: 5  # Connection pool size
  #   max_overflow: 10  # Maximum overflow connections
  #   pool_timeout: 30  # Connection timeout in seconds
  #   pool_recycle: 1800  # Connection recycle time in seconds
  #   
  #   # Vector extensions
  #   vector_extension: "auto"  # Options: auto, pgvector, sqlite_vss, sqlite_vec, none
  #   create_vector_index: true  # Whether to create vector indexes automatically
  #   vector_index_type: "ivfflat"  # For PostgreSQL: ivfflat, hnsw
  
  # SQLite-based storage configuration (uncomment to use)
  # backend: sqlite
  # path: "./data"  # Path where the SQLite database file will be stored
  # sqlite:
  #   # Extensions configuration
  #   sqlite_extensions:
  #     use_sqlean: true  # Whether to use sqlean.py (provides more SQLite extensions)
  #     auto_discover: true  # Whether to automatically discover and load vector extensions
  #   
  #   # Vector search configuration
  #   vector_extensions:
  #     preferred: "auto"  # Options: auto, vec0, vss0, none
  #     create_tables: true  # Whether to create vector tables on startup
  #     populate_existing: true  # Whether to populate vector tables with existing embeddings

embedding:
  enabled: true
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimensions: 384  # Embedding dimensions, used by database vector search
  
  # OpenAI embedding configuration (uncomment to use)
  # provider: "openai"  # Change from default sentence-transformers to OpenAI
  # model: "text-embedding-3-small"  # OpenAI embedding model name
  # dimensions: 1536  # Dimensions for the model (1536 for text-embedding-3-small, 3072 for text-embedding-3-large)
  # openai:
  #   api_key: "your-openai-api-key"  # Can also be set via OPENAI_API_KEY environment variable
  #   batch_size: 10  # Number of texts to embed in a single API call
  #   retry_count: 3  # Number of retries on API failure
  #   retry_delay: 1  # Delay between retries in seconds
  #   timeout: 60  # Timeout for API calls in seconds
  #   max_tokens: 8191  # Maximum tokens per text (8191 for text-embedding-3-small/large)
  #   cache_enabled: true  # Whether to cache embeddings
  #   cache_size: 1000  # Maximum number of embeddings to cache in memory

content_sources:
  # File content source
  - name: "local-files"
    type: "file"
    base_path: "./documents"
    file_pattern: "**/*"
    include_extensions: ["md", "txt", "pdf", "docx", "html"]
    exclude_extensions: ["tmp", "bak"]
    watch_for_changes: true
    recursive: true
    max_link_depth: 2
  
  # JIRA content source
  - name: "project-tickets"
    type: "jira"
    base_url: "https://your-company.atlassian.net"
    username: "jira_user@example.com"
    api_token: "your-jira-api-token"
    projects: ["PROJ", "FEAT"]
    issue_types: ["Bug", "Story", "Task"]
    statuses: ["In Progress", "To Do", "In Review"] 
    include_closed: false
    max_results: 100
    include_description: true
    include_comments: true
    include_attachments: false
    include_subtasks: true
    include_linked_issues: true
    include_custom_fields: ["customfield_10001", "customfield_10002"]
    max_link_depth: 1
  
  # MongoDB content source
  - name: "document-db"
    type: "mongodb"
    connection_string: "mongodb://localhost:27017/"
    database_name: "your_database"
    collection_name: "documents"
    query: {"status": "active"}
    projection: {"_id": 1, "title": 1, "content": 1, "metadata": 1, "updated_at": 1}
    id_field: "_id"
    content_field: "content"
    timestamp_field: "updated_at"
    limit: 1000
    sort_by: [["updated_at", -1]]
    follow_references: true
    reference_field: "related_docs"
    max_link_depth: 1
    
  # S3 content source
  - name: "cloud-documents"
    type: "s3"
    bucket_name: "your-document-bucket"
    prefix: "documents/"
    region_name: "us-west-2"
    aws_access_key_id: "your-access-key"
    aws_secret_access_key: "your-secret-key"
    assume_role_arn: "arn:aws:iam::123456789012:role/S3AccessRole"  # Optional
    endpoint_url: null  # For S3-compatible storage
    include_extensions: ["md", "txt", "pdf", "docx", "html"]
    exclude_extensions: ["tmp", "bak", "log"]
    include_prefixes: ["documents/important/", "documents/shared/"]  # Optional
    exclude_prefixes: ["documents/archive/", "documents/backup/"]  # Optional
    include_patterns: []  # Optional regex patterns
    exclude_patterns: []  # Optional regex patterns
    recursive: true
    max_depth: 5
    detect_mimetype: true
    temp_dir: "/tmp"
    delete_after_processing: true
    local_link_mode: "relative"  # Can be relative, absolute, or none
    max_link_depth: 2
    
  # ServiceNow content source
  - name: "it-service-management"
    type: "servicenow"
    base_url: "https://your-instance.service-now.com"
    username: "servicenow_user"
    api_token: "your-servicenow-api-token"
    # Alternatively, use password authentication
    # password: "your-password"
    
    # Content type settings
    include_knowledge: true
    include_incidents: true
    include_service_catalog: true
    include_cmdb: true
    
    # Filter settings
    knowledge_query: "workflow_state=published"  # ServiceNow knowledge API query
    incident_query: "active=true^priority<=2"  # ServiceNow table API query for incidents
    service_catalog_query: "active=true^category=hardware"  # Query for service catalog items
    cmdb_query: "sys_class_name=cmdb_ci_server"  # Query for CMDB items
    include_patterns: [".*prod.*", ".*critical.*"]  # Regex patterns to include
    exclude_patterns: [".*test.*", ".*dev.*"]  # Regex patterns to exclude
    limit: 100  # Maximum number of items to retrieve
    max_link_depth: 1  # For following links between ServiceNow items
    
  # Web content source
  - name: "web-content"
    type: "web"
    base_url: "https://www.example.com"  # Optional base URL for relative paths
    url_list:  # List of URLs to fetch
      - "https://www.example.com/docs/overview"
      - "https://www.example.com/docs/tutorials"
      - "https://www.example.com/blog"
    url_list_file: "./urls.txt"  # Optional path to file containing URLs (one per line)
    refresh_interval: 86400  # Refresh interval in seconds (default: 1 day)
    headers:  # Custom headers for requests
      User-Agent: "Mozilla/5.0 (compatible; DoculyzerBot/1.0)"
      Accept-Language: "en-US,en;q=0.9"
    authentication:  # Optional authentication
      type: "basic"  # Can be "basic" or "bearer"
      username: "web_user"
      password: "web_password"
      # For bearer token:
      # type: "bearer"
      # token: "your-access-token"
    include_patterns:  # Regex patterns to include when following links
      - "/docs/.*"
      - "/blog/[0-9]{4}/.*"
    exclude_patterns:  # Regex patterns to exclude when following links
      - ".*\\.pdf$"
      - "/archived/.*"
    max_link_depth: 3  # Maximum depth for following links
    
  # Confluence content source
  - name: "team-knowledge-base"
    type: "confluence"
    base_url: "https://your-company.atlassian.net"
    username: "confluence_user@example.com"
    api_token: "your-confluence-api-token"
    # Alternatively, use password authentication
    # password: "your-password"
    
    # Space configuration
    spaces: ["TEAM", "PROJ", "DOCS"]  # List of space keys to include (empty list fetches all accessible spaces)
    exclude_personal_spaces: true  # Skip personal spaces when fetching all spaces
    
    # Content type settings
    include_pages: true  # Include regular Confluence pages
    include_blogs: true  # Include blog posts
    include_comments: false  # Include comments on pages/blogs
    include_attachments: false  # Include file attachments
    
    # Content filtering
    include_patterns: ["^Project.*", "^Guide.*"]  # Regex patterns to include (matches against title)
    exclude_patterns: ["^Draft.*", "^WIP.*"]  # Regex patterns to exclude
    
    # Advanced settings
    expand_macros: true  # Expand Confluence macros in content
    link_pattern: "/wiki/spaces/[^/]+/pages/(\\d+)"  # Regex pattern to extract page IDs from links
    limit: 500  # Maximum number of items to retrieve
    max_link_depth: 2  # For following links between Confluence pages
  
  # Example of a blob-based database content source
  - name: "database-blobs"
    type: "database"
    connection_string: "postgresql://user:password@localhost:5432/mydatabase"
    query: "SELECT * FROM documents"
    id_column: "doc_id"
    content_column: "content_blob" 
    metadata_columns: ["title", "author", "created_date"]
    timestamp_column: "updated_at"
    
  # Example of a JSON-structured database content source
  - name: "database-json"
    type: "database"
    connection_string: "mysql://user:password@localhost:3306/customer_db"
    query: "SELECT * FROM customers"
    id_column: "customer_id"
    json_mode: true
    json_columns: ["first_name", "last_name", "email", "address", "phone_number", "signup_date"]
    metadata_columns: ["account_status", "customer_type"]
    timestamp_column: "last_modified"
    json_include_metadata: true  # Include metadata columns in the JSON document
    
  # Example of automatic column discovery (all columns except ID will be in JSON)
  - name: "database-json-auto"
    type: "database"
    connection_string: "sqlite:///local_database.db"
    query: "products"  # Simple table name
    id_column: "product_id"
    json_mode: true
    # No json_columns specified - will automatically use all non-ID columns
    metadata_columns: ["category", "supplier"]
    timestamp_column: "updated_at"
    
  # Example with a complex query
  - name: "database-complex-query"
    type: "database"
    connection_string: "mssql+pyodbc://user:password@server/database"
    query: "SELECT o.order_id, c.customer_name, o.order_date, p.product_name, oi.quantity 
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN products p ON oi.product_id = p.product_id"
    id_column: "order_id"
    json_mode: true
    json_columns: ["customer_name", "order_date", "product_name", "quantity"]
    timestamp_column: "order_date"

relationship_detection:
  enabled: true

logging:
  level: "INFO"
  file: "./logs/doculyzer.log"
```

# Here's what I've tested so far
- SQLite storage (with and within vector search plugins)
- Web Content Source
- File Content Source
- Content type: MD, HTML, XLSX, PDF, XML, CSV, DOCX, PPTX

