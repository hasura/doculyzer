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

# Optional: Install embedding providers based on your needs
pip install sentence-transformers  # For HuggingFace embeddings
pip install openai                # For OpenAI embeddings
pip install fastembed             # For FastEmbed embeddings (new!)
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
from doculyzer.embeddings import get_embedding_generator

# Get the configured embedding generator
embedding_generator = get_embedding_generator(config)
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
| HuggingFace | Standard | High | 384-768 | Local | `pip install sentence-transformers` |
| OpenAI | Fast | Very High | 1536-3072 | Remote (API) | `pip install openai` |
| FastEmbed | Very Fast (15x) | High | 384-1024 | Local | `pip install fastembed` |

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
  # Provider options: "huggingface", "openai", "fastembed"
  provider: "huggingface"
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimensions: 384  # Embedding dimensions, used by database vector search
  
  # OpenAI embedding configuration
  # provider: "openai"  # Change from default huggingface to OpenAI
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
  
  # FastEmbed embedding configuration
  # provider: "fastembed"  # Use the new FastEmbed provider
  # model: "BAAI/bge-small-en-v1.5"  # FastEmbed model name
  # dimensions: 384  # Dimensions for the model
  # fastembed:
  #   cache_dir: "./model_cache"  # Where to cache downloaded models
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
  
  # More content sources as in the original config...
  # (Additional content sources omitted for brevity)

relationship_detection:
  enabled: true

logging:
  level: "INFO"
  file: "./logs/doculyzer.log"
```

# Verified Compatibility

Tested and working with:
- SQLite storage (with and without vector search plugins)
- Web Content Source
- File Content Source
- Content types: MD, HTML, XLSX, PDF, XML, CSV, DOCX, PPTX
- Embedding providers: HuggingFace, OpenAI, FastEmbed
