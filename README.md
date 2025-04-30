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
