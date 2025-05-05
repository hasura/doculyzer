import logging
import os

from dotenv import load_dotenv

from .config import Config
from .embeddings import EmbeddingGenerator

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

_config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))


def ingest_documents(config: Config, source_configs=None, max_link_depth=None):
    """
    Ingest documents from configured content sources, including following links to specified depth.
    Uses global visited tracking to prevent duplicate processing across all sources.
    Skip documents that haven't changed since their last processing.

    Args:
        config: Configuration object
        source_configs: Optional list of content source configs (overrides config)
        max_link_depth: Optional override for link depth (overrides source config)

    Returns:
        Dictionary with statistics about ingested documents
    """
    from .content_source.factory import get_content_source
    from .embeddings import get_embedding_generator
    from .relationships import create_relationship_detector

    logger.debug("Starting document ingestion process")

    # Ensure config is a Config instance
    if not isinstance(config, Config):
        from doculyzer import Config as ConfigClass
        if isinstance(config, dict):
            # Create Config instance from dictionary
            config_obj = ConfigClass()
            config_obj.config = config
            config = config_obj
        elif isinstance(config, str):
            # Create Config instance from path
            config = ConfigClass(config)
        else:
            raise ValueError("config must be a Config instance, dictionary, or path string")

    # 1. Document database
    logger.debug("Getting document database from config")
    db = config.get_document_database()
    logger.debug(f"Using database instance: {db}")

    # 2. Embedding generator (if enabled)
    embedding_generator = None
    if config.is_embedding_enabled():
        logger.debug("Initializing embedding generator")
        embedding_generator = get_embedding_generator(config)
        logger.info(f"Initialized embedding generator with model {config.get_embedding_model()}")
        logger.debug(f"Embedding generator details: {embedding_generator}")
    else:
        logger.debug("Embeddings not enabled in config")

    # 3. Relationship detector
    logger.debug("Creating relationship detector")
    relationship_detector = create_relationship_detector(
        config.get_relationship_detection_config(),
        embedding_generator
    )
    logger.info("Initialized relationship detector")
    logger.debug(f"Relationship detector: {relationship_detector}")

    # Track statistics
    stats = {
        "documents": 0,
        "elements": 0,
        "relationships": 0,
        "unchanged_documents": 0  # New counter for unchanged documents
    }

    # Get content sources to process
    sources_to_process = source_configs or config.get_content_sources()
    logger.debug(f"Processing {len(sources_to_process)} content sources")

    # Global set to track ALL visited documents across all sources
    global_visited_docs = set()
    logger.debug("Initialized global visited document tracking")

    # Process each content source
    processed_docs = set()  # Track fully processed docs within this source
    for idx, source_config in enumerate(sources_to_process):
        source_type = source_config.get('type')
        source_name = source_config.get('name')
        logger.debug(f"Processing source {idx + 1}/{len(sources_to_process)}: {source_name} ({source_type})")

        # Override max_link_depth if specified
        if max_link_depth is not None:
            original_depth = source_config.get('max_link_depth', 1)
            source_config['max_link_depth'] = max_link_depth
            logger.debug(f"Overriding max_link_depth from {original_depth} to {max_link_depth}")

        # Create content source
        logger.debug(f"Creating content source for {source_name}")
        source = get_content_source(source_config)
        logger.debug(f"Content source created: {source}")

        # Get document list
        logger.debug(f"Listing documents from source {source_name}")
        documents = source.list_documents()
        logger.info(f"Found {len(documents)} documents in source {source_name}")

        # Process each document
        logger.debug(f"Starting to process {len(documents)} documents from source {source_name}")

        for doc_idx, doc in enumerate(documents):
            doc_id = doc['id']
            logger.debug(f"Processing document {doc_idx + 1}/{len(documents)}: {doc_id}")
            _ingest_document_recursively(
                source, doc_id, db, relationship_detector, embedding_generator,
                processed_docs, stats, source_config.get('max_link_depth', 1),
                global_visited_docs
            )
            logger.debug(f"Completed document {doc_idx + 1}/{len(documents)}: {doc_id}")

        logger.debug(f"Completed processing source {source_name}")

    # Generate cross-document container relationships if embedding is enabled
    if embedding_generator and stats['documents'] > 0:
        logger.info("Computing cross-document container relationships")

        # Fix: processed_docs is a set of processed document IDs, not a dictionary
        processed_doc_ids = list(global_visited_docs.intersection(processed_docs))

        logger.info(f"Found {len(processed_doc_ids)} changed documents for relationship analysis")
        relationship_count = _compute_cross_document_container_relationships(db, processed_doc_ids)
        stats['semantic_relationships'] = relationship_count

    # Log summary
    logger.info(
        f"Processed {stats['documents']} documents with {stats['elements']} elements "
        f"and {stats['relationships']} relationships"
    )
    if stats['unchanged_documents'] > 0:
        logger.info(f"Skipped {stats['unchanged_documents']} unchanged documents")

    logger.debug(f"Total unique documents visited: {len(global_visited_docs)}")
    logger.debug("Document ingestion process completed")

    return stats


def _ingest_document_recursively(source, doc_id, db, relationship_detector,
                                 embedding_generator: EmbeddingGenerator, processed_docs, stats,
                                 max_depth, global_visited_docs, current_depth=0):
    """
    Recursively ingest a document and its linked documents with global visited tracking.
    Skip documents that haven't changed since their last processing.

    Args:
        source: Content source
        doc_id: Document ID
        db: Document database
        relationship_detector: Relationship detector
        embedding_generator: Embedding generator
        processed_docs: Set of processed document IDs for current source
        stats: Statistics dictionary to update
        max_depth: Maximum link depth to follow
        global_visited_docs: Global set of all visited document IDs
        current_depth: Current depth in the recursion
    """
    from .document_parser.factory import get_parser_for_content

    logger.debug(f"Recursively ingesting document: {doc_id} (depth: {current_depth}/{max_depth})")

    # Skip if already processed in this source
    if doc_id in processed_docs:
        logger.debug(f"Skipping already processed document in this source: {doc_id}")
        return

    # Skip if already visited globally (except for root documents at depth 0)
    if current_depth > 0 and doc_id in global_visited_docs:
        logger.debug(f"Skipping globally visited document: {doc_id}")
        return

    # Skip if max depth reached
    if current_depth > max_depth:
        logger.debug(f"Skipping document due to max depth reached: {doc_id} (depth: {current_depth}/{max_depth})")
        return

    # Check if document has already been processed in a previous run and hasn't changed
    last_processed_info = db.get_last_processed_info(doc_id)
    if last_processed_info:
        try:
            # If we have processing history for this document, check if it has changed
            if not source.has_changed(doc_id, last_processed_info.get("last_modified")):
                logger.debug(f"Document unchanged since last processing: {doc_id}")
                stats['unchanged_documents'] += 1

                # Still mark as visited to prevent following the same links again
                processed_docs.add(doc_id)
                global_visited_docs.add(doc_id)

                return

            # If we have content hash, we could also compare that
            # This is useful for sources where modification time isn't reliable
            if "content_hash" in last_processed_info:
                # Peek at the content hash without fully processing
                try:
                    # Get just enough info to check the content hash
                    doc_content_peek = source.fetch_document(doc_id)
                    if doc_content_peek.get("content_hash") == last_processed_info["content_hash"]:
                        logger.debug(f"Document content unchanged (verified by hash): {doc_id}")
                        stats['unchanged_documents'] += 1

                        # Still mark as visited to prevent following the same links again
                        processed_docs.add(doc_id)
                        global_visited_docs.add(doc_id)

                        return
                except Exception as e:
                    logger.warning(f"Error peeking at document content: {str(e)}")
                    # Continue with full processing
        except Exception as e:
            logger.warning(f"Error checking if document has changed: {str(e)}")
            # Continue with full processing

    # Mark as processed in this source and globally visited
    processed_docs.add(doc_id)
    global_visited_docs.add(doc_id)
    logger.debug(f"Marked document as processed and globally visited: {doc_id}")

    try:
        # Fetch document content
        logger.debug(f"Fetching content for document: {doc_id}")
        doc_content = source.fetch_document(doc_id)
        logger.debug(f"Document content fetched, size: {len(doc_content.get('content', ''))}")

        # Create parser
        logger.debug(f"Creating parser for document: {doc_id}")
        parser = get_parser_for_content(doc_content)
        logger.debug(f"Parser created: {parser.__class__.__name__}")

        # Parse document
        logger.debug(f"Parsing document: {doc_id}")
        parsed_doc = parser.parse(doc_content)
        logger.debug(f"Document parsed. Found {len(parsed_doc.get('elements', []))} elements")

        # Detect relationships
        links = parsed_doc.get('links', [])
        relationships = parsed_doc.get('relationships', [])
        logger.debug(f"Detecting relationships. Found {len(links)} links in document")
        relationships.extend(relationship_detector.detect_relationships(
            parsed_doc['document'],
            parsed_doc['elements'],
            links
        ))
        logger.debug(f"Detected {len(relationships)} relationships")

        # Store document
        logger.debug(f"Storing document in database: {doc_id}")
        db.store_document(parsed_doc['document'], parsed_doc['elements'], relationships)
        logger.debug(f"Document stored: {doc_id}")

        # Update processing history
        content_hash = doc_content.get("content_hash", "")
        if content_hash:
            db.update_processing_history(doc_id, content_hash)

        # Generate embeddings if enabled
        if embedding_generator:
            logger.debug(f"Generating embeddings for {len(parsed_doc['elements'])} elements")

            # Generate embeddings using a consistent interface
            embeddings = embedding_generator.generate_from_elements(parsed_doc['elements'])

            # Store embeddings
            for element_id, embedding in embeddings.items():
                db.store_embedding(element_id, embedding)

            logger.debug(f"Generated and stored {len(embeddings)} embeddings")

        # Update statistics
        stats['documents'] += 1
        stats['elements'] += len(parsed_doc['elements'])
        stats['relationships'] += len(relationships)
        logger.debug(
            f"Updated stats: docs={stats['documents']}, elements={stats['elements']}, relationships={stats['relationships']}")

        # Follow links if not at max depth
        if current_depth < max_depth:
            logger.debug(f"Following links for document {doc_id} at depth {current_depth}")
            # Use source's follow_links method to get linked documents, passing global visited set
            linked_docs = source.follow_links(
                doc_content['content'],
                doc_id,
                current_depth,
                global_visited_docs
            )
            logger.debug(f"Found {len(linked_docs)} linked documents to follow")

            # Process each linked document
            for link_idx, linked_doc in enumerate(linked_docs):
                linked_id = linked_doc['id']
                logger.debug(f"Processing linked document {link_idx + 1}/{len(linked_docs)}: {linked_id}")
                _ingest_document_recursively(
                    source, linked_id, db, relationship_detector,
                    embedding_generator, processed_docs, stats,
                    max_depth, global_visited_docs, current_depth + 1
                )
                logger.debug(f"Completed processing linked document: {linked_id}")
        else:
            logger.debug(f"Not following links due to max depth: {current_depth}/{max_depth}")

        logger.debug(f"Successfully completed processing document: {doc_id}")

    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {str(e)}")
        import traceback
        logger.debug(f"Exception traceback for {doc_id}: {traceback.format_exc()}")


def _compute_cross_document_container_relationships(db, processed_doc_ids):
    """
    Compute semantic relationships between containers across documents.

    Args:
        db: DocumentDatabase instance
        processed_doc_ids: List of document IDs that were processed

    Returns:
        Number of relationships created
    """
    logger.info(f"Computing cross-document container relationships for {len(processed_doc_ids)} documents")

    # Container element types we're interested in
    container_types = ["body", "div", "list", "header", "section", "title", "h1", "h2", "h3", "h4", "h5", "h6"]
    similarity_threshold = (_config.config.get('relationship_detection', {})
                            .get('cross_document_semantic', {}).get('similarity_threshold'))
    if similarity_threshold is None:
        return

    # Get all container elements from processed documents
    processed_containers = []

    for doc_id in processed_doc_ids:
        # Get elements from the document
        elements = db.get_document_elements(doc_id)

        # Filter for container elements
        containers = [e for e in elements if e["element_type"] in container_types]

        # Store with document context
        for container in containers:
            processed_containers.append({
                "element": container,
                "doc_id": doc_id
            })

    logger.debug(f"Found {len(processed_containers)} container elements in processed documents")

    # Delete existing semantic relationships for processed elements
    for container_info in processed_containers:
        container = container_info["element"]
        element_id = container["element_id"]
        db.delete_relationships_for_element(element_id, "semantic_section")

    # Compute new relationships
    new_relationships = []
    relationship_count = 0

    # Create a map of processed containers by element_id for quick lookup
    processed_container_map = {
        container_info["element"]["element_id"]: container_info
        for container_info in processed_containers
    }

    # For each processed container
    for container_info in processed_containers:
        container = container_info["element"]
        source = container_info["doc_id"]
        element_pk = container["element_pk"]

        # Get embedding
        embedding = db.get_embedding(element_pk)
        if not embedding:
            continue

        # Search for similar containers in other documents
        filter_criteria = {
            "element_type": container_types,
            "exclude_doc_source": [source]  # Exclude containers from the same document
        }

        similar_containers = db.search_by_embedding(
            embedding,
            limit=20,
            filter_criteria=filter_criteria
        )

        # Process results
        for target_id, similarity in similar_containers:
            # Skip if similarity is below threshold
            if similarity < similarity_threshold:
                continue

            # Skip if this is another processed container we've already compared to
            if target_id in processed_container_map:
                # # Only create the relationship once (when we process the first element)
                # if target_id < element_id:
                #     continue
                target_element = db.get_element(target_id)
                target_doc_id = processed_container_map[target_id]["doc_id"]
            else:
                # Get document ID for this container
                target_element = db.get_element(target_id)
                if not target_element:
                    continue

                target_doc_id = target_element["doc_id"]

            # Create relationship
            new_relationships.append({
                "relationship_id": f"sem_rel_{element_id}_{target_element['element_id']}",
                "source_id": element_id,
                "relationship_type": "semantic_section",
                "target_reference": target_element['element_id'],
                "metadata": {
                    "similarity_score": similarity,
                    "cross_document": True,
                    "source_doc_id": source,
                    "target_doc_id": target_doc_id
                }
            })
            relationship_count += 1

    # Store all new relationships
    for relationship in new_relationships:
        db.store_relationship(relationship)

    logger.info(f"Created {relationship_count} cross-document semantic relationships")
    return relationship_count
