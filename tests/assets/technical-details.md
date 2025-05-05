---
title: Technical Details of Document Pointer System
author: Engineering Team
created: 2023-06-18
tags: [documentation, technical]
---

# Technical Details

This document provides in-depth technical information about the Document Pointer System.

## Data Model

The system uses a relational data model with the following key entities:

### Documents

Documents represent the root container for content from a source.

| Field | Description |
|-------|-------------|
| doc_id | Unique identifier |
| doc_type | Type (markdown, html, etc.) |
| source | Source identifier |
| metadata | Additional metadata |
| content_hash | Hash of original content |

### Elements

Elements are structural components of a document.

| Field | Description |
|-------|-------------|
| element_id | Unique identifier |
| doc_id | Parent document |
| element_type | Type (header, paragraph, etc.) |
| parent_id | Parent element |
| content_preview | Preview of content |
| content_location | Pointer to original content |

### Relationships

Relationships connect elements to each other.

| Field | Description |
|-------|-------------|
| relationship_id | Unique identifier |
| source_id | Source element |
| relationship_type | Type of relationship |
| target_reference | Target element or reference |
| metadata | Additional metadata |

## Embedding Generation

The system uses [Sentence Transformers](https://www.sbert.net/) to generate vector embeddings for semantic search. The default model is `all-MiniLM-L6-v2`, which provides a good balance of performance and quality.

## Parsing Process

When a document is ingested, it goes through several stages:

1. Content is retrieved from the source
2. The appropriate parser is selected based on content type
3. The parser extracts structural elements and links
4. Embeddings are generated for each element
5. Relationships are detected between elements
6. All data is stored in the document database

See the [Introduction](introduction.md) for a high-level overview of the system architecture.
