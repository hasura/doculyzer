"""
functions.py

This is an example of how you can use the Python SDK's built-in Function connector to easily write Python code.
When you add a Python Lambda connector to your Hasura project, this file is generated for you!

In this file you'll find code examples that will help you get up to speed with the usage of the Hasura lambda connector.
If you are an old pro and already know what is going on you can get rid of these example functions and start writing your own code.
"""
import json
from typing import List, Optional, Dict, Any
import os
import asyncio
import aiohttp

from hasura_ndc import start
from hasura_ndc.function_connector import FunctionConnector
from hasura_ndc.instrumentation import \
    with_active_span  # If you aren't planning on adding additional tracing spans, you don't need this!
from opentelemetry.trace import \
    get_tracer, \
    get_current_span  # If you aren't planning on adding additional tracing spans, you don't need this either!
from pydantic import \
    Field, BaseModel  # You only need this import if you plan to have complex inputs/outputs, which function similar to how frameworks like FastAPI do

# Define a minimal ElementFlat type for our return values
class ElementFlat(BaseModel):
    element_pk: int = Field(
        description="Primary key of the element in the document store. Used for direct element retrieval and unique identification."
    )
    score: Optional[float] = Field(
        default=None,
        description="Semantic similarity score (-1 to 1) indicating how closely the content matches conceptually. Higher scores mean more relevant matches."
    )
    element_id: str = Field(
        default="",
        description="Unique identifier of the specific matching element within its document. Used for element-level referencing and relationships."
    )
    element_type: str = Field(
        default="",
        description="Type of document element (e.g., section, paragraph, table, list) indicating the structural context of the match. Helps understand content organization."
    )
    content_preview: str = Field(
        default="",
        description="Abbreviated preview of the matching content, suitable for display. May be truncated for large elements. Use text field for complete content."
    )
    doc_id: str = Field(
        default="",
        description="Unique identifier of the document containing this match. Use to locate or reference source documents."
    )
    content_location: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Location details of the content within its document, such as page numbers, sections, or coordinates. Format varies by document type."
    )
    source: Optional[str] = Field(
        default=None,
        description="Origin or system source of the document. Useful for tracking document provenance and filtering results by system."
    )
    text: Optional[str] = Field(
        default=None,
        description="Complete text content of the matching element when resolve_text is true. Provides full context of the match without structural formatting."
    )
    content: Optional[str] = Field(
        default=None,
        description="Complete structured content of the matching element when resolve_content is true. Preserves document-specific formatting and structure."
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="Identifier of the element's parent when include_parents is true. Used to understand containment relationships and document hierarchy."
    )
    path: Optional[str] = Field(
        default=None,
        description="Full path showing element's location in document hierarchy. Format depends on document type. Useful for understanding context and navigation."
    )
    content_hash: Optional[str] = Field(
        default=None,
        description="Hash identifier of the element's content. Useful for tracking content changes, versioning, or deduplication."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional document or element metadata key-value pairs. May include custom attributes, tags, or system-specific information."
    )

connector = FunctionConnector()

# This last section shows you how to add OTEL tracing to any of your functions!
tracer = get_tracer("document_search.server") # You only need a tracer if you plan to add additional Otel spans

# Configuration for the web server
SEARCH_SERVER_URL = os.environ.get('DOCUMENTS_URI', 'http://localhost:5000')
SEARCH_API_KEY = os.environ.get('SEARCH_API_KEY')  # Optional API key

@connector.register_query
async def search_documents(
        search_for: str,
        include_parents: Optional[bool] = Field(
            default=None,
            description="Include containing elements (e.g., sections containing matching paragraphs) to provide fuller context. Parent elements help understand where matches fit in the document structure. Defaults to False."
        ),
        resolve_content: Optional[bool] = Field(
            default=None,
            description="Include complete structured content of matching elements. Useful when document structure (like XML or JSON) contains important context beyond plain text. Defaults to False."
        ),
        resolve_text: Optional[bool] = Field(
            default=None,
            description="Include complete text content of matching elements. Useful when previews are insufficient and full text context is needed. Defaults to False."
        ),
        limit: Optional[int] = Field(
            default=None,
            description="Maximum results to return. Higher limits find more matches but may include less relevant content. Consider balancing with min_score to maintain relevance quality. Defaults to 10."
        ),
        min_score: Optional[float] = Field(
            default=None,
            description="Semantic similarity threshold (-1 to 1). Higher values ensure closer conceptual matches: 0.7+ for exact concepts, 0.5+ for closely related, 0.3+ for broadly related, 0.1+ for exploratory searches. Defaults to 0."
        )
) -> List[ElementFlat]:

    """
    This performs semantic similarity search to find relevant content across documents, returning both the matching elements
    and their related context. The search identifies conceptually similar content even when exact words don't match, making
    it ideal for finding relevant documentation based on natural language descriptions or questions.

    The function breaks documents into searchable elements (paragraphs, lists, tables, etc.) and understands their relationships:
    - Structural relationships: parent/child elements (e.g., a section containing paragraphs), siblings (adjacent elements)
    - Explicit relationships: links or references between elements (if the document format supports it)
    - Semantic relationships: elements with similar meaning or topic, even if using different words

    Common uses:
    - Finding relevant documentation for customer inquiries
    - Locating policy information based on topic descriptions
    - Discovering related content across document sections
    - Matching technical documentation to user questions

    Parameters:
    :param resolve_text: This will provide the complete textual version of the matching element.
    :param resolve_content: This will provide the complete content (meaning any structural decorators or tags like formatting) of the matching element.
    :param search_for: Natural language text to search with. Can be a question ("How do I dispute a charge?"),
        description ("Customer asking about wire transfers"), or topic ("account security policies").
        The search uses semantic similarity, so exact word matches aren't needed.
    :param min_score: Similarity threshold (-1 to 1). Higher values mean closer conceptual matches:
        - 0.7+: Nearly exact concept matches
        - 0.5+: Closely related content
        - 0.3+: Broadly related content
        - 0.1+: Exploratory searches
        Defaults to 0.
    :param include_parents: Whether to include containing elements (e.g., the section containing a matching paragraph)
        to provide fuller context. Useful when matching content is part of a larger relevant section. Defaults to False.
    :param limit: Maximum number of results to return. Higher limits find more matches but may include less relevant content.
        Defaults to 10.

    Returns:
    A SearchResults object containing matching elements with:
    - doc_id: Identifier of the containing document
    - element_type: Type of matching element (section, paragraph, list, etc.)
    - text: Full text of the matching element when available
    - content_preview: Preview of the matching content (may be truncated)
    - score: Semantic similarity score (-1 to 1) indicating conceptual relevance
    - path: Full element path showing location in document hierarchy

    Example matches for "customer asking about wire transfer fees":
    - Exact match: Section about wire transfer fee schedule (score: 0.85)
    - Related: Paragraph about international transfer costs (score: 0.65)
    - Broader: Table of all service fees (score: 0.45)
    """
    async def work(_search_for, _limit, _min_score, _include_parents, _resolve_text, _resolve_content) -> List[ElementFlat]:
        _span = get_current_span()

        # Set defaults
        if not isinstance(_limit, int):
            _limit = 10

        if not isinstance(_min_score, float):
            _min_score = 0.0

        if not isinstance(_include_parents, bool):
            _include_parents = False

        if not isinstance(_resolve_text, bool):
            _resolve_text = False

        if not isinstance(_resolve_content, bool):
            _resolve_content = False

        # Prepare request headers
        headers = {
            'Content-Type': 'application/json'
        }
        if SEARCH_API_KEY:
            headers['X-API-Key'] = SEARCH_API_KEY

        # Prepare request payload
        payload = {
            'query': _search_for,
            'limit': _limit,
            'include_parents': _include_parents,
            'min_score': _min_score,
            'text': _resolve_text,
            'content': _resolve_content,
            'flat': True
        }

        try:
            # Make HTTP request to the search server
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{SEARCH_SERVER_URL}/api/search",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Search server error {response.status}: {error_text}")

                    response_data = await response.json()

        except Exception as e:
            _span.set_attribute("error", str(e))
            _span.set_attribute("search_error", "HTTP request failed")
            raise Exception(f"Failed to search documents: {str(e)}")

        try:
            search_tree = response_data.get('search_tree', [])
            search_tree = [ElementFlat(**item) for item in search_tree]
            _span.set_attribute("result_count", len(search_tree))
            return search_tree

        except Exception as e:
            _span.set_attribute("error", str(e))
            _span.set_attribute("processing_error", "Failed to process search results")
            raise Exception(f"Failed to serialize search documents: {str(e)}")

    return await with_active_span(
        tracer,
        "Search Documents",
        lambda span: work(search_for, limit, min_score, include_parents, resolve_content, resolve_text),
        {
            "search_for": search_for,
            "limit": str(limit),
            "min_score": str(min_score),
            "include_parents": str(include_parents),
        })

if __name__ == "__main__":
    start(connector)
