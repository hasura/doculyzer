"""
functions.py

This is an example of how you can use the Python SDK's built-in Function connector to easily write Python code.
When you add a Python Lambda connector to your Hasura project, this file is generated for you!

In this file you'll find code examples that will help you get up to speed with the usage of the Hasura lambda connector.
If you are an old pro and already know what is going on you can get rid of these example functions and start writing your own code.
"""
from typing import List, Optional
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
    Field  # You only need this import if you plan to have complex inputs/outputs, which function similar to how frameworks like FastAPI do
import time

from doculyzer import ingest_documents
from doculyzer.storage import flatten_hierarchy, ElementFlat

connector = FunctionConnector()

# This last section shows you how to add OTEL tracing to any of your functions!
tracer = get_tracer("document_search.server") # You only need a tracer if you plan to add additional Otel spans

# Configuration for the web server
SEARCH_SERVER_URL = os.environ.get('DOCUMENTS_URI', 'http://localhost:5000')
SEARCH_API_KEY = os.environ.get('SEARCH_API_KEY')  # Optional API key

@connector.register_query
async def search_documents(
        search_for: str,
        include_parents: Optional[bool] = Field(default=None, description="Whether to include parent elements in the search results. Defaults to False."),
        resolve_content: Optional[bool] = Field(default=None, description="Whether to include the fully resolved element content. Defaults to False."),
        resolve_text: Optional[bool] = Field(default=None, description="Whether to include the fully resolved text content. Defaults to False."),
        limit: Optional[int] = Field(description="An integer specifying the maximum number of search results to return. Defaults to 10.", default=None),
        min_score: Optional[float] = Field(default=None, description="Min similarity score to consider a match. 0 is neutral. 1 is perfect match. -1 is no match. Defaults to 0.")) -> List[ElementFlat]:
    """
    This performs a similarity search to identify individual elements (like paragraphs, list items, or tables) in a document
    and returns the type of elements, the content of those elements and a preview of its related items.
    Items may be related structurally, like parent, child, sibling, explicitly like a link (if the document type
    supports that), and semantically, like a similar word or phrase.

    :param include_parents: Whether to include parent elements in the search results. Defaults to False.
    :param min_score: Min similarity score to consider a match. 0 is neutral. 1 is perfect match. -1 is no match.
    :param search_for: A string representing the query text to search for in the documents.
    :param limit: An integer specifying the maximum number of search results to return. Defaults to 10.
    :return: A SearchResults object containing the search results matching the query.
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

                    _span.set_attribute("result_count", len(response_data.get('search_tree')))

        except Exception as e:
            _span.set_attribute("error", str(e))
            _span.set_attribute("search_error", "HTTP request failed")
            raise Exception(f"Failed to search documents: {str(e)}")

        return response_data.get('search_tree')

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
