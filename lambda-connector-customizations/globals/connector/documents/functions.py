"""
functions.py

This is an example of how you can use the Python SDK's built-in Function connector to easily write Python code.
When you add a Python Lambda connector to your Hasura project, this file is generated for you!

In this file you'll find code examples that will help you get up to speed with the usage of the Hasura lambda connector.
If you are an old pro and already know what is going on you can get rid of these example functions and start writing your own code.
"""
from doculyzer import search_with_content
from hasura_ndc import start
from hasura_ndc.instrumentation import with_active_span # If you aren't planning on adding additional tracing spans, you don't need this!
from opentelemetry.trace import get_tracer # If you aren't planning on adding additional tracing spans, you don't need this either!
from hasura_ndc.function_connector import FunctionConnector
from pydantic import BaseModel, Field # You only need this import if you plan to have complex inputs/outputs, which function similar to how frameworks like FastAPI do
from typing import Annotated, List, Optional
from doculyzer.search import search_with_content, SearchResult
from doculyzer import ingest_documents

connector = FunctionConnector()

# This last section shows you how to add OTEL tracing to any of your functions!
tracer = get_tracer("ndc-sdk-python.server") # You only need a tracer if you plan to add additional Otel spans

@connector.register_query
async def search_documents(
        search_for: str,
        limit: Optional[int] = Field(description="An integer specifying the maximum number of search results to return. Defaults to 10.", default=None),
        min_score: Optional[float] = Field(default=None, description="Min similarity score to consider a match. 0 is neutral. 1 is perfect match. -1 is no match. Defaults to 0.")) -> List[SearchResult]:
    """
    This performs a similarity search to identify individual elements (like paragraphs, list items, or tables) in a document
    and returns the type of elements, the content of those elements and a preview of its related items.
    Items may be related structurally, like parent, child, sibling, explicitly like a link (if the document type
    supports that), and semantically, like a similar word or phrase.

    :param min_score: Min similarity score to consider a match. 0 is neutral. 1 is perfect match. -1 is no match.
    :param search_for: A string representing the query text to search for in the documents.
    :param limit: An integer specifying the maximum number of search results to return. Defaults to 10.
    :return: A SearchResults object containing the search results matching the query.
    """
    limit = limit or 10
    min_score = min_score or 0

    def work(_span, work_response):
        return search_with_content(search_for, limit, min_score = min_score)

    return await with_active_span(
        tracer,
        "Search Documents",
        lambda span: work(span, search_for),
        {"search_for": search_for})


import threading
import time


@connector.register_mutation
async def update_documents() -> bool:
    """
    Starts a document ingestion process on another thread.
    Returns immediately with success if the ingestion process starts successfully.

    :return: A boolean indicating whether the ingestion process was started successfully.
    """

    def work(_span, _):
        # Start the ingestion task in a new thread
        thread = threading.Thread(target=ingest_documents, daemon=True)
        thread.start()

        # Record that we've started the thread
        _span.set_attribute("thread.started", True)

        # Return true immediately to indicate successful start
        return True

    return await with_active_span(
        tracer,
        "Trigger Document Ingestion",
        lambda span: work(span, None),
        {})

if __name__ == "__main__":
    start(connector)
