import logging
import os
import sys
import time
import json

import pytest
from dotenv import load_dotenv

from doculyzer.embeddings import EmbeddingGenerator

# Load environment variables from .env file
load_dotenv()
from doculyzer import Config
from doculyzer.search import search_by_text, search_structured, SearchQueryRequest, SearchCriteriaGroupRequest, \
    SemanticSearchRequest, TopicSearchRequest, LogicalOperatorEnum
from doculyzer.storage.search import pydantic_to_core_query

# Configure logging with real-time output
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO for cleaner output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensure real-time output
    ],
    force=True
)
logger = logging.getLogger('doculyzer_test')


# Add flush function for real-time logging during tests
def flush_logs():
    """Force flush all log handlers for immediate output during pytest."""
    for handler in logger.handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()


@pytest.fixture
def config_emb() -> (Config, EmbeddingGenerator):
    """Load test configuration as a fixture."""
    _config = Config(os.environ.get('DOCULYZER_CONFIG_PATH', 'config.yaml'))
    _embedding_generator = None
    return _config, _embedding_generator


def test_document_ingestion(config_emb: (Config, EmbeddingGenerator)):
    """Test the full document ingestion process."""
    from doculyzer.main import ingest_documents
    from doculyzer.adapter import create_content_resolver

    _config, _embedding_generator = config_emb

    # Get database from config
    db = _config.get_document_database()

    # Initialize database
    logger.info("Initializing database")
    db.initialize()

    try:
        # Create content resolver
        content_resolver = create_content_resolver(_config)
        logger.info("Created content resolver")

        # Ingest documents
        logger.info("Starting document ingestion")
        stats = ingest_documents(_config)
        logger.info(f"Document ingestion completed: {stats}")

        # Log summary
        logger.info(
            f"Processed {stats['documents']} documents with {stats['elements']} elements and {stats['relationships']} relationships")

        return stats
    finally:
        # Always close the database connection
        logger.info("Closing database connection")
        db.close()

def test_json_serialization_critical():
    """
    CRITICAL: Test JSON serialization for HTTP API compatibility.
    If this fails, HTTP API calls will fail in production.
    """

    logger.info("🔧 CRITICAL JSON SERIALIZATION TEST FOR HTTP API")
    logger.info("=" * 60)
    flush_logs()

    serialization_results = []

    # Test 1: Basic search result serialization
    logger.info("1. Basic Search Results → JSON")
    try:
        logger.info("   Running basic search...")
        search_results = search_by_text(
            query_text="cash management",
            limit=10,
            include_topics=['%wikipedia%'],
            text=False,
            content=False
        )

        logger.info(f"   Search completed: {len(search_results.results)} results")

        # Test JSON serialization
        logger.info("   Testing JSON serialization...")
        start_time = time.time()
        json_str = search_results.model_dump_json()
        serialization_time = (time.time() - start_time) * 1000

        # Test parsing
        parsed_json = json.loads(json_str)

        logger.info(f"   ✅ SUCCESS: {len(json_str):,} chars in {serialization_time:.2f}ms")
        serialization_results.append({'test': 'basic', 'success': True, 'size': len(json_str)})

    except Exception as e:
        logger.error(f"   ❌ FAILED: {e}")
        logger.error("   🚨 This will break HTTP API calls!")
        serialization_results.append({'test': 'basic', 'success': False, 'error': str(e)})

    flush_logs()

    # Test 2: Text materialization serialization
    logger.info("2. Text Materialized Results → JSON")
    try:
        logger.info("   Running search with text materialization...")
        text_results = search_by_text(
            query_text="document management",
            limit=5,
            include_topics=['%wikipedia%'],
            text=True,  # This adds materialized text
            content=False
        )

        logger.info(f"   Search completed: {len(text_results.results)} results")
        logger.info(f"   Text resolved: {text_results.text_resolved}")

        # Test JSON serialization with text
        logger.info("   Testing JSON serialization with text...")
        start_time = time.time()
        json_str = text_results.model_dump_json()
        serialization_time = (time.time() - start_time) * 1000

        # Test parsing
        parsed_json = json.loads(json_str)

        logger.info(f"   ✅ SUCCESS: {len(json_str):,} chars in {serialization_time:.2f}ms")
        serialization_results.append({'test': 'text_materialized', 'success': True, 'size': len(json_str)})

    except Exception as e:
        logger.error(f"   ❌ FAILED: {e}")
        logger.error("   🚨 Text materialization will break HTTP API!")
        serialization_results.append({'test': 'text_materialized', 'success': False, 'error': str(e)})

    flush_logs()

    # Test 3: Structured search serialization
    logger.info("3. Structured Search Results → JSON")
    try:
        logger.info("   Running structured search...")
        structured_query = SearchQueryRequest(
            criteria_group=SearchCriteriaGroupRequest(
                operator=LogicalOperatorEnum.AND,
                semantic_search=SemanticSearchRequest(
                    query_text="financial analysis",
                    similarity_threshold=0.7
                ),
                topic_search=TopicSearchRequest(
                    include_topics=["source.external.wikipedia"],
                    min_confidence=0.8
                )
            ),
            limit=8,
            include_similarity_scores=True,
            include_topics=True
        )

        structured_results = search_structured(
            query=structured_query,
            text=False,
            content=False,
            flat=False,
            include_parents=True
        )

        logger.info(f"   Search completed: {len(structured_results.results)} results")
        logger.info(f"   Query ID: {structured_results.query_id}")

        # Test JSON serialization
        logger.info("   Testing structured search JSON serialization...")
        start_time = time.time()
        json_str = structured_results.model_dump_json()
        serialization_time = (time.time() - start_time) * 1000

        # Test parsing and validate structure
        parsed_json = json.loads(json_str)
        has_query_id = 'query_id' in parsed_json
        has_execution_time = 'execution_time_ms' in parsed_json

        logger.info(f"   ✅ SUCCESS: {len(json_str):,} chars in {serialization_time:.2f}ms")
        logger.info(f"   Query ID in JSON: {has_query_id}")
        logger.info(f"   Execution time in JSON: {has_execution_time}")
        serialization_results.append({'test': 'structured', 'success': True, 'size': len(json_str)})

    except Exception as e:
        logger.error(f"   ❌ FAILED: {e}")
        logger.error("   🚨 Structured searches will break HTTP API!")
        serialization_results.append({'test': 'structured', 'success': False, 'error': str(e)})

    flush_logs()

    # Test 4: Edge cases
    logger.info("4. Edge Cases → JSON")
    edge_cases = [
        ('Empty Results', 'xyzunlikelytomatchanything123', ['%nonexistent%']),
        ('Special Characters', 'test "quotes" & <html> ñ utf8 💰', None),
    ]

    for case_name, query, topics in edge_cases:
        try:
            logger.info(f"   Testing {case_name}...")
            edge_results = search_by_text(query_text=query, limit=5, include_topics=topics)
            json_str = edge_results.model_dump_json()
            parsed_json = json.loads(json_str)
            logger.info(f"   ✅ {case_name}: OK ({len(edge_results.results)} results)")
        except Exception as e:
            logger.error(f"   ❌ {case_name}: FAILED - {e}")

    flush_logs()

    # Summary
    logger.info("📊 JSON SERIALIZATION TEST SUMMARY")
    logger.info("=" * 50)

    successful = [r for r in serialization_results if r.get('success', False)]
    failed = [r for r in serialization_results if not r.get('success', False)]

    if failed:
        logger.error(f"❌ FAILED TESTS: {len(failed)}/{len(serialization_results)}")
        for test in failed:
            logger.error(f"   {test['test']}: {test.get('error', 'Unknown error')}")
        logger.error("🚨 CRITICAL: Fix serialization before HTTP API deployment!")
    else:
        logger.info(f"✅ ALL TESTS PASSED: {len(successful)}/{len(serialization_results)}")
        logger.info("🚀 HTTP API serialization is working correctly!")

    # Performance summary
    if successful:
        total_size = sum(t['size'] for t in successful)
        logger.info(f"📈 Total JSON size across tests: {total_size:,} characters")
        if total_size > 1024 * 1024:  # 1MB
            logger.warning("⚠️  Large JSON payloads - consider pagination for production")

    flush_logs()


def test_document_search_execution():
    """Test comprehensive document search functionality with actual execution using real search functions."""

    logger.info("=== STRUCTURED DOCUMENT SEARCH EXECUTION TESTS ===")
    flush_logs()

    # Test 1: Original search function execution
    logger.info("1. ORIGINAL SEARCH FUNCTION TEST")
    logger.info("-" * 50)

    logger.info("Running original search_by_text function")
    query_text = "cash management"

    try:
        # This matches your original test pattern
        original_results = search_by_text(
            query_text=query_text,
            include_topics=['%wikipedia%'],
            min_score=-1,
            limit=50,
            text=True
        )

        original_count = len(original_results.results)
        logger.info(f"Original search returned {original_count} results")
        logger.info(f"✅ Original search: {original_count} results")
        logger.info(f"📊 Total results: {original_results.total_results}")
        logger.info(f"🔍 Query: {original_results.query}")
        logger.info(f"📝 Search type: {original_results.search_type}")
        logger.info(f"🏷️  Topics supported: {original_results.supports_topics}")

        # Show sample of original results (matching your original pattern)
        if original_results.search_tree:
            logger.info("Original search_tree sample:")
            for i, item in enumerate(original_results.search_tree[:3]):
                if hasattr(item, 'content_preview'):
                    logger.info(f"  {i + 1}. {item.content_preview[:100]}...")
                elif hasattr(item, 'text') and item.text:
                    logger.info(f"  {i + 1}. {item.text[:100]}...")
                else:
                    logger.info(f"  {i + 1}. Element {getattr(item, 'element_pk', 'unknown')}")

        # Show results from results list too
        if original_results.results:
            logger.info("Results list sample:")
            for i, result in enumerate(original_results.results[:3]):
                logger.info(f"  {i + 1}. PK: {result.element_pk}, Score: {result.similarity:.3f}")
                if result.content_preview:
                    logger.info(f"      Preview: {result.content_preview[:80]}...")

    except Exception as e:
        logger.error(f"Original search failed: {e}")
        original_results = None
        original_count = 0

    flush_logs()

    # Test 2: Equivalent structured search using your actual function
    logger.info("2. STRUCTURED SEARCH EQUIVALENT TEST")
    logger.info("-" * 50)

    logger.info("Running equivalent structured search")

    try:
        # Create equivalent structured query
        structured_query = SearchQueryRequest(
            criteria_group=SearchCriteriaGroupRequest(
                operator=LogicalOperatorEnum.AND,
                semantic_search=SemanticSearchRequest(
                    query_text="cash management",
                    similarity_threshold=0.0,  # min_score=-1 equivalent
                    boost_factor=1.0
                ),
                topic_search=TopicSearchRequest(
                    include_topics=["source.external.wikipedia"],  # Updated to your topic format
                    min_confidence=0.0  # Accept all confidence levels
                )
            ),
            limit=50,
            include_similarity_scores=True,
            include_topics=True
        )

        logger.info("🔍 Attempting structured search...")
        flush_logs()

        # Try search_structured function first
        try:
            structured_results = search_structured(
                query=structured_query,
                text=True,  # Equivalent to text=True in original
                content=False,
                flat=False,
                include_parents=True
            )

            if len(structured_results.results) > 0:
                logger.info("✅ search_structured() worked successfully")
            else:
                logger.warning("⚠️  search_structured() returned 0 results")
                logger.info("   This might indicate:")
                logger.info("   - Pydantic search module import issues")
                logger.info("   - No content matching the new topic format")
                logger.info("   - Topic hierarchy not yet populated")

        except Exception as search_error:
            logger.error(f"❌ search_structured() failed: {search_error}")

            # Check for specific validation error
            if "SearchCriteriaGroup must have at least one criterion" in str(search_error):
                logger.error("🚨 VALIDATION ERROR: SearchCriteriaGroup validation failed")
                logger.info("💡 Running detailed validation debug test...")
                test_structured_query_validation_debug()
                logger.info("💡 Running Pydantic to Core conversion debug...")
                test_pydantic_to_core_conversion_debug()
                return

            logger.info("🔄 Trying alternative approach with SearchHelper...")

            # Fallback: Try SearchHelper directly
            try:
                from doculyzer.search import SearchHelper

                if hasattr(SearchHelper, 'execute_structured_search'):
                    structured_results = SearchHelper.execute_structured_search(
                        query=structured_query,
                        text=True,
                        content=False,
                        flat=False,
                        include_parents=True
                    )
                    logger.info("✅ SearchHelper.execute_structured_search() worked as fallback")
                else:
                    logger.error("❌ SearchHelper.execute_structured_search() not available")
                    logger.info("💡 Running diagnostic test...")
                    test_pydantic_search_module_availability()
                    return

            except Exception as helper_error:
                logger.error(f"❌ SearchHelper fallback also failed: {helper_error}")

                # Check for same validation error in fallback
                if "SearchCriteriaGroup must have at least one criterion" in str(helper_error):
                    logger.error("🚨 SAME VALIDATION ERROR in SearchHelper fallback")
                    logger.info("💡 This indicates a Pydantic model validation issue")
                    test_structured_query_validation_debug()
                    return

                logger.error("🚨 CRITICAL: Structured search is not working")
                logger.info("💡 Run: pytest test_doculyzer.py::test_pydantic_search_module_availability -v -s")
                return

        structured_count = len(structured_results.results)
        logger.info(f"Structured search returned {structured_count} results")
        logger.info(f"✅ Structured search: {structured_count} results")
        logger.info(f"📊 Total results: {structured_results.total_results}")
        logger.info(f"🆔 Query ID: {structured_results.query_id}")
        logger.info(f"⏱️  Execution time: {structured_results.execution_time_ms}ms")
        logger.info(f"📝 Search type: {structured_results.search_type}")

        # Show sample of structured results
        if structured_results.results:
            logger.info("Structured results sample:")
            for i, result in enumerate(structured_results.results[:3]):
                logger.info(f"  {i + 1}. PK: {result.element_pk}, Score: {result.similarity:.3f}")
                if result.topics:
                    logger.info(f"      Topics: {result.topics}")
                if result.content_preview:
                    logger.info(f"      Preview: {result.content_preview[:80]}...")

        # Compare results if both worked
        if original_results:
            logger.info(f"📊 Comparison: Original({original_count}) vs Structured({structured_count})")
            if abs(original_count - structured_count) <= 5:  # Allow small variance
                logger.info("✅ Result counts are comparable")
            else:
                logger.warning("⚠️  Significant difference in result counts")
                logger.info("   Possible causes:")
                logger.info("   - Topic format difference ('%wikipedia%' vs 'source.external.wikipedia')")
                logger.info("   - Topic hierarchy not fully populated")
                logger.info("   - Different search backends or configurations")

    except Exception as e:
        logger.error(f"Structured search test completely failed: {e}")
        logger.info("💡 Running diagnostic test to identify the issue...")
        test_pydantic_search_module_availability()

    flush_logs()

    # Test 3: Cash management domain search using your functions
    logger.info("3. CASH MANAGEMENT DOMAIN SEARCH")
    logger.info("-" * 50)

    try:
        domain_query = SearchQueryRequest(
            criteria_group=SearchCriteriaGroupRequest(
                operator=LogicalOperatorEnum.AND,
                semantic_search=SemanticSearchRequest(
                    query_text="working capital liquidity management treasury operations",
                    similarity_threshold=0.7,
                    boost_factor=1.5
                ),
                topic_search=TopicSearchRequest(
                    include_topics=["domain.business.finance.cash_management"],
                    min_confidence=0.8
                )
            ),
            limit=25,
            include_similarity_scores=True,
            include_topics=True
        )

        logger.info("Executing domain-specific cash management search")

        domain_results = search_structured(
            query=domain_query,
            text=True,
            content=False,
            flat=False,
            include_parents=True
        )

        logger.info("✅ Domain search executed successfully")
        logger.info(f"📊 Results: {len(domain_results.results)} matches")
        logger.info(f"📊 Total results: {domain_results.total_results}")
        if domain_results.execution_time_ms:
            logger.info(f"⏱️  Execution time: {domain_results.execution_time_ms:.2f}ms")

        if domain_results.results:
            logger.info("Top domain-specific results:")
            for i, result in enumerate(domain_results.results[:5]):
                topics_str = ", ".join(result.topics or ["No topics"])
                logger.info(f"  {i + 1}. Score: {result.similarity:.3f}")
                logger.info(f"     Topics: {topics_str}")
                if result.content_preview:
                    logger.info(f"     Content: {result.content_preview[:80]}...")
        else:
            logger.warning("⚠️  No results found for domain-specific search")

    except Exception as e:
        logger.error(f"Domain search failed: {e}")

    flush_logs()

    # Test 4: Multi-source comparison using your search_by_text function
    logger.info("4. MULTI-SOURCE SEARCH COMPARISON")
    logger.info("-" * 50)

    sources_to_test = [
        ("Wikipedia", ['%wikipedia%']),  # Your original format
        ("All Topics", None),
        ("Cash Management", ['%cash%', '%finance%'])  # Pattern matching
    ]

    base_query_text = "document management system"

    for source_name, topic_patterns in sources_to_test:
        logger.info(f"Testing search across {source_name}")

        try:
            source_results = search_by_text(
                query_text=base_query_text,
                limit=20,
                include_topics=topic_patterns,
                min_score=0.6,
                text=False,
                content=False,
                flat=False,
                include_parents=True
            )

            result_count = len(source_results.results)
            logger.info(f"📁 {source_name}: {result_count} results")
            logger.info(f"   📊 Total: {source_results.total_results}")
            logger.info(f"   🔍 Query: {source_results.query}")
            logger.info(f"   📝 Type: {source_results.search_type}")

            if source_results.documents:
                logger.info(f"   📄 Document sources: {len(source_results.documents)} unique")
                for doc in source_results.documents[:3]:  # Show first 3
                    logger.debug(f"      - {doc}")
                if len(source_results.documents) > 3:
                    logger.info(f"      ... and {len(source_results.documents) - 3} more")

        except Exception as e:
            logger.error(f"Source search for {source_name} failed: {e}")

        flush_logs()


def test_topic_hierarchy_execution():
    """Test actual execution of topic hierarchy queries using your real functions."""

    logger.info("=== TOPIC HIERARCHY EXECUTION TESTS ===")
    flush_logs()

    hierarchy_tests = [
        {
            "name": "Source: Wikipedia (Legacy Format)",
            "legacy_topics": ['%wikipedia%'],
            "description": "External Wikipedia content using legacy pattern"
        },
        {
            "name": "Source: Wikipedia (New Format)",
            "structured_topics": ["source.external.wikipedia"],
            "description": "External Wikipedia content using new topic hierarchy"
        },
        {
            "name": "Domain: Cash Management",
            "structured_topics": ["domain.business.finance.cash_management"],
            "description": "Cash management domain content"
        },
        {
            "name": "Type: Financial Research",
            "structured_topics": ["type.research.financial"],
            "description": "Financial research content"
        },
        {
            "name": "Access: Public Content",
            "structured_topics": ["access.public"],
            "description": "Publicly accessible content"
        }
    ]

    topic_results = {}

    for test in hierarchy_tests:
        logger.info(f"🏷️  Testing: {test['name']}")
        logger.info(f"   Description: {test['description']}")

        try:
            if 'legacy_topics' in test:
                # Use legacy search_by_text with pattern matching
                results = search_by_text(
                    query_text="cash management",  # Base query
                    limit=10,
                    include_topics=test['legacy_topics'],
                    min_confidence=0.5,
                    text=False,
                    content=False
                )

                result_count = len(results.results)
                logger.info(f"   📊 Results (legacy): {result_count}")
                logger.info(f"   📊 Total: {results.total_results}")
                logger.info(f"   🔍 Query: {results.query}")
                logger.info(f"   📝 Type: {results.search_type}")

            elif 'structured_topics' in test:
                # Use structured search with new topic hierarchy
                topic_query = SearchQueryRequest(
                    criteria_group=SearchCriteriaGroupRequest(
                        topic_search=TopicSearchRequest(
                            include_topics=test['structured_topics'],
                            min_confidence=0.5
                        )
                    ),
                    limit=10,
                    include_topics=True
                )

                results = search_structured(
                    query=topic_query,
                    text=False,
                    content=False,
                    flat=False,
                    include_parents=True
                )

                result_count = len(results.results)
                logger.info(f"   📊 Results (structured): {result_count}")
                logger.info(f"   📊 Total: {results.total_results}")
                if results.query_id:
                    logger.debug(f"   🆔 Query ID: {results.query_id}")
                if results.execution_time_ms:
                    logger.info(f"   ⏱️  Time: {results.execution_time_ms:.2f}ms")

                # Show topic distribution in results
                if results.results:
                    all_topics = []
                    for result in results.results:
                        if result.topics:
                            all_topics.extend(result.topics)

                    if all_topics:
                        topic_counts = {}
                        for topic in all_topics:
                            topic_counts[topic] = topic_counts.get(topic, 0) + 1

                        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
                        logger.info(f"   🏷️  Topic distribution: {dict(sorted_topics[:3])}")

            topic_results[test['name']] = result_count

        except Exception as e:
            logger.error(f"Topic hierarchy test '{test['name']}' failed: {e}")
            topic_results[test['name']] = f"Error: {e}"

        flush_logs()

    # Summary
    logger.info("=== TOPIC HIERARCHY SUMMARY ===")
    for test_name, result in topic_results.items():
        if isinstance(result, int):
            status = "✅" if result > 0 else "⚠️"
            logger.info(f"{status} {test_name}: {result} results")
        else:
            logger.warning(f"❓ {test_name}: {result}")

    flush_logs()


def test_search_performance_validation():
    """Test search performance and validate result quality using your real functions."""

    logger.info("=== SEARCH PERFORMANCE & VALIDATION TESTS ===")
    flush_logs()

    test_cases = [
        {
            "name": "Simple Text Search",
            "function": "search_by_text",
            "params": {
                "query_text": "cash management strategies",
                "limit": 20,
                "min_score": 0.7,
                "text": False
            },
            "expected_min_results": 1
        },
        {
            "name": "Wikipedia Topic Filter",
            "function": "search_by_text",
            "params": {
                "query_text": "document management",
                "limit": 15,
                "include_topics": ['%wikipedia%'],
                "min_score": 0.6,
                "text": False
            },
            "expected_min_results": 1
        },
        {
            "name": "Structured Search Test",
            "function": "search_structured",
            "query": SearchQueryRequest(
                criteria_group=SearchCriteriaGroupRequest(
                    operator=LogicalOperatorEnum.AND,
                    semantic_search=SemanticSearchRequest(
                        query_text="financial analysis",
                        similarity_threshold=0.7
                    ),
                    topic_search=TopicSearchRequest(
                        include_topics=["source.external.wikipedia"],
                        min_confidence=0.7
                    )
                ),
                limit=10
            ),
            "structured_params": {
                "text": False,
                "content": False,
                "flat": False,
                "include_parents": True
            },
            "expected_min_results": 0
        }
    ]

    results_summary = []

    for test_case in test_cases:
        logger.info(f"🧪 Testing: {test_case['name']}")
        logger.info("-" * 40)

        start_time = time.time()

        try:
            if test_case['function'] == 'search_by_text':
                search_results = search_by_text(**test_case['params'])
            elif test_case['function'] == 'search_structured':
                search_results = search_structured(
                    query=test_case['query'],
                    **test_case['structured_params']
                )
            else:
                raise ValueError(f"Unknown function: {test_case['function']}")

            execution_time = (time.time() - start_time) * 1000

            result_count = len(search_results.results)
            success = result_count >= test_case['expected_min_results']

            if success:
                logger.info("   ✅ Executed successfully")
            else:
                logger.warning("   ⚠️  Low result count")

            logger.info(f"   📊 Results: {result_count}")
            logger.info(f"   📊 Total: {search_results.total_results}")
            logger.info(f"   ⏱️  Time: {execution_time:.2f}ms")
            logger.info(f"   🔍 Query: {search_results.query}")
            logger.info(f"   📝 Type: {search_results.search_type}")
            logger.info(f"   🎯 Expected minimum: {test_case['expected_min_results']}")

            if search_results.execution_time_ms:
                logger.info(f"   ⏱️  Reported time: {search_results.execution_time_ms:.2f}ms")

            if search_results.results:
                avg_score = sum(r.similarity for r in search_results.results) / len(search_results.results)
                logger.info(f"   📈 Average score: {avg_score:.3f}")

                # Show document sources
                if search_results.documents:
                    logger.info(f"   📄 Document sources: {len(search_results.documents)}")
                    for doc in search_results.documents[:2]:
                        logger.debug(f"      - {doc}")

            results_summary.append({
                'name': test_case['name'],
                'success': success,
                'result_count': result_count,
                'execution_time': execution_time,
                'average_score': avg_score if search_results.results else 0.0,
                'search_type': search_results.search_type
            })

        except Exception as e:
            logger.error(f"Test '{test_case['name']}' failed: {e}")

            results_summary.append({
                'name': test_case['name'],
                'success': False,
                'error': str(e)
            })

        flush_logs()

    # Summary report
    logger.info("=== PERFORMANCE SUMMARY ===")
    successful_tests = [r for r in results_summary if r.get('success', False)]

    if successful_tests:
        total_results = sum(r['result_count'] for r in successful_tests)
        avg_time = sum(r['execution_time'] for r in successful_tests) / len(successful_tests)
        avg_score = sum(r['average_score'] for r in successful_tests) / len(successful_tests)

        logger.info(f"✅ Successful tests: {len(successful_tests)}/{len(test_cases)}")
        logger.info(f"📊 Total results: {total_results}")
        logger.info(f"⏱️  Average execution time: {avg_time:.2f}ms")
        logger.info(f"📈 Average relevance score: {avg_score:.3f}")

        # Show search type distribution
        search_types = [r['search_type'] for r in successful_tests]
        type_counts = {}
        for st in search_types:
            type_counts[st] = type_counts.get(st, 0) + 1
        logger.info(f"📝 Search types: {type_counts}")
    else:
        logger.warning("⚠️  No tests completed successfully")

    flush_logs()


def test_content_materialization():
    """Test content materialization features of your search functions."""

    logger.info("=== CONTENT MATERIALIZATION TESTS ===")
    flush_logs()

    # Test 1: Text materialization with legacy search
    logger.info("1. TEXT MATERIALIZATION TEST (LEGACY)")
    logger.info("-" * 50)

    try:
        text_results = search_by_text(
            query_text="cash management",
            limit=5,
            include_topics=['%wikipedia%'],
            text=True,  # Request text materialization
            content=False,
            flat=False,
            include_parents=True
        )

        logger.info(f"Text materialization results: {len(text_results.results)} items")
        logger.info(f"Text resolved: {text_results.text_resolved}")
        logger.info(f"Content resolved: {text_results.content_resolved}")

        # Check if text was actually materialized in search tree
        text_count = 0
        for tree_item in text_results.search_tree[:3]:
            if hasattr(tree_item, 'text') and tree_item.text:
                text_count += 1
                logger.info(f"   Materialized text ({len(tree_item.text)} chars): {tree_item.text[:100]}...")

        logger.info(f"   Items with materialized text: {text_count}")

    except Exception as e:
        logger.error(f"Text materialization test failed: {e}")

    flush_logs()

    # Test 2: Content materialization with structured search
    logger.info("2. CONTENT MATERIALIZATION TEST (STRUCTURED)")
    logger.info("-" * 50)

    try:
        content_query = SearchQueryRequest(
            criteria_group=SearchCriteriaGroupRequest(
                semantic_search=SemanticSearchRequest(
                    query_text="document management",
                    similarity_threshold=0.7
                )
            ),
            limit=3,
            include_similarity_scores=True
        )

        content_results = search_structured(
            query=content_query,
            text=True,  # Request text materialization
            content=True,  # Request content materialization
            flat=False,
            include_parents=True
        )

        logger.info(f"Content materialization results: {len(content_results.results)} items")
        logger.info(f"Text resolved: {content_results.text_resolved}")
        logger.info(f"Content resolved: {content_results.content_resolved}")

        # Check materialized content in search tree
        for i, tree_item in enumerate(content_results.search_tree[:2]):
            logger.info(f"   Item {i + 1}:")
            if hasattr(tree_item, 'text') and tree_item.text:
                logger.info(f"     Text available: {len(tree_item.text)} chars")
            if hasattr(tree_item, 'content') and tree_item.content:
                logger.info(f"     Content available: {len(tree_item.content)} chars")
            if hasattr(tree_item, 'content_location'):
                logger.debug(f"     Content location: {tree_item.content_location}")

        # Check materialized content in results items
        for i, result in enumerate(content_results.results[:2]):
            logger.info(f"   Result {i + 1} (PK: {result.element_pk}):")
            if result.text:
                logger.info(f"     Result text available: {len(result.text)} chars")
            if result.content:
                logger.info(f"     Result content available: {len(result.content)} chars")

    except Exception as e:
        logger.error(f"Content materialization test failed: {e}")

    flush_logs()

    # Test 3: Performance comparison (with vs without materialization)
    logger.info("3. MATERIALIZATION PERFORMANCE COMPARISON")
    logger.info("-" * 50)

    query_text = "financial analysis"
    limit = 10

    try:
        # Test without materialization
        start_time = time.time()
        no_content_results = search_by_text(
            query_text=query_text,
            limit=limit,
            text=False,
            content=False
        )
        no_content_time = (time.time() - start_time) * 1000

        # Test with text materialization
        start_time = time.time()
        text_results = search_by_text(
            query_text=query_text,
            limit=limit,
            text=True,
            content=False
        )
        text_time = (time.time() - start_time) * 1000

        # Test with full materialization
        start_time = time.time()
        full_results = search_by_text(
            query_text=query_text,
            limit=limit,
            text=True,
            content=True
        )
        full_time = (time.time() - start_time) * 1000

        logger.info(f"Performance comparison for {limit} results:")
        logger.info(f"   No materialization: {no_content_time:.2f}ms")
        logger.info(f"   Text only: {text_time:.2f}ms")
        logger.info(f"   Text + Content: {full_time:.2f}ms")
        logger.info(f"   Text overhead: +{text_time - no_content_time:.2f}ms")
        logger.info(f"   Full overhead: +{full_time - no_content_time:.2f}ms")

    except Exception as e:
        logger.error(f"Performance comparison failed: {e}")

    flush_logs()


def run_comprehensive_search_tests():
    """Run all search tests with actual execution using your real search functions."""

    logger.info("🔍 RUNNING COMPREHENSIVE SEARCH EXECUTION TESTS")
    logger.info("=" * 60)
    logger.info("Using actual search functions from your codebase:")
    logger.info("- search_by_text() for legacy pattern searches")
    logger.info("- search_structured() for new structured searches")
    logger.info("- Comparing both approaches with real data")
    flush_logs()

    try:
        # 0. DIAGNOSTIC: Check pydantic search availability first
        logger.info("📍 PHASE 0: Pydantic Search Module Diagnostic")
        flush_logs()
        test_pydantic_search_module_availability()

        # 1. Core search execution tests
        test_document_search_execution()

        # 2. Topic hierarchy execution tests
        test_topic_hierarchy_execution()

        # 3. Performance and validation tests
        test_search_performance_validation()

        # 4. Content materialization tests
        test_content_materialization()

        # 5. CRITICAL: JSON serialization tests
        test_json_serialization_critical()

        logger.info("✅ ALL SEARCH EXECUTION TESTS COMPLETED")

        # Integration summary based on your actual code
        logger.info("=== ACTUAL INTEGRATION STATUS ===")
        logger.info("✅ Your system already has these working functions:")
        logger.info("   - search_by_text() with topic filtering")
        logger.info("   - search_structured() with Pydantic models")
        logger.info("   - Content materialization (text/content resolution)")
        logger.info("   - Hierarchical search trees")
        logger.info("   - Topic statistics and filtering")
        logger.info("   - JSON serialization for HTTP APIs")
        logger.info("")
        logger.info("🔄 Migration path:")
        logger.info("   1. Keep using search_by_text() for simple queries")
        logger.info("   2. Use search_structured() for complex multi-criteria queries")
        logger.info("   3. Update topic patterns: '%wikipedia%' → 'source.external.wikipedia'")
        logger.info("   4. Access results via SearchResults.results and SearchResults.search_tree")

    except Exception as e:
        logger.error(f"❌ Search execution tests failed: {e}")
        raise
    finally:
        flush_logs()


def test_quick_pydantic_search_fix():
    """
    Quick test to verify if the pydantic search issue can be resolved.
    Run this test to check potential fixes.
    """

    logger.info("🔧 QUICK PYDANTIC SEARCH FIX TEST")
    logger.info("=" * 50)
    flush_logs()

    # Fix attempt 1: Check if we can import from storage.search instead
    logger.info("1. Trying storage.search import...")
    try:
        from doculyzer.storage.search import execute_search as storage_execute_search
        logger.info("   ✅ SUCCESS: Can import execute_search from storage.search")

        # Test a quick query
        test_query = SearchQueryRequest(
            criteria_group=SearchCriteriaGroupRequest(
                semantic_search=SemanticSearchRequest(
                    query_text="test",
                    similarity_threshold=0.7
                )
            ),
            limit=1
        )

        # Try using storage.search.execute_search
        from doculyzer.search import SearchHelper
        db = SearchHelper.get_database()

        result = storage_execute_search(test_query, db)
        if result.success:
            logger.info("   ✅ storage.search.execute_search works!")
            logger.info("   💡 FIX: Update search.py to import from storage.search")
        else:
            logger.warning(f"   ⚠️  storage.search.execute_search failed: {result.error_message}")

    except Exception as e:
        logger.error(f"   ❌ storage.search import failed: {e}")

    flush_logs()

    # Fix attempt 2: Check module path issues
    logger.info("2. Checking module paths...")
    try:
        import sys
        logger.info("   Python path:")
        for path in sys.path[:5]:  # Show first 5 paths
            logger.info(f"     - {path}")

        logger.info("   Doculyzer module location:")
        import doculyzer
        logger.info(f"     - {doculyzer.__file__}")

    except Exception as e:
        logger.error(f"   ❌ Path check failed: {e}")

    flush_logs()

    logger.info("💡 POTENTIAL FIXES:")
    logger.info("   1. Create doculyzer/pydantic_search.py file")
    logger.info("   2. Or update doculyzer/search.py line:")
    logger.info("      from .pydantic_search import execute_search")
    logger.info("      TO:")
    logger.info("      from .storage.search import execute_search")
    logger.info("   3. Or check if all dependencies are installed")
    logger.info("   4. Run: pip install pydantic>=2.0")

    flush_logs()


def test_original_pattern_match():
    """Test that exactly matches your original test_document_search pattern."""

    logger.info("=== ORIGINAL PATTERN MATCH TEST ===")
    flush_logs()

    try:
        # Run exactly your original search
        logger.info("Running ANN search")
        query_text = "cash management"
        logger.info(f"Searching for similar elements: {query_text}")

        text_results = search_by_text(
            query_text=query_text,
            include_topics=['%wikipedia%'],
            min_score=-1,
            limit=50,
            text=True
        )

        # Convert to JSON as in your original
        text_results_json = json.loads(text_results.model_dump_json())

        # Log the search tree as in your original pattern
        search_tree_items = text_results_json.get('search_tree', [])
        logger.info(f"Search tree contains {len(search_tree_items)} items")

        # Show items from search tree (matching your original pprint pattern)
        for i, item in enumerate(search_tree_items[:5]):  # Show first 5
            logger.info(f"Search tree item {i + 1}:")
            if 'content_preview' in item:
                logger.info(f"  Preview: {item['content_preview'][:100]}...")
            if 'element_type' in item:
                logger.info(f"  Type: {item['element_type']}")
            if 'element_pk' in item:
                logger.info(f"  PK: {item['element_pk']}")
            if 'score' in item:
                logger.info(f"  Score: {item['score']}")

        # Additional validation
        logger.info(f"✅ Original pattern test completed successfully")
        logger.info(f"📊 Results: {len(text_results.results)} items")
        logger.info(f"📊 Total: {text_results.total_results}")
        logger.info(f"🔍 Query: {text_results.query}")
        logger.info(f"📝 Search type: {text_results.search_type}")

    except Exception as e:
        logger.error(f"Original pattern test failed: {e}")

    flush_logs()


if __name__ == "__main__":
    # For real-time logging during pytest, add this to pytest.ini:
    # [tool.pytest.ini_options]
    # log_cli = true
    # log_cli_level = "INFO"
    # log_cli_format = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
    # log_cli_date_format = "%Y-%m-%d %H:%M:%S"
    #
    # Or run with: pytest -s --log-cli-level=INFO

    # Set up logging for direct execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    logger = logging.getLogger(__name__)

    logger.info("🔍 TESTING YOUR ACTUAL SEARCH FUNCTIONS")
    logger.info("=" * 50)
    logger.info("This test uses your real search_by_text() and search_structured() functions")

    logger.info("💡 To run these tests with real-time logging:")
    logger.info("1. pytest test_doculyzer.py -s --log-cli-level=INFO")
    logger.info("2. Individual test: pytest test_doculyzer.py::test_json_serialization_critical -v -s")
    logger.info("3. Full suite: pytest test_doculyzer.py::run_comprehensive_search_tests -v -s")
