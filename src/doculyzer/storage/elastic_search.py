"""
Complete Elasticsearch Implementation with Structured Search Support

This module provides a comprehensive Elasticsearch implementation of the DocumentDatabase
with full structured search capabilities, matching the abstract base class requirements.
It leverages Elasticsearch's advanced features including full-text search, aggregations,
and vector search to provide comprehensive search functionality.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, TYPE_CHECKING

import time

# Import types for type checking only
if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    # Define type aliases for type checking
    VectorType = NDArray[np.float32]
else:
    # Runtime type aliases - use generic Python types
    VectorType = List[float]

from .element_relationship import ElementRelationship
from .base import DocumentDatabase
from .element_element import ElementBase, ElementType, ElementHierarchical

# Import structured search components
from .structured_search import (
    StructuredSearchQuery, SearchCriteriaGroup, BackendCapabilities, SearchCapability,
    UnsupportedSearchError, TextSearchCriteria, EmbeddingSearchCriteria, DateSearchCriteria,
    TopicSearchCriteria, MetadataSearchCriteria, ElementSearchCriteria,
    LogicalOperator, DateRangeOperator, SimilarityOperator
)

logger = logging.getLogger(__name__)

# Define global flags for availability - will be set at runtime
ELASTICSEARCH_AVAILABLE = False
NUMPY_AVAILABLE = False

# Try to import Elasticsearch library at runtime
try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import NotFoundError, RequestError
    from elasticsearch.helpers import bulk, scan

    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    logger.warning("elasticsearch not available. Install with 'pip install elasticsearch'.")

    # Create a placeholder for type checking
    class Elasticsearch:
        def __init__(self, *args, **kwargs):
            pass

# Try to import NumPy conditionally
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("NumPy not available. Will use slower pure Python vector operations.")

# Try to import the config
try:
    from ..config import Config

    config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
except Exception as e:
    logger.warning(f"Error configuring Elasticsearch provider: {str(e)}")
    config = None


class ElasticsearchDocumentDatabase(DocumentDatabase):
    """Elasticsearch implementation of document database with comprehensive structured search support."""

    def __init__(self, conn_params: Dict[str, Any]):
        """
        Initialize Elasticsearch document database.

        Args:
            conn_params: Connection parameters for Elasticsearch
                Core connection:
                - hosts: List of Elasticsearch hosts (default: ['localhost:9200'])
                - username: Optional username for authentication
                - password: Optional password for authentication
                - ca_certs: Optional path to CA certificates
                - verify_certs: Whether to verify SSL certificates (default: True)
                - index_prefix: Prefix for all indices (default: 'doculyzer')

                Vector search:
                - vector_dimension: Dimension of embedding vectors (default: 384)

                Text storage and indexing options:
                - store_full_text: Whether to store full text for retrieval (default: True)
                - index_full_text: Whether to index full text for search (default: True)
                - compress_full_text: Whether to enable compression for stored text (default: False)
                - full_text_max_length: Maximum length for full text, truncate if longer (default: None)

                Common configurations:
                - Search + Storage: store_full_text=True, index_full_text=True (default, best search quality)
                - Search only: store_full_text=False, index_full_text=True (saves storage space)
                - Storage only: store_full_text=True, index_full_text=False (for retrieval without search)
                - Neither: store_full_text=False, index_full_text=False (minimal storage, preview only)
        """
        self.conn_params = conn_params

        # Extract connection parameters
        hosts = conn_params.get('hosts', ['localhost:9200'])
        username = conn_params.get('username')
        password = conn_params.get('password')
        ca_certs = conn_params.get('ca_certs')
        verify_certs = conn_params.get('verify_certs', True)
        self.index_prefix = conn_params.get('index_prefix', 'doculyzer')

        # Build Elasticsearch client configuration
        es_config = {
            'hosts': hosts,
            'verify_certs': verify_certs,
            'request_timeout': 30,
            'retry_on_timeout': True,
            'max_retries': 3
        }

        if username and password:
            es_config['http_auth'] = (username, password)

        if ca_certs:
            es_config['ca_certs'] = ca_certs

        # Define index names
        self.documents_index = f"{self.index_prefix}_documents"
        self.elements_index = f"{self.index_prefix}_elements"
        self.relationships_index = f"{self.index_prefix}_relationships"
        self.history_index = f"{self.index_prefix}_history"
        self.embeddings_index = f"{self.index_prefix}_embeddings"
        self.dates_index = f"{self.index_prefix}_dates"

        # Initialize Elasticsearch client to None - will be created in initialize()
        self.es = None

        # Auto-increment counter
        self.element_pk_counter = 0

        # Configuration for vector search
        self.vector_dimension = conn_params.get('vector_dimension', 384)
        if config:
            self.vector_dimension = config.config.get('embedding', {}).get('dimensions', self.vector_dimension)

        # Text storage and indexing configuration
        self.store_full_text = conn_params.get('store_full_text', True)  # Store full text for retrieval
        self.index_full_text = conn_params.get('index_full_text', True)  # Index full text for search
        self.compress_full_text = conn_params.get('compress_full_text', False)  # Compress stored text
        self.full_text_max_length = conn_params.get('full_text_max_length', None)  # Truncate long text

        self.embedding_generator = None

    # ========================================
    # STRUCTURED SEARCH IMPLEMENTATION
    # ========================================

    def get_backend_capabilities(self) -> BackendCapabilities:
        """
        Elasticsearch supports comprehensive search capabilities with excellent performance.
        """
        supported = {
            # Core search types
            SearchCapability.TEXT_SIMILARITY,
            SearchCapability.EMBEDDING_SIMILARITY,
            SearchCapability.FULL_TEXT_SEARCH,
            SearchCapability.VECTOR_SEARCH,

            # Date capabilities
            SearchCapability.DATE_FILTERING,
            SearchCapability.DATE_RANGE_QUERIES,
            SearchCapability.FISCAL_YEAR_DATES,
            SearchCapability.RELATIVE_DATES,
            SearchCapability.DATE_AGGREGATIONS,

            # Topic capabilities
            SearchCapability.TOPIC_FILTERING,
            SearchCapability.TOPIC_LIKE_PATTERNS,
            SearchCapability.TOPIC_CONFIDENCE_FILTERING,

            # Metadata capabilities
            SearchCapability.METADATA_EXACT,
            SearchCapability.METADATA_LIKE,
            SearchCapability.METADATA_RANGE,
            SearchCapability.METADATA_EXISTS,
            SearchCapability.NESTED_METADATA,

            # Element capabilities
            SearchCapability.ELEMENT_TYPE_FILTERING,
            SearchCapability.ELEMENT_HIERARCHY,
            SearchCapability.ELEMENT_RELATIONSHIPS,

            # Logical operations
            SearchCapability.LOGICAL_AND,
            SearchCapability.LOGICAL_OR,
            SearchCapability.LOGICAL_NOT,
            SearchCapability.NESTED_QUERIES,

            # Scoring and ranking
            SearchCapability.CUSTOM_SCORING,
            SearchCapability.SIMILARITY_THRESHOLDS,
            SearchCapability.BOOST_FACTORS,
            SearchCapability.SCORE_COMBINATION,

            # Advanced features
            SearchCapability.RESULT_HIGHLIGHTING,
            SearchCapability.FACETED_SEARCH,
        }

        return BackendCapabilities(supported)

    def execute_structured_search(self, query: StructuredSearchQuery) -> List[Dict[str, Any]]:
        """
        Execute a structured search query using Elasticsearch's capabilities.
        """
        if not self.es:
            raise ValueError("Database not initialized")

        # Validate query support
        missing = self.validate_query_support(query)
        if missing:
            raise UnsupportedSearchError(missing)

        try:
            # Execute the root criteria group
            raw_results = self._execute_criteria_group(query.criteria_group)

            # Process and enrich results
            final_results = self._process_search_results(raw_results, query)

            # Apply pagination
            start_idx = query.offset or 0
            end_idx = start_idx + query.limit

            return final_results[start_idx:end_idx]

        except Exception as e:
            logger.error(f"Error executing structured search: {str(e)}")
            return []

    def _execute_criteria_group(self, group: SearchCriteriaGroup) -> List[Dict[str, Any]]:
        """Execute a single criteria group and return scored results."""

        # Collect results from all criteria in this group
        all_results = []

        # Execute individual criteria
        if group.text_criteria:
            text_results = self._execute_text_criteria(group.text_criteria)
            all_results.append(("text", text_results))

        if group.embedding_criteria:
            embedding_results = self._execute_embedding_criteria(group.embedding_criteria)
            all_results.append(("embedding", embedding_results))

        if group.date_criteria:
            date_results = self._execute_date_criteria(group.date_criteria)
            all_results.append(("date", date_results))

        if group.topic_criteria:
            topic_results = self._execute_topic_criteria(group.topic_criteria)
            all_results.append(("topic", topic_results))

        if group.metadata_criteria:
            metadata_results = self._execute_metadata_criteria(group.metadata_criteria)
            all_results.append(("metadata", metadata_results))

        if group.element_criteria:
            element_results = self._execute_element_criteria(group.element_criteria)
            all_results.append(("element", element_results))

        # Execute sub-groups recursively
        for sub_group in group.sub_groups:
            sub_results = self._execute_criteria_group(sub_group)
            all_results.append(("subgroup", sub_results))

        # Combine results based on the group's logical operator
        return self._combine_results(all_results, group.operator)

    def _execute_text_criteria(self, criteria: TextSearchCriteria) -> List[Dict[str, Any]]:
        """Execute text similarity search using Elasticsearch's capabilities."""
        try:
            # Use Elasticsearch's multi-match query for text search
            search_fields = ["content_preview^2"]  # Boost content_preview
            if self.index_full_text:
                search_fields.append("full_text")

            # Execute text search
            text_query = {
                "multi_match": {
                    "query": criteria.query_text,
                    "fields": search_fields,
                    "type": "best_fields"
                }
            }

            search_body = {
                "query": text_query,
                "size": 1000
            }

            result = self.es.search(index=self.elements_index, body=search_body)
            text_scores = {int(hit['_source']["element_pk"]): float(hit['_score'])
                           for hit in result['hits']['hits']}

            # Also perform vector search if available
            vector_scores = {}
            try:
                query_embedding = self._generate_embedding(criteria.query_text)
                vector_results = self.search_by_embedding(query_embedding, limit=1000)
                vector_scores = {pk: score for pk, score in vector_results}
            except Exception as e:
                logger.warning(f"Vector search failed in text criteria: {str(e)}")

            # Combine and filter results
            filtered_results = []
            all_element_pks = set(text_scores.keys()) | set(vector_scores.keys())

            for element_pk in all_element_pks:
                # Calculate hybrid score (text + vector)
                text_score = text_scores.get(element_pk, 0.0)
                vector_score = vector_scores.get(element_pk, 0.0)

                # Normalize text score (ES scores can vary widely)
                normalized_text_score = min(text_score / 10.0, 1.0)

                # Weighted combination
                hybrid_score = 0.4 * normalized_text_score + 0.6 * vector_score

                if self._compare_similarity(hybrid_score, criteria.similarity_threshold, criteria.similarity_operator):
                    filtered_results.append({
                        'element_pk': element_pk,
                        'scores': {
                            'text_similarity': hybrid_score * criteria.boost_factor
                        }
                    })

            return filtered_results

        except Exception as e:
            logger.error(f"Error executing text criteria: {str(e)}")
            return []

    def _execute_embedding_criteria(self, criteria: EmbeddingSearchCriteria) -> List[Dict[str, Any]]:
        """Execute direct embedding vector search."""
        try:
            similarity_results = self.search_by_embedding(
                criteria.embedding_vector,
                limit=1000,
                filter_criteria=None
            )

            filtered_results = []
            for element_pk, similarity in similarity_results:
                if self._compare_similarity(similarity, criteria.similarity_threshold, criteria.similarity_operator):
                    filtered_results.append({
                        'element_pk': element_pk,
                        'scores': {
                            'embedding_similarity': similarity * criteria.boost_factor
                        }
                    })

            return filtered_results

        except Exception as e:
            logger.error(f"Error executing embedding criteria: {str(e)}")
            return []

    def _execute_date_criteria(self, criteria: DateSearchCriteria) -> List[Dict[str, Any]]:
        """Execute date-based filtering using Elasticsearch date range queries."""
        try:
            # Build date filter based on operator
            if criteria.operator == DateRangeOperator.WITHIN:
                element_pks = self._get_element_pks_in_date_range(criteria.start_date, criteria.end_date)

            elif criteria.operator == DateRangeOperator.AFTER:
                element_pks = self._get_element_pks_in_date_range(criteria.exact_date, None)

            elif criteria.operator == DateRangeOperator.BEFORE:
                element_pks = self._get_element_pks_in_date_range(None, criteria.exact_date)

            elif criteria.operator == DateRangeOperator.EXACTLY:
                # For exactly, we need a tight range around the date
                start_of_day = criteria.exact_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day = criteria.exact_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                element_pks = self._get_element_pks_in_date_range(start_of_day, end_of_day)

            elif criteria.operator == DateRangeOperator.RELATIVE_DAYS:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=criteria.relative_value)
                element_pks = self._get_element_pks_in_date_range(start_date, end_date)

            elif criteria.operator == DateRangeOperator.RELATIVE_MONTHS:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=criteria.relative_value * 30)  # Approximate
                element_pks = self._get_element_pks_in_date_range(start_date, end_date)

            elif criteria.operator == DateRangeOperator.FISCAL_YEAR:
                # Assume fiscal year starts in July (customize as needed)
                start_date = datetime(criteria.year - 1, 7, 1)
                end_date = datetime(criteria.year, 6, 30, 23, 59, 59)
                element_pks = self._get_element_pks_in_date_range(start_date, end_date)

            elif criteria.operator == DateRangeOperator.CALENDAR_YEAR:
                start_date = datetime(criteria.year, 1, 1)
                end_date = datetime(criteria.year, 12, 31, 23, 59, 59)
                element_pks = self._get_element_pks_in_date_range(start_date, end_date)

            elif criteria.operator == DateRangeOperator.QUARTER:
                quarter_starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
                quarter_ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}

                start_month, start_day = quarter_starts[criteria.quarter]
                end_month, end_day = quarter_ends[criteria.quarter]

                start_date = datetime(criteria.year, start_month, start_day)
                end_date = datetime(criteria.year, end_month, end_day, 23, 59, 59)
                element_pks = self._get_element_pks_in_date_range(start_date, end_date)

            # Also filter by specificity levels if needed
            if criteria.specificity_levels:
                element_pks = self._filter_by_specificity(element_pks, criteria.specificity_levels)

            # Convert to result format
            results = []
            for element_pk in element_pks:
                results.append({
                    'element_pk': element_pk,
                    'scores': {
                        'date_relevance': 1.0  # Could calculate date relevance score
                    }
                })

            return results

        except Exception as e:
            logger.error(f"Error executing date criteria: {str(e)}")
            return []

    def _execute_topic_criteria(self, criteria: TopicSearchCriteria) -> List[Dict[str, Any]]:
        """Execute topic-based filtering using Elasticsearch's capabilities."""
        try:
            topic_results = self.search_by_text_and_topics(
                search_text=None,
                include_topics=criteria.include_topics,
                exclude_topics=criteria.exclude_topics,
                min_confidence=criteria.min_confidence,
                limit=1000
            )

            results = []
            for result in topic_results:
                results.append({
                    'element_pk': result['element_pk'],
                    'scores': {
                        'topic_confidence': result['confidence'] * criteria.boost_factor
                    }
                })

            return results

        except Exception as e:
            logger.error(f"Error executing topic criteria: {str(e)}")
            return []

    def _execute_metadata_criteria(self, criteria: MetadataSearchCriteria) -> List[Dict[str, Any]]:
        """Execute metadata-based filtering using Elasticsearch's JSON queries."""
        try:
            # Build Elasticsearch query for metadata filtering
            filter_queries = []

            # Add exact matches
            for key, value in criteria.exact_matches.items():
                filter_queries.append({
                    "wildcard": {
                        "metadata_json": f'*"{key}":"{value}"*'
                    }
                })

            # Add LIKE patterns
            for key, pattern in criteria.like_patterns.items():
                wildcard_pattern = self._convert_like_to_wildcard(pattern)
                filter_queries.append({
                    "wildcard": {
                        "metadata_json": f'*"{key}":*{wildcard_pattern}*'
                    }
                })

            # Add range filters
            for key, range_filter in criteria.range_filters.items():
                # For range queries on metadata, we'll use script queries
                conditions = []
                if 'gte' in range_filter:
                    conditions.append(f"params.value >= {range_filter['gte']}")
                if 'lte' in range_filter:
                    conditions.append(f"params.value <= {range_filter['lte']}")
                if 'gt' in range_filter:
                    conditions.append(f"params.value > {range_filter['gt']}")
                if 'lt' in range_filter:
                    conditions.append(f"params.value < {range_filter['lt']}")

                if conditions:
                    script_query = {
                        "script": {
                            "script": {
                                "source": f"if (params._source.metadata != null && params._source.metadata['{key}'] != null) {{ def value = params._source.metadata['{key}']; return {' && '.join(conditions)}; }} return false;",
                                "params": range_filter
                            }
                        }
                    }
                    filter_queries.append(script_query)

            # Add exists filters
            for key in criteria.exists_filters:
                filter_queries.append({
                    "wildcard": {
                        "metadata_json": f'*"{key}":*'
                    }
                })

            if not filter_queries:
                return []

            # Execute query
            search_body = {
                "query": {
                    "bool": {
                        "filter": filter_queries
                    }
                },
                "size": 1000
            }

            result = self.es.search(index=self.elements_index, body=search_body)
            element_pks = [int(hit['_source']["element_pk"]) for hit in result['hits']['hits']]

            results_list = []
            for element_pk in element_pks:
                results_list.append({
                    'element_pk': element_pk,
                    'scores': {
                        'metadata_relevance': 1.0
                    }
                })

            return results_list

        except Exception as e:
            logger.error(f"Error executing metadata criteria: {str(e)}")
            return []

    def _execute_element_criteria(self, criteria: ElementSearchCriteria) -> List[Dict[str, Any]]:
        """Execute element-based filtering using Elasticsearch queries."""
        try:
            filter_queries = []

            # Add element type filter
            if criteria.element_types:
                type_values = self.prepare_element_type_query(criteria.element_types)
                if type_values:
                    if len(type_values) == 1:
                        filter_queries.append({"term": {"element_type": type_values[0]}})
                    else:
                        filter_queries.append({"terms": {"element_type": type_values}})

            # Add document ID filters
            if criteria.doc_ids:
                filter_queries.append({"terms": {"doc_id": criteria.doc_ids}})

            if criteria.exclude_doc_ids:
                filter_queries.append({
                    "bool": {
                        "must_not": {
                            "terms": {"doc_id": criteria.exclude_doc_ids}
                        }
                    }
                })

            # Add content length filters using script queries
            if criteria.content_length_min is not None:
                filter_queries.append({
                    "script": {
                        "script": {
                            "source": f"doc['content_preview'].size() > 0 && doc['content_preview'].value.length() >= {criteria.content_length_min}"
                        }
                    }
                })

            if criteria.content_length_max is not None:
                filter_queries.append({
                    "script": {
                        "script": {
                            "source": f"doc['content_preview'].size() > 0 && doc['content_preview'].value.length() <= {criteria.content_length_max}"
                        }
                    }
                })

            # Add parent element filters
            if criteria.parent_element_ids:
                filter_queries.append({"terms": {"parent_id": criteria.parent_element_ids}})

            # Build and execute query
            if not filter_queries:
                # No filters specified, return all elements
                search_body = {"query": {"match_all": {}}, "size": 1000}
            else:
                search_body = {
                    "query": {
                        "bool": {
                            "filter": filter_queries
                        }
                    },
                    "size": 1000
                }

            result = self.es.search(index=self.elements_index, body=search_body)
            element_pks = [int(hit['_source']["element_pk"]) for hit in result['hits']['hits']]

            results_list = []
            for element_pk in element_pks:
                results_list.append({
                    'element_pk': element_pk,
                    'scores': {
                        'element_match': 1.0
                    }
                })

            return results_list

        except Exception as e:
            logger.error(f"Error executing element criteria: {str(e)}")
            return []

    def _combine_results(self, all_results: List[Tuple[str, List[Dict[str, Any]]]],
                         operator: LogicalOperator) -> List[Dict[str, Any]]:
        """Combine results from multiple criteria using logical operators."""

        if not all_results:
            return []

        if len(all_results) == 1:
            return all_results[0][1]  # Return the single result set

        # Extract just the result lists
        result_sets = [results for _, results in all_results]

        if operator == LogicalOperator.AND:
            return self._intersect_results(result_sets)
        elif operator == LogicalOperator.OR:
            return self._union_results(result_sets)
        elif operator == LogicalOperator.NOT:
            # NOT operation: first set minus all other sets
            if len(result_sets) >= 2:
                return self._subtract_results(result_sets[0], result_sets[1:])
            else:
                return result_sets[0]

        return []

    @staticmethod
    def _intersect_results(result_sets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Find intersection of multiple result sets."""
        if not result_sets:
            return []

        # Get element_pks from all sets and combine scores
        element_pk_sets = []
        element_scores = {}  # element_pk -> combined scores

        for result_set in result_sets:
            pk_set = set()
            for result in result_set:
                element_pk = result['element_pk']
                pk_set.add(element_pk)

                # Accumulate scores
                if element_pk not in element_scores:
                    element_scores[element_pk] = {}

                for score_type, score_value in result.get('scores', {}).items():
                    if score_type not in element_scores[element_pk]:
                        element_scores[element_pk][score_type] = []
                    element_scores[element_pk][score_type].append(score_value)

            element_pk_sets.append(pk_set)

        # Find intersection
        common_pks = element_pk_sets[0]
        for pk_set in element_pk_sets[1:]:
            common_pks = common_pks.intersection(pk_set)

        # Build result list
        results = []
        for element_pk in common_pks:
            results.append({
                'element_pk': element_pk,
                'scores': element_scores[element_pk]
            })

        return results

    @staticmethod
    def _union_results(result_sets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Find union of multiple result sets."""
        element_scores = {}  # element_pk -> combined scores

        for result_set in result_sets:
            for result in result_set:
                element_pk = result['element_pk']

                if element_pk not in element_scores:
                    element_scores[element_pk] = {}

                for score_type, score_value in result.get('scores', {}).items():
                    if score_type not in element_scores[element_pk]:
                        element_scores[element_pk][score_type] = []
                    element_scores[element_pk][score_type].append(score_value)

        # Build result list
        results = []
        for element_pk, scores in element_scores.items():
            results.append({
                'element_pk': element_pk,
                'scores': scores
            })

        return results

    @staticmethod
    def _subtract_results(base_set: List[Dict[str, Any]],
                          subtract_sets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Subtract multiple sets from base set."""
        base_pks = {result['element_pk'] for result in base_set}

        # Collect all PKs to subtract
        subtract_pks = set()
        for subtract_set in subtract_sets:
            for result in subtract_set:
                subtract_pks.add(result['element_pk'])

        # Return base results that are not in subtract sets
        final_pks = base_pks - subtract_pks

        return [result for result in base_set if result['element_pk'] in final_pks]

    def _process_search_results(self, raw_results: List[Dict[str, Any]],
                                query: StructuredSearchQuery) -> List[Dict[str, Any]]:
        """Process and enrich search results."""

        # Calculate combined scores
        for result in raw_results:
            result['final_score'] = self._calculate_combined_score(
                result.get('scores', {}),
                query.score_combination,
                query.custom_weights
            )

        # Sort by final score
        raw_results.sort(key=lambda x: x['final_score'], reverse=True)

        # Enrich with element details
        enriched_results = []
        for result in raw_results:
            element_pk = result['element_pk']
            element = self.get_element(element_pk)

            if not element:
                continue

            enriched_result = {
                'element_pk': element_pk,
                'element_id': element.get('element_id'),
                'doc_id': element.get('doc_id'),
                'element_type': element.get('element_type'),
                'content_preview': element.get('content_preview'),
                'final_score': result['final_score']
            }

            if query.include_similarity_scores:
                enriched_result['scores'] = result.get('scores', {})

            if query.include_metadata:
                enriched_result['metadata'] = element.get('metadata', {})

            if query.include_topics:
                enriched_result['topics'] = self.get_embedding_topics(element_pk)

            if query.include_element_dates:
                element_id = element.get('element_id')
                if element_id:
                    enriched_result['extracted_dates'] = self.get_element_dates(element_id)
                    enriched_result['date_count'] = len(enriched_result['extracted_dates'])

            enriched_results.append(enriched_result)

        return enriched_results

    @staticmethod
    def _calculate_combined_score(scores: Dict[str, List[float] | float],
                                  combination_method: str,
                                  weights: Dict[str, float]) -> float:
        """Calculate final combined score from multiple score types."""

        if not scores:
            return 0.0

        # Average scores of the same type
        avg_scores = {}
        for score_type, score_list in scores.items():
            if isinstance(score_list, list) and len(score_list) > 0:
                avg_scores[score_type] = sum(score_list) / len(score_list)
            elif isinstance(score_list, float):
                avg_scores[score_type] = score_list

        if not avg_scores:
            return 0.0

        if combination_method == "multiply":
            final_score = 1.0
            for score_type, score in avg_scores.items():
                weight = weights.get(score_type, 1.0)
                final_score *= (score * weight)
            return final_score

        elif combination_method == "add":
            final_score = 0.0
            for score_type, score in avg_scores.items():
                weight = weights.get(score_type, 1.0)
                final_score += (score * weight)
            return final_score

        elif combination_method == "max":
            weighted_scores = []
            for score_type, score in avg_scores.items():
                weight = weights.get(score_type, 1.0)
                weighted_scores.append(score * weight)
            return max(weighted_scores)

        elif combination_method == "weighted_avg":
            total_weighted_score = 0.0
            total_weight = 0.0
            for score_type, score in avg_scores.items():
                weight = weights.get(score_type, 1.0)
                total_weighted_score += (score * weight)
                total_weight += weight
            return total_weighted_score / total_weight if total_weight > 0 else 0.0

        return 0.0

    @staticmethod
    def _compare_similarity(similarity: float, threshold: float,
                            operator: SimilarityOperator) -> bool:
        """Compare similarity score against threshold using specified operator."""
        if operator == SimilarityOperator.GREATER_THAN:
            return similarity > threshold
        elif operator == SimilarityOperator.GREATER_EQUAL:
            return similarity >= threshold
        elif operator == SimilarityOperator.LESS_THAN:
            return similarity < threshold
        elif operator == SimilarityOperator.LESS_EQUAL:
            return similarity <= threshold
        elif operator == SimilarityOperator.EQUALS:
            return abs(similarity - threshold) < 0.001  # Small epsilon for float comparison
        return False

    def _generate_embedding(self, search_text: str) -> List[float]:
        """Generate embedding for search text."""
        try:
            if self.embedding_generator is None:
                from ..embeddings import get_embedding_generator
                config_instance = config or Config()
                self.embedding_generator = get_embedding_generator(config_instance)

            return self.embedding_generator.generate(search_text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def _get_element_pks_in_date_range(self, start_date: Optional[datetime],
                                       end_date: Optional[datetime]) -> List[int]:
        """Get element_pks that have dates within the specified range using Elasticsearch date queries."""
        if not (start_date or end_date):
            return []

        try:
            # Build Elasticsearch date range query
            range_query = {}
            if start_date and end_date:
                # Both dates specified
                range_query["timestamp"] = {
                    "gte": start_date.timestamp(),
                    "lte": end_date.timestamp()
                }
            elif start_date:
                # Only start date
                range_query["timestamp"] = {"gte": start_date.timestamp()}
            else:
                # Only end date
                range_query["timestamp"] = {"lte": end_date.timestamp()}

            search_body = {
                "query": {"range": range_query},
                "size": 10000,
                "aggs": {
                    "unique_elements": {
                        "terms": {
                            "field": "element_pk",
                            "size": 10000
                        }
                    }
                }
            }

            result = self.es.search(index=self.dates_index, body=search_body)

            # Get unique element PKs
            element_pks = []
            if 'aggregations' in result:
                for bucket in result['aggregations']['unique_elements']['buckets']:
                    element_pks.append(int(bucket['key']))

            return element_pks

        except Exception as e:
            logger.error(f"Error getting element PKs in date range: {str(e)}")
            return []

    def _filter_by_specificity(self, element_pks: List[int],
                               allowed_levels: List[str]) -> List[int]:
        """Filter element PKs by date specificity levels."""
        if not element_pks or not allowed_levels:
            return element_pks

        try:
            # Build query to filter by specificity
            search_body = {
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"element_pk": element_pks}},
                            {"terms": {"specificity_level": allowed_levels}}
                        ]
                    }
                },
                "size": 10000,
                "aggs": {
                    "unique_elements": {
                        "terms": {
                            "field": "element_pk",
                            "size": 10000
                        }
                    }
                }
            }

            result = self.es.search(index=self.dates_index, body=search_body)

            # Get unique element PKs
            filtered_pks = []
            if 'aggregations' in result:
                for bucket in result['aggregations']['unique_elements']['buckets']:
                    filtered_pks.append(int(bucket['key']))

            return filtered_pks

        except Exception as e:
            logger.error(f"Error filtering by specificity: {str(e)}")
            return element_pks

    # ========================================
    # CORE INFRASTRUCTURE METHODS (EXISTING)
    # ========================================

    def initialize(self) -> None:
        """Initialize the database by connecting to Elasticsearch and creating indices if needed."""
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError("elasticsearch is required for Elasticsearch support")

        try:
            # Create Elasticsearch client
            self.es = Elasticsearch(**{k: v for k, v in self.conn_params.items()
                                       if k not in ['index_prefix', 'vector_dimension']})

            # Test connection
            if not self.es.ping():
                raise ConnectionError("Could not connect to Elasticsearch")

            logger.info("Connected to Elasticsearch successfully")

            # Create indices if they don't exist
            self._create_indices()

            # Initialize element_pk counter
            self._initialize_counter()

            logger.info("Elasticsearch document database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Elasticsearch database: {str(e)}")
            raise

    def _create_indices(self) -> None:
        """Create Elasticsearch indices with appropriate mappings."""

        # Documents index mapping
        documents_mapping = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "doc_type": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "standard"},
                    "content_hash": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": False},  # Store as-is
                    "metadata_json": {"type": "text"},  # For searching
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            }
        }

        # Elements index mapping
        elements_mapping = {
            "mappings": {
                "properties": {
                    "element_id": {"type": "keyword"},
                    "element_pk": {"type": "long"},
                    "doc_id": {"type": "keyword"},
                    "element_type": {"type": "keyword"},
                    "parent_id": {"type": "keyword"},
                    "content_preview": {"type": "text", "analyzer": "standard"},
                    "metadata": {"type": "object", "enabled": False},
                    "metadata_json": {"type": "text"},
                    "bbox": {"type": "object", "enabled": False},
                    "page_number": {"type": "integer"},
                    "topics": {"type": "keyword"}  # Multivalued field for topics
                }
            }
        }

        # Configure full_text field based on settings
        if self.index_full_text or self.store_full_text:
            full_text_config = {"type": "text"}

            if self.index_full_text:
                full_text_config["analyzer"] = "standard"
            else:
                # Store but don't index
                full_text_config["index"] = False

            if not self.store_full_text:
                # Index but don't store
                full_text_config["store"] = False
            elif self.compress_full_text:
                # Enable the best compression for stored text
                full_text_config["store"] = True

            elements_mapping["mappings"]["properties"]["full_text"] = full_text_config

        # Relationships index mapping
        relationships_mapping = {
            "mappings": {
                "properties": {
                    "relationship_id": {"type": "keyword"},
                    "source_id": {"type": "keyword"},
                    "relationship_type": {"type": "keyword"},
                    "target_reference": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": False},
                    "metadata_json": {"type": "text"}
                }
            }
        }

        # History index mapping
        history_mapping = {
            "mappings": {
                "properties": {
                    "source_id": {"type": "keyword"},
                    "content_hash": {"type": "keyword"},
                    "last_modified": {"type": "date"},
                    "processing_count": {"type": "integer"}
                }
            }
        }

        # Embeddings index mapping
        embeddings_mapping = {
            "mappings": {
                "properties": {
                    "element_pk": {"type": "long"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.vector_dimension,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "dimensions": {"type": "integer"},
                    "topics": {"type": "keyword"},  # Multi-valued field
                    "topics_json": {"type": "text"},
                    "confidence": {"type": "float"},
                    "created_at": {"type": "date"}
                }
            }
        }

        # Dates index mapping
        dates_mapping = {
            "mappings": {
                "properties": {
                    "element_id": {"type": "keyword"},
                    "element_pk": {"type": "long"},
                    "date_type": {"type": "keyword"},  # 'absolute', 'relative', etc.
                    "timestamp": {"type": "date"},
                    "original_text": {"type": "text"},
                    "confidence": {"type": "float"},
                    "metadata": {"type": "object", "enabled": False}
                }
            }
        }

        # Create indices
        indices_to_create = [
            (self.documents_index, documents_mapping),
            (self.elements_index, elements_mapping),
            (self.relationships_index, relationships_mapping),
            (self.history_index, history_mapping),
            (self.embeddings_index, embeddings_mapping),
            (self.dates_index, dates_mapping)
        ]

        for index_name, mapping in indices_to_create:
            if not self.es.indices.exists(index=index_name):
                index_settings = {}

                # Add compression settings if enabled for elements index
                if index_name == self.elements_index and self.compress_full_text:
                    index_settings = {
                        "settings": {
                            "index": {
                                "codec": "best_compression"
                            }
                        },
                        **mapping
                    }
                else:
                    index_settings = mapping

                self.es.indices.create(index=index_name, body=index_settings)
                logger.info(f"Created Elasticsearch index: {index_name}")
            else:
                logger.debug(f"Elasticsearch index already exists: {index_name}")

        # Log text storage configuration
        config_info = self.get_text_storage_config()
        logger.info(f"Text storage config - Store: {config_info['store_full_text']}, "
                    f"Index: {config_info['index_full_text']}, "
                    f"Compress: {config_info['compress_full_text']}")
        if config_info['full_text_max_length']:
            logger.info(f"Full text will be truncated to {config_info['full_text_max_length']} characters")

    def _initialize_counter(self) -> None:
        """Initialize the element_pk counter based on highest existing value."""
        try:
            # Search for highest element_pk
            query = {
                "size": 1,
                "sort": [{"element_pk": {"order": "desc"}}],
                "query": {"match_all": {}}
            }

            result = self.es.search(index=self.elements_index, body=query)

            if result['hits']['total']['value'] > 0:
                self.element_pk_counter = int(result['hits']['hits'][0]['_source']['element_pk'])
                logger.info(f"Initialized element_pk counter to {self.element_pk_counter}")
            else:
                self.element_pk_counter = 0
                logger.info("No existing elements found, element_pk counter set to 0")

        except Exception as e:
            logger.error(f"Error initializing counter: {str(e)}")
            self.element_pk_counter = 0

    def close(self) -> None:
        """Close the database connection."""
        if self.es:
            self.es.close()
        self.es = None

    def get_last_processed_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about when a document was last processed."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            query = {
                "query": {
                    "term": {"source_id": source_id}
                }
            }

            result = self.es.search(index=self.history_index, body=query)

            if result['hits']['total']['value'] == 0:
                return None

            return result['hits']['hits'][0]['_source']

        except Exception as e:
            logger.error(f"Error getting processing history for {source_id}: {str(e)}")
            return None

    def update_processing_history(self, source_id: str, content_hash: str) -> None:
        """Update the processing history for a document."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Check if record exists
            existing = self.get_last_processed_info(source_id)
            processing_count = 1  # Default for new records

            if existing:
                processing_count = existing.get("processing_count", 0) + 1

            # Create or update record
            record = {
                "source_id": source_id,
                "content_hash": content_hash,
                "last_modified": time.time(),
                "processing_count": processing_count
            }

            self.es.index(index=self.history_index, id=source_id, body=record)
            logger.debug(f"Updated processing history for {source_id}")

        except Exception as e:
            logger.error(f"Error updating processing history for {source_id}: {str(e)}")

    def store_document(self, document: Dict[str, Any], elements: List[Dict[str, Any]],
                       relationships: List[Dict[str, Any]]) -> None:
        """Store a document with its elements and relationships."""
        if not self.es:
            raise ValueError("Database not initialized")

        source = document.get("source", "")
        content_hash = document.get("content_hash", "")

        # Check if document already exists with this source
        if source:
            existing_doc = self.get_document(source)
            if existing_doc:
                # Document exists, update it
                doc_id = existing_doc["doc_id"]
                document["doc_id"] = doc_id

                # Update all elements to use the existing doc_id
                for element in elements:
                    element["doc_id"] = doc_id

                self.update_document(doc_id, document, elements, relationships)
                return

        # New document, proceed with creation
        doc_id = document["doc_id"]

        try:
            # Prepare document for Elasticsearch
            es_document = {**document, "created_at": document.get("created_at", time.time()),
                           "updated_at": document.get("updated_at", time.time())}

            # Convert metadata to JSON if it's a dict
            if isinstance(es_document.get("metadata"), dict):
                es_document["metadata_json"] = json.dumps(es_document["metadata"])

            # Store document
            self.es.index(index=self.documents_index, id=doc_id, body=es_document)

            # Process elements for bulk indexing
            bulk_actions = []
            for element in elements:
                es_element = {**element}

                # Generate element_pk if not present
                if "element_pk" not in es_element:
                    self.element_pk_counter += 1
                    es_element["element_pk"] = self.element_pk_counter
                    # Store back in original element
                    element["element_pk"] = es_element["element_pk"]

                # Extract and process full content based on configuration
                if "full_content" in element:
                    full_content = element["full_content"]

                    # Apply length limit if configured
                    if self.full_text_max_length and len(full_content) > self.full_text_max_length:
                        full_content = full_content[:self.full_text_max_length] + "..."
                        logger.debug(
                            f"Truncated full_text for element {es_element['element_id']} to {self.full_text_max_length} characters")

                    # Store/index full text based on configuration
                    if self.store_full_text or self.index_full_text:
                        es_element["full_text"] = full_content

                    # Always remove the original full_content field to avoid duplication
                    if "full_content" in es_element:
                        del es_element["full_content"]

                # Convert metadata to JSON if it's a dict
                if isinstance(es_element.get("metadata"), dict):
                    es_element["metadata_json"] = json.dumps(es_element["metadata"])

                bulk_actions.append({
                    "_index": self.elements_index,
                    "_id": es_element["element_id"],
                    "_source": es_element
                })

            # Bulk index elements
            if bulk_actions:
                bulk(self.es, bulk_actions)

            # Process relationships for bulk indexing
            bulk_actions = []
            for rel in relationships:
                es_rel = {**rel}

                # Convert metadata to JSON if it's a dict
                if isinstance(es_rel.get("metadata"), dict):
                    es_rel["metadata_json"] = json.dumps(es_rel["metadata"])

                bulk_actions.append({
                    "_index": self.relationships_index,
                    "_id": es_rel["relationship_id"],
                    "_source": es_rel
                })

            # Bulk index relationships
            if bulk_actions:
                bulk(self.es, bulk_actions)

            # Update processing history
            if source:
                self.update_processing_history(source, content_hash)

            logger.info(
                f"Stored document {doc_id} with {len(elements)} elements and {len(relationships)} relationships")

        except Exception as e:
            logger.error(f"Error storing document {doc_id}: {str(e)}")
            raise

    def update_document(self, doc_id: str, document: Dict[str, Any],
                        elements: List[Dict[str, Any]],
                        relationships: List[Dict[str, Any]]) -> None:
        """Update an existing document."""
        if not self.es:
            raise ValueError("Database not initialized")

        # Check if document exists
        existing_doc = self.get_document(doc_id)
        if not existing_doc:
            raise ValueError(f"Document not found: {doc_id}")

        try:
            # Update document timestamps
            document["updated_at"] = time.time()
            if "created_at" not in document:
                document["created_at"] = existing_doc.get("created_at", time.time())

            # Get existing elements to clean up embeddings and dates
            existing_elements = self.get_document_elements(doc_id)
            existing_element_pks = [int(elem.get("element_pk", 0)) for elem in existing_elements]
            existing_element_ids = [elem["element_id"] for elem in existing_elements]

            # Delete existing document elements
            delete_query = {"query": {"term": {"doc_id": doc_id}}}
            self.es.delete_by_query(index=self.elements_index, body=delete_query)

            # Delete existing embeddings for document elements
            if existing_element_pks:
                delete_query = {"query": {"terms": {"element_pk": existing_element_pks}}}
                self.es.delete_by_query(index=self.embeddings_index, body=delete_query)

            # Delete existing dates for document elements
            if existing_element_ids:
                delete_query = {"query": {"terms": {"element_id": existing_element_ids}}}
                self.es.delete_by_query(index=self.dates_index, body=delete_query)

            # Delete existing relationships for document elements
            if existing_element_ids:
                delete_query = {"query": {"terms": {"source_id": existing_element_ids}}}
                self.es.delete_by_query(index=self.relationships_index, body=delete_query)

            # Prepare updated document
            es_document = {**document}
            if isinstance(es_document.get("metadata"), dict):
                es_document["metadata_json"] = json.dumps(es_document["metadata"])

            # Store updated document
            self.es.index(index=self.documents_index, id=doc_id, body=es_document)

            # Process and store updated elements and relationships (same as store_document)
            bulk_actions = []
            for element in elements:
                es_element = {**element}

                if "element_pk" not in es_element:
                    self.element_pk_counter += 1
                    es_element["element_pk"] = self.element_pk_counter
                    element["element_pk"] = es_element["element_pk"]

                # Extract and process full content based on configuration
                if "full_content" in element:
                    full_content = element["full_content"]

                    # Apply length limit if configured
                    if self.full_text_max_length and len(full_content) > self.full_text_max_length:
                        full_content = full_content[:self.full_text_max_length] + "..."
                        logger.debug(
                            f"Truncated full_text for element {es_element['element_id']} to {self.full_text_max_length} characters")

                    # Store/index full text based on configuration
                    if self.store_full_text or self.index_full_text:
                        es_element["full_text"] = full_content

                    # Always remove the original full_content field to avoid duplication
                    if "full_content" in es_element:
                        del es_element["full_content"]

                if isinstance(es_element.get("metadata"), dict):
                    es_element["metadata_json"] = json.dumps(es_element["metadata"])

                bulk_actions.append({
                    "_index": self.elements_index,
                    "_id": es_element["element_id"],
                    "_source": es_element
                })

            if bulk_actions:
                bulk(self.es, bulk_actions)

            # Process relationships
            bulk_actions = []
            for rel in relationships:
                es_rel = {**rel}
                if isinstance(es_rel.get("metadata"), dict):
                    es_rel["metadata_json"] = json.dumps(es_rel["metadata"])

                bulk_actions.append({
                    "_index": self.relationships_index,
                    "_id": es_rel["relationship_id"],
                    "_source": es_rel
                })

            if bulk_actions:
                bulk(self.es, bulk_actions)

            # Update processing history
            source = document.get("source", "")
            content_hash = document.get("content_hash", "")
            if source:
                self.update_processing_history(source, content_hash)

            logger.info(
                f"Updated document {doc_id} with {len(elements)} elements and {len(relationships)} relationships")

        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {str(e)}")
            raise

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Try to get by doc_id
            try:
                result = self.es.get(index=self.documents_index, id=doc_id)
                document = result['_source']
            except NotFoundError:
                # Try to get by source field
                query = {"query": {"term": {"source": doc_id}}}
                result = self.es.search(index=self.documents_index, body=query)

                if result['hits']['total']['value'] == 0:
                    return None

                document = result['hits']['hits'][0]['_source']

            # Parse metadata_json if present
            if "metadata_json" in document and not document.get("metadata"):
                try:
                    document["metadata"] = json.loads(document["metadata_json"])
                except:
                    pass

            return document

        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {str(e)}")
            return None

    def get_document_elements(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get elements for a document."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # First try to get document by doc_id to handle case where source is provided
            document = self.get_document(doc_id)
            if document:
                doc_id = document["doc_id"]

            # Get elements
            query = {
                "query": {"term": {"doc_id": doc_id}},
                "size": 10000,
                "sort": [{"element_pk": {"order": "asc"}}]
            }

            result = self.es.search(index=self.elements_index, body=query)

            elements = []
            for hit in result['hits']['hits']:
                element = hit['_source']

                # Parse metadata_json if present
                if "metadata_json" in element and not element.get("metadata"):
                    try:
                        element["metadata"] = json.loads(element["metadata_json"])
                    except:
                        pass

                elements.append(element)

            return elements

        except Exception as e:
            logger.error(f"Error getting document elements for {doc_id}: {str(e)}")
            return []

    def get_document_relationships(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get relationships for a document."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Get all element IDs for this document
            elements = self.get_document_elements(doc_id)
            element_ids = [element["element_id"] for element in elements]

            if not element_ids:
                return []

            # Find relationships involving these elements
            query = {
                "query": {"terms": {"source_id": element_ids}},
                "size": 10000
            }

            result = self.es.search(index=self.relationships_index, body=query)

            relationships = []
            for hit in result['hits']['hits']:
                relationship = hit['_source']

                # Parse metadata_json if present
                if "metadata_json" in relationship and not relationship.get("metadata"):
                    try:
                        relationship["metadata"] = json.loads(relationship["metadata_json"])
                    except:
                        pass

                relationships.append(relationship)

            return relationships

        except Exception as e:
            logger.error(f"Error getting document relationships for {doc_id}: {str(e)}")
            return []

    def get_element(self, element_id_or_pk: Union[str, int]) -> Optional[Dict[str, Any]]:
        """Get element by ID or PK."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Try to interpret as element_pk (integer) first
            try:
                element_pk = int(element_id_or_pk)
                query = {"query": {"term": {"element_pk": element_pk}}}
                result = self.es.search(index=self.elements_index, body=query)
            except (ValueError, TypeError):
                # If not an integer, treat as element_id (string)
                try:
                    result = self.es.get(index=self.elements_index, id=element_id_or_pk)
                    element = result['_source']

                    # Parse metadata_json if present
                    if "metadata_json" in element and not element.get("metadata"):
                        try:
                            element["metadata"] = json.loads(element["metadata_json"])
                        except:
                            pass

                    return element
                except NotFoundError:
                    return None

            if result['hits']['total']['value'] == 0:
                return None

            element = result['hits']['hits'][0]['_source']

            # Parse metadata_json if present
            if "metadata_json" in element and not element.get("metadata"):
                try:
                    element["metadata"] = json.loads(element["metadata_json"])
                except:
                    pass

            return element

        except Exception as e:
            logger.error(f"Error getting element {element_id_or_pk}: {str(e)}")
            return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all associated elements and relationships."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Check if document exists
            document = self.get_document(doc_id)
            if not document:
                return False

            # Get all elements for this document to clean up
            elements = self.get_document_elements(doc_id)
            element_pks = [int(elem.get("element_pk", 0)) for elem in elements]
            element_ids = [elem["element_id"] for elem in elements]

            # Delete embeddings for these elements
            if element_pks:
                delete_query = {"query": {"terms": {"element_pk": element_pks}}}
                self.es.delete_by_query(index=self.embeddings_index, body=delete_query)

            # Delete dates for these elements
            if element_ids:
                delete_query = {"query": {"terms": {"element_id": element_ids}}}
                self.es.delete_by_query(index=self.dates_index, body=delete_query)

            # Delete relationships involving these elements
            if element_ids:
                delete_query = {"query": {"terms": {"source_id": element_ids}}}
                self.es.delete_by_query(index=self.relationships_index, body=delete_query)

            # Delete elements
            delete_query = {"query": {"term": {"doc_id": doc_id}}}
            self.es.delete_by_query(index=self.elements_index, body=delete_query)

            # Delete document
            self.es.delete(index=self.documents_index, id=doc_id)

            logger.info(f"Deleted document {doc_id} with {len(element_ids)} elements")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False

    # ========================================
    # SEARCH METHODS (EXISTING)
    # ========================================

    def find_documents(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find documents matching query with support for LIKE patterns."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Build Elasticsearch query
            es_query = {"match_all": {}}  # Default to all documents
            filters = []

            if query:
                for key, value in query.items():
                    if key == "metadata":
                        # Handle metadata exact matches
                        for meta_key, meta_value in value.items():
                            filters.append({
                                "wildcard": {
                                    "metadata_json": f'*"{meta_key}":"{meta_value}"*'
                                }
                            })
                    elif key == "metadata_like":
                        # Handle metadata LIKE patterns
                        for meta_key, meta_value in value.items():
                            pattern = self._convert_like_to_wildcard(meta_value)
                            filters.append({
                                "wildcard": {
                                    "metadata_json": f'*"{meta_key}":*{pattern}*'
                                }
                            })
                    elif key.endswith("_like") or key.endswith("_ilike"):
                        # Handle LIKE patterns
                        field_name = key[:-5] if key.endswith("_like") else key[:-6]
                        pattern = self._convert_like_to_wildcard(value)
                        filters.append({"wildcard": {field_name: pattern}})
                    elif isinstance(value, list):
                        # Handle list values
                        filters.append({"terms": {key: value}})
                    else:
                        # Simple equality
                        filters.append({"term": {key: value}})

            # Build final query
            if filters:
                es_query = {"bool": {"filter": filters}}

            search_body = {
                "query": es_query,
                "size": limit
            }

            result = self.es.search(index=self.documents_index, body=search_body)

            documents = []
            for hit in result['hits']['hits']:
                document = hit['_source']

                # Parse metadata_json if present
                if "metadata_json" in document and not document.get("metadata"):
                    try:
                        document["metadata"] = json.loads(document["metadata_json"])
                    except:
                        pass

                documents.append(document)

            return documents

        except Exception as e:
            logger.error(f"Error finding documents: {str(e)}")
            return []

    def find_elements(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Find elements matching query with support for LIKE patterns and ElementType enums."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Build Elasticsearch query
            es_query = {"match_all": {}}
            filters = []

            if query:
                for key, value in query.items():
                    if key == "metadata":
                        # Handle metadata exact matches
                        for meta_key, meta_value in value.items():
                            filters.append({
                                "wildcard": {
                                    "metadata_json": f'*"{meta_key}":"{meta_value}"*'
                                }
                            })
                    elif key == "metadata_like":
                        # Handle metadata LIKE patterns
                        for meta_key, meta_value in value.items():
                            pattern = self._convert_like_to_wildcard(meta_value)
                            filters.append({
                                "wildcard": {
                                    "metadata_json": f'*"{meta_key}":*{pattern}*'
                                }
                            })
                    elif key.endswith("_like") or key.endswith("_ilike"):
                        # Handle LIKE patterns
                        field_name = key[:-5] if key.endswith("_like") else key[:-6]
                        pattern = self._convert_like_to_wildcard(value)
                        filters.append({"wildcard": {field_name: pattern}})
                    elif key == "element_type":
                        # Handle ElementType enums, strings, and lists
                        type_values = self.prepare_element_type_query(value)
                        if type_values:
                            if len(type_values) == 1:
                                filters.append({"term": {"element_type": type_values[0]}})
                            else:
                                filters.append({"terms": {"element_type": type_values}})
                    elif isinstance(value, list):
                        # Handle other list values
                        filters.append({"terms": {key: value}})
                    else:
                        # Simple equality
                        filters.append({"term": {key: value}})

            # Build final query
            if filters:
                es_query = {"bool": {"filter": filters}}

            search_body = {
                "query": es_query,
                "size": limit
            }

            result = self.es.search(index=self.elements_index, body=search_body)

            elements = []
            for hit in result['hits']['hits']:
                element = hit['_source']

                # Parse metadata_json if present
                if "metadata_json" in element and not element.get("metadata"):
                    try:
                        element["metadata"] = json.loads(element["metadata_json"])
                    except:
                        pass

                elements.append(element)

            return elements

        except Exception as e:
            logger.error(f"Error finding elements: {str(e)}")
            return []

    def search_elements_by_content(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search elements by content."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Build search fields based on configuration
            search_fields = ["content_preview"]
            if self.index_full_text:
                search_fields.append("full_text")

            query = {
                "query": {
                    "multi_match": {
                        "query": search_text,
                        "fields": search_fields,
                        "type": "best_fields"
                    }
                },
                "size": limit
            }

            result = self.es.search(index=self.elements_index, body=query)

            elements = []
            for hit in result['hits']['hits']:
                element = hit['_source']
                element['_score'] = hit['_score']  # Include relevance score

                # Parse metadata_json if present
                if "metadata_json" in element and not element.get("metadata"):
                    try:
                        element["metadata"] = json.loads(element["metadata_json"])
                    except:
                        pass

                elements.append(element)

            return elements

        except Exception as e:
            logger.error(f"Error searching elements by content: {str(e)}")
            return []

    # ========================================
    # EMBEDDING METHODS (EXISTING)
    # ========================================

    def store_embedding(self, element_pk: int, embedding: VectorType) -> None:
        """Store embedding for an element."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Verify element exists
            element = self.get_element(element_pk)
            if not element:
                raise ValueError(f"Element not found: {element_pk}")

            # Create embedding document
            embedding_doc = {
                "element_pk": element_pk,
                "embedding": embedding,
                "dimensions": len(embedding),
                "topics": [],
                "confidence": 1.0,
                "created_at": time.time()
            }

            # Store in embeddings index
            self.es.index(index=self.embeddings_index, id=str(element_pk), body=embedding_doc)
            logger.debug(f"Stored embedding for element {element_pk}")

        except Exception as e:
            logger.error(f"Error storing embedding for element {element_pk}: {str(e)}")
            raise

    def get_embedding(self, element_pk: int) -> Optional[VectorType]:
        """Get embedding for an element."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            result = self.es.get(index=self.embeddings_index, id=str(element_pk))
            return result['_source'].get("embedding")
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error getting embedding for element {element_pk}: {str(e)}")
            return None

    def search_by_embedding(self, query_embedding: VectorType, limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Search elements by embedding similarity with optional filtering."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Build Elasticsearch kNN query
            knn_query = {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": limit,
                "num_candidates": limit * 10  # Search more candidates for better results
            }

            # Add filters if provided
            if filter_criteria:
                # Get element IDs that match the filter criteria
                matching_elements = self.find_elements(filter_criteria, limit=10000)
                if not matching_elements:
                    return []

                element_pks = [int(elem["element_pk"]) for elem in matching_elements]
                knn_query["filter"] = {"terms": {"element_pk": element_pks}}

            search_body = {
                "knn": knn_query,
                "size": limit
            }

            result = self.es.search(index=self.embeddings_index, body=search_body)

            similarities = []
            for hit in result['hits']['hits']:
                element_pk = int(hit['_source']['element_pk'])
                similarity = float(hit['_score'])
                similarities.append((element_pk, similarity))

            return similarities

        except Exception as e:
            logger.error(f"Error searching by embedding: {str(e)}")
            return []

    def search_by_text(self, search_text: str, limit: int = 10,
                       filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Search elements by semantic similarity to the provided text."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Generate embedding for search text
            if self.embedding_generator is None:
                try:
                    from ..embeddings import get_embedding_generator
                    config_instance = config or Config()
                    self.embedding_generator = get_embedding_generator(config_instance)
                except Exception as e:
                    logger.warning(f"Could not load embedding generator: {str(e)}")
                    # Fall back to text search only
                    return self._fallback_text_search(search_text, limit, filter_criteria)

            query_embedding = self.embedding_generator.generate(search_text)

            # Perform hybrid search: combine text search and vector search
            text_results = self._text_search_scores(search_text, limit * 2, filter_criteria)
            vector_results = self.search_by_embedding(query_embedding, limit, filter_criteria)

            # Merge and rank results
            return self._merge_search_results(text_results, vector_results, limit)

        except Exception as e:
            logger.error(f"Error in semantic search by text: {str(e)}")
            return []

    def _text_search_scores(self, search_text: str, limit: int,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Perform text search and return element_pk, score tuples."""
        try:
            # Build filters
            filters = []
            if filter_criteria:
                for key, value in filter_criteria.items():
                    if isinstance(value, list):
                        filters.append({"terms": {key: value}})
                    else:
                        filters.append({"term": {key: value}})

            # Build search fields based on configuration
            search_fields = ["content_preview^2"]  # Always boost content_preview
            if self.index_full_text:
                search_fields.append("full_text")

            # Build query
            query = {
                "multi_match": {
                    "query": search_text,
                    "fields": search_fields,
                    "type": "best_fields"
                }
            }

            if filters:
                query = {"bool": {"must": query, "filter": filters}}

            search_body = {
                "query": query,
                "size": limit
            }

            result = self.es.search(index=self.elements_index, body=search_body)

            text_scores = []
            for hit in result['hits']['hits']:
                element_pk = int(hit['_source']['element_pk'])
                score = float(hit['_score']) / 10.0  # Normalize ES text scores
                text_scores.append((element_pk, score))

            return text_scores

        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            return []

    def _fallback_text_search(self, search_text: str, limit: int,
                              filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Fallback to text search only when embedding generator is not available."""
        return self._text_search_scores(search_text, limit, filter_criteria)

    @staticmethod
    def _merge_search_results(text_results: List[Tuple[int, float]],
                              vector_results: List[Tuple[int, float]],
                              limit: int) -> List[Tuple[int, float]]:
        """Merge text and vector search results with weighted scoring."""

        # Convert to dictionaries for easier merging
        text_scores = {pk: score for pk, score in text_results}
        vector_scores = {pk: score for pk, score in vector_results}

        # Combine scores
        combined_scores = {}
        all_pks = set(text_scores.keys()) | set(vector_scores.keys())

        for pk in all_pks:
            text_score = text_scores.get(pk, 0.0)
            vector_score = vector_scores.get(pk, 0.0)

            # Weighted average: 30% text, 70% vector
            final_score = 0.3 * text_score + 0.7 * vector_score
            combined_scores[pk] = final_score

        # Sort by score and return top results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]

    def get_outgoing_relationships(self, element_pk: int) -> List[ElementRelationship]:
        """Find all relationships where the specified element_pk is the source."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Get the element to find its element_id
            element = self.get_element(element_pk)
            if not element:
                logger.warning(f"Element with PK {element_pk} not found")
                return []

            element_id = element.get("element_id")
            if not element_id:
                logger.warning(f"Element with PK {element_pk} has no element_id")
                return []

            element_type = element.get("element_type", "")

            # Search for relationships where this element is the source
            query = {
                "query": {"term": {"source_id": element_id}},
                "size": 10000
            }

            result = self.es.search(index=self.relationships_index, body=query)

            relationships = []
            for hit in result['hits']['hits']:
                rel_doc = hit['_source']

                # Get target element if it exists
                target_reference = rel_doc.get("target_reference", "")
                target_element = None
                target_element_pk = None
                target_element_type = None
                target_content_preview = None

                if target_reference:
                    target_element = self.get_element(target_reference)
                    if target_element:
                        target_element_pk = target_element.get("element_pk")
                        target_element_type = target_element.get("element_type")
                        target_content_preview = target_element.get("content_preview", "")

                # Parse metadata if it exists
                metadata = {}
                if "metadata_json" in rel_doc:
                    try:
                        metadata = json.loads(rel_doc["metadata_json"])
                    except:
                        metadata = rel_doc.get("metadata", {})

                # Create relationship object
                relationship = ElementRelationship(
                    relationship_id=rel_doc.get("relationship_id", ""),
                    source_id=element_id,
                    source_element_pk=element_pk,
                    source_element_type=element_type,
                    relationship_type=rel_doc.get("relationship_type", ""),
                    target_reference=target_reference,
                    target_element_pk=target_element_pk,
                    target_element_type=target_element_type,
                    target_content_preview=target_content_preview,
                    doc_id=rel_doc.get("doc_id"),
                    metadata=metadata,
                    is_source=True
                )

                relationships.append(relationship)

            return relationships

        except Exception as e:
            logger.error(f"Error getting outgoing relationships for element {element_pk}: {str(e)}")
            return []

    # ========================================
    # DATE STORAGE AND SEARCH METHODS (EXISTING)
    # ========================================

    def store_element_dates(self, element_id: str, dates: List[Dict[str, Any]]) -> None:
        """Store extracted dates associated with an element."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Get element info
            element = self.get_element(element_id)
            if not element:
                raise ValueError(f"Element not found: {element_id}")

            element_pk = element.get("element_pk")

            # Prepare bulk actions for dates
            bulk_actions = []
            for i, date_info in enumerate(dates):
                date_doc = {
                    **date_info,
                    "element_id": element_id,
                    "element_pk": element_pk
                }

                doc_id = f"{element_id}_{i}"
                bulk_actions.append({
                    "_index": self.dates_index,
                    "_id": doc_id,
                    "_source": date_doc
                })

            # Bulk index dates
            if bulk_actions:
                bulk(self.es, bulk_actions)

            logger.debug(f"Stored {len(dates)} dates for element {element_id}")

        except Exception as e:
            logger.error(f"Error storing dates for element {element_id}: {str(e)}")
            raise

    def get_element_dates(self, element_id: str) -> List[Dict[str, Any]]:
        """Get all dates associated with an element."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            query = {
                "query": {"term": {"element_id": element_id}},
                "size": 10000
            }

            result = self.es.search(index=self.dates_index, body=query)

            dates = [hit['_source'] for hit in result['hits']['hits']]
            return dates

        except Exception as e:
            logger.error(f"Error getting dates for element {element_id}: {str(e)}")
            return []

    def store_embedding_with_dates(self, element_pk: int, embedding: VectorType,
                                   dates: List[Dict[str, Any]]) -> None:
        """Store both embedding and dates for an element in a single operation."""
        # Store embedding
        self.store_embedding(element_pk, embedding)

        # Get element_id for storing dates
        element = self.get_element(element_pk)
        if element:
            element_id = element.get("element_id")
            if element_id:
                self.store_element_dates(element_id, dates)

    def delete_element_dates(self, element_id: str) -> bool:
        """Delete all dates associated with an element."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            delete_query = {"query": {"term": {"element_id": element_id}}}
            result = self.es.delete_by_query(index=self.dates_index, body=delete_query)

            deleted_count = result.get('deleted', 0)
            return deleted_count > 0

        except Exception as e:
            logger.error(f"Error deleting dates for element {element_id}: {str(e)}")
            return False

    def search_elements_by_date_range(self, start_date, end_date, limit: int = 100) -> List[Dict[str, Any]]:
        """Find elements that contain dates within a specified range."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Convert datetime objects to timestamps if necessary
            start_timestamp = start_date.timestamp() if hasattr(start_date, 'timestamp') else start_date
            end_timestamp = end_date.timestamp() if hasattr(end_date, 'timestamp') else end_date

            # Search for dates in range
            query = {
                "query": {
                    "range": {
                        "timestamp": {
                            "gte": start_timestamp,
                            "lte": end_timestamp
                        }
                    }
                },
                "size": 10000,
                "aggs": {
                    "unique_elements": {
                        "terms": {
                            "field": "element_id",
                            "size": limit
                        }
                    }
                }
            }

            result = self.es.search(index=self.dates_index, body=query)

            # Get unique element IDs
            element_ids = []
            if 'aggregations' in result:
                for bucket in result['aggregations']['unique_elements']['buckets']:
                    element_ids.append(bucket['key'])

            # Get element details
            elements = []
            for element_id in element_ids:
                element = self.get_element(element_id)
                if element:
                    elements.append(element)

            return elements

        except Exception as e:
            logger.error(f"Error searching elements by date range: {str(e)}")
            return []

    def search_by_text_and_date_range(self, search_text: str, start_date=None,
                                      end_date=None, limit: int = 10) -> List[Tuple[int, float]]:
        """Search elements by semantic similarity AND date range."""
        # Get elements in date range if dates are provided
        date_filtered_elements = None
        if start_date and end_date:
            date_elements = self.search_elements_by_date_range(start_date, end_date, limit=10000)
            date_filtered_elements = [elem["element_pk"] for elem in date_elements]

        # Apply date filter to search criteria
        filter_criteria = {}
        if date_filtered_elements is not None:
            filter_criteria["element_pk"] = date_filtered_elements

        return self.search_by_text(search_text, limit, filter_criteria)

    def search_by_embedding_and_date_range(self, query_embedding: VectorType,
                                           start_date=None, end_date=None,
                                           limit: int = 10) -> List[Tuple[int, float]]:
        """Search elements by embedding similarity AND date range."""
        # Get elements in date range if dates are provided
        date_filtered_elements = None
        if start_date and end_date:
            date_elements = self.search_elements_by_date_range(start_date, end_date, limit=10000)
            date_filtered_elements = [elem["element_pk"] for elem in date_elements]

        # Apply date filter to search criteria
        filter_criteria = {}
        if date_filtered_elements is not None:
            filter_criteria["element_pk"] = date_filtered_elements

        return self.search_by_embedding(query_embedding, limit, filter_criteria)

    def get_elements_with_dates(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all elements that have associated dates."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Get unique element IDs that have dates
            query = {
                "size": 0,
                "aggs": {
                    "unique_elements": {
                        "terms": {
                            "field": "element_id",
                            "size": limit
                        }
                    }
                }
            }

            result = self.es.search(index=self.dates_index, body=query)

            # Get element details
            elements = []
            if 'aggregations' in result:
                for bucket in result['aggregations']['unique_elements']['buckets']:
                    element_id = bucket['key']
                    element = self.get_element(element_id)
                    if element:
                        elements.append(element)

            return elements

        except Exception as e:
            logger.error(f"Error getting elements with dates: {str(e)}")
            return []

    def get_date_statistics(self) -> Dict[str, Any]:
        """Get statistics about dates in the database."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            query = {
                "size": 0,
                "aggs": {
                    "total_dates": {"value_count": {"field": "timestamp"}},
                    "unique_elements": {"cardinality": {"field": "element_id"}},
                    "date_range": {
                        "stats": {"field": "timestamp"}
                    },
                    "date_types": {
                        "terms": {"field": "date_type"}
                    }
                }
            }

            result = self.es.search(index=self.dates_index, body=query)

            if 'aggregations' not in result:
                return {}

            aggs = result['aggregations']
            stats = {
                "total_dates": aggs['total_dates']['value'],
                "unique_elements": aggs['unique_elements']['value'],
                "earliest_date": aggs['date_range']['min'],
                "latest_date": aggs['date_range']['max'],
                "date_types": {bucket['key']: bucket['doc_count']
                               for bucket in aggs['date_types']['buckets']}
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting date statistics: {str(e)}")
            return {}

    # ========================================
    # TOPIC SUPPORT METHODS (ENHANCED)
    # ========================================

    def supports_topics(self) -> bool:
        """Indicate whether this backend supports topic-aware embeddings."""
        return True

    def store_embedding_with_topics(self, element_pk: int, embedding: VectorType,
                                    topics: List[str], confidence: float = 1.0) -> None:
        """Store embedding for an element with topic assignments."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Verify element exists
            element = self.get_element(element_pk)
            if not element:
                raise ValueError(f"Element not found: {element_pk}")

            # Create enhanced embedding document
            embedding_doc = {
                "element_pk": element_pk,
                "embedding": embedding,
                "dimensions": len(embedding),
                "topics": topics,
                "topics_json": json.dumps(topics),
                "confidence": confidence,
                "created_at": time.time()
            }

            # Store in embeddings index
            self.es.index(index=self.embeddings_index, id=str(element_pk), body=embedding_doc)
            logger.debug(f"Stored embedding with topics for element {element_pk}")

        except Exception as e:
            logger.error(f"Error storing embedding with topics for element {element_pk}: {str(e)}")
            raise

    def search_by_text_and_topics(self, search_text: str = None,
                                  include_topics: Optional[List[str]] = None,
                                  exclude_topics: Optional[List[str]] = None,
                                  min_confidence: float = 0.7,
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """Search elements by text with topic filtering."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            # Generate embedding for search text if provided
            query_embedding = None
            if search_text:
                if self.embedding_generator is None:
                    from ..embeddings import get_embedding_generator
                    config_instance = config or Config()
                    self.embedding_generator = get_embedding_generator(config_instance)

                query_embedding = self.embedding_generator.generate(search_text)

            # Build Elasticsearch query
            filters = [{"range": {"confidence": {"gte": min_confidence}}}]

            # Add topic filtering
            if include_topics:
                include_queries = []
                for pattern in include_topics:
                    wildcard_pattern = self._convert_like_to_wildcard(pattern)
                    include_queries.append({"wildcard": {"topics": wildcard_pattern}})

                if include_queries:
                    if len(include_queries) == 1:
                        filters.append(include_queries[0])
                    else:
                        filters.append({"bool": {"should": include_queries}})

            if exclude_topics:
                for pattern in exclude_topics:
                    wildcard_pattern = self._convert_like_to_wildcard(pattern)
                    filters.append({"bool": {"must_not": {"wildcard": {"topics": wildcard_pattern}}}})

            # Build query
            if query_embedding:
                # Use kNN search with filters
                search_body = {
                    "knn": {
                        "field": "embedding",
                        "query_vector": query_embedding,
                        "k": limit,
                        "num_candidates": limit * 10,
                        "filter": {"bool": {"filter": filters}}
                    },
                    "size": limit
                }
            else:
                # Just filter by topics and confidence
                search_body = {
                    "query": {"bool": {"filter": filters}},
                    "size": limit
                }

            result = self.es.search(index=self.embeddings_index, body=search_body)

            # Process results
            filtered_results = []
            for hit in result['hits']['hits']:
                doc = hit['_source']
                result_dict = {
                    'element_pk': int(doc['element_pk']),
                    'confidence': float(doc.get('confidence', 1.0)),
                    'topics': doc.get('topics', []),
                    'similarity': float(hit.get('_score', 0.0))
                }
                filtered_results.append(result_dict)

            return filtered_results

        except Exception as e:
            logger.error(f"Error in topic-aware search: {str(e)}")
            return []

    def get_topic_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about topic distribution across embeddings."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            query = {
                "size": 0,
                "aggs": {
                    "topics": {
                        "terms": {
                            "field": "topics",
                            "size": 1000
                        },
                        "aggs": {
                            "avg_confidence": {
                                "avg": {"field": "confidence"}
                            },
                            "unique_documents": {
                                "cardinality": {"script": "doc['element_pk'].value"}
                            }
                        }
                    }
                }
            }

            result = self.es.search(index=self.embeddings_index, body=query)

            topic_stats = {}
            if 'aggregations' in result:
                for bucket in result['aggregations']['topics']['buckets']:
                    topic = bucket['key']
                    topic_stats[topic] = {
                        'embedding_count': bucket['doc_count'],
                        'document_count': bucket['unique_documents']['value'],
                        'avg_embedding_confidence': bucket['avg_confidence']['value']
                    }

            return topic_stats

        except Exception as e:
            logger.error(f"Error getting topic statistics: {str(e)}")
            return {}

    def get_embedding_topics(self, element_pk: int) -> List[str]:
        """Get topics assigned to a specific embedding."""
        if not self.es:
            raise ValueError("Database not initialized")

        try:
            result = self.es.get(index=self.embeddings_index, id=str(element_pk))
            return result['_source'].get('topics', [])
        except NotFoundError:
            return []
        except Exception as e:
            logger.error(f"Error getting topics for element {element_pk}: {str(e)}")
            return []

    # ========================================
    # DATE UTILITY METHODS (NEW)
    # ========================================

    def get_date_range_for_element(self, element_id: str) -> Optional[Tuple[datetime, datetime]]:
        """Get the date range (earliest, latest) for an element."""
        dates = self.get_element_dates(element_id)
        if not dates:
            return None

        timestamps = [d['timestamp'] for d in dates if 'timestamp' in d and d['timestamp'] is not None]
        if not timestamps:
            return None

        earliest = datetime.fromtimestamp(min(timestamps))
        latest = datetime.fromtimestamp(max(timestamps))
        return earliest, latest

    def count_dates_for_element(self, element_id: str) -> int:
        """Count the number of dates associated with an element."""
        dates = self.get_element_dates(element_id)
        return len(dates)

    def get_elements_by_year(self, year: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get elements that contain dates from a specific year."""
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        return self.search_elements_by_date_range(start_date, end_date, limit)

    def get_elements_by_month(self, year: int, month: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get elements that contain dates from a specific month."""
        import calendar
        start_date = datetime(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        end_date = datetime(year, month, last_day, 23, 59, 59)
        return self.search_elements_by_date_range(start_date, end_date, limit)

    def update_element_dates(self, element_id: str, dates: List[Dict[str, Any]]) -> None:
        """Update dates for an element (delete old, store new)."""
        self.delete_element_dates(element_id)
        self.store_element_dates(element_id, dates)

    # ========================================
    # HIERARCHY METHODS (NEW)
    # ========================================

    def get_results_outline(self, elements: List[Tuple[int, float]]) -> List[ElementHierarchical]:
        """
        For search results, create a hierarchical outline showing element ancestry.
        """
        # Dictionary to store element_pk -> score mapping for quick lookup
        element_scores = {element_pk: score for element_pk, score in elements}

        # Set to track processed element_pks to avoid duplicates
        processed_elements = set()

        # Final result structure
        result_tree: List[ElementHierarchical] = []

        # Process each element from the search results
        for element_pk, score in elements:
            if element_pk in processed_elements:
                continue

            # Find the complete ancestry path for this element
            ancestry_path = self._get_element_ancestry_path(element_pk)

            if not ancestry_path:
                continue

            # Mark this element as processed
            processed_elements.add(element_pk)

            # Start with the root level
            current_level = result_tree

            # Process each ancestor from root to the target element
            for i, ancestor in enumerate(ancestry_path):
                ancestor_pk = ancestor.element_pk

                # Check if this ancestor is already in the current level
                existing_idx = None
                for idx, existing_element in enumerate(current_level):
                    if existing_element.element_pk == ancestor_pk:
                        existing_idx = idx
                        break

                if existing_idx is not None:
                    # Ancestor exists, get its children
                    current_level = current_level[existing_idx].child_elements
                else:
                    # Ancestor doesn't exist, add it with its score
                    ancestor_score = element_scores.get(ancestor_pk)
                    children = []
                    ancestor.score = ancestor_score
                    h_ancestor = ancestor.to_hierarchical()
                    h_ancestor.child_elements = children
                    current_level.append(h_ancestor)
                    current_level = children

        return result_tree

    def _get_element_ancestry_path(self, element_pk: int) -> List[ElementBase]:
        """Get the complete ancestry path for an element, from root to the element itself."""

        # Get the element
        element_dict = self.get_element(element_pk)
        if not element_dict:
            return []

        # Convert to ElementBase instance
        element = ElementBase(**element_dict)

        # Start building the ancestry path with the element itself
        ancestry = [element]

        # Track to avoid circular references
        visited = {element_pk}

        # Current element to process
        current_pk = element_pk

        # Traverse up the hierarchy using parent_id
        while True:
            # Get the current element
            current_element = self.get_element(current_pk)
            if not current_element:
                break

            # Get parent ID
            parent_id = current_element.get('parent_id')
            if not parent_id:
                break

            # Get the parent element
            parent_dict = self.get_element(parent_id)
            if not parent_dict:
                break

            # Check for circular references
            parent_pk = parent_dict.get('element_pk') or parent_dict.get('pk') or parent_dict.get('id')
            if parent_pk in visited:
                break

            # Convert to ElementBase
            parent = ElementBase(**parent_dict)

            # Add to visited set
            visited.add(parent_pk)

            # Add parent to the beginning of the ancestry list (root first)
            ancestry.insert(0, parent)

            # Move up to the parent
            current_pk = parent_id

        return ancestry

    # ========================================
    # CONFIGURATION AND UTILITY METHODS (EXISTING)
    # ========================================

    def get_text_storage_config(self) -> Dict[str, Any]:
        """
        Get current text storage and indexing configuration.

        Returns:
            Dictionary with current text storage settings
        """
        return {
            'store_full_text': self.store_full_text,
            'index_full_text': self.index_full_text,
            'compress_full_text': self.compress_full_text,
            'full_text_max_length': self.full_text_max_length,
            'search_capabilities': {
                'can_search_full_text': self.index_full_text,
                'can_retrieve_full_text': self.store_full_text,
                'search_fields': ['content_preview'] + (['full_text'] if self.index_full_text else [])
            }
        }

    def get_storage_size_estimate(self) -> Dict[str, str]:
        """
        Get estimated storage usage based on configuration.

        Returns:
            Dictionary with storage estimates
        """
        estimates = {
            'full_text_storage': 'High' if self.store_full_text else 'None',
            'full_text_index': 'High' if self.index_full_text else 'None',
            'overall_storage': 'High'
        }

        if not self.store_full_text and not self.index_full_text:
            estimates['overall_storage'] = 'Minimal (preview only)'
        elif not self.store_full_text:
            estimates['overall_storage'] = 'Medium (search index only)'
        elif not self.index_full_text:
            estimates['overall_storage'] = 'Medium (storage only)'

        return estimates

    # ========================================
    # HELPER METHODS (EXISTING)
    # ========================================

    @staticmethod
    def _convert_like_to_wildcard(like_pattern: str) -> str:
        """Convert SQL LIKE pattern to Elasticsearch wildcard pattern."""
        # Convert % to * (match any characters)
        pattern = like_pattern.replace('%', '*')
        # Convert _ to ? (match single character)
        pattern = pattern.replace('_', '?')
        return pattern

    @staticmethod
    def _cosine_similarity_numpy(vec1: VectorType, vec2: VectorType) -> float:
        """Calculate cosine similarity between two vectors using NumPy."""
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for this method but not available")

        # Make sure vectors are the same length
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]

        # Convert to numpy arrays
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        # Calculate dot product
        dot_product = np.dot(vec1_np, vec2_np)

        # Calculate magnitudes
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def _cosine_similarity_python(vec1: VectorType, vec2: VectorType) -> float:
        """Calculate cosine similarity between two vectors using pure Python."""
        # Make sure vectors are the same length
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Calculate magnitudes
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        # Check for zero magnitudes
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Calculate cosine similarity
        return float(dot_product / (magnitude1 * magnitude2))


if __name__ == "__main__":
    # Example demonstrating structured search with Elasticsearch
    conn_params = {
        'hosts': ['localhost:9200'],
        'index_prefix': 'doculyzer'
    }

    db = ElasticsearchDocumentDatabase(conn_params)
    db.initialize()

    # Show backend capabilities
    capabilities = db.get_backend_capabilities()
    print(f"Elasticsearch supports {len(capabilities.supported)} capabilities:")
    for cap in sorted(capabilities.get_supported_list()):
        print(f"  ✓ {cap}")

    # Example structured search
    from .structured_search import SearchQueryBuilder, LogicalOperator

    query = (SearchQueryBuilder()
             .with_operator(LogicalOperator.AND)
             .text_search("machine learning algorithms", similarity_threshold=0.8)
             .last_days(30)
             .topics(include=["ml*", "ai*"])
             .element_types(["header", "paragraph"])
             .include_dates(True)
             .include_topics_in_results(True)
             .build())

    print(f"\nExecuting structured search...")
    print(f"Query capabilities required: {len(query.get_required_capabilities())}")

    # Validate query
    missing = db.validate_query_support(query)
    if missing:
        print(f"Missing capabilities: {[m.value for m in missing]}")
    else:
        print("Query fully supported!")

        # Execute the search
        results = db.execute_structured_search(query)
        print(f"Found {len(results)} results")

        for result in results[:3]:  # Show first 3 results
            print(f"  - {result['element_id']}: {result['final_score']:.3f}")
