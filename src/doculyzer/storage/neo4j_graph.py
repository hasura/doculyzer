"""
Neo4j Implementation with Structured Search Support

This module provides a complete Neo4j implementation of the DocumentDatabase
with full structured search capabilities. It leverages Neo4j's graph database features
including Cypher queries, graph relationships, and JSON properties to provide comprehensive search.
"""

import datetime
import json
import logging
import os
from typing import Optional, Dict, Any, List, Tuple, Union, TYPE_CHECKING

import time

from .element_element import ElementHierarchical, ElementBase

# Import types for type checking only - these won't be imported at runtime
if TYPE_CHECKING:
    from neo4j import GraphDatabase, Driver, Session
    from neo4j.exceptions import ServiceUnavailable, AuthError
    import numpy as np
    from numpy.typing import NDArray

    # Define type aliases for type checking
    VectorType = NDArray[np.float32]  # NumPy array type for vectors
    Neo4jDriverType = Driver  # Neo4j driver type
    Neo4jSessionType = Session  # Neo4j session type
else:
    # Runtime type aliases - use generic Python types
    VectorType = List[float]  # Generic list of floats for vectors
    Neo4jDriverType = Any  # Generic type for Neo4j driver
    Neo4jSessionType = Any  # Generic type for Neo4j session

from .base import DocumentDatabase
from .element_relationship import ElementRelationship
from .element_element import ElementType  # Import existing enum

# Import structured search components
from .structured_search import (
    StructuredSearchQuery, SearchCriteriaGroup, BackendCapabilities, SearchCapability,
    UnsupportedSearchError, TextSearchCriteria, EmbeddingSearchCriteria, DateSearchCriteria,
    TopicSearchCriteria, MetadataSearchCriteria, ElementSearchCriteria,
    LogicalOperator, DateRangeOperator, SimilarityOperator
)

# Setup logger
logger = logging.getLogger(__name__)

# Define global flags for availability - these will be set at runtime
NEO4J_AVAILABLE = False
NUMPY_AVAILABLE = False

# Try to import Neo4j conditionally at runtime
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError

    NEO4J_AVAILABLE = True
except ImportError:
    logger.warning("Neo4j driver not available. Install with 'pip install neo4j'.")
    GraphDatabase = None
    ServiceUnavailable = Exception  # Fallback type for exception handling
    AuthError = Exception  # Fallback type for exception handling

# Try to import NumPy conditionally at runtime
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
    logger.warning(f"Error configuring Neo4j provider: {str(e)}")
    config = None


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()  # Convert date/datetime to ISO 8601 string
        return super().default(obj)


class Neo4jDocumentDatabase(DocumentDatabase):
    """Neo4j implementation of document database with structured search support."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j document database.

        Args:
            uri: Neo4j connection URI (e.g., 'bolt://localhost:7687')
            user: Neo4j username
            password: Neo4j password
            database: Neo4j database name (default is 'neo4j')
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Neo4jDriverType = None
        self.embedding_generator = None
        self.vector_dimension = None
        if config:
            self.vector_dimension = config.config.get('embedding', {}).get('dimensions', 384)
        else:
            self.vector_dimension = 384  # Default if config not available

    # ========================================
    # STRUCTURED SEARCH IMPLEMENTATION
    # ========================================

    def get_backend_capabilities(self) -> BackendCapabilities:
        """
        Neo4j supports most search capabilities through Cypher queries and graph relationships.
        """
        supported = {
            # Core search types
            SearchCapability.TEXT_SIMILARITY,
            SearchCapability.EMBEDDING_SIMILARITY,
            SearchCapability.FULL_TEXT_SEARCH,

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
            SearchCapability.FACETED_SEARCH,
            # Note: Neo4j doesn't have built-in result highlighting like PostgreSQL full-text search
        }

        return BackendCapabilities(supported)

    def execute_structured_search(self, query: StructuredSearchQuery) -> List[Dict[str, Any]]:
        """
        Execute a structured search query using Neo4j's Cypher query language.
        """
        if not self.driver:
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
            start_idx = query.offset
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
        """Execute text similarity search using embeddings."""
        try:
            # Generate embedding for the query text
            query_embedding = self._generate_embedding(criteria.query_text)

            # Perform similarity search
            similarity_results = self.search_by_embedding(
                query_embedding,
                limit=1000,  # Get many results for filtering
                filter_criteria=None
            )

            # Filter by similarity threshold and operator
            filtered_results = []
            for element_pk, similarity in similarity_results:
                if self._compare_similarity(similarity, criteria.similarity_threshold, criteria.similarity_operator):
                    filtered_results.append({
                        'element_pk': element_pk,
                        'scores': {
                            'text_similarity': similarity * criteria.boost_factor
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
        """Execute date-based filtering using Neo4j date functions."""
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
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=criteria.relative_value)
                element_pks = self._get_element_pks_in_date_range(start_date, end_date)

            elif criteria.operator == DateRangeOperator.RELATIVE_MONTHS:
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=criteria.relative_value * 30)  # Approximate
                element_pks = self._get_element_pks_in_date_range(start_date, end_date)

            elif criteria.operator == DateRangeOperator.FISCAL_YEAR:
                # Assume fiscal year starts in July (customize as needed)
                start_date = datetime.datetime(criteria.year - 1, 7, 1)
                end_date = datetime.datetime(criteria.year, 6, 30, 23, 59, 59)
                element_pks = self._get_element_pks_in_date_range(start_date, end_date)

            elif criteria.operator == DateRangeOperator.CALENDAR_YEAR:
                start_date = datetime.datetime(criteria.year, 1, 1)
                end_date = datetime.datetime(criteria.year, 12, 31, 23, 59, 59)
                element_pks = self._get_element_pks_in_date_range(start_date, end_date)

            elif criteria.operator == DateRangeOperator.QUARTER:
                quarter_starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
                quarter_ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}

                start_month, start_day = quarter_starts[criteria.quarter]
                end_month, end_day = quarter_ends[criteria.quarter]

                start_date = datetime.datetime(criteria.year, start_month, start_day)
                end_date = datetime.datetime(criteria.year, end_month, end_day, 23, 59, 59)
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
        """Execute topic-based filtering using Neo4j JSON operators."""
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
        """Execute metadata-based filtering using Neo4j JSON operators."""
        try:
            with self.driver.session(database=self.database) as session:
                # Build Cypher query for metadata filtering
                cypher_query = "MATCH (e:Element) WHERE 1=1"
                params = {}

                # Add exact matches
                param_counter = 0
                for key, value in criteria.exact_matches.items():
                    param_name = f"exact_value_{param_counter}"
                    cypher_query += f" AND e.metadata CONTAINS '{key}\":\"{value}\"'"
                    param_counter += 1

                # Add LIKE patterns
                for key, pattern in criteria.like_patterns.items():
                    # Neo4j doesn't have direct JSON LIKE, so we use CONTAINS for simple patterns
                    if pattern.startswith('%') and pattern.endswith('%'):
                        # %text% -> contains
                        search_value = pattern[1:-1]
                        cypher_query += f" AND e.metadata CONTAINS '{key}' AND e.metadata CONTAINS '{search_value}'"
                    elif pattern.endswith('%'):
                        # text% -> starts with (approximate)
                        search_value = pattern[:-1]
                        cypher_query += f" AND e.metadata CONTAINS '{key}\":\"{search_value}'"
                    elif pattern.startswith('%'):
                        # %text -> ends with (approximate)
                        search_value = pattern[1:]
                        cypher_query += f" AND e.metadata CONTAINS '{search_value}\"'"

                # Add range filters (requires parsing JSON in Neo4j)
                for key, range_filter in criteria.range_filters.items():
                    # This is a simplified approach - in production you might want to use APOC procedures
                    if 'gte' in range_filter:
                        cypher_query += f" AND apoc.convert.getJsonProperty(e.metadata, '{key}') >= {range_filter['gte']}"
                    if 'lte' in range_filter:
                        cypher_query += f" AND apoc.convert.getJsonProperty(e.metadata, '{key}') <= {range_filter['lte']}"
                    if 'gt' in range_filter:
                        cypher_query += f" AND apoc.convert.getJsonProperty(e.metadata, '{key}') > {range_filter['gt']}"
                    if 'lt' in range_filter:
                        cypher_query += f" AND apoc.convert.getJsonProperty(e.metadata, '{key}') < {range_filter['lt']}"

                # Add exists filters
                for key in criteria.exists_filters:
                    cypher_query += f" AND e.metadata CONTAINS '{key}'"

                cypher_query += " RETURN id(e) AS element_pk LIMIT 1000"

                # Execute query
                result = session.run(cypher_query, params)
                element_pks = [record["element_pk"] for record in result]

            results = []
            for element_pk in element_pks:
                results.append({
                    'element_pk': element_pk,
                    'scores': {
                        'metadata_relevance': 1.0
                    }
                })

            return results

        except Exception as e:
            logger.error(f"Error executing metadata criteria: {str(e)}")
            return []

    def _execute_element_criteria(self, criteria: ElementSearchCriteria) -> List[Dict[str, Any]]:
        """Execute element-based filtering using Neo4j."""
        try:
            with self.driver.session(database=self.database) as session:
                # Build Cypher query for element filtering
                cypher_query = "MATCH (e:Element) WHERE 1=1"
                params = {}

                # Add element type filter
                if criteria.element_types:
                    type_values = self._prepare_element_type_query(criteria.element_types)
                    if type_values:
                        if len(type_values) == 1:
                            cypher_query += " AND e.element_type = $element_type"
                            params["element_type"] = type_values[0]
                        else:
                            cypher_query += " AND e.element_type IN $element_types"
                            params["element_types"] = type_values

                # Add document ID filters
                if criteria.doc_ids:
                    cypher_query += " AND e.doc_id IN $doc_ids"
                    params["doc_ids"] = criteria.doc_ids

                if criteria.exclude_doc_ids:
                    cypher_query += " AND NOT e.doc_id IN $exclude_doc_ids"
                    params["exclude_doc_ids"] = criteria.exclude_doc_ids

                # Add content length filters
                if criteria.content_length_min is not None:
                    cypher_query += " AND size(e.content_preview) >= $content_length_min"
                    params["content_length_min"] = criteria.content_length_min

                if criteria.content_length_max is not None:
                    cypher_query += " AND size(e.content_preview) <= $content_length_max"
                    params["content_length_max"] = criteria.content_length_max

                # Add parent element filters
                if criteria.parent_element_ids:
                    cypher_query += " AND e.parent_id IN $parent_element_ids"
                    params["parent_element_ids"] = criteria.parent_element_ids

                cypher_query += " RETURN id(e) AS element_pk LIMIT 1000"

                # Execute query
                result = session.run(cypher_query, params)
                element_pks = [record["element_pk"] for record in result]

            results = []
            for element_pk in element_pks:
                results.append({
                    'element_pk': element_pk,
                    'scores': {
                        'element_match': 1.0
                    }
                })

            return results

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
    def _calculate_combined_score(scores: Dict[str, List[float]],
                                  combination_method: str,
                                  weights: Dict[str, float]) -> float:
        """Calculate final combined score from multiple score types."""

        if not scores:
            return 0.0

        # Average scores of the same type
        avg_scores = {}
        for score_type, score_list in scores.items():
            if score_list:
                avg_scores[score_type] = sum(score_list) / len(score_list)

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
            from ..embeddings import get_embedding_generator

            if self.embedding_generator is None:
                if not config:
                    logger.error("Config not available for embedding generator")
                    raise ValueError("Config not available")
                self.embedding_generator = get_embedding_generator(config)

            return self.embedding_generator.generate(search_text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def _get_element_pks_in_date_range(self, start_date: Optional[datetime.datetime],
                                       end_date: Optional[datetime.datetime]) -> List[int]:
        """Get element_pks that have dates within the specified range using Neo4j."""
        if not (start_date or end_date):
            return []

        with self.driver.session(database=self.database) as session:
            # Build Cypher query for date range filtering
            cypher_query = """
                MATCH (e:Element)
                WHERE EXISTS {
                    MATCH (e)-[:HAS_DATE]->(d:ExtractedDate)
                    WHERE 1=1
            """
            params = {}

            if start_date:
                cypher_query += " AND d.timestamp_value >= $start_timestamp"
                params["start_timestamp"] = start_date.timestamp()

            if end_date:
                cypher_query += " AND d.timestamp_value <= $end_timestamp"
                params["end_timestamp"] = end_date.timestamp()

            cypher_query += """
                }
                RETURN DISTINCT id(e) AS element_pk
            """

            result = session.run(cypher_query, params)
            return [record["element_pk"] for record in result]

    def _filter_by_specificity(self, element_pks: List[int],
                               allowed_levels: List[str]) -> List[int]:
        """Filter element PKs by date specificity levels."""
        if not element_pks or not allowed_levels:
            return element_pks

        with self.driver.session(database=self.database) as session:
            # Query to get element PKs that have dates with allowed specificity levels
            cypher_query = """
                MATCH (e:Element)-[:HAS_DATE]->(d:ExtractedDate)
                WHERE id(e) IN $element_pks
                AND d.specificity_level IN $allowed_levels
                RETURN DISTINCT id(e) AS element_pk
            """

            result = session.run(cypher_query, {
                "element_pks": element_pks,
                "allowed_levels": allowed_levels
            })

            return [record["element_pk"] for record in result]

    # ========================================
    # ALL EXISTING METHODS (unchanged from previous implementation)
    # ========================================

    def initialize(self) -> None:
        """Initialize the database by creating constraints and indexes."""
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not installed. Please install with: pip install neo4j")

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )

            # Test the connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")

            # Create constraints and indexes
            self._create_constraints_and_indexes()
            logger.info(f"Successfully connected to Neo4j at {self.uri}")

        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            self.driver = None

    def _create_constraints_and_indexes(self) -> None:
        """Create necessary constraints and indexes for optimal performance."""
        with self.driver.session(database=self.database) as session:
            # Create constraints for unique IDs
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS ON (d:Document) ASSERT d.doc_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS ON (e:Element) ASSERT e.element_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS ON (d:Document) ASSERT d.source IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS ON (h:ProcessingHistory) ASSERT h.source_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS ON (emb:Embedding) ASSERT emb.element_pk IS UNIQUE"
            ]

            # Create indexes for faster lookups
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (e:Element) ON (e.doc_id)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Element) ON (e.element_type)",
                "CREATE INDEX IF NOT EXISTS FOR (r:RELATES_TO) ON (r.relationship_type)",
                "CREATE INDEX IF NOT EXISTS FOR (emb:Embedding) ON (emb.confidence)",
                "CREATE INDEX IF NOT EXISTS FOR (emb:Embedding) ON (emb.created_at)",
                # Enhanced search indexes
                "CREATE INDEX IF NOT EXISTS FOR (e:Element) ON (e.content_preview)",
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.source)",
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.doc_type)"
            ]

            # Execute all constraints and indexes
            for query in constraints + indexes:
                try:
                    session.run(query)
                except Exception as e:
                    logger.warning(f"Error creating constraint or index: {str(e)}")

    def get_last_processed_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about when a document was last processed."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (h:ProcessingHistory {source_id: $source_id})
                RETURN h.source_id AS source_id, 
                       h.content_hash AS content_hash,
                       h.last_modified AS last_modified,
                       h.processing_count AS processing_count
                """,
                source_id=source_id
            )

            record = result.single()
            if not record:
                return None

            return {
                "source_id": record["source_id"],
                "content_hash": record["content_hash"],
                "last_modified": record["last_modified"],
                "processing_count": record["processing_count"]
            }

    def update_processing_history(self, source_id: str, content_hash: str) -> None:
        """Update the processing history for a document."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # Check if record exists and get processing count
            result = session.run(
                """
                MATCH (h:ProcessingHistory {source_id: $source_id})
                RETURN h.processing_count AS processing_count
                """,
                source_id=source_id
            )

            record = result.single()
            processing_count = 1  # Default for new records

            if record:
                processing_count = record["processing_count"] + 1

                # Update existing record
                session.run(
                    """
                    MATCH (h:ProcessingHistory {source_id: $source_id})
                    SET h.content_hash = $content_hash,
                        h.last_modified = $timestamp,
                        h.processing_count = $processing_count
                    """,
                    source_id=source_id,
                    content_hash=content_hash,
                    timestamp=time.time(),
                    processing_count=processing_count
                )
            else:
                # Create new record
                session.run(
                    """
                    CREATE (h:ProcessingHistory {
                        source_id: $source_id,
                        content_hash: $content_hash,
                        last_modified: $timestamp,
                        processing_count: $processing_count
                    })
                    """,
                    source_id=source_id,
                    content_hash=content_hash,
                    timestamp=time.time(),
                    processing_count=processing_count
                )

            logger.debug(f"Updated processing history for {source_id}")

    def store_document(self, document: Dict[str, Any], elements: List[Dict[str, Any]],
                       relationships: List[Dict[str, Any]]) -> None:
        """
        Store a document with its elements and relationships.
        If a document with the same source already exists, update it instead.

        Args:
            document: Document metadata
            elements: Document elements
            relationships: Element relationships
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        source = document.get("source", "")
        content_hash = document.get("content_hash", "")

        with self.driver.session(database=self.database) as session:
            # Check if document already exists
            result = session.run(
                """
                MATCH (d:Document {source: $source})
                RETURN d.doc_id AS doc_id
                """,
                source=source
            )

            record = result.single()
            if record:
                # Document exists, update it
                doc_id = record["doc_id"]
                document["doc_id"] = doc_id  # Use existing doc_id

                # Update all elements to use the existing doc_id
                for element in elements:
                    element["doc_id"] = doc_id

                self.update_document(doc_id, document, elements, relationships)
                return

            # New document, proceed with creation
            doc_id = document["doc_id"]

            # Store document
            metadata_json = json.dumps(document.get("metadata", {}), cls=DateTimeEncoder)

            session.run(
                """
                CREATE (d:Document {
                    doc_id: $doc_id,
                    doc_type: $doc_type,
                    source: $source,
                    content_hash: $content_hash,
                    metadata: $metadata,
                    created_at: $created_at,
                    updated_at: $updated_at
                })
                """,
                doc_id=doc_id,
                doc_type=document.get("doc_type", ""),
                source=source,
                content_hash=content_hash,
                metadata=metadata_json,
                created_at=document.get("created_at", time.time()),
                updated_at=document.get("updated_at", time.time())
            )

            # Store elements and create relationships to document
            element_pk_map = {}  # Maps element_id to Neo4j node id

            for element in elements:
                element_id = element["element_id"]
                metadata_json = json.dumps(element.get("metadata", {}))
                content_preview = element.get("content_preview", "")

                if len(content_preview) > 100:
                    content_preview = content_preview[:100] + "..."

                # Create the element node and link to document
                result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    CREATE (e:Element {
                        element_id: $element_id,
                        doc_id: $doc_id,
                        element_type: $element_type,
                        parent_id: $parent_id,
                        content_preview: $content_preview,
                        content_location: $content_location,
                        content_hash: $content_hash,
                        metadata: $metadata
                    })
                    CREATE (e)-[:BELONGS_TO]->(d)
                    RETURN id(e) AS node_id
                    """,
                    doc_id=element.get("doc_id", ""),
                    element_id=element_id,
                    element_type=element.get("element_type", ""),
                    parent_id=element.get("parent_id", ""),
                    content_preview=content_preview,
                    content_location=element.get("content_location", ""),
                    content_hash=element.get("content_hash", ""),
                    metadata=metadata_json
                )

                record = result.single()
                if record:
                    element_pk_map[element_id] = record["node_id"]
                    # Store the element_pk back in the element
                    element["element_pk"] = record["node_id"]

            # Create parent-child relationships between elements
            for element in elements:
                if element.get("parent_id"):
                    session.run(
                        """
                        MATCH (child:Element {element_id: $element_id})
                        MATCH (parent:Element {element_id: $parent_id})
                        CREATE (child)-[:CHILD_OF]->(parent)
                        """,
                        element_id=element["element_id"],
                        parent_id=element["parent_id"]
                    )

            # Store custom relationships
            for relationship in relationships:
                relationship_id = relationship["relationship_id"]
                metadata_json = json.dumps(relationship.get("metadata", {}))
                source_id = relationship.get("source_id", "")
                target_reference = relationship.get("target_reference", "")
                relationship_type = relationship.get("relationship_type", "")

                # Create the relationship between elements
                session.run(
                    """
                    MATCH (source:Element {element_id: $source_id})
                    MATCH (target:Element {element_id: $target_reference})
                    CREATE (source)-[r:RELATES_TO {
                        relationship_id: $relationship_id,
                        relationship_type: $relationship_type,
                        metadata: $metadata
                    }]->(target)
                    """,
                    source_id=source_id,
                    target_reference=target_reference,
                    relationship_id=relationship_id,
                    relationship_type=relationship_type,
                    metadata=metadata_json
                )

            # Update processing history
            if source:
                self.update_processing_history(source, content_hash)

    def update_document(self, doc_id: str, document: Dict[str, Any],
                        elements: List[Dict[str, Any]],
                        relationships: List[Dict[str, Any]]) -> None:
        """
        Update an existing document by removing it and then reinserting.
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # Check if document exists
            result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                RETURN d.doc_id AS doc_id
                """,
                doc_id=doc_id
            )

            if not result.single():
                raise ValueError(f"Document not found: {doc_id}")

            # Delete the document, which will cascade to elements and relationships
            self.delete_document(doc_id)

            # Use store_document to insert everything
            self.store_document(document, elements, relationships)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (d:Document)
                WHERE d.doc_id = $doc_id OR d.source = $doc_id
                RETURN d
                """,
                doc_id=doc_id
            )

            record = result.single()
            if not record:
                return None

            document = dict(record["d"])

            # Convert metadata from JSON
            try:
                document["metadata"] = json.loads(document["metadata"])
            except (json.JSONDecodeError, TypeError):
                document["metadata"] = {}

            return document

    def get_document_elements(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get elements for a document."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (d:Document)
                WHERE d.doc_id = $doc_id OR d.source = $doc_id
                MATCH (e:Element)-[:BELONGS_TO]->(d)
                RETURN e, id(e) AS element_pk
                """,
                doc_id=doc_id
            )

            elements = []
            for record in result:
                element = dict(record["e"])
                element["element_pk"] = record["element_pk"]

                # Convert metadata from JSON
                try:
                    element["metadata"] = json.loads(element["metadata"])
                except (json.JSONDecodeError, TypeError):
                    element["metadata"] = {}

                elements.append(element)

            return elements

    def get_document_relationships(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get relationships for a document."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                MATCH (e:Element)-[:BELONGS_TO]->(d)
                MATCH (e)-[r:RELATES_TO]->(target:Element)
                RETURN r.relationship_id AS relationship_id,
                       e.element_id AS source_id,
                       r.relationship_type AS relationship_type,
                       target.element_id AS target_reference,
                       r.metadata AS metadata
                """,
                doc_id=doc_id
            )

            relationships = []
            for record in result:
                relationship = {
                    "relationship_id": record["relationship_id"],
                    "source_id": record["source_id"],
                    "relationship_type": record["relationship_type"],
                    "target_reference": record["target_reference"],
                }

                # Convert metadata from JSON
                try:
                    relationship["metadata"] = json.loads(record["metadata"])
                except (json.JSONDecodeError, TypeError):
                    relationship["metadata"] = {}

                relationships.append(relationship)

            return relationships

    def get_element(self, element_pk: Union[int, str]) -> Optional[Dict[str, Any]]:
        """Get element by ID."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # If element_pk is numeric, treat as a Neo4j node ID
            if str(element_pk).isdigit():
                result = session.run(
                    """
                    MATCH (e:Element)
                    WHERE id(e) = $element_pk
                    RETURN e, id(e) AS element_pk
                    """,
                    element_pk=int(element_pk)
                )
            else:
                # Treat as element_id string
                result = session.run(
                    """
                    MATCH (e:Element {element_id: $element_id})
                    RETURN e, id(e) AS element_pk
                    """,
                    element_id=str(element_pk)
                )

            record = result.single()
            if not record:
                return None

            element = dict(record["e"])
            element["element_pk"] = record["element_pk"]

            # Convert metadata from JSON
            try:
                element["metadata"] = json.loads(element["metadata"])
            except (json.JSONDecodeError, TypeError):
                element["metadata"] = {}

            return element

    def get_outgoing_relationships(self, element_pk: Union[int, str]) -> List[ElementRelationship]:
        """
        Find all relationships where the specified element_pk is the source.

        Args:
            element_pk: The primary key of the element or element_id string

        Returns:
            List of ElementRelationship objects where the specified element is the source
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        relationships = []

        with self.driver.session(database=self.database) as session:
            # Get the element to find its element_id and type
            if str(element_pk).isdigit():
                element_result = session.run(
                    """
                    MATCH (e:Element)
                    WHERE id(e) = $element_pk
                    RETURN e.element_id AS element_id, e.element_type AS element_type
                    """,
                    element_pk=int(element_pk)
                )
            else:
                element_result = session.run(
                    """
                    MATCH (e:Element {element_id: $element_id})
                    RETURN e.element_id AS element_id, e.element_type AS element_type, id(e) AS element_pk
                    """,
                    element_id=str(element_pk)
                )

            element_record = element_result.single()
            if not element_record:
                logger.warning(f"Element with PK {element_pk} not found")
                return []

            element_id = element_record["element_id"]
            element_type = element_record["element_type"]
            element_pk = element_record.get("element_pk", element_pk)

            # Find relationships with target element information
            result = session.run(
                """
                MATCH (source:Element {element_id: $element_id})-[r:RELATES_TO]->(target:Element)
                RETURN r.relationship_id AS relationship_id,
                       r.relationship_type AS relationship_type,
                       r.metadata AS metadata,
                       source.element_id AS source_id,
                       target.element_id AS target_reference,
                       target.element_type AS target_element_type,
                       target.content_preview AS target_content_preview,
                       id(target) AS target_element_pk,
                       source.doc_id AS doc_id
                """,
                element_id=element_id
            )

            for record in result:
                # Convert metadata from JSON
                try:
                    metadata = json.loads(record["metadata"]) if record["metadata"] else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                # Create relationship object
                relationship = ElementRelationship(
                    relationship_id=record["relationship_id"],
                    source_id=element_id,
                    source_element_pk=element_pk,
                    source_element_type=element_type,
                    relationship_type=record["relationship_type"],
                    target_reference=record["target_reference"],
                    target_element_pk=record["target_element_pk"],
                    target_element_type=record["target_element_type"],
                    target_content_preview=record["target_content_preview"],
                    doc_id=record["doc_id"],
                    metadata=metadata,
                    is_source=True
                )

                relationships.append(relationship)

        return relationships

    def find_documents(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find documents matching query with support for pattern matching.

        Args:
            limit: Maximum number of results
            query: Query parameters. Use '_like' suffix for pattern matching.
                   Examples:
                   - {"doc_type": "pdf"} - exact match
                   - {"source_like": "%reports%"} - pattern match (uses CONTAINS)
                   - {"source_starts": "annual"} - starts with pattern
                   - {"source_ends": ".pdf"} - ends with pattern
                   - {"metadata": {"author": "John"}} - metadata exact match
                   - {"metadata_like": {"title": "%annual%"}} - metadata pattern match

        Returns:
            List of matching documents
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # Start with base query
            cypher_query = "MATCH (d:Document)"
            params = {}
            conditions = []

            # Apply filters if provided
            if query:
                for key, value in query.items():
                    if key == "metadata":
                        # Metadata filters require special handling
                        for meta_key, meta_value in value.items():
                            # For simplicity, we'll check if the JSON contains this key/value
                            conditions.append(f"d.metadata CONTAINS ${meta_key}_value")
                            params[f"{meta_key}_value"] = f'"{meta_key}":"{meta_value}"'
                    elif key == "metadata_like":
                        # Metadata pattern filters
                        for meta_key, meta_value in value.items():
                            pattern = self._convert_like_to_neo4j_pattern(meta_value)
                            if pattern["type"] == "contains":
                                conditions.append(f"d.metadata CONTAINS ${meta_key}_pattern")
                                params[f"{meta_key}_pattern"] = pattern["value"]
                            elif pattern["type"] == "starts":
                                conditions.append(f"d.metadata CONTAINS ${meta_key}_start")
                                params[f"{meta_key}_start"] = f'"{meta_key}":"{pattern["value"]}'
                            elif pattern["type"] == "ends":
                                conditions.append(f"d.metadata CONTAINS ${meta_key}_end")
                                params[f"{meta_key}_end"] = pattern["value"]
                    elif key.endswith("_like"):
                        # Pattern matching for regular fields
                        field_name = key[:-5]  # Remove '_like' suffix
                        pattern = self._convert_like_to_neo4j_pattern(value)
                        if pattern["type"] == "contains":
                            conditions.append(f"d.{field_name} CONTAINS ${field_name}_contains")
                            params[f"{field_name}_contains"] = pattern["value"]
                        elif pattern["type"] == "starts":
                            conditions.append(f"d.{field_name} STARTS WITH ${field_name}_starts")
                            params[f"{field_name}_starts"] = pattern["value"]
                        elif pattern["type"] == "ends":
                            conditions.append(f"d.{field_name} ENDS WITH ${field_name}_ends")
                            params[f"{field_name}_ends"] = pattern["value"]
                        elif pattern["type"] == "regex":
                            conditions.append(f"d.{field_name} =~ ${field_name}_regex")
                            params[f"{field_name}_regex"] = pattern["value"]
                    elif key.endswith("_starts"):
                        # Starts with pattern
                        field_name = key[:-7]  # Remove '_starts' suffix
                        conditions.append(f"d.{field_name} STARTS WITH ${field_name}_starts")
                        params[f"{field_name}_starts"] = value
                    elif key.endswith("_ends"):
                        # Ends with pattern
                        field_name = key[:-5]  # Remove '_ends' suffix
                        conditions.append(f"d.{field_name} ENDS WITH ${field_name}_ends")
                        params[f"{field_name}_ends"] = value
                    elif key.endswith("_contains"):
                        # Contains pattern
                        field_name = key[:-9]  # Remove '_contains' suffix
                        conditions.append(f"d.{field_name} CONTAINS ${field_name}_contains")
                        params[f"{field_name}_contains"] = value
                    elif isinstance(value, list):
                        # Handle list fields with IN clause
                        conditions.append(f"d.{key} IN ${key}")
                        params[key] = value
                    else:
                        # Exact match for regular fields
                        conditions.append(f"d.{key} = ${key}")
                        params[key] = value

            if conditions:
                cypher_query += " WHERE " + " AND ".join(conditions)

            # Add return and limit
            cypher_query += f" RETURN d LIMIT {limit}"

            # Execute query
            result = session.run(cypher_query, params)

            documents = []
            for record in result:
                doc = dict(record["d"])

                # Convert metadata from JSON
                try:
                    doc["metadata"] = json.loads(doc["metadata"])
                except (json.JSONDecodeError, TypeError):
                    doc["metadata"] = {}

                documents.append(doc)

            return documents

    def find_elements(self, query: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements matching query with support for pattern matching and ElementType enums.
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # Start with base query
            cypher_query = "MATCH (e:Element)"
            params = {}
            conditions = []

            # Apply filters if provided
            if query:
                for key, value in query.items():
                    if key == "metadata":
                        # Metadata filters require special handling
                        for meta_key, meta_value in value.items():
                            conditions.append(f"e.metadata CONTAINS ${meta_key}_value")
                            params[f"{meta_key}_value"] = f'"{meta_key}":"{meta_value}"'
                    elif key == "metadata_like":
                        # Metadata pattern filters
                        for meta_key, meta_value in value.items():
                            pattern = self._convert_like_to_neo4j_pattern(meta_value)
                            if pattern["type"] == "contains":
                                conditions.append(f"e.metadata CONTAINS ${meta_key}_pattern")
                                params[f"{meta_key}_pattern"] = pattern["value"]
                            elif pattern["type"] == "starts":
                                conditions.append(f"e.metadata CONTAINS ${meta_key}_start")
                                params[f"{meta_key}_start"] = f'"{meta_key}":"{pattern["value"]}'
                            elif pattern["type"] == "ends":
                                conditions.append(f"e.metadata CONTAINS ${meta_key}_end")
                                params[f"{meta_key}_end"] = pattern["value"]
                    elif key.endswith("_like"):
                        # Pattern matching for regular fields
                        field_name = key[:-5]  # Remove '_like' suffix
                        pattern = self._convert_like_to_neo4j_pattern(value)
                        if pattern["type"] == "contains":
                            conditions.append(f"e.{field_name} CONTAINS ${field_name}_contains")
                            params[f"{field_name}_contains"] = pattern["value"]
                        elif pattern["type"] == "starts":
                            conditions.append(f"e.{field_name} STARTS WITH ${field_name}_starts")
                            params[f"{field_name}_starts"] = pattern["value"]
                        elif pattern["type"] == "ends":
                            conditions.append(f"e.{field_name} ENDS WITH ${field_name}_ends")
                            params[f"{field_name}_ends"] = pattern["value"]
                        elif pattern["type"] == "regex":
                            conditions.append(f"e.{field_name} =~ ${field_name}_regex")
                            params[f"{field_name}_regex"] = pattern["value"]
                    elif key.endswith("_starts"):
                        # Starts with pattern
                        field_name = key[:-7]  # Remove '_starts' suffix
                        conditions.append(f"e.{field_name} STARTS WITH ${field_name}_starts")
                        params[f"{field_name}_starts"] = value
                    elif key.endswith("_ends"):
                        # Ends with pattern
                        field_name = key[:-5]  # Remove '_ends' suffix
                        conditions.append(f"e.{field_name} ENDS WITH ${field_name}_ends")
                        params[f"{field_name}_ends"] = value
                    elif key.endswith("_contains"):
                        # Contains pattern
                        field_name = key[:-9]  # Remove '_contains' suffix
                        conditions.append(f"e.{field_name} CONTAINS ${field_name}_contains")
                        params[f"{field_name}_contains"] = value
                    elif key == "element_type":
                        # Handle ElementType enums, strings, and lists
                        type_values = self._prepare_element_type_query(value)
                        if type_values:
                            if len(type_values) == 1:
                                conditions.append("e.element_type = $element_type")
                                params["element_type"] = type_values[0]
                            else:
                                conditions.append("e.element_type IN $element_types")
                                params["element_types"] = type_values
                    elif isinstance(value, list):
                        # Handle other list fields with IN clause
                        conditions.append(f"e.{key} IN ${key}")
                        params[key] = value
                    else:
                        # Exact match for regular fields
                        conditions.append(f"e.{key} = ${key}")
                        params[key] = value

            if conditions:
                cypher_query += " WHERE " + " AND ".join(conditions)

            # Add return and limit
            cypher_query += f" RETURN e, id(e) AS element_pk LIMIT {limit}"

            # Execute query
            result = session.run(cypher_query, params)

            elements = []
            for record in result:
                element = dict(record["e"])
                element["element_pk"] = record["element_pk"]

                # Convert metadata from JSON
                try:
                    element["metadata"] = json.loads(element["metadata"])
                except (json.JSONDecodeError, TypeError):
                    element["metadata"] = {}

                elements.append(element)

            return elements

    def search_elements_by_content(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search elements by content preview."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (e:Element)
                WHERE e.content_preview CONTAINS $search_text
                RETURN e, id(e) AS element_pk
                LIMIT $limit
                """,
                search_text=search_text,
                limit=limit
            )

            elements = []
            for record in result:
                element = dict(record["e"])
                element["element_pk"] = record["element_pk"]

                # Convert metadata from JSON
                try:
                    element["metadata"] = json.loads(element["metadata"])
                except (json.JSONDecodeError, TypeError):
                    element["metadata"] = {}

                elements.append(element)

            return elements

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all associated elements and relationships."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # Check if document exists
            result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                RETURN d.doc_id AS doc_id
                """,
                doc_id=doc_id
            )

            if not result.single():
                return False

            # Delete the document and all its elements and relationships
            session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                OPTIONAL MATCH (e:Element)-[:BELONGS_TO]->(d)
                OPTIONAL MATCH (e)-[r:RELATES_TO]->()
                OPTIONAL MATCH ()-[r2:RELATES_TO]->(e)
                OPTIONAL MATCH (e)-[r3:CHILD_OF]->()
                OPTIONAL MATCH ()-[r4:CHILD_OF]->(e)
                OPTIONAL MATCH (e)-[r5:BELONGS_TO]->()
                OPTIONAL MATCH (emb:Embedding)-[r6:EMBEDDING_OF]->(e)
                DELETE r, r2, r3, r4, r5, r6, emb, e, d
                """,
                doc_id=doc_id
            )

            return True

    def store_relationship(self, relationship: Dict[str, Any]) -> None:
        """
        Store a relationship between elements.

        Args:
            relationship: Relationship data
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # Convert metadata to JSON
            metadata_json = json.dumps(relationship.get("metadata", {}))

            # Create the relationship
            session.run(
                """
                MATCH (source:Element {element_id: $source_id})
                MATCH (target:Element {element_id: $target_reference})
                MERGE (source)-[r:RELATES_TO {relationship_id: $relationship_id}]->(target)
                SET r.relationship_type = $relationship_type,
                    r.metadata = $metadata
                """,
                relationship_id=relationship["relationship_id"],
                source_id=relationship.get("source_id", ""),
                target_reference=relationship.get("target_reference", ""),
                relationship_type=relationship.get("relationship_type", ""),
                metadata=metadata_json
            )

    def delete_relationships_for_element(self, element_id: str, relationship_type: str = None) -> None:
        """
        Delete relationships for an element.

        Args:
            element_id: Element ID
            relationship_type: Optional relationship type to filter by
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # Delete relationships where element is the source
            if relationship_type:
                session.run(
                    """
                    MATCH (source:Element {element_id: $element_id})-[r:RELATES_TO]->(target)
                    WHERE r.relationship_type = $relationship_type
                    DELETE r
                    """,
                    element_id=element_id,
                    relationship_type=relationship_type
                )

                # Delete relationships where element is the target
                session.run(
                    """
                    MATCH (source)-[r:RELATES_TO]->(target:Element {element_id: $element_id})
                    WHERE r.relationship_type = $relationship_type
                    DELETE r
                    """,
                    element_id=element_id,
                    relationship_type=relationship_type
                )
            else:
                # Delete all relationships regardless of type
                session.run(
                    """
                    MATCH (source:Element {element_id: $element_id})-[r:RELATES_TO]->()
                    DELETE r
                    """,
                    element_id=element_id
                )

                session.run(
                    """
                    MATCH ()-[r:RELATES_TO]->(target:Element {element_id: $element_id})
                    DELETE r
                    """,
                    element_id=element_id
                )

    # ========================================
    # ENHANCED SEARCH HELPER METHODS
    # ========================================

    @staticmethod
    def _prepare_element_type_query(element_types: Union[
        ElementType,
        List[ElementType],
        str,
        List[str],
        None
    ]) -> Optional[List[str]]:
        """
        Prepare element type values for database queries using existing ElementType enum.

        Args:
            element_types: ElementType enum(s), string(s), or None

        Returns:
            List of string values for database query, or None
        """
        if element_types is None:
            return None

        if isinstance(element_types, ElementType):
            return [element_types.value]
        elif isinstance(element_types, str):
            return [element_types]
        elif isinstance(element_types, list):
            result = []
            for et in element_types:
                if isinstance(et, ElementType):
                    result.append(et.value)
                elif isinstance(et, str):
                    result.append(et)
            return result if result else None

        return None

    @staticmethod
    def _convert_like_to_neo4j_pattern(like_pattern: str) -> Dict[str, str]:
        """
        Convert SQL LIKE pattern to Neo4j pattern matching.

        Args:
            like_pattern: SQL LIKE pattern (e.g., "%text%", "text%", "%text")

        Returns:
            Dictionary with pattern type and value
        """
        if like_pattern.startswith('%') and like_pattern.endswith('%'):
            # %text% -> CONTAINS
            return {"type": "contains", "value": like_pattern[1:-1]}
        elif like_pattern.endswith('%'):
            # text% -> STARTS WITH
            return {"type": "starts", "value": like_pattern[:-1]}
        elif like_pattern.startswith('%'):
            # %text -> ENDS WITH
            return {"type": "ends", "value": like_pattern[1:]}
        else:
            # No wildcards - treat as exact match or convert to regex for more complex patterns
            if '_' in like_pattern:
                # Convert _ to . for regex
                regex_pattern = like_pattern.replace('_', '.')
                return {"type": "regex", "value": f".*{regex_pattern}.*"}
            else:
                return {"type": "contains", "value": like_pattern}

    def get_element_types_by_category(self):
        """
        Get categorized lists of ElementType enums from your existing enum.

        Returns:
            Dictionary with categorized element types
        """
        return {
            "text_elements": [
                ElementType.HEADER,
                ElementType.PARAGRAPH,
                ElementType.BLOCKQUOTE,
                ElementType.TEXT_BOX
            ],

            "structural_elements": [
                ElementType.ROOT,
                ElementType.PAGE,
                ElementType.BODY,
                ElementType.PAGE_HEADER,
                ElementType.PAGE_FOOTER
            ],

            "list_elements": [
                ElementType.LIST,
                ElementType.LIST_ITEM
            ],

            "table_elements": [
                ElementType.TABLE,
                ElementType.TABLE_ROW,
                ElementType.TABLE_HEADER_ROW,
                ElementType.TABLE_CELL,
                ElementType.TABLE_HEADER
            ],

            "media_elements": [
                ElementType.IMAGE,
                ElementType.CHART,
                ElementType.SHAPE,
                ElementType.SHAPE_GROUP
            ],

            "code_elements": [
                ElementType.CODE_BLOCK
            ],

            "presentation_elements": [
                ElementType.SLIDE,
                ElementType.SLIDE_NOTES,
                ElementType.PRESENTATION_BODY,
                ElementType.SLIDE_MASTERS,
                ElementType.SLIDE_TEMPLATES,
                ElementType.SLIDE_LAYOUT,
                ElementType.SLIDE_MASTER
            ],

            "data_elements": [
                ElementType.JSON_OBJECT,
                ElementType.JSON_ARRAY,
                ElementType.JSON_FIELD,
                ElementType.JSON_ITEM
            ],

            "xml_elements": [
                ElementType.XML_ELEMENT,
                ElementType.XML_TEXT,
                ElementType.XML_LIST,
                ElementType.XML_OBJECT
            ]
        }

    def find_elements_by_category(self, category: str, **other_filters) -> List[Dict[str, Any]]:
        """
        Find elements by predefined category using your existing ElementType enum.

        Args:
            category: Category name from get_element_types_by_category()
            **other_filters: Additional filter criteria

        Returns:
            List of matching elements

        Examples:
            find_elements_by_category("text_elements")
            find_elements_by_category("table_elements", content_preview_like="%data%")
        """
        categories = self.get_element_types_by_category()

        if category not in categories:
            raise ValueError(f"Unknown category: {category}. Available: {list(categories.keys())}")

        element_types = categories[category]
        query = {"element_type": element_types}
        query.update(other_filters)

        return self.find_elements(query)

    def supports_like_patterns(self) -> bool:
        """Neo4j supports pattern matching through CONTAINS, STARTS WITH, ENDS WITH."""
        return True

    def supports_case_insensitive_like(self) -> bool:
        """Neo4j pattern matching is case-sensitive by default."""
        return False

    def supports_element_type_enums(self) -> bool:
        """Neo4j supports ElementType enum integration."""
        return True

    def create_search_indexes(self):
        """
        Create additional indexes to optimize pattern matching searches.
        Call this after database initialization for better performance.
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            try:
                # Text indexes for better pattern matching performance
                additional_indexes = [
                    "CREATE TEXT INDEX IF NOT EXISTS FOR (e:Element) ON (e.content_preview)",
                    "CREATE TEXT INDEX IF NOT EXISTS FOR (e:Element) ON (e.element_type)",
                    "CREATE TEXT INDEX IF NOT EXISTS FOR (d:Document) ON (d.source)",
                    "CREATE TEXT INDEX IF NOT EXISTS FOR (d:Document) ON (d.doc_type)",
                    # Composite indexes for common query patterns
                    "CREATE INDEX IF NOT EXISTS FOR (e:Element) ON (e.doc_id, e.element_type)",
                    "CREATE INDEX IF NOT EXISTS FOR (e:Element) ON (e.element_type, e.content_preview)"
                ]

                for index_query in additional_indexes:
                    try:
                        session.run(index_query)
                    except Exception as e:
                        logger.debug(f"Could not create index: {str(e)}")

                logger.info("Created additional search optimization indexes for Neo4j")

            except Exception as e:
                logger.warning(f"Could not create search optimization indexes: {str(e)}")

    # ========================================
    # ENHANCED EMBEDDING FUNCTIONS
    # ========================================

    def store_embedding(self, element_pk: Union[int, str], embedding: VectorType) -> None:
        """Store embedding for an element."""
        if not self.driver:
            raise ValueError("Database not initialized")

        # Convert embedding to a JSON string for storage
        embedding_json = json.dumps(embedding)

        with self.driver.session(database=self.database) as session:
            if str(element_pk).isdigit():
                # Using Neo4j internal ID
                session.run(
                    """
                    MATCH (e:Element)
                    WHERE id(e) = $element_pk
                    MERGE (emb:Embedding {element_pk: $element_pk})
                    SET emb.embedding = $embedding,
                        emb.dimensions = $dimensions,
                        emb.topics = $topics,
                        emb.confidence = $confidence,
                        emb.created_at = $created_at
                    MERGE (emb)-[:EMBEDDING_OF]->(e)
                    """,
                    element_pk=int(element_pk),
                    embedding=embedding_json,
                    dimensions=len(embedding),
                    topics=json.dumps([]),  # Default to empty topics
                    confidence=1.0,  # Default confidence
                    created_at=time.time()
                )
            else:
                # Using element_id string
                session.run(
                    """
                    MATCH (e:Element {element_id: $element_id})
                    WITH e, id(e) AS element_pk
                    MERGE (emb:Embedding {element_pk: element_pk})
                    SET emb.embedding = $embedding,
                        emb.dimensions = $dimensions,
                        emb.topics = $topics,
                        emb.confidence = $confidence,
                        emb.created_at = $created_at
                    MERGE (emb)-[:EMBEDDING_OF]->(e)
                    """,
                    element_id=str(element_pk),
                    embedding=embedding_json,
                    dimensions=len(embedding),
                    topics=json.dumps([]),  # Default to empty topics
                    confidence=1.0,  # Default confidence
                    created_at=time.time()
                )

    def get_embedding(self, element_pk: Union[int, str]) -> Optional[VectorType]:
        """Get embedding for an element."""
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            if str(element_pk).isdigit():
                result = session.run(
                    """
                    MATCH (emb:Embedding {element_pk: $element_pk})
                    RETURN emb.embedding AS embedding
                    """,
                    element_pk=int(element_pk)
                )
            else:
                result = session.run(
                    """
                    MATCH (e:Element {element_id: $element_id})
                    WITH id(e) AS element_pk
                    MATCH (emb:Embedding {element_pk: element_pk})
                    RETURN emb.embedding AS embedding
                    """,
                    element_id=str(element_pk)
                )

            record = result.single()
            if not record:
                return None

            # Convert from JSON string back to list
            try:
                return json.loads(record["embedding"])
            except (json.JSONDecodeError, TypeError):
                return None

    def search_by_embedding(self, query_embedding: VectorType, limit: int = 10,
                            filter_criteria: Dict[str, Any] = None) -> List[Tuple[Union[int, str], float]]:
        """
        Search elements by embedding similarity.
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        # Always use fallback implementation since most Neo4j instances won't have vector extensions
        return self._fallback_embedding_search(query_embedding, limit, filter_criteria)

    def _fallback_embedding_search(self, query_embedding: VectorType, limit: int = 10,
                                   filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Fallback implementation for embedding search.
        This processes embeddings in Python instead of in the database.
        """
        # Check if NumPy is available for optimized calculation
        if NUMPY_AVAILABLE:
            return self._fallback_embedding_search_numpy(query_embedding, limit, filter_criteria)
        else:
            return self._fallback_embedding_search_pure_python(query_embedding, limit, filter_criteria)

    def _fallback_embedding_search_numpy(self, query_embedding: VectorType, limit: int = 10,
                                         filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """NumPy implementation of fallback embedding search."""
        # Convert query embedding to numpy array
        query_np = np.array(query_embedding)

        with self.driver.session(database=self.database) as session:
            # Build query to fetch embeddings
            cypher_query = """
            MATCH (emb:Embedding)-[:EMBEDDING_OF]->(e:Element)-[:BELONGS_TO]->(d:Document)
            WHERE emb.dimensions = $dimensions
            """

            params = {
                "dimensions": len(query_embedding)
            }

            # Add filter criteria if provided
            if filter_criteria:
                conditions = []
                for key, value in filter_criteria.items():
                    if key == "element_type" and isinstance(value, list):
                        conditions.append("e.element_type IN $element_types")
                        params["element_types"] = value
                    elif key == "doc_id" and isinstance(value, list):
                        conditions.append("e.doc_id IN $doc_ids")
                        params["doc_ids"] = value
                    elif key == "exclude_doc_id" and isinstance(value, list):
                        conditions.append("NOT e.doc_id IN $exclude_doc_ids")
                        params["exclude_doc_ids"] = value
                    elif key == "exclude_doc_source" and isinstance(value, list):
                        conditions.append("NOT d.source IN $exclude_sources")
                        params["exclude_sources"] = value
                    else:
                        conditions.append(f"e.{key} = ${key}")
                        params[key] = value

                if conditions:
                    cypher_query += " AND " + " AND ".join(conditions)

            # Complete the query
            cypher_query += """
            RETURN id(e) AS element_pk, emb.embedding AS embedding
            """

            # Execute query
            result = session.run(cypher_query, params)

            # Process results in Python
            similarities = []
            for record in result:
                element_pk = record["element_pk"]
                embedding_json = record["embedding"]

                try:
                    # Parse embedding
                    embedding = json.loads(embedding_json)
                    embedding_np = np.array(embedding)

                    # Calculate cosine similarity
                    similarity = self._cosine_similarity_numpy(query_np, embedding_np)
                    similarities.append((element_pk, similarity))
                except Exception as e:
                    logger.warning(f"Error processing embedding: {str(e)}")

            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]

    def _fallback_embedding_search_pure_python(self, query_embedding: VectorType, limit: int = 10,
                                               filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """Pure Python implementation of fallback embedding search when NumPy is not available."""
        with self.driver.session(database=self.database) as session:
            # Build query to fetch embeddings
            cypher_query = """
            MATCH (emb:Embedding)-[:EMBEDDING_OF]->(e:Element)-[:BELONGS_TO]->(d:Document)
            WHERE emb.dimensions = $dimensions
            """

            params = {
                "dimensions": len(query_embedding)
            }

            # Add filter criteria if provided
            if filter_criteria:
                conditions = []
                for key, value in filter_criteria.items():
                    if key == "element_type" and isinstance(value, list):
                        conditions.append("e.element_type IN $element_types")
                        params["element_types"] = value
                    elif key == "doc_id" and isinstance(value, list):
                        conditions.append("e.doc_id IN $doc_ids")
                        params["doc_ids"] = value
                    elif key == "exclude_doc_id" and isinstance(value, list):
                        conditions.append("NOT e.doc_id IN $exclude_doc_ids")
                        params["exclude_doc_ids"] = value
                    elif key == "exclude_doc_source" and isinstance(value, list):
                        conditions.append("NOT d.source IN $exclude_sources")
                        params["exclude_sources"] = value
                    else:
                        conditions.append(f"e.{key} = ${key}")
                        params[key] = value

                if conditions:
                    cypher_query += " AND " + " AND ".join(conditions)

            # Complete the query
            cypher_query += """
            RETURN id(e) AS element_pk, emb.embedding AS embedding
            """

            # Execute query
            result = session.run(cypher_query, params)

            # Process results in Python
            similarities = []
            for record in result:
                element_pk = record["element_pk"]
                embedding_json = record["embedding"]

                try:
                    # Parse embedding
                    embedding = json.loads(embedding_json)

                    # Calculate cosine similarity using pure Python
                    similarity = self._cosine_similarity_python(query_embedding, embedding)
                    similarities.append((element_pk, similarity))
                except Exception as e:
                    logger.warning(f"Error processing embedding: {str(e)}")

            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]

    def search_by_text(self, search_text: str, limit: int = 10,
                       filter_criteria: Dict[str, Any] = None) -> List[Tuple[int, float]]:
        """
        Search elements by semantic similarity to the provided text.

        Args:
            search_text: Text to search for semantically
            limit: Maximum number of results
            filter_criteria: Optional dictionary with criteria to filter results

        Returns:
            List of (element_id, similarity_score) tuples
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        try:
            # Initialize embedding generator if not already done
            if self.embedding_generator is None:
                try:
                    from ..embeddings import get_embedding_generator
                    if config:
                        self.embedding_generator = get_embedding_generator(config)
                    else:
                        logger.error("Config not available for embedding generator")
                        raise ValueError("Config not available")
                except ImportError as e:
                    logger.error(f"Embedding generator not available: {str(e)}")
                    raise ValueError("Embedding libraries are not installed.")

            # Generate embedding for the search text
            query_embedding = self.embedding_generator.generate(search_text)

            # Use the embedding to search
            return self.search_by_embedding(query_embedding, limit, filter_criteria)

        except Exception as e:
            logger.error(f"Error in semantic search by text: {str(e)}")
            return []

    # ========================================
    # NEW: TOPIC SUPPORT METHODS
    # ========================================

    def supports_topics(self) -> bool:
        """
        Indicate whether this backend supports topic-aware embeddings.

        Returns:
            True since Neo4j implementation now supports topics
        """
        return True

    def store_embedding_with_topics(self, element_pk: Union[int, str], embedding: VectorType,
                                    topics: List[str], confidence: float = 1.0) -> None:
        """
        Store embedding for an element with topic assignments.

        Args:
            element_pk: Element primary key
            embedding: Vector embedding
            topics: List of topic strings (e.g., ['security.policy', 'compliance'])
            confidence: Overall confidence in this embedding/topic assignment
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        # Convert embedding and topics to JSON strings for storage
        embedding_json = json.dumps(embedding)
        topics_json = json.dumps(topics)

        with self.driver.session(database=self.database) as session:
            if str(element_pk).isdigit():
                # Using Neo4j internal ID
                session.run(
                    """
                    MATCH (e:Element)
                    WHERE id(e) = $element_pk
                    MERGE (emb:Embedding {element_pk: $element_pk})
                    SET emb.embedding = $embedding,
                        emb.dimensions = $dimensions,
                        emb.topics = $topics,
                        emb.confidence = $confidence,
                        emb.created_at = $created_at
                    MERGE (emb)-[:EMBEDDING_OF]->(e)
                    """,
                    element_pk=int(element_pk),
                    embedding=embedding_json,
                    dimensions=len(embedding),
                    topics=topics_json,
                    confidence=confidence,
                    created_at=time.time()
                )
            else:
                # Using element_id string
                session.run(
                    """
                    MATCH (e:Element {element_id: $element_id})
                    WITH e, id(e) AS element_pk
                    MERGE (emb:Embedding {element_pk: element_pk})
                    SET emb.embedding = $embedding,
                        emb.dimensions = $dimensions,
                        emb.topics = $topics,
                        emb.confidence = $confidence,
                        emb.created_at = $created_at
                    MERGE (emb)-[:EMBEDDING_OF]->(e)
                    """,
                    element_id=str(element_pk),
                    embedding=embedding_json,
                    dimensions=len(embedding),
                    topics=topics_json,
                    confidence=confidence,
                    created_at=time.time()
                )

    def search_by_text_and_topics(self, search_text: str = None,
                                  include_topics: Optional[List[str]] = None,
                                  exclude_topics: Optional[List[str]] = None,
                                  min_confidence: float = 0.7,
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search elements by text with topic filtering using pattern matching.

        Args:
            search_text: Text to search for semantically (optional)
            include_topics: Topic patterns to include (e.g., ['security*', '*.policy*'])
            exclude_topics: Topic patterns to exclude (e.g., ['deprecated*'])
            min_confidence: Minimum confidence threshold for embeddings
            limit: Maximum number of results

        Returns:
            List of dictionaries with keys:
            - element_pk: Element primary key
            - similarity: Similarity score (if search_text provided)
            - confidence: Overall embedding confidence
            - topics: List of assigned topic strings
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        try:
            # Generate embedding for search text if provided
            query_embedding = None
            if search_text:
                if self.embedding_generator is None:
                    from ..embeddings import get_embedding_generator
                    if config:
                        self.embedding_generator = get_embedding_generator(config)
                    else:
                        logger.error("Config not available for embedding generator")
                        return []

                query_embedding = self.embedding_generator.generate(search_text)

            return self._search_by_text_and_topics_fallback(
                query_embedding, include_topics, exclude_topics, min_confidence, limit
            )

        except Exception as e:
            logger.error(f"Error in topic-aware search: {str(e)}")
            return []

    def _search_by_text_and_topics_fallback(self, query_embedding: Optional[VectorType] = None,
                                            include_topics: Optional[List[str]] = None,
                                            exclude_topics: Optional[List[str]] = None,
                                            min_confidence: float = 0.7,
                                            limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback search using Python similarity calculation with topic filtering."""

        with self.driver.session(database=self.database) as session:
            # Base query to get embeddings with confidence filtering
            cypher_query = """
            MATCH (emb:Embedding)-[:EMBEDDING_OF]->(e:Element)-[:BELONGS_TO]->(d:Document)
            WHERE emb.confidence >= $min_confidence
            RETURN id(e) AS element_pk, emb.embedding AS embedding, 
                   emb.confidence AS confidence, emb.topics AS topics
            """

            params = {"min_confidence": min_confidence}

            # Execute query
            result = session.run(cypher_query, params)

            results = []
            for record in result:
                element_pk = record["element_pk"]
                embedding_json = record["embedding"]
                confidence = record["confidence"]
                topics_json = record["topics"]

                try:
                    # Parse topics
                    topics = json.loads(topics_json) if topics_json else []
                except (json.JSONDecodeError, TypeError):
                    topics = []

                # Apply topic filtering
                if not self._matches_topic_filters(topics, include_topics, exclude_topics):
                    continue

                result_dict = {
                    'element_pk': element_pk,
                    'confidence': float(confidence),
                    'topics': topics
                }

                # Calculate similarity if we have a query embedding
                if query_embedding:
                    try:
                        embedding = json.loads(embedding_json)
                        if NUMPY_AVAILABLE:
                            similarity = self._cosine_similarity_numpy(query_embedding, embedding)
                        else:
                            similarity = self._cosine_similarity_python(query_embedding, embedding)
                        result_dict['similarity'] = float(similarity)
                    except Exception as e:
                        logger.warning(f"Error calculating similarity for element {element_pk}: {str(e)}")
                        result_dict['similarity'] = 0.0
                else:
                    result_dict['similarity'] = 1.0  # No text search, all results have equal similarity

                results.append(result_dict)

            # Sort by similarity if we calculated it
            if query_embedding:
                results.sort(key=lambda x: x['similarity'], reverse=True)

            return results[:limit]

    @staticmethod
    def _matches_topic_filters(topics: List[str],
                               include_topics: Optional[List[str]] = None,
                               exclude_topics: Optional[List[str]] = None) -> bool:
        """Check if topics match the include/exclude filters using pattern matching."""
        import fnmatch

        # Check include filters - at least one must match
        if include_topics:
            include_match = False
            for topic in topics:
                for pattern in include_topics:
                    if fnmatch.fnmatch(topic, pattern):
                        include_match = True
                        break
                if include_match:
                    break

            if not include_match:
                return False

        # Check exclude filters - none should match
        if exclude_topics:
            for topic in topics:
                for pattern in exclude_topics:
                    if fnmatch.fnmatch(topic, pattern):
                        return False

        return True

    def get_topic_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about topic distribution across embeddings.

        Returns:
            Dictionary mapping topic strings to statistics
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        try:
            topic_stats = {}

            with self.driver.session(database=self.database) as session:
                # Get all embeddings with topics
                result = session.run("""
                    MATCH (emb:Embedding)-[:EMBEDDING_OF]->(e:Element)-[:BELONGS_TO]->(d:Document)
                    WHERE emb.topics IS NOT NULL
                    RETURN emb.topics AS topics, emb.confidence AS confidence, d.doc_id AS doc_id
                """)

                for record in result:
                    try:
                        topics = json.loads(record["topics"]) if record["topics"] else []
                        confidence = record["confidence"]
                        doc_id = record["doc_id"]

                        for topic in topics:
                            if topic not in topic_stats:
                                topic_stats[topic] = {
                                    'embedding_count': 0,
                                    'document_ids': set(),
                                    'confidences': []
                                }

                            topic_stats[topic]['embedding_count'] += 1
                            topic_stats[topic]['confidences'].append(confidence)
                            if doc_id:
                                topic_stats[topic]['document_ids'].add(doc_id)
                    except (json.JSONDecodeError, TypeError):
                        continue

            # Calculate final statistics
            final_stats = {}
            for topic, stats in topic_stats.items():
                final_stats[topic] = {
                    'embedding_count': stats['embedding_count'],
                    'document_count': len(stats['document_ids']),
                    'avg_embedding_confidence': sum(stats['confidences']) / len(stats['confidences'])
                }

            return final_stats

        except Exception as e:
            logger.error(f"Error getting topic statistics: {str(e)}")
            return {}

    def get_embedding_topics(self, element_pk: Union[int, str]) -> List[str]:
        """
        Get topics assigned to a specific embedding.

        Args:
            element_pk: Element primary key

        Returns:
            List of topic strings assigned to this embedding
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        try:
            with self.driver.session(database=self.database) as session:
                if str(element_pk).isdigit():
                    result = session.run(
                        """
                        MATCH (emb:Embedding {element_pk: $element_pk})
                        RETURN emb.topics AS topics
                        """,
                        element_pk=int(element_pk)
                    )
                else:
                    result = session.run(
                        """
                        MATCH (e:Element {element_id: $element_id})
                        WITH id(e) AS element_pk
                        MATCH (emb:Embedding {element_pk: element_pk})
                        RETURN emb.topics AS topics
                        """,
                        element_id=str(element_pk)
                    )

                record = result.single()
                if not record or record["topics"] is None:
                    return []

                try:
                    return json.loads(record["topics"])
                except (json.JSONDecodeError, TypeError):
                    return []

        except Exception as e:
            logger.error(f"Error getting topics for element {element_pk}: {str(e)}")
            return []

    @staticmethod
    def _cosine_similarity_numpy(vec1: VectorType, vec2: VectorType) -> float:
        """
        Calculate cosine similarity between two vectors using NumPy.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (between -1 and 1)
        """
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
        """
        Calculate cosine similarity between two vectors using pure Python.
        This is a fallback when NumPy is not available.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (between -1 and 1)
        """
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

    # ========================================
    # EXISTING HIERARCHY METHODS (adapted for Neo4j)
    # ========================================

    def get_results_outline(self, elements: List[Tuple[int, float]]) -> List["ElementHierarchical"]:
        """
        For an arbitrary list of element pk search results, finds the root node of the source, and each
        ancestor element, to create a root -> element array of arrays like this:
        [(<parent element>, score, [children])]

        (Note score is None if the element was not in the results param)

        Then each additional element is analyzed, its hierarchy materialized, and merged into
        the final result.
        """
        from .element_element import ElementHierarchical

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
                    current_level = current_level[existing_idx].child_elements  # Get children list
                else:
                    # Ancestor doesn't exist, add it with its score (or None if not in search results)
                    ancestor_score = element_scores.get(ancestor_pk)
                    children = []
                    ancestor.score = ancestor_score
                    h_ancestor = ancestor.to_hierarchical()
                    h_ancestor.child_elements = children
                    current_level.append(h_ancestor)
                    current_level = children

        return result_tree

    def _get_element_ancestry_path(self, element_pk: int) -> List["ElementBase"]:
        """
        Get the complete ancestry path for an element, from root to the element itself.

        Uses parent_id to find parents instead of relationships.
        """
        from .element_element import ElementBase

        # Get the element
        element_dict = self.get_element(element_pk)
        if not element_dict:
            return []

        # Convert to ElementElement instance
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
            parent_pk = parent_dict.get('id') or parent_dict.get('pk') or parent_dict.get('element_id')
            if parent_pk in visited:
                break

            # Convert to ElementElement
            parent = ElementBase(**parent_dict)

            # Add to visited set
            visited.add(parent_pk)

            # Add parent to the beginning of the ancestry list (root first)
            ancestry.insert(0, parent)

            # Move up to the parent
            current_pk = parent_id

        return ancestry

    # ========================================
    # DATE STORAGE AND SEARCH METHODS (Neo4j specific implementation)
    # ========================================

    def store_element_dates(self, element_id: str, dates: List[Dict[str, Any]]) -> None:
        """
        Store extracted dates associated with an element using Neo4j relationships.

        Args:
            element_id: Element ID
            dates: List of date dictionaries from ExtractedDate.to_dict()
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            # First, delete existing date relationships for this element
            session.run("""
                MATCH (e:Element {element_id: $element_id})-[r:HAS_DATE]->(d:ExtractedDate)
                DELETE r, d
            """, element_id=element_id)

            # Store each date as a separate node with relationship
            for date_dict in dates:
                session.run("""
                    MATCH (e:Element {element_id: $element_id})
                    CREATE (d:ExtractedDate {
                        original_text: $original_text,
                        parsed_date: $parsed_date,
                        timestamp_value: $timestamp_value,
                        confidence: $confidence,
                        specificity_level: $specificity_level,
                        date_format: $date_format,
                        metadata: $metadata
                    })
                    CREATE (e)-[:HAS_DATE]->(d)
                """,
                            element_id=element_id,
                            original_text=date_dict.get('original_text', ''),
                            parsed_date=date_dict.get('parsed_date', ''),
                            timestamp_value=date_dict.get('timestamp'),
                            confidence=date_dict.get('confidence', 1.0),
                            specificity_level=date_dict.get('specificity_level', ''),
                            date_format=date_dict.get('date_format', ''),
                            metadata=json.dumps(date_dict.get('metadata', {}))
                            )

    def get_element_dates(self, element_id: str) -> List[Dict[str, Any]]:
        """
        Get all dates associated with an element.

        Args:
            element_id: Element ID

        Returns:
            List of date dictionaries, empty list if none found
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (e:Element {element_id: $element_id})-[:HAS_DATE]->(d:ExtractedDate)
                RETURN d
                ORDER BY d.timestamp_value
            """, element_id=element_id)

            dates = []
            for record in result:
                date_node = dict(record['d'])

                # Convert back to expected format
                date_dict = {
                    'original_text': date_node.get('original_text', ''),
                    'parsed_date': date_node.get('parsed_date', ''),
                    'timestamp': date_node.get('timestamp_value'),
                    'confidence': date_node.get('confidence', 1.0),
                    'specificity_level': date_node.get('specificity_level', ''),
                    'date_format': date_node.get('date_format', ''),
                    'metadata': json.loads(date_node.get('metadata', '{}'))
                }
                dates.append(date_dict)

            return dates

    def store_embedding_with_dates(self, element_id: str, embedding: List[float],
                                   dates: List[Dict[str, Any]]) -> None:
        """
        Store both embedding and dates for an element in a single operation.

        Args:
            element_id: Element ID
            embedding: Vector embedding
            dates: List of extracted date dictionaries
        """
        # Store embedding and dates separately in Neo4j
        self.store_embedding(element_id, embedding)
        self.store_element_dates(element_id, dates)

    def delete_element_dates(self, element_id: str) -> bool:
        """
        Delete all dates associated with an element.

        Args:
            element_id: Element ID

        Returns:
            True if dates were deleted, False if none existed
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (e:Element {element_id: $element_id})-[r:HAS_DATE]->(d:ExtractedDate)
                DELETE r, d
                RETURN count(r) AS deleted_count
            """, element_id=element_id)

            record = result.single()
            return record and record['deleted_count'] > 0

    def search_elements_by_date_range(self, start_date: datetime.datetime, end_date: datetime.datetime,
                                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find elements that contain dates within a specified range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            limit: Maximum number of results

        Returns:
            List of element dictionaries that contain dates in the range
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (e:Element)-[:HAS_DATE]->(d:ExtractedDate)
                WHERE d.timestamp_value >= $start_timestamp 
                AND d.timestamp_value <= $end_timestamp
                RETURN DISTINCT e, id(e) AS element_pk
                LIMIT $limit
            """,
                                 start_timestamp=start_date.timestamp(),
                                 end_timestamp=end_date.timestamp(),
                                 limit=limit)

            elements = []
            for record in result:
                element = dict(record['e'])
                element['element_pk'] = record['element_pk']

                # Convert metadata from JSON
                try:
                    element["metadata"] = json.loads(element.get("metadata", "{}"))
                except (json.JSONDecodeError, TypeError):
                    element["metadata"] = {}

                elements.append(element)

            return elements

    def search_by_text_and_date_range(self,
                                      search_text: str,
                                      start_date: Optional[datetime.datetime] = None,
                                      end_date: Optional[datetime.datetime] = None,
                                      limit: int = 10) -> List[Tuple[int, float]]:
        """
        Search elements by semantic similarity AND date range.

        Args:
            search_text: Text to search for semantically
            start_date: Optional start of date range
            end_date: Optional end of date range
            limit: Maximum number of results

        Returns:
            List of (element_id, similarity_score) tuples
        """
        # First get elements in date range if specified
        date_element_pks = None
        if start_date or end_date:
            date_element_pks = set()

            with self.driver.session(database=self.database) as session:
                cypher_query = """
                    MATCH (e:Element)-[:HAS_DATE]->(d:ExtractedDate)
                    WHERE 1=1
                """
                params = {}

                if start_date:
                    cypher_query += " AND d.timestamp_value >= $start_timestamp"
                    params["start_timestamp"] = start_date.timestamp()

                if end_date:
                    cypher_query += " AND d.timestamp_value <= $end_timestamp"
                    params["end_timestamp"] = end_date.timestamp()

                cypher_query += " RETURN DISTINCT id(e) AS element_pk"

                result = session.run(cypher_query, params)
                date_element_pks = {record["element_pk"] for record in result}

        # Perform text similarity search
        text_results = self.search_by_text(search_text, limit=limit * 2)  # Get more to allow for filtering

        # Filter by date results if we have them
        if date_element_pks is not None:
            filtered_results = []
            for element_pk, similarity in text_results:
                if element_pk in date_element_pks:
                    filtered_results.append((element_pk, similarity))
            return filtered_results[:limit]
        else:
            return text_results[:limit]

    def search_by_embedding_and_date_range(self,
                                           query_embedding: List[float],
                                           start_date: Optional[datetime.datetime] = None,
                                           end_date: Optional[datetime.datetime] = None,
                                           limit: int = 10) -> List[Tuple[int, float]]:
        """
        Search elements by embedding similarity AND date range.

        Args:
            query_embedding: Query embedding vector
            start_date: Optional start of date range
            end_date: Optional end of date range
            limit: Maximum number of results

        Returns:
            List of (element_id, similarity_score) tuples
        """
        # First get elements in date range if specified
        date_element_pks = None
        if start_date or end_date:
            date_element_pks = set()

            with self.driver.session(database=self.database) as session:
                cypher_query = """
                    MATCH (e:Element)-[:HAS_DATE]->(d:ExtractedDate)
                    WHERE 1=1
                """
                params = {}

                if start_date:
                    cypher_query += " AND d.timestamp_value >= $start_timestamp"
                    params["start_timestamp"] = start_date.timestamp()

                if end_date:
                    cypher_query += " AND d.timestamp_value <= $end_timestamp"
                    params["end_timestamp"] = end_date.timestamp()

                cypher_query += " RETURN DISTINCT id(e) AS element_pk"

                result = session.run(cypher_query, params)
                date_element_pks = {record["element_pk"] for record in result}

        # Perform embedding similarity search
        embedding_results = self.search_by_embedding(query_embedding, limit=limit * 2)

        # Filter by date results if we have them
        if date_element_pks is not None:
            filtered_results = []
            for element_pk, similarity in embedding_results:
                if element_pk in date_element_pks:
                    filtered_results.append((element_pk, similarity))
            return filtered_results[:limit]
        else:
            return embedding_results[:limit]

    def get_elements_with_dates(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all elements that have associated dates.

        Args:
            limit: Maximum number of results

        Returns:
            List of element dictionaries that have dates
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (e:Element)-[:HAS_DATE]->(:ExtractedDate)
                RETURN DISTINCT e, id(e) AS element_pk
                LIMIT $limit
            """, limit=limit)

            elements = []
            for record in result:
                element = dict(record['e'])
                element['element_pk'] = record['element_pk']

                # Convert metadata from JSON
                try:
                    element["metadata"] = json.loads(element.get("metadata", "{}"))
                except (json.JSONDecodeError, TypeError):
                    element["metadata"] = {}

                elements.append(element)

            return elements

    def get_date_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about dates in the database.

        Returns:
            Dictionary with date statistics
        """
        if not self.driver:
            raise ValueError("Database not initialized")

        try:
            with self.driver.session(database=self.database) as session:
                # Get basic counts and statistics
                result = session.run("""
                    MATCH (d:ExtractedDate)
                    RETURN 
                        count(d) AS total_dates,
                        min(d.timestamp_value) AS earliest_timestamp,
                        max(d.timestamp_value) AS latest_timestamp,
                        avg(d.confidence) AS avg_confidence
                """)

                record = result.single()
                if not record:
                    return {}

                stats = {
                    'total_dates': record['total_dates'],
                    'avg_confidence': record['avg_confidence'],
                    'earliest_date': None,
                    'latest_date': None
                }

                if record['earliest_timestamp']:
                    stats['earliest_date'] = datetime.datetime.fromtimestamp(record['earliest_timestamp']).isoformat()
                if record['latest_timestamp']:
                    stats['latest_date'] = datetime.datetime.fromtimestamp(record['latest_timestamp']).isoformat()

                # Get count of elements with dates
                result = session.run("""
                    MATCH (e:Element)-[:HAS_DATE]->(:ExtractedDate)
                    RETURN count(DISTINCT e) AS elements_with_dates
                """)

                record = result.single()
                if record:
                    stats['elements_with_dates'] = record['elements_with_dates']

                # Get specificity level distribution
                result = session.run("""
                    MATCH (d:ExtractedDate)
                    RETURN d.specificity_level AS level, count(d) AS count
                    ORDER BY count DESC
                """)

                specificity_dist = {}
                for record in result:
                    level = record['level'] or 'unknown'
                    specificity_dist[level] = record['count']

                stats['specificity_distribution'] = specificity_dist

                return stats

        except Exception as e:
            logger.error(f"Error getting date statistics: {str(e)}")
            return {}


if __name__ == "__main__":
    # Example demonstrating structured search with Neo4j
    db = Neo4jDocumentDatabase("bolt://localhost:7687", "neo4j", "password")
    db.initialize()

    # Show backend capabilities
    capabilities = db.get_backend_capabilities()
    print(f"Neo4j supports {len(capabilities.supported)} capabilities:")
    for cap in sorted(capabilities.get_supported_list()):
        print(f"  ✓ {cap}")

    # Example structured search
    from .structured_search import SearchQueryBuilder, LogicalOperator

    query = (SearchQueryBuilder()
             .with_operator(LogicalOperator.AND)
             .text_search("machine learning algorithms", similarity_threshold=0.8)
             .last_days(30)
             .topics(include=["ml%", "ai%"])
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
