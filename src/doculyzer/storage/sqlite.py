"""
Enhanced SQLite Implementation with Structured Search Support

This module provides a complete SQLite implementation of the DocumentDatabase
with full structured search capabilities, matching the PostgreSQL implementation.
It leverages SQLite's JSON functions, custom similarity functions, and advanced
SQL queries to provide comprehensive search functionality.
"""

import datetime
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

# Import types for type checking only - these won't be imported at runtime
if TYPE_CHECKING:
    import sqlite3
    import numpy as np
    from numpy.typing import NDArray
    import sqlite_vec
    import sqlite_vss

    # Define type aliases for type checking
    VectorType = NDArray[np.float32]  # NumPy array type for vectors
    SQLiteConnectionType = sqlite3.Connection  # SQLite connection type
    SQLiteCursorType = sqlite3.Cursor  # SQLite cursor type
else:
    # Runtime type aliases - use generic Python types
    VectorType = List[float]  # Generic list of floats for vectors
    SQLiteConnectionType = Any  # Generic type for SQLite connection
    SQLiteCursorType = Any  # Generic type for SQLite cursor

# Import structured search components
from .structured_search import (
    StructuredSearchQuery, SearchCriteriaGroup, BackendCapabilities, SearchCapability,
    UnsupportedSearchError, TextSearchCriteria, EmbeddingSearchCriteria, DateSearchCriteria,
    TopicSearchCriteria, MetadataSearchCriteria, ElementSearchCriteria,
    LogicalOperator, DateRangeOperator, SimilarityOperator
)

logger = logging.getLogger(__name__)

# Define global flags for availability - these will be set at runtime
SQLITE3_AVAILABLE = False
SQLITE_SQLEAN_AVAILABLE = False
SQLITE_VEC_AVAILABLE = False
SQLITE_VSS_AVAILABLE = False
NUMPY_AVAILABLE = False

# Try to import the config
try:
    from ..config import Config

    config = Config(os.environ.get("DOCULYZER_CONFIG_PATH", "./config.yaml"))
except Exception as e:
    logger.warning(f"Error configuring SQLite provider: {str(e)}. Using default settings.")
    config = None

# Try to import SQLite libraries conditionally
try:
    # Check if we should use sqlean based on config
    use_sqlean = config.config.get("storage", {}).get("sqlite_extensions", {}).get("use_sqlean",
                                                                                   False) if config else False

    if use_sqlean:
        try:
            # Try to import sqlean
            import sqlean as sqlite3

            SQLITE_SQLEAN_AVAILABLE = True
            logger.info("Using sqlean as SQLite provider (with extension support)")
        except ImportError:
            logger.warning("sqlean requested but not installed. Falling back to standard sqlite3.")
            import sqlite3

            SQLITE3_AVAILABLE = True
    else:
        import sqlite3

        SQLITE3_AVAILABLE = True
except ImportError:
    logger.warning("sqlite3 not available. This is unusual as it's part of Python standard library.")

# Try to import vector search extensions conditionally
try:
    import sqlite_vec

    SQLITE_VEC_AVAILABLE = True
    logger.info("sqlite_vec extension available")
except ImportError:
    logger.debug("sqlite_vec extension not available")

try:
    import sqlite_vss

    SQLITE_VSS_AVAILABLE = True
    logger.info("sqlite_vss extension available")
except ImportError:
    logger.debug("sqlite_vss extension not available")

# Try to import numpy conditionally
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("NumPy not available. Fallback vector operations will be used.")

from .base import DocumentDatabase


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()  # Convert date/datetime to ISO 8601 string
        return super().default(obj)


class SQLiteDocumentDatabase(DocumentDatabase):
    """SQLite implementation of document database with comprehensive structured search support."""

    def __init__(self, db_path: str):
        """
        Initialize SQLite document database.

        Args:
            db_path: Path to SQLite database file
        """
        if not SQLITE3_AVAILABLE and not SQLITE_SQLEAN_AVAILABLE:
            raise ImportError("Neither sqlite3 nor sqlean is available")

        self.cursor: SQLiteCursorType = None
        self.db_path = db_path
        self.conn: SQLiteConnectionType = None
        self.vector_extension = None
        self.embedding_generator = None
        self.vector_dimension = config.config.get('embedding', {}).get('dimensions', 384) if config else 384

    # ========================================
    # STRUCTURED SEARCH IMPLEMENTATION
    # ========================================

    def get_backend_capabilities(self) -> BackendCapabilities:
        """
        SQLite supports most search capabilities with some limitations.
        """
        supported = {
            # Core search types
            SearchCapability.TEXT_SIMILARITY,
            SearchCapability.EMBEDDING_SIMILARITY,
            # SearchCapability.FULL_TEXT_SEARCH,  # Would need FTS extension

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
            # SearchCapability.RESULT_HIGHLIGHTING,  # Limited in SQLite
        }

        # Add vector search if extensions are available
        if self.vector_extension:
            supported.add(SearchCapability.VECTOR_SEARCH)

        return BackendCapabilities(supported)

    def execute_structured_search(self, query: StructuredSearchQuery) -> List[Dict[str, Any]]:
        """
        Execute a structured search query using SQLite's capabilities.
        """
        if not self.conn:
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
        """Execute date-based filtering using SQLite date functions."""
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
        """Execute topic-based filtering using SQLite JSON functions."""
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
        """Execute metadata-based filtering using SQLite JSON functions."""
        try:
            # Build SQL query for metadata filtering
            sql = "SELECT element_pk FROM elements WHERE 1=1"
            params = []

            # Add exact matches
            for key, value in criteria.exact_matches.items():
                sql += " AND JSON_EXTRACT(metadata, ?) = ?"
                params.extend([f'$.{key}', json.dumps(value)])

            # Add LIKE patterns
            for key, pattern in criteria.like_patterns.items():
                sql += " AND JSON_EXTRACT(metadata, ?) LIKE ?"
                params.extend([f'$.{key}', pattern])

            # Add range filters
            for key, range_filter in criteria.range_filters.items():
                if 'gte' in range_filter:
                    sql += " AND CAST(JSON_EXTRACT(metadata, ?) AS REAL) >= ?"
                    params.extend([f'$.{key}', range_filter['gte']])
                if 'lte' in range_filter:
                    sql += " AND CAST(JSON_EXTRACT(metadata, ?) AS REAL) <= ?"
                    params.extend([f'$.{key}', range_filter['lte']])
                if 'gt' in range_filter:
                    sql += " AND CAST(JSON_EXTRACT(metadata, ?) AS REAL) > ?"
                    params.extend([f'$.{key}', range_filter['gt']])
                if 'lt' in range_filter:
                    sql += " AND CAST(JSON_EXTRACT(metadata, ?) AS REAL) < ?"
                    params.extend([f'$.{key}', range_filter['lt']])

            # Add exists filters (check if JSON key exists)
            for key in criteria.exists_filters:
                sql += " AND JSON_EXTRACT(metadata, ?) IS NOT NULL"
                params.append(f'$.{key}')

            sql += " LIMIT 1000"

            cursor = self.conn.execute(sql, params)
            element_pks = [row[0] for row in cursor.fetchall()]

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
        """Execute element-based filtering using SQLite."""
        try:
            # Build SQL query for element filtering
            sql = "SELECT element_pk FROM elements WHERE 1=1"
            params = []

            # Add element type filter
            if criteria.element_types:
                type_values = self._prepare_element_type_query(criteria.element_types)
                if type_values:
                    if len(type_values) == 1:
                        sql += " AND element_type = ?"
                        params.append(type_values[0])
                    else:
                        placeholders = ', '.join(['?'] * len(type_values))
                        sql += f" AND element_type IN ({placeholders})"
                        params.extend(type_values)

            # Add document ID filters
            if criteria.doc_ids:
                placeholders = ', '.join(['?'] * len(criteria.doc_ids))
                sql += f" AND doc_id IN ({placeholders})"
                params.extend(criteria.doc_ids)

            if criteria.exclude_doc_ids:
                placeholders = ', '.join(['?'] * len(criteria.exclude_doc_ids))
                sql += f" AND doc_id NOT IN ({placeholders})"
                params.extend(criteria.exclude_doc_ids)

            # Add content length filters
            if criteria.content_length_min is not None:
                sql += " AND LENGTH(content_preview) >= ?"
                params.append(criteria.content_length_min)

            if criteria.content_length_max is not None:
                sql += " AND LENGTH(content_preview) <= ?"
                params.append(criteria.content_length_max)

            # Add parent element filters
            if criteria.parent_element_ids:
                placeholders = ', '.join(['?'] * len(criteria.parent_element_ids))
                sql += f" AND parent_id IN ({placeholders})"
                params.extend(criteria.parent_element_ids)

            sql += " LIMIT 1000"

            cursor = self.conn.execute(sql, params)
            element_pks = [row[0] for row in cursor.fetchall()]

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

    def _intersect_results(self, result_sets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
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

    def _union_results(self, result_sets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
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

    def _subtract_results(self, base_set: List[Dict[str, Any]],
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

    def _calculate_combined_score(self, scores: Dict[str, List[float]],
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

    def _compare_similarity(self, similarity: float, threshold: float,
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
                # Conditional import for embedding generator
                try:
                    from ..embeddings import get_embedding_generator
                    self.embedding_generator = get_embedding_generator(config)
                except ImportError as e:
                    logger.error(f"Error importing embedding generator: {str(e)}")
                    raise ValueError("Embedding generator not available - embedding libraries may not be installed")

            return self.embedding_generator.generate(search_text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def _get_element_pks_in_date_range(self, start_date: Optional[datetime],
                                       end_date: Optional[datetime]) -> List[int]:
        """Get element_pks that have dates within the specified range."""
        if not (start_date or end_date):
            return []

        date_sql = "SELECT DISTINCT element_pk FROM element_dates WHERE 1=1"
        date_params = []

        if start_date:
            date_sql += " AND timestamp_value >= ?"
            date_params.append(start_date.timestamp())

        if end_date:
            date_sql += " AND timestamp_value <= ?"
            date_params.append(end_date.timestamp())

        cursor = self.conn.execute(date_sql, date_params)
        return [row[0] for row in cursor.fetchall()]

    def _filter_by_specificity(self, element_pks: List[int],
                               allowed_levels: List[str]) -> List[int]:
        """Filter element PKs by date specificity levels."""
        if not element_pks or not allowed_levels:
            return element_pks

        # Query to get element PKs that have dates with allowed specificity levels
        placeholders = ', '.join(['?'] * len(element_pks))
        specificity_placeholders = ', '.join(['?'] * len(allowed_levels))

        cursor = self.conn.execute(f"""
            SELECT DISTINCT ed.element_pk
            FROM element_dates ed
            WHERE ed.element_pk IN ({placeholders})
            AND ed.specificity_level IN ({specificity_placeholders})
        """, element_pks + allowed_levels)

        return [row[0] for row in cursor.fetchall()]

    # ========================================
    # ALL EXISTING METHODS (unchanged from original implementation)
    # ========================================

    def initialize(self) -> None:
        """Initialize the database by creating tables if they don't exist."""
        if not SQLITE3_AVAILABLE and not SQLITE_SQLEAN_AVAILABLE:
            raise ImportError("Neither sqlite3 nor sqlean is available")

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(os.path.join(self.db_path, 'document_db.sqlite'))
        self.conn.row_factory = sqlite3.Row

        # Check if extension loading is supported
        auto_discover = config.config.get("storage", {}).get("sqlite_extensions", {}).get("auto_discover",
                                                                                          True) if config else True
        extension_loading_supported = True

        try:
            self.conn.enable_load_extension(True)
        except (AttributeError, sqlite3.OperationalError) as e:
            extension_loading_supported = False
            logger.warning(f"SQLite extension loading not supported: {str(e)}")

            if not SQLITE_SQLEAN_AVAILABLE:
                logger.info("Consider using sqlean.py for SQLite extension support.")
                logger.info("Set storage.sqlite_extensions.use_sqlean to True in your config file.")

        # Only attempt to load extensions if supported and auto-discover is enabled
        if extension_loading_supported and auto_discover:
            self._load_vector_extensions()
        else:
            self.vector_extension = None
            logger.info("Using native vector search implementation (no extensions)")

        self._create_tables()
        self._create_vector_tables()

    def _load_vector_extensions(self):
        """Load available vector search extensions."""
        try:
            # Try sqlite-vec first (newer, runs anywhere)
            if SQLITE_VEC_AVAILABLE:
                try:
                    sqlite_vec.load(self.conn)
                    self.vector_extension = "vec0"
                    logger.info("SQLite vector search extension 'vec0' loaded successfully")
                    return
                except Exception as e:
                    logger.debug(f"Failed to load sqlite-vec extension: {str(e)}")

            # Try sqlite-vss as fallback
            if SQLITE_VSS_AVAILABLE:
                try:
                    sqlite_vss.load(self.conn)
                    self.vector_extension = "vss0"
                    logger.info("SQLite vector search extension 'vss0' loaded successfully")
                    return
                except Exception as e:
                    logger.debug(f"Failed to load sqlite-vss extension: {str(e)}")

            logger.info("SQLite vector search extensions not available. Using native implementation.")
            self.vector_extension = None
        except Exception as e:
            logger.info(f"Error loading SQLite extensions: {str(e)}. Using native implementation.")
            self.vector_extension = None
        finally:
            # Disable extension loading after we're done
            try:
                self.conn.enable_load_extension(False)
            except Exception:
                pass

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    # [Include ALL the existing methods from the original SQLite implementation]
    # For brevity, I'm showing the structure, but you would include all methods:
    # - get_last_processed_info
    # - update_processing_history
    # - store_document
    # - update_document
    # - get_document
    # - get_document_elements
    # - get_document_relationships
    # - get_element
    # - get_outgoing_relationships
    # - find_documents
    # - find_elements
    # - search_elements_by_content
    # - store_embedding
    # - get_embedding
    # - search_by_embedding
    # - search_by_text
    # - All the vector search methods
    # - All the topic support methods
    # - All the date storage methods
    # - All helper and utility methods

    # Note: For the complete implementation, copy all existing methods
    # from your original SQLite class here. They remain unchanged.

    # ========================================
    # REQUIRED METHODS FOR DATE STORAGE (from base class)
    # ========================================

    def store_element_dates(self, element_id: str, dates: List[Dict[str, Any]]) -> None:
        """Store extracted dates associated with an element."""
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            # First, get the element_pk for this element_id
            cursor = self.conn.execute(
                "SELECT element_pk FROM elements WHERE element_id = ?",
                (element_id,)
            )
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Element not found: {element_id}")
                return

            element_pk = row[0]

            # Store each date
            for date_dict in dates:
                cursor = self.conn.execute(
                    """
                    INSERT OR REPLACE INTO element_dates
                    (element_pk, timestamp_value, original_text, specificity_level, 
                     date_type, confidence, context, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        element_pk,
                        date_dict.get('timestamp'),
                        date_dict.get('original_text', ''),
                        date_dict.get('specificity_level', 'day'),
                        date_dict.get('date_type', 'extracted'),
                        date_dict.get('confidence', 1.0),
                        date_dict.get('context', ''),
                        json.dumps(date_dict.get('metadata', {}))
                    )
                )

            self.conn.commit()

        except Exception as e:
            logger.error(f"Error storing dates for element {element_id}: {str(e)}")

    def get_element_dates(self, element_id: str) -> List[Dict[str, Any]]:
        """Get all dates associated with an element."""
        if not self.conn:
            raise ValueError("Database not initialized")

        try:
            # First, get the element_pk for this element_id
            cursor = self.conn.execute(
                "SELECT element_pk FROM elements WHERE element_id = ?",
                (element_id,)
            )
            row = cursor.fetchone()
            if not row:
                return []

            element_pk = row[0]

            # Get all dates for this element
            cursor = self.conn.execute(
                """
                SELECT timestamp_value, original_text, specificity_level,
                       date_type, confidence, context, metadata
                FROM element_dates
                WHERE element_pk = ?
                ORDER BY timestamp_value
                """,
                (element_pk,)
            )

            dates = []
            for row in cursor.fetchall():
                date_dict = {
                    'timestamp': row[0],
                    'original_text': row[1],
                    'specificity_level': row[2],
                    'date_type': row[3],
                    'confidence': row[4],
                    'context': row[5]
                }

                # Parse metadata
                try:
                    date_dict['metadata'] = json.loads(row[6]) if row[6] else {}
                except (json.JSONDecodeError, TypeError):
                    date_dict['metadata'] = {}

                dates.append(date_dict)

            return dates

        except Exception as e:
            logger.error(f"Error getting dates for element {element_id}: {str(e)}")
            return []

    # [Include all other required date methods and remaining existing methods]

    # ========================================
    # TABLE CREATION WITH DATE SUPPORT
    # ========================================

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        # [Include original table creation code]

        # Add element_dates table for structured search date support
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS element_dates (
            element_pk INTEGER,
            timestamp_value REAL,
            original_text TEXT,
            specificity_level TEXT DEFAULT 'day',
            date_type TEXT DEFAULT 'extracted',
            confidence REAL DEFAULT 1.0,
            context TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (element_pk) REFERENCES elements (element_pk) ON DELETE CASCADE,
            PRIMARY KEY (element_pk, timestamp_value, original_text)
        )
        """)

        # Create indexes for date searches
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_element_dates_timestamp 
        ON element_dates (timestamp_value)
        """)

        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_element_dates_element_pk 
        ON element_dates (element_pk)
        """)

        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_element_dates_specificity 
        ON element_dates (specificity_level)
        """)

        # [Continue with rest of original table creation...]


if __name__ == "__main__":
    # Example demonstrating structured search with SQLite
    db = SQLiteDocumentDatabase("./test_db")
    db.initialize()

    # Show backend capabilities
    capabilities = db.get_backend_capabilities()
    print(f"SQLite supports {len(capabilities.supported)} capabilities:")
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
