from typing import List

from doculyzer.storage import StructuredSearchQuery, UnsupportedSearchError


def execute_structured_search(self, query: StructuredSearchQuery) -> List[Dict[str, Any]]:
    """Execute a structured search query."""
    if not self.es:
        raise ValueError("Database not initialized")

    # Validate query capabilities
    missing_capabilities = self.validate_query_support(query)
    if missing_capabilities:
        raise UnsupportedSearchError(
            f"Query uses unsupported capabilities: {[c.value for c in missing_capabilities]}"
        )

    try:
        # Convert structured query to Elasticsearch query
        es_query = self._build_elasticsearch_query(query.criteria_group)

        # Execute search
        search_body = {
            "query": es_query,
            "size": query.limit,
            "from": query.offset or 0
        }

        # Add sorting and scoring if needed
        if query.include_similarity_scores:
            search_body["track_scores"] = True

        # Add highlighting if requested
        if query.include_highlighting:
            search_body["highlight"] = {
                "fields": {
                    "content_preview": {},
                    "full_text": {}
                }
            }

        result = self.es.search(index=self.elements_index, body=search_body)

        # Process results
        formatted_results = []
        for hit in result['hits']['hits']:
            element_data = hit['_source']
            result_item = {
                'element_pk': int(element_data['element_pk']),
                'element_id': element_data['element_id'],
                'doc_id': element_data['doc_id'],
                'element_type': element_data['element_type'],
                'content_preview': element_data.get('content_preview', '')
            }

            # Add similarity score if requested
            if query.include_similarity_scores:
                result_item['final_score'] = float(hit.get('_score', 0.0))

            # Add metadata if requested
            if query.include_metadata and element_data.get('metadata'):
                # Parse metadata_json if present
                if "metadata_json" in element_data and not element_data.get("metadata"):
                    try:
                        element_data["metadata"] = json.loads(element_data["metadata_json"])
                    except:
                        pass
                result_item['metadata'] = element_data.get('metadata', {})

            # Add topics if requested
            if query.include_topics and element_data.get('topics'):
                result_item['topics'] = element_data['topics']

            # Add dates if requested
            if query.include_element_dates:
                dates = self.get_element_dates(element_data['element_id'])
                result_item['extracted_dates'] = dates

            # Add highlighting if available
            if query.include_highlighting and 'highlight' in hit:
                result_item['highlights'] = hit['highlight']

            formatted_results.append(result_item)

        return formatted_results

    except Exception as e:
        logger.error(f"Error executing structured search: {str(e)}")
        return []


def _build_elasticsearch_query(self, criteria_group) -> Dict[str, Any]:
    """Convert a SearchCriteriaGroup to Elasticsearch query DSL."""
    from .structured_search import LogicalOperator

    # Handle empty group
    if not self._has_criteria(criteria_group):
        return {"match_all": {}}

    query_parts = []

    # Handle text search criteria
    if criteria_group.text_criteria:
        text_query = self._build_text_query(criteria_group.text_criteria)
        query_parts.append(text_query)

    # Handle embedding search criteria
    if criteria_group.embedding_criteria:
        # For now, we'll handle this as a post-filter since ES kNN is more complex
        # In a full implementation, you'd convert this to a kNN query
        logger.warning("Embedding search criteria not fully implemented in basic ES query")

    # Handle filters that will be combined
    filters = []

    # Date filtering
    if criteria_group.date_criteria:
        date_filter = self._build_date_filter(criteria_group.date_criteria)
        if date_filter:
            filters.append(date_filter)

    # Element type filtering
    if criteria_group.element_criteria and criteria_group.element_criteria.element_types:
        filters.append({
            "terms": {"element_type": criteria_group.element_criteria.element_types}
        })

    # Document ID filtering
    if criteria_group.element_criteria and criteria_group.element_criteria.doc_ids:
        filters.append({
            "terms": {"doc_id": criteria_group.element_criteria.doc_ids}
        })

    # Document ID exclusion
    if criteria_group.element_criteria and criteria_group.element_criteria.exclude_doc_ids:
        filters.append({
            "bool": {"must_not": {"terms": {"doc_id": criteria_group.element_criteria.exclude_doc_ids}}}
        })

    # Metadata filtering
    if criteria_group.metadata_criteria:
        metadata_filters = self._build_metadata_filters(criteria_group.metadata_criteria)
        filters.extend(metadata_filters)

    # Topic filtering
    if criteria_group.topic_criteria:
        topic_filters = self._build_topic_filters(criteria_group.topic_criteria)
        filters.extend(topic_filters)

    # Content length filtering
    if criteria_group.element_criteria:
        content_filters = self._build_content_length_filters(criteria_group.element_criteria)
        filters.extend(content_filters)

    # Handle sub-groups recursively
    sub_group_queries = []
    for sub_group in criteria_group.sub_groups:
        sub_query = self._build_elasticsearch_query(sub_group)
        sub_group_queries.append(sub_query)

    # Combine everything based on the operator
    if criteria_group.operator == LogicalOperator.AND:
        return self._combine_and(query_parts, filters, sub_group_queries)
    elif criteria_group.operator == LogicalOperator.OR:
        return self._combine_or(query_parts, filters, sub_group_queries)
    elif criteria_group.operator == LogicalOperator.NOT:
        return self._combine_not(query_parts, filters, sub_group_queries)
    else:
        # Default to AND
        return self._combine_and(query_parts, filters, sub_group_queries)


def _has_criteria(self, criteria_group) -> bool:
    """Check if a criteria group has any criteria."""
    return any([
        criteria_group.text_criteria,
        criteria_group.embedding_criteria,
        criteria_group.date_criteria,
        criteria_group.topic_criteria,
        criteria_group.metadata_criteria,
        criteria_group.element_criteria,
        criteria_group.sub_groups
    ])


def _build_text_query(self, text_criteria) -> Dict[str, Any]:
    """Build text search query from TextSearchCriteria."""
    # Determine search fields
    search_fields = text_criteria.search_fields if text_criteria.search_fields else [
        "content_preview^2"  # Boost content_preview by default
    ]

    # Add full_text field if indexing is enabled
    if self.index_full_text and "full_text" not in search_fields:
        search_fields.append("full_text")

    query = {
        "multi_match": {
            "query": text_criteria.query_text,
            "fields": search_fields,
            "type": "best_fields"
        }
    }

    # Apply boost factor if not default
    if text_criteria.boost_factor != 1.0:
        query = {
            "function_score": {
                "query": query,
                "boost": text_criteria.boost_factor
            }
        }

    return query


def _build_date_filter(self, date_criteria) -> Optional[Dict[str, Any]]:
    """Build date filter from DateSearchCriteria."""
    from .structured_search import DateRangeOperator
    from datetime import datetime, timedelta

    # This is a simplified implementation
    # In a full implementation, you'd need to join with the dates index
    # For now, we'll filter by element creation/update dates as a proxy

    current_time = datetime.now()

    if date_criteria.operator == DateRangeOperator.WITHIN:
        return {
            "range": {
                "created_at": {
                    "gte": date_criteria.start_date.timestamp(),
                    "lte": date_criteria.end_date.timestamp()
                }
            }
        }
    elif date_criteria.operator == DateRangeOperator.AFTER:
        return {
            "range": {
                "created_at": {
                    "gt": date_criteria.exact_date.timestamp()
                }
            }
        }
    elif date_criteria.operator == DateRangeOperator.BEFORE:
        return {
            "range": {
                "created_at": {
                    "lt": date_criteria.exact_date.timestamp()
                }
            }
        }
    elif date_criteria.operator == DateRangeOperator.RELATIVE_DAYS:
        cutoff_date = current_time - timedelta(days=date_criteria.relative_value)
        return {
            "range": {
                "created_at": {
                    "gte": cutoff_date.timestamp()
                }
            }
        }
    elif date_criteria.operator == DateRangeOperator.RELATIVE_MONTHS:
        # Approximate months as 30 days
        cutoff_date = current_time - timedelta(days=date_criteria.relative_value * 30)
        return {
            "range": {
                "created_at": {
                    "gte": cutoff_date.timestamp()
                }
            }
        }

    # For other date operators, return None for now
    # In a full implementation, you'd implement fiscal year, quarter, etc.
    return None


def _build_metadata_filters(self, metadata_criteria) -> List[Dict[str, Any]]:
    """Build metadata filters from MetadataSearchCriteria."""
    filters = []

    # Exact matches
    for key, value in metadata_criteria.exact_matches.items():
        if isinstance(value, list):
            filters.append({"terms": {f"metadata.{key}": value}})
        else:
            filters.append({"term": {f"metadata.{key}": value}})

    # LIKE patterns
    for key, pattern in metadata_criteria.like_patterns.items():
        wildcard_pattern = self._convert_like_to_wildcard(pattern)
        filters.append({"wildcard": {f"metadata.{key}": wildcard_pattern}})

    # Range filters
    for key, range_config in metadata_criteria.range_filters.items():
        filters.append({"range": {f"metadata.{key}": range_config}})

    # Existence filters
    for field in metadata_criteria.exists_filters:
        filters.append({"exists": {"field": f"metadata.{field}"}})

    return filters


def _build_topic_filters(self, topic_criteria) -> List[Dict[str, Any]]:
    """Build topic filters from TopicSearchCriteria."""
    filters = []

    # Include topics
    if topic_criteria.include_topics:
        if topic_criteria.require_all_included:
            # All topics must match (AND)
            for pattern in topic_criteria.include_topics:
                wildcard_pattern = self._convert_like_to_wildcard(pattern)
                filters.append({"wildcard": {"topics": wildcard_pattern}})
        else:
            # Any topic can match (OR)
            should_queries = []
            for pattern in topic_criteria.include_topics:
                wildcard_pattern = self._convert_like_to_wildcard(pattern)
                should_queries.append({"wildcard": {"topics": wildcard_pattern}})

            if should_queries:
                filters.append({"bool": {"should": should_queries, "minimum_should_match": 1}})

    # Exclude topics
    for pattern in topic_criteria.exclude_topics:
        wildcard_pattern = self._convert_like_to_wildcard(pattern)
        filters.append({
            "bool": {"must_not": {"wildcard": {"topics": wildcard_pattern}}}
        })

    return filters


def _build_content_length_filters(self, element_criteria) -> List[Dict[str, Any]]:
    """Build content length filters from ElementSearchCriteria."""
    filters = []

    # This would require storing content length as a field
    # For now, we'll use a script-based approach on content_preview
    range_config = {}

    if element_criteria.content_length_min is not None:
        range_config["gte"] = element_criteria.content_length_min

    if element_criteria.content_length_max is not None:
        range_config["lte"] = element_criteria.content_length_max

    if range_config:
        # Use a script to calculate content length
        filters.append({
            "script": {
                "script": {
                    "source": "doc['content_preview'].value.length()",
                    "params": range_config
                }
            }
        })

    return filters


def _combine_and(self, query_parts: List[Dict], filters: List[Dict],
                 sub_queries: List[Dict]) -> Dict[str, Any]:
    """Combine query parts with AND logic."""
    must_clauses = query_parts + sub_queries

    if not must_clauses and not filters:
        return {"match_all": {}}

    if not must_clauses:
        # Only filters
        return {"bool": {"filter": filters}}

    if not filters:
        # Only queries
        if len(must_clauses) == 1:
            return must_clauses[0]
        return {"bool": {"must": must_clauses}}

    # Both queries and filters
    return {"bool": {"must": must_clauses, "filter": filters}}


def _combine_or(self, query_parts: List[Dict], filters: List[Dict],
                sub_queries: List[Dict]) -> Dict[str, Any]:
    """Combine query parts with OR logic."""
    should_clauses = query_parts + filters + sub_queries

    if not should_clauses:
        return {"match_all": {}}

    return {"bool": {"should": should_clauses, "minimum_should_match": 1}}


def _combine_not(self, query_parts: List[Dict], filters: List[Dict],
                 sub_queries: List[Dict]) -> Dict[str, Any]:
    """Combine query parts with NOT logic."""
    must_not_clauses = query_parts + filters + sub_queries

    if not must_not_clauses:
        return {"match_all": {}}

    return {"bool": {"must_not": must_not_clauses}}


def validate_query_support(self, query: StructuredSearchQuery) -> List:
    """Validate that this backend can execute the given query."""
    from .structured_search import validate_query_capabilities

    backend_capabilities = self.get_backend_capabilities()
    return validate_query_capabilities(query, backend_capabilities)
