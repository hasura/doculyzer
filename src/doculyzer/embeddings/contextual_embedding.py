from typing import Optional, List, Dict, Any

from .base import EmbeddingGenerator
from ..adapter import create_content_resolver
from ..config import Config


class ContextualEmbeddingGenerator(EmbeddingGenerator):
    """
    Embedding generator that includes context for better semantic understanding.

    This generator creates embeddings that include context from surrounding elements,
    creating overlapping context windows to improve semantic search quality.
    """

    def __init__(self,
                 _config: Config,
                 base_generator: EmbeddingGenerator,
                 window_size: int = 3,
                 overlap_size: int = 1,
                 predecessor_count: int = 1,
                 successor_count: int = 1,
                 ancestor_depth: int = 1,
                 child_count: int = 1):
        """
        Initialize the contextual embedding generator.

        Args:
            base_generator: Base embedding generator
            window_size: Number of elements in context window
            overlap_size: Number of elements to overlap between windows
            predecessor_count: Number of preceding elements to include
            successor_count: Number of following elements to include
            ancestor_depth: Number of ancestral levels to include
        """
        super().__init__(_config)
        self.base_generator = base_generator
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.predecessor_count = predecessor_count
        self.successor_count = successor_count
        self.ancestor_depth = ancestor_depth
        self.child_count = child_count

    def generate(self, text: str, context: Optional[List[str]] = None) -> List[float]:
        """
        Generate embedding for text with context.

        Args:
            text: Main text to embed
            context: List of context texts (optional)

        Returns:
            Vector embedding
        """
        if not context:
            # No context, just generate embedding for text
            return self.base_generator.generate(text)

        # Combine text with context
        combined_text = self._combine_text_with_context(text, context)

        # Generate embedding for combined text
        return self.base_generator.generate(combined_text)

    def generate_batch(self, texts: List[str], contexts: Optional[List[List[str]]] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with contexts.

        Args:
            texts: List of input texts
            contexts: List of context lists (optional)

        Returns:
            List of vector embeddings
        """
        if not contexts:
            # No contexts, just generate embeddings for texts
            return self.base_generator.generate_batch(texts)

        # Ensure contexts list has same length as texts
        if len(contexts) != len(texts):
            raise ValueError("Length of contexts must match length of texts")

        # Combine texts with contexts
        combined_texts = [
            self._combine_text_with_context(text, context)
            for text, context in zip(texts, contexts)
        ]

        # Generate embeddings for combined texts
        return self.base_generator.generate_batch(combined_texts)

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.base_generator.get_dimensions()

    def get_model_name(self) -> str:
        """Get embedding model name."""
        return f"contextual-{self.base_generator.get_model_name()}"

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.base_generator.clear_cache()

    @staticmethod
    def _combine_text_with_context(text: str, context: List[str]) -> str:
        """
        Combine text with context texts.

        Args:
            text: Main text
            context: List of context texts

        Returns:
            Combined text
        """
        # Use a simple approach: concatenate with separators
        combined = text

        for ctx in context:
            combined += f"\n\n{ctx}"

        return combined

    @staticmethod
    def is_number(value: str) -> bool:
        """
        Check if the given string represents an integer or a float.

        Args:
            value: The string to check.
        Returns:
            True if the string is an int or a float, False otherwise.
        """
        try:
            float(value)  # Try converting to float (handles integers too)
            return True
        except ValueError:
            return False

    def generate_from_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Generate contextual embeddings for document elements, with size handling:
        - Skip root elements that exceed the size threshold
        - Truncate non-root elements that exceed the size threshold
        """
        # Build element hierarchy
        hierarchy = self._build_element_hierarchy(elements)
        resolver = create_content_resolver(self._config)

        # Define maximum content size for effective embedding (approximate word count)
        max_words_for_embedding = 500

        # Generate embeddings with context
        embeddings = {}

        for element in elements:
            element_pk = element["element_pk"]

            # Get full text content for all elements using the resolver
            content = resolver.resolve_content(element.get('content_location'), text=True)

            # Skip if no meaningful content
            if not content and not self.is_number(content):
                continue

            # Check content length
            word_count = len(content.split())
            if word_count > max_words_for_embedding:
                # For root elements, skip entirely
                if element["element_type"] == "root":
                    continue

                # For non-root elements, truncate to threshold
                content = " ".join(content.split()[:max_words_for_embedding])

            # Get context elements
            context_elements = self._get_context_elements(element, elements, hierarchy)

            # Get context contents using the resolver for text
            context_contents = []
            for ctx_element in context_elements:
                ctx_content = resolver.resolve_content(ctx_element.get('content_location'), text=True)
                if ctx_content and not self.is_number(ctx_content):
                    # Also check size of context elements and truncate if needed
                    ctx_words = len(ctx_content.split())
                    if ctx_words > max_words_for_embedding:
                        ctx_content = " ".join(ctx_content.split()[:max_words_for_embedding])
                    context_contents.append(ctx_content)

            # Generate embedding with context
            embedding = self.generate(content, context_contents)
            embeddings[element_pk] = embedding

        return embeddings

    @staticmethod
    def _build_element_hierarchy(elements: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Build element hierarchy for context lookup.

        Args:
            elements: List of document elements

        Returns:
            Dictionary mapping parent_id to list of child element_ids
        """
        hierarchy = {}

        for element in elements:
            parent_id = element.get("parent_id")
            element_id = element["element_id"]

            if parent_id:
                if parent_id not in hierarchy:
                    hierarchy[parent_id] = []

                hierarchy[parent_id].append(element_id)

        return hierarchy

    def _get_context_elements(self, element: Dict[str, Any],
                              all_elements: List[Dict[str, Any]],
                              hierarchy: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Get context elements for an element.

        This includes:
        - Ancestors up to configured depth (skipping those with blank content)
        - Meaningful predecessors (elements that come before in document order)
        - Meaningful successors (elements that come after in document order)
        - A limited number of meaningful children (directly nested elements)

        Args:
            element: Element to get context for
            all_elements: List of all elements
            hierarchy: Element hierarchy

        Returns:
            List of context elements
        """
        element_id = element["element_id"]
        context_ids = set()

        # Build a mapping from element_id to element for quicker lookups
        id_to_element = {e["element_id"]: e for e in all_elements}

        # Add ancestors up to configured depth
        current_element = element
        current_depth = 0
        ancestors_added = 0

        while ancestors_added < self.ancestor_depth:
            parent_id = current_element.get("parent_id")
            if not parent_id:
                break  # No more ancestors

            # Find parent element to continue up the hierarchy
            parent_element = id_to_element.get(parent_id)
            if not parent_element:
                break  # Parent not found

            # Only include parent if it has content and is not an empty container
            if (parent_element.get("content_preview") and
                    parent_element["element_type"] != "root" and
                    not self._is_empty_container(parent_element)):
                context_ids.add(parent_id)
                ancestors_added += 1

            # Move up to the next level, even if we skipped this parent
            current_element = parent_element
            current_depth += 1

            # Safety check - don't go too far up (avoid infinite loops)
            if current_depth > 10:  # Arbitrary depth limit
                break

        # Find meaningful predecessors and successors
        current_index = -1
        for i, e in enumerate(all_elements):
            if e["element_id"] == element_id:
                current_index = i
                break

        if current_index >= 0:
            # Get meaningful predecessors (elements that come before)
            pred_count = 0
            i = current_index - 1

            while i >= 0 and pred_count < self.predecessor_count:
                pred_element = all_elements[i]

                # Skip elements that:
                # 1. Are root elements
                # 2. Don't have content (empty content_preview)
                # 3. Are just container elements
                if (pred_element["element_type"] != "root" and
                        pred_element.get("content_preview") and
                        not self._is_empty_container(pred_element)):
                    context_ids.add(pred_element["element_id"])
                    pred_count += 1

                i -= 1

            # Get meaningful successors (elements that come after)
            succ_count = 0
            i = current_index + 1

            while i < len(all_elements) and succ_count < self.successor_count:
                succ_element = all_elements[i]

                # Same filtering as for predecessors
                if (succ_element["element_type"] != "root" and
                        succ_element.get("content_preview") and
                        not self._is_empty_container(succ_element)):
                    context_ids.add(succ_element["element_id"])
                    succ_count += 1

                i += 1

        # Add a limited number of meaningful children
        if element_id in hierarchy and self.child_count > 0:
            children_added = 0

            for child_id in hierarchy[element_id]:
                # Apply same filtering as for predecessors/successors
                child_element = id_to_element.get(child_id)
                if (child_element and
                        child_element["element_type"] != "root" and
                        child_element.get("content_preview") and
                        not self._is_empty_container(child_element)):
                    context_ids.add(child_id)
                    children_added += 1
                    if children_added >= self.child_count:
                        break

        # Convert IDs to elements
        context_elements = []
        for context_id in context_ids:
            if context_id in id_to_element:
                context_elements.append(id_to_element[context_id])

        return context_elements

    @staticmethod
    def _is_empty_container(element: Dict[str, Any]) -> bool:
        """
        Check if an element is just an empty container (like a div with no text).

        Args:
            element: The element to check

        Returns:
            True if the element is an empty container, False otherwise
        """
        # Consider these element types as potential empty containers
        container_types = ["div", "span", "article", "section", "nav", "aside"]

        # Check if it's a container type
        if element["element_type"] in container_types:
            # Check if it has no meaningful content
            content = element.get("content_preview", "").strip()
            return not content or content in ["", "..."]

        return False
