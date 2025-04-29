from abc import ABC, abstractmethod


class ContentResolver(ABC):
    """Abstract base class for content resolvers."""

    @abstractmethod
    def resolve_content(self, content_location: str) -> str:
        """
        Resolve a content pointer to actual content.

        Args:
            content_location: Content location pointer

        Returns:
            Resolved content as string
        """
        pass

    @abstractmethod
    def supports_location(self, content_location: str) -> bool:
        """
        Check if this resolver supports a content location.

        Args:
            content_location: Content location pointer

        Returns:
            True if supported, False otherwise
        """
        pass

    @abstractmethod
    def get_document_binary(self, content_location: str) -> bytes:
        """
        Get the containing document as a binary blob.

        Args:
            content_location: Content location pointer

        Returns:
            Document binary content

        Raises:
            ValueError: If document cannot be retrieved
        """
        pass
