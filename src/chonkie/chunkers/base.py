"""Base Class for All Chunkers."""
from abc import ABC, abstractmethod

class BaseChunker(ABC):
    """Base class for all chunkers."""

    def __init__(self, **kwargs):
        """Initialize the chunker with any necessary parameters."""
        self.params = kwargs

    @abstractmethod
    def chunk(self, text: str) -> list:
        """Chunk the given text into smaller pieces."""
        pass

    @abstractmethod
    def chunk(self, text: str) -> list:
        """Chunk the given text.

        Args:
            text (str): The text to chunk.
        Returns:
            List of Chunks containing the chunked text and other metadata.

        """
        pass

