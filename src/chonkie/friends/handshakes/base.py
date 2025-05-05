"""Base class for Handshakes."""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Sequence,
    Union,
)

from chonkie.types import Chunk

# TODO: Move this to inside the BaseHandshake class
# Why is this even outside the class?
# def _generate_default_id(*args: Any) -> str:
#     """Generate a default UUID."""
#     return str(uuid.uuid4())


class BaseHandshake(ABC):
    """Abstract base class for Handshakes."""

    @abstractmethod
    def write(self, chunk: Chunk) -> Any:
        """Write a single chunk to the vector database.

        Args:
            chunk (Chunk): The chunk to write.

        Returns:
            Any: The result from the database write operation.

        """
        raise NotImplementedError

    def write_batch(self, chunks: Sequence[Chunk]) -> Any:
        """Write a batch of chunks to the vector database.

        Args:
            chunks (Sequence[Chunk]): The sequence of chunks to write.

        Returns:
            Any: The result from the database batch write operation.

        """
        raise NotImplementedError

    async def awrite(self, chunk: Chunk) -> Any:
        """Asynchronously write a single chunk to the vector database.

        Args:
            chunk (Chunk): The chunk to write.

        Returns:
            Any: The result from the database write operation.

        """
        raise NotImplementedError("Async write not implemented for this handshake.")

    async def awrite_batch(self, chunks: Sequence[Chunk]) -> Any:
        """Asynchronously write a batch of chunks to the vector database.

        Args:
            chunks (Sequence[Chunk]): The sequence of chunks to write.

        Returns:
            Any: The result from the database batch write operation.

        """
        raise NotImplementedError("Async batch write not implemented for this handshake.")

    def __call__(self, chunks: Union[Chunk, Sequence[Chunk]]) -> Any:
        """Write chunks using the default batch method when the instance is called.

        Args:
            chunks (Union[Chunk, Sequence[Chunk]]): A single chunk or a sequence of chunks.

        Returns:
            Any: The result from the database write operation.

        """
        if isinstance(chunks, Chunk):
            return self.write(chunks)
        elif isinstance(chunks, Sequence):
            return self.write_batch(chunks)
        else:
            raise TypeError("Input must be a Chunk or a sequence of Chunks.")

