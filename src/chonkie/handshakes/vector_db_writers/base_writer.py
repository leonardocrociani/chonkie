"""Base class for vector database handshakes."""

import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from chonkie.types import Chunk


def _generate_default_id(*args: Any) -> str:
    """Generate a default UUID."""
    return str(uuid.uuid4())


class BaseVectorWriter(ABC):
    """Abstract base class for vector database handshakes."""

    def __init__(
        self,
        id_generator: Optional[Callable[..., str]] = None,
        metadata_fields: Optional[List[str]] = None,
    ) -> None:
        """Initialize the BaseVectorWriter.

        Args:
            id_generator (Optional[Callable[..., str]]): A function to generate document IDs.
                If None, defaults to UUID generation. The function will receive the chunk object
                and its index in the batch as arguments.
            metadata_fields (Optional[List[str]]): A specific list of chunk attributes
                to include as metadata. If None, all attributes except 'text' and 'embedding'
                will be included.

        """
        self.id_generator = id_generator or _generate_default_id
        self.metadata_fields = metadata_fields

    def _prepare_data(
        self,
        chunks: Sequence[Chunk],
        use_chunk_embeddings: bool = True,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[str], List[Optional[List[float]]], List[Dict[str, Any]]]:
        """Prepare chunks for insertion into a vector database.

        Args:
            chunks (Sequence[Chunk]): The chunks to prepare.
            use_chunk_embeddings (bool): Whether to use the value associate with `chunk.embeddings` as embeddings.
            extra_metadata (Optional[Dict[str, Any]]): Extra static metadata to add to each chunk.

        Returns:
            Tuple[List[str], List[str], List[Optional[List[float]]], List[Dict[str, Any]]]:
                A tuple containing lists of IDs, documents, embeddings, and metadata.

        Raises:
            ValueError: If use_chunk_embeddings is True and a chunk lacks an embedding.
            AttributeError: If a specified metadata field does not exist on a chunk.

        """
        ids: List[str] = []
        documents: List[str] = []
        embeddings: List[Optional[List[float]]] = []
        metadatas: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            # Generate ID
            ids.append(self.id_generator(chunk, i))

            # Get document text
            documents.append(chunk.text)

            # Get embedding
            embedding = getattr(chunk, "embedding", None)
            if use_chunk_embeddings and embedding is None:
                raise ValueError(
                    f"Chunk at index {i} is missing the 'embedding' attribute, "
                    "which is required for this handshake."
                )
            embeddings.append(embedding.tolist() if embedding is not None else None) # Assuming numpy array, convert

            # Prepare metadata
            metadata: Dict[str, Any] = {}
            if self.metadata_fields is None:
                # Default: include all except text and embedding
                for field, value in chunk.__dict__.items():
                    if field not in ["text", "embedding"]:
                        # Handle potentially complex types like tree-sitter nodes
                        if hasattr(value, '__dict__') or isinstance(value, (list, tuple)):
                             # Attempt a simple representation or skip
                             try:
                                 metadata[field] = repr(value) # Basic representation
                             except Exception:
                                 metadata[field] = f"<{type(value).__name__} object (unserializable)>"
                        else:
                            metadata[field] = value
            else:
                # Include only specified fields
                for field in self.metadata_fields:
                    if not hasattr(chunk, field):
                        raise AttributeError(
                            f"Chunk at index {i} does not have the specified metadata field '{field}'."
                        )
                    value = getattr(chunk, field)
                    # Handle potentially complex types
                    if hasattr(value, '__dict__') or isinstance(value, (list, tuple)):
                         try:
                             metadata[field] = repr(value)
                         except Exception:
                             metadata[field] = f"<{type(value).__name__} object (unserializable)>"
                    else:
                        metadata[field] = value


            # Add extra metadata if provided
            if extra_metadata:
                metadata.update(extra_metadata)

            metadatas.append(metadata)

        return ids, documents, embeddings, metadatas

    @abstractmethod
    def write(self, chunk: Chunk, **kwargs: Any) -> Any:
        """Write a single chunk to the vector database.

        Args:
            chunk (Chunk): The chunk to write.
            **kwargs: Additional keyword arguments for the specific database write operation.

        Returns:
            Any: The result from the database write operation.

        """
        raise NotImplementedError

    @abstractmethod
    def write_batch(self, chunks: Sequence[Chunk], **kwargs: Any) -> Any:
        """Write a batch of chunks to the vector database.

        Args:
            chunks (Sequence[Chunk]): The sequence of chunks to write.
            **kwargs: Additional keyword arguments for the specific database write operation.

        Returns:
            Any: The result from the database batch write operation.

        """
        raise NotImplementedError

    async def awrite(self, chunk: Chunk, **kwargs: Any) -> Any:
        """Asynchronously write a single chunk to the vector database.

        Args:
            chunk (Chunk): The chunk to write.
            **kwargs: Additional keyword arguments for the specific database write operation.

        Returns:
            Any: The result from the database write operation.

        """
        raise NotImplementedError("Async write not implemented for this handshake.")

    async def awrite_batch(self, chunks: Sequence[Chunk], **kwargs: Any) -> Any:
        """Asynchronously write a batch of chunks to the vector database.

        Args:
            chunks (Sequence[Chunk]): The sequence of chunks to write.
            **kwargs: Additional keyword arguments for the specific database write operation.

        Returns:
            Any: The result from the database batch write operation.

        """
        raise NotImplementedError("Async batch write not implemented for this handshake.")

    def __call__(self, chunks: Union[Chunk, Sequence[Chunk]], **kwargs: Any) -> Any:
        """Write chunks using the default batch method when the instance is called.

        Args:
            chunks (Union[Chunk, Sequence[Chunk]]): A single chunk or a sequence of chunks.
            **kwargs: Additional keyword arguments for the write operation.

        Returns:
            Any: The result from the database write operation.

        """
        if isinstance(chunks, Chunk):
            return self.write(chunks, **kwargs)
        elif isinstance(chunks, Sequence):
            return self.write_batch(chunks, **kwargs)
        else:
            raise TypeError("Input must be a Chunk or a sequence of Chunks.")

