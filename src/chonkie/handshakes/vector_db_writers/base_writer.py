"""Base class for vector database handshakes."""

import uuid
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from chonkie.types import Chunk


def _generate_default_id(*args: Any) -> str:
    """Generate a default UUID."""
    return str(uuid.uuid4())


class BaseVectorWriter(ABC):
    """Abstract base class for vector database handshakes."""

    def __init__(
        self,
        id_generator: Optional[Callable[..., str]] = None,
        metadata_fields: Union[Literal["all"], List[str], None] = None,
    ) -> None:
        """Initialize the BaseVectorWriter.

        Args:
            id_generator (Optional[Callable[..., str]]): A function to generate document IDs.
                If None, defaults to UUID generation. The function will receive the chunk object
                and its index in the batch as arguments.
            metadata_fields (Union[Literal["all"], List[str], None]): Controls which chunk attributes
                are included as metadata.
                - If "all", all attributes except 'text' and 'embedding' are included.
                - If a list of strings, only those specific attributes are included.
                - If None (default), no chunk attributes are included as metadata (only extra_metadata).

        """
        self.id_generator = id_generator or _generate_default_id
        self.metadata_fields = metadata_fields

    def _prepare_data(
        self,
        chunks: Sequence[Chunk],
        embedding_required: bool = True,  # Add embedding_required parameter
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[str], List[Optional[List[float]]], List[Dict[str, Any]]]:
        """Prepare chunks for insertion into a vector database.

        Args:
            chunks (Sequence[Chunk]): The chunks to prepare.
            embedding_required (bool): Whether to raise an error if a chunk lacks an embedding. Defaults to True.
            extra_metadata (Optional[Dict[str, Any]]): Extra static metadata to add to each chunk.

        Returns:
            Tuple[List[str], List[str], List[Optional[List[float]]], List[Dict[str, Any]]]:
                A tuple containing lists of IDs, documents, embeddings, and metadata.

        Raises:
            ValueError: If embedding_required is True and a chunk lacks an embedding.
            AttributeError: If a specified metadata field does not exist on a chunk when metadata_fields is a list.

        """
        ids: List[str] = []
        documents: List[str] = []
        embeddings: List[Optional[List[float]]] = []
        metadatas: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            # Generate ID using the chunk itself if it has an 'id' attribute, otherwise generate
            chunk_id = getattr(chunk, "id", None)
            if chunk_id is None:
                ids.append(self.id_generator(chunk, i))
            else:
                # Ensure the provided ID is a string
                ids.append(str(chunk_id))

            # Get document text
            documents.append(chunk.text)

            # Get embedding
            embedding = getattr(chunk, "embedding", None)
            if embedding_required and embedding is None:  # Use embedding_required flag
                raise ValueError(
                    f"Chunk at index {i} (ID: {ids[-1]}) is missing the 'embedding' attribute, "
                    "which is required when embedding_required is True."
                )
            # Assuming numpy array or similar, convert to list if not None
            embeddings.append(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)

            # Prepare metadata based on self.metadata_fields
            metadata: Dict[str, Any] = {}
            if self.metadata_fields == "all":
                # Include all except text and embedding
                for field, value in chunk.__dict__.items():
                    if field not in ["text", "embedding"]:
                        # Handle potentially complex types like tree-sitter nodes
                        if hasattr(value, '__dict__') or isinstance(value, (list, tuple)):
                            # Attempt a simple representation or skip
                            try:
                                metadata[field] = repr(value)  # Basic representation
                            except Exception:
                                metadata[field] = f"<{type(value).__name__} object (unserializable)>"
                        else:
                            metadata[field] = value
            elif isinstance(self.metadata_fields, list):
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
            # If self.metadata_fields is None, metadata remains empty {} by default

            # Add extra metadata if provided, regardless of metadata_fields setting
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

