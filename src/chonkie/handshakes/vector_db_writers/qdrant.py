"""Qdrant Handshake for writing chunks to Qdrant."""

from typing import Any, Callable, List, Optional, Sequence

from chonkie.types import Chunk

from .base_writer import BaseVectorWriter

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import Distance, VectorParams
except ImportError:
    raise ImportError(
        "qdrant-client is not installed. Please install it to use QdrantHandshake: "
        "`pip install 'chonkie[handshakes]'` or `pip install qdrant-client`"
    )


class QdrantHandshake(BaseVectorWriter):
    """Handshake for writing chunks to Qdrant."""

    def __init__(
        self,
        collection_name: str,
        client: Optional[QdrantClient] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_size: Optional[int] = None, # Required if creating collection
        distance: Distance = Distance.COSINE, # Required if creating collection
        id_generator: Optional[Callable[..., str]] = None,
        metadata_fields: Optional[List[str]] = None,
        create_collection_if_not_exists: bool = True,
        **kwargs: Any # For QdrantClient init (e.g., port, grpc_port, prefer_grpc)
    ) -> None:
        """Initialize the QdrantHandshake.

        Args:
            collection_name (str): The name of the Qdrant collection.
            client (Optional[QdrantClient]): An existing Qdrant client instance.
                If None, a new client will be created based on url, api_key, etc.
            url (Optional[str]): URL for the Qdrant instance (e.g., "http://localhost:6333").
            api_key (Optional[str]): API key for Qdrant Cloud.
            vector_size (Optional[int]): Dimension of vectors. Required if creating collection.
            distance (Distance): Distance metric for vectors. Required if creating collection.
            id_generator (Optional[Callable[..., str]]): Function to generate document IDs.
            metadata_fields (Optional[List[str]]): Specific chunk attributes for metadata.
            create_collection_if_not_exists (bool): Create the collection if it doesn't exist.
            **kwargs: Additional arguments passed to QdrantClient constructor if client is None.

        """
        super().__init__(id_generator=id_generator, metadata_fields=metadata_fields)
        self.collection_name = collection_name

        if client:
            self.client = client
        else:
            self.client = QdrantClient(url=url, api_key=api_key, **kwargs)

        if create_collection_if_not_exists:
            try:
                self.client.get_collection(collection_name=self.collection_name)
            except Exception: # Broad exception for collection not found or connection issues
                if vector_size is None:
                    raise ValueError("vector_size must be provided when creating a new Qdrant collection.")
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance),
                )
        else:
             # Ensure collection exists if not creating
             try:
                 self.client.get_collection(collection_name=self.collection_name)
             except Exception as e:
                 raise ValueError(f"Collection '{self.collection_name}' not found and create_collection_if_not_exists is False.") from e


    def _prepare_qdrant_points(
        self, chunks: Sequence[Chunk], **kwargs: Any
    ) -> List[models.PointStruct]:
        """Prepare Qdrant PointStruct objects from chunks."""
        ids, _, embeddings, metadatas = self._prepare_data(
            chunks,
            embedding_required=True, # Qdrant requires embeddings
            extra_metadata=kwargs.pop("extra_metadata", None)
        )

        points = []
        for i in range(len(ids)):
            if embeddings[i] is None:
                 # This should theoretically not happen due to embedding_required=True
                 raise ValueError(f"Chunk at index {i} is missing required embedding for Qdrant.")
            # Qdrant uses 'payload' for metadata
            points.append(
                models.PointStruct(
                    id=ids[i],
                    vector=embeddings[i], # type: ignore # Already checked for None
                    payload=metadatas[i] if metadatas[i] else None # Qdrant expects Optional[dict]
                )
            )
        return points

    def write(self, chunk: Chunk, wait: bool = True, **kwargs: Any) -> Any:
        """Write a single chunk to the Qdrant collection.

        Args:
            chunk (Chunk): The chunk to write.
            wait (bool): Wait for the operation to complete.
            **kwargs: Additional arguments for preparing Qdrant points (e.g., extra_metadata).

        Returns:
            Any: The result from the Qdrant upsert operation.

        """
        return self.write_batch([chunk], wait=wait, **kwargs)

    def write_batch(self, chunks: Sequence[Chunk], wait: bool = True, **kwargs: Any) -> Any:
        """Write a batch of chunks to the Qdrant collection.

        Args:
            chunks (Sequence[Chunk]): The sequence of chunks to write.
            wait (bool): Wait for the operation to complete.
            **kwargs: Additional arguments for preparing Qdrant points (e.g., extra_metadata).

        Returns:
            Any: The result from the Qdrant upsert operation.

        """
        points = self._prepare_qdrant_points(chunks, **kwargs)
        if not points:
            return None # Nothing to write

        return self.client.upsert(
            collection_name=self.collection_name, points=points, wait=wait
        )

    async def awrite(self, chunk: Chunk, **kwargs: Any) -> Any:
        """Asynchronously write a single chunk to the Qdrant collection.

        Args:
            chunk (Chunk): The chunk to write.
            **kwargs: Additional arguments for preparing Qdrant points (e.g., extra_metadata).

        Returns:
            Any: The result from the Qdrant upsert operation.

        """
        return await self.awrite_batch([chunk], **kwargs)

    async def awrite_batch(self, chunks: Sequence[Chunk], **kwargs: Any) -> Any:
        """Asynchronously write a batch of chunks to the Qdrant collection.

        Args:
            chunks (Sequence[Chunk]): The sequence of chunks to write.
            **kwargs: Additional arguments for preparing Qdrant points (e.g., extra_metadata).

        Returns:
            Any: The result from the Qdrant upsert operation.

        """
        points = self._prepare_qdrant_points(chunks, **kwargs)
        if not points:
            return None # Nothing to write

        # Qdrant client uses the same method for sync/async if initialized correctly
        # (e.g., QdrantClient(prefer_grpc=True) often enables async under the hood)
        # Or use specific async methods if available/needed depending on client setup.
        # Assuming the standard client handles async context correctly:
        return await self.client.upsert(
             collection_name=self.collection_name, points=points, wait=False # Typically don't wait in async
        )

