"""ChromaDB Handshake for Chonkie."""

from typing import Any, Callable, Dict, List, Optional, Sequence

from chonkie.types import Chunk

from .base_writer import BaseVectorWriter

try:
    import chromadb
    from chromadb import Collection
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    raise ImportError(
        "chromadb is not installed. Please install it to use ChromaHandshake: "
        "`pip install 'chonkie[handshakes]'` or `pip install chromadb`"
    )


class ChromaHandshake(BaseVectorWriter):
    """Handshake for writing chunks to ChromaDB."""

    def __init__(
        self,
        collection_name: str,
        client: Optional[chromadb.ClientAPI] = None,
        client_settings: Optional[Settings] = None,
        id_generator: Optional[Callable[..., str]] = None,
        metadata_fields: Optional[List[str]] = None,
        create_collection_if_not_exists: bool = False,
        collection_metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[
            Any
        ] = embedding_functions.DefaultEmbeddingFunction(),  # Use Chroma's default if none provided by user chunks
    ) -> None:
        """Initialize the ChromaHandshake.

        Args:
            collection_name (str): The name of the ChromaDB collection.
            client (Optional[chromadb.ClientAPI]): An existing ChromaDB client instance.
                If None, a new client will be created based on client_settings.
            client_settings (Optional[Settings]): Settings for creating a new ChromaDB client.
                Used only if `client` is None. Defaults to ephemeral (in-memory) client.
            id_generator (Optional[Callable[..., str]]): Function to generate document IDs.
            metadata_fields (Optional[List[str]]): Specific chunk attributes for metadata.
            create_collection_if_not_exists (bool): Create the collection if it doesn't exist.
            collection_metadata (Optional[Dict[str, Any]]): Metadata for collection creation.
            embedding_function (Optional[Any]): Embedding function for Chroma collection.
                Defaults to Chroma's default. Set to None if you are always providing embeddings
                with your chunks and don't want Chroma to generate any.

        """
        super().__init__(
            id_generator=id_generator, metadata_fields=metadata_fields
        )

        if client:
            self.client = client
        else:
            self.client = chromadb.Client(
                client_settings or Settings()
            )  # Default ephemeral (in-memory) client

        self.collection: Collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                metadata=collection_metadata,
                embedding_function=embedding_function,  # Pass the embedding function here
            )
            if create_collection_if_not_exists
            else self.client.get_collection(name=collection_name)
        )

    def write(self, chunk: Chunk, **kwargs: Any) -> None:
        """Write a single chunk to the ChromaDB collection.

        Args:
            chunk (Chunk): The chunk to write.
            **kwargs: Passed directly to Chroma's `collection.add` method.

        """
        self.write_batch([chunk], **kwargs)

    def write_batch(self, chunks: Sequence[Chunk], **kwargs: Any) -> None:
        """Write a batch of chunks to the ChromaDB collection.

        Chunks without an 'embedding' attribute will rely on the collection's
        embedding function (if configured).

        Args:
            chunks (Sequence[Chunk]): The sequence of chunks to write.
            **kwargs: Passed directly to Chroma's `collection.add` method.

        """
        # Chroma can generate embeddings, so we don't strictly require them here.
        # However, if provided, they should be used.
        ids, documents, embeddings, metadatas = self._prepare_data(
            chunks,
            embedding_required=False,  # Chroma can handle missing embeddings
            extra_metadata=kwargs.pop("extra_metadata", None),
        )

        # Filter out None embeddings if Chroma is expected to generate them
        # If an embedding function is configured in Chroma, it expects either all embeddings or none.
        # Pass embeddings only if *all* chunks have them.
        has_embeddings = all(e is not None for e in embeddings)
        embeddings_to_pass = embeddings if has_embeddings else None

        # Chroma expects Optional[List[Dict]] for metadata
        metadatas_to_pass = [m if m else None for m in metadatas]  # type: ignore

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings_to_pass,  # Pass embeddings only if all chunks have them
            metadatas=metadatas_to_pass,  # type: ignore
            **kwargs,
        )
