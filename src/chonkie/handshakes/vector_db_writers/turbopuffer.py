"""Turbopuffer Handshake for writing chunks to Turbopuffer."""

import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from chonkie.types import Chunk

from .base_writer import BaseVectorWriter

try:
    import turbopuffer as tpuf
except ImportError:
    raise ImportError(
        "turbopuffer is not installed. Please install it to use TurbopufferHandshake: "
        "`pip install 'chonkie[handshakes]'` or `pip install turbopuffer-py`"
    )


class TurbopufferHandshake(BaseVectorWriter):
    """Handshake for writing chunks to Turbopuffer."""

    def __init__(
        self,
        namespace: str,
        api_key: Optional[str] = None,
        id_generator: Optional[Callable[..., str]] = None,
        metadata_fields: Optional[List[str]] = None,
    ) -> None:
        """Initialize the TurbopufferHandshake.

        Args:
            namespace (str): The name of the Turbopuffer namespace.
            api_key (Optional[str]): Turbopuffer API key. Reads from TURBOPUFFER_API_KEY
                environment variable if not provided.
            id_generator (Optional[Callable[..., str]]): Function to generate document IDs.
            metadata_fields (Optional[List[str]]): Specific chunk attributes for metadata.

        """
        super().__init__(id_generator=id_generator, metadata_fields=metadata_fields)

        self.api_key = api_key or os.environ.get("TURBOPUFFER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Turbopuffer API key not provided and TURBOPUFFER_API_KEY environment variable is not set."
            )

        # Turbopuffer client is namespace-based
        self.namespace = tpuf.Namespace(namespace, api_key=self.api_key)

    def _prepare_tpuf_data(
        self, chunks: Sequence[Chunk], **kwargs: Any
    ) -> Dict[str, Any]:
        """Prepare data in Turbopuffer's expected dictionary format."""
        ids, texts, embeddings, metadatas = self._prepare_data(
            chunks=chunks,
            embedding_required=True, # Turbopuffer requires embeddings
            extra_metadata=kwargs.pop("extra_metadata", None)
        )

        if any(e is None for e in embeddings):
             # This should theoretically not happen due to embedding_required=True
             raise ValueError("One or more chunks are missing required embeddings for Turbopuffer.")

        # Turbopuffer expects dict format: {'ids': [...], 'vectors': [...], 'attributes': {...}}
        data = {
            "ids": ids,
            "vectors": embeddings,
            "texts": texts,
            "attributes": {},
        }

        # Collect attributes - Turbopuffer expects attributes as {'attr_name': [val1, val2,...]}
        if metadatas:
            all_keys = set(key for meta_dict in metadatas if meta_dict for key in meta_dict)
            for key in all_keys:
                data["attributes"][key] = [
                    meta.get(key) if meta else None for meta in metadatas # type: ignore
                ]

        return data


    def write(self, chunk: Chunk, **kwargs: Any) -> Any:
        """Write a single chunk to the Turbopuffer namespace.

        Args:
            chunk (Chunk): The chunk to write.
            **kwargs: Additional arguments for preparing Turbopuffer data (e.g., extra_metadata)
                      or passed to `namespace.upsert`.

        Returns:
            Any: The result from the Turbopuffer upsert operation.

        """
        return self.write_batch([chunk], **kwargs)

    def write_batch(self, chunks: Sequence[Chunk], **kwargs: Any) -> Any:
        """Write a batch of chunks to the Turbopuffer namespace.

        Args:
            chunks (Sequence[Chunk]): The sequence of chunks to write.
            **kwargs: Additional arguments for preparing Turbopuffer data (e.g., extra_metadata)
                      or passed to `namespace.upsert`.

        Returns:
            Any: The result from the Turbopuffer upsert operation.

        """
        data_to_upsert = self._prepare_tpuf_data(chunks, **kwargs)
        if not data_to_upsert["ids"]:
            return None # Nothing to write

        # Pass remaining kwargs to upsert if needed (e.g., distance_metric)
        return self.namespace.upsert(data=data_to_upsert, **kwargs)

    # Turbopuffer client is synchronous as of early 2024.
    async def awrite(self, chunk: Chunk, **kwargs: Any) -> Any:
        """Asynchronously write a single chunk (Not Implemented)."""
        raise NotImplementedError("Async write not implemented for TurbopufferHandshake.")

    async def awrite_batch(self, chunks: Sequence[Chunk], **kwargs: Any) -> Any:
        """Asynchronously write a batch of chunks (Not Implemented)."""
        raise NotImplementedError("Async batch write not implemented for TurbopufferHandshake.")

