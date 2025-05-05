"""Chroma Handshake to export Chonkie's Chunks into a Chroma collection."""

from typing import Sequence, Union

from chonkie.types import Chunk

from .base import BaseHandshake


class ChromaHandshake(BaseHandshake):
    """Chroma Handshake to export Chonkie's Chunks into a Chroma collection."""

    def __init__(self, 
                collection_name: str = "chonkie"):
        """Initialize the Chroma Handshake."""
        super().__init__()
        self.collection_name = collection_name

    def write(self, chunks: Union[Chunk, Sequence[Chunk]]) -> None:
        """Write the Chunks to the Chroma collection."""
        pass
