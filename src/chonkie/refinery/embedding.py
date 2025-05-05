"""Embedding Refinery."""

from typing import Any, Dict, List, Union

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.types import Chunk

from .base import BaseRefinery


class EmbeddingsRefinery(BaseRefinery):
    """Embedding Refinery.
    
    Embeds the text of the chunks using the embedding model and 
    adds the embeddings to the chunks for use in downstream tasks
    like upserting into a vector database.

    Args:
        embedding_model: The embedding model to use.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        embedding_model: Union[str, BaseEmbeddings, AutoEmbeddings],
        **kwargs: Dict[str, Any]
    ) -> None:
        """Initialize the EmbeddingRefinery."""
        super().__init__()

        # Check if the model is a string
        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model, **kwargs)
        elif isinstance(embedding_model, BaseEmbeddings):
            self.embedding_model = embedding_model
        else:
            raise ValueError("Model must be a string or a BaseEmbeddings instance.")

    def is_available(self) -> bool:
        """Check if the embedding model is available."""
        return self.embedding_model.is_available()
        
    def refine(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine the chunks."""
        texts = [chunk.text for chunk in chunks]
        embeds = self.embedding_model.embed_batch(texts)
        for chunk, embed in zip(chunks, embeds):
            chunk.embedding = embed
        return chunks
    
    # NOTE: The __call__ method here is taking in "input" and not "chunks"
    # because we want to be able to use the EmbeddingRefinery as a EmbeddingFunction 
    # in ChromaHandshake -> which seems to only accept a __call__ method with "input"
    # and not "chunks"
    def __call__(self, input: List[Chunk]) -> List[Chunk]:
        """Call the EmbeddingRefinery."""
        return self.refine(input)

    def __repr__(self) -> str:
        """Represent the EmbeddingRefinery."""
        return f"EmbeddingsRefinery(embedding_model={self.embedding_model})"

    @property
    def dimension(self) -> int:
        """Dimension of the embedding model."""
        return self.embedding_model.dimension
