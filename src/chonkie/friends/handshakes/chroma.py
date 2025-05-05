"""Chroma Handshake to export Chonkie's Chunks into a Chroma collection."""

import importlib.util as importutil
import warnings
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union
from uuid import NAMESPACE_OID, uuid5

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.refinery import EmbeddingsRefinery
from chonkie.types import Chunk

from .base import BaseHandshake
from .utils import generate_random_collection_name

if TYPE_CHECKING:
    try:
        import chromadb
    except ImportError:
        chromadb = Any

# NOTE: We're doing a bit of a hack here to work with embeddings from inside Chonkie
#  since we can't have a EmbeddingFunction without having ChromaDB in the base install. 
# So we put the input of EmbeddingsRefinery to have "input" and not "chunks" so it 
# looks and feels like a normal chromadb.EmbeddingFunction. 
# Personally, I don't quite like this, but it's the best we can do for now. 

class ChromaHandshake(BaseHandshake):
    """Chroma Handshake to export Chonkie's Chunks into a Chroma collection."""

    def __init__(self, 
                client: Optional["chromadb.Client"] = None,
                collection_name: Union[str, Literal["random"]] = "random", 
                embedding_model: Union[str, BaseEmbeddings, AutoEmbeddings, EmbeddingsRefinery] = "minishlab/potion-retrieval-32M"
                ) -> None:
        """Initialize the Chroma Handshake."""
        # Warn the user that ChromaHandshake is experimental
        warnings.warn("Chonkie's ChromaHandshake is experimental and may change in the future. Not all Chonkie features are supported yet.", FutureWarning)
                    
        super().__init__()
        
        # Lazy importing the dependencies
        self._import_dependencies()

        # Initialize Chroma client
        if client is None:
            self.client = chromadb.Client()
        else:
            self.client = client

        # Initialize the EmbeddingRefinery internally!
        if isinstance(embedding_model, EmbeddingsRefinery):
            self.embedding_refinery = embedding_model
        else:
            self.embedding_refinery = EmbeddingsRefinery(embedding_model)

        # If the collection name is not random, create the collection
        if collection_name != "random":
            self.collection_name = collection_name
            self.collection = self.client.get_or_create_collection(self.collection_name)
        else:
            # Keep generating random collection names until we find one that doesn't exist
            while True:
                self.collection_name = generate_random_collection_name()
                try:
                    self.collection = self.client.get_or_create_collection(self.collection_name)
                    break
                except Exception:
                    pass
            print(f"🦛 Chonkie created a new collection in ChromaDB: {self.collection_name}")
        
        # Now that we have a collection, we can write the Chunks to it!

    def _is_available(self) -> bool:
        """Check if the dependencies are available."""
        return importutil.find_spec("chromadb") is not None

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if self._is_available():
            global chromadb
            import chromadb
        else:
            raise ImportError("ChromaDB is not installed. " +
                              "Please install it with `pip install chonkie[chroma]`.")


    def _generate_id(self, index: int, chunk: Chunk) -> str:
        """Generate a unique index name for the Chunk."""
        return str(
            uuid5(
                NAMESPACE_OID, 
                f"{self.collection_name}::chunk-{index}:{chunk.text}"
            )
        )
    def _generate_metadata(self, chunk: Chunk) -> dict:
        """Generate the metadata for the Chunk."""
        return {
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "token_count": chunk.token_count,
        }


    def write(self, chunks: Union[Chunk, Sequence[Chunk]]) -> None:
        """Write the Chunks to the Chroma collection."""
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        # Generate the ids and metadata
        ids = [self._generate_id(index, chunk) for (index, chunk) in enumerate(chunks)]
        metadata = [self._generate_metadata(chunk) for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        
        # Write the Chunks to the Chroma collection
        # Since this uses the `upsert` method, if the same index and same chunk text already exist, it will update the existing Chunk — which would only be the case if the Chunk has a different embedding
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadata,
        )

        print(f"🦛 Chonkie wrote {len(chunks)} Chunks to the Chroma collection: {self.collection_name}")
    

