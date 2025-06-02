"""Psycopg3 Handshake to export Chonkie's Chunks into a PostgreSQL database with pgvector."""

import importlib.util as importutil
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from uuid import NAMESPACE_OID, uuid5

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.types import Chunk

from .base import BaseHandshake

if TYPE_CHECKING:
    import numpy as np
    import psycopg


class PsycopgHandshake(BaseHandshake):
    """Psycopg3 Handshake to export Chonkie's Chunks into a PostgreSQL database with pgvector.
    
    This handshake allows storing Chonkie chunks in PostgreSQL with vector embeddings
    using the pgvector extension and psycopg3 driver.

    Args:
        connection: The psycopg3 connection to use.
        table_name: The name of the table to store chunks in.
        embedding_model: The embedding model to use for generating embeddings.
        vector_dimensions: The number of dimensions for the vector embeddings.
        create_table: Whether to create the table if it doesn't exist.

    """

    def __init__(
        self,
        connection: "psycopg.Connection",
        table_name: str = "chonkie_chunks",
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-retrieval-32M",
        vector_dimensions: Optional[int] = None,
        create_table: bool = True,
    ) -> None:
        """Initialize the Psycopg3 Handshake.
        
        Args:
            connection: The psycopg3 connection to use.
            table_name: The name of the table to store chunks in.
            embedding_model: The embedding model to use for generating embeddings.
            vector_dimensions: The number of dimensions for the vector embeddings.
            create_table: Whether to create the table if it doesn't exist.

        """
        # Warn the user that PsycopgHandshake is experimental
        warnings.warn(
            "Chonkie's PsycopgHandshake is experimental and may change in the future. "
            "Not all Chonkie features are supported yet.",
            FutureWarning
        )
                    
        super().__init__()
        
        # Lazy importing the dependencies
        self._import_dependencies()

        self.connection = connection
        self.table_name = table_name
        self.create_table = create_table
        
        # Initialize the embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        elif isinstance(embedding_model, BaseEmbeddings):
            self.embedding_model = embedding_model
        else:
            raise ValueError("embedding_model must be a string or a BaseEmbeddings instance.")

        # Determine vector dimensions
        if vector_dimensions is None:
            # Get dimensions from a test embedding
            test_embedding = self.embedding_model.embed("test")
            self.vector_dimensions = len(test_embedding)
        else:
            self.vector_dimensions = vector_dimensions

        # Setup the database
        self._setup_database()

    def _is_available(self) -> bool:
        """Check if the dependencies are available."""
        return (
            importutil.find_spec("psycopg") is not None and
            importutil.find_spec("pgvector") is not None
        )

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if not self._is_available():
            raise ImportError(
                "psycopg and pgvector are not installed. "
                "Please install them with `pip install chonkie[psycopg]`."
            )
        
        global psycopg, register_vector
        import psycopg
        from pgvector.psycopg import register_vector

    def _setup_database(self) -> None:
        """Setup the database with the pgvector extension and table."""
        with self.connection.cursor() as cur:
            # Enable the pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Register the vector type
            register_vector(self.connection)
            
            # Create the table if needed
            if self.create_table:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        embedding vector({self.vector_dimensions}),
                        start_index INTEGER,
                        end_index INTEGER,
                        token_count INTEGER,
                        metadata JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
            
            self.connection.commit()

    def _generate_id(self, index: int, chunk: Chunk) -> str:
        """Generate a unique ID for the chunk."""
        return str(
            uuid5(
                NAMESPACE_OID, 
                f"{self.table_name}::chunk-{index}:{chunk.text}"
            )
        )

    def _generate_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """Generate metadata for the chunk."""
        metadata = {
            "chunk_type": type(chunk).__name__,
        }
        
        # Add chunk-specific metadata
        if hasattr(chunk, "sentences") and chunk.sentences:
            metadata["sentence_count"] = len(chunk.sentences)
        
        if hasattr(chunk, "words") and chunk.words:
            metadata["word_count"] = len(chunk.words)
            
        if hasattr(chunk, "language") and chunk.language:
            metadata["language"] = chunk.language
            
        return metadata

    def write(self, chunks: Union[Chunk, Sequence[Chunk]]) -> List[str]:
        """Write chunks to the PostgreSQL database.
        
        Args:
            chunks: A single chunk or sequence of chunks to write.
            
        Returns:
            List[str]: List of IDs of the inserted chunks.
        """
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        inserted_ids = []
        
        with self.connection.cursor() as cur:
            for index, chunk in enumerate(chunks):
                # Generate ID and metadata
                chunk_id = self._generate_id(index, chunk)
                metadata = self._generate_metadata(chunk)
                
                # Generate embedding
                embedding = self.embedding_model.embed(chunk.text)
                
                # Insert the chunk
                cur.execute(f"""
                    INSERT INTO {self.table_name} 
                    (id, text, embedding, start_index, end_index, token_count, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        start_index = EXCLUDED.start_index,
                        end_index = EXCLUDED.end_index,
                        token_count = EXCLUDED.token_count,
                        metadata = EXCLUDED.metadata
                """, (
                    chunk_id,
                    chunk.text,
                    embedding,
                    chunk.start_index,
                    chunk.end_index,
                    chunk.token_count,
                    metadata
                ))
                
                inserted_ids.append(chunk_id)
            
            self.connection.commit()

        print(f"🦛 Chonkie wrote {len(chunks)} chunks to PostgreSQL table: {self.table_name}")
        return inserted_ids

    def search(
        self, 
        query: str, 
        limit: int = 5, 
        distance_metric: str = "l2"
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity.
        
        Args:
            query: The query text to search for.
            limit: Maximum number of results to return.
            distance_metric: Distance metric to use ('l2', 'cosine', 'inner_product').
            
        Returns:
            List[Dict[str, Any]]: List of similar chunks with metadata.
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.embed(query)
        
        # Map distance metrics to PostgreSQL operators
        operators = {
            "l2": "<->",
            "cosine": "<=>", 
            "inner_product": "<#>"
        }
        
        if distance_metric not in operators:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        
        operator = operators[distance_metric]
        
        with self.connection.cursor() as cur:
            cur.execute(f"""
                SELECT 
                    id, 
                    text, 
                    start_index, 
                    end_index, 
                    token_count, 
                    metadata,
                    embedding {operator} %s AS distance
                FROM {self.table_name}
                ORDER BY embedding {operator} %s
                LIMIT %s
            """, (query_embedding, query_embedding, limit))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "id": row[0],
                    "text": row[1],
                    "start_index": row[2],
                    "end_index": row[3],
                    "token_count": row[4],
                    "metadata": row[5],
                    "distance": float(row[6])
                })
            
            return results

    def create_index(
        self, 
        index_type: str = "hnsw", 
        distance_metric: str = "l2",
        **index_params: Any
    ) -> None:
        """Create an index on the embedding column for faster similarity search.
        
        Args:
            index_type: Type of index to create ('hnsw' or 'ivfflat').
            distance_metric: Distance metric for the index ('l2', 'cosine', 'inner_product').
            **index_params: Additional parameters for the index.
        """
        # Map distance metrics to operator classes
        opclasses = {
            "l2": "vector_l2_ops",
            "cosine": "vector_cosine_ops",
            "inner_product": "vector_ip_ops"
        }
        
        if distance_metric not in opclasses:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        
        opclass = opclasses[distance_metric]
        index_name = f"{self.table_name}_embedding_{index_type}_{distance_metric}_idx"
        
        with self.connection.cursor() as cur:
            if index_type == "hnsw":
                m = index_params.get("m", 16)
                ef_construction = index_params.get("ef_construction", 64)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self.table_name} 
                    USING hnsw (embedding {opclass})
                    WITH (m = {m}, ef_construction = {ef_construction})
                """)
            elif index_type == "ivfflat":
                lists = index_params.get("lists", 100)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self.table_name} 
                    USING ivfflat (embedding {opclass})
                    WITH (lists = {lists})
                """)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            self.connection.commit()
            
        print(f"🦛 Created {index_type} index on {self.table_name}.embedding using {distance_metric}")

    def __repr__(self) -> str:
        """Return the string representation of the PsycopgHandshake."""
        return f"PsycopgHandshake(table_name={self.table_name}, vector_dimensions={self.vector_dimensions})"