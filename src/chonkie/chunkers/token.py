"""Chonkie: Token Chunker

Token based chunker for text processing.
This chunker splits text into chunks based on a specified token limit.
It uses a tokenizer to count tokens and create chunks of text.
"""

from __future__ import annotations
import warnings
from typing import Any, Literal
from chonkie.tokenizer import Tokenizer
from chonkie.types import Chunk
from chonkie.chunkers.base import BaseChunker


class TokenChunker(BaseChunker):
    """Split text into chunks based on token count."""

    def __init__(
        self,
        tokenizer: str | Any = "gpt2",
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        return_type: Literal["texts", "chunks"] = "chunks",
    ):
        """Initialize the TokenChunker.

        Args:
            tokenizer (str | Any): The tokenizer to use for token counting.
            chunk_size (int): The maximum number of tokens per chunk.
            chunk_overlap (int): The number of overlapping tokens between chunks.
            return_type (Literal["texts", "chunks"]): The type of output to return.
        """
        super().__init__(tokenizer)
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap must be non-negative.")
        if isinstance(chunk_overlap, int) and chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size.")
        if return_type not in ["texts", "chunks"]:
            raise ValueError("Return type must be either 'texts' or 'chunks'.")
        self.chunk_size = chunk_size
        self.chunk_overlap = (
            chunk_overlap
            if isinstance(chunk_overlap, int)
            else int(chunk_overlap * chunk_size)
        )
        self.return_type = return_type
        self._multiprocessing = False

        
