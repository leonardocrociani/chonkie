"""Chonkie: Sentence Chunker
Split text into chunks based on sentence boundaries.
"""

from __future__ import annotations
from chonkie.chunkers.base import BaseChunker

class SentenceChunker(BaseChunker):
    """Split text into chunks based on sentence boundaries and token limits."""

    