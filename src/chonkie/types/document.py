"""Document type for Chonkie."""

from dataclasses import dataclass
from typing import List

from .base import Chunk


@dataclass
class Document:
    """Document type for Chonkie.
    
    Document allows us to encapsulate a text and its chunks, along with any additional 
    metadata. It becomes essential when dealing with complex chunking use-cases, such
    as dealing with in-line images, tables, or other non-text data. Documents are also 
    useful to give meaning when you want to chunk text that is already chunked, possibly
    with different chunkers.

    Args:
        text: The text of the document.
        chunks: The chunks of the document.

    """

    text: str
    chunks: List[Chunk]
