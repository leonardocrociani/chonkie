"""Document type for Chonkie."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

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
        id: The id of the document. If not provided, a random uuid will be generated.
        text: The complete text of the document.
        chunks: The chunks of the document.
        metadata: Any additional metadata you want to store about the document.
        
    """

    id: Optional[str] = field(default_factory=str)
    text: str = field(default_factory=str)
    chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
