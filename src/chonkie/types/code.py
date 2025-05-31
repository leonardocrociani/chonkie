"""Module containing CodeChunker types."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from chonkie.types.base import Chunk

if TYPE_CHECKING:
    from tree_sitter import Node


@dataclass
class CodeChunk(Chunk):
    """Code chunk with metadata."""

    lang: Optional[str] = None
    nodes: Optional[List["Node"]] = None

    def to_dict(self) -> dict:
        """Return the Chunk as a dictionary."""
        result = super().to_dict()
        result["lang"] = self.lang
        result["nodes"] = None  # TODO: Add support for dict nodes
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "CodeChunk":
        """Create a Chunk object from a dictionary."""
        return cls(**data)