"""Module containing the associated types for the LateChunker."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

from .base import Chunk
from .sentence import Sentence

if TYPE_CHECKING:
    import numpy as np


@dataclass
class LateSentence(Sentence):
    """Class to represent the late sentence."""

    start_token: int
    end_token: int


@dataclass
class LateChunk(Chunk):
    """Class to represent the late chunk.

    Attributes:
        text (str): The text of the chunk.
        start_index (int): The start index of the chunk.
        end_index (int): The end index of the chunk.
        token_count (int): The number of tokens in the chunk.
        start_token (int): The start token of the chunk.
        end_token (int): The end token of the chunk.
        sentences (list[LateSentence]): The sentences in the chunk.
        embedding (Optional[np.ndarray]): The embedding of the chunk.

    """

    start_token: int = 0
    end_token: int = 0
    sentences: list[LateSentence] = field(default_factory=list)
    embedding: Optional["np.ndarray"] = field(default=None)

    def to_dict(self) -> dict:
        """Return the LateChunk as a dictionary."""
        return self.__dict__()

    @classmethod
    def from_dict(cls, data: Dict) -> "LateChunk":
        """Create a LateChunk from a dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        """Return a string representation of the LateChunk."""
        return (
            f"LateChunk(start_token={self.start_token}, "
            f"end_token={self.end_token}, "
            f"sentence_count={len(self.sentences)}, "
            f"embedding={self.embedding})"
        )
