"""Module for Chonkie's Porters.

Porters allow the user to _export_ data from chonkie into a variety of formats for saving on disk or cloud blob storage. Porters make the implicit assumption that the data is not being used for querying, but rather for saving.
"""

from abc import ABC, abstractmethod
from typing import List

from chonkie.types import Chunk


class BasePorter(ABC):
    """Abstract base class for Chonkie's Porters."""

    @abstractmethod
    def export(self, chunks: List[Chunk]) -> None:
        """Export the chunks to the desired format."""
        raise NotImplementedError("Subclasses must implement this method.")
