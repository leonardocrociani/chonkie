"""BaseFetcher is the base class for all fetchers."""

from abc import ABC, abstractmethod


class BaseFetcher(ABC):
    """BaseFetcher is the base class for all fetchers."""

    def __init__(self):
        """Initialize the BaseFetcher."""
        pass

    @abstractmethod
    def fetch(self):
        """Fetch data from the source."""
        raise NotImplementedError("Subclasses must implement fetch()")
