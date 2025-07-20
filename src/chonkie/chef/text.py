"""TextChef is a chef that processes text data."""

from .base import BaseChef


class TextChef(BaseChef):
    """TextChef is a chef that processes text data."""

    def process(self, path: str) -> str:
        """Process the text data."""
        with open(path, "r") as file:
            return file.read()

    def __repr__(self) -> str:
        """Return a string representation of the chef."""
        return f"{self.__class__.__name__}()"
