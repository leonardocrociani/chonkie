"""FileFetcher is a fetcher that fetches data from a file."""

from .base import BaseFetcher


class FileFetcher(BaseFetcher):
    """FileFetcher is a fetcher that fetches data from a file."""

    def __init__(self) -> None:
        """Initialize the FileFetcher."""
        super().__init__()

    def fetch(self, path: str) -> str:
        """Fetch data from the file.

        Returns:
            str: The data fetched from the file.

        """
        with open(path, "r") as f:
            return str(f.read())

    def __call__(self, path: str) -> str:
        """Fetch data from the file.

        Returns:
            str: The data fetched from the file.

        """
        return self.fetch(path)
