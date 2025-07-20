"""FileFetcher is a fetcher that fetches data from a file."""

from .base import BaseFetcher


class FileFetcher(BaseFetcher):
    """FileFetcher is a fetcher that fetches data from a file.

    Args:
        path (str): The path to the file to fetch data from.

    """

    def __init__(self, path: str) -> None:
        """Initialize the FileFetcher.

        Args:
            path (str): The path to the file to fetch data from.

        """
        super().__init__()
        self.path = path

    def fetch(self) -> str:
        """Fetch data from the file.

        Returns:
            str: The data fetched from the file.

        """
        with open(self.path, "r") as f:
            return f.read()
