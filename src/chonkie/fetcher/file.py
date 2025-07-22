"""FileFetcher is a fetcher that fetches paths of files from local directories."""

from pathlib import Path
from typing import List, Optional

from chonkie.pipeline.registry import fetcher
from .base import BaseFetcher


@fetcher("file")
class FileFetcher(BaseFetcher):
    """FileFetcher is a fetcher that fetches paths of files from local directories."""

    def __init__(self) -> None:
        """Initialize the FileFetcher."""
        super().__init__()

    def fetch(self, dir: str, ext: Optional[List[str]] = None) -> List[Path]:
        """Fetch files from a directory.

        Args:
            dir (str): The directory to fetch files from.
            ext (Optional[List[str]]): The file extensions to fetch.

        Returns:
            List[Path]: The list of files fetched from the directory.

        """
        # Reads the entire directory and returns a list of files with the specified extension
        return [
            file
            for file in Path(dir).iterdir()
            if file.is_file() and (ext is None or file.suffix in ext)
        ]

    def fetch_file(self, dir: str, name: str) -> Path:  # type: ignore[override]
        """Given a directory and a file name, return the path to the file.

        NOTE: This method is mostly for uniformity across fetchers since one may require to
        get a file from an online database.
        """
        # We should search the directory for the file
        for file in Path(dir).iterdir():
            if file.is_file() and file.name == name:
                return file
        raise FileNotFoundError(f"File {name} not found in directory {dir}")

    def __call__(self, path: Optional[str] = None, dir: Optional[str] = None, ext: Optional[List[str]] = None) -> List[Path]:  # type: ignore[override]
        """Fetch files from a directory or a single file path.

        Args:
            path (Optional[str]): Path to a single file to fetch.
            dir (Optional[str]): The directory to fetch files from.
            ext (Optional[List[str]]): The file extensions to fetch (only used with dir).

        Returns:
            List[Path]: The list of files fetched.

        Raises:
            ValueError: If neither path nor dir is provided, or if both are provided.
            FileNotFoundError: If the specified path or directory doesn't exist.

        """
        if path is not None and dir is not None:
            raise ValueError("Cannot specify both 'path' and 'dir'. Use 'path' for single file or 'dir' for directory.")
        
        if path is None and dir is None:
            raise ValueError("Must specify either 'path' for single file or 'dir' for directory.")
        
        if path is not None:
            # Single file mode
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if not file_path.is_file():
                raise ValueError(f"Path is not a file: {path}")
            return [file_path]
        
        else:
            # Directory mode (existing behavior)
            return self.fetch(dir, ext)
