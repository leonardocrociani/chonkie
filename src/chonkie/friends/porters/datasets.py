"""DatasetsPorter to convert Chunks into datasets format for storage."""

from typing import Any, Dict, Union

from datasets import Dataset, DatasetDict

from chonkie.types import Chunk

from .base import BasePorter


class DatasetsPorter(BasePorter):
    """Porter to convert Chunks into datasets format for storage."""

    def __init__(self):
        """Initialize the DatasetsPorter."""
        super().__init__()

    def export(
        self,
        chunks: list[Chunk],
        return_ds: bool = False,
        dataset_path: str = "chuncked_data",
        **kwargs: Dict[str, Any],
    ) -> Union[DatasetDict, None]:
        """Export a list of Chunk objects into a Hugging Face Dataset and optionally save it to disk.

        Args:
            chunks (list[Chunk]): The list of Chunk objects to export.
            return_ds (bool, optional): If True, returns the Dataset object instead of saving it to disk. Defaults to False.
            dataset_path (str, optional): The path where the dataset will be saved if return_ds is False. Defaults to "chuncked_data".
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            Union[DatasetDict, None]: Returns the Dataset object if return_ds is True, otherwise saves the dataset to disk and returns None.

        """
        dataset = Dataset.from_list([chunk.to_dict() for chunk in chunks])
        if return_ds:
            return dataset
        else:
            dataset.save_to_disk(dataset_path)

    def __call__(
        self,
        chunks: list[Chunk],
        return_ds: bool = False,
        dataset_path: str = "chuncked_data",
        **kwargs: Dict[str, Any],
    ) -> Union[DatasetDict, None]:
        """Export a list of Chunk objects into a Hugging Face Dataset and optionally save it to disk.

        Args:
            chunks (list[Chunk]): The list of Chunk objects to export.
            return_ds (bool, optional): If True, returns the Dataset object instead of saving it to disk. Defaults to False.
            dataset_path (str, optional): The path where the dataset will be saved if return_ds is False. Defaults to "chuncked_data".
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            Union[DatasetDict, None]: Returns the Dataset object if return_ds is True, otherwise saves the dataset to disk and returns None.

        """
        return self.export(chunks, return_ds, dataset_path, **kwargs)
