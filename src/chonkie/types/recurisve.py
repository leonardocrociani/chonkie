"""Custom types for recursive chunking."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from chonkie.types.base import Chunk


@dataclass
class RecursiveLevel:
    """Class to express chunking delimiters at different levels.

    Attributes:
        delimiters (List[str] | str): List of delimiters for the chunking level. **NOTE:** If using whitespace as a delimiter, no other delimiters should be included.
        include_delim (Literal["prev", "next", None] | None): Whether to include the delimiter in the previous chunk, next chunk, or not at all.
    """

    delimiters: list[str] | str
    include_delim: Literal["prev", "next", None] | None = None

    def __post_init__(self):
        # More than one delimiter with one of them being whitespace
        if (
            isinstance(self.delimiters, list)
            and any(delim.strip() == "" for delim in self.delimiters)
            and len(self.delimiters) > 1
        ):
            raise ValueError(
                "Cannot combine whitespace with other delimiters. "
                "Please use only whitespace or a custom list of delimiter."
            )
        if isinstance(self.delimiters, str) and self.delimiters == "":
            raise ValueError("Delimiter cannot be an empty string.")

        if isinstance(self.delimiters, list):
            for delim in self.delimiters:
                if not isinstance(delim, str):
                    raise ValueError("All delimiters must be strings.")
                elif delim == "":
                    raise ValueError("Delimiter cannot be an empty string.")

    def to_dict(self) -> dict:
        """Return the RecursiveLevel as a dictionary."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "RecursiveLevel":
        """Create a RecursiveLevel object from a dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        """Return a string representation of the RecursiveLevel."""
        return (
            f"RecursiveLevel(delimiters={self.delimiters}, "
            f"include_delim={self.include_delim})"
        )


@dataclass
class RecursiveRules:
    """Expression rules for recursive chunking."""

    levels: list[RecursiveLevel] | RecursiveLevel | None = None

    def __post_init__(self):
        if self.levels is None:
            paragraphs = RecursiveLevel(
                delimiters=["\n", "\n\n", "\r\n"], include_delim="prev"
            )
            sentences = RecursiveLevel(
                delimiters=[
                    ".",
                    "!",
                    "?",
                ],
                include_delim="prev",
            )
            pauses = RecursiveLevel(
                delimiters=[
                    "{",
                    "}",
                    '"',
                    "[",
                    "]",
                    "<",
                    ">",
                    "(",
                    ")",
                    ":",
                    ";",
                    ",",
                    "—",
                    "|",
                    "~",
                    "-",
                    "...",
                    "`",
                    "'",
                ],
                include_delim="prev",
            )
            word = RecursiveLevel(delimiters=" ", include_delim="prev")
            token = RecursiveLevel(delimiters=None, include_delim="prev")
            self.levels = [paragraphs, sentences, pauses, word, token]
        elif isinstance(self.levels, RecursiveLevel):
            self.levels.__post_init__()
        elif isinstance(self.levels, list):
            for level in self.levels:
                level.__post_init__()
        else:
            raise ValueError(
                "Levels must be a RecursiveLevel object or a list of RecursiveLevel objects."
            )

    def __repr__(self) -> str:
        """Return a string representation of the RecursiveRules."""
        return f"RecursiveRules(levels={self.levels})"

    def __len__(self) -> int:
        return len(self.levels)

    def __getitem__(self, index: int) -> RecursiveLevel:
        if isinstance(self.levels, list):
            return self.levels[index]
        raise TypeError(
            "Levels must be a list of RecursiveLevel objects to use indexing."
        )

    def __iter__(self):
        if isinstance(self.levels, list):
            return iter(self.levels)
        raise TypeError(
            "Levels must be a list of RecursiveLevel objects to use iteration."
        )

    @classmethod
    def from_dict(cls, data: dict) -> "RecursiveRules":
        """Create a RecursiveRules object from a dictionary."""
        dict_levels = data.pop("levels")
        object_levels = None
        if dict_levels is not None:
            if isinstance(dict_levels, dict):
                object_levels = RecursiveLevel.from_dict(dict_levels)
            elif isinstance(dict_levels, list):
                object_levels = [
                    RecursiveLevel.from_dict(d_level) for d_level in dict_levels
                ]
        return cls(levels=object_levels)

    def to_dict(self) -> dict:
        """Return the RecursiveRules as a dictionary."""
        result = dict()
        result["levels"] = None
        if isinstance(self.levels, RecursiveLevel):
            result["levels"] = self.levels.to_dict()
        elif isinstance(self.levels, list):
            result["levels"] = [level.to_dict() for level in self.levels]
        else:
            raise ValueError(
                "Levels must be a RecursiveLevel object or a list of RecursiveLevel objects."
            )
        return result


@dataclass
class RecursiveChunk(Chunk):
    """Class to represent recursive chunks.

    Attributes:
        recursive_level (int | None): The level of recursion for the chunk, if any.
    """

    recursive_level: int | None = None

    def str_repr(self) -> str:
        """Return a string representation of the RecursiveChunk."""
        return (
            f"RecursiveChunk(text={self.text}, start_index={self.start_index}, "
            f"end_index={self.end_index}, token_count={self.token_count}, "
            f"recursive_level={self.recursive_level})"
        )

    def __repr__(self) -> str:
        """Return a string representation of the RecursiveChunk."""
        return self.str_repr()

    def __str__(self):
        """Return a string representation of the RecursiveChunk."""
        return self.str_repr()

    def to_dict(self) -> dict:
        """Return the RecursiveChunk as a dictionary."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> RecursiveChunk:
        """Create a RecursiveChunk object from a dictionary."""
        return cls(**data)
