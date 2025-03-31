"""Module for abstracting tokeinization logic."""

from __future__ import annotations

from collections import defaultdict
import importlib
import inspect
from typing import Any, Callable
import warnings


class Tokenizer:
    """Unified tokenizer interface for Chonkie.

    Args:
        tokenizer: Tokenizer identifier or instance.

    Raises:
        ImportError: If the specified tokenizer is not available.
    """

    def __init__(self, tokenizer: str | Callable | Any = "gpt2"):
        """Initialize the Tokenizer with a specified tokenizer."""
        if isinstance(tokenizer, str):
            self.tokenizer = self._load_tokenizer(tokenizer)
        else:
            self.tokenizer = tokenizer

        self._backend = self._get_backend(tokenizer)

    def _load_tokenizer(self, tokenizer: str):
        """Load the tokenizer based on the identifier."""
        if tokenizer.casefold == "character".casefold():
            return CharacterTokenizer()
        elif tokenizer.casefold() == "word".casefold():
            return WordTokenizer()

        # Try tokenizers first
        if importlib.util.find_spec("tokenizers") is not None:
            try:
                from tokenizers import Tokenizer

                return Tokenizer.from_pretrained(tokenizer)
            except Exception:
                warnings.warn(
                    "Could not find 'tokenizers'. Falling back to 'tiktoken'."
                )
        else:
            warnings.warn(
                "Could not find 'tokenizers'. Falling back to 'tiktoken'."
            )

        # Try tiktoken
        if importlib.util.find_spec("tiktoken") is not None:
            try:
                from tiktoken import get_encoding

                return get_encoding(tokenizer)
            except Exception:
                warnings.warn(
                    "Could not find 'tiktoken'. Falling back to 'transformers'."
                )
        else:
            warnings.warn(
                "Could not find 'tiktoken'. Falling back to 'transformers'."
            )

        # Try transformers as last resort
        if importlib.util.find_spec("transformers") is not None:
            try:
                from transformers import AutoTokenizer

                return AutoTokenizer.from_pretrained(tokenizer)
            except Exception:
                raise ValueError(
                    "Tokenizer not found in transformers, tokenizers, or tiktoken"
                )
        raise ValueError(
            "Tokenizer not found in transformers, tokenizers, or tiktoken"
        )

    def _get_backend(self, tokenizer: str):
        """Get the tokenizer instance based on the identifier."""
        supported_backends = [
            "chonkie",
            "transformers",
            "tokenizers",
            "tiktoken",
        ]
        for backend in supported_backends:
            if backend in str(type(self.tokenizer)):
                return backend
        if (
            callable(self.tokenizer)
            or inspect.isfunction(self.tokenizer)
            or inspect.ismethod(self.tokenizer)
        ):
            return "callable"
        raise ValueError(
            f"Unsupported tokenizer backend: {type(self.tokenizer)}"
        )

class CharacterTokenizer:
    """Character-based tokenizer."""

    def __init__(self):
        """Initialize the CharacterTokenizer."""
        self.vocab = [" "]  # Default vocabulary with space
        self.token2id = defaultdict(lambda: len(self.vocab))
        self.token2id[" "]  # Add space to the vocabulary


class WordTokenizer:
    """Word-based tokenizer."""

    def __init__(self):
        """Initialize the WordTokenizer."""
        self.vocab = [" "]  # Default vocabulary with space
        self.token2id = defaultdict(lambda: len(self.vocab))
        self.token2id[" "]  # Add space to the vocabulary
