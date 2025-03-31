"""Chonkie: Token Chunker

Token based chunker for text processing.
This chunker splits text into chunks based on a specified token limit.
It uses a tokenizer to count tokens and create chunks of text.
"""

from __future__ import annotations
from typing import Any, Literal, Sequence

from tqdm import trange
from chonkie.types.base import Chunk
from chonkie.chunkers.base import BaseChunker


class TokenChunker(BaseChunker):
    """Split text into chunks based on token count."""

    def __init__(
        self,
        tokenizer: str | Any = "gpt2",
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        return_type: Literal["texts", "chunks"] = "chunks",
    ):
        """Initialize the TokenChunker.

        Args:
            tokenizer (str | Any): The tokenizer to use for token counting.
            chunk_size (int): The maximum number of tokens per chunk.
            chunk_overlap (int): The number of overlapping tokens between chunks.
            return_type (Literal["texts", "chunks"]): The type of output to return.
        """
        super().__init__(tokenizer)
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap must be non-negative.")
        if isinstance(chunk_overlap, int) and chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size.")
        if return_type not in ["texts", "chunks"]:
            raise ValueError("Return type must be either 'texts' or 'chunks'.")
        self.chunk_size = chunk_size
        self.chunk_overlap = (
            chunk_overlap
            if isinstance(chunk_overlap, int)
            else int(chunk_overlap * chunk_size)
        )
        self.return_type = return_type
        self._multiprocessing = (
            False  # Disable multiprocessing for Token chunker
        )

    def __call__(
        self,
        text: str | Sequence[str],
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> Sequence[Chunk] | Sequence[Sequence[Chunk]]:
        """Call TokenChunker with the given args.

        Args:
            text (str | List[str]): The text to chunk.
            batch_size (int): The number of texts to process in parallel.
            show_progress (bool): Whether to show progress.

        Returns:
            Sequence[Chunk] | Sequence[Sequence[Chunk]]: A list of chunks or a list of lists of chunks.
        """
        if isinstance(text, str):
            return self.chunk(text)
        elif isinstance(text, Sequence):
            return self.chunk_batch(text, batch_size, show_progress)
        else:
            raise ValueError("Input must be a string or a list of strings.")

    def __repr__(self):
        """Return a string representation of the TokenChunker."""
        return (
            f"TokenChunker(tokenizer={self.tokenizer}, chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, return_type='{self.return_type}')"
        )

    def _get_token_groups(
        self, tokens: Sequence[int]
    ) -> Sequence[Sequence[int]]:
        """Returns a sequence of chunks from a list of tokens."""  # Updated docstring

        all_chunks = []

        if not tokens:
            return all_chunks

        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            raise ValueError("Chunk overlap must be less than chunk size")

        for start in range(0, len(tokens), step):
            end = min(start + self.chunk_size, len(tokens))
            chunk = tokens[start:end]
            all_chunks.append(chunk)

        return all_chunks  # Return the complete list of chunks

    def _create_chunks(
        self,
        texts: Sequence[str],
        groups: Sequence[Sequence[int]],
        token_counts: Sequence[int],
    ) -> Sequence[Chunk]:
        if self.chunk_overlap == 0:
            overlaps = [0] * len(groups)
        else:
            overlapping_txts = self.tokenizer.decode_batch(
                [
                    groups[-self.chunk_overlap :]
                    if (len(group) > self.chunk_overlap)
                    else group
                    for group in groups
                ]
            )
            overlaps = [len(txt) for txt in overlapping_txts]

        chunks = []
        curr_index = 0
        for chunk_text, overlap_length, token_count in zip(
            texts, overlaps, token_counts
        ):
            start_index = curr_index
            end_index = start_index + len(chunk_text)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    token_count=token_count,
                )
            )
            curr_index = end_index - overlap_length

        return chunks

    def _process_batch(self, texts: Sequence[str]) -> Sequence[Sequence[Chunk]]:
        token_batch = self.tokenizer.encode_batch(texts)
        result = []

        for tokens in token_batch:
            if not tokens:
                result.append([])
                continue

            groups = self._get_token_groups(tokens)
            if self.return_type == "texts":
                result.append(self.tokenizer.decode_batch(groups))
            elif self.return_type == "chunks":
                chunks = self._create_chunks(
                    texts=self.tokenizer.decode_batch(groups),
                    groups=groups,
                    token_counts=[len(group) for group in groups],
                )
                result.append(chunks)
            else:
                raise ValueError(
                    f"Invalid return type: {self.return_type}. Expected 'texts' or 'chunks'."
                )

        return result

    def chunk(self, text) -> Sequence[Chunk]:
        """Chunk the given text into smaller pieces based on token count.

        Args:
            text (str): The text to chunk.

        Returns:
            Sequence[Chunk]: A list of chunks created from the input text.
        """
        if not text or text.strip():
            return []

        tokens = self.tokenizer.encode(text)
        groups = self._get_token_groups(tokens)

        if self.return_type == "texts":
            return self.tokenizer.decode_batch(groups)
        if self.return_type == "chunks":
            return self._create_chunks(
                # Decode texts to standardize the text format and create expected indices.
                texts=self.tokenizer.decode_batch(groups),
                groups=groups,
                token_counts=[len(group) for group in groups],
            )

    def chunk_batch(
        self, texts: Sequence[str], batch_size: int = 1, show_progress=True
    ) -> Sequence[Sequence[Chunk]]:
        """Chunk a batch of texts with token chunking.

        Args:
            texts (Sequence[str]): The texts to chunk.
            batch_size (int): The number of texts to process in parallel.
            show_progress (bool): Whether to show progress.

        Returns:
            Sequence[Sequence[Chunk]]: A list of lists of chunks created from the input texts.
        """
        chunks = []
        for i in trange(
            0,
            len(texts),
            batch_size,
            desc="🦛",
            disable=not show_progress,
            unit="batch",
            bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} batches chunked [{elapsed}<{remaining}, {rate_fmt}] 🌱",
            ascii=" o",
        ):
            batch_texts = texts[i : min(i + batch_size, len(texts))]
            chunks.extend(self._process_batch(batch_texts))
        return chunks
