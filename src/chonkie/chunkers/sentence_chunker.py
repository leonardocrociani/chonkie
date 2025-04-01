"""Chonkie: Sentence Chunker
Split text into chunks based on sentence boundaries.
"""

from __future__ import annotations
from bisect import bisect_left
from itertools import accumulate
from typing import Any, Callable, Literal, Sequence
import warnings
from chonkie.chunkers.base import BaseChunker
from chonkie.types.base import Chunk
from chonkie.types.sentence import Sentence, SentenceChunk


class SentenceChunker(BaseChunker):
    """Split text into chunks based on sentence boundaries and token limits."""

    _CHARS_PER_TOKEN = 6

    def __init__(
        self,
        tokenizer_or_token_counter: str | Callable | Any = "gpt2",
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12,
        include_delim: Literal["prev", "next"] | None = "prev",
        delim: str | list[str] = [".", "!", "?", "\n"],
        return_type: Literal["texts", "chunks"] = "chunks",
        approximate: bool = True,
    ):
        """Initialize the SentenceChunker.

        Args:
            tokenizer_or_token_counter (str | Any): The tokenizer or token counter to use.
            chunk_size (int): The maximum number of tokens per chunk.
            chunk_overlap (int): The number of overlapping tokens between chunks.
            min_sentences_per_chunk (int): Minimum number of sentences per chunk.
            min_characters_per_sentence (int): Minimum number of characters per sentence.
            approximate (bool): Whether to use approximate token counting.
            delim (str | list[str]): Delimiters for sentence splitting.
            include_delim (Literal["prev", "next"] | None): Whether to include delimiters in the output. If 'prev', include the delimiter before the sentence. If 'next', include it after. If None, do not include.
            return_type (Literal["texts", "chunks"]): The type of output to return.

        Raises:
            ValueError: if chunk_size is less than or equal to 0.
            ValueError: if chunk_overlap is negative.
            ValueError: if chunk_overlap is greater than or equal to chunk_size.
            ValueError: if min_sentences_per_chunk is less than 1.
            ValueError: if min_characters_per_sentence is less than 1.
            ValueError: if delim is not a string or list of strings.
            ValueError: if include_delim is not 'start', 'end', or None.
            ValueError: if return_type is not 'texts' or 'chunks'.
        """
        super().__init__(tokenizer_or_token_counter)
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap must be non-negative.")
        if isinstance(chunk_overlap, int) and chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size.")
        if min_sentences_per_chunk < 1:
            raise ValueError("Minimum sentences per chunk must be at least 1.")
        if min_characters_per_sentence < 1:
            raise ValueError(
                "Minimum characters per sentence must be at least 1."
            )
        if delim is None:
            raise ValueError(
                "Delim cannot be None. Must be a string or list of strings."
            )
        if include_delim not in ["prev", "next", None]:
            raise ValueError(
                "Include delim must be either 'prev', 'next', or None."
            )
        if return_type not in ["texts", "chunks"]:
            raise ValueError("Return type must be either 'texts' or 'chunks'.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.approximate = approximate
        self.delim = delim
        self.include_delim = include_delim
        self.return_type = return_type
        self.separator = "<!!CHONKIE_SENTENCE_SEPARATOR_INTERNAL!!>"

    def __repr__(self):
        """Return a string representation of the SentenceChunker."""
        return (
            f"SentenceChunker(tokenizer={self.tokenizer}, chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, min_sentences_per_chunk={self.min_sentences_per_chunk}, "
            f"min_characters_per_sentence={self.min_characters_per_sentence}, "
            f"approximate={self.approximate}, delim={self.delim}, "
            f"include_delim='{self.include_delim}', return_type='{self.return_type}')"
        )

    def _split(self, text: str) -> list[Sentence]:
        """Split text into sentences based on delimiters.

        Args:
            text (str): The text to split.

        Returns:
            list[Sentence]: List of Sentence objects.
        """
        for delim in self.delim:
            if self.include_delim == "next":
                text = text.replace(delim, self.sep + delim)
            elif self.include_delim == "prev":
                text = text.replace(delim, delim + self.sep)
            else:
                text = text.replace(delim, self.sep)

        splits = list(filter(None, text.split(self.sep)))

        # Combine short splits with the sentence before it.
        current = ""
        sentences = []
        for split in splits:
            if current:
                current += split
                sentences.append(current)
                current = ""
            elif len(split) < self.min_characters_per_sentence:
                current += split
            else:
                sentences.append(split)

            if len(current) >= self.min_characters_per_sentence:
                sentences.append(current)
                current = ""

        if current:
            sentences.append(current)

        return sentences

    def _estimate_token_count(self, sentences: str | list[str]) -> int:
        """Estimate the token count of the text.

        Args:
            text (str): The text to estimate tokens for.

        Returns:
            int: Estimated token count.
        """
        if isinstance(sentences, str):
            return max(1, len(sentences) // self._CHARS_PER_TOKEN)
        elif isinstance(sentences, list):
            return [
                max(1, len(sentence) // self._CHARS_PER_TOKEN)
                for sentence in sentences
            ]
        raise ValueError(
            f"Invalid type for sentences: {type(sentences)}. Expected str or list[str]."
        )

    def _prepare_sentences(self, text: str) -> Sequence[Sentence]:
        """Convert text into a list of Sentence objects with position and token count information."""

        sentences = self._split(text)
        if not sentences:
            return []

        # Calculate positions and lengths in a single pass
        positions = []
        current_position = 0

        for sentence in sentences:
            positions.append(current_position)
            current_position += len(sentence)

        # Determine token counts based on configuration
        if self.approximate:
            token_counts = self._estimate_token_count(sentences)
        else:
            token_counts = self.tokenizer.count_tokens_batch(sentences)

        # Construct Sentence objects using zip
        return [
            Sentence(
                text=sentence,
                start_index=position,
                end_index=position + len(sentence),
                token_count=token_count,
            )
            for sentence, position, token_count in zip(
                sentences, positions, token_counts
            )
        ]

    def _return_chunk(
        self, sentences: Sequence[Sentence], token_count: int
    ) -> SentenceChunk | str:
        """Return a SentenceChunk object with the given sentences and text."""
        text = "".join([sentence.text for sentence in sentences])
        if self.return_type == "chunks":
            return SentenceChunk(
                text=text,
                start_index=sentences[0].start_index,
                end_index=sentences[-1].end_index,
                token_count=token_count,
                sentences=sentences,
            )
        elif self.return_type == "texts":
            return text
        raise ValueError(
            f"Invalid return type: {self.return_type}. Expected 'texts' or 'chunks'."
        )

    def chunk(self, text: str) -> list[Chunk]:
        """Chunk the text into smaller pieces based on sentence boundaries.

        Args:
            text (str): The text to chunk.

        Returns:
            list[Chunk]: List of SentenceChunk objects.
        """
        if not text or not text.strip():
            return []

        sentences = self._prepare_sentences(text)
        if not sentences:
            return []

        chunks: list[SentenceChunk] = []
        feedback = 1.0
        position = 0
        token_sum = list(
            accumulate((s.token_count for s in sentences), initial=0)
        )

        while position < len(sentences):
            token_sum = [int(s * feedback) for s in token_sum]

            # break by bisect left
            target_tokens = token_sum[position] + self.chunk_size
            break_index = max(
                min(bisect_left(token_sum, target_tokens) - 1, len(sentences)),
                position + 1,
            )

            # Handle minimum sentences requirement
            if break_index - position < self.min_sentences_per_chunk:
                # If we can't meet the minimum_sentences requirement, we need to warn the user
                if position + self.min_sentences_per_chunk <= len(sentences):
                    break_index = position + self.min_sentences_per_chunk
                else:
                    warnings.warn(
                        f"min_sentneces_per_chunk {self.min_sentences_per_chunk} could not be satisfied for all chunks."
                        + f"{len(sentences) - position} sentences in the last chunk."
                        + "Either decrease the min_sentences_per_chunk or increase the chunk_size."
                    )
                    break_index = len(sentences)

            # Estimate token count for the chunk
            estimate = max(1, token_sum[break_index] - token_sum[position])

            # Get candidate sentences and verify token count
            chunk_sentences = sentences[position:break_index]
            chunk_text = "".join(s.text for s in chunk_sentences)
            actual = max(1, self.tokenizer.count_tokens(chunk_text))

            # Adjust feedback based on the difference between estimated and actual token count
            feedback = max(0.01, 1 - ((estimate - actual) / estimate))

            # Backoff if the actual token count exceeds the chunk size
            while (
                actual > self.chunk_size
                and len(chunk_sentences) > self.min_sentences_per_chunk
            ):
                break_index -= 1
                chunk_sentences = sentences[position:break_index]
                chunk_text = "".join(s.text for s in chunk_sentences)
                actual = self.tokenizer.count_tokens(chunk_text)

            chunks.append(self._create_chunk(chunk_sentences, actual))

            if self.chunk_overlap > 0 and break_index < len(sentences):
                overlap_tokens = 0
                overlap_index = break_index - 1

                while (
                    overlap_index > position
                    and overlap_tokens < self.chunk_overlap
                ):
                    sent = sentences[overlap_index]
                    next_tokens = (
                        overlap_tokens
                        + sent.token_count
                        + 1  # +1 for the space
                    )
                    if next_tokens > self.chunk_overlap:
                        break
                    overlap_tokens = next_tokens
                    overlap_index -= 1

                # Move position to after the overlap
                position = overlap_index + 1
            else:
                position = break_index

        return chunks
