"""Refinery for adding overlap to chunks."""

import warnings
from typing import Any, Callable, Dict, List, Literal, Union

from chonkie.refinery.base import BaseRefinery
from chonkie.tokenizer import Tokenizer
from chonkie.types import Chunk, RecursiveLevel, RecursiveRules

# TODO: Fix the way that float context size is handled.
# Currently, it just estimates the context size to token count
# but it should ideally handle it on a chunk by chunk basis.

# TODO: Add support for `justified` method which is the best of
# both prefix and suffix overlap.

# TODO: The OverlapRefinery is so slow right now (TwT)
# We need to find a way to speed it up.

class OverlapRefinery(BaseRefinery):
    """Refinery for adding overlap to chunks."""

    def __init__(
        self,
        tokenizer_or_token_counter: Union[str, Callable, Any] = "character",
        context_size: Union[int, float] = 0.25,
        mode: Literal["token", "recursive"] = "token",
        method: Literal["suffix", "prefix"] = "suffix",
        rules: RecursiveRules = RecursiveRules(),
        merge: bool = True,
        inplace: bool = True,
    ) -> None:
        """Initialize the refinery.

        When a tokenizer or token counter is not provided, the refinery 
        defaults to character-level overlap. Otherwise, the refinery will
        use the tokenizer or token counter to calculate the overlap.
        
        Args:
            tokenizer_or_token_counter: The tokenizer or token counter to use. Defaults to None.
            context_size: The size of the context to add to the chunks.
            mode: The mode to use for overlapping. Could be token or recursive.
            method: The method to use for the context. Could be suffix or prefix.
            rules: The rules to use for the recursive overlap. Defaults to RecursiveRules().
            merge: Whether to merge the context with the chunk. Defaults to True.
            inplace: Whether to modify the chunks in place or make a copy. Defaults to True.

        """
        # Check if the context size is a valid number
        if isinstance(context_size, float) and (context_size <= 0 or context_size > 1):
            raise ValueError("Context size must be a number between 0 and 1.")
        elif isinstance(context_size, int) and context_size <= 0:
            raise ValueError("Context size must be a positive integer.")
        if mode not in ["token", "recursive"]:
            raise ValueError("Mode must be one of: token, recursive.")
        if method not in ["suffix", "prefix"]:
            raise ValueError("Method must be one of: suffix, prefix.")
        if not isinstance(merge, bool):
            raise ValueError("Merge must be a boolean.")
        if not isinstance(inplace, bool):
            raise ValueError("Inplace must be a boolean.")

        # Initialize the refinery
        self.tokenizer = Tokenizer(tokenizer_or_token_counter)
        self.context_size = context_size
        self.mode = mode
        self.method = method
        self.merge = merge
        self.inplace = inplace
        self.rules = rules
        self.sep = '✄'
        
        # Performance optimization: Add caches for repeated operations
        self._token_cache: Dict[str, list] = {}  # Cache for tokenizer.encode() results
        self._count_cache: Dict[str, int] = {}  # Cache for token count results


    def _is_available(self) -> bool:
        """Check if the refinery is available."""
        return True

    def _split_text(self, text: str, recursive_level: RecursiveLevel) -> list[str]:
        """Split the text into chunks using the delimiters."""
        # At every delimiter, replace it with the sep
        if recursive_level.whitespace:
            splits = text.split(" ")
        elif recursive_level.delimiters:
            if recursive_level.include_delim == "prev":
                for delimiter in recursive_level.delimiters:
                    text = text.replace(delimiter, delimiter + self.sep)
            elif recursive_level.include_delim == "next":
                for delimiter in recursive_level.delimiters:
                    text = text.replace(delimiter, self.sep + delimiter)
            else:
                for delimiter in recursive_level.delimiters:
                    text = text.replace(delimiter, self.sep)
            splits = [split for split in text.split(self.sep) if split != ""]
        else:
            # Encode, Split, and Decode
            encoded = self.tokenizer.encode(text)
            context_size_int = int(self.context_size) if isinstance(self.context_size, (int, float)) else self.context_size
            token_splits = [
                encoded[i : i + context_size_int]
                for i in range(0, len(encoded), context_size_int)]
            splits = list(self.tokenizer.decode_batch(token_splits))

        # Some splits may not be meaningful yet.
        # This will be handled during chunk creation.
        return splits

    def _get_token_counts_cached(self, splits: List[str]) -> List[int]:
        """Get token counts with caching for performance optimization."""
        cached_counts = []
        uncached_texts = []
        uncached_indices = []
        
        # Separate cached and uncached texts
        for i, split in enumerate(splits):
            if split in self._count_cache:
                cached_counts.append((i, self._count_cache[split]))
            else:
                uncached_texts.append(split)
                uncached_indices.append(i)
        
        # Batch process uncached texts
        if uncached_texts:
            new_counts = list(self.tokenizer.count_tokens_batch(uncached_texts))
            # Cache the results
            for text, count in zip(uncached_texts, new_counts):
                self._count_cache[text] = count
            
            # Add to cached_counts
            for idx, count in zip(uncached_indices, new_counts):
                cached_counts.append((idx, count))
        
        # Sort by original index and extract counts
        cached_counts.sort(key=lambda x: x[0])
        return [count for _, count in cached_counts]

    def _group_splits(self, splits: List[str], token_counts: List[int]) -> List[str]:
        """Group the splits.

        Args:
            splits: The splits to merge.
            token_counts: The token counts of the splits.

        Returns:
            The grouped splits.

        """
        group = []
        current_token_count = 0
        for token_count, split in zip(token_counts, splits):
            if current_token_count + token_count < self.context_size:
                group.append(split)
                current_token_count += token_count
            else:
                break
        return group


    def _prefix_overlap_token(self, chunk: Chunk) -> str:
        """Calculate token-based overlap context using tokenizer.

        Takes a larger window of text from the chunk end, tokenizes it,
        and selects exactly context_size tokens worth of text.

        Args:
            chunk: The chunk to calculate the overlap context for.

        Returns:
            The overlap context.

        """
        # Performance optimization: Cache tokenization results
        if chunk.text in self._token_cache:
            tokens = self._token_cache[chunk.text]
        else:
            tokens = self.tokenizer.encode(chunk.text)
            self._token_cache[chunk.text] = tokens
            
        if self.context_size > len(tokens):
            warnings.warn("Context size is greater than the chunk size. The entire chunk will be returned as the context.")
            return chunk.text
        else:
            assert isinstance(self.context_size, int), "Context size must be an integer."
            return self.tokenizer.decode(tokens[-self.context_size:])
    
    def _recursive_overlap(self, text: str, level: int, method: Literal["prefix", "suffix"]) -> str:
        """Calculate recursive overlap context.
        
        Args:
            text: The text to calculate the overlap context for.
            level: The recursive level to use.
            method: The method to use for the context.
        
        Returns:
            The overlap context.

        """
        if text == "":
            return ""
            
        # Check if we've exceeded the available recursive levels
        if level >= len(self.rules):
            return text
            
        # Split the Chunk text based on the recursive rules
        recursive_level = self.rules[level]
        if recursive_level is None:
            return text
        splits = self._split_text(text, recursive_level)

        if method == "prefix":
            splits = splits[::-1]

        # Performance optimization: Get token counts with caching
        token_counts = self._get_token_counts_cached(splits)

        # Group the splits
        grouped_splits = self._group_splits(splits, token_counts)

        # If the grouped splits is empty, then we need to recursively split the first split
        if not grouped_splits:
            return self._recursive_overlap(splits[0], level+1, method)

        if method == "prefix":
            grouped_splits = grouped_splits[::-1]

        # Return the final context
        context = "".join(grouped_splits)
        return context

    
    def _prefix_overlap_recursive(self, chunk: Chunk) -> str:
        """Calculate recursive overlap context.

        Takes a larger window of text from the chunk end, tokenizes it,
        and selects exactly context_size tokens worth of text.
        
        """
        return self._recursive_overlap(chunk.text, 0, "prefix")

    def _get_prefix_overlap_context(self, chunk: Chunk) -> str:
        """Get the prefix overlap context.

        Args:
            chunk: The chunk to get the prefix overlap context for.

        """
        # Route to the appropriate method
        if self.mode == "token":
            return self._prefix_overlap_token(chunk)
        elif self.mode == "recursive":
            return self._prefix_overlap_recursive(chunk)
        else:
            raise ValueError("Mode must be one of: token, recursive.")

    def _refine_prefix(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine the prefix of the chunk.
        
        Args:
            chunks: The chunks to refine.

        Returns:
            The refined chunks.

        """
        # Iterate over the chunks till the second to last chunk
        for i, chunk in enumerate(chunks[1:]):
            # Get the previous chunk, since i starts from 0 
            prev_chunk = chunks[i]

            # Calculate the overlap context
            context = self._get_prefix_overlap_context(prev_chunk)

            # Set it as a part of the chunk
            setattr(chunk, "context", context)
            
            # Merge the context if merge is True
            if self.merge:
                chunk.text = context + chunk.text
                # Note: We don't adjust start_index/end_index when adding context
                # because they should represent the original document positions.
                # The context is additional information, not part of the original chunk position.

                # Performance optimization: Update the token count with caching
                if self.tokenizer:
                    if context in self._count_cache:
                        context_tokens = self._count_cache[context]
                    else:
                        context_tokens = self.tokenizer.count_tokens(context)
                        self._count_cache[context] = context_tokens
                    chunk.token_count += context_tokens

        return chunks


    def _suffix_overlap_token(self, chunk: Chunk) -> str:
        """Calculate token-based overlap context using tokenizer.

        Takes a larger window of text from the chunk start, tokenizes it,
        and selects exactly context_size tokens worth of text.
        
        """
        # Performance optimization: Cache tokenization results
        if chunk.text in self._token_cache:
            tokens = self._token_cache[chunk.text]
        else:
            tokens = self.tokenizer.encode(chunk.text)
            self._token_cache[chunk.text] = tokens
            
        if self.context_size > len(tokens):
            warnings.warn("Context size is greater than the chunk size. The entire chunk will be returned as the context.")
            return chunk.text
        else:
            assert isinstance(self.context_size, int), "Context size must be an integer."
            return self.tokenizer.decode(tokens[:self.context_size])
    
    def _suffix_overlap_recursive(self, chunk: Chunk) -> str:
        """Calculate recursive overlap context.

        Takes a larger window of text from the chunk start, tokenizes it,
        and selects exactly context_size tokens worth of text.
        
        """
        return self._recursive_overlap(chunk.text, 0, "suffix")

    def _get_suffix_overlap_context(self, chunk: Chunk) -> str:
        """Get the suffix overlap context.

        Args:
            chunk: The chunk to get the suffix overlap context for.

        """
        # Route to the appropriate method
        if self.mode == "token":
            return self._suffix_overlap_token(chunk)
        elif self.mode == "recursive":
            return self._suffix_overlap_recursive(chunk)
        else:
            raise ValueError("Mode must be one of: token, recursive.")

    def _refine_suffix(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine the suffix of the chunk.
        
        Args:
            chunks: The chunks to refine.

        Returns:
            The refined chunks.

        """
        # Iterate over the chunks till the second to last chunk
        for i, chunk in enumerate(chunks[:-1]):
            # Get the previous chunk
            prev_chunk = chunks[i+1]

            # Calculate the overlap context
            context = self._get_suffix_overlap_context(prev_chunk)

            # Set it as a part of the chunk
            setattr(chunk, "context", context)
            
            # Merge the context if merge is True
            if self.merge:
                chunk.text = chunk.text + context
                # Note: We don't adjust start_index/end_index when adding context
                # because they should represent the original document positions.
                # The context is additional information, not part of the original chunk position.

                # Performance optimization: Update the token count with caching
                if self.tokenizer:
                    if context in self._count_cache:
                        context_tokens = self._count_cache[context]
                    else:
                        context_tokens = self.tokenizer.count_tokens(context)
                        self._count_cache[context] = context_tokens
                    chunk.token_count += context_tokens

        return chunks

    def _get_overlap_context_size(self, chunks: List[Chunk]) -> int:
        """Get the overlap context size.
        
        Args:
            chunks: The chunks to get the overlap context size for.

        """
        # Calculate context size for each call (float context size depends on chunk set)
        if isinstance(self.context_size, float):
            return int(self.context_size * max(chunk.token_count for chunk in chunks))
        else:
            return self.context_size


    def refine(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine the chunks based on the overlap.
        
        Args:
            chunks: The chunks to refine.

        Returns:
            The refined chunks.

        """
        # Check if the chunks are empty
        if not chunks:
            return chunks

        # Check if all the chunks are of the same type
        if len(set(type(chunk) for chunk in chunks)) > 1:
            raise ValueError("All chunks must be of the same type.")

        # If inplace is False, make a copy of the chunks
        if not self.inplace:
            chunks = [chunk.copy() for chunk in chunks]
        
        # Get a rough estimate of the overlap if the context size is a float
        self.context_size = self._get_overlap_context_size(chunks)

        # Refine the chunks based on the method
        if self.method == "prefix":
            return self._refine_prefix(chunks)
        elif self.method == "suffix":
            return self._refine_suffix(chunks)
        else:
            raise ValueError("Method must be one of: prefix, suffix.")
        
    def __repr__(self) -> str:
        """Return the string representation of the refinery."""
        return (f"OverlapRefinery(tokenizer={self.tokenizer}, "
                f"context_size={self.context_size}, "
                f"mode={self.mode}, method={self.method}, "
                f"merge={self.merge}, inplace={self.inplace})")
