"""Test the Chonkie Cloud Recursive Chunker."""

import os

import pytest

from chonkie.cloud import RecursiveChunker
from chonkie.types import RecursiveRules


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_recursive_chunker_initialization() -> None:
    """Test that the recursive chunker can be initialized."""
    # Check if the chunk_size < 0 raises an error
    with pytest.raises(ValueError):
        RecursiveChunker(tokenizer_or_token_counter="gpt2", chunk_size=-1)

    # Check if the min_characters_per_chunk < 1 raises an error
    with pytest.raises(ValueError):
        RecursiveChunker(tokenizer_or_token_counter="gpt2", chunk_size=512, min_characters_per_chunk=-1)

    # Check if the return_type is not "texts" or "chunks" raises an error
    with pytest.raises(ValueError):
        RecursiveChunker(tokenizer_or_token_counter="gpt2", chunk_size=512, return_type="not_a_string")
    
    # Finally, check if the attributes are set correctly
    chunker = RecursiveChunker(tokenizer_or_token_counter="gpt2", chunk_size=512)
    assert chunker.tokenizer_or_token_counter == "gpt2"
    assert chunker.chunk_size == 512
    assert chunker.min_characters_per_chunk == 12
    assert chunker.return_type == "chunks"
    assert isinstance(chunker.rules, RecursiveRules)

@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_recursive_chunker_single_sentence() -> None:
    """Test that the Recursive Chunker works with a single sentence."""
    recursive_chunker = RecursiveChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
    )

    result = recursive_chunker("Hello, world!")
    assert len(result) == 1
    assert result[0]["text"] == "Hello, world!"
    assert result[0]["token_count"] == 4
    assert result[0]["start_index"] == 0
    assert result[0]["end_index"] == 13


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_recursive_chunker_batch() -> None:
    """Test that the Recursive Chunker works with a batch of texts."""
    recursive_chunker = RecursiveChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
    )

    result = recursive_chunker(["Hello, world!", "This is another sentence.", "This is a third sentence."])
    assert len(result) == 3
    assert result[0][0]["text"] == "Hello, world!"
    assert result[0][0]["token_count"] == 4
    assert result[0][0]["start_index"] == 0
    assert result[0][0]["end_index"] == 13


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_recursive_chunker_empty_text() -> None:
    """Test that the Recursive Chunker works with an empty text."""
    recursive_chunker = RecursiveChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
    )

    result = recursive_chunker("")
    assert len(result) == 0