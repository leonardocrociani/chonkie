"""Test for the Chonkie Cloud Token Chunker class."""

import os

import pytest

from chonkie.cloud import TokenChunker


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_token_chunker_initialization() -> None:
    """Test that the token chunker can be initialized."""
    # Check if the chunk_size < 0 raises an error
    with pytest.raises(ValueError):
        TokenChunker(tokenizer="gpt2", chunk_size=-1, chunk_overlap=0)

    # Check if the chunk_overlap < 0 raises an error
    with pytest.raises(ValueError):
        TokenChunker(tokenizer="gpt2", chunk_size=512, chunk_overlap=-1)
    
    # Check if the return_type is not "texts" or "chunks" raises an error
    with pytest.raises(ValueError):
        TokenChunker(tokenizer="gpt2", chunk_size=512, chunk_overlap=0, return_type="bad_return_type")

    # Finally, check if the attributes are set correctly
    chunker = TokenChunker(tokenizer="gpt2", chunk_size=512, chunk_overlap=0)
    assert chunker.tokenizer == "gpt2"
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 0
    assert chunker.return_type == "chunks"

@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_token_chunker_simple() -> None:
    """Test that the token chunker works."""
    token_chunker = TokenChunker(
        tokenizer="gpt2",
        chunk_size=512,
        chunk_overlap=0,
    )
    result = token_chunker("Hello, world!")

    # Check the result
    assert isinstance(result, list) and isinstance(result[0], dict) and len(result) == 1
    assert result[0]["text"] == "Hello, world!"
    assert result[0]["token_count"] == 4
    assert result[0]["start_index"] == 0
    assert result[0]["end_index"] == 13

@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_token_chunker_multiple_sentences() -> None:
    """Test that the token chunker works with a complex text."""
    token_chunker = TokenChunker(
        tokenizer="gpt2",
        chunk_size=5,
        chunk_overlap=0,
    )
    text = "This is one sentence. This is another sentence. This is a third sentence."
    result = token_chunker(text)

    # Check the result
    assert len(result) > 1
    assert isinstance(result, list)
    assert all(isinstance(item, dict) for item in result)
    assert all(isinstance(item["text"], str) for item in result)
    assert all(isinstance(item["token_count"], int) for item in result)
    assert all(isinstance(item["start_index"], int) for item in result)
    assert all(isinstance(item["end_index"], int) for item in result)

@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_token_chunker_batch() -> None:
    """Test that the token chunker works with a batch of texts."""
    token_chunker = TokenChunker(
        tokenizer="gpt2",
        chunk_size=512,
        chunk_overlap=0,
    )
    texts = ["Hello, world!", "This is another sentence.", "This is a third sentence."]
    result = token_chunker(texts)

    # Check the result
    assert len(result) == len(texts)
    assert isinstance(result, list)
    assert all(isinstance(item, list) for item in result), \
        f"Expected a list of lists, got {type(result)}"
    assert all(isinstance(item, dict) for item in result[0]), \
        f"Expected a list of dictionaries, got {type(result[0])}"
    assert all(isinstance(item["text"], str) for item in result[0]), \
        f"Expected a list of dictionaries with a 'text' key, got {type(result[0])}"
    assert all(isinstance(item["token_count"], int) for item in result[0]), \
        f"Expected a list of dictionaries with a 'token_count' key, got {type(result[0])}"
    assert all(isinstance(item["start_index"], int) for item in result[0]), \
        f"Expected a list of dictionaries with a 'start_index' key, got {type(result[0])}"
    assert all(isinstance(item["end_index"], int) for item in result[0]), \
        f"Expected a list of dictionaries with a 'end_index' key, got {type(result[0])}"


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_token_chunker_return_type() -> None:
    """Test that the token chunker works with a return type."""
    token_chunker = TokenChunker(
        tokenizer="gpt2",
        chunk_size=512,
        chunk_overlap=0,
        return_type="texts",
    )
    result = token_chunker("Hello, world!")

    # Check the result
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)