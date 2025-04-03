"""Test for the Chonkie Cloud Sentence Chunker class."""

import os

import pytest

from chonkie.cloud import SentenceChunker


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_sentence_chunker_initialization() -> None:
    """Test that the sentence chunker can be initialized."""
    # Check if the chunk_size < 0 raises an error
    with pytest.raises(ValueError):
        SentenceChunker(tokenizer_or_token_counter="gpt2", chunk_size=-1, chunk_overlap=0)

    # Check if the chunk_overlap < 0 raises an error
    with pytest.raises(ValueError):
        SentenceChunker(tokenizer_or_token_counter="gpt2", chunk_size=512, chunk_overlap=-1)

    # Check if the min_sentences_per_chunk < 1 raises an error
    with pytest.raises(ValueError):
        SentenceChunker(tokenizer_or_token_counter="gpt2", chunk_size=512, chunk_overlap=0, min_sentences_per_chunk=-1)

    # Check if the min_characters_per_sentence < 1 raises an error
    with pytest.raises(ValueError):
        SentenceChunker(tokenizer_or_token_counter="gpt2", chunk_size=512, chunk_overlap=0, min_characters_per_sentence=-1)
    
    # Check if the approximate is not a boolean
    with pytest.raises(ValueError):
        SentenceChunker(tokenizer_or_token_counter="gpt2", chunk_size=512, chunk_overlap=0, approximate="not_a_boolean")

    # Check if the include_delim is not a string
    with pytest.raises(ValueError):
        SentenceChunker(tokenizer_or_token_counter="gpt2", chunk_size=512, chunk_overlap=0, include_delim="not_a_string")

    # Check if the return_type is not a string
    with pytest.raises(ValueError):
        SentenceChunker(tokenizer_or_token_counter="gpt2", chunk_size=512, chunk_overlap=0, return_type="not_a_string")

    # Finally, check if the attributes are set correctly
    chunker = SentenceChunker(tokenizer_or_token_counter="gpt2", chunk_size=512, chunk_overlap=0)
    assert chunker.tokenizer_or_token_counter == "gpt2"
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 0
    assert chunker.min_sentences_per_chunk == 1
    assert chunker.min_characters_per_sentence == 12
    assert chunker.approximate == True
    assert chunker.delim == [".", "!", "?", "\n"]
    assert chunker.include_delim == "prev"
    assert chunker.return_type == "chunks"

def test_cloud_sentence_chunker_simple() -> None:
    """Test that the sentence chunker works."""
    sentence_chunker = SentenceChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        chunk_overlap=0,
    )
    result = sentence_chunker("Hello, world!")

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
def test_cloud_sentence_chunker_multiple_sentences() -> None:
    """Test that the sentence chunker works with a complex text."""
    sentence_chunker = SentenceChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=5,
        chunk_overlap=0,
    )
    text = "This is one sentence. This is another sentence. This is a third sentence."
    result = sentence_chunker(text)

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
def test_cloud_sentence_chunker_batch() -> None:
    """Test that the sentence chunker works with a batch of texts."""
    sentence_chunker = SentenceChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        chunk_overlap=0,
    )
    texts = ["Hello, world!", "This is another sentence.", "This is a third sentence."]
    result = sentence_chunker(texts)

    # Check the result
    assert len(result) == len(texts)
    assert isinstance(result, list)
    assert all(isinstance(item, list) for item in result)
    assert all(isinstance(item, dict) for item in result[0])
    assert all(isinstance(item["text"], str) for item in result[0])
    assert all(isinstance(item["token_count"], int) for item in result[0])
    assert all(isinstance(item["start_index"], int) for item in result[0])