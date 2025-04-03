"""Test the Chonkie Cloud Semantic Chunker."""

import os

import pytest

from chonkie.cloud import SemanticChunker


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_semantic_chunker_initialization() -> None:
    """Test that the semantic chunker can be initialized."""
    # Check if the chunk_size < 0 raises an error
    with pytest.raises(ValueError):
        SemanticChunker(chunk_size=-1)

    # Check if the threshold is a str but not "auto"
    with pytest.raises(ValueError):
        SemanticChunker(threshold="not_auto")

    # Check if the threshold is a number but not between 0 and 1
    with pytest.raises(ValueError):
        SemanticChunker(threshold=1.1)

    # Check if the threshold is a number but not between 0 and 1
    with pytest.raises(ValueError):
        SemanticChunker(threshold=-0.1)
    
    # Check if the similarity window is not a positive integer
    with pytest.raises(ValueError):
        SemanticChunker(similarity_window=-1)

    # Check if the min_sentences is not a positive integer
    with pytest.raises(ValueError):
        SemanticChunker(min_sentences=-1)

    # Check if the min_chunk_size is not a positive integer
    with pytest.raises(ValueError):
        SemanticChunker(min_chunk_size=-1)

    # Check if the min_characters_per_sentence is not a positive integer
    with pytest.raises(ValueError):
        SemanticChunker(min_characters_per_sentence=-1)

    # Check if the threshold_step is not a number between 0 and 1
    with pytest.raises(ValueError):
        SemanticChunker(threshold_step=-0.1)

    # Check if the threshold_step is not a number between 0 and 1
    with pytest.raises(ValueError):
        SemanticChunker(threshold_step=1.1)

    # Check if the delim is not a string or a list of strings
    with pytest.raises(ValueError):
        SemanticChunker(delim=1)

    # Check if the include_delim is not a string or a list of strings
    with pytest.raises(ValueError):
        SemanticChunker(include_delim=1)

    # Check if the return_type is not "chunks" or "texts"
    with pytest.raises(ValueError):
        SemanticChunker(return_type="not_a_string")

    # Finally, check if the attributes are set correctly
    chunker = SemanticChunker(chunk_size=512)
    assert chunker.embedding_model == "minishlab/potion-base-32M"
    assert chunker.chunk_size == 512
    assert chunker.threshold == "auto"
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2
    assert chunker.min_characters_per_sentence == 12
    assert chunker.threshold_step == 0.01
    assert chunker.delim == [".", "!", "?", "\n"]
    assert chunker.include_delim == "prev"
    assert chunker.return_type == "chunks"

@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_semantic_chunker_single_sentence() -> None:
    """Test that the Semantic Chunker works with a single sentence."""
    semantic_chunker = SemanticChunker(
        chunk_size=512,
    )

    result = semantic_chunker("Hello, world!")
    assert len(result) == 1
    assert result[0]["text"] == "Hello, world!"
    assert result[0]["token_count"] == 4
    assert result[0]["start_index"] == 0
    assert result[0]["end_index"] == 13


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_semantic_chunker_batch() -> None:
    """Test that the Semantic Chunker works with a batch of texts."""
    semantic_chunker = SemanticChunker(
        chunk_size=512,
    )
    result = semantic_chunker(["Hello, world!", "This is another sentence.", "This is a third sentence."])
    assert len(result) == 3
    assert result[0][0]["text"] == "Hello, world!"
    assert result[0][0]["token_count"] == 4
    assert result[0][0]["start_index"] == 0
    assert result[0][0]["end_index"] == 13


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_semantic_chunker_empty_text() -> None:
    """Test that the Semantic Chunker works with an empty text."""
    semantic_chunker = SemanticChunker(
        chunk_size=512,
    )

    result = semantic_chunker("")
    assert len(result) == 0
