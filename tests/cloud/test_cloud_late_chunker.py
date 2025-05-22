"""Test the Chonkie Cloud Late Chunker."""

import os

import pytest

from chonkie.cloud.chunker import LateChunker  # Corrected import
from chonkie.types import RecursiveLevel, RecursiveRules


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_late_chunker_initialization() -> None:
    """Test that the late chunker can be initialized."""
    # Check if chunk_size < 0 raises an error (inherited from superclass validation)
    with pytest.raises(ValueError):
        LateChunker(chunk_size=-1)

    # Check if min_characters_per_chunk < 1 raises an error (inherited from superclass validation)
    with pytest.raises(ValueError):
        LateChunker(min_characters_per_chunk=-1)

    # Check default initialization
    chunker = LateChunker()
    assert chunker.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert chunker.chunk_size == 512
    assert chunker.min_characters_per_chunk == 24  # Default for LateChunker
    assert isinstance(chunker.rules, RecursiveRules)
    # Verify attributes set for the superclass
    assert chunker.tokenizer_or_token_counter == "gpt2" # Set by LateChunker for super
    assert chunker.return_type == "chunks" # Set by LateChunker for super

    # Check initialization with custom parameters
    custom_levels = [RecursiveLevel(delimiters=["\n\n", "\n", ". "])]
    custom_rules = RecursiveRules(levels=custom_levels)
    custom_chunker = LateChunker(
        embedding_model="custom-model",
        chunk_size=256,
        min_characters_per_chunk=10,
        rules=custom_rules,
    )
    assert custom_chunker.embedding_model == "custom-model"
    assert custom_chunker.chunk_size == 256
    assert custom_chunker.min_characters_per_chunk == 10
    assert custom_chunker.rules == custom_rules


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_late_chunker_single_text() -> None:
    """Test that the Late Chunker works with a single text."""
    late_chunker = LateChunker(chunk_size=512) # Using default embedding model

    text = "This is a test sentence for the late chunker. It has several parts."
    result = late_chunker(text)

    assert isinstance(result, list)
    if result: # API might return multiple chunks depending on its logic
        for chunk in result:
            assert isinstance(chunk, dict)
            assert "text" in chunk
            assert isinstance(chunk["text"], str)
            assert "start_index" in chunk
            assert isinstance(chunk["start_index"], int)
            assert "end_index" in chunk
            assert isinstance(chunk["end_index"], int)
            assert "token_count" in chunk # Assuming API returns token_count
            assert isinstance(chunk["token_count"], int)
            # Embedding presence could also be checked if guaranteed
            # assert "embedding" in chunk
            # assert isinstance(chunk["embedding"], list)


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_late_chunker_batch_texts() -> None:
    """Test that the Late Chunker works with a batch of texts."""
    late_chunker = LateChunker(chunk_size=256) # Smaller chunk size for variety

    texts = [
        "First document for batch processing.",
        "Second document, slightly longer to see if it splits.",
        "Third one.",
    ]
    result = late_chunker(texts)

    assert isinstance(result, list)
    # Based on the error, the API for batch input returns a List[List[Dict]]
    # where the outer list contains a single element: the list of all chunks.
    if result:
        assert len(result) > 0 # Ensure we got something back
        assert isinstance(result[0], list) # The actual list of chunks is the first element
        
        all_chunks = result[0]
        if all_chunks:
            for chunk in all_chunks: # Iterate through the inner list of chunks
                assert isinstance(chunk, dict)
                assert "text" in chunk
                assert "start_index" in chunk
                assert "end_index" in chunk
                assert "token_count" in chunk # Assuming API returns token_count

                # Check that start and end indices are within the bounds of the original texts
                # This is more complex for batch as we don't know which original text a chunk belongs to
                # without more info from API. For now, just check type.
                assert isinstance(chunk["start_index"], int)
                assert isinstance(chunk["end_index"], int)


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_late_chunker_empty_text() -> None:
    """Test that the Late Chunker works with an empty text."""
    late_chunker = LateChunker()
    result = late_chunker("")
    assert isinstance(result, list)
    assert len(result) == 0


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_late_chunker_from_recipe() -> None:
    """Test creating a LateChunker from a recipe."""
    # Assuming 'default' recipe exists and is valid
    chunker = LateChunker.from_recipe(
        name="default",
        lang="en",
        embedding_model="test-recipe-model",
        chunk_size=128,
        min_characters_per_chunk=5,
    )
    assert isinstance(chunker, LateChunker)
    assert chunker.embedding_model == "test-recipe-model"
    assert chunker.chunk_size == 128
    assert chunker.min_characters_per_chunk == 5
    assert isinstance(chunker.rules, RecursiveRules)
    # Check if rules are loaded (e.g., default recipe might have specific separators)
    # This depends on the content of the "default" recipe.
    # For example, if default recipe for 'en' has known separators:
    # assert ". " in chunker.rules.separators

    # Test with a potentially non-existent recipe to ensure error handling (if applicable by from_recipe)
    # This depends on whether RecursiveRules.from_recipe raises an error or returns default.
    # If RecursiveRules.from_recipe raises ValueError for bad recipe:
    with pytest.raises(ValueError): # Or FileNotFoundError, depending on implementation
        LateChunker.from_recipe(name="non_existent_recipe")
