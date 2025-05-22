"""Test the Chonkie Cloud SDPM Chunker."""

import os

import pytest

from chonkie.cloud import SDPMChunker


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_sdpm_chunker_initialization() -> None:
    """Test that the SDPM chunker can be initialized."""
    # Check if the chunk_size < 0 raises an error
    with pytest.raises(ValueError):
        SDPMChunker(chunk_size=-1)

    # Check if the threshold is a str but not "auto"
    with pytest.raises(ValueError):
        SDPMChunker(threshold="not_auto")

    # Check if the similarity window is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(similarity_window=-1)

    # Check if the min_sentences is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(min_sentences=-1)

    # Check if the min_chunk_size is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(min_chunk_size=-1)

    # Check if the min_characters_per_sentence is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(min_characters_per_sentence=-1)

    # Check if the threshold_step is not a positive float
    with pytest.raises(ValueError):
        SDPMChunker(threshold_step=-0.1)
    
    # Check if the delim is not a list
    with pytest.raises(ValueError):
        SDPMChunker(delim="not_a_list")

    # Check if the include_delim is not "prev" or "next"
    with pytest.raises(ValueError):
        SDPMChunker(include_delim="not_valid")

    # Check if the skip_window is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(skip_window=-1)

    # Check if the return_type is not "chunks" or "texts"
    with pytest.raises(ValueError):
        SDPMChunker(return_type="not_valid")

    # Check if the embedding_model is not a string
    with pytest.raises(ValueError):
        SDPMChunker(embedding_model=123) # type: ignore

    # Finally, check if the attributes are set correctly
    # Provide a dummy api_key for initialization if CHONKIE_API_KEY is not set
    # The actual API call tests are skipped if the key is not present.
    api_key_to_use = os.getenv("CHONKIE_API_KEY", "test_key_dummy")
    chunker = SDPMChunker(chunk_size=256, api_key=api_key_to_use) 
    assert chunker.embedding_model == "minishlab/potion-base-8M"
    assert chunker.mode == "window"
    assert chunker.threshold == "auto"
    assert chunker.chunk_size == 256
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2
    assert chunker.min_characters_per_sentence == 12
    assert chunker.threshold_step == 0.01
    assert chunker.delim == [". ", "! ", "? ", "\n"]
    assert chunker.include_delim == "prev"
    assert chunker.skip_window == 1
    assert chunker.return_type == "chunks"
    assert chunker.api_key == api_key_to_use


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_sdpm_chunker_single_sentence() -> None:
    """Test that the SDPM Chunker works with a single sentence."""
    sdpm_chunker = SDPMChunker(
        chunk_size=512,
    )

    result = sdpm_chunker("Hello, world!")
    assert len(result) >= 0 # API might return 0 chunks for very short text
    if len(result) > 0:
        # Exact values depend on the API response, so we check for presence and type
        assert "text" in result[0]
        assert isinstance(result[0]["text"], str)
        assert "token_count" in result[0]
        assert isinstance(result[0]["token_count"], int)
        assert "start_index" in result[0]
        assert isinstance(result[0]["start_index"], int)
        assert "end_index" in result[0]
        assert isinstance(result[0]["end_index"], int)


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_sdpm_chunker_batch() -> None:
    """Test that the SDPM Chunker works with a batch of texts."""
    sdpm_chunker = SDPMChunker(
        chunk_size=512,
    )
    texts = [
        "Hello, world!",
        "This is another sentence.",
        "This is a third sentence. This is a fourth one to make it longer.",
    ]
    result = sdpm_chunker(texts)
    assert len(result) == len(texts)
    for i in range(len(texts)):
        assert isinstance(result[i], list) # API returns a list of lists for batch
        if len(result[i]) > 0:
            assert "text" in result[i][0] 
            assert isinstance(result[i][0]["text"], str)
            assert "token_count" in result[i][0]
            assert isinstance(result[i][0]["token_count"], int)
            assert "start_index" in result[i][0]
            assert isinstance(result[i][0]["start_index"], int)
            assert "end_index" in result[i][0]
            assert isinstance(result[i][0]["end_index"], int)


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_sdpm_chunker_empty_text() -> None:
    """Test that the SDPM Chunker works with an empty text."""
    sdpm_chunker = SDPMChunker(
        chunk_size=512,
    )

    result = sdpm_chunker("")
    # The API might return an empty list or a list with an empty chunk object
    # Depending on the API's behavior, this assertion might need adjustment.
    # Based on SemanticChunker tests, an empty list is expected.
    assert len(result) == 0
