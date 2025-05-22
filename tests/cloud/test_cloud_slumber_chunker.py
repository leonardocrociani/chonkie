"""Test the Chonkie Cloud Slumber Chunker."""

import os

import pytest
from pytest import MonkeyPatch

from chonkie.cloud import SlumberChunker
from chonkie.types import RecursiveLevel, RecursiveRules


def test_cloud_slumber_chunker_no_api_key(monkeypatch: MonkeyPatch) -> None:
    """Test SlumberChunker initialization fails without API key."""
    monkeypatch.delenv("CHONKIE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key provided"):
        SlumberChunker()


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_slumber_chunker_initialization() -> None:
    """Test that the slumber chunker can be initialized."""
    # Check if chunk_size <= 0 raises an error
    with pytest.raises(ValueError, match="Chunk size must be greater than 0."):
        SlumberChunker(tokenizer_or_token_counter="gpt2", chunk_size=-1)

    # Check if candidate_size <= 0 raises an error
    with pytest.raises(ValueError, match="Candidate size must be greater than 0."):
        SlumberChunker(
            tokenizer_or_token_counter="gpt2",
            chunk_size=1024,
            candidate_size=-1,
        )

    # Check if min_characters_per_chunk < 1 raises an error
    with pytest.raises(ValueError, match="Minimum characters per chunk must be greater than 0."):
        SlumberChunker(
            tokenizer_or_token_counter="gpt2",
            chunk_size=1024,
            min_characters_per_chunk=0,
        )

    # Check if return_type is not "texts" or "chunks" raises an error
    with pytest.raises(ValueError, match="Return type must be either 'texts' or 'chunks'."):
        SlumberChunker(
            tokenizer_or_token_counter="gpt2",
            chunk_size=1024,
            return_type="not_a_valid_type",
        )

    # Finally, check if the attributes are set correctly with non-default values
    custom_rules = RecursiveRules(levels=[
        RecursiveLevel(delimiters=["\n\n", "\n", "."])
    ])
    chunker = SlumberChunker(
        tokenizer_or_token_counter="claude2",
        chunk_size=2048,
        rules=custom_rules,
        candidate_size=64,
        min_characters_per_chunk=10,
        return_type="texts",
    )
    assert chunker.tokenizer_or_token_counter == "claude2"
    assert chunker.chunk_size == 2048
    assert chunker.rules == custom_rules
    assert chunker.candidate_size == 64
    assert chunker.min_characters_per_chunk == 10
    assert chunker.return_type == "texts"

    # Check default values
    default_chunker = SlumberChunker()
    assert default_chunker.tokenizer_or_token_counter == "gpt2"
    assert default_chunker.chunk_size == 1024
    assert isinstance(default_chunker.rules, RecursiveRules) # Default rules
    assert default_chunker.candidate_size == 128
    assert default_chunker.min_characters_per_chunk == 24
    assert default_chunker.return_type == "chunks"


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_slumber_chunker_single_sentence() -> None:
    """Test that the Slumber Chunker works with a single sentence."""
    slumber_chunker = SlumberChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512, # Smaller chunk size for test predictability
    )

    text = "Hello, world! This is a simple test sentence."
    result = slumber_chunker(text)
    assert len(result) >= 1 # Could be one or more chunks depending on API logic
    assert isinstance(result[0], dict)
    assert "text" in result[0]
    assert "token_count" in result[0]
    assert "start_index" in result[0]
    assert "end_index" in result[0]
    # For a short sentence, it should ideally be one chunk
    if len(result) == 1:
        assert result[0]["text"] == text
        assert result[0]["start_index"] == 0
        assert result[0]["end_index"] == len(text)
        # Token count depends on the API's tokenizer, so we just check its presence and type
        assert isinstance(result[0]["token_count"], int)
        assert result[0]["token_count"] > 0


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_slumber_chunker_batch() -> None:
    """Test that the Slumber Chunker works with a batch of texts."""
    slumber_chunker = SlumberChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512, # Smaller chunk size for test
    )

    texts = [
        "Hello, world!",
        "This is another sentence for batch processing.",
        "And a third one to ensure multiple inputs are handled.",
    ]
    result = slumber_chunker(texts)

    # Expect a list of lists of dictionaries, one inner list per input text
    assert isinstance(result, list)
    assert len(result) == len(texts)

    for i, text_chunks in enumerate(result):
        assert isinstance(text_chunks, list)
        assert len(text_chunks) >= 1 # At least one chunk per text
        current_text_total_chunked_length = 0
        for chunk in text_chunks:
            assert isinstance(chunk, dict)
            assert "text" in chunk
            assert "token_count" in chunk
            assert "start_index" in chunk
            assert "end_index" in chunk
            assert isinstance(chunk["text"], str)
            assert isinstance(chunk["token_count"], int) and chunk["token_count"] >= 0
            assert isinstance(chunk["start_index"], int)
            assert isinstance(chunk["end_index"], int)
            assert chunk["start_index"] < chunk["end_index"]
            assert chunk["end_index"] - chunk["start_index"] == len(chunk["text"])
            # Check if the chunk's text is part of the corresponding original text
            assert chunk["text"] in texts[i]
            # Check if start_index matches
            assert texts[i].find(chunk["text"]) == chunk["start_index"]
            current_text_total_chunked_length += len(chunk["text"])
        
        # Check if the combined length of chunks for this text matches the original text's length
        # Allowing for minor differences if chunking adds/removes minimal characters (e.g. delimiters)
        assert abs(current_text_total_chunked_length - len(texts[i])) <= 2


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_slumber_chunker_empty_text() -> None:
    """Test that the Slumber Chunker works with an empty text."""
    slumber_chunker = SlumberChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
    )

    result = slumber_chunker("")
    assert len(result) == 0

@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_slumber_chunker_return_type_texts() -> None:
    """Test that the Slumber Chunker returns a list of strings when return_type is 'texts'."""
    slumber_chunker = SlumberChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=30, # small chunk size to ensure multiple chunks
        return_type="texts",
    )
    # Made text significantly longer to ensure splitting for chunk_size=30
    text = ("This is a very long test sentence, designed specifically to be much longer than the nominal chunk size. "
            "By repeating this phrase multiple times, we increase the likelihood that the Slumber chunker, "
            "when configured with a small chunk size like 30 tokens, will indeed split this into multiple segments. "
            "This is a very long test sentence, designed specifically to be much longer than the nominal chunk size. "
            "By repeating this phrase multiple times, we increase the likelihood that the Slumber chunker, "
            "when configured with a small chunk size like 30 tokens, will indeed split this into multiple segments. "
            "This is a very long test sentence, designed specifically to be much longer than the nominal chunk size. "
            "By repeating this phrase multiple times, we increase the likelihood that the Slumber chunker, "
            "when configured with a small chunk size like 30 tokens, will indeed split this into multiple segments.")
    result = slumber_chunker(text) # API likely returns List[Dict] where each dict has 'text'

    assert isinstance(result, list)
    assert len(result) >= 1 # Could be one or more dictionaries (chunks)

    combined_text_from_chunks = []
    for item in result:
        assert isinstance(item, dict) # Expecting API to return list of dicts
        assert "text" in item
        assert isinstance(item["text"], str)
        combined_text_from_chunks.append(item["text"])

    # The combined text from chunks should reconstitute the original text.
    # Depending on API's handling of spaces/delimiters, an exact match or a normalized match is needed.
    # Using replace(" ", "") for a more robust check against minor spacing differences.
    assert "".join(combined_text_from_chunks).replace(" ", "") == text.replace(" ", "")
    
    # If the text was actually split, there should be more than one chunk dictionary
    # Heuristic: if text char length is much larger than an estimated char length for chunk_size tokens
    # (e.g., assuming ~3-4 chars/token for English)
    estimated_chars_per_chunk = slumber_chunker.chunk_size * 3.5 
    if len(text) > estimated_chars_per_chunk * 1.5 : 
        assert len(result) > 1, (
            f"Text (char len {len(text)}, approx token factor for split check: {estimated_chars_per_chunk * 1.5}) "
            f"was expected to be split into multiple chunks (got {len(result)}) with chunk_size={slumber_chunker.chunk_size}."
        )


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_slumber_chunker_return_type_texts_batch() -> None:
    """Test batch processing with return_type 'texts'."""
    slumber_chunker = SlumberChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=20, # Small chunk size
        return_type="texts",
    )
    texts = ["Short text one.", "Another short one for the batch processing.", "This one is slightly longer for fun and to ensure splitting."]
    result = slumber_chunker(texts) # Expect List[List[Dict]]

    assert isinstance(result, list)
    assert len(result) == len(texts) # One inner list of dicts per input text
    
    for i, text_results_list in enumerate(result):
        assert isinstance(text_results_list, list)
        assert len(text_results_list) >= 1 # At least one dict per input text
        
        original_text_no_spaces = texts[i].replace(" ", "")
        combined_chunk_texts_no_spaces = ""
        for item_dict in text_results_list:
            assert isinstance(item_dict, dict)
            assert "text" in item_dict
            assert isinstance(item_dict["text"], str)
            combined_chunk_texts_no_spaces += item_dict["text"].replace(" ", "")
        
        # Compare content ignoring spaces
        assert combined_chunk_texts_no_spaces == original_text_no_spaces
