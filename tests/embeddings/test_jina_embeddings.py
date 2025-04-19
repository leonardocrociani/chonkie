"""Test suite for JinaEmbeddings."""
import os

import numpy as np
import pytest
from transformers import PreTrainedTokenizerFast

from chonkie.embeddings.jina import JinaEmbeddings

# --- Fixtures ---

@pytest.fixture(scope="module")
def embedding_model():
    """Fixture to create a JinaEmbeddings instance using environment API key.

    Returns:
        JinaEmbeddings: An initialized JinaEmbeddings instance or skips if key missing.

    """
    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        pytest.skip("Skipping Jina integration tests because JINA_API_KEY is not defined")
    return JinaEmbeddings(api_key=api_key)

@pytest.fixture
def sample_text():
    """Fixture for a single sample text.

    Returns:
        str: A sample text string.

    """
    return "This is a sample text for testing Jina embeddings."

@pytest.fixture
def sample_texts():
    """Fixture for a batch of sample texts.

    Returns:
        List[str]: A list of sample text strings.

    """
    return [
        "This is the first sample text for Jina.",
        "Here is another example sentence for batch processing.",
        "Testing Jina embeddings with multiple sentences.",
    ]

# --- Test Cases ---    

# Decorator to skip tests if API key is not available
skip_if_no_key = pytest.mark.skipif(
    "JINA_API_KEY" not in os.environ,
    reason="Skipping test because JINA_API_KEY is not defined",
)

@skip_if_no_key
def test_initialization_with_env_key(embedding_model):
    """Test JinaEmbeddings initialization using environment API key and defaults.

    Args:
        embedding_model: The JinaEmbeddings instance fixture.

    """
    assert embedding_model.model == "jina-embeddings-v3"
    assert embedding_model.task == "text-matching"
    assert embedding_model.late_chunking is True
    assert embedding_model.embedding_type == "float"
    assert embedding_model.api_key is not None
    assert embedding_model.headers["Authorization"].startswith("Bearer ")
    assert embedding_model.url == 'https://api.jina.ai/v1/embeddings'



@skip_if_no_key
def test_embed_single_text(embedding_model, sample_text):
    """Test embedding a single text using the live Jina API.

    Args:
        embedding_model: The JinaEmbeddings instance fixture.
        sample_text: The single sample text fixture.

    """
    embedding = embedding_model.embed([sample_text])
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.dimension,)


@skip_if_no_key
def test_embed_batch_texts_live(embedding_model, sample_texts):
    """Test embedding a batch of texts using the live Jina API.

    Args:
        embedding_model: The JinaEmbeddings instance fixture.
        sample_texts: The batch of sample texts fixture.

    """
    embeddings = embedding_model.embed_batch(sample_texts)
    assert isinstance(embeddings, list)
    if embeddings: 
        assert len(embeddings) == len(sample_texts)


@skip_if_no_key
def test_count_tokens_single_text(embedding_model, sample_text):
    """Test counting tokens for a single text using the live Jina API.

    Args:
        embedding_model: The JinaEmbeddings instance fixture.
        sample_text: The single sample text fixture.

    """
    token_count = embedding_model.count_tokens(sample_text)
    assert isinstance(token_count, int)
    assert token_count > 0

@skip_if_no_key
def test_count_tokens_batch_texts(embedding_model, sample_texts):
    """Test counting tokens for a batch of texts using the live Jina API.

    Args:
        embedding_model: The JinaEmbeddings instance fixture.
        sample_texts: The batch of sample texts fixture.

    """
    token_counts = embedding_model.count_tokens_batch(sample_texts)
    assert isinstance(token_counts, list)
    assert len(token_counts) == len(sample_texts)
    assert all(isinstance(count, int) for count in token_counts)
    assert all(count > 0 for count in token_counts)

@skip_if_no_key
def test_similarity(embedding_model, sample_texts):
    """Test similarity calculation between two embeddings from the live Jina API.

    Args:
        embedding_model: The JinaEmbeddings instance fixture.
        sample_texts: The batch of sample texts fixture.

    """
    if len(sample_texts) < 2:
        pytest.skip("Need at least two sample texts for similarity test")

    # Embed only the first two texts using the embed method.
    embedding1 = embedding_model.embed([sample_texts[0]])
    embedding2 = embedding_model.embed([sample_texts[1]])

    similarity_score = embedding_model.similarity(embedding1, embedding2)
    assert isinstance(similarity_score, (float, np.floating))
    assert 0 <= similarity_score <= 1

@skip_if_no_key
def test_dimension_property(embedding_model):
    """Test the dimension property returns the correct value.

    Args:
        embedding_model: The JinaEmbeddings instance fixture.

    """
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension == 1024 # Check default dimension



@skip_if_no_key
def test_get_tokenizer_or_token_counter(embedding_model, sample_text):
    """Test get_tokenizer_or_token_counter returns the correct tokenizer instance.

    Args:
        embedding_model: The JinaEmbeddings instance fixture.
        sample_text: The single sample text fixture.

    """
    tokenizer_obj = embedding_model.get_tokenizer_or_token_counter()

    assert tokenizer_obj is embedding_model._tokenizer

    assert isinstance(tokenizer_obj, PreTrainedTokenizerFast)

    tokens = tokenizer_obj.tokenize(sample_text)
    assert tokens is not None
    assert isinstance(tokens, list)
    if len(sample_text.strip()) > 0:
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
    else:
        assert len(tokens) == 0 

    expected_token_count_via_method = embedding_model.count_tokens(sample_text)
    assert len(tokens) == expected_token_count_via_method

@skip_if_no_key
def test_repr(embedding_model):
    """Test the __repr__ method.

    Args:
        embedding_model: The JinaEmbeddings instance fixture.

    """
    repr_str = repr(embedding_model)
    assert isinstance(repr_str, str)
    assert repr_str.startswith("JinaEmbeddings")


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])