"""Test suite for VoyageAIEmbeddings."""

import os
import pytest
import numpy as np
from chonkie.embeddings.voyageai import VoyageAIEmbeddings


@pytest.fixture
def dummy_embedding_model():
    """Fixture for a VoyageAIEmbeddings instance with a dummy API key for non-API tests."""
    return VoyageAIEmbeddings(model="voyage-3", api_key="dummy_key")

@pytest.fixture
def api_embedding_model():
    """Fixture for a VoyageAIEmbeddings instance with the actual API key for API tests."""
    api_key = os.getenv("VOYAGEAI_API_KEY")
    if not api_key:
        pytest.skip("VOYAGEAI_API_KEY not set")
    return VoyageAIEmbeddings(model="voyage-3", api_key=api_key)

@pytest.fixture
def sample_short_text():
    """Fixture for a short sample text."""
    return "This is a short text."

@pytest.fixture
def sample_long_text():
    """Fixture for a long sample text likely exceeding token limits."""
    return "a " * 100000  # Approximately 100,000 tokens

@pytest.fixture
def sample_texts():
    """Fixture for a list of sample texts for batch testing."""
    return [
        "First sample text.",
        "Another example sentence.",
        "Multiple sentences for testing.",
    ]


def test_initialization_valid_model(dummy_embedding_model):
    """Test that VoyageAIEmbeddings initializes with a valid model name."""
    assert dummy_embedding_model.model == "voyage-3"
    assert dummy_embedding_model._dimension == 1024
    assert dummy_embedding_model._token_limit == 32000
    assert dummy_embedding_model.batch_size == 128
    assert dummy_embedding_model.truncation is True

def test_initialization_invalid_model():
    """Test that VoyageAIEmbeddings does not initializes and raises ValueError."""
    with pytest.raises(ValueError, match="Model 'invalid_model' not available"):
        VoyageAIEmbeddings(model="invalid_model", api_key="dummy_key")

def test_initialization_no_api_key():
    """Test that VoyageAIEmbeddings raises ValueError if no API key is provided."""
    if "VOYAGEAI_API_KEY" in os.environ:
        del os.environ["VOYAGEAI_API_KEY"]
    with pytest.raises(ValueError, match="VoyageAI API key not found"):
        VoyageAIEmbeddings(model="voyage-3")

def test_initialization_invalid_output_dimension():
    """Test that VoyageAIEmbeddings raises ValueError for invalid output_dimension."""
    with pytest.raises(ValueError, match="Invalid output_dimension"):
        VoyageAIEmbeddings(model="voyage-3", api_key="dummy_key", output_dimension=300)

def test_initialization_batch_size_cap():
    """Test that batch_size is capped at 128."""
    embedding = VoyageAIEmbeddings(model="voyage-3", api_key="dummy_key", batch_size=200)
    assert embedding.batch_size == 128


def test_count_tokens_short_text(dummy_embedding_model, sample_short_text):
    """Test token counting for a short text."""
    tokens = dummy_embedding_model.count_tokens(sample_short_text)
    assert isinstance(tokens, int)
    assert tokens > 0

def test_count_tokens_long_text_truncation(dummy_embedding_model, sample_long_text):
    """Test token counting for a long text with truncation."""
    dummy_embedding_model.truncation = True
    tokens = dummy_embedding_model.count_tokens(sample_long_text)
    assert tokens <= dummy_embedding_model._token_limit

def test_count_tokens_long_text_no_truncation(dummy_embedding_model, sample_long_text):
    """Test token counting for a long text with truncation disabled."""
    dummy_embedding_model.truncation = False
    tokens = dummy_embedding_model.count_tokens(sample_long_text)
    # Tokens may exceed _token_limit; behavior depends on tokenizer

def test_count_tokens_batch(dummy_embedding_model, sample_texts):
    """Test token counting for a batch of texts."""
    token_counts = dummy_embedding_model.count_tokens_batch(sample_texts)
    assert isinstance(token_counts, list)
    assert len(token_counts) == len(sample_texts)
    for count in token_counts:
        assert isinstance(count, int)
        assert count > 0

# Synchronous Embedding Tests
@pytest.mark.skipif(
    "VOYAGEAI_API_KEY" not in os.environ,
    reason="Skipping test because VOYAGEAI_API_KEY is not defined",
)
def test_embed_short_text(api_embedding_model, sample_short_text):
    """Test synchronous embedding of a short text."""
    embedding = api_embedding_model.embed(sample_short_text)
    assert isinstance(embedding, np.ndarray)
    expected_dim = api_embedding_model.output_dimension or api_embedding_model._dimension
    assert embedding.shape == (expected_dim,)
    assert embedding.dtype == np.float32

@pytest.mark.skipif(
    "VOYAGEAI_API_KEY" not in os.environ,
    reason="Skipping test because VOYAGEAI_API_KEY is not defined",
)
def test_embed_batch(api_embedding_model, sample_texts):
    """Test synchronous batch embedding."""
    embeddings = api_embedding_model.embed_batch(sample_texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    expected_dim = api_embedding_model.output_dimension or api_embedding_model._dimension
    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (expected_dim,)
        assert emb.dtype == np.float32

@pytest.mark.skipif(
    "VOYAGEAI_API_KEY" not in os.environ,
    reason="Skipping test because VOYAGEAI_API_KEY is not defined",
)
def test_embed_with_output_dimension():
    """Test embedding with a custom output_dimension."""
    api_key = os.getenv("VOYAGEAI_API_KEY")
    embedding_model = VoyageAIEmbeddings(model="voyage-3", api_key=api_key, output_dimension=256)
    embedding = embedding_model.embed("text")
    assert embedding.shape == (256,)

# Asynchronous Embedding Tests
@pytest.mark.anyio
@pytest.mark.skipif(
    "VOYAGEAI_API_KEY" not in os.environ,
    reason="Skipping test because VOYAGEAI_API_KEY is not defined",
)
async def test_aembed_short_text(api_embedding_model, sample_short_text):
    """Test asynchronous embedding of a short text."""
    embedding = await api_embedding_model.aembed(sample_short_text)
    assert isinstance(embedding, np.ndarray)
    expected_dim = api_embedding_model.output_dimension or api_embedding_model._dimension
    assert embedding.shape == (expected_dim,)
    assert embedding.dtype == np.float32

@pytest.mark.anyio
@pytest.mark.skipif(
    "VOYAGEAI_API_KEY" not in os.environ,
    reason="Skipping test because VOYAGEAI_API_KEY is not defined",
)
async def test_aembed_batch(api_embedding_model, sample_texts):
    """Test asynchronous batch embedding."""
    embeddings = await api_embedding_model.aembed_batch(sample_texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_texts)
    expected_dim = api_embedding_model.output_dimension or api_embedding_model._dimension
    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (expected_dim,)
        assert emb.dtype == np.float32

# Utility Tests
def test_dimension_property(dummy_embedding_model):
    """Test that VoyageAIEmbeddings correctly calculates the dimension property."""
    assert dummy_embedding_model.dimension == 1024

def test_is_available():
    """Test that VoyageAIEmbeddings correctly checks if it is available."""
    assert VoyageAIEmbeddings.is_available() is True

def test_repr(dummy_embedding_model):
    """Test that VoyageAIEmbeddings correctly returns a string representation."""
    assert repr(dummy_embedding_model) == "VoyageEmbeddings(model=voyage-3)"

def test_get_tokenizer_or_token_counter(dummy_embedding_model):
    """Test retrieval of the tokenizer."""
    tokenizer = dummy_embedding_model.get_tokenizer_or_token_counter()
    assert hasattr(tokenizer, "encode")  # Basic check for tokenizer functionality

if __name__ == "__main__":
    pytest.main()