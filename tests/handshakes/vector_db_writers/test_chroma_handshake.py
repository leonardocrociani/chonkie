"""Test suite for the ChromaHandshake class."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock the chromadb library before importing the class under test
chromadb_mock = MagicMock()
collection_mock = MagicMock()
chromadb_mock.Client.return_value.get_or_create_collection.return_value = (
    collection_mock
)
chromadb_mock.Client.return_value.get_collection.return_value = collection_mock
chromadb_mock.utils.embedding_functions.DefaultEmbeddingFunction.return_value = "default_ef"

with patch.dict("sys.modules", {"chromadb": chromadb_mock}):
    from chonkie.handshakes.vector_db_writers.chroma import ChromaHandshake
    from chonkie.types import Chunk


@pytest.fixture
def mock_chroma_client():
    """Fixture for a mocked ChromaDB ClientAPI object."""
    client = MagicMock()
    client.get_or_create_collection.return_value = collection_mock
    client.get_collection.return_value = collection_mock
    return client


@pytest.fixture
def sample_chunks_chroma():
    """Fixture for sample Chunk objects for Chroma tests."""
    return [
        Chunk(
            text="Chroma chunk 1.",
            embedding=np.array([0.1, 0.2], dtype=np.float32),
            id="ch1",
            metadata={"doc": "docA", "page": 1},
            token_count=3,
            start_index=0,
            end_index=16,
        ),
        Chunk(
            text="Chroma chunk 2.",
            embedding=np.array([0.3, 0.4], dtype=np.float32),
            id="ch2",
            metadata={"doc": "docA", "page": 2},
            token_count=3,
            start_index=17,
            end_index=33,
        ),
    ]


@pytest.fixture
def sample_chunks_chroma_no_embeddings():
    """Fixture for sample Chunk objects without embeddings for Chroma tests."""
    return [
        Chunk(
            text="No embedding here.",
            id="ch3",
            metadata={"doc": "docB"},
            token_count=3,
            start_index=0,
            end_index=18,
        ),
        Chunk(
            text="Another without embedding.",
            id="ch4",
            metadata={"doc": "docB"},
            token_count=3,
            start_index=19,
            end_index=44,
        ),
    ]


# Test Initialization
def test_init_with_existing_client(mock_chroma_client):
    """Test initialization with an existing client."""
    handshake = ChromaHandshake(
        collection_name="test_coll",
        client=mock_chroma_client,
        create_collection_if_not_exists=False,
    )
    assert handshake.client == mock_chroma_client
    mock_chroma_client.get_collection.assert_called_once_with(name="test_coll")
    mock_chroma_client.get_or_create_collection.assert_not_called()
    assert handshake.collection == collection_mock


@patch("chonkie.handshakes.vector_db_writers.chroma.chromadb.Client")
def test_init_without_client_create_collection(mock_client_constructor):
    """Test initialization without client, creating collection."""
    mock_client_instance = mock_client_constructor.return_value
    mock_client_instance.get_or_create_collection.return_value = collection_mock
    custom_settings = MagicMock(spec=chromadb_mock.config.Settings)
    custom_ef = MagicMock()

    handshake = ChromaHandshake(
        collection_name="new_coll",
        client_settings=custom_settings,
        create_collection_if_not_exists=True,
        collection_metadata={"purpose": "testing"},
        embedding_function=custom_ef,
    )

    mock_client_constructor.assert_called_once_with(custom_settings)
    assert handshake.client == mock_client_instance
    mock_client_instance.get_or_create_collection.assert_called_once_with(
        name="new_coll",
        metadata={"purpose": "testing"},
        embedding_function=custom_ef,
    )
    mock_client_instance.get_collection.assert_not_called()
    assert handshake.collection == collection_mock


@patch("chonkie.handshakes.vector_db_writers.chroma.chromadb.Client")
def test_init_without_client_default_settings_and_ef(mock_client_constructor):
    """Test initialization uses default settings and EF if none provided."""
    mock_client_instance = mock_client_constructor.return_value
    mock_client_instance.get_or_create_collection.return_value = collection_mock

    handshake = ChromaHandshake(collection_name="default_coll")

    # Check if called with default Settings()
    mock_client_constructor.assert_called_once()
    assert isinstance(
        mock_client_constructor.call_args[0][0], chromadb_mock.config.Settings
    )

    mock_client_instance.get_or_create_collection.assert_called_once_with(
        name="default_coll",
        metadata=None,
        embedding_function="default_ef",  # The mocked default EF
    )
    assert handshake.collection == collection_mock


def test_init_get_collection_not_found(mock_chroma_client):
    """Test error if collection not found and create_collection_if_not_exists is False."""
    mock_chroma_client.get_collection.side_effect = Exception("Not found")
    with pytest.raises(Exception, match="Not found"):
        ChromaHandshake(
            collection_name="nonexistent",
            client=mock_chroma_client,
            create_collection_if_not_exists=False,
        )
    mock_chroma_client.get_collection.assert_called_once_with(
        name="nonexistent"
    )


# Test write and write_batch
def test_write_calls_write_batch(mock_chroma_client, sample_chunks_chroma):
    """Test that write calls write_batch."""
    handshake = ChromaHandshake(
        collection_name="test_coll", client=mock_chroma_client
    )
    handshake.write_batch = MagicMock()  # Mock write_batch

    handshake.write(sample_chunks_chroma[0], extra_arg="value")

    handshake.write_batch.assert_called_once_with(
        [sample_chunks_chroma[0]], extra_arg="value"
    )


def test_write_batch_with_embeddings(mock_chroma_client, sample_chunks_chroma):
    """Test write_batch with chunks that have embeddings."""
    handshake = ChromaHandshake(
        collection_name="test_coll", client=mock_chroma_client
    )
    handshake.collection = collection_mock  # Ensure collection is the mock

    handshake.write_batch(sample_chunks_chroma, extra_metadata={"batch": "b1"})

    ids_expected = ["ch1", "ch2"]
    docs_expected = ["Chroma chunk 1.", "Chroma chunk 2."]
    embeds_expected = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    meta_expected = [
        {
            "doc": "docA",
            "page": 1,
            "token_count": 3,
            "start_index": 0,
            "end_index": 16,
            "batch": "b1",
        },
        {
            "doc": "docA",
            "page": 2,
            "token_count": 3,
            "start_index": 17,
            "end_index": 33,
            "batch": "b1",
        },
    ]

    collection_mock.add.assert_called_once()
    call_args, call_kwargs = collection_mock.add.call_args
    assert call_kwargs["ids"] == ids_expected
    assert call_kwargs["documents"] == docs_expected
    assert len(call_kwargs["embeddings"]) == len(embeds_expected)
    for got, expected in zip(call_kwargs["embeddings"], embeds_expected):
        np.testing.assert_array_equal(got, expected)
    assert call_kwargs["metadatas"] == meta_expected


def test_write_batch_without_embeddings(
    mock_chroma_client, sample_chunks_chroma_no_embeddings
):
    """Test write_batch with chunks that do not have embeddings."""
    handshake = ChromaHandshake(
        collection_name="test_coll", client=mock_chroma_client
    )
    handshake.collection = collection_mock

    handshake.write_batch(sample_chunks_chroma_no_embeddings)

    ids_expected = ["ch3", "ch4"]
    docs_expected = ["No embedding here.", "Another without embedding."]
    meta_expected = [
        {"doc": "docB", "token_count": 3, "start_index": 0, "end_index": 18},
        {"doc": "docB", "token_count": 3, "start_index": 19, "end_index": 44},
    ]

    collection_mock.add.assert_called_once_with(
        ids=ids_expected,
        documents=docs_expected,
        embeddings=None,  # Should be None as not all chunks had embeddings
        metadatas=meta_expected,
    )


def test_write_batch_mixed_embeddings(
    mock_chroma_client, sample_chunks_chroma, sample_chunks_chroma_no_embeddings
):
    """Test write_batch with a mix of chunks with and without embeddings."""
    handshake = ChromaHandshake(
        collection_name="test_coll", client=mock_chroma_client
    )
    handshake.collection = collection_mock
    mixed_chunks = [
        sample_chunks_chroma[0],
        sample_chunks_chroma_no_embeddings[0],
    ]

    handshake.write_batch(mixed_chunks)

    ids_expected = ["ch1", "ch3"]
    docs_expected = ["Chroma chunk 1.", "No embedding here."]
    meta_expected = [
        {
            "doc": "docA",
            "page": 1,
            "token_count": 3,
            "start_index": 0,
            "end_index": 16,
        },
        {"doc": "docB", "token_count": 3, "start_index": 0, "end_index": 18},
    ]

    collection_mock.add.assert_called_once_with(
        ids=ids_expected,
        documents=docs_expected,
        embeddings=None,  # Should be None as not all chunks had embeddings
        metadatas=meta_expected,
    )


def test_write_batch_empty_list(mock_chroma_client):
    """Test write_batch with an empty list of chunks."""
    handshake = ChromaHandshake(
        collection_name="test_coll", client=mock_chroma_client
    )
    handshake.collection = collection_mock

    handshake.write_batch([])

    collection_mock.add.assert_not_called()


# Test Async Methods
@pytest.mark.asyncio
async def test_awrite_not_implemented(mock_chroma_client, sample_chunks_chroma):
    """Test that awrite raises NotImplementedError."""
    handshake = ChromaHandshake(
        collection_name="test_coll", client=mock_chroma_client
    )
    with pytest.raises(NotImplementedError):
        await handshake.awrite(sample_chunks_chroma[0])


@pytest.mark.asyncio
async def test_awrite_batch_not_implemented(
    mock_chroma_client, sample_chunks_chroma
):
    """Test that awrite_batch raises NotImplementedError."""
    handshake = ChromaHandshake(
        collection_name="test_coll", client=mock_chroma_client
    )
    with pytest.raises(NotImplementedError):
        await handshake.awrite_batch(sample_chunks_chroma)
