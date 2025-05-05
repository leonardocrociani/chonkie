"""Unit tests for the TurbopufferHandshake class."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock the turbopuffer library before importing the class under test
tpuf_mock = MagicMock()
with patch.dict("sys.modules", {"turbopuffer": tpuf_mock}):
    from chonkie.handshakes.vector_db_writers.turbopuffer import (
        TurbopufferHandshake,
    )
    from chonkie.types import Chunk


@pytest.fixture
def mock_namespace():
    """Fixture for a mocked Turbopuffer Namespace object."""
    return MagicMock()


@pytest.fixture
def mock_tpuf_namespace_init(mock_namespace):
    """Fixture to mock the __init__ of turbopuffer.Namespace."""
    with patch(
        "turbopuffer.Namespace", return_value=mock_namespace
    ) as mock_init:
        yield mock_init


@pytest.fixture
def sample_chunks():
    """Fixture for sample Chunk objects."""
    return [
        Chunk(
            text="Chunk 1 text.",
            embedding=np.array([0.1, 0.2], dtype=np.float32),
            id="c1",
            metadata={"doc_id": "doc1", "source": "file1.txt"},
            token_count=3,
            start_index=0,
            end_index=15,
        ),
        Chunk(
            text="Chunk 2.",
            embedding=np.array([0.3, 0.4], dtype=np.float32),
            id="c2",
            metadata={"doc_id": "doc1", "extra": "data"},
            token_count=2,
            start_index=16,
            end_index=24,
        ),
        Chunk(
            text="Chunk 3 no meta.",
            embedding=np.array([0.5, 0.6], dtype=np.float32),
            id="c3",
            token_count=4,
            start_index=25,
            end_index=41,
        ),
    ]


@pytest.fixture
def sample_chunks_no_embeddings():
    """Fixture for sample Chunk objects without embeddings."""
    return [
        Chunk(text="No embedding here.", id="c4", metadata={"doc_id": "doc2"}),
    ]


# Test Initialization
def test_init_success_with_key(mock_tpuf_namespace_init):
    """Test successful initialization with API key provided."""
    handshake = TurbopufferHandshake(namespace="test-ns", api_key="test-key")
    mock_tpuf_namespace_init.assert_called_once_with(
        "test-ns", api_key="test-key"
    )
    assert handshake.namespace == mock_tpuf_namespace_init.return_value
    assert handshake.api_key == "test-key"


@patch.dict(os.environ, {"TURBOPUFFER_API_KEY": "env-key"})
def test_init_success_with_env_key(mock_tpuf_namespace_init):
    """Test successful initialization using environment variable for API key."""
    handshake = TurbopufferHandshake(namespace="test-ns")
    mock_tpuf_namespace_init.assert_called_once_with(
        "test-ns", api_key="env-key"
    )
    assert handshake.namespace == mock_tpuf_namespace_init.return_value
    assert handshake.api_key == "env-key"


@patch.dict(os.environ, {}, clear=True)
def test_init_fail_no_key(mock_tpuf_namespace_init):
    """Test ValueError is raised when no API key is provided or found in env."""
    with pytest.raises(ValueError, match="Turbopuffer API key not provided"):
        TurbopufferHandshake(namespace="test-ns")
    mock_tpuf_namespace_init.assert_not_called()


# Test _prepare_tpuf_data
def test_prepare_tpuf_data_basic(mock_tpuf_namespace_init, sample_chunks):
    """Test _prepare_tpuf_data with basic chunks."""
    handshake = TurbopufferHandshake(namespace="test-ns", api_key="test-key")
    prepared_data = handshake._prepare_tpuf_data(sample_chunks)

    assert prepared_data["ids"] == ["c1", "c2", "c3"]
    np.testing.assert_array_equal(
        prepared_data["vectors"],
        [
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([0.3, 0.4], dtype=np.float32),
            np.array([0.5, 0.6], dtype=np.float32),
        ],
    )
    assert (
        "text" not in prepared_data["attributes"]
    )  # text should not be included by default
    assert prepared_data["attributes"]["doc_id"] == ["doc1", "doc1", None]
    assert prepared_data["attributes"]["source"] == ["file1.txt", None, None]
    assert prepared_data["attributes"]["extra"] == [None, "data", None]
    assert prepared_data["attributes"]["token_count"] == [
        3,
        2,
        4,
    ]  # Default metadata


def test_prepare_tpuf_data_with_metadata_fields(
    mock_tpuf_namespace_init, sample_chunks
):
    """Test _prepare_tpuf_data with specific metadata_fields."""
    handshake = TurbopufferHandshake(
        namespace="test-ns",
        api_key="test-key",
        metadata_fields=["doc_id", "source"],  # Only include these
    )
    prepared_data = handshake._prepare_tpuf_data(sample_chunks)

    assert prepared_data["ids"] == ["c1", "c2", "c3"]
    assert "extra" not in prepared_data["attributes"]
    assert (
        "token_count" not in prepared_data["attributes"]
    )  # Not explicitly requested
    assert prepared_data["attributes"]["doc_id"] == ["doc1", "doc1", None]
    assert prepared_data["attributes"]["source"] == ["file1.txt", None, None]


def test_prepare_tpuf_data_with_extra_metadata(
    mock_tpuf_namespace_init, sample_chunks
):
    """Test _prepare_tpuf_data with extra_metadata."""
    handshake = TurbopufferHandshake(namespace="test-ns", api_key="test-key")
    extra = {"batch_id": "batch123", "processed_at": "now"}
    prepared_data = handshake._prepare_tpuf_data(
        sample_chunks, extra_metadata=extra
    )

    assert prepared_data["attributes"]["batch_id"] == ["batch123"] * 3
    assert prepared_data["attributes"]["processed_at"] == ["now"] * 3
    # Ensure original metadata is still present
    assert prepared_data["attributes"]["doc_id"] == ["doc1", "doc1", None]


def test_prepare_tpuf_data_missing_embedding_raises_error(
    mock_tpuf_namespace_init, sample_chunks_no_embeddings
):
    """Test _prepare_tpuf_data raises ValueError if embeddings are missing."""
    handshake = TurbopufferHandshake(namespace="test-ns", api_key="test-key")
    with pytest.raises(
        ValueError, match="One or more chunks are missing required embeddings"
    ):
        handshake._prepare_tpuf_data(sample_chunks_no_embeddings)


def test_prepare_tpuf_data_empty_list(mock_tpuf_namespace_init):
    """Test _prepare_tpuf_data with an empty list of chunks."""
    handshake = TurbopufferHandshake(namespace="test-ns", api_key="test-key")
    prepared_data = handshake._prepare_tpuf_data([])
    assert prepared_data == {"ids": [], "vectors": [], "attributes": {}}


# Test write and write_batch
def test_write_calls_write_batch(
    mock_tpuf_namespace_init, mock_namespace, sample_chunks
):
    """Test that write calls write_batch."""
    handshake = TurbopufferHandshake(namespace="test-ns", api_key="test-key")
    handshake.write_batch = MagicMock(
        return_value="upsert_result"
    )  # Mock write_batch

    result = handshake.write(sample_chunks[0], distance_metric="cosine")

    handshake.write_batch.assert_called_once_with(
        [sample_chunks[0]], distance_metric="cosine"
    )
    assert result == "upsert_result"


def test_write_batch_success(
    mock_tpuf_namespace_init, mock_namespace, sample_chunks
):
    """Test successful batch write operation."""
    handshake = TurbopufferHandshake(namespace="test-ns", api_key="test-key")
    mock_namespace.upsert.return_value = "upsert_success"

    # Mock prepare data to control its output
    prepared_data_mock = {
        "ids": ["c1", "c2", "c3"],
        "vectors": [
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
            np.array([0.5, 0.6]),
        ],
        "attributes": {
            "doc_id": ["doc1", "doc1", None],
            "token_count": [3, 2, 4],
        },
    }
    handshake._prepare_tpuf_data = MagicMock(return_value=prepared_data_mock)

    result = handshake.write_batch(
        sample_chunks, distance_metric="euclidean", extra_metadata={"run": 1}
    )

    handshake._prepare_tpuf_data.assert_called_once_with(
        sample_chunks, extra_metadata={"run": 1}
    )
    mock_namespace.upsert.assert_called_once_with(
        data=prepared_data_mock, distance_metric="euclidean"
    )
    assert result == "upsert_success"


def test_write_batch_no_chunks(mock_tpuf_namespace_init, mock_namespace):
    """Test write_batch when no chunks are provided (or prepared data is empty)."""
    handshake = TurbopufferHandshake(namespace="test-ns", api_key="test-key")

    # Mock prepare data to return empty IDs
    prepared_data_mock = {"ids": [], "vectors": [], "attributes": {}}
    handshake._prepare_tpuf_data = MagicMock(return_value=prepared_data_mock)

    result = handshake.write_batch([])

    handshake._prepare_tpuf_data.assert_called_once_with(
        [], extra_metadata=None
    )
    mock_namespace.upsert.assert_not_called()
    assert result is None


# Test Async Methods
@pytest.mark.asyncio
async def test_awrite_not_implemented(mock_tpuf_namespace_init, sample_chunks):
    """Test that awrite raises NotImplementedError."""
    handshake = TurbopufferHandshake(namespace="test-ns", api_key="test-key")
    with pytest.raises(NotImplementedError):
        await handshake.awrite(sample_chunks[0])


@pytest.mark.asyncio
async def test_awrite_batch_not_implemented(
    mock_tpuf_namespace_init, sample_chunks
):
    """Test that awrite_batch raises NotImplementedError."""
    handshake = TurbopufferHandshake(namespace="test-ns", api_key="test-key")
    with pytest.raises(NotImplementedError):
        await handshake.awrite_batch(sample_chunks)
