"""Unit tests for QdrantHandshake class."""

from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np
import pytest

# Mock the qdrant_client library before importing the class under test
qdrant_client_mock = MagicMock()
models_mock = MagicMock()
qdrant_client_mock.models = models_mock
qdrant_client_mock.http.models.Distance = MagicMock()
qdrant_client_mock.http.models.VectorParams = MagicMock()
qdrant_client_mock.QdrantClient = MagicMock()

# Define specific distance values if needed for tests
qdrant_client_mock.http.models.Distance.COSINE = "Cosine"
qdrant_client_mock.http.models.Distance.EUCLID = "Euclid"


with patch.dict("sys.modules", {"qdrant_client": qdrant_client_mock}):
    from chonkie.handshakes.vector_db_writers.qdrant import QdrantHandshake
    from chonkie.types import Chunk


@pytest.fixture
def mock_qdrant_client():
    """Fixture for a mocked QdrantClient object."""
    client = MagicMock()
    # Mock async upsert separately if needed, or make upsert an AsyncMock
    client.upsert = AsyncMock(return_value="upsert_result")
    client.get_collection = MagicMock()
    client.recreate_collection = MagicMock()
    return client


@pytest.fixture
def sample_chunks_qdrant():
    """Fixture for sample Chunk objects for Qdrant tests (embeddings required)."""
    return [
        Chunk(
            text="Qdrant chunk one.",
            embedding=np.array([0.5, 0.6], dtype=np.float32),
            id="qd1",
            metadata={"source": "fileX", "type": "para"},
            token_count=3,
            start_index=0,
            end_index=18,
        ),
        Chunk(
            text="Qdrant chunk two.",
            embedding=np.array([0.7, 0.8], dtype=np.float32),
            id="qd2",
            metadata={"source": "fileY", "type": "list"},
            token_count=3,
            start_index=19,
            end_index=37,
        ),
    ]


@pytest.fixture
def sample_chunks_qdrant_no_embeddings():
    """Fixture for sample Chunk objects without embeddings (should cause errors)."""
    return [
        Chunk(
            text="No embedding here.", id="qd3", metadata={"source": "fileZ"}
        ),
    ]


# Test Initialization
def test_init_with_existing_client(mock_qdrant_client):
    """Test initialization with an existing client."""
    mock_qdrant_client.get_collection.return_value = (
        True  # Simulate collection exists
    )
    handshake = QdrantHandshake(
        collection_name="existing_coll",
        client=mock_qdrant_client,
        create_collection_if_not_exists=False,
    )
    assert handshake.client == mock_qdrant_client
    assert handshake.collection_name == "existing_coll"
    mock_qdrant_client.get_collection.assert_called_once_with(
        collection_name="existing_coll"
    )
    mock_qdrant_client.recreate_collection.assert_not_called()


@patch("chonkie.handshakes.vector_db_writers.qdrant.QdrantClient")
def test_init_without_client_create_collection(mock_client_constructor):
    """Test initialization without client, creating collection."""
    mock_client_instance = mock_client_constructor.return_value
    mock_client_instance.get_collection.side_effect = Exception(
        "Not found"
    )  # Simulate collection doesn't exist

    handshake = QdrantHandshake(
        collection_name="new_q_coll",
        url="http://qdrant:6333",
        api_key="q-key",
        vector_size=128,
        distance=qdrant_client_mock.http.models.Distance.EUCLID,
        create_collection_if_not_exists=True,
        prefer_grpc=True,
    )

    mock_client_constructor.assert_called_once_with(
        url="http://qdrant:6333", api_key="q-key", prefer_grpc=True
    )
    assert handshake.client == mock_client_instance
    mock_client_instance.get_collection.assert_called_once_with(
        collection_name="new_q_coll"
    )
    qdrant_client_mock.http.models.VectorParams.assert_called_once_with(
        size=128, distance=qdrant_client_mock.http.models.Distance.EUCLID
    )
    mock_client_instance.recreate_collection.assert_called_once_with(
        collection_name="new_q_coll",
        vectors_config=qdrant_client_mock.http.models.VectorParams.return_value,
    )


@patch("chonkie.handshakes.vector_db_writers.qdrant.QdrantClient")
def test_init_create_collection_missing_vector_size(mock_client_constructor):
    """Test ValueError if creating collection without vector_size."""
    mock_client_instance = mock_client_constructor.return_value
    mock_client_instance.get_collection.side_effect = Exception("Not found")

    with pytest.raises(ValueError, match="vector_size must be provided"):
        QdrantHandshake(
            collection_name="new_q_coll",
            create_collection_if_not_exists=True,
            # vector_size is missing
        )


def test_init_collection_not_found_and_not_create(mock_qdrant_client):
    """Test ValueError if collection not found and create_collection_if_not_exists is False."""
    mock_qdrant_client.get_collection.side_effect = Exception(
        "Collection not found error"
    )
    with pytest.raises(
        ValueError,
        match="not found and create_collection_if_not_exists is False",
    ):
        QdrantHandshake(
            collection_name="nonexistent_q",
            client=mock_qdrant_client,
            create_collection_if_not_exists=False,
        )
    mock_qdrant_client.get_collection.assert_called_once_with(
        collection_name="nonexistent_q"
    )


# Test _prepare_qdrant_points
def test_prepare_qdrant_points_success(
    mock_qdrant_client, sample_chunks_qdrant
):
    """Test successful preparation of Qdrant points."""
    handshake = QdrantHandshake(
        collection_name="test", client=mock_qdrant_client, vector_size=2
    )  # vector_size needed for init check mock
    extra_meta = {"global": "value"}
    points = handshake._prepare_qdrant_points(
        sample_chunks_qdrant, extra_metadata=extra_meta
    )

    assert len(points) == 2
    models_mock.PointStruct.assert_has_calls([
        call(
            id="qd1",
            vector=sample_chunks_qdrant[0].embedding,
            payload={
                "source": "fileX",
                "type": "para",
                "token_count": 3,
                "start_index": 0,
                "end_index": 18,
                "global": "value",
            },
        ),
        call(
            id="qd2",
            vector=sample_chunks_qdrant[1].embedding,
            payload={
                "source": "fileY",
                "type": "list",
                "token_count": 3,
                "start_index": 19,
                "end_index": 37,
                "global": "value",
            },
        ),
    ])
    # Check the actual returned objects (assuming PointStruct mock returns the call args)
    assert points[0] == models_mock.PointStruct.call_args_list[0][1]
    assert points[1] == models_mock.PointStruct.call_args_list[1][1]


def test_prepare_qdrant_points_missing_embedding_raises_error(
    mock_qdrant_client, sample_chunks_qdrant_no_embeddings
):
    """Test _prepare_qdrant_points raises ValueError if embeddings are missing."""
    handshake = QdrantHandshake(
        collection_name="test", client=mock_qdrant_client, vector_size=2
    )
    # The error should be raised by _prepare_data called within _prepare_qdrant_points
    with pytest.raises(
        ValueError, match="One or more chunks are missing required embeddings"
    ):
        handshake._prepare_qdrant_points(sample_chunks_qdrant_no_embeddings)


def test_prepare_qdrant_points_empty_list(mock_qdrant_client):
    """Test _prepare_qdrant_points with an empty list."""
    handshake = QdrantHandshake(
        collection_name="test", client=mock_qdrant_client, vector_size=2
    )
    points = handshake._prepare_qdrant_points([])
    assert points == []
    models_mock.PointStruct.assert_not_called()


# Test write and write_batch
def test_write_calls_write_batch(mock_qdrant_client, sample_chunks_qdrant):
    """Test that write calls write_batch."""
    handshake = QdrantHandshake(
        collection_name="test", client=mock_qdrant_client, vector_size=2
    )
    handshake.write_batch = MagicMock(
        return_value="batch_result"
    )  # Mock write_batch

    result = handshake.write(
        sample_chunks_qdrant[0], wait=False, extra_metadata={"single": True}
    )

    handshake.write_batch.assert_called_once_with(
        [sample_chunks_qdrant[0]], wait=False, extra_metadata={"single": True}
    )
    assert result == "batch_result"


def test_write_batch_success(mock_qdrant_client, sample_chunks_qdrant):
    """Test successful batch write operation."""
    handshake = QdrantHandshake(
        collection_name="q_coll", client=mock_qdrant_client, vector_size=2
    )
    mock_points = [MagicMock(), MagicMock()]
    handshake._prepare_qdrant_points = MagicMock(return_value=mock_points)
    mock_qdrant_client.upsert.return_value = (
        "upsert_success"  # Reset mock return value
    )

    result = handshake.write_batch(
        sample_chunks_qdrant, wait=True, extra_metadata={"run": 5}
    )

    handshake._prepare_qdrant_points.assert_called_once_with(
        sample_chunks_qdrant, extra_metadata={"run": 5}
    )
    mock_qdrant_client.upsert.assert_called_once_with(
        collection_name="q_coll", points=mock_points, wait=True
    )
    assert result == "upsert_success"


def test_write_batch_no_chunks(mock_qdrant_client):
    """Test write_batch when no chunks are provided."""
    handshake = QdrantHandshake(
        collection_name="q_coll", client=mock_qdrant_client, vector_size=2
    )
    handshake._prepare_qdrant_points = MagicMock(
        return_value=[]
    )  # Simulate no points prepared

    result = handshake.write_batch([])

    handshake._prepare_qdrant_points.assert_called_once_with(
        [], extra_metadata=None
    )
    mock_qdrant_client.upsert.assert_not_called()
    assert result is None


# Test Async Methods
@pytest.mark.asyncio
async def test_awrite_calls_awrite_batch(
    mock_qdrant_client, sample_chunks_qdrant
):
    """Test that awrite calls awrite_batch."""
    handshake = QdrantHandshake(
        collection_name="test", client=mock_qdrant_client, vector_size=2
    )
    handshake.awrite_batch = AsyncMock(
        return_value="async_batch_result"
    )  # Mock awrite_batch

    result = await handshake.awrite(
        sample_chunks_qdrant[0], extra_metadata={"async_single": True}
    )

    handshake.awrite_batch.assert_awaited_once_with(
        [sample_chunks_qdrant[0]], extra_metadata={"async_single": True}
    )
    assert result == "async_batch_result"


@pytest.mark.asyncio
async def test_awrite_batch_success(mock_qdrant_client, sample_chunks_qdrant):
    """Test successful async batch write operation."""
    handshake = QdrantHandshake(
        collection_name="async_q_coll", client=mock_qdrant_client, vector_size=2
    )
    mock_points = [MagicMock(), MagicMock()]
    handshake._prepare_qdrant_points = MagicMock(return_value=mock_points)
    # Ensure the client's upsert is awaitable and returns something
    mock_qdrant_client.upsert = AsyncMock(return_value="async_upsert_done")

    result = await handshake.awrite_batch(
        sample_chunks_qdrant, extra_metadata={"async_run": 1}
    )

    handshake._prepare_qdrant_points.assert_called_once_with(
        sample_chunks_qdrant, extra_metadata={"async_run": 1}
    )
    # Qdrant client's async upsert might implicitly handle wait=False
    mock_qdrant_client.upsert.assert_awaited_once_with(
        collection_name="async_q_coll",
        points=mock_points,
        wait=False,  # awrite_batch defaults to wait=False
    )
    assert result == "async_upsert_done"


@pytest.mark.asyncio
async def test_awrite_batch_no_chunks(mock_qdrant_client):
    """Test awrite_batch when no chunks are provided."""
    handshake = QdrantHandshake(
        collection_name="async_q_coll", client=mock_qdrant_client, vector_size=2
    )
    handshake._prepare_qdrant_points = MagicMock(
        return_value=[]
    )  # Simulate no points prepared
    mock_qdrant_client.upsert = AsyncMock()  # Reset mock

    result = await handshake.awrite_batch([])

    handshake._prepare_qdrant_points.assert_called_once_with(
        [], extra_metadata=None
    )
    mock_qdrant_client.upsert.assert_not_awaited()
    assert result is None
