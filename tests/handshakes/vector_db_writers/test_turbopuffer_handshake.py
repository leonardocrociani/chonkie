# """Unit tests for the TurbopufferHandshake class."""

# import os
# import uuid
# from unittest.mock import patch

# import numpy as np
# import pytest

# # Try importing turbopuffer, skip tests if unavailable
# tpuf = pytest.importorskip("turbopuffer")

# from chonkie.handshakes.vector_db_writers.turbopuffer import (
#     TurbopufferHandshake,
# )
# from chonkie.types import Chunk

# # --- Fixtures ---

# # Fixture to generate a unique namespace for each test function
# # to avoid collisions when running against a real Turbopuffer instance.
# @pytest.fixture(scope="function")
# def test_namespace_name():
#     """Generate a unique namespace name for test isolation."""
#     return f"chonkie-test-{uuid.uuid4()}"


# # Fixture to provide a TurbopufferHandshake instance.
# # It requires TURBOPUFFER_API_KEY environment variable.
# # Skips the test if the key is not found.
# # Cleans up the namespace after the test.
# @pytest.fixture(scope="function")
# def turbopuffer_handshake(test_namespace_name):
#     """Fixture providing an initialized TurbopufferHandshake and cleanup."""
#     api_key = os.environ.get("TURBOPUFFER_API_KEY")
#     if not api_key:
#         pytest.skip(
#             "TURBOPUFFER_API_KEY environment variable not set. Skipping Turbopuffer integration tests."
#         )

#     # Use suppress to ignore errors if namespace doesn't exist during cleanup
#     try:
#         # Initialize the handshake (which creates the namespace implicitly on write)
#         handshake = TurbopufferHandshake(
#             namespace=test_namespace_name, api_key=api_key
#         )
#         yield handshake
#     finally:
#         # Cleanup: Delete the test namespace
#         try:
#             # Need a direct client instance to delete
#             client = tpuf.Namespace(test_namespace_name, api_key=api_key)
#             client.delete_all()
#         except Exception as e:
#             print(f"Warning: Failed to clean up Turbopuffer namespace {test_namespace_name}: {e}")


# @pytest.fixture
# def sample_chunks():
#     """Fixture for sample Chunk objects."""
#     chunks_data = [
#         {
#             "text": "Chunk 1 text.",
#             "token_count": 3,
#             "start_index": 0,
#             "end_index": 15,
#             "id": "c1",
#             "embedding": np.array([0.1, 0.2], dtype=np.float32),
#             "metadata": {"doc_id": "doc1", "source": "file1.txt"},
#         },
#         {
#             "text": "Chunk 2.",
#             "token_count": 2,
#             "start_index": 16,
#             "end_index": 24,
#             "id": "c2",
#             "embedding": np.array([0.3, 0.4], dtype=np.float32),
#             "metadata": {"doc_id": "doc1", "extra": "data"},
#         },
#         {
#             "text": "Chunk 3 no meta.",
#             "token_count": 4,
#             "start_index": 25,
#             "end_index": 41,
#             "id": "c3",
#             "embedding": np.array([0.5, 0.6], dtype=np.float32),
#             "metadata": {}, # Empty metadata
#         },
#     ]
#     chunks = []
#     for data in chunks_data:
#         chunk = Chunk(
#             text=data["text"],
#             start_index=data["start_index"],
#             end_index=data["end_index"],
#             token_count=data["token_count"],
#         )
#         chunk.id = data["id"]
#         chunk.embedding = data["embedding"]
#         # Add metadata items as attributes
#         for key, value in data["metadata"].items():
#             setattr(chunk, key, value)
#         chunks.append(chunk)
#     return chunks


# @pytest.fixture
# def sample_chunks_no_embeddings():
#     """Fixture for sample Chunk objects without embeddings."""
#     chunk_data = {
#         "text": "No embedding here.",
#         "id": "c4",
#         "metadata": {"doc_id": "doc2"},
#         # Assuming default/dummy values for indices and token_count if not critical
#         "start_index": 0,
#         "end_index": 18,
#         "token_count": 4,
#     }
#     chunk = Chunk(
#         text=chunk_data["text"],
#         start_index=chunk_data["start_index"],
#         end_index=chunk_data["end_index"],
#         token_count=chunk_data["token_count"],
#     )
#     chunk.id = chunk_data["id"]
#     # chunk.embedding is intentionally omitted
#     for key, value in chunk_data["metadata"].items():
#         setattr(chunk, key, value)
#     return [chunk]


# # --- Test Initialization ---

# # Note: Direct initialization tests now depend on the API key being present.
# # The fixture handles skipping if the key is missing.

# def test_init_success_with_key(test_namespace_name):
#     """Test successful initialization with API key provided directly."""
#     api_key = os.environ.get("TURBOPUFFER_API_KEY")
#     if not api_key:
#         pytest.skip("TURBOPUFFER_API_KEY needed for this test.")

#     handshake = TurbopufferHandshake(
#         namespace=test_namespace_name, api_key=api_key
#     )
#     assert handshake.namespace is not None
#     assert handshake.namespace.name == test_namespace_name
#     assert handshake.api_key == api_key


# @pytest.mark.skipif(
#     not os.environ.get("TURBOPUFFER_API_KEY"),
#     reason="TURBOPUFFER_API_KEY needed for this test.",
# )
# def test_init_success_with_env_key(test_namespace_name):
#     """Test successful initialization using environment variable for API key."""
#     # Environment variable is implicitly used by the fixture/constructor
#     handshake = TurbopufferHandshake(namespace=test_namespace_name)
#     assert handshake.namespace is not None
#     assert handshake.namespace.name == test_namespace_name
#     assert handshake.api_key == os.environ.get("TURBOPUFFER_API_KEY")


# @patch.dict(os.environ, {}, clear=True)
# def test_init_fail_no_key():
#     """Test ValueError is raised when no API key is provided or found in env."""
#     with pytest.raises(ValueError, match="Turbopuffer API key not provided"):
#         TurbopufferHandshake(namespace="test-ns-fail")


# # --- Test _prepare_tpuf_data ---
# # These tests check the data transformation logic, which doesn't require a live connection.
# # We still need to instantiate the Handshake, so a dummy key/namespace is fine here,
# # or we can reuse the fixture logic carefully. Let's instantiate directly for simplicity.

# def test_prepare_tpuf_data_basic(sample_chunks):
#     """Test _prepare_tpuf_data with basic chunks."""
#     # Instantiation needs an API key, even if not used for connection here.
#     # Use a dummy key as we won't connect.
#     handshake = TurbopufferHandshake(namespace="dummy-ns", api_key="dummy-key")
#     prepared_data = handshake._prepare_tpuf_data(sample_chunks)

#     assert prepared_data["ids"] == ["c1", "c2", "c3"]
#     np.testing.assert_array_equal(
#         prepared_data["vectors"],
#         [
#             np.array([0.1, 0.2], dtype=np.float32),
#             np.array([0.3, 0.4], dtype=np.float32),
#             np.array([0.5, 0.6], dtype=np.float32),
#         ],
#     )
#     assert (
#         "text" not in prepared_data["attributes"]
#     )  # text should not be included by default
#     assert prepared_data["attributes"]["doc_id"] == ["doc1", "doc1", None]
#     assert prepared_data["attributes"]["source"] == ["file1.txt", None, None]
#     assert prepared_data["attributes"]["extra"] == [None, "data", None]
#     assert prepared_data["attributes"]["token_count"] == [
#         3,
#         2,
#         4,
#     ]  # Default metadata


# def test_prepare_tpuf_data_with_metadata_fields(sample_chunks):
#     """Test _prepare_tpuf_data with specific metadata_fields."""
#     handshake = TurbopufferHandshake(
#         namespace="dummy-ns",
#         api_key="dummy-key",
#         metadata_fields=["doc_id", "source"],  # Only include these
#     )
#     prepared_data = handshake._prepare_tpuf_data(sample_chunks)

#     assert prepared_data["ids"] == ["c1", "c2", "c3"]
#     assert "extra" not in prepared_data["attributes"]
#     assert (
#         "token_count" not in prepared_data["attributes"]
#     )  # Not explicitly requested
#     assert prepared_data["attributes"]["doc_id"] == ["doc1", "doc1", None]
#     assert prepared_data["attributes"]["source"] == ["file1.txt", None, None]


# def test_prepare_tpuf_data_with_extra_metadata(sample_chunks):
#     """Test _prepare_tpuf_data with extra_metadata."""
#     handshake = TurbopufferHandshake(namespace="dummy-ns", api_key="dummy-key")
#     extra = {"batch_id": "batch123", "processed_at": "now"}
#     prepared_data = handshake._prepare_tpuf_data(
#         sample_chunks, extra_metadata=extra
#     )

#     assert prepared_data["attributes"]["batch_id"] == ["batch123"] * 3
#     assert prepared_data["attributes"]["processed_at"] == ["now"] * 3
#     # Ensure original metadata is still present
#     assert prepared_data["attributes"]["doc_id"] == ["doc1", "doc1", None]


# def test_prepare_tpuf_data_missing_embedding_raises_error(
#     sample_chunks_no_embeddings,
# ):
#     """Test _prepare_tpuf_data raises ValueError if embeddings are missing."""
#     handshake = TurbopufferHandshake(namespace="dummy-ns", api_key="dummy-key")
#     with pytest.raises(
#         ValueError, match="One or more chunks are missing required embeddings"
#     ):
#         handshake._prepare_tpuf_data(sample_chunks_no_embeddings)


# def test_prepare_tpuf_data_empty_list():
#     """Test _prepare_tpuf_data with an empty list of chunks."""
#     handshake = TurbopufferHandshake(namespace="dummy-ns", api_key="dummy-key")
#     prepared_data = handshake._prepare_tpuf_data([])
#     assert prepared_data == {"ids": [], "vectors": [], "attributes": {}}


# # --- Test write and write_batch (Integration Tests) ---
# # These tests now interact with the actual Turbopuffer service via the fixture.

# def test_write_batch_success(turbopuffer_handshake, sample_chunks):
#     """Test successful batch write operation to Turbopuffer."""
#     result = turbopuffer_handshake.write_batch(
#         sample_chunks, distance_metric="cosine_dist" # Use valid metric like cosine_dist
#     )

#     # Basic check: Turbopuffer client usually returns None on success for upsert
#     assert result is None


# def test_write_calls_write_batch(turbopuffer_handshake, sample_chunks):
#     """Test that write calls write_batch and successfully writes a single chunk."""
#     chunk_to_write = sample_chunks[0]
#     result = turbopuffer_handshake.write(
#         chunk_to_write, distance_metric="cosine_dist"
#     )
#     assert result is None


# def test_write_batch_no_chunks(turbopuffer_handshake):
#     """Test write_batch when no chunks are provided."""
#     result = turbopuffer_handshake.write_batch([])
#     assert result is None


# # --- Test Async Methods ---
# # Keep NotImplementedError tests as they reflect the current state.

# @pytest.mark.asyncio
# async def test_awrite_not_implemented(turbopuffer_handshake, sample_chunks):
#     """Test that awrite raises NotImplementedError."""
#     with pytest.raises(NotImplementedError):
#         await turbopuffer_handshake.awrite(sample_chunks[0])


# @pytest.mark.asyncio
# async def test_awrite_batch_not_implemented(turbopuffer_handshake, sample_chunks):
#     """Test that awrite_batch raises NotImplementedError."""
#     with pytest.raises(NotImplementedError):
#         await turbopuffer_handshake.awrite_batch(sample_chunks)
