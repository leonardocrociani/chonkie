# """Unit tests for QdrantHandshake class using an in-memory Qdrant instance."""

# import uuid
# from contextlib import suppress

# import numpy as np
# import pytest

# # Try importing qdrant_client, skip tests if unavailable
# qdrant_client = pytest.importorskip("qdrant_client")
# from qdrant_client import QdrantClient, models

# from chonkie.handshakes.vector_db_writers.qdrant import QdrantHandshake
# from chonkie.types import Chunk

# # --- Fixtures ---

# @pytest.fixture(scope="function")
# def test_collection_name():
#     """Generate a unique collection name for each test function."""
#     return f"chonkie-test-coll-{uuid.uuid4()}"


# @pytest.fixture(scope="function")
# def qdrant_client_in_memory():
#     """Fixture for an in-memory QdrantClient."""
#     client = QdrantClient(":memory:")
#     yield client
#     # No explicit cleanup needed for :memory: client


# @pytest.fixture(scope="function")
# def qdrant_handshake(qdrant_client_in_memory, test_collection_name):
#     """Fixture for a QdrantHandshake using an in-memory client."""
#     # Use a small default vector size for tests
#     vector_size = 2
#     handshake = QdrantHandshake(
#         collection_name=test_collection_name,
#         client=qdrant_client_in_memory,
#         vector_size=vector_size,
#         distance=models.Distance.COSINE,
#         create_collection_if_not_exists=True,
#     )
#     # Ensure collection exists after handshake initialization
#     try:
#         qdrant_client_in_memory.get_collection(collection_name=test_collection_name)
#     except Exception as e:
#         pytest.fail(f"Collection {test_collection_name} was not created: {e}")

#     yield handshake
#     # Clean up the collection after the test
#     with suppress(Exception): # Ignore errors if collection doesn't exist
#          qdrant_client_in_memory.delete_collection(collection_name=test_collection_name)


# @pytest.fixture
# def sample_chunks_qdrant():
#     """Fixture for sample Chunk objects for Qdrant tests (embeddings required)."""
#     chunk1 = Chunk(
#         text="Qdrant chunk one.",
#         token_count=3,
#         start_index=0,
#         end_index=18,
#     )
#     chunk1.embedding = np.array([0.5, 0.6], dtype=np.float32)
#     chunk1.id = "qd1"
#     # Assign metadata as attributes
#     chunk1.source = "fileX"
#     chunk1.type = "para"

#     chunk2 = Chunk(
#         text="Qdrant chunk two.",
#         token_count=3,
#         start_index=19,
#         end_index=37,
#     )
#     chunk2.embedding = np.array([0.7, 0.8], dtype=np.float32)
#     chunk2.id = "qd2"
#     # Assign metadata as attributes
#     chunk2.source = "fileY"
#     chunk2.type = "list"

#     chunk3 = Chunk( # Chunk with no metadata initially
#         text="Qdrant chunk three.",
#         token_count=3,
#         start_index=38,
#         end_index=57,
#     )
#     chunk3.embedding = np.array([0.1, 0.9], dtype=np.float32)
#     chunk3.id = "qd3"

#     return [chunk1, chunk2, chunk3]


# @pytest.fixture
# def sample_chunks_qdrant_no_embeddings():
#     """Fixture for sample Chunk objects without embeddings (should cause errors)."""
#     chunk_no_emb = Chunk(
#         text="No embedding here.",
#         # Assuming default/dummy values for required fields if needed by tests
#         token_count=3,
#         start_index=0,
#         end_index=18,
#     )
#     chunk_no_emb.id = "qd_no_emb"
#     # Assign metadata as attributes
#     chunk_no_emb.source = "fileZ"
#     return [chunk_no_emb]


# # --- Test Initialization ---

# def test_init_with_existing_client_create_collection(qdrant_client_in_memory, test_collection_name):
#     """Test initialization with an existing client and creating the collection."""
#     vector_size = 4 # Different size for this test
#     handshake = QdrantHandshake(
#         collection_name=test_collection_name,
#         client=qdrant_client_in_memory,
#         vector_size=vector_size,
#         distance=models.Distance.EUCLID,
#         create_collection_if_not_exists=True,
#     )
#     assert handshake.client == qdrant_client_in_memory
#     assert handshake.collection_name == test_collection_name

#     # Verify collection was created
#     collection_info = qdrant_client_in_memory.get_collection(collection_name=test_collection_name)
#     assert collection_info.vectors_config.params.size == vector_size
#     assert collection_info.vectors_config.params.distance == models.Distance.EUCLID


# def test_init_with_existing_client_collection_exists(qdrant_client_in_memory, test_collection_name):
#     """Test initialization when the collection already exists."""
#     vector_size = 8
#     distance = models.Distance.DOT
#     # Pre-create the collection
#     qdrant_client_in_memory.recreate_collection(
#         collection_name=test_collection_name,
#         vectors_config=models.VectorParams(size=vector_size, distance=distance)
#     )

#     handshake = QdrantHandshake(
#         collection_name=test_collection_name,
#         client=qdrant_client_in_memory,
#         create_collection_if_not_exists=False, # Should not try to recreate
#         # vector_size/distance shouldn't matter if not creating
#     )
#     assert handshake.client == qdrant_client_in_memory
#     assert handshake.collection_name == test_collection_name

#     # Verify collection still exists with original config
#     collection_info = qdrant_client_in_memory.get_collection(collection_name=test_collection_name)
#     assert collection_info.vectors_config.params.size == vector_size
#     assert collection_info.vectors_config.params.distance == distance


# def test_init_without_client_create_collection(test_collection_name):
#     """Test initialization without providing a client instance (should create one)."""
#     vector_size = 16
#     handshake = QdrantHandshake(
#         collection_name=test_collection_name,
#         location=":memory:", # Use in-memory location
#         vector_size=vector_size,
#         distance=models.Distance.COSINE,
#         create_collection_if_not_exists=True,
#     )
#     assert isinstance(handshake.client, QdrantClient)
#     assert handshake.collection_name == test_collection_name

#     # Verify collection was created in the new client instance
#     collection_info = handshake.client.get_collection(collection_name=test_collection_name)
#     assert collection_info.vectors_config.params.size == vector_size
#     assert collection_info.vectors_config.params.distance == models.Distance.COSINE


# def test_init_create_collection_missing_vector_size(qdrant_client_in_memory, test_collection_name):
#     """Test ValueError if creating collection without vector_size."""
#     with pytest.raises(ValueError, match="vector_size must be provided"):
#         QdrantHandshake(
#             collection_name=test_collection_name,
#             client=qdrant_client_in_memory,
#             create_collection_if_not_exists=True,
#             # vector_size is missing
#         )


# def test_init_collection_not_found_and_not_create(qdrant_client_in_memory, test_collection_name):
#     """Test ValueError if collection not found and create_collection_if_not_exists is False."""
#     with pytest.raises(
#         ValueError, # Qdrant client raises UnexpectedResponseError or similar
#         match=f"Collection `{test_collection_name}` not found and create_collection_if_not_exists is False.",
#     ):
#         QdrantHandshake(
#             collection_name=test_collection_name, # Does not exist yet
#             client=qdrant_client_in_memory,
#             create_collection_if_not_exists=False,
#         )


# # --- Test _prepare_qdrant_points ---
# # This is an internal method, but testing it ensures data transformation logic is correct.
# # We still need a handshake instance to call it.

# def test_prepare_qdrant_points_success(qdrant_handshake, sample_chunks_qdrant):
#     """Test successful preparation of Qdrant points."""
#     extra_meta = {"global": "value"}
#     points = qdrant_handshake._prepare_qdrant_points(
#         sample_chunks_qdrant, extra_metadata=extra_meta
#     )

#     assert len(points) == 3
#     assert isinstance(points[0], models.PointStruct)
#     assert points[0].id == "qd1"
#     np.testing.assert_array_equal(points[0].vector, sample_chunks_qdrant[0].embedding)
#     assert points[0].payload == {
#         "source": "fileX",
#         "type": "para",
#         "token_count": 3,
#         "start_index": 0,
#         "end_index": 18,
#         "global": "value", # Extra metadata included
#     }

#     assert isinstance(points[1], models.PointStruct)
#     assert points[1].id == "qd2"
#     np.testing.assert_array_equal(points[1].vector, sample_chunks_qdrant[1].embedding)
#     assert points[1].payload == {
#         "source": "fileY",
#         "type": "list",
#         "token_count": 3,
#         "start_index": 19,
#         "end_index": 37,
#         "global": "value",
#     }

#     assert isinstance(points[2], models.PointStruct)
#     assert points[2].id == "qd3"
#     np.testing.assert_array_equal(points[2].vector, sample_chunks_qdrant[2].embedding)
#     assert points[2].payload == {
#         # No original metadata, only default + extra
#         "token_count": 3,
#         "start_index": 38,
#         "end_index": 57,
#         "global": "value",
#     }


# def test_prepare_qdrant_points_missing_embedding_raises_error(
#     qdrant_handshake, sample_chunks_qdrant_no_embeddings
# ):
#     """Test _prepare_qdrant_points raises ValueError if embeddings are missing."""
#     with pytest.raises(
#         ValueError, match="One or more chunks are missing required embeddings"
#     ):
#         qdrant_handshake._prepare_qdrant_points(sample_chunks_qdrant_no_embeddings)


# def test_prepare_qdrant_points_empty_list(qdrant_handshake):
#     """Test _prepare_qdrant_points with an empty list."""
#     points = qdrant_handshake._prepare_qdrant_points([])
#     assert points == []


# # --- Test write and write_batch (Integration with :memory: client) ---

# def test_write_calls_write_batch(qdrant_handshake, sample_chunks_qdrant):
#     """Test that write calls write_batch and successfully writes a single chunk."""
#     chunk_to_write = sample_chunks_qdrant[0]
#     extra_meta = {"single_write": True}

#     # Use the real write method
#     result = qdrant_handshake.write(
#         chunk_to_write, wait=True, extra_metadata=extra_meta
#     )

#     # Check result (Qdrant upsert result is complex, just check it's not None/error)
#     assert result is not None

#     # Verify the point exists in the collection
#     retrieved_points = qdrant_handshake.client.retrieve(
#         collection_name=qdrant_handshake.collection_name,
#         ids=[chunk_to_write.id],
#         with_payload=True,
#         with_vectors=True,
#     )
#     assert len(retrieved_points) == 1
#     point = retrieved_points[0]
#     assert point.id == chunk_to_write.id
#     np.testing.assert_array_equal(point.vector, chunk_to_write.embedding)
#     assert point.payload["source"] == chunk_to_write.metadata["source"]
#     assert point.payload["single_write"] is True # Check extra metadata


# def test_write_batch_success(qdrant_handshake, sample_chunks_qdrant):
#     """Test successful batch write operation."""
#     extra_meta = {"batch_run": 5}
#     result = qdrant_handshake.write_batch(
#         sample_chunks_qdrant, wait=True, extra_metadata=extra_meta
#     )

#     # Check result
#     assert result is not None

#     # Verify points exist
#     ids_to_check = [c.id for c in sample_chunks_qdrant]
#     retrieved_points = qdrant_handshake.client.retrieve(
#         collection_name=qdrant_handshake.collection_name,
#         ids=ids_to_check,
#         with_payload=True,
#     )
#     assert len(retrieved_points) == len(sample_chunks_qdrant)
#     retrieved_ids = {p.id for p in retrieved_points}
#     assert retrieved_ids == set(ids_to_check)
#     # Check extra metadata on one point
#     assert retrieved_points[0].payload["batch_run"] == 5


# def test_write_batch_no_chunks(qdrant_handshake):
#     """Test write_batch when no chunks are provided."""
#     result = qdrant_handshake.write_batch([])
#     assert result is None # Should do nothing and return None

#     # Verify collection is still empty (or has 0 points if created empty)
#     count = qdrant_handshake.client.count(
#         collection_name=qdrant_handshake.collection_name
#     )
#     assert count.count == 0


# # --- Test Async Methods (Integration with :memory: client) ---

# @pytest.mark.asyncio
# async def test_awrite_calls_awrite_batch(qdrant_handshake, sample_chunks_qdrant):
#     """Test that awrite calls awrite_batch and successfully writes a single chunk."""
#     chunk_to_write = sample_chunks_qdrant[1] # Use a different chunk
#     extra_meta = {"async_single": True}

#     result = await qdrant_handshake.awrite(
#         chunk_to_write, extra_metadata=extra_meta
#     ) # wait=False is default for async

#     # Check result
#     assert result is not None

#     # Verify the point exists (use sync client for verification for simplicity)
#     # Note: In-memory client state is shared between sync/async calls
#     retrieved_points = qdrant_handshake.client.retrieve(
#         collection_name=qdrant_handshake.collection_name,
#         ids=[chunk_to_write.id],
#         with_payload=True,
#         with_vectors=True,
#     )
#     assert len(retrieved_points) == 1
#     point = retrieved_points[0]
#     assert point.id == chunk_to_write.id
#     np.testing.assert_array_equal(point.vector, chunk_to_write.embedding)
#     assert point.payload["type"] == chunk_to_write.metadata["type"]
#     assert point.payload["async_single"] is True


# @pytest.mark.asyncio
# async def test_awrite_batch_success(qdrant_handshake, sample_chunks_qdrant):
#     """Test successful async batch write operation."""
#     extra_meta = {"async_batch_run": 1}
#     result = await qdrant_handshake.awrite_batch(
#         sample_chunks_qdrant, extra_metadata=extra_meta
#     )

#     # Check result
#     assert result is not None

#     # Verify points exist
#     ids_to_check = [c.id for c in sample_chunks_qdrant]
#     retrieved_points = qdrant_handshake.client.retrieve(
#         collection_name=qdrant_handshake.collection_name,
#         ids=ids_to_check,
#         with_payload=True,
#     )
#     assert len(retrieved_points) == len(sample_chunks_qdrant)
#     retrieved_ids = {p.id for p in retrieved_points}
#     assert retrieved_ids == set(ids_to_check)
#     # Check extra metadata on one point
#     assert retrieved_points[1].payload["async_batch_run"] == 1


# @pytest.mark.asyncio
# async def test_awrite_batch_no_chunks(qdrant_handshake):
#     """Test awrite_batch when no chunks are provided."""
#     result = await qdrant_handshake.awrite_batch([])
#     assert result is None # Should do nothing and return None

#     # Verify collection is still empty
#     count = qdrant_handshake.client.count(
#         collection_name=qdrant_handshake.collection_name
#     )
#     assert count.count == 0