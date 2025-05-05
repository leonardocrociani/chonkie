"""Test suite for the ChromaHandshake class."""

import time
from unittest.mock import MagicMock

import chromadb
import numpy as np
import pytest
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from chonkie.handshakes.vector_db_writers.chroma import ChromaHandshake
from chonkie.types import Chunk


# Use a real in-memory client for testing
@pytest.fixture(scope="function")  # Recreate client for each test function
def ephemeral_chroma_client():
    """Fixture for a ChromaDB EphemeralClient object."""
    client = chromadb.EphemeralClient(
        settings=Settings(allow_reset=True, is_persistent=False)
    )
    yield client
    # Explicitly reset the client to tear down the system instance
    try:
        client.reset()
    except Exception:
        # Ignore exceptions during cleanup
        pass


@pytest.fixture
def sample_chunks_chroma():
    """Fixture for sample Chunk objects for Chroma tests."""
    chunk1 = Chunk(
        text="Chroma chunk 1.",
        token_count=3,
        start_index=0,
        end_index=16,
    )
    chunk1.id = "ch1"
    chunk1.embedding = np.array([0.1, 0.2], dtype=np.float32)
    # Assign metadata as attributes
    chunk1.custom_field = "value1"

    chunk2 = Chunk(
        text="Chroma chunk 2.",
        token_count=3,
        start_index=17,
        end_index=33,
    )
    chunk2.id = "ch2"
    chunk2.embedding = np.array([0.3, 0.4], dtype=np.float32)
    # Assign metadata as attributes
    chunk2.custom_field = "value2"

    return [chunk1, chunk2]


@pytest.fixture
def sample_chunks_chroma_no_embeddings():
    """Fixture for sample Chunk objects without embeddings for Chroma tests."""
    chunks = [
        Chunk(
            text="No embedding here.",
            token_count=3,
            start_index=0,
            end_index=18,
        ),
        Chunk(
            text="Another without embedding.",
            token_count=3,
            start_index=19,
            end_index=44,
        ),
    ]
    chunks[0].id = "ch3"
    chunks[1].id = "ch4"
    chunks[0].custom_field = "value3"
    chunks[1].custom_field = "value4"
    return chunks


def unique_collection_name(prefix="test_coll"):
    """Generate unique collection names for test isolation."""
    return f"{prefix}_{int(time.time() * 1000)}_{np.random.randint(1000)}"


# Test Initialization
def test_init_with_existing_client_get_existing(ephemeral_chroma_client):
    """Test initialization with an existing client and existing collection."""
    coll_name = unique_collection_name()
    existing_collection = ephemeral_chroma_client.create_collection(coll_name)

    handshake = ChromaHandshake(
        collection_name=coll_name,
        client=ephemeral_chroma_client,
        create_collection_if_not_exists=False,
    )
    assert handshake.client == ephemeral_chroma_client
    assert handshake.collection.name == existing_collection.name
    assert handshake.collection.id == existing_collection.id


def test_init_with_existing_client_create_new(ephemeral_chroma_client):
    """Test initialization with an existing client, creating a new collection."""
    coll_name = unique_collection_name("new_coll")
    custom_ef = embedding_functions.DefaultEmbeddingFunction()

    handshake = ChromaHandshake(
        collection_name=coll_name,
        client=ephemeral_chroma_client,
        create_collection_if_not_exists=True,
        collection_metadata={"purpose": "testing_create"},
        embedding_function=custom_ef,
    )

    assert handshake.client == ephemeral_chroma_client
    created_collection = ephemeral_chroma_client.get_collection(
        name=coll_name, embedding_function=custom_ef
    )
    assert created_collection is not None
    assert created_collection.name == coll_name
    assert created_collection.metadata == {"purpose": "testing_create"}
    assert handshake.collection.name == created_collection.name


def test_init_without_client_create_collection():
    """Test initialization without client, creating collection."""
    coll_name = unique_collection_name("no_client_coll")
    internal_client_settings = Settings(allow_reset=True, is_persistent=False)
    dummy_ef = embedding_functions.DefaultEmbeddingFunction()

    handshake = ChromaHandshake(
        collection_name=coll_name,
        client_settings=internal_client_settings,
        create_collection_if_not_exists=True,
        collection_metadata={"purpose": "testing_internal"},
        embedding_function=dummy_ef,
    )

    assert isinstance(handshake.client, chromadb.ClientAPI)
    created_collection = handshake.client.get_collection(name=coll_name)
    assert created_collection is not None
    assert created_collection.name == coll_name
    assert created_collection.metadata == {"purpose": "testing_internal"}
    assert handshake.collection.name == created_collection.name


def test_init_without_client_default_settings_and_ef():
    """Test initialization uses default settings and EF if none provided."""
    coll_name = unique_collection_name("default_coll")
    handshake = None
    ephemeral_settings = Settings(allow_reset=True, is_persistent=False)
    try:
        handshake = ChromaHandshake(
            collection_name=coll_name,
            client_settings=ephemeral_settings,
            create_collection_if_not_exists=True,
        )

        assert isinstance(handshake.client, chromadb.ClientAPI)
        created_collection = handshake.client.get_collection(name=coll_name)
        assert created_collection is not None
        assert created_collection.name == coll_name
        assert handshake.collection.name == created_collection.name
    finally:
        if handshake and hasattr(handshake, "client") and handshake.client:
            try:
                if hasattr(handshake.client, "reset"):
                    handshake.client.reset()
            except Exception:
                pass


def test_init_get_collection_not_found(ephemeral_chroma_client):
    """Test error if collection not found and create_collection_if_not_exists is False."""
    coll_name = unique_collection_name("nonexistent")
    with pytest.raises(Exception):
        ChromaHandshake(
            collection_name=coll_name,
            client=ephemeral_chroma_client,
            create_collection_if_not_exists=False,
        )
    with pytest.raises(Exception):
        ephemeral_chroma_client.get_collection(name=coll_name)


# Test write and write_batch
def test_write_calls_write_batch(ephemeral_chroma_client, sample_chunks_chroma):
    """Test that write calls write_batch."""
    coll_name = unique_collection_name()
    handshake = ChromaHandshake(
        collection_name=coll_name,
        client=ephemeral_chroma_client,
        embedding_function=None,
        create_collection_if_not_exists=True,
    )
    handshake.write_batch = MagicMock()

    handshake.write(sample_chunks_chroma[0], extra_arg="value")

    handshake.write_batch.assert_called_once_with(
        [sample_chunks_chroma[0]], extra_arg="value"
    )


def test_write_batch_with_embeddings(
    ephemeral_chroma_client, sample_chunks_chroma
):
    """Test write_batch with chunks that have embeddings."""
    coll_name = unique_collection_name()
    handshake = ChromaHandshake(
        collection_name=coll_name,
        client=ephemeral_chroma_client,
        embedding_function=None,
        create_collection_if_not_exists=True,
        metadata_fields=[
            "token_count",
            "start_index",
            "end_index",
            "custom_field",
        ],
    )

    extra_batch_meta = {"batch": "b1", "source": "test_fixture"}
    handshake.write_batch(sample_chunks_chroma, extra_metadata=extra_batch_meta)

    results = handshake.collection.get(
        ids=[c.id for c in sample_chunks_chroma],
        include=["metadatas", "documents", "embeddings"],
    )

    assert sorted(results["ids"]) == sorted(["ch1", "ch2"])
    assert len(results["documents"]) == 2
    assert len(results["metadatas"]) == 2
    assert len(results["embeddings"]) == 2

    idx1 = results["ids"].index("ch1")
    idx2 = results["ids"].index("ch2")

    assert results["documents"][idx1] == "Chroma chunk 1."
    expected_meta1 = {
        "token_count": 3,
        "start_index": 0,
        "end_index": 16,
        "custom_field": "value1",
        "batch": "b1",
        "source": "test_fixture",
    }
    assert results["metadatas"][idx1] == expected_meta1
    np.testing.assert_allclose(
        results["embeddings"][idx1], [0.1, 0.2], atol=1e-6
    )

    assert results["documents"][idx2] == "Chroma chunk 2."
    expected_meta2 = {
        "token_count": 3,
        "start_index": 17,
        "end_index": 33,
        "custom_field": "value2",
        "batch": "b1",
        "source": "test_fixture",
    }
    assert results["metadatas"][idx2] == expected_meta2
    np.testing.assert_allclose(
        results["embeddings"][idx2], [0.3, 0.4], atol=1e-6
    )
