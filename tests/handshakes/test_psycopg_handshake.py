"""Test the PsycopgHandshake class."""
import uuid
from typing import List
from unittest.mock import Mock, patch

import pytest

# Try to import psycopg and pgvector, skip tests if unavailable
try:
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg.types.json import Json
    psycopg_available = True
except ImportError:
    psycopg = None
    Json = None
    register_vector = None
    psycopg_available = False

from chonkie.types import Chunk

if psycopg_available:
    from chonkie.friends.handshakes.psycopg import PsycopgHandshake

# Mark all tests in this module to be skipped if psycopg/pgvector is not installed
pytestmark = pytest.mark.skipif(not psycopg_available, reason="psycopg or pgvector not installed")


@pytest.fixture(autouse=True)
def mock_embeddings():
    """Mock AutoEmbeddings to avoid downloading models in CI."""
    with patch('chonkie.embeddings.AutoEmbeddings.get_embeddings') as mock_get_embeddings:
        # Create a mock embedding model
        mock_embedding = Mock()
        
        # Mock the embed method to return consistent results
        def mock_embed_single(text):
            return [0.1] * 128
        
        # Mock the embed_batch method to return the right number of embeddings
        def mock_embed_batch(texts):
            return [[0.1] * 128] * len(texts)  # Return one embedding per text
        
        mock_embedding.embed.side_effect = mock_embed_single
        mock_embedding.embed_batch.side_effect = mock_embed_batch
        mock_embedding.dimension = 128
        mock_get_embeddings.return_value = mock_embedding
        yield mock_get_embeddings


@pytest.fixture
def mock_connection():
    """Mock psycopg connection for testing."""
    mock_conn = Mock()
    mock_cursor = Mock()
    
    # Setup cursor context manager using MagicMock for proper context manager support
    from unittest.mock import MagicMock
    mock_cursor_cm = MagicMock()
    mock_cursor_cm.__enter__.return_value = mock_cursor
    mock_cursor_cm.__exit__.return_value = None
    mock_conn.cursor.return_value = mock_cursor_cm
    
    # Mock cursor methods
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = []
    mock_conn.commit.return_value = None
    
    return mock_conn


# Sample Chunks for testing
SAMPLE_CHUNKS: List[Chunk] = [
    Chunk(text="This is the first chunk.", start_index=0, end_index=25, token_count=6),
    Chunk(text="This is the second chunk.", start_index=26, end_index=52, token_count=6),
    Chunk(text="Another chunk follows.", start_index=53, end_index=75, token_count=4),
]


def test_psycopg_handshake_is_available():
    """Test the _is_available check."""
    with patch('chonkie.friends.handshakes.psycopg.importutil.find_spec') as mock_find_spec:
        # Mock both psycopg and pgvector as available
        mock_find_spec.side_effect = lambda name: Mock() if name in ['psycopg', 'pgvector'] else None
        
        handshake = PsycopgHandshake.__new__(PsycopgHandshake)  # Create without calling __init__
        assert handshake._is_available() is True
        
        # Test when psycopg is not available
        mock_find_spec.side_effect = lambda name: Mock() if name == 'pgvector' else None
        assert handshake._is_available() is False
        
        # Test when pgvector is not available
        mock_find_spec.side_effect = lambda name: Mock() if name == 'psycopg' else None
        assert handshake._is_available() is False


def test_psycopg_handshake_init_default(mock_connection):
    """Test PsycopgHandshake initialization with default settings."""
    with patch.object(PsycopgHandshake, '_import_dependencies'), \
         patch.object(PsycopgHandshake, '_setup_database'):
        handshake = PsycopgHandshake(mock_connection)
        
        assert handshake.connection == mock_connection
        assert handshake.table_name == "chonkie_chunks"
        assert handshake.create_table is True
        assert handshake.vector_dimensions == 128  # From mocked embedding


def test_psycopg_handshake_init_custom(mock_connection):
    """Test PsycopgHandshake initialization with custom settings."""
    with patch.object(PsycopgHandshake, '_import_dependencies'), \
         patch.object(PsycopgHandshake, '_setup_database'):
        handshake = PsycopgHandshake(
            connection=mock_connection,
            table_name="custom_chunks",
            vector_dimensions=256,
            create_table=False
        )
        
        assert handshake.table_name == "custom_chunks"
        assert handshake.vector_dimensions == 256
        assert handshake.create_table is False


def test_psycopg_handshake_generate_id(mock_connection):
    """Test the _generate_id method."""
    with patch.object(PsycopgHandshake, '_import_dependencies'), \
         patch.object(PsycopgHandshake, '_setup_database'):
        handshake = PsycopgHandshake(mock_connection)
        chunk = SAMPLE_CHUNKS[0]
        index = 0
        
        expected_id_str = str(uuid.uuid5(
            uuid.NAMESPACE_OID, 
            f"{handshake.table_name}::chunk-{index}:{chunk.text}"
        ))
        generated_id = handshake._generate_id(index, chunk)
        assert generated_id == expected_id_str


def test_psycopg_handshake_generate_metadata(mock_connection):
    """Test the _generate_metadata method."""
    with patch.object(PsycopgHandshake, '_import_dependencies'), \
         patch.object(PsycopgHandshake, '_setup_database'):
        handshake = PsycopgHandshake(mock_connection)
        chunk = SAMPLE_CHUNKS[0]
        
        metadata = handshake._generate_metadata(chunk)
        
        assert metadata["chunk_type"] == "Chunk"
        # Basic chunk shouldn't have sentences, words, or language attributes
        assert "sentence_count" not in metadata
        assert "word_count" not in metadata
        assert "language" not in metadata


def test_psycopg_handshake_generate_metadata_with_attributes(mock_connection):
    """Test the _generate_metadata method with chunk attributes."""
    with patch.object(PsycopgHandshake, '_import_dependencies'), \
         patch.object(PsycopgHandshake, '_setup_database'):
        handshake = PsycopgHandshake(mock_connection)
        
        # Create a mock chunk with additional attributes
        chunk = Mock()
        chunk.text = "Sample text"
        chunk.start_index = 0
        chunk.end_index = 11
        chunk.token_count = 2
        chunk.sentences = ["Sentence 1", "Sentence 2"]
        chunk.words = ["Sample", "text"]
        chunk.language = "en"
        
        metadata = handshake._generate_metadata(chunk)
        
        assert metadata["chunk_type"] == "Mock"
        assert metadata["sentence_count"] == 2
        assert metadata["word_count"] == 2
        assert metadata["language"] == "en"


def test_psycopg_handshake_write_single_chunk(mock_connection):
    """Test writing a single Chunk."""
    with patch.object(PsycopgHandshake, '_import_dependencies'), \
         patch.object(PsycopgHandshake, '_setup_database'), \
         patch('chonkie.friends.handshakes.psycopg.Json', side_effect=lambda x: x, create=True) as mock_json:
        
        handshake = PsycopgHandshake(mock_connection)
        chunk = SAMPLE_CHUNKS[0]
        
        result = handshake.write(chunk)
        
        # Should return list of IDs
        assert isinstance(result, list)
        assert len(result) == 1
        
        # Verify cursor execute was called for insertion
        mock_cursor = mock_connection.cursor.return_value.__enter__.return_value
        mock_cursor.execute.assert_called()
        mock_connection.commit.assert_called()
        
        # Verify Json was called for metadata
        mock_json.assert_called()


def test_psycopg_handshake_write_multiple_chunks(mock_connection):
    """Test writing multiple Chunks."""
    with patch.object(PsycopgHandshake, '_import_dependencies'), \
         patch.object(PsycopgHandshake, '_setup_database'), \
         patch('chonkie.friends.handshakes.psycopg.Json', side_effect=lambda x: x, create=True) as mock_json:
        
        handshake = PsycopgHandshake(mock_connection)
        
        result = handshake.write(SAMPLE_CHUNKS)
        
        # Should return list of IDs equal to number of chunks
        assert isinstance(result, list)
        assert len(result) == len(SAMPLE_CHUNKS)
        
        # Verify cursor execute was called for each chunk
        mock_cursor = mock_connection.cursor.return_value.__enter__.return_value
        assert mock_cursor.execute.call_count == len(SAMPLE_CHUNKS)
        mock_connection.commit.assert_called()
        
        # Verify Json was called for each chunk's metadata
        assert mock_json.call_count == len(SAMPLE_CHUNKS)


def test_psycopg_handshake_search(mock_connection):
    """Test similarity search functionality."""
    with patch.object(PsycopgHandshake, '_import_dependencies'), \
         patch.object(PsycopgHandshake, '_setup_database'):
        handshake = PsycopgHandshake(mock_connection)
        
        # Mock return data for search
        mock_cursor = mock_connection.cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.return_value = [
            ("id1", "Similar text", 0, 25, 5, {"chunk_type": "Chunk"}, 0.5),
            ("id2", "Another similar text", 26, 50, 4, {"chunk_type": "Chunk"}, 0.7)
        ]
        
        results = handshake.search("test query", limit=2)
        
        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[0]["text"] == "Similar text"
        assert results[0]["distance"] == 0.5
        
        # Verify search query was executed
        mock_cursor.execute.assert_called()


def test_psycopg_handshake_search_invalid_metric(mock_connection):
    """Test similarity search with invalid distance metric."""
    with patch.object(PsycopgHandshake, '_import_dependencies'), \
         patch.object(PsycopgHandshake, '_setup_database'):
        handshake = PsycopgHandshake(mock_connection)
        
        with pytest.raises(ValueError, match="Unsupported distance metric"):
            handshake.search("test query", distance_metric="invalid")


def test_psycopg_handshake_create_index_hnsw(mock_connection):
    """Test creating HNSW index."""
    with patch.object(PsycopgHandshake, '_import_dependencies'), \
         patch.object(PsycopgHandshake, '_setup_database'):
        handshake = PsycopgHandshake(mock_connection)
        
        handshake.create_index(index_type="hnsw", distance_metric="l2", m=32, ef_construction=128)
        
        # Verify index creation query was executed
        mock_cursor = mock_connection.cursor.return_value.__enter__.return_value
        mock_cursor.execute.assert_called()
        mock_connection.commit.assert_called()


def test_psycopg_handshake_create_index_ivfflat(mock_connection):
    """Test creating IVFFlat index."""
    with patch.object(PsycopgHandshake, '_import_dependencies'), \
         patch.object(PsycopgHandshake, '_setup_database'):
        handshake = PsycopgHandshake(mock_connection)
        
        handshake.create_index(index_type="ivfflat", distance_metric="cosine", lists=200)
        
        # Verify index creation query was executed
        mock_cursor = mock_connection.cursor.return_value.__enter__.return_value
        mock_cursor.execute.assert_called()
        mock_connection.commit.assert_called()


def test_psycopg_handshake_create_index_invalid_type(mock_connection):
    """Test creating index with invalid type."""
    with patch.object(PsycopgHandshake, '_import_dependencies'), \
         patch.object(PsycopgHandshake, '_setup_database'):
        handshake = PsycopgHandshake(mock_connection)
        
        with pytest.raises(ValueError, match="Unsupported index type"):
            handshake.create_index(index_type="invalid")


def test_psycopg_handshake_create_index_invalid_metric(mock_connection):
    """Test creating index with invalid distance metric."""
    with patch.object(PsycopgHandshake, '_import_dependencies'), \
         patch.object(PsycopgHandshake, '_setup_database'):
        handshake = PsycopgHandshake(mock_connection)
        
        with pytest.raises(ValueError, match="Unsupported distance metric"):
            handshake.create_index(distance_metric="invalid")


def test_psycopg_handshake_repr(mock_connection):
    """Test the __repr__ method."""
    with patch.object(PsycopgHandshake, '_import_dependencies'), \
         patch.object(PsycopgHandshake, '_setup_database'):
        handshake = PsycopgHandshake(mock_connection, table_name="test_table")
        expected_repr = f"PsycopgHandshake(table_name=test_table, vector_dimensions={handshake.vector_dimensions})"
        assert repr(handshake) == expected_repr


def test_psycopg_handshake_import_dependencies_error():
    """Test import dependencies failure."""
    with patch.object(PsycopgHandshake, '_is_available', return_value=False):
        handshake = PsycopgHandshake.__new__(PsycopgHandshake)  # Create without calling __init__
        
        with pytest.raises(ImportError, match="psycopg and pgvector are not installed"):
            handshake._import_dependencies()


def test_psycopg_handshake_import_dependencies_success():
    """Test successful import of dependencies."""
    with patch('chonkie.friends.handshakes.psycopg.importutil.find_spec') as mock_find_spec:
        # Mock both psycopg and pgvector as available
        mock_find_spec.side_effect = lambda name: Mock() if name in ['psycopg', 'pgvector'] else None
        
        handshake = PsycopgHandshake.__new__(PsycopgHandshake)  # Create without calling __init__
        
        # Mock the imports to avoid actual import
        with patch('builtins.__import__') as mock_import:
            handshake._import_dependencies()
            
            # Verify the imports were attempted
            # Should import psycopg, Json from psycopg.types.json, and register_vector from pgvector.psycopg
            assert mock_import.call_count >= 2