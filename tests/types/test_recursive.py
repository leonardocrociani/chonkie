"""Test recursive types."""

import pytest

from chonkie.types import RecursiveChunk, RecursiveLevel, RecursiveRules


def test_recursive_level_init():
    """Test RecursiveLevel initialization."""
    level = RecursiveLevel(custom_delimiters=["\n", "."])
    assert level.custom_delimiters == ["\n", "."]
    assert not level.whitespace_delimiter
    assert level.include_delim == "prev"


def test_recursive_level_raises_error():
    """Test RecursiveLevel validation."""
    with pytest.raises(NotImplementedError):
        RecursiveLevel(whitespace_delimiter=True, custom_delimiters=["."])

    with pytest.raises(ValueError):
        RecursiveLevel(custom_delimiters=[1, 2])

    with pytest.raises(ValueError):
        RecursiveLevel(custom_delimiters=[""])

    with pytest.raises(ValueError):
        RecursiveLevel(custom_delimiters=[" "])


def test_recursive_level_serialization():
    """Test RecursiveLevel serialization and deserialization."""
    level = RecursiveLevel(custom_delimiters=["\n", "."])
    level_dict = level.to_dict()
    reconstructed = RecursiveLevel.from_dict(level_dict)
    assert reconstructed.custom_delimiters == ["\n", "."]
    assert not reconstructed.whitespace_delimiter
    assert reconstructed.include_delim == "prev"


# RecursiveRules Tests
def test_recursive_rules_default_init():
    """Test RecursiveRules default initialization."""
    rules = RecursiveRules()
    assert len(rules.levels) == 5
    assert all(isinstance(level, RecursiveLevel) for level in rules.levels)


def test_recursive_rules_custom_init():
    """Test RecursiveRules custom initialization."""
    levels = [
        RecursiveLevel(custom_delimiters=["\n"]),
        RecursiveLevel(custom_delimiters=["."]),
    ]
    rules = RecursiveRules(levels=levels)
    assert len(rules.levels) == 2
    assert rules.levels == levels


def test_recursive_rules_serialization():
    """Test RecursiveRules serialization and deserialization."""
    levels = [
        RecursiveLevel(custom_delimiters=["\n"]),
        RecursiveLevel(custom_delimiters=["."]),
    ]
    rules = RecursiveRules(levels=levels)
    rules_dict = rules.to_dict()
    reconstructed = RecursiveRules.from_dict(rules_dict)
    assert len(reconstructed.levels) == 2
    assert all(
        isinstance(level, RecursiveLevel) for level in reconstructed.levels
    )


# RecursiveChunk Tests
def test_recursive_chunk_init():
    """Test RecursiveChunk initialization."""
    chunk = RecursiveChunk(
        text="test chunk",
        start_index=0,
        end_index=10,
        token_count=2,
        recursive_level=1,
    )
    assert chunk.text == "test chunk"
    assert chunk.recursive_level == 1


def test_recursive_chunk_serialization():
    """Test RecursiveChunk serialization/deserialization."""
    chunk = RecursiveChunk(
        text="test chunk",
        start_index=0,
        end_index=10,
        token_count=2,
        recursive_level=1,
    )
    chunk_dict = chunk.to_dict()
    reconstructed = RecursiveChunk.from_dict(chunk_dict)
    assert reconstructed.recursive_level == 1
    assert reconstructed.text == chunk.text
