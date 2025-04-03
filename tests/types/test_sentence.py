"""Unit tests for Sentence and SentenceChunk classes."""

import pytest

from chonkie import Sentence, SentenceChunk


def test_sentence_init():
    """Test Sentence initialization."""
    sentence = Sentence(
        text="Ratatouille is a movie.", start_index=0, end_index=14, token_count=3
    )
    assert sentence.text == "Ratatouille is a movie."
    assert sentence.start_index == 0
    assert sentence.end_index == 14
    assert sentence.token_count == 3


def test_sentence_raises_error():
    """Test Sentence raises error for illegal field values."""
    with pytest.raises(ValueError):
        Sentence(text=9000, start_index=0, end_index=14, token_count=3)

    with pytest.raises(ValueError):
        Sentence(
            text="Ratatouille is a movie.", start_index=-1, end_index=14, token_count=3
        )

    with pytest.raises(ValueError):
        Sentence(
            text="Ratatouille is a movie.", start_index=0, end_index=-1, token_count=3
        )

    with pytest.raises(ValueError):
        Sentence(
            text="Ratatouille is a movie.", start_index=0, end_index=14, token_count=-1
        )

    with pytest.raises(ValueError):
        Sentence(
            text="Ratatouille is a movie.",
            start_index=10,
            end_index=5,
            token_count=3,
        )


def test_sentence_serialization():
    """Test Sentence serialization/deserialization."""
    sentence = Sentence(
        text="Ratatouille is a movie.", start_index=0, end_index=14, token_count=3
    )
    sentence_dict = sentence.to_dict()
    restored = Sentence.from_dict(sentence_dict)
    assert sentence.text == restored.text
    assert sentence.token_count == restored.token_count


def test_sentence_chunk_init():
    """Test SentenceChunk initialization."""
    sentences = [
        Sentence("First sentence.", 0, 14, 3),
        Sentence("Second sentence.", 15, 30, 3),
    ]
    chunk = SentenceChunk(
        text="Wall-E is a Pixar movie. Ratatouille is another one.",
        start_index=0,
        end_index=30,
        token_count=6,
        sentences=sentences,
    )
    assert chunk.text == "Wall-E is a Pixar movie. Ratatouille is another one."
    assert len(chunk.sentences) == 2
    assert all(isinstance(s, Sentence) for s in chunk.sentences)


def test_sentence_chunk_serialization():
    """Test SentenceChunk serialization/deserialization."""
    sentences = [
        Sentence("First sentence.", 0, 14, 3),
        Sentence("Second sentence.", 15, 30, 3),
    ]
    chunk = SentenceChunk(
        text="Wall-E is a Pixar movie. Ratatouille is another one.",
        start_index=0,
        end_index=30,
        token_count=6,
        sentences=sentences,
    )
    chunk_dict = chunk.to_dict()
    restored = SentenceChunk.from_dict(chunk_dict)
    assert len(restored.sentences) == 2
    assert all(isinstance(s, Sentence) for s in restored.sentences)
