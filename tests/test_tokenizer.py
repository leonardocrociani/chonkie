import pytest

from typing import List


@pytest.fixture
def sample_text() -> str:
    """Fixture to provide sample text for testing."""
    return """The quick brown fox jumps over the lazy dog.
    This classic pangram contains all the letters of the English alphabet.
    It's often used for testing typefaces and keyboard layouts.
    Text chunking, the process you are working on, 
    involves dividing a larger text into smaller, contiguous pieces or 'chunks'.
    This is fundamental in many Natural Language Processing (NLP) tasks.
    For instance, large documents might be chunked into paragraphs or sections 
    before feeding them into a machine learning model due to memory constraints 
    or to process contextually relevant blocks. 
    Other applications include displaying text incrementally in user interfaces 
    or preparing data for certain types of linguistic analysis. 
    Effective chunking might consider sentence boundaries 
    (using periods, question marks, exclamation points), 
    paragraph breaks (often marked by double newlines), 
    or simply aim for fixed-size chunks based on character or word counts. 
    The ideal strategy depends heavily on the specific downstream application. 
    Testing should cover various scenarios, including text with short sentences, 
    long sentences, multiple paragraphs, and potentially unusual punctuation or spacing."""

@pytest.fixture
def sample_text_list() -> List[str]:
    """Fixture to provide a list of sample text for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "This classic pangram contains all the letters of the English alphabet.",
        "It's often used for testing typefaces and keyboard layouts.",
        "Text chunking, the process you are working on, involves dividing a larger text into smaller, contiguous pieces or 'chunks'.",
        "This is fundamental in many Natural Language Processing (NLP) tasks.",
        "For instance, large documents might be chunked into paragraphs or sections before feeding them into a machine learning model due to memory constraints or to process contextually relevant blocks.",
        "Other applications include displaying text incrementally in user interfaces or preparing data for certain types of linguistic analysis.",
        "Effective chunking might consider sentence boundaries (using periods, question marks, exclamation points), paragraph breaks (often marked by double newlines), or simply aim for fixed-size chunks based on character or word counts.",
        "The ideal strategy depends heavily on the specific downstream application.",
        "Testing should cover various scenarios, including text with short sentences, long sentences, multiple paragraphs, and potentially unusual punctuation or spacing."
    ]

