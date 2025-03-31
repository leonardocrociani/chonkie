import tiktoken
from chonkie.chunkers.tokenizer import (
    CharacterTokenizer,
    WordTokenizer,
    Tokenizer,
)
from tokenizers import Tokenizer as HFTokenizer
from transformers import AutoTokenizer
from typing import Callable
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
        "Testing should cover various scenarios, including text with short sentences, long sentences, multiple paragraphs, and potentially unusual punctuation or spacing.",
    ]


@pytest.fixture
def char_tokenizer() -> CharacterTokenizer:
    """Character tokenizer fixture."""
    return CharacterTokenizer()


@pytest.fixture
def word_tokenizer() -> WordTokenizer:
    """Word tokenizer fixture."""
    return WordTokenizer()


@pytest.fixture
def hf_tokenizer() -> HFTokenizer:
    """Create a HuggingFace tokenizer fixture."""
    return HFTokenizer.from_pretrained("gpt2")


@pytest.fixture
def tiktoken_tokenizer() -> tiktoken.Encoding:
    """Create a Tiktoken tokenizer fixture."""
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def transformers_tokenizer() -> AutoTokenizer:
    """Create a Transformer tokenizer fixture."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def callable_tokenizer() -> Callable[[str], int]:
    """Create a callable tokenizer fixture."""
    return lambda text: len(text.split())
