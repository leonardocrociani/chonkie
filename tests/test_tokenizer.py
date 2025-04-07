"""Unit tests for the tokenizer module."""

from typing import Callable

import pytest
import tiktoken
from tokenizers import Tokenizer as HFTokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from chonkie.tokenizer import (
    CharacterTokenizer,
    Tokenizer,
    WordTokenizer,
)


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
def sample_text_list() -> list[str]:
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
def character_tokenizer() -> CharacterTokenizer:
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
def transformers_tokenizer() -> PreTrainedTokenizerFast:
    """Create a Transformer tokenizer fixture."""
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("gpt2")
    return tokenizer


@pytest.fixture
def callable_tokenizer() -> Callable[[str], int]:
    """Create a callable tokenizer fixture."""
    return lambda text: len(text.split())

@pytest.mark.parametrize(
    "backend_str",
    [
        "hf_tokenizer",
        "tiktoken_tokenizer",
        "transformers_tokenizer",
        "callable_tokenizer",
    ],
)
def test_backend_selection(request: pytest.FixtureRequest, backend_str: str) -> None:
    """Test that the tokenizer correctly selects the backend based on given string."""
    try:
        tokenizer = Tokenizer(request.getfixturevalue(backend_str))
    except Exception as e:
        pytest.skip(f"Skipping test with backend {backend_str}: {str(e)}")

    assert tokenizer._backend in [
        "transformers",
        "tokenizers",
        "tiktoken",
        "callable",
    ]


@pytest.mark.parametrize(
    "model_name", ["gpt2", "cl100k_base", "p50k_base"]
)
def test_string_init(model_name: str) -> None:
    """Test initialization of tokenizer with different model strings."""
    try:
        tokenizer = Tokenizer(model_name)
        assert tokenizer is not None
        assert tokenizer._backend in [
            "transformers",
            "tokenizers",
            "tiktoken",
        ]
    except ImportError as e:
        pytest.skip(f"Could not import tokenizer for {model_name}: {str(e)}")
    except Exception as e:
        if "not found in model".casefold() in str(e).casefold():
            pytest.skip(f"Skipping test with {model_name}. Backend not available")
        else:
            raise e


@pytest.mark.parametrize(
    "backend_str",
    ["hf_tokenizer", "tiktoken_tokenizer", "transformers_tokenizer"],
)
def test_encode_decode(
    request: pytest.FixtureRequest, backend_str: str, sample_text: str
) -> None:
    """Test encoding and decoding with different backends."""
    try:
        tokenizer = request.getfixturevalue(backend_str)
        tokenizer = Tokenizer(tokenizer)
    except Exception as e:
        pytest.skip(f"Skipping test with backend {backend_str}: {str(e)}")

    # Encode, Decode and Compare
    tokens = tokenizer.encode(sample_text)
    assert len(tokens) > 0
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)

    if tokenizer._backend != "callable":
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)
        assert decoded == sample_text


@pytest.mark.parametrize(
    "model_name", ["gpt2", "cl100k_base", "p50k_base"]
)
def test_string_init_encode_decode(model_name: str) -> None:
    """Test basic functionality of string initialized models."""
    try:
        tokenizer = Tokenizer(model_name)
        assert tokenizer is not None
        assert tokenizer._backend in [
            "transformers",
            "tokenizers",
            "tiktoken",
        ]
        test_string = "Testing tokenizer_string_init_basic for Chonkie Tokenizers."
        tokens = tokenizer.encode(test_string)
        assert len(tokens) > 0
        assert isinstance(tokens, list)
        assert all(isinstance(token, int) for token in tokens)

        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)
        # Check if decoded strings preserves original words
        for word in [
            "testing",
            "Chonkie",
            "Tokenizers",
        ]:
            assert word.lower() in decoded.lower()
    except ImportError as e:
        pytest.skip(
            f"Skipping test. Could not import tokenizer for {model_name}: {str(e)}"
        )
    except Exception as e:
        if "not found in model".casefold() in str(e).casefold():
            pytest.skip(f"Skipping test with {model_name}. Backend not available")
        else:
            raise e


@pytest.mark.parametrize(
    "backend_str",
    [
        "hf_tokenizer",
        "tiktoken_tokenizer",
        "transformers_tokenizer",
        "callable_tokenizer",
    ],
)
def test_token_counting(
    request: pytest.FixtureRequest,
    backend_str: str,
    sample_text: str,
) -> None:
    """Test token counting with different backends."""
    try:
        tokenizer = request.getfixturevalue(backend_str)
        tokenizer = Tokenizer(tokenizer)
    except Exception as e:
        pytest.skip(f"Skipping test with backend {backend_str}: {str(e)}")

    count = tokenizer.count_tokens(sample_text)
    assert isinstance(count, int)
    assert count > 0

    # Verify count matches encoded length
    if tokenizer._backend != "callable":
        assert count == len(tokenizer.encode(sample_text))


@pytest.mark.parametrize(
    "backend_str",
    ["hf_tokenizer", "tiktoken_tokenizer", "transformers_tokenizer"],
)
def test_batch_encode_decode(
    request: pytest.FixtureRequest,
    backend_str: str,
    sample_text_list: list[str],
) -> None:
    """Test batch encoding and decoding with different backends."""
    try:
        tokenizer = request.getfixturevalue(backend_str)
        tokenizer = Tokenizer(tokenizer)
    except Exception as e:
        pytest.skip(f"Skipping test with backend {backend_str}: {str(e)}")

    batch_encoded = tokenizer.encode_batch(sample_text_list)
    assert isinstance(batch_encoded, list)
    assert len(batch_encoded) == len(sample_text_list)
    assert all(isinstance(tokens, list) for tokens in batch_encoded)
    assert all(len(tokens) > 0 for tokens in batch_encoded)
    assert all(
        all(isinstance(token, int) for token in tokens) for tokens in batch_encoded
    )

    if tokenizer._backend != "callable":
        batch_decoded = tokenizer.decode_batch(batch_encoded)
        assert isinstance(batch_decoded, list)
        assert len(batch_decoded) == len(sample_text_list)
        assert all(isinstance(text, str) for text in batch_decoded)
        assert batch_decoded == sample_text_list


@pytest.mark.parametrize(
    "backend_str",
    ["hf_tokenizer", "tiktoken_tokenizer", "transformers_tokenizer"],
)
def test_batch_counting(
    request: pytest.FixtureRequest,
    backend_str: str,
    sample_text_list: list[str],
) -> None:
    """Test batch token counting with different backends."""
    try:
        tokenizer = request.getfixturevalue(backend_str)
        tokenizer = Tokenizer(tokenizer)
    except Exception as e:
        pytest.skip(f"Skipping test with backend {backend_str}: {str(e)}")

    # Test batch token count
    counts = tokenizer.count_tokens_batch(sample_text_list)
    assert isinstance(counts, list)
    assert len(counts) == len(sample_text_list)
    assert all(isinstance(c, int) for c in counts)
    assert all(c > 0 for c in counts)

    # Verify counts match encoded lengths
    if tokenizer._backend != "callable":
        encoded_lengths = [
            len(tokens) for tokens in tokenizer.encode_batch(sample_text_list)
        ]
        assert counts == encoded_lengths


def test_tokenizer_raises_error_with_invalid_tokenizer() -> None:
    """Test if Tokenizer raises ValueError when initialized with an invalid tokenizer."""
    with pytest.raises(ValueError):
        Tokenizer(object())


def test_raises_correct_error() -> None:
    """Test if tokenizers raise expected errors."""
    tokenizer = Tokenizer(lambda x: len(x))

    assert tokenizer.count_tokens("test") == 4

    with pytest.raises(NotImplementedError):
        tokenizer.encode(
            "Ratatouille or Wall-E? Tell us which is the best Pixar movie on Discord."
        )

    with pytest.raises(NotImplementedError):
        tokenizer.decode([0, 1, 2])

    with pytest.raises(NotImplementedError):
        tokenizer.encode_batch(["I", "Like", "Ratatouille", "Personally"])


### WordTokenizer Tests ###
def test_word_tokenizer_init(word_tokenizer: WordTokenizer) -> None:
    """Test WordTokenizer initialization."""
    assert word_tokenizer.vocab == [" "]
    assert len(word_tokenizer.token2id) == 1
    assert word_tokenizer.token2id[" "] == 0


def test_word_tokenizer_encode_decode(
    word_tokenizer: WordTokenizer, sample_text: str
) -> None:
    """Test WordTokenizer encoding and decoding."""
    tokens = word_tokenizer.encode(sample_text)
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)

    decoded = word_tokenizer.decode(tokens)
    assert isinstance(decoded, str)
    assert decoded.strip() == sample_text.strip()


def test_word_tokenizer_batch_encode_decode(
    word_tokenizer: WordTokenizer, sample_text_list: list[str]
) -> None:
    """Test batch encode and decode with WordTokenizer."""
    encoded_batch = word_tokenizer.encode_batch(sample_text_list)
    assert isinstance(encoded_batch, list)
    assert all(isinstance(tokens, list) for tokens in encoded_batch)

    decoded_batch = word_tokenizer.decode_batch(encoded_batch)
    assert isinstance(decoded_batch, list)
    assert all(isinstance(text, str) for text in decoded_batch)
    for decoded_text, original_text in zip(decoded_batch, sample_text_list):
        assert decoded_text.strip() == original_text.strip()


def test_word_tokenizer_vocab_appends_new_words(
    word_tokenizer: WordTokenizer,
) -> None:
    """Test WordTokenizer appends new words to the vocabulary."""
    initial_vocab_size = len(word_tokenizer.vocab)
    test_str = "every tech bro should watch wall-e"
    word_tokenizer.encode(test_str)
    assert len(word_tokenizer.vocab) > initial_vocab_size
    for word in test_str.split():
        assert word in word_tokenizer.vocab


def test_word_tokenizer_repr() -> None:
    """Test string representation of tokenizers."""
    word_tokenizer = WordTokenizer()
    assert str(word_tokenizer) == "WordTokenizer(vocab_size=1)"


def test_word_tokenizer_multiple_encodings(
    word_tokenizer: WordTokenizer,
) -> None:
    """Test that vocabulary changes as expected over multiple encodings."""
    str1 = "Wall-E is truly a masterpiece that should be required viewing."
    str2 = "Ratatouille is truly a delightful film that every kid should watch."

    # Test WordTokenizer
    word_tokenizer.encode(str1)
    vocab_size1 = len(word_tokenizer.get_vocab())
    word_tokenizer.encode(str2)
    vocab_size2 = len(word_tokenizer.get_vocab())

    assert vocab_size2 > vocab_size1
    assert "Wall-E" in word_tokenizer.get_vocab()
    assert "Ratatouille" in word_tokenizer.get_vocab()
    assert word_tokenizer.get_token2id()["truly"] == word_tokenizer.encode("truly")[0]


### CharacterTokenizer Tests ###
def test_character_tokenizer_init(
    character_tokenizer: CharacterTokenizer,
) -> None:
    """Test CharacterTokenizer initialization."""
    assert character_tokenizer.vocab == [" "]
    assert len(character_tokenizer.token2id) == 1
    assert character_tokenizer.token2id[" "] == 0


def test_character_tokenizer_encode_decode(
    character_tokenizer: CharacterTokenizer, sample_text: str
) -> None:
    """Test encoding and decoding with CharacterTokenizer."""
    tokens = character_tokenizer.encode(sample_text)
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)
    assert len(tokens) == len(sample_text)

    decoded = character_tokenizer.decode(tokens)
    assert isinstance(decoded, str)
    assert decoded == sample_text


def test_character_tokenizer_count_tokens(
    character_tokenizer: CharacterTokenizer,
    sample_text: str,
    sample_text_list: list[str],
) -> None:
    """Test token counting with CharacterTokenizer."""
    count = character_tokenizer.count_tokens(sample_text)
    assert count == len(sample_text)


def test_character_tokenizer_batch_encode_decode(
    character_tokenizer: CharacterTokenizer, sample_text_list: list[str]
) -> None:
    """Test batch encoding and decoding with CharacterTokenizer."""
    batch_encoded = character_tokenizer.encode_batch(sample_text_list)
    assert isinstance(batch_encoded, list)
    assert all(isinstance(tokens, list) for tokens in batch_encoded)
    assert all(
        len(tokens) == len(text)
        for tokens, text in zip(batch_encoded, sample_text_list)
    )

    batch_decoded = character_tokenizer.decode_batch(batch_encoded)
    assert isinstance(batch_decoded, list)
    assert all(isinstance(text, str) for text in batch_decoded)
    assert batch_decoded == sample_text_list


def test_character_tokenizer_count_tokens_batch(
    character_tokenizer: CharacterTokenizer,
    sample_text_list: list[str],
) -> None:
    """Test batch token counting with CharacterTokenizer."""
    counts = character_tokenizer.count_tokens_batch(sample_text_list)
    assert counts == [len(text) for text in sample_text_list]


def test_character_tokenizer_repr() -> None:
    """Test string representation of tokenizers."""
    character_tokenizer = CharacterTokenizer()
    assert str(character_tokenizer) == "CharacterTokenizer(vocab_size=1)"


def test_character_tokenizer_vocab_and_mapping(
    character_tokenizer: CharacterTokenizer, sample_text: str
) -> None:
    """Test vocabulary evolution in CharacterTokenizer."""
    # Initial state
    assert character_tokenizer.get_vocab() == [" "]
    assert dict(character_tokenizer.get_token2id()) == {" ": 0}

    character_tokenizer.encode(sample_text)

    # Encoding text should add vocabulary
    # and update token2id mapping
    vocab = character_tokenizer.get_vocab()
    token2id = character_tokenizer.get_token2id()

    # Spot check vocabulary
    assert len(vocab) > 1

    assert isinstance(token2id, dict)
    assert all(isinstance(token, str) for token in token2id.keys())
    assert all(isinstance(idx, int) for idx in token2id.values())
    assert token2id[" "] == 0

    # Verify mapping consistency
    for token in vocab:
        assert token in token2id
        assert vocab[token2id[token]] == token

    for char in sample_text:
        assert char in vocab
        assert char in token2id


def test_character_tokenizer_multiple_encodings(
    character_tokenizer: CharacterTokenizer,
) -> None:
    """Test that vocabulary changes as expected over multiple encodings."""
    text1 = "Wall-E is truly a masterpiece that should be required viewing."
    text2 = "Ratatouille is truly a delightful film that every kid should watch."

    character_tokenizer.encode(text1)
    vocab_size1 = len(character_tokenizer.get_vocab())
    character_tokenizer.encode(text2)
    vocab_size2 = len(character_tokenizer.get_vocab())

    assert vocab_size2 > vocab_size1
    assert "u" in character_tokenizer.get_vocab()
    assert character_tokenizer.get_token2id()["u"] == character_tokenizer.encode("u")[0]
