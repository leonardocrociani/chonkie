"""Test for SDPMChunker class."""

import os

import pytest

from chonkie import SDPMChunker
from chonkie.embeddings import CohereEmbeddings, Model2VecEmbeddings, OpenAIEmbeddings
from chonkie.types import SemanticChunk


@pytest.fixture
def sample_text():
    """Sample text for testing the SemanticChunker.

    Returns:
        str: A paragraph of text about text chunking in RAG applications.

    """
    text = """The process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that leverages recent advances in transformer-based language models to achieve a more effective balance between these competing requirements."""
    return text


@pytest.fixture
def embedding_model():
    """Fixture that returns a Model2Vec embedding model for testing.

    Returns:
        Model2VecEmbeddings: A Model2Vec model initialized with 'minishlab/potion-base-8M'

    """
    return Model2VecEmbeddings("minishlab/potion-base-8M")


@pytest.fixture
def openai_embedding_model():
    """Fixture that returns an OpenAI embedding model for testing.

    Returns:
        OpenAIEmbeddings: An OpenAI model initialized with 'text-embedding-3-small'
            and the API key from environment variables.

    """
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)


@pytest.fixture
def cohere_embedding_model():
    """Fixture that returns an Cohere embedding model for testing.

    Returns:
        CohereEmbeddings: An Cohere model initialized with 'embed-english-light-v3.0'
            and the API key from environment variables.

    """
    api_key = os.environ.get("COHERE_API_KEY")
    return CohereEmbeddings(model="embed-english-light-v3.0", api_key=api_key)


@pytest.fixture
def sample_complex_markdown_text():
    """Fixture that returns a sample markdown text with complex formatting.

    Returns:
        str: A markdown text containing various formatting elements like headings,
            lists, code blocks, links, images and blockquotes.

    """
    text = """# Heading 1
    This is a paragraph with some **bold text** and _italic text_.
    ## Heading 2
    - Bullet point 1
    - Bullet point 2 with `inline code`
    ```python
    # Code block
    def hello_world():
        print("Hello, world!")
    ```
    Another paragraph with [a link](https://example.com) and an image:
    ![Alt text](https://example.com/image.jpg)
    > A blockquote with multiple lines
    > that spans more than one line.
    Finally, a paragraph at the end.
    """
    return text


def test_sdpm_chunker_initialization(embedding_model):
    """Test that the SemanticChunker can be initialized with required parameters."""
    chunker = SDPMChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.threshold == 0.5
    assert chunker.mode == "window"
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2
    assert chunker.skip_window == 1


def test_sdpm_chunker_initialization_sentence_transformer():
    """Test that the SemanticChunker can be initialized with SentenceTransformer model."""
    chunker = SDPMChunker(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        threshold=0.5,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.threshold == 0.5
    assert chunker.mode == "window"
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2


@pytest.mark.skipif(
    "COHERE_API_KEY" not in os.environ,
    reason="Skipping test because COHERE_API_KEY is not defined",
)
def test_sdpm_chunker_initialization_cohere(cohere_embedding_model):
    """Test that the SemanticChunker can be initialized with required parameters."""
    chunker = SDPMChunker(
        embedding_model=cohere_embedding_model,
        chunk_size=512,
        threshold=0.5,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.threshold == 0.5
    assert chunker.mode == "window"
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2


def test_sdpm_chunker_chunking(embedding_model, sample_text):
    """Test that the SemanticChunker can chunk a sample text."""
    chunker = SDPMChunker(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert isinstance(chunks[0], SemanticChunk)
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])
    assert all([chunk.sentences is not None for chunk in chunks])


def test_semantic_chunker_empty_text(embedding_model):
    """Test that the SemanticChunker can handle empty text input."""
    chunker = SDPMChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk("")

    assert len(chunks) == 0


def test_semantic_chunker_single_sentence(embedding_model):
    """Test that the SemanticChunker can handle text with a single sentence."""
    chunker = SDPMChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )
    chunks = chunker.chunk("This is a single sentence.")

    assert len(chunks) == 1
    assert chunks[0].text == "This is a single sentence."
    assert len(chunks[0].sentences) == 1


def test_semantic_chunker_repr(embedding_model):
    """Test that the SemanticChunker has a string representation."""
    chunker = SDPMChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
    )

    expected = (
        f"SemanticChunker(model={chunker.embedding_model}, "
        f"chunk_size={chunker.chunk_size}, "
        f"mode={chunker.mode}, "
        f"threshold={chunker.threshold}, "
        f"similarity_window={chunker.similarity_window}, "
        f"min_sentences={chunker.min_sentences}, "
        f"min_chunk_size={chunker.min_chunk_size}, "
        f"return_type={chunker.return_type})"
    )
    assert repr(chunker) == expected


def test_sdpm_chunker_reconstruction(embedding_model, sample_text):
    """Test that the SemanticChunker can reconstruct the original text."""
    chunker = SDPMChunker(
        embedding_model=embedding_model, chunk_size=512, threshold=0.5
    )
    chunks = chunker.chunk(sample_text)
    assert sample_text == "".join([chunk.text for chunk in chunks])


def test_sdpm_chunker_reconstruction_complex_md(
    embedding_model, sample_complex_markdown_text
):
    """Test that the SemanticChunker can reconstruct the original text."""
    chunker = SDPMChunker(
        embedding_model=embedding_model, chunk_size=512, threshold=0.5
    )
    chunks = chunker.chunk(sample_complex_markdown_text)
    assert sample_complex_markdown_text == "".join([chunk.text for chunk in chunks])


def test_sdpm_chunker_reconstruction_batch(embedding_model, sample_text):
    """Test that the SemanticChunker can reconstruct the original text."""
    chunker = SDPMChunker(
        embedding_model=embedding_model, chunk_size=512, threshold=0.5
    )
    chunks = chunker.chunk_batch([sample_text] * 10)[-1]
    assert sample_text == "".join([chunk.text for chunk in chunks])


def test_sdpm_chunker_return_type(embedding_model, sample_text):
    """Test that SemanticChunker's return type is correctly set."""
    chunker = SDPMChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        threshold=0.5,
        return_type="texts",
    )
    chunks = chunker.chunk(sample_text)
    tokenizer = embedding_model.get_tokenizer_or_token_counter()
    assert all([type(chunk) is str for chunk in chunks])
    assert all([len(tokenizer.encode(chunk)) <= 512 for chunk in chunks])
