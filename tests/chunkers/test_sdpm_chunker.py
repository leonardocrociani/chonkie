"""Test for SDPMChunker class."""

import os

import pytest

from chonkie import SDPMChunker
from chonkie.embeddings import CohereEmbeddings, Model2VecEmbeddings, OpenAIEmbeddings


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
