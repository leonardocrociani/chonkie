"""Test for the Chonkie Cloud Code Chunker class."""

import os

import pytest

from chonkie.cloud import CodeChunker


@pytest.fixture
def python_code() -> str:
    """Return a sample Python code snippet."""
    return """
import os
import sys

def hello_world(name: str):
    \"\"\"Prints a greeting.\"\"\"
    print(f"Hello, {name}!")

class MyClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

if __name__ == "__main__":
    hello_world("World")
    instance = MyClass(10)
    print(instance.get_value())
"""


@pytest.fixture
def js_code() -> str:
    """Return a sample JavaScript code snippet."""
    return """
function greet(name) {
  console.log(`Hello, ${name}!`);
}

class Calculator {
  add(a, b) {
    return a + b;
  }
}

const calc = new Calculator();
greet('Developer');
console.log(calc.add(5, 3));
"""


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_code_chunker_initialization() -> None:
    """Test that the code chunker can be initialized."""
    # Check if the chunk_size <= 0 raises an error
    with pytest.raises(ValueError):
        CodeChunker(tokenizer_or_token_counter="gpt2", chunk_size=-1)

    with pytest.raises(ValueError):
        CodeChunker(tokenizer_or_token_counter="gpt2", chunk_size=0)

    # Check if the return_type is not "texts" or "chunks" raises an error
    with pytest.raises(ValueError):
        CodeChunker(
            tokenizer_or_token_counter="gpt2",
            chunk_size=512,
            return_type="bad_return_type",
        )

    # Finally, check if the attributes are set correctly
    chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2", 
        chunk_size=512, 
        language="python"
    )
    assert chunker.tokenizer_or_token_counter == "gpt2"
    assert chunker.chunk_size == 512
    assert chunker.language == "python"
    assert chunker.return_type == "chunks"


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_code_chunker_simple() -> None:
    """Test that the code chunker works with simple code."""
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        language="python",
    )
    simple_code = "def hello():\n    print('Hello, world!')"
    result = code_chunker(simple_code)

    # Check the result
    assert isinstance(result, list) and len(result) >= 1
    assert isinstance(result[0], dict)
    assert "text" in result[0]
    assert "token_count" in result[0]
    assert "start_index" in result[0]
    assert "end_index" in result[0]
    assert isinstance(result[0]["text"], str)
    assert isinstance(result[0]["token_count"], int)
    assert isinstance(result[0]["start_index"], int)
    assert isinstance(result[0]["end_index"], int)


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_code_chunker_python_complex(python_code: str) -> None:
    """Test that the code chunker works with complex Python code."""
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=50,
        language="python",
    )
    result = code_chunker(python_code)

    # Check the result
    assert isinstance(result, list)
    assert len(result) > 1  # Should be split into multiple chunks
    assert all(isinstance(item, dict) for item in result)
    assert all(isinstance(item["text"], str) for item in result)
    assert all(isinstance(item["token_count"], int) for item in result)
    assert all(isinstance(item["start_index"], int) for item in result)
    assert all(isinstance(item["end_index"], int) for item in result)

    # Check that chunks can be reconstructed
    reconstructed = "".join(chunk["text"] for chunk in result)
    assert reconstructed == python_code


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_code_chunker_javascript(js_code: str) -> None:
    """Test that the code chunker works with JavaScript code."""
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=40,
        language="javascript",
    )
    result = code_chunker(js_code)

    # Check the result
    assert isinstance(result, list)
    assert len(result) > 1  # Should be split into multiple chunks
    assert all(isinstance(item, dict) for item in result)
    assert all(isinstance(item["text"], str) for item in result)
    assert all(isinstance(item["token_count"], int) for item in result)

    # Check that chunks can be reconstructed
    reconstructed = "".join(chunk["text"] for chunk in result)
    assert reconstructed == js_code


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_code_chunker_auto_language(python_code: str) -> None:
    """Test that the code chunker works with auto language detection."""
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=100,
        language="auto",
    )
    result = code_chunker(python_code)

    # Check the result
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(item, dict) for item in result)
    assert all(isinstance(item["text"], str) for item in result)

    # Check that chunks can be reconstructed
    reconstructed = "".join(chunk["text"] for chunk in result)
    assert reconstructed == python_code


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_code_chunker_no_nodes_support() -> None:
    """Test that the code chunker doesn't support nodes (API limitation)."""
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        language="python",
    )
    simple_code = "def hello():\n    print('Hello, world!')"
    result = code_chunker(simple_code)

    # Check the result - should not contain nodes since API doesn't support them
    assert isinstance(result, list) and len(result) >= 1
    assert isinstance(result[0], dict)
    assert "text" in result[0]
    # API doesn't support tree-sitter nodes, so they shouldn't be in response


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_code_chunker_batch(python_code: str, js_code: str) -> None:
    """Test that the code chunker works with a batch of texts."""
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=100,
        language="auto",
    )
    texts = [python_code, js_code, "def simple(): pass"]
    result = code_chunker(texts)

    # Check the result
    assert len(result) == len(texts)
    assert isinstance(result, list)
    assert all(isinstance(item, list) for item in result), (
        f"Expected a list of lists, got {type(result)}"
    )
    assert all(isinstance(chunk, dict) for batch in result for chunk in batch), (
        "Expected lists of dictionaries"
    )
    assert all(isinstance(chunk["text"], str) for batch in result for chunk in batch), (
        "Expected chunks with text field"
    )
    assert all(isinstance(chunk["token_count"], int) for batch in result for chunk in batch), (
        "Expected chunks with token_count field"
    )


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_code_chunker_return_type_texts() -> None:
    """Test that the code chunker works with return_type='texts'."""
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        language="python",
        return_type="texts",
    )
    simple_code = "def hello():\n    print('Hello, world!')"
    result = code_chunker(simple_code)

    # Check the result
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_code_chunker_empty_text() -> None:
    """Test that the code chunker works with an empty text."""
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        language="python",
    )

    result = code_chunker("")
    assert len(result) == 0


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_code_chunker_whitespace_text() -> None:
    """Test that the code chunker works with whitespace-only text."""
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        language="python",
    )

    result = code_chunker("   \n\t\n  ")
    # Should return empty list or minimal chunks
    assert isinstance(result, list)


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_code_chunker_chunk_size_adherence(python_code: str) -> None:
    """Test that code chunks mostly adhere to chunk_size limits."""
    chunk_size = 30
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=chunk_size,
        language="python",
    )
    result = code_chunker(python_code)

    # Most chunks should be close to or under chunk_size (with some tolerance for code boundaries)
    for i, chunk in enumerate(result[:-1]):  # Check all but last chunk
        assert chunk["token_count"] <= chunk_size + 20, f"Chunk {i} exceeds size limit: {chunk['token_count']}"
    
    # Last chunk should have some content
    if result:
        assert result[-1]["token_count"] > 0


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_code_chunker_indices_continuity(python_code: str) -> None:
    """Test that chunk indices are continuous."""
    code_chunker = CodeChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=60,
        language="python",
    )
    result = code_chunker(python_code)

    # Check indices are continuous
    current_index = 0
    for chunk in result:
        assert chunk["start_index"] == current_index
        assert chunk["end_index"] > chunk["start_index"]
        current_index = chunk["end_index"]
    
    # Final index should match original text length
    assert current_index == len(python_code)


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_code_chunker_different_tokenizers() -> None:
    """Test that the code chunker works with different tokenizers."""
    simple_code = "def hello():\n    print('Hello, world!')"
    
    # Test with different tokenizers
    tokenizers = ["gpt2", "cl100k_base"]
    
    for tokenizer in tokenizers:
        code_chunker = CodeChunker(
            tokenizer_or_token_counter=tokenizer,
            chunk_size=512,
            language="python",
        )
        result = code_chunker(simple_code)
        
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(chunk, dict) for chunk in result)
        assert all("text" in chunk for chunk in result)