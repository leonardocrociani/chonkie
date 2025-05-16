# scripts/test_cython_chunker.py
from chonkie.chunker.c_extensions.token_chunker import generate_token_groups, process_batch_chunks
import array
from chonkie import Tokenizer
from chonkie.chunker.token import TokenChunker
import timeit
import numpy as np

def test_manual() -> None:
    # Test single document
    tokens = array.array('i', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = generate_token_groups(tokens, chunk_size=3, chunk_overlap=1)
    print("Single document result:", [list(chunk) for chunk in result])
    
    # Test batch
    token_arrays = [
        array.array('i', [1, 2, 3, 4, 5]),
        array.array('i', [6, 7, 8, 9, 10])
    ]
    result = process_batch_chunks(token_arrays, chunk_size=3, chunk_overlap=1)
    print("Batch result:", [[list(chunk) for chunk in group] for group in result])

def test_tokenizer_to_chunks(tokenizer: Tokenizer) -> None:
    text = "This is a test sentence for chunking."
    tokens = tokenizer.encode(text)
    token_array = array.array('i', tokens)
    token_groups = generate_token_groups(token_array, chunk_size=3, chunk_overlap=1)
    print("Token groups (cython):", [list(chunk) for chunk in token_groups])

    # Pure Python approach using TokenChunker
    token_chunker = TokenChunker(tokenizer=tokenizer, chunk_size=3, chunk_overlap=1, return_type="texts")
    # TokenChunker.chunk returns a list of strings (since return_type="texts")
    # To get the token groups, we need to use the private _token_group_generator
    token_groups_py = list(token_chunker._token_group_generator(tokens))
    print("Token groups (pure python):", token_groups_py)

def benchmark_token_group_generation(tokenizer: Tokenizer, chunk_size: int = 16, chunk_overlap: int = 4, n_runs: int = 100) -> None:
    text = (
        "This is a test sentence for chunking. " * 1000  # Make it long for benchmarking
    )
    tokens = tokenizer.encode(text)
    token_array = array.array('i', tokens)
    token_chunker = TokenChunker(tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap, return_type="texts")

    def cython_chunk() -> None:
        list(generate_token_groups(token_array, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    def python_chunk() -> None:
        list(token_chunker._token_group_generator(tokens))

    cython_time = timeit.timeit(cython_chunk, number=n_runs)
    python_time = timeit.timeit(python_chunk, number=n_runs)

    print(f"Cython token group generation: {cython_time / n_runs * 1000:.3f} ms per run (avg over {n_runs})")
    print(f"Pure Python token group generation: {python_time / n_runs * 1000:.3f} ms per run (avg over {n_runs})")

def benchmark_batch_token_group_generation(tokenizer: Tokenizer, chunk_size: int = 16, chunk_overlap: int = 4, n_runs: int = 100, batch_size: int = 32) -> None:
    # Create a batch of texts
    base_text = "This is a test sentence for chunking. " * 20
    texts = [base_text + str(i) for i in range(batch_size)]
    # Encode all texts
    tokens_batch = [tokenizer.encode(text) for text in texts]
    # Prepare array.array('i') batch for cython
    token_arrays = [array.array('i', tokens) for tokens in tokens_batch]
    token_chunker = TokenChunker(tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap, return_type="texts")

    def cython_batch_chunk() -> None:
        process_batch_chunks(token_arrays, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def python_batch_chunk() -> None:
        [list(token_chunker._token_group_generator(tokens)) for tokens in tokens_batch]

    cython_time = timeit.timeit(cython_batch_chunk, number=n_runs)
    python_time = timeit.timeit(python_batch_chunk, number=n_runs)

    print(f"Cython batch token group generation: {cython_time / n_runs * 1000:.3f} ms per run (avg over {n_runs})")
    print(f"Pure Python batch token group generation: {python_time / n_runs * 1000:.3f} ms per run (avg over {n_runs})")

def benchmark_batch_token_group_generation_numpy(tokenizer: Tokenizer, chunk_size: int = 16, chunk_overlap: int = 4, n_runs: int = 100, batch_size: int = 32) -> None:
    # Create a batch of texts
    base_text = "This is a test sentence for chunking. " * 20
    texts = [base_text + str(i) for i in range(batch_size)]
    # Encode all texts
    tokens_batch = [tokenizer.encode(text) for text in texts]
    # Prepare numpy arrays for cython
    token_arrays = [np.array(tokens, dtype=np.int32) for tokens in tokens_batch]
    print("First numpy array dtype:", token_arrays[0].dtype, "shape:", token_arrays[0].shape)
    token_chunker = TokenChunker(tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap, return_type="texts")

    def cython_batch_chunk() -> None:
        process_batch_chunks(token_arrays, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def python_batch_chunk() -> None:
        [list(token_chunker._token_group_generator(tokens)) for tokens in tokens_batch]

    cython_time = timeit.timeit(cython_batch_chunk, number=n_runs)
    python_time = timeit.timeit(python_batch_chunk, number=n_runs)

    print(f"Cython batch token group generation (numpy): {cython_time / n_runs * 1000:.3f} ms per run (avg over {n_runs})")
    print(f"Pure Python batch token group generation: {python_time / n_runs * 1000:.3f} ms per run (avg over {n_runs})")

def test_batch_with_python_lists() -> None:
    # Test batch with plain Python lists
    token_arrays = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10]
    ]
    try:
        result = process_batch_chunks(token_arrays, chunk_size=3, chunk_overlap=1)
        print("Batch result (python lists):", [[list(chunk) for chunk in group] for group in result])
    except Exception as e:
        print("Exception in batch with python lists:", e)

def profile_token_chunker_large_text() -> None:
    tokenizer = Tokenizer("gpt2")
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=32, chunk_overlap=8, return_type="chunks")
    large_text = ("This is a test sentence for chunking. " * 10000).strip()
    chunks = chunker.chunk(large_text)
    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk: {chunks[0]}")
    print(f"Last chunk: {chunks[-1]}")

if __name__ == "__main__":
    # test_manual()
    tokenizer = Tokenizer("gpt2")
    # test_tokenizer_to_chunks(tokenizer)
    # benchmark_token_group_generation(tokenizer)
    # benchmark_batch_token_group_generation(tokenizer)
    # benchmark_batch_token_group_generation_numpy(tokenizer)
    # test_batch_with_python_lists()
    # Uncomment the next line to profile TokenChunker on a large string:
    profile_token_chunker_large_text()