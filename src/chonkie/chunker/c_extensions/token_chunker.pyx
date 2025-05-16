# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import cython
from cpython.list cimport PyList_New, PyList_Append
from libc.stdlib cimport malloc, free

def generate_token_groups(int[:] tokens, int chunk_size, int chunk_overlap):
    print("[Cython] generate_token_groups: tokens type:", type(tokens), "len:", len(tokens))
    cdef:
        int i, start, end
        int step = chunk_size - chunk_overlap
        int n_tokens = len(tokens)
        list result = PyList_New(0)
        int[:] current_chunk
    
    for start in range(0, n_tokens, step):
        end = min(start + chunk_size, n_tokens)
        current_chunk = tokens[start:end]
        print(f"[Cython] start: {start}, end: {end}, current_chunk: {list(current_chunk)}")
        PyList_Append(result, list(current_chunk))
        if end == n_tokens:
            break
    
    print("[Cython] Returning result from generate_token_groups, len:", len(result))
    return result

def process_batch_chunks(list token_arrays, int chunk_size, int chunk_overlap):
    print("[Cython] process_batch_chunks: token_arrays type:", type(token_arrays), "len:", len(token_arrays))
    cdef:
        int i
        int n_arrays = len(token_arrays)
        list result = PyList_New(0)
        int[:] tokens
        list token_groups
    
    for i in range(n_arrays):
        print(f"[Cython] process_batch_chunks: processing array {i}, type: {type(token_arrays[i])}, len: {len(token_arrays[i])}")
        tokens = token_arrays[i]
        print(f"[Cython] process_batch_chunks: before generate_token_groups for array {i}")
        token_groups = generate_token_groups(tokens, chunk_size, chunk_overlap)
        print(f"[Cython] process_batch_chunks: after generate_token_groups for array {i}")
        PyList_Append(result, token_groups)
    
    return result
