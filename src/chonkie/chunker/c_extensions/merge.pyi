"""Stub file for merge C extension."""

from typing import List, Tuple

def _merge_splits(
    splits: List[str], 
    token_counts: List[int], 
    chunk_size: int, 
    combine_whitespace: bool = False
) -> Tuple[List[str], List[int]]:
    """Merge text splits into chunks based on token counts.
    
    Args:
        splits: List of text splits to merge
        token_counts: Token count for each split
        chunk_size: Maximum tokens per chunk
        combine_whitespace: Whether to combine with whitespace
        
    Returns:
        Tuple of (merged_chunks, combined_token_counts)

    """
    ...

def find_merge_indices(
    token_counts: List[int],
    chunk_size: int,
    start_index: int = 0
) -> List[int]:
    """Find optimal merge indices for token counts.
    
    Args:
        token_counts: List of token counts for each segment
        chunk_size: Maximum tokens per chunk
        start_index: Starting index for merging
        
    Returns:
        List of merge indices

    """
    ...