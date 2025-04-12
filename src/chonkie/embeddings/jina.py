"""Module for Jina AI embeddings integration."""

import os
import warnings
from typing import List, Optional

import numpy as np
import requests
from numpy.typing import NDArray

from .base import BaseEmbeddings


class JinaEmbeddings(BaseEmbeddings):
    """Jina embeddings implementation using their API."""

    def __init__(
            self,
            model: str = "jina-embeddings-v3",
            task: str = "text-matching",
            late_chunking: bool = True,
            embedding_type: str = "float",
            dimensions: int = 1024,
            api_key: Optional[str] = None,
            batch_size: int = 128,
            max_retries: int = 3
    ):
        """Initialize Jina embeddings.

        Args:
            model: Name of the Jina embedding model to use
            task: Task for the Jina model
            late_chunking: Whether to use late chunking
            embedding_type: Type of the embedding
            dimensions: Dimensions of the embedding
            api_key: Jina API key (if not provided, looks for JINA_API_KEY env var)
            batch_size: Maximum number of texts to embed in one API call
            max_retries: Maximum number of retries for API calls

        """
        super().__init__()
        self.model = model
        self.task = task
        self.late_chunking = late_chunking
        self._dimension = dimensions
        self.embedding_type = embedding_type
        self._batch_size = batch_size
        self._max_retries = max_retries
        
        self.api_key = self._get_api_key(api_key)        
        self.url = 'https://api.jina.ai/v1/embeddings'
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    def _get_api_key(self, api_key: Optional[str] = None) -> str:
        api_key = api_key or os.getenv("JINA_API_KEY")
        if not api_key:
            raise ValueError("Jina API key is required. Provide via api_key parameter or JINA_API_KEY environment variable")
        return api_key
    def embed(self, texts: List[str]) -> NDArray[np.float32]:
        """Embed the first text in a list using the Jina embeddings API.

        Note: This method processes only the first text even if the list contains multiple texts.

        Args:
            texts: List containing the text(s) to embed.

        Returns:
            Numpy array with the embedding for the first text in the input list.

        Raises:
            ValueError: If the input `texts` list is empty.
            requests.exceptions.RequestException: If the API request fails after retries.

        """
        if not texts:
            raise ValueError("At least one text must be provided")
            
        data = {
            "model": self.model,
            "task": self.task,
            "late_chunking": self.late_chunking,
            "embedding_type": self.embedding_type,
            "dimensions": self._dimension,
            "input": texts
        }

        for attempt in range(self._max_retries):
            try:
                response = requests.post(self.url, json=data, headers=self.headers)
                response.raise_for_status()
                vector = response.json()
                data = vector.get('data')
                if not data or not data[0] or 'embedding' not in data[0]:
                    raise ValueError(f"Unexpected API response format: {vector}")
                return np.array(data[0]['embedding'], dtype=np.float32)            
            except requests.exceptions.RequestException as e:
                if attempt == self._max_retries - 1:
                    raise
                warnings.warn(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
    
    def embed_batch(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Embed multiple texts using the Jina embeddings API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of numpy arrays with embeddings for each text.

        Raises:
            requests.exceptions.HTTPError: If the initial API request for a batch fails
                and the batch contained only one text.
            ValueError: If the API response format is unexpected, or if the fallback
                to single embedding fails for a text within a failed batch.
            requests.exceptions.RequestException: If an API request fails after all retries
                (either batch or single fallback).

        """
        if not texts:
            return []
            
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            payload = {
                "model": self.model,
                "task": self.task,
                "late_chunking": self.late_chunking,
                "embedding_type": self.embedding_type,
                "dimensions": self._dimension,
                "input": batch
            }
            
            try:
                response = requests.post(self.url, json=payload, headers=self.headers)
                response.raise_for_status()
                response_data = response.json()
                embeddings = [
                    np.array(item['embedding'], dtype=np.float32) 
                    for item in response_data['data']
                ]
                all_embeddings.extend(embeddings)
            except requests.exceptions.HTTPError as e:
                if len(batch) > 1:
                    warnings.warn(f"Batch embedding failed: {str(e)}. Trying one by one...")
                    # Fall back to single embeddings
                    single_embeddings = [self.embed([t]) if isinstance(t, str) else np.array([]) for idx, t in enumerate(batch)] # or raise an error
                    all_embeddings.extend(single_embeddings)
                else:
                    raise ValueError(f"Failed to embed text: {batch} due to: {e}")                    
        return all_embeddings

    def similarity(self, u: NDArray[np.float32], v: NDArray[np.float32]) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            u: First embedding vector
            v: Second embedding vector
            
        Returns:
            Cosine similarity between u and v (float between -1 and 1)

        """
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return 0.0  # or raise an exception, depending on desired behavior
        return float(np.dot(u, v) / (norm_u * norm_v))    
    def count_tokens(self, text: str, tokenizer: str = 'cl100k_base') -> int:
        """Count tokens in text using the Jina segmenter.

        Args:
            text: The input text.
            tokenizer: The tokenizer model to use (default: 'cl100k_base').

        Returns:
            The number of tokens in the text.

        Raises:
            requests.exceptions.RequestException: If the API request fails after retries.

        """
        if not text:
            return 0
            
        url = 'https://api.jina.ai/v1/segment'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._get_api_key()}'
        }

        data = {
            "content": text,
            "tokenizer": tokenizer
        }
        
        for attempt in range(self._max_retries):
            try:
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                return response.json()['num_tokens']
            except requests.exceptions.RequestException as e:
                if attempt == self._max_retries - 1:
                    raise
                warnings.warn(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts."""
        return [self.count_tokens(text) for text in texts]
    
    @property
    def dimension(self) -> int:
        """Return the dimensions of the embeddings."""
        return self._dimension
        
    def get_tokenizer_or_token_counter(self):
        """Get the tokenizer or token counter for the embeddings."""
        return self.count_tokens
    
    def __repr__(self) -> str:
        """Return a string representation of the JinaEmbeddings instance."""
        return f"JinaEmbeddings(model={self.model}, dimensions={self._dimension})"