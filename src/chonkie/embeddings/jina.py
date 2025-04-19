"""Module for Jina AI embeddings integration."""

import os
import warnings
from typing import List, Optional

import numpy as np
import requests
from numpy.typing import NDArray
from transformers import AutoTokenizer

from .base import BaseEmbeddings


class JinaEmbeddings(BaseEmbeddings):
    """Jina embeddings implementation using their API."""

    AVAILABLE_MODELS = {
        "jina-embeddings-v3": 1024 
    }

    def __init__(
            self,
            model: str = "jina-embeddings-v3",
            task: str = "text-matching",
            late_chunking: bool = True,
            embedding_type: str = "float",
            api_key: Optional[str] = None,
            batch_size: int = 128,
            max_retries: int = 3
    ):
        """Initialize Jina embeddings.

        Args:
            model (str): Name of the Jina embedding model to use.
            task (str): Task for the Jina model.
            late_chunking (bool): Whether to use late chunking.
            embedding_type (str): Type of the embedding.
            api_key (Optional[str]): Jina API key (if not provided, looks for
                JINA_API_KEY env var).
            batch_size (int): Maximum number of texts to embed in one API call.
            max_retries (int): Maximum number of retries for API calls.

        """
        super().__init__()

        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model} not available. Choose from: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.model = model
        self.task = task
        self.late_chunking = late_chunking
        self._dimension = self.AVAILABLE_MODELS[model]
        self.embedding_type = embedding_type
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3')
        self.api_key = self._get_api_key(api_key)        
        self.url = 'https://api.jina.ai/v1/embeddings'
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _get_api_key(self, api_key: Optional[str] = None) -> str:
        """Retrieve the Jina API key from parameter or environment variable.

        Args:
            api_key (Optional[str]): The API key passed directly.

        Returns:
            str: The validated Jina API key.

        Raises:
            ValueError: If the API key is not provided either via the parameter
                or the JINA_API_KEY environment variable.

        """
        key = api_key or os.getenv("JINA_API_KEY")
        if not key:
            raise ValueError("Jina API key is required. Provide via api_key parameter or JINA_API_KEY environment variable")
        return key
    def embed(self, texts: List[str]) -> NDArray[np.float32]:
        """Embed the first text in a list using the Jina embeddings API.

        Note: This method processes only the first text even if the list contains multiple texts, it is for embedding single text.

        Args:
            texts (List[str]): List containing the text(s) to embed.

        Returns:
            NDArray[np.float32]: Numpy array with the embedding for the first text.

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
            texts (List[str]): List of texts to embed.

        Returns:
            List[NDArray[np.float32]]: List of numpy arrays with embeddings for each text.

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
                    warnings.warn(f"Fallback to single embeddings due to: {str(e)}")
                    # Fall back to single embeddings
                    single_embeddings = []
                    for t in batch:
                        if isinstance(t, str):
                            single_embeddings.append(self.embed([t]))
                        else:
                            raise ValueError(f"Invalid text type found in batch: {type(t)}")
                    all_embeddings.extend(single_embeddings)
                else:
                    raise ValueError(f"Failed to embed text: {batch} due to: {e}")                    
        return all_embeddings

    def similarity(self, u: NDArray[np.float32], v: NDArray[np.float32]) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            u (NDArray[np.float32]): First embedding vector.
            v (NDArray[np.float32]): Second embedding vector.

        Returns:
            float: Cosine similarity between u and v.

        """
        return np.divide(
            np.dot(u, v), np.linalg.norm(u) * np.linalg.norm(v), dtype=float
        )

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the input text using Jina's tokenizer.

        Args:
            text (str): Input text to tokenize. Leading/trailing whitespace is ignored.
 
        Returns:
            int: Number of tokens (using the same tokenization rules as the embedding model).

        """
        if not text.strip():
            return 0

        tokens = self._tokenizer.tokenize(text)
        return len(tokens)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts.

        Args:
            texts (List[str]): List of texts to count tokens for.

        Returns:
            List[int]: A list containing the token count for each text.

        """
        return [self.count_tokens(text) for text in texts]
    
    @property
    def dimension(self) -> int:
        """Return the dimensions of the embeddings.

        Returns:
            int: The embedding dimension size.

        """
        return self._dimension
        
    def get_tokenizer_or_token_counter(self):
        """Get the tokenizer instance used by the embeddings model.

        Returns:
            PreTrainedTokenizerFast: The Hugging Face tokenizer instance.

        """
        return self._tokenizer

    def __repr__(self) -> str:
        """Return a string representation of the JinaEmbeddings instance.

        Returns:
            str: A string representation of the instance.

        """
        return f"JinaEmbeddings(model={self.model}, dimensions={self._dimension})"