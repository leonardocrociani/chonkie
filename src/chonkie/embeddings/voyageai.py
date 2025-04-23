import importlib
import os
import warnings
from typing import TYPE_CHECKING, Any, List, Literal, Optional

from .base import BaseEmbeddings



if TYPE_CHECKING:
    try: 
        import numpy as np
        from transformers import AutoTokenizer
        import voyageai
    except ImportError:
        np = Any # type: ignore
        AutoTokenizer = Any # type: ignore
        voyageai = Any # type: ignore

class VoyageAIEmbeddings(BaseEmbeddings):
    """
    Voyage Embeddings client for interfacing with the VoyageAI API.

    This class provides synchronous and asynchronous methods to obtain embeddings
    for single texts or batches, with optional truncation and configurable output dimension.
    """

    # Supported models with (dimension, max_tokens)
    AVAILABLE_MODELS = {
        "voyage-3-large": (1024, 32000), # The best general-purpose and multilingual retrieval quality
        "voyage-3": (1024, 32000), # Optimized for general-purpose and multilingual retrieval quality 
        "voyage-3-lite": (512, 32000), # Optimized for latency and cost
        "voyage-code-3": (1024, 32000), # Optimized for code retrieval
        "voyage-finance-2": (1024, 32000), # Optimized for finance retrieval and RAG
        "voyage-law-2": (1024, 16000), # Optimized for legal retrieval and RAG
        "voyage-code-2": (1536, 16000), # Optimized for code retrieval (17% better than alternatives) / Previous generation of code embeddings
    }
    DEFAULT_MODEL = "voyage-3"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        output_dimension: Optional[Literal[256, 512, 1024, 2048]] = None,
        batch_size: int = 128,
        truncation: bool = True,
    ):
        """
        Initialize the VoyageAI embeddings client.

        Args:
            model: Name of the Voyage model to use (must be in AVAILABLE_MODELS).
            api_key: API key for authentication (or set VOYAGEAI_API_KEY env var).
            max_retries: Maximum retry attempts for API calls.
            timeout: Timeout in seconds for API requests.
            output_dimension: Optional target embedding dimension.
            batch_size: Number of texts per batch (max 128).
            truncation: Whether to truncate inputs exceeding model token limit.

        Raises:
            ValueError: If model is unsupported or invalid output_dimension.
            ImportError: If voyageai package is not installed.
        """
        super().__init__()

        # Lazy import dependencies
        self._import_dependencies()

        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model!r} not available. Choose from: {list(self.AVAILABLE_MODELS)}"
            )
        self.model = model
        self._token_limit = self.AVAILABLE_MODELS[model][1]
        self._dimension = self.AVAILABLE_MODELS[model][0]

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(f"voyageai/{model}", trust_remote_code=True)
        except Exception as e:
            raise ValueError(f"Failed to initialize tokenizer for model {model}: {e}")

        # API clients
        key = api_key or os.getenv("VOYAGEAI_API_KEY")
        if not key:
            raise ValueError(
                "VoyageAI API key not found. "
                "Set `api_key` or VOYAGEAI_API_KEY environment variable."
            )
        self._client = voyageai.Client(
            api_key=key, max_retries=max_retries, timeout=timeout
        )
        self._aclient = voyageai.AsyncClient(
            api_key=key, max_retries=max_retries, timeout=timeout
        )

        self.truncation = truncation
        self.batch_size = min(batch_size, 128)
        if output_dimension is not None and output_dimension not in {256, 512, 1024, 2048}:
            raise ValueError(
                f"Invalid output_dimension: {output_dimension}. Must be one of {256,512,1024,2048} or None."
            )
        self.output_dimension = output_dimension

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a given text, applying truncation if enabled."""
        return len(
        self._tokenizer.encode(
            text,
            truncation=self.truncation,
            max_length=self._token_limit,
        )
    )
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts."""
        output = self._tokenizer.batch_encode_plus(
            texts,
            truncation=self.truncation,
            max_length=self._token_limit,
            return_length=True,
        )
        return output["length"]

    def embed(self, text: str, input_type: Literal["query", "document"] = None) -> "np.ndarray":
        """
        Obtain embedding for a single text synchronously.

        Args:
            text: The input string to embed.
            input_type: Optional tag indicating 'query' or 'document'.
        Returns:
            A NumPy array of the embedding vector.
        """
        tokens = self.count_tokens(text)
        if tokens > self._token_limit and self.truncation:
            warnings.warn(
                f"Input has {tokens} tokens (>{self._token_limit}); truncating.",
                UserWarning,
            )
        try:
            response = self._client.embed(
                texts=[text],
                model=self.model,
                input_type=input_type,
                truncation=self.truncation,
                output_dimension=self.output_dimension
            )
        except Exception as e:
            raise RuntimeError(f"VoyageAI API error during embedding: {e}") from e
        
        return np.array(response.embeddings[0], dtype=np.float32)

    async def aembed(self, text: str, input_type: Literal["query", "document"] = None) -> "np.ndarray":
        """
        Obtain embedding for a single text asynchronously.

        Args:
            text: The input string to embed.
            input_type: Optional tag indicating 'query' or 'document'.
        Returns:
            A NumPy array of the embedding vector.
        """
        tokens = self.count_tokens(text)
        if tokens > self._token_limit and self.truncation:
            warnings.warn(
                f"Input has {tokens} tokens (>{self._token_limit}); truncating.",
                UserWarning,
            )

        try:
            response = await self._aclient.embed(
                texts=[text],
                model=self.model,
                input_type=input_type,
                truncation=self.truncation,
                output_dimension=self.output_dimension
            )
        except Exception as e:
            raise RuntimeError(f"VoyageAI API error during embedding: {e}") from e
        
        return np.array(response.embeddings[0], dtype=np.float32)

    def embed_batch(self, texts: List[str], input_type: Literal["query", "document"] = None) -> List["np.ndarray"]:
        """
        Obtain embeddings for a batch of texts synchronously.

        Args:
            texts: List of input strings to embed.
            input_type: Optional tag indicating 'query' or 'document'.
        Returns:
            List of NumPy arrays representing embedding vectors.
        """
        embeddings: List["np.ndarray"] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Check token counts and warn if necessary
            token_counts = self.count_tokens_batch(batch)
            if self.truncation:
                for text, count in zip(batch, token_counts):
                    if count > self._token_limit:
                        warnings.warn(
                            f"Text has {count} tokens which exceeds the model's limit of {self._token_limit}. "
                            "It will be truncated."
                        )
            try:
                response = self._client.embed(
                    texts=batch,
                    model=self.model,
                    input_type=input_type,
                    truncation=self.truncation,
                    output_dimension=self.output_dimension
                )
                embeddings.extend(np.array(emb, dtype=np.float32) for emb in response.embeddings)
            except Exception as e:
                raise RuntimeError(f"VoyageAI API error during embedding: {e}") from e
            
        return embeddings

    async def aembed_batch(self, texts: List[str], input_type: Literal["query", "document"] = None) -> List["np.ndarray"]:
        """
        Obtain embeddings for a batch of texts asynchronously.

        Args:
            texts: List of input strings to embed.
            input_type: Optional tag indicating 'query' or 'document'.
        Returns:
            List of NumPy arrays representing embedding vectors.
        """
        embeddings: List["np.ndarray"] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Check token counts and warn if necessary
            token_counts = self.count_tokens_batch(batch)
            if self.truncation:
                for text, count in zip(batch, token_counts):
                    if count > self._token_limit:
                        warnings.warn(
                            f"Text has {count} tokens which exceeds the model's limit of {self._token_limit}. "
                            "It will be truncated."
                        )
            try:
                response = await self._aclient.embed(
                    texts=batch,
                    model=self.model,
                    input_type=input_type,
                    truncation=self.truncation,
                    output_dimension=self.output_dimension
                )
                embeddings.extend(np.array(emb, dtype=np.float32) for emb in response.embeddings)
            except Exception as e:
                raise RuntimeError(f"VoyageAI API error during embedding: {e}") from e
            
        return embeddings

    @classmethod
    def is_available(cls) -> bool:
        """Check if the voyageai package is available."""
        return importlib.util.find_spec("voyageai") is not None

    @classmethod
    def _import_dependencies(cls) -> None:
        """Lazy import dependencies if they are not already imported.""" 
        if cls.is_available():
            global np, AutoTokenizer, voyageai
            import numpy as np
            from transformers import AutoTokenizer
            import voyageai
        else:
            raise ImportError("One (or more) of the following packages is not available: numpy, AutoTokenizer or voyageai." +
             " Please install it via `pip install chonkie[voyageai]`")

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def get_tokenizer_or_token_counter(self) -> Any:
        """Return a tokenizers tokenizer object of the current model."""
        return self._tokenizer

    def __repr__(self) -> str:
        """Return a string representation of the VoyageEmbeddings object."""
        return f"VoyageEmbeddings(model={self.model})"
