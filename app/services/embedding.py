"""
Embedding Service
Provides OpenAI embedding functionality for runtime use.

Core logic adapted from scripts/embedder.py for use as a reusable service.
"""

import time
from typing import Optional

import numpy as np
from openai import OpenAI


class EmbeddingService:
    """
    Service for generating embeddings using OpenAI API.
    
    Features:
    - Single text and batch embedding
    - Automatic text truncation for token limits
    - Batch overflow handling (splits large batches)
    - Retry with exponential backoff
    
    Example:
        service = EmbeddingService(api_key="sk-...")
        
        # Single text
        embedding = service.embed_text("労働基準法第一条")
        
        # Batch
        embeddings = service.embed_batch(["text1", "text2", "text3"])
    """
    
    # Default configuration
    DEFAULT_MODEL = "text-embedding-3-large"
    DEFAULT_DIMENSIONS = 3072
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    MAX_TOKENS_PER_TEXT = 6000  # Safe limit for single text
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Initialize EmbeddingService.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name
            dimensions: Output embedding dimensions
            max_retries: Maximum retry attempts for failed requests
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self.max_retries = max_retries
    
    def _truncate_text(self, text: str, max_tokens: int = MAX_TOKENS_PER_TEXT) -> str:
        """
        Truncate text to fit within token limit.
        
        Japanese characters can be 2-3 tokens each.
        Using conservative ratio: 1 char = 3 tokens.
        
        Args:
            text: Input text
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated text (or original if within limit)
        """
        max_chars = max_tokens // 3
        if len(text) <= max_chars:
            return text
        return text[:max_chars]
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Rough estimate of tokens for Japanese/mixed text.
        
        Japanese characters are typically 1-2 tokens each.
        This is a conservative estimate.
        """
        return len(text) // 2
    
    def embed_text(self, text: str, truncate: bool = True) -> list[float]:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            truncate: If True, truncate long texts to fit token limit
            
        Returns:
            Embedding vector as list of floats
        """
        if truncate:
            text = self._truncate_text(text)
        
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            dimensions=self.dimensions
        )
        return response.data[0].embedding
    
    def embed_text_numpy(self, text: str, truncate: bool = True) -> np.ndarray:
        """
        Embed a single text, returning numpy array.
        
        Args:
            text: Text to embed
            truncate: If True, truncate long texts
            
        Returns:
            Embedding as numpy array of shape (dimensions,)
        """
        embedding = self.embed_text(text, truncate=truncate)
        return np.array(embedding, dtype=np.float32)
    
    def embed_batch(
        self,
        texts: list[str],
        truncate: bool = True,
        retry_count: int = 0,
    ) -> list[list[float]]:
        """
        Embed multiple texts in batch.
        
        Handles token overflow by splitting batch or processing individually.
        Includes retry with exponential backoff for transient errors.
        
        Args:
            texts: List of texts to embed
            truncate: If True, truncate long texts
            retry_count: Internal retry counter
            
        Returns:
            List of embedding vectors
        """
        if truncate:
            texts = [self._truncate_text(t) for t in texts]
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions
            )
            return [item.embedding for item in response.data]
        
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a token overflow error
            if "maximum context length" in error_str or "8192 tokens" in error_str:
                if len(texts) > 1:
                    # Split batch in half and try again
                    mid = len(texts) // 2
                    first_half = self.embed_batch(texts[:mid], truncate=False)
                    second_half = self.embed_batch(texts[mid:], truncate=False)
                    return first_half + second_half
                else:
                    # Single text too long - truncate more aggressively
                    truncated = self._truncate_text(texts[0], max_tokens=4000)
                    return [self.embed_text(truncated, truncate=False)]
            
            # For other errors, retry with backoff
            if retry_count < self.max_retries:
                delay = self.RETRY_DELAY * (retry_count + 1)
                time.sleep(delay)
                return self.embed_batch(texts, truncate=False, retry_count=retry_count + 1)
            else:
                raise RuntimeError(f"Failed after {self.max_retries} retries: {e}")
    
    def embed_batch_numpy(
        self,
        texts: list[str],
        truncate: bool = True,
    ) -> np.ndarray:
        """
        Embed multiple texts, returning numpy array.
        
        Args:
            texts: List of texts to embed
            truncate: If True, truncate long texts
            
        Returns:
            Numpy array of shape (n_texts, dimensions)
        """
        embeddings = self.embed_batch(texts, truncate=truncate)
        return np.array(embeddings, dtype=np.float32)
    
    def process_in_batches(
        self,
        texts: list[str],
        batch_size: int = 100,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Process large list of texts in batches.
        
        Useful for embedding many texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            show_progress: If True, print progress
            
        Returns:
            Numpy array of shape (n_texts, dimensions)
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embed_batch_numpy(batch_texts)
            all_embeddings.append(batch_embeddings)
            
            if show_progress:
                batch_num = i // batch_size + 1
                print(f"Processed batch {batch_num}/{total_batches}")
        
        return np.vstack(all_embeddings)

