# Embedding Service
# TODO: Implement for runtime embedding (if needed)
# Note: Batch embedding is in scripts/embedder.py

from openai import OpenAI


class EmbeddingService:
    """Service for generating embeddings using OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        # TODO: Implement
        pass
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in batch."""
        # TODO: Implement
        pass
