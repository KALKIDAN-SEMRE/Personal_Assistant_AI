"""
Embedding service abstraction for generating text embeddings.
Provider-agnostic design supporting multiple embedding backends.
"""
import logging
from typing import List, Optional
import numpy as np
import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Abstracted embedding service for generating text embeddings.
    Supports multiple providers (mock, OpenAI, Ollama, etc.)
    """
    
    def __init__(self):
        self.provider = config.settings.embedding_provider
        self.embedding_dimension = config.settings.embedding_dimension
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding vector for a text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if self.provider == "mock":
            return self._mock_embed(text)
        elif self.provider == "openai":
            return await self._openai_embed(text)
        elif self.provider == "ollama":
            return await self._ollama_embed(text)
        else:
            logger.warning(f"Unknown embedding provider '{self.provider}', using mock")
            return self._mock_embed(text)
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if self.provider == "openai":
            # OpenAI supports batch embedding
            return await self._openai_embed_batch(texts)
        else:
            # For other providers, embed sequentially
            return [await self.embed(text) for text in texts]
    
    def _mock_embed(self, text: str) -> List[float]:
        """
        Mock embedding using simple hash-based approach.
        For development/testing only - not semantically meaningful.
        """
        # Simple deterministic hash-based embedding
        hash_value = hash(text)
        np.random.seed(hash_value % (2**32))
        embedding = np.random.normal(0, 0.1, self.embedding_dimension).tolist()
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (np.array(embedding) / norm).tolist()
        return embedding
    
    async def _openai_embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            from openai import AsyncOpenAI
            
            if not config.settings.openai_api_key:
                logger.error("OpenAI API key not configured for embeddings")
                return self._mock_embed(text)
            
            client = AsyncOpenAI(api_key=config.settings.openai_api_key)
            
            response = await client.embeddings.create(
                model=config.settings.openai_embedding_model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return self._mock_embed(text)
    
    async def _openai_embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch using OpenAI API."""
        try:
            from openai import AsyncOpenAI
            
            if not config.settings.openai_api_key:
                logger.error("OpenAI API key not configured for embeddings")
                return [self._mock_embed(text) for text in texts]
            
            client = AsyncOpenAI(api_key=config.settings.openai_api_key)
            
            response = await client.embeddings.create(
                model=config.settings.openai_embedding_model,
                input=texts
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            return [self._mock_embed(text) for text in texts]
    
    async def _ollama_embed(self, text: str) -> List[float]:
        """
        Generate embedding using Ollama embeddings API.
        Note: Ollama may not have a dedicated embeddings endpoint,
        so this falls back to mock for now.
        """
        # Ollama doesn't have a standard embeddings API endpoint
        # This would need to be implemented based on Ollama's specific API
        logger.warning("Ollama embeddings not fully implemented, using mock")
        return self._mock_embed(text)


# Global embedding service instance
embedding_service = EmbeddingService()
