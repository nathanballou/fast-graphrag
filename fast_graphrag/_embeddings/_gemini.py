"""Gemini embeddings service implementation."""

import os
import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from numpy.typing import NDArray
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from fast_graphrag._llm._base import BaseEmbeddingService
from fast_graphrag._exceptions import EmbeddingError
from fast_graphrag._utils import logger


@dataclass
class GeminiEmbedderConfig:
    """Configuration for the Gemini embedder."""
    model: str = field(default="models/embedding-001")
    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    batch_size: int = field(default=10)
    delay_between_batches: float = field(default=0.1)


class GeminiEmbedder(BaseEmbeddingService):
    """Gemini embedder implementation."""
    
    def __init__(self, config: GeminiEmbedderConfig):
        """Initialize the Gemini embedder."""
        self.config = config
        genai.configure(api_key=config.api_key)
        self._embed_fn = lambda text: genai.embed_content(
            model=self.config.model,
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_single_embedding(self, text: str) -> NDArray[np.float32]:
        """Get embedding for a single text with retry logic."""
        try:
            # Use the embedding method
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._embed_fn(text)
            )
            
            # Extract embedding values from response
            if not result or not hasattr(result, 'embedding'):
                raise ValueError(f"Invalid response format from API: {result}")
                
            # Convert embedding values to numpy array
            embedding_values = result.embedding
            if not embedding_values:
                raise ValueError("Empty embedding returned from API")
                
            return np.array(embedding_values, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts."""
        embeddings = []
        for batch in await self._batch_texts(texts):
            try:
                results = await asyncio.gather(
                    *[self._get_embedding(text) for text in batch]
                )
                embeddings.extend(results)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                raise
        if not embeddings:
            raise ValueError("Failed to get any embeddings")
        return embeddings

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._embed_fn(text)
            )
            if isinstance(result, dict) and 'embedding' in result:
                return result['embedding']
            logger.error(f"Invalid response format from API: {result}")
            raise ValueError(f"Invalid response format from API: {result}")
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    async def encode(
        self, texts: List[str], show_progress: bool = False
    ) -> List[NDArray[np.float32]]:
        """Encode a list of texts into embeddings."""
        try:
            return await self.get_embeddings(texts)
        except Exception as e:
            raise EmbeddingError(f"Failed to get embeddings: {str(e)}") from e

    async def _batch_texts(self, texts: list[str]) -> list[list[str]]:
        """Split texts into batches."""
        if not texts:
            return []
        
        batches = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batches.append(batch)
            if i + self.config.batch_size < len(texts):
                await asyncio.sleep(self.config.delay_between_batches)
        return batches 