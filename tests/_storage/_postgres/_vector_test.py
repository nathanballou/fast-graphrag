"""Test PostgreSQL vector storage implementation."""

from typing import Dict, AsyncGenerator
import pytest
import numpy as np
from numpy.typing import NDArray

from fast_graphrag._storage._postgres._config import PostgresStorageConfig
from fast_graphrag._storage._postgres._vector import PostgresVectorStorage, PostgresVectorStorageConfig

class TestPostgresVectorStorage:
    """Test PostgreSQL vector storage implementation."""

    @pytest.fixture
    async def vector_storage(self, postgres_config: PostgresStorageConfig) -> AsyncGenerator[PostgresVectorStorage[NDArray[np.float32], str], None]:
        """Create a vector storage instance."""
        config = PostgresVectorStorageConfig(
            schema=postgres_config.schema,
            vectors_table="test_vectors"
        )
        storage = PostgresVectorStorage[NDArray[np.float32], str](
            config=postgres_config,
            vector_config=config
        )
        # Initialize storage
        await storage.insert_start()
        try:
            yield storage
        finally:
            await storage.insert_done()

    @pytest.mark.asyncio
    async def test_insert_and_get(self, vector_storage: AsyncGenerator[PostgresVectorStorage[NDArray[np.float32], str], None]):
        """Test inserting and retrieving vectors."""
        storage = await anext(vector_storage)
        # Test data
        vector_id = "test1"
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Insert vector
        await storage.set(vector_id, vector)

        # Retrieve vector
        result = await storage.get(vector_id)
        assert result is not None
        np.testing.assert_array_almost_equal(result, vector)

    @pytest.mark.asyncio
    async def test_get_many(self, vector_storage: AsyncGenerator[PostgresVectorStorage[NDArray[np.float32], str], None]):
        """Test retrieving multiple vectors."""
        storage = await anext(vector_storage)
        # Test data
        vectors: Dict[str, NDArray[np.float32]] = {
            "test1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "test2": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            "test3": np.array([7.0, 8.0, 9.0], dtype=np.float32)
        }

        # Insert vectors
        for vector_id, vector in vectors.items():
            await storage.set(vector_id, vector)

        # Retrieve vectors
        async for vector_id, vector in storage.get_many(vectors.keys()):
            np.testing.assert_array_almost_equal(vector, vectors[vector_id])

    @pytest.mark.asyncio
    async def test_delete(self, vector_storage: AsyncGenerator[PostgresVectorStorage[NDArray[np.float32], str], None]):
        """Test deleting vectors."""
        storage = await anext(vector_storage)
        # Test data
        vector_id = "test1"
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Insert and then delete
        await storage.set(vector_id, vector)
        await storage.delete(vector_id)

        # Verify deletion
        result = await storage.get(vector_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_many(self, vector_storage: AsyncGenerator[PostgresVectorStorage[NDArray[np.float32], str], None]):
        """Test deleting multiple vectors."""
        storage = await anext(vector_storage)
        # Test data
        vectors: Dict[str, NDArray[np.float32]] = {
            "test1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "test2": np.array([4.0, 5.0, 6.0], dtype=np.float32)
        }

        # Insert vectors
        for vector_id, vector in vectors.items():
            await storage.set(vector_id, vector)

        # Delete vectors
        await storage.delete_many(vectors.keys())

        # Verify deletion
        async for vector_id, vector in storage.get_many(vectors.keys()):
            assert False, f"Vector {vector_id} should have been deleted"

    @pytest.mark.asyncio
    async def test_clear(self, vector_storage: AsyncGenerator[PostgresVectorStorage[NDArray[np.float32], str], None]):
        """Test clearing all vectors."""
        storage = await anext(vector_storage)
        # Test data
        vectors: Dict[str, NDArray[np.float32]] = {
            "test1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "test2": np.array([4.0, 5.0, 6.0], dtype=np.float32)
        }

        # Insert vectors
        for vector_id, vector in vectors.items():
            await storage.set(vector_id, vector)

        # Clear all vectors
        await storage.clear()

        # Verify all vectors are deleted
        async for vector_id, vector in storage.get_many(vectors.keys()):
            assert False, "All vectors should have been deleted"

    @pytest.mark.asyncio
    async def test_get_all(self, vector_storage: AsyncGenerator[PostgresVectorStorage[NDArray[np.float32], str], None]):
        """Test retrieving all vectors."""
        storage = await anext(vector_storage)
        # Test data
        vectors: Dict[str, NDArray[np.float32]] = {
            "test1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "test2": np.array([4.0, 5.0, 6.0], dtype=np.float32)
        }

        # Insert vectors
        for vector_id, vector in vectors.items():
            await storage.set(vector_id, vector)

        # Get all vectors and verify
        found_vectors: Dict[str, NDArray[np.float32]] = {}
        async for vector_id, vector in storage.get_all():
            found_vectors[vector_id] = vector

        assert len(found_vectors) == len(vectors)
        for vector_id, vector in vectors.items():
            np.testing.assert_array_almost_equal(found_vectors[vector_id], vector) 