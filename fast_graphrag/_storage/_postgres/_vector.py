"""PostgreSQL vector storage implementation."""

import json
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

import asyncpg
import numpy as np
from numpy.typing import NDArray

from fast_graphrag._storage._base import BaseVectorStorage
from fast_graphrag._types import GTId

from ._config import PostgresStorageConfig


TEmbedding = TypeVar("TEmbedding", bound=NDArray[np.float32])
TId = TypeVar("TId")


@dataclass
class PostgresVectorStorageConfig:
    """PostgreSQL vector storage configuration."""

    schema: str = "public"
    vectors_table: str = "vectors"


@dataclass
class PostgresVectorStorage(BaseVectorStorage[TEmbedding, TId], Generic[TEmbedding, TId]):
    """PostgreSQL vector storage implementation."""

    config: PostgresStorageConfig
    vector_config: PostgresVectorStorageConfig = field(default_factory=PostgresVectorStorageConfig)
    _pool: asyncpg.Pool | None = None
    _in_progress: bool = False

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool."""
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=1,
                max_size=self.config.pool_size,
                ssl=self.config.ssl,
            )
        return self._pool

    async def _insert_start(self):
        """Initialize the connection pool and create tables if they don't exist."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Create vectors table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.vector_config.vectors_table} (
                    id TEXT PRIMARY KEY,
                    vector FLOAT4[] NOT NULL
                );
            """)
        self._in_progress = True

    async def _insert_done(self):
        """Commit the transaction."""
        self._in_progress = False

    async def _query_start(self):
        """Initialize the connection pool."""
        await self._get_pool()
        self._in_progress = True

    async def _query_done(self):
        """Release resources."""
        self._in_progress = False

    async def get(self, vector_id: TId) -> Optional[TEmbedding]:
        """Get a vector from storage."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT vector FROM {self.vector_config.vectors_table}
                WHERE id = $1
                """,
                str(vector_id),
            )
            if row:
                return cast(TEmbedding, np.array(row["vector"], dtype=np.float32))
            return None

    async def set(self, vector_id: TId, vector: TEmbedding) -> None:
        """Set a vector in storage."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.vector_config.vectors_table} (id, vector)
                VALUES ($1, $2)
                ON CONFLICT (id) DO UPDATE
                SET vector = EXCLUDED.vector
                """,
                str(vector_id),
                vector.tolist(),
            )

    async def get_many(self, vector_ids: Iterable[TId]) -> AsyncIterator[Tuple[TId, TEmbedding]]:
        """Get multiple vectors from storage."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, vector FROM {self.vector_config.vectors_table}
                WHERE id = ANY($1::text[])
                """,
                [str(vector_id) for vector_id in vector_ids],
            )
            for row in rows:
                yield cast(TId, row["id"]), cast(TEmbedding, np.array(row["vector"], dtype=np.float32))

    async def delete(self, vector_id: TId) -> None:
        """Delete a vector from storage."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                DELETE FROM {self.vector_config.vectors_table}
                WHERE id = $1
                """,
                str(vector_id),
            )

    async def delete_many(self, vector_ids: Iterable[TId]) -> None:
        """Delete multiple vectors from storage."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                DELETE FROM {self.vector_config.vectors_table}
                WHERE id = ANY($1::text[])
                """,
                [str(vector_id) for vector_id in vector_ids],
            )

    async def clear(self) -> None:
        """Delete all vectors from storage."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self.vector_config.vectors_table}")

    async def get_all(self) -> AsyncIterator[Tuple[TId, TEmbedding]]:
        """Get all vectors from storage."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, vector FROM {self.vector_config.vectors_table}
                """,
            )
            for row in rows:
                yield cast(TId, row["id"]), cast(TEmbedding, np.array(row["vector"], dtype=np.float32)) 