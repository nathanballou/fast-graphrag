"""PostgreSQL blob storage implementation."""

import pickle
from dataclasses import dataclass
from typing import Generic

import asyncpg

from fast_graphrag._storage._base import BaseBlobStorage
from fast_graphrag._types import GTBlob

from ._config import PostgresStorageConfig
from ._utils import get_connection_pool


@dataclass
class PostgresBlobStorage(BaseBlobStorage[GTBlob], Generic[GTBlob]):
    """PostgreSQL blob storage implementation."""

    config: PostgresStorageConfig
    _pool: asyncpg.Pool | None = None
    _in_progress: bool = False

    async def _insert_start(self):
        """Initialize the connection pool and create the table if it doesn't exist."""
        if not self._pool:
            self._pool = await get_connection_pool(self.config)
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS blobs (
                        id INTEGER PRIMARY KEY DEFAULT 1,
                        data BYTEA NOT NULL,
                        CHECK (id = 1)
                    );
                """)
        self._in_progress = True

    async def _insert_done(self):
        """Commit the transaction."""
        self._in_progress = False

    async def _query_start(self):
        """Initialize the connection pool."""
        if not self._pool:
            self._pool = await get_connection_pool(self.config)
        self._in_progress = True

    async def _query_done(self):
        """Release resources."""
        self._in_progress = False

    async def get(self) -> GTBlob | None:
        """Get the blob from storage."""
        if not self._pool:
            await self._query_start()
        
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT data FROM blobs WHERE id = 1")
            if row:
                return pickle.loads(row["data"])
            return None

    async def set(self, blob: GTBlob) -> None:
        """Set the blob in storage."""
        if not self._pool:
            await self._insert_start()
        
        assert self._pool is not None
        data = pickle.dumps(blob)
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO blobs (id, data) 
                VALUES (1, $1)
                ON CONFLICT (id) DO UPDATE 
                SET data = EXCLUDED.data
            """, data) 