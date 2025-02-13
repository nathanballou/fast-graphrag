"""PostgreSQL key-value storage implementation."""

from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar, cast

import asyncpg

from fast_graphrag._storage._base import BaseIndexedKeyValueStorage
from fast_graphrag._types import GTKey, GTValue
from fast_graphrag._utils import logger

from ._config import PostgresStorageConfig
from ._utils import execute_query, get_connection_pool


@dataclass
class PostgresIndexedKeyValueStorage(BaseIndexedKeyValueStorage[GTKey, GTValue]):
    """PostgreSQL key-value storage implementation."""

    config: PostgresStorageConfig
    _pool: Optional[asyncpg.Pool] = None
    _in_progress: Optional[bool] = None

    async def _insert_start(self) -> None:
        """Initialize the connection pool and create the table if it doesn't exist."""
        if not self._pool:
            self._pool = await get_connection_pool(self.config)
        
        # Create table if it doesn't exist
        await execute_query(
            self._pool,
            f"""
            CREATE TABLE IF NOT EXISTS {self.config.schema}.key_values (
                id SERIAL PRIMARY KEY,
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                idx INTEGER NOT NULL GENERATED ALWAYS AS IDENTITY,
                value JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(namespace, key)
            );
            CREATE INDEX IF NOT EXISTS idx_key_values_namespace_key ON {self.config.schema}.key_values(namespace, key);
            CREATE INDEX IF NOT EXISTS idx_key_values_namespace_idx ON {self.config.schema}.key_values(namespace, idx);
            """
        )
        self._in_progress = True

    async def _insert_end(self) -> None:
        """Clean up after inserting."""
        self._in_progress = False

    async def _query_start(self) -> None:
        """Initialize the connection pool for querying."""
        if not self._pool:
            self._pool = await get_connection_pool(self.config)
        self._in_progress = True

    async def _query_end(self) -> None:
        """Clean up after querying."""
        self._in_progress = False

    async def size(self) -> int:
        """Get the number of key-value pairs."""
        assert self._pool is not None, "Connection pool not initialized"
        assert self.namespace is not None, "Namespace not set"
        
        result = await execute_query(
            self._pool,
            f"""
            SELECT COUNT(*) FROM {self.config.schema}.key_values
            WHERE namespace = $1
            """,
            self.namespace.name
        )
        return result[0][0]

    async def get(self, keys: Iterable[GTKey]) -> Iterable[Optional[GTValue]]:
        """Get values for the given keys."""
        assert self._pool is not None, "Connection pool not initialized"
        assert self.namespace is not None, "Namespace not set"
        
        keys = list(keys)
        if not keys:
            return []
        
        result = await execute_query(
            self._pool,
            f"""
            SELECT key, value FROM {self.config.schema}.key_values
            WHERE namespace = $1 AND key = ANY($2)
            """,
            self.namespace.name,
            keys
        )
        
        # Create a mapping of keys to values
        value_map = {row[0]: row[1] for row in result}
        
        # Return values in the same order as keys
        return [value_map.get(key) for key in keys]

    async def get_by_index(self, indices: Iterable[int]) -> Iterable[Optional[GTValue]]:
        """Get values for the given indices."""
        assert self._pool is not None, "Connection pool not initialized"
        assert self.namespace is not None, "Namespace not set"
        
        indices = list(indices)
        if not indices:
            return []
        
        result = await execute_query(
            self._pool,
            f"""
            SELECT idx, value FROM {self.config.schema}.key_values
            WHERE namespace = $1 AND idx = ANY($2)
            """,
            self.namespace.name,
            indices
        )
        
        # Create a mapping of indices to values
        value_map = {row[0]: row[1] for row in result}
        
        # Return values in the same order as indices
        return [value_map.get(idx) for idx in indices]

    async def get_index(self, keys: Iterable[GTKey]) -> Iterable[Optional[int]]:
        """Get indices for the given keys."""
        assert self._pool is not None, "Connection pool not initialized"
        assert self.namespace is not None, "Namespace not set"
        
        keys = list(keys)
        if not keys:
            return []
        
        result = await execute_query(
            self._pool,
            f"""
            SELECT key, idx FROM {self.config.schema}.key_values
            WHERE namespace = $1 AND key = ANY($2)
            """,
            self.namespace.name,
            keys
        )
        
        # Create a mapping of keys to indices
        index_map = {row[0]: row[1] for row in result}
        
        # Return indices in the same order as keys
        return [index_map.get(key) for key in keys]

    async def upsert(self, keys: Iterable[GTKey], values: Iterable[GTValue]) -> None:
        """Upsert key-value pairs."""
        assert self._pool is not None, "Connection pool not initialized"
        assert self.namespace is not None, "Namespace not set"
        
        keys = list(keys)
        values = list(values)
        if not keys or not values:
            return
        
        await execute_query(
            self._pool,
            f"""
            INSERT INTO {self.config.schema}.key_values (namespace, key, value)
            VALUES ($1, unnest($2::text[]), unnest($3::jsonb[]))
            ON CONFLICT (namespace, key)
            DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            self.namespace.name,
            keys,
            [cast(Dict[str, Any], value) for value in values]
        )

    async def upsert_by_index(self, indices: Iterable[int], values: Iterable[GTValue]) -> None:
        """Upsert values at the given indices."""
        assert self._pool is not None, "Connection pool not initialized"
        assert self.namespace is not None, "Namespace not set"
        
        indices = list(indices)
        values = list(values)
        if not indices or not values:
            return
        
        await execute_query(
            self._pool,
            f"""
            UPDATE {self.config.schema}.key_values
            SET value = data_table.value,
                updated_at = CURRENT_TIMESTAMP
            FROM (
                SELECT unnest($2::int[]) as idx,
                       unnest($3::jsonb[]) as value
            ) as data_table
            WHERE namespace = $1
            AND key_values.idx = data_table.idx
            """,
            self.namespace.name,
            indices,
            [cast(Dict[str, Any], value) for value in values]
        )

    async def delete(self, keys: Iterable[GTKey]) -> None:
        """Delete key-value pairs."""
        assert self._pool is not None, "Connection pool not initialized"
        assert self.namespace is not None, "Namespace not set"
        
        keys = list(keys)
        if not keys:
            return
        
        await execute_query(
            self._pool,
            f"""
            DELETE FROM {self.config.schema}.key_values
            WHERE namespace = $1 AND key = ANY($2)
            """,
            self.namespace.name,
            keys
        )

    async def delete_by_index(self, indices: Iterable[int]) -> None:
        """Delete values at the given indices."""
        assert self._pool is not None, "Connection pool not initialized"
        assert self.namespace is not None, "Namespace not set"
        
        indices = list(indices)
        if not indices:
            return
        
        await execute_query(
            self._pool,
            f"""
            DELETE FROM {self.config.schema}.key_values
            WHERE namespace = $1 AND idx = ANY($2)
            """,
            self.namespace.name,
            indices
        )

    async def mask_new(self, keys: Iterable[GTKey]) -> Iterable[bool]:
        """Check which keys are new (not in storage)."""
        assert self._pool is not None, "Connection pool not initialized"
        assert self.namespace is not None, "Namespace not set"
        
        keys = list(keys)
        if not keys:
            return []
        
        result = await execute_query(
            self._pool,
            f"""
            SELECT key FROM {self.config.schema}.key_values
            WHERE namespace = $1 AND key = ANY($2)
            """,
            self.namespace.name,
            keys
        )
        
        # Create a set of existing keys
        existing_keys = {row[0] for row in result}
        
        # Return True for keys that don't exist
        return [key not in existing_keys for key in keys] 