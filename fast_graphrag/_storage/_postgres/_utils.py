"""PostgreSQL utilities for connection handling."""

from typing import Any, Dict, Optional, Union

import asyncpg

from ._config import PostgresStorageConfig


async def get_connection_pool(config: PostgresStorageConfig) -> asyncpg.Pool:
    """Create a connection pool for PostgreSQL."""
    return await asyncpg.create_pool(
        host=config.host,
        port=config.port,
        database=config.database,
        user=config.user,
        password=config.password,
        min_size=1,
        max_size=config.pool_size,
        ssl=config.ssl,
    )


async def execute_query(
    pool: asyncpg.Pool,
    query: str,
    *args: Any,
    fetch_type: str = "all",
) -> Union[Optional[asyncpg.Record], Optional[list[asyncpg.Record]]]:
    """Execute a query and return results based on fetch_type.
    
    Args:
        pool: The connection pool
        query: The SQL query to execute
        *args: Query parameters
        fetch_type: One of "all", "row", or None
    
    Returns:
        Query results based on fetch_type
    """
    async with pool.acquire() as conn:
        # Set search path for AGE
        await conn.execute("SET search_path = ag_catalog, ag_catalog, public")
        
        if fetch_type == "all":
            return await conn.fetch(query, *args)
        elif fetch_type == "row":
            return await conn.fetchrow(query, *args)
        else:
            await conn.execute(query, *args)
            return None


async def create_schema_if_not_exists(pool: asyncpg.Pool, schema: str) -> None:
    """Create a schema if it doesn't exist."""
    async with pool.acquire() as conn:
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")


async def set_schema(pool: asyncpg.Pool, schema: str) -> None:
    """Set the search path to use the specified schema."""
    async with pool.acquire() as conn:
        await conn.execute(f"SET search_path TO {schema}, public")


async def initialize_age_graph(pool: asyncpg.Pool, graph_name: str) -> None:
    """Initialize AGE extension and create a graph.
    
    Args:
        pool: The connection pool
        graph_name: Name of the graph to create
    """
    async with pool.acquire() as conn:
        # Create the AGE extension if it doesn't exist
        await conn.execute("CREATE EXTENSION IF NOT EXISTS age")
        
        # Set the search path to include the AGE schema
        await conn.execute("SET search_path = ag_catalog, public")
        
        # Create the graph if it doesn't exist
        await conn.execute(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = '{graph_name}') THEN
                PERFORM ag_catalog.create_graph('{graph_name}');
            END IF;
        END
        $$;
        SET search_path = ag_catalog, "{graph_name}", public;
        """) 