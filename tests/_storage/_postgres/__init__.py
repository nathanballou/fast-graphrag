"""PostgreSQL storage test utilities."""

import os
from typing import AsyncGenerator, Any
import pytest
import asyncpg
from dotenv import load_dotenv

from fast_graphrag._storage._postgres._config import PostgresStorageConfig

# Load environment variables
load_dotenv()

@pytest.fixture
def postgres_config() -> PostgresStorageConfig:
    """Get PostgreSQL configuration from environment variables."""
    return PostgresStorageConfig(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        database=os.getenv("PG_DATABASE", "graphrag"),
        user=os.getenv("PG_USER", "fastrag"),
        password=os.getenv("PG_PASSWORD", "fastrag"),
        schema="test",
        ssl=False,
    )

@pytest.fixture
async def postgres_pool(postgres_config: PostgresStorageConfig) -> AsyncGenerator[asyncpg.Pool, None]:
    """Create a connection pool for testing."""
    # Create pool with setup function to load AGE and set search path
    async def setup_connection(conn: asyncpg.Connection) -> None:
        """Set up connection with AGE extension and proper search path."""
        try:
            # Create AGE extension if it doesn't exist
            await conn.execute("CREATE EXTENSION IF NOT EXISTS age CASCADE")
            # Load AGE extension
            await conn.execute("LOAD 'age'")
            # Set search path to include AGE schema first
            await conn.execute("SET search_path = ag_catalog, public")
            # Then add the test schema to the search path
            await conn.execute(f"SET search_path = ag_catalog, {postgres_config.schema}, public")
        except Exception as e:
            print(f"Error setting up connection: {e}")
            raise
    
    pool = await asyncpg.create_pool(
        host=postgres_config.host,
        port=postgres_config.port,
        database=postgres_config.database,
        user=postgres_config.user,
        password=postgres_config.password,
        min_size=1,
        max_size=1,
        ssl=postgres_config.ssl,
        setup=setup_connection,
    )
    
    # Create test schema and initialize
    async with pool.acquire() as conn:
        try:
            # Drop existing schema if it exists
            await conn.execute(f"DROP SCHEMA IF EXISTS {postgres_config.schema} CASCADE")
            await conn.execute(f"CREATE SCHEMA {postgres_config.schema}")
            
            # Create test graph
            await conn.execute("""
            SELECT * FROM ag_catalog.create_graph('test_graph')
            WHERE NOT EXISTS (
                SELECT 1 FROM ag_graph WHERE name = 'test_graph'
            )
            """)
        except Exception as e:
            print(f"Error initializing schema: {e}")
            raise
    
    yield pool
    
    # Cleanup
    async with pool.acquire() as conn:
        await conn.execute(f"DROP SCHEMA IF EXISTS {postgres_config.schema} CASCADE")
    
    await pool.close()

# Mark all tests in this directory as async
def pytest_collection_modifyitems(items: list[Any]) -> None:
    """Mark all tests as async."""
    for item in items:
        item.add_marker(pytest.mark.asyncio) 