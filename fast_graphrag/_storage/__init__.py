"""Storage module for GraphRAG."""

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Type, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from fast_graphrag._types import BTEdge, BTNode

from ._base import (
    BaseBlobStorage,
    BaseGraphStorage,
    BaseIndexedKeyValueStorage,
    BaseVectorStorage,
)
from ._default import (
    DefaultGraphStorage,
    DefaultIndexedKeyValueStorage,
    DefaultVectorStorage,
    HNSWVectorStorageConfig,
    IGraphStorageConfig,
    StorageConfig,
)
from ._postgres import (
    PostgresGraphStorage,
    PostgresGraphStorageConfig,
    PostgresIndexedKeyValueStorage,
    PostgresStorageConfig,
    PostgresVectorStorage,
    PostgresVectorStorageConfig,
)

# Type variables for storage classes
TEntity = TypeVar("TEntity", bound=BTNode)
TRelation = TypeVar("TRelation", bound=BTEdge)
TId = TypeVar("TId")
TIndex = TypeVar("TIndex")
TEmbedding = TypeVar("TEmbedding", bound=NDArray[np.float32])
THash = TypeVar("THash")
TChunk = TypeVar("TChunk")


@dataclass
class StorageBackend(Generic[TEntity, TRelation, TId, TIndex, TEmbedding, THash, TChunk]):
    """Container for storage instances."""
    graph_storage: BaseGraphStorage[TEntity, TRelation, TId]
    entity_storage: BaseVectorStorage[TIndex, TEmbedding]
    chunk_storage: BaseIndexedKeyValueStorage[THash, TChunk]


def create_storage_backend(
    storage_type: str,
    working_dir: str,
    pg_host: str,
    pg_port: int,
    pg_database: str,
    pg_user: str,
    pg_password: str,
    embedding_dim: int,
    entity_cls: Type[TEntity],
    relation_cls: Type[TRelation],
    graph_config: Optional[IGraphStorageConfig] = None
) -> StorageBackend[TEntity, TRelation, Any, Any, NDArray[np.float32], Any, Any]:
    """Create storage instances based on configuration.

    Args:
        storage_type: Type of storage backend ("file" or "postgres")
        working_dir: Working directory for file-based storage
        pg_host: PostgreSQL host
        pg_port: PostgreSQL port
        pg_database: PostgreSQL database
        pg_user: PostgreSQL user
        pg_password: PostgreSQL password
        embedding_dim: Dimension of embeddings for vector storage
        entity_cls: Class type for entities
        relation_cls: Class type for relations
        graph_config: Optional graph configuration for file storage

    Returns:
        StorageBackend: Container with initialized storage instances
    """
    if storage_type == "postgres":
        # Create base PostgreSQL config
        config = PostgresStorageConfig(
            host=pg_host,
            port=pg_port,
            database=pg_database,
            user=pg_user,
            password=pg_password,
            schema="public",  # Default schema
            ssl=False,  # Default SSL
            pool_size=5  # Default pool size
        )

        # Create graph storage with config
        graph_config = PostgresGraphStorageConfig(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password,
            schema=config.schema,
            ssl=config.ssl,
            pool_size=config.pool_size,
            graph_name=config.schema,
            node_label="Node",
            edge_label="Edge"
        )

        graph_storage = PostgresGraphStorage[TEntity, TRelation, Any](config=graph_config)

        # Create vector storage with config
        vector_config = PostgresVectorStorageConfig(
            schema=config.schema,
            vectors_table="vectors"
        )

        entity_storage = PostgresVectorStorage[Any, NDArray[np.float32]](
            config=config,
            vector_config=vector_config
        )  # type: ignore
        entity_storage.embedding_dim = embedding_dim

        # Create chunk storage with config
        chunk_storage = PostgresIndexedKeyValueStorage[Any, Any](config=config)

        return StorageBackend(
            graph_storage=graph_storage,
            entity_storage=entity_storage,
            chunk_storage=chunk_storage
        )
    elif storage_type == "file":  # file backend
        if graph_config is None:
            # Create a default IGraphStorageConfig if not provided
            graph_config = IGraphStorageConfig(
                node_cls=cast(Type[BTNode], entity_cls),
                edge_cls=cast(Type[BTEdge], relation_cls)
            )
        config = StorageConfig(backend="file", graph=graph_config)
        graph_storage = DefaultGraphStorage[TEntity, TRelation, Any](config=config)
        entity_storage = DefaultVectorStorage[Any, NDArray[np.float32]](config=config)
        entity_storage.embedding_dim = embedding_dim
        chunk_storage = DefaultIndexedKeyValueStorage[Any, Any](config=config)

        return StorageBackend(
            graph_storage=graph_storage,
            entity_storage=entity_storage,
            chunk_storage=chunk_storage
        )
    else:
        raise ValueError(f"Invalid storage_type: {storage_type}. Must be 'file' or 'postgres'.")

__all__ = [
    "BaseBlobStorage",
    "BaseGraphStorage",
    "BaseIndexedKeyValueStorage", 
    "BaseVectorStorage",
    "DefaultGraphStorage",
    "DefaultVectorStorage",
    "DefaultIndexedKeyValueStorage",
    "PostgresGraphStorage",
    "PostgresVectorStorage", 
    "PostgresIndexedKeyValueStorage",
    "StorageBackend",
    "create_storage_backend"
]
