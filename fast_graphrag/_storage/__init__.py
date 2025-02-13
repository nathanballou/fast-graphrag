"""Storage module for GraphRAG."""

from dataclasses import dataclass
from typing import Any, Dict, Generic, Type, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from ._base import BaseBlobStorage, BaseGraphStorage, BaseIndexedKeyValueStorage, BaseVectorStorage
from fast_graphrag._types import BTNode, BTEdge
from ._default import (
    DefaultGraphStorage, DefaultVectorStorage, DefaultIndexedKeyValueStorage,
    StorageConfig, IGraphStorageConfig, HNSWVectorStorageConfig
)
from ._postgres import (
    PostgresGraphStorage, PostgresVectorStorage, PostgresIndexedKeyValueStorage,
    PostgresStorageConfig, PostgresGraphStorageConfig, PostgresVectorStorageConfig
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
    storage_options: Dict[str, Any],
    embedding_dim: int,
    entity_cls: Type[TEntity],
    relation_cls: Type[TRelation],
) -> StorageBackend[TEntity, TRelation, Any, Any, NDArray[np.float32], Any, Any]:
    """Create storage instances based on configuration.
    
    Args:
        storage_type: Type of storage backend ("file" or "postgres")
        storage_options: Configuration options for the storage backend
        embedding_dim: Dimension of embeddings for vector storage
        entity_cls: Class type for entities
        relation_cls: Class type for relations
        
    Returns:
        StorageBackend: Container with initialized storage instances
    """
    if storage_type == "postgres":
        # Extract graph-specific options
        graph_options = {
            "graph_name": storage_options.pop("graph_name", None),
            "node_label": storage_options.pop("node_label", "Node"),
            "edge_label": storage_options.pop("edge_label", "Edge")
        }
        
        # Extract vector-specific options
        vector_options = {
            "vectors_table": storage_options.pop("vectors_table", "vectors")
        }
        
        # Create base PostgreSQL config
        config = PostgresStorageConfig(**storage_options)
        
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
            graph_name=graph_options["graph_name"] or config.schema,
            node_label=graph_options["node_label"],
            edge_label=graph_options["edge_label"]
        )
        
        graph_storage = PostgresGraphStorage[TEntity, TRelation, Any](config=graph_config)
        
        # Create vector storage with config
        vector_config = PostgresVectorStorageConfig(
            schema=config.schema,
            vectors_table=vector_options["vectors_table"]
        )
        
        entity_storage = PostgresVectorStorage[Any, NDArray[np.float32]](
            config=config,
            vector_config=vector_config
        )
        entity_storage.embedding_dim = embedding_dim
        
        # Create chunk storage with config
        chunk_storage = PostgresIndexedKeyValueStorage[Any, Any](config=config)
        
        return StorageBackend(
            graph_storage=graph_storage,
            entity_storage=entity_storage,
            chunk_storage=chunk_storage
        )
    else:  # file backend
        # Create graph config first
        graph_config = IGraphStorageConfig(
            node_cls=cast(Type[BTNode], entity_cls),
            edge_cls=cast(Type[BTEdge], relation_cls)
        )
        
        # Get vector config from options or create default
        vector_config = storage_options.get("vector") or HNSWVectorStorageConfig(
            ef_construction=128,
            M=64,
            ef_search=96,
            num_threads=-1
        )
        
        # Create storage config with all required fields
        config = StorageConfig(
            backend="file",
            file=storage_options,  # Pass the entire storage_options dict
            postgres=None,
            graph=graph_config,
            vector=vector_config
        )
        
        # Create storage instances
        graph_storage = DefaultGraphStorage[TEntity, TRelation, Any](config=config)
        entity_storage = DefaultVectorStorage[Any, NDArray[np.float32]](config=config)
        entity_storage.embedding_dim = embedding_dim
        chunk_storage = DefaultIndexedKeyValueStorage[Any, Any](config=config)
        
        return StorageBackend(
            graph_storage=graph_storage,
            entity_storage=entity_storage,
            chunk_storage=chunk_storage
        )

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
