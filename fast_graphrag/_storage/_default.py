"""Default storage implementations for fast-graphrag."""

from dataclasses import dataclass
from typing import Any, Dict, Generic, Literal, Optional, Type, TypeVar, Union, cast

from fast_graphrag._storage._base import (
    BaseBlobStorage,
    BaseGraphStorage,
    BaseIndexedKeyValueStorage,
    BaseVectorStorage,
)
from fast_graphrag._storage._blob_pickle import PickleBlobStorage
from fast_graphrag._storage._gdb_igraph import IGraphStorage, IGraphStorageConfig
from fast_graphrag._storage._ikv_pickle import PickleIndexedKeyValueStorage
from fast_graphrag._storage._postgres import (
    PostgresBlobStorage,
    PostgresGraphStorage,
    PostgresGraphStorageConfig,
    PostgresIndexedKeyValueStorage,
    PostgresStorageConfig,
    PostgresVectorStorage,
    PostgresVectorStorageConfig,
)
from fast_graphrag._storage._vdb_hnswlib import HNSWVectorStorage, HNSWVectorStorageConfig
from fast_graphrag._types import BTNode, BTEdge

# Type variables for storage classes
TNode = TypeVar("TNode", bound=BTNode)
TEdge = TypeVar("TEdge", bound=BTEdge)
TId = TypeVar("TId")
TBlob = TypeVar("TBlob")
TKey = TypeVar("TKey")
TValue = TypeVar("TValue")
TEmbedding = TypeVar("TEmbedding")


@dataclass
class StorageConfig:
    """Configuration for storage backends."""
    backend: Literal["file", "postgres"] = "file"
    file: Optional[Dict[str, Any]] = None
    postgres: Optional[PostgresStorageConfig] = None
    vector: Optional[Union[HNSWVectorStorageConfig, PostgresVectorStorageConfig]] = None
    graph: Optional[Union[IGraphStorageConfig[BTNode, BTEdge], PostgresGraphStorageConfig]] = None


def get_storage_class(
    storage_type: Literal["blob", "ikv", "vector", "graph"],
    backend: Literal["file", "postgres"],
) -> Union[
    Type[BaseBlobStorage[Any]],
    Type[BaseIndexedKeyValueStorage[Any, Any]],
    Type[BaseVectorStorage[Any, Any]],
    Type[BaseGraphStorage[BTNode, BTEdge, Any]],
]:
    """Get the appropriate storage class based on type and backend."""
    storage_classes = {
        "file": {
            "blob": PickleBlobStorage,
            "ikv": PickleIndexedKeyValueStorage,
            "vector": HNSWVectorStorage,
            "graph": IGraphStorage,
        },
        "postgres": {
            "blob": PostgresBlobStorage,
            "ikv": PostgresIndexedKeyValueStorage,
            "vector": PostgresVectorStorage,
            "graph": PostgresGraphStorage,
        },
    }
    return storage_classes[backend][storage_type]


class DefaultBlobStorage(Generic[TBlob]):
    def __new__(cls, config: StorageConfig) -> BaseBlobStorage[TBlob]:
        storage_class = cast(Type[BaseBlobStorage[TBlob]], get_storage_class("blob", config.backend))
        storage_config = config.postgres if config.backend == "postgres" else config.file
        return storage_class(config=storage_config)


class DefaultIndexedKeyValueStorage(Generic[TKey, TValue]):
    def __new__(cls, config: StorageConfig) -> BaseIndexedKeyValueStorage[TKey, TValue]:
        storage_class = cast(Type[BaseIndexedKeyValueStorage[TKey, TValue]], get_storage_class("ikv", config.backend))
        storage_config = config.postgres if config.backend == "postgres" else config.file
        return storage_class(config=storage_config)


class DefaultVectorStorage(Generic[TId, TEmbedding]):
    def __new__(cls, config: StorageConfig) -> BaseVectorStorage[TId, TEmbedding]:
        storage_class = cast(Type[BaseVectorStorage[TId, TEmbedding]], get_storage_class("vector", config.backend))
        storage_config = config.postgres if config.backend == "postgres" else config.file
        if config.vector:
            if isinstance(config.vector, PostgresVectorStorageConfig):
                storage_config = PostgresVectorStorageConfig(**vars(config.vector))
            else:
                storage_config = config.vector
        return storage_class(config=storage_config)


class DefaultGraphStorage(Generic[TNode, TEdge, TId]):
    def __new__(cls, config: StorageConfig) -> BaseGraphStorage[TNode, TEdge, TId]:
        storage_class = cast(Type[BaseGraphStorage[TNode, TEdge, TId]], get_storage_class("graph", config.backend))
        if config.backend == "postgres":
            if config.graph and isinstance(config.graph, PostgresGraphStorageConfig):
                storage_config = PostgresGraphStorageConfig(**vars(config.graph))
            else:
                storage_config = config.postgres
        else:  # file backend
            storage_config = config.graph
            if not storage_config or not isinstance(storage_config, IGraphStorageConfig):
                raise ValueError("IGraphStorageConfig is required for file storage")
        return storage_class(config=storage_config)


__all__ = [
    "StorageConfig",
    "DefaultBlobStorage",
    "DefaultIndexedKeyValueStorage",
    "DefaultVectorStorage",
    "DefaultGraphStorage",
]
