"""Top-level package for GraphRAG."""

__all__ = ["GraphRAG", "QueryParam"]

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os

from fast_graphrag._llm import DefaultEmbeddingService, DefaultLLMService
from fast_graphrag._llm._base import BaseEmbeddingService
from fast_graphrag._llm._llm_openai import BaseLLMService
from fast_graphrag._policies._graph_upsert import (
    DefaultGraphUpsertPolicy,
    EdgeUpsertPolicy_UpsertIfValidNodes,
    NodeUpsertPolicy_SummarizeDescription,
)
from fast_graphrag._policies._ranking import RankingPolicy_TopK, RankingPolicy_WithThreshold
from fast_graphrag._services import (
    DefaultChunkingService,
    DefaultInformationExtractionService,
    DefaultStateManagerService,
)
from fast_graphrag._storage import create_storage_backend
from fast_graphrag._storage._namespace import Workspace
from fast_graphrag._types import TChunk, TEmbedding, TEntity, THash, TId, TRelation

from ._graphrag import BaseGraphRAG, QueryParam


@dataclass
class GraphRAG(BaseGraphRAG[TEmbedding, THash, TChunk, TEntity, TRelation, TId]):
    """A class representing a Graph-based Retrieval-Augmented Generation system."""

    working_dir: str = field()
    domain: str = field()
    example_queries: str = field()
    entity_types: List[str] = field()
    n_checkpoints: int = field(default=5)
    storage_type: str = field(default="file")
    storage_options: Optional[Dict[str, Any]] = field(default=None)
    llm_service: BaseLLMService = field(default_factory=DefaultLLMService)
    embedding_service: BaseEmbeddingService = field(default_factory=DefaultEmbeddingService)

    def __post_init__(self):
        """Initialize the GraphRAG class with sensible defaults."""
        # Set up default chunking service
        self.chunking_service = DefaultChunkingService()
        
        # Set up default information extraction service
        self.information_extraction_service = DefaultInformationExtractionService(
            graph_upsert=DefaultGraphUpsertPolicy(
                config=NodeUpsertPolicy_SummarizeDescription.Config(),
                nodes_upsert_cls=NodeUpsertPolicy_SummarizeDescription,
                edges_upsert_cls=EdgeUpsertPolicy_UpsertIfValidNodes,
            )
        )

        # Set up storage options
        if self.storage_options is None:
            self.storage_options = {}
            
        # Configure storage based on type
        storage_options = {}
        if self.storage_type == "file":
            storage_options["working_dir"] = self.working_dir
        else:  # postgres
            storage_options.update({
                "host": os.getenv("PG_HOST", "localhost"),
                "port": int(os.getenv("PG_PORT", "5432")),
                "database": os.getenv("PG_DATABASE", "fastrag"),
                "user": os.getenv("PG_USER", "fastrag"),
                "password": os.getenv("PG_PASSWORD", "fastrag"),
                "schema": "graphrag"
            })

        # Create storage instances using factory
        storage = create_storage_backend(
            storage_type=self.storage_type,
            storage_options=storage_options,
            embedding_dim=self.embedding_service.embedding_dim,
            entity_cls=TEntity,
            relation_cls=TRelation
        )

        # Set up default ranking policies
        entity_ranking_policy = RankingPolicy_WithThreshold(
            RankingPolicy_WithThreshold.Config(threshold=0.005)
        )
        relation_ranking_policy = RankingPolicy_TopK(
            RankingPolicy_TopK.Config(top_k=64)
        )
        chunk_ranking_policy = RankingPolicy_TopK(
            RankingPolicy_TopK.Config(top_k=8)
        )

        # Initialize state manager
        self.state_manager = DefaultStateManagerService(
            workspace=Workspace.new(self.working_dir, keep_n=self.n_checkpoints),
            embedding_service=self.embedding_service,
            graph_storage=storage.graph_storage,
            entity_storage=storage.entity_storage,
            chunk_storage=storage.chunk_storage,
            entity_ranking_policy=entity_ranking_policy,
            relation_ranking_policy=relation_ranking_policy,
            chunk_ranking_policy=chunk_ranking_policy,
            node_upsert_policy=NodeUpsertPolicy_SummarizeDescription(),
            edge_upsert_policy=EdgeUpsertPolicy_UpsertIfValidNodes()
        )
