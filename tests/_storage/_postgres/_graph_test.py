"""Tests for PostgreSQL graph storage implementation."""

import pytest
from typing import AsyncGenerator, Dict, Any, Optional, Iterable, List
from dataclasses import dataclass

from fast_graphrag._storage._postgres import PostgresGraphStorage, PostgresGraphStorageConfig
from fast_graphrag._storage._postgres._config import PostgresStorageConfig
from fast_graphrag._types import BTNode, BTEdge, TIndex

@dataclass
class TestNode(BTNode):
    """Test node class."""
    name: str
    type: str
    data: Dict[str, Any]

@dataclass
class TestEdge(BTEdge):
    """Test edge class."""
    source: str
    target: str
    type: str
    data: Dict[str, Any]

    @staticmethod
    def to_attrs(edge: Optional[Any] = None, edges: Optional[Iterable[Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        return {}

class TestPostgresGraphStorage:
    """Test PostgreSQL graph storage implementation."""

    @pytest.fixture
    async def graph_storage(self, postgres_config: PostgresStorageConfig) -> AsyncGenerator[PostgresGraphStorage[TestNode, TestEdge, str], None]:
        """Create a graph storage instance."""
        # Create graph storage config
        config = PostgresGraphStorageConfig()
        config.host = postgres_config.host
        config.port = postgres_config.port
        config.database = postgres_config.database
        config.user = postgres_config.user
        config.password = postgres_config.password
        config.schema = postgres_config.schema
        config.pool_size = postgres_config.pool_size
        config.ssl = postgres_config.ssl
        config.graph_name = "test_graph"
        config.node_label = "TestNode"
        config.edge_label = "TestEdge"
        
        # Create storage instance
        storage = PostgresGraphStorage[TestNode, TestEdge, str](config=config)
        
        # Initialize storage and AGE
        await storage._insert_start()
        
        try:
            yield storage
        finally:
            await storage._insert_end()
            # Clean up will be handled by the postgres_pool fixture

    @pytest.mark.asyncio
    async def test_node_operations(self, graph_storage: AsyncGenerator[PostgresGraphStorage[TestNode, TestEdge, str], None]):
        """Test node operations."""
        storage = await anext(graph_storage)
        
        # Create test node
        node = TestNode(
            name="test_node",
            type="test",
            data={"key": "value"}
        )
        
        # Test node insertion
        node_index = await storage.upsert_node(node, None)
        assert node_index is not None
        
        # Test node retrieval
        retrieved_node, retrieved_index = await storage.get_node(node)
        assert retrieved_node is not None
        assert retrieved_index == node_index
        assert retrieved_node.name == node.name
        assert retrieved_node.type == node.type
        assert retrieved_node.data == node.data

    @pytest.mark.asyncio
    async def test_edge_operations(self, graph_storage: AsyncGenerator[PostgresGraphStorage[TestNode, TestEdge, str], None]):
        """Test edge operations."""
        storage = await anext(graph_storage)
        
        # Create test nodes
        source_node = TestNode(name="source", type="test", data={})
        target_node = TestNode(name="target", type="test", data={})
        
        # Insert nodes
        source_index = await storage.upsert_node(source_node, None)
        target_index = await storage.upsert_node(target_node, None)
        
        # Create and insert edge
        edge = TestEdge(
            source=source_node.name,
            target=target_node.name,
            type="test_relation",
            data={"weight": 1.0}
        )
        
        edge_index = await storage.upsert_edge(edge, None)
        assert edge_index is not None
        
        # Test edge retrieval
        retrieved_edges = [edge async for edge in storage.get_edges(source_node.name, target_node.name)]
        assert len(retrieved_edges) == 1
        retrieved_edge, retrieved_index = retrieved_edges[0]
        assert retrieved_index == edge_index
        assert retrieved_edge.source == edge.source
        assert retrieved_edge.target == edge.target
        assert retrieved_edge.type == edge.type
        assert retrieved_edge.data == edge.data

    @pytest.mark.asyncio
    async def test_batch_operations(self, graph_storage: AsyncGenerator[PostgresGraphStorage[TestNode, TestEdge, str], None]):
        """Test batch operations."""
        storage = await anext(graph_storage)
        
        # Create test nodes
        nodes = [
            TestNode(name=f"node_{i}", type="test", data={}) 
            for i in range(3)
        ]
        
        # Create test edges
        edges = [
            TestEdge(
                source="node_0",
                target="node_1",
                type="test_relation",
                data={}
            ),
            TestEdge(
                source="node_1",
                target="node_2",
                type="test_relation",
                data={}
            )
        ]
        
        # Insert nodes
        node_indices = []
        for node in nodes:
            idx = await storage.upsert_node(node, None)
            assert idx is not None
            node_indices.append(idx)
        
        # Insert edges in batch
        edge_indices = await storage.insert_edges(edges=edges)
        assert len(edge_indices) == len(edges)

    @pytest.mark.asyncio
    async def test_graph_metrics(self, graph_storage: AsyncGenerator[PostgresGraphStorage[TestNode, TestEdge, str], None]):
        """Test graph metrics."""
        storage = await anext(graph_storage)
        
        # Create test nodes
        nodes = [
            TestNode(name=f"node_{i}", type="test", data={})
            for i in range(3)
        ]
        
        # Insert nodes
        for node in nodes:
            await storage.upsert_node(node, None)
        
        # Create and insert edges
        edges = [
            TestEdge(
                source="node_0",
                target="node_1",
                type="test_relation",
                data={}
            ),
            TestEdge(
                source="node_1",
                target="node_2",
                type="test_relation",
                data={}
            )
        ]
        await storage.insert_edges(edges=edges)
        
        # Test metrics
        node_count = await storage.node_count()
        edge_count = await storage.edge_count()
        
        assert node_count == len(nodes)
        assert edge_count == len(edges) 