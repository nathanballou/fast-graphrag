"""PostgreSQL graph storage implementation."""

from typing import Optional, TypeVar, Union, Tuple, Iterable, List, cast, Any, AsyncGenerator, Dict, Generic
import asyncpg
from scipy.sparse import csr_matrix

from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._types import BTNode, BTEdge, TIndex
from fast_graphrag._utils import logger
from fast_graphrag._storage._postgres._config import PostgresGraphStorageConfig

from ._config import PostgresStorageConfig, PostgresGraphStorageConfig
from ._utils import execute_query, get_connection_pool, initialize_age_graph


TNode = TypeVar("TNode", bound=BTNode)
TEdge = TypeVar("TEdge", bound=BTEdge)
TId = TypeVar("TId")


class PostgresGraphStorage(BaseGraphStorage[TNode, TEdge, TId]):
    """PostgreSQL graph storage implementation."""

    config: PostgresGraphStorageConfig
    _pool: Optional[asyncpg.Pool] = None
    _in_progress: Optional[bool] = False

    async def _insert_start(self) -> None:
        """Initialize the connection pool and create the graph if it doesn't exist."""
        if not self._pool:
            self._pool = await get_connection_pool(self.config)
        await initialize_age_graph(self._pool, self.config.graph_name)
        self._in_progress = True

    async def _insert_end(self) -> None:
        """Clean up after inserting."""
        self._in_progress = False

    async def _query_start(self) -> None:
        """Initialize the connection pool for querying."""
        if not self._pool:
            self._pool = await get_connection_pool(self.config)
        await initialize_age_graph(self._pool, self.config.graph_name)
        self._in_progress = True

    async def _query_end(self) -> None:
        """Clean up after querying."""
        self._in_progress = False

    async def node_count(self) -> int:
        """Get the number of nodes in the graph."""
        if not self._pool:
            await self._query_start()
        assert self._pool is not None
        
        result = await execute_query(
            self._pool,
            f"""
            SELECT count(*) FROM ag_catalog.cypher('{self.config.graph_name}', $$
                MATCH (n:{self.config.node_label})
                RETURN count(n)
            $$) as (count bigint)
            """,
            fetch_type="row"
        )
        return cast(int, result[0]) if result else 0

    async def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        if not self._pool:
            await self._query_start()
        assert self._pool is not None
        
        result = await execute_query(
            self._pool,
            f"""
            SELECT count(*) FROM ag_catalog.cypher('{self.config.graph_name}', $$
                MATCH ()-[r:{self.config.edge_label}]->()
                RETURN count(r)
            $$) as (count bigint)
            """,
            fetch_type="row"
        )
        return cast(int, result[0]) if result else 0

    async def upsert_node(self, node: TNode, node_index: Union[TIndex, None]) -> TIndex:
        """Insert or update a node."""
        if not self._pool:
            await self._insert_start()
        
        # Convert node data to a format AGE can handle
        node_data = {
            "name": str(node.name),
            "type": str(node.type),
            "data": dict(node.data)
        }
        
        result = await execute_query(
            self._pool,
            f"""
            SELECT * FROM ag_catalog.cypher('"{self.config.graph_name}"', $$
            MERGE (n:{self.config.node_label} {{name: $node_name}})
            ON CREATE SET n += $node_data
            ON MATCH SET n += $node_data
            RETURN id(n) as id
            $$, jsonb_build_object('node_name', $1::text, 'node_data', $2::jsonb)) as (id agtype)
            """,
            str(node.name),
            node_data,
            fetch_type="row"
        )
        return cast(TIndex, int(result["id"]))

    async def upsert_edge(self, edge: TEdge, edge_index: Union[TIndex, None]) -> TIndex:
        """Insert or update an edge."""
        if not self._pool:
            await self._insert_start()
        
        # Convert edge data to a format AGE can handle
        edge_data = {
            "type": str(edge.type),
            "data": dict(edge.data)
        }
        
        result = await execute_query(
            self._pool,
            f"""
            SELECT * FROM ag_catalog.cypher('"{self.config.graph_name}"', $$
            MATCH (s:{self.config.node_label} {{name: $source_name}}), (t:{self.config.node_label} {{name: $target_name}})
            MERGE (s)-[r:{self.config.edge_label}]->(t)
            ON CREATE SET r += $edge_data
            ON MATCH SET r += $edge_data
            RETURN id(r) as id
            $$, jsonb_build_object('source_name', $1::text, 'target_name', $2::text, 'edge_data', $3::jsonb)) as (id agtype)
            """,
            str(edge.source),
            str(edge.target),
            edge_data,
            fetch_type="row"
        )
        return cast(TIndex, int(result["id"]))

    async def get_node(self, node: Union[TNode, TId]) -> Union[Tuple[TNode, TIndex], Tuple[None, None]]:
        """Get a node and its index by ID or node data."""
        if not self._pool:
            await self._query_start()
        assert self._pool is not None
        
        node_name = node.name if isinstance(node, BTNode) else str(node)
        result = await execute_query(
            self._pool,
            f"""
            SELECT * FROM ag_catalog.cypher('"{self.config.graph_name}"', $$
                MATCH (n:{self.config.node_label})
                WHERE n.name = $node_name
                RETURN properties(n) as props, id(n)
            $$, jsonb_build_object('node_name', $1::text)) as (props jsonb, id bigint)
            """,
            node_name,
            fetch_type="row"
        )
        
        if result:
            node_props = result["props"]
            return cast(TNode, TNode(
                name=node_props["name"],
                type=node_props["type"],
                data=node_props["data"]
            )), cast(TIndex, int(result["id"]))
        return None, None

    async def get_node_by_index(self, index: TIndex) -> Union[TNode, None]:
        """Get a node by its index."""
        if not self._pool:
            await self._query_start()
        assert self._pool is not None
        
        result = await execute_query(
            self._pool,
            f"""
            SELECT * FROM ag_catalog.cypher('"{self.config.graph_name}"', $$
                MATCH (n:{self.config.node_label})
                WHERE id(n) = $node_id
                RETURN properties(n) as props
            $$, jsonb_build_object('node_id', $1::bigint)) as (props jsonb)
            """,
            int(index),
            fetch_type="row"
        )
        
        if result:
            node_props = result["props"]
            return cast(TNode, TNode(
                name=node_props["name"],
                type=node_props["type"],
                data=node_props["data"]
            ))
        return None

    async def get_edges(self, source_node: Union[TId, TIndex], target_node: Union[TId, TIndex]) -> Iterable[Tuple[TEdge, TIndex]]:
        """Get edges between two nodes."""
        if not self._pool:
            await self._query_start()
        assert self._pool is not None
        
        result = await execute_query(
            self._pool,
            f"""
            SELECT * FROM ag_catalog.cypher('"{self.config.graph_name}"', $$
                MATCH (s)-[r:{self.config.edge_label}]->(t)
                WHERE id(s) = $source_id AND id(t) = $target_id
                RETURN properties(r) as props, id(r) as idx, s.name as source, t.name as target
            $$, jsonb_build_object('source_id', $1::bigint, 'target_id', $2::bigint)) as (props jsonb, idx bigint, source text, target text)
            """,
            int(source_node),
            int(target_node),
            fetch_type="all"
        )
        
        edges = []
        for row in result:
            edge = cast(TEdge, TEdge(
                source=row["source"],
                target=row["target"],
                type=row["props"]["type"],
                data=row["props"]["data"]
            ))
            edges.append((edge, cast(TIndex, int(row["idx"]))))
        return edges

    async def get_edge_indices(self, source_node: Union[TId, TIndex], target_node: Union[TId, TIndex]) -> Iterable[TIndex]:
        """Get edge indices between two nodes."""
        if not self._pool:
            await self._query_start()
        assert self._pool is not None
        
        result = await execute_query(
            self._pool,
            f"""
            SELECT * FROM ag_catalog.cypher('"{self.config.graph_name}"', $$
                MATCH (s)-[r:{self.config.edge_label}]->(t)
                WHERE id(s) = $source_id AND id(t) = $target_id
                RETURN id(r) as idx
            $$, jsonb_build_object('source_id', $1::bigint, 'target_id', $2::bigint)) as (idx agtype)
            """,
            int(source_node),
            int(target_node),
            fetch_type="all"
        )
        
        return [cast(TIndex, int(row["idx"])) for row in result]

    async def get_edge_by_index(self, index: TIndex) -> Union[TEdge, None]:
        """Get an edge by its index."""
        if not self._pool:
            await self._query_start()
        assert self._pool is not None
        
        result = await execute_query(
            self._pool,
            f"""
            SELECT * FROM ag_catalog.cypher('"{self.config.graph_name}"', $$
                MATCH (s)-[r:{self.config.edge_label}]->(t)
                WHERE id(r) = $edge_id
                RETURN r, s.name as source, t.name as target
            $$, jsonb_build_object('edge_id', $1::bigint)) as (edge agtype, source text, target text)
            """,
            int(index),
            fetch_type="row"
        )
        
        if result:
            edge_data = result["edge"]
            return cast(TEdge, TEdge(
                source=result["source"],
                target=result["target"],
                type=edge_data["type"],
                data=edge_data["data"]
            ))
        return None

    async def are_neighbours(self, source_node: Union[TId, TIndex], target_node: Union[TId, TIndex]) -> bool:
        """Check if two nodes are neighbours."""
        if not self._pool:
            await self._query_start()
        assert self._pool is not None
        
        result = await execute_query(
            self._pool,
            f"""
            SELECT * FROM ag_catalog.cypher('"{self.config.graph_name}"', $$
                MATCH (s)-[:{self.config.edge_label}]->(t)
                WHERE id(s) = $source_id AND id(t) = $target_id
                RETURN count(*) > 0 as connected
            $$, jsonb_build_object('source_id', $1::bigint, 'target_id', $2::bigint)) as (connected boolean)
            """,
            int(source_node),
            int(target_node),
            fetch_type="row"
        )
        return bool(result["connected"]) if result else False

    async def insert_edges(self, edges: Optional[Iterable[TEdge]] = None, indices: Optional[Iterable[Tuple[TIndex, TIndex]]] = None) -> List[TIndex]:
        """Insert multiple edges."""
        if not self._pool:
            await self._insert_start()
        assert self._pool is not None
        
        if edges:
            # Insert edges with full data
            result_indices = []
            for edge in edges:
                idx = await self.upsert_edge(edge, None)
                result_indices.append(idx)
            return result_indices
        
        elif indices:
            # Insert edges between node pairs
            result_indices = []
            for source, target in indices:
                result = await execute_query(
                    self._pool,
                    f"""
                    SELECT * FROM ag_catalog.cypher('"{self.config.graph_name}"', $$
                        MATCH (s), (t)
                        WHERE id(s) = $source_id AND id(t) = $target_id
                        MERGE (s)-[r:{self.config.edge_label}]->(t)
                        RETURN id(r) as idx
                    $$, jsonb_build_object('source_id', $1::bigint, 'target_id', $2::bigint)) as (idx agtype)
                    """,
                    int(source),
                    int(target),
                    fetch_type="row"
                )
                result_indices.append(cast(TIndex, int(result["idx"])))
            return result_indices
        
        else:
            raise ValueError("Must provide either edges or indices")

    async def delete_edges_by_index(self, indices: Iterable[TIndex]) -> None:
        """Delete edges by their indices."""
        if not self._pool:
            await self._insert_start()
        assert self._pool is not None
        
        await execute_query(
            self._pool,
            f"""
            SELECT * FROM ag_catalog.cypher('{self.config.graph_name}', $$
                UNWIND $1 as idx
                MATCH ()-[r:{self.config.edge_label}]->()
                WHERE id(r) = idx
                DELETE r
            $$) as (result agtype)
            """,
            [int(idx) for idx in indices]
        )