"""PostgreSQL storage configuration."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class PostgresStorageConfig:
    """Configuration for PostgreSQL storage."""
    host: str = field(default="localhost")
    port: int = field(default=5432)
    database: str = field(default="fastrag")
    user: str = field(default="fastrag")
    password: str = field(default="fastrag")
    schema: str = field(default="public")
    pool_size: int = field(default=10)
    ssl: bool = field(default=False)


@dataclass
class PostgresGraphStorageConfig(PostgresStorageConfig):
    """Configuration for PostgreSQL graph storage."""
    graph_name: str = field(default="graph")
    node_label: str = field(default="node")
    edge_label: str = field(default="edge")