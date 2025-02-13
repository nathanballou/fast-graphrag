"""Test configuration and shared fixtures."""

import pytest

pytest_plugins = [
    "tests._storage._postgres",  # Import PostgreSQL fixtures
] 