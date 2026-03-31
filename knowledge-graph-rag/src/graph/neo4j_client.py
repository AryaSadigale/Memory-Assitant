# FILE: src/graph/neo4j_client.py
# PURPOSE: Provide an async Neo4j driver wrapper with safe parameterized query helpers.

import time
from typing import Any, Dict, List, Optional

from loguru import logger
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable


class Neo4jConnectionError(RuntimeError):
    """Raised when the Neo4j client cannot establish a connection."""


class Neo4jClient:
    """Async Neo4j driver wrapper."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        """Initialize the Neo4j client."""
        self.uri = uri
        self.user = user
        self.password = password
        self.driver: Optional[Any] = None

    async def connect(self) -> None:
        """Create the async driver and verify connectivity."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            await self.driver.verify_connectivity()
            logger.info("Connected to Neo4j at {}", self.uri)
        except (ServiceUnavailable, Neo4jError, OSError) as exc:
            raise Neo4jConnectionError(f"Failed to connect to Neo4j at {self.uri}") from exc

    async def close(self) -> None:
        """Close the Neo4j driver if it exists."""
        if self.driver is not None:
            await self.driver.close()
            self.driver = None

    async def run_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a read query and return all rows as dictionaries."""
        if self.driver is None:
            raise Neo4jConnectionError("Neo4j driver is not connected.")
        started = time.perf_counter()
        async with self.driver.session() as session:
            result = await session.run(query, params or {})
            records = [record.data() async for record in result]
        duration = time.perf_counter() - started
        if duration > 1.0:
            logger.warning("Slow Neo4j query completed in {:.2f}s", duration)
        return records

    async def run_write(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Run a write query without returning rows."""
        if self.driver is None:
            raise Neo4jConnectionError("Neo4j driver is not connected.")
        started = time.perf_counter()
        async with self.driver.session() as session:
            result = await session.run(query, params or {})
            await result.consume()
        duration = time.perf_counter() - started
        if duration > 1.0:
            logger.warning("Slow Neo4j write completed in {:.2f}s", duration)

    async def run_batch_write(self, query: str, batch: List[Dict[str, Any]]) -> None:
        """Run an UNWIND batch write for efficient inserts or updates."""
        if self.driver is None:
            raise Neo4jConnectionError("Neo4j driver is not connected.")
        started = time.perf_counter()
        async with self.driver.session() as session:
            result = await session.run(query, {"batch": batch})
            await result.consume()
        duration = time.perf_counter() - started
        if duration > 1.0:
            logger.warning("Slow Neo4j batch write completed in {:.2f}s", duration)
