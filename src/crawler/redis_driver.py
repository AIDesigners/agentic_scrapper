"""
Redis driver module with both real and mock implementations.

This module provides a unified Redis client that can operate in either
real mode (connecting to an actual Redis server) or mock mode (using
in-memory storage for testing/debugging).

Usage:
    # Use real Redis (default)
    redis_client = RedisDBClient(host="localhost", port=6379, db=0, readonly=False)
    
    # Use mock Redis
    redis_client = RedisDBClient(use_mock=True)
    
    # Or use environment variable DEBUG
    import os
    os.environ['DEBUG'] = 'true'  # Will automatically use mock
"""

import logging
from typing import Optional, Set, Dict, Any

logger = logging.getLogger(__name__)
logger.propagate = False
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('(%(funcName)s:%(lineno)d) %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

import asyncio


class RedisDBClient:
    """
    Unified Redis client that supports both real and mock implementations.
    
    This class acts as a facade that delegates to either the real Redis
    client implementation or a mock implementation based on the `use_mock`
    parameter or the DEBUG environment variable.
    
    Attributes:
        CRAWLER_VISITED_URLS: The key used for storing visited URLs.
    
    Example:
        # Real Redis
        async with RedisDBClient(use_mock=False) as client:
            result = await client.check_and_add("https://example.com")
        
        # Mock Redis
        async with RedisDBClient(use_mock=True) as client:
            result = await client.check_and_add("https://example.com")
    """

    CRAWLER_VISITED_URLS = "crawler_visited_urls"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        readonly: bool = False,
        use_mock: Optional[bool] = None
    ) -> None:
        """
        Initialize the Redis client.
        
        Args:
            host: Redis server hostname.
            port: Redis server port.
            db: Redis database number.
            readonly: If True, only checks URLs without adding them.
            use_mock: If True, uses mock implementation. If False, uses real Redis.
                      If None, checks the DEBUG environment variable.
        """
        # Determine if we should use mock implementation
        if use_mock is None:
            import os
            use_mock = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')
        
        self._use_mock = use_mock
        self.readonly = readonly
        self._client = None
        self._host = host
        self._port = port
        self._db = db
        self._connected = False
        self._visited_urls: Set[str] = set()

    @property
    def host(self) -> str:
        return self._host

    @host.setter
    def host(self, value: str) -> None:
        if self._client is not None:
            raise RuntimeError("Cannot change host while connected. Close connection first.")
        self._host = value

    @property
    def port(self) -> int:
        return self._port

    @port.setter
    def port(self, value: int) -> None:
        if self._client is not None:
            raise RuntimeError("Cannot change port while connected. Close connection first.")
        self._port = value

    @property
    def db(self) -> int:
        return self._db

    @db.setter
    def db(self, value: int) -> None:
        if self._client is not None:
            raise RuntimeError("Cannot change db while connected. Close connection first.")
        self._db = value

    def get_host(self) -> str:
        """Get the Redis server host."""
        return self._host

    def get_port(self) -> int:
        """Get the Redis server port."""
        return self._port

    async def connect(self) -> int:
        """
        Connect to Redis (real or mock based on configuration).
        
        Returns:
            0 on success, -1 on failure.
        """
        if self._connected:
            logger.error("Redis client is already opened!")
            return -1

        if self._use_mock:
            # Mock implementation
            try:
                self._connected = True
                self._visited_urls = set()
                logger.info(f"Mock Redis connected to {self._host}:{self._port}")
                return 0
            except Exception as e:
                logger.error(f"Failed to connect Redis (mock), error: {e}")
                return -1
        else:
            # Real implementation
            try:
                import redis.asyncio as redis
                self._client = await redis.Redis(host=self._host, port=self._port, db=self._db, decode_responses=True)
                await self._client.ping()
                self._connected = True
                logger.info(f"Redis connected to {self._host}:{self._port}")
                return 0
            except Exception as e:
                logger.error(f"Failed to connect Redis, error: {e}")
                self._client = None
                return -1

    async def disconnect(self) -> None:
        """
        Gracefully disconnect and clean up all resources.
        
        This method reverses the connection process by closing the connection
        in the correct order and clearing internal state.
        """
        if self._use_mock:
            self._connected = False
            self._visited_urls = set()
            logger.info("Mock Redis connection closed.")
        else:
            if self._client is not None:
                await self._client.close()
                self._client = None
            self._connected = False
            logger.info("Redis connection closed.")

    async def restart(self) -> int:
        """
        Restart the Redis connection by disconnecting and reconnecting.
        
        This is useful for recovering from errors or when wanting to refresh
        the connection.
        
        Returns:
            int: Error code from connect() method.
        """
        await self.disconnect()
        return await self.connect()

    async def close(self) -> None:
        """
        Close the Redis connection (alias for disconnect).
        
        This method is kept for backward compatibility.
        """
        await self.disconnect()

    async def check_and_add(self, url: str) -> int:
        """
        Check if a URL has been visited and add it if not already present.
        
        Returns:
            1 if URL is new and was added (or would be added if readonly)
            0 if URL already exists
            -1 on error
        """
        if self._use_mock:
            return await self._check_and_add_mock(url)
        else:
            return await self._check_and_add_real(url)

    async def _check_and_add_real(self, url: str) -> int:
        """Real Redis implementation of check_and_add."""
        if self._client is None:
            logger.error("Redis client is not yet opened!")
            return -1
        try:
            if self.readonly:
                # sismember returns 1 if it exists, 0 if it doesn't.
                # We return 1 (New) if it's NOT a member, 0 (Exists) if it is.
                return 0 if await self._client.sismember(self.CRAWLER_VISITED_URLS, url) else 1

            # SADD is natively atomic. Returns 1 if added, 0 if it already existed.
            return await self._client.sadd(self.CRAWLER_VISITED_URLS, url)

        except Exception as e:
            logger.error(f"Redis error: {e}")
            return -1

    async def _check_and_add_mock(self, url: str) -> int:
        """Mock Redis implementation of check_and_add."""
        if not self._connected:
            logger.error("Redis client is not yet opened!")
            return -1
        try:
            if url in self._visited_urls:
                logger.debug(f"URL already visited: {url}")
                return 0
            else:
                self._visited_urls.add(url)
                logger.debug(f"Added URL to visited set: {url}")
                return 1
        except Exception as e:
            logger.error(f"Redis error: {e}")
            return -1

    # Mock-only methods (for testing purposes)
    async def visited_url_exists(self, url: str) -> bool:
        """Check if a URL has been visited (for testing purposes)."""
        if self._use_mock:
            return url in self._visited_urls
        # For real implementation, just use check_and_add in readonly mode
        self.readonly = True
        result = await self._check_and_add_real(url)
        return result == 0

    async def get_visited_count(self) -> int:
        """Get the total number of visited URLs (for testing purposes)."""
        if self._use_mock:
            return len(self._visited_urls)
        return -1

    async def get_visited_urls(self) -> Set[str]:
        """Get a copy of all visited URLs (for testing purposes)."""
        if self._use_mock:
            return self._visited_urls.copy()
        return set()

    async def clear_visited(self) -> None:
        """Clear all visited URLs (for testing purposes)."""
        if self._use_mock:
            self._visited_urls.clear()
            logger.debug("Cleared all visited URLs")
        else:
            logger.warning("clear_visited is only available in mock mode")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
