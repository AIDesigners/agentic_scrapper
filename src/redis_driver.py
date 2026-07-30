"""
Redis driver module.

This module provides a unified Redis client. In debug mode (DEBUG env var),
readonly is enforced and writes go to internal storage while reads still
check the real Redis server.

Usage:
    # Production (default)
    redis_client = RedisDBClient(host="localhost", port=6379, db=0)

    # Debug mode (readonly enforced)
    redis_client = RedisDBClient(readonly=True)

    # Or use environment variable DEBUG
    import os
    os.environ['DEBUG'] = 'true'  # Will automatically enable readonly mode
"""

import logging
from typing import Set

logger = logging.getLogger(__name__)
logger.propagate = False
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('(%(funcName)s:%(lineno)d) %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class RedisDBClient:
    """
    Unified Redis client that always connects to a real Redis server.

    In readonly mode, reads check Redis while writes go to internal storage
    so the server remains unchanged.

    Attributes:
        CRAWLER_VISITED_URLS: The key used for storing visited URLs.

    Example:
        # Production
        async with RedisDBClient() as client:
            await client.connect()
            result = await client.check_and_add("https://example.com")

        # Debug mode
        async with RedisDBClient(readonly=True) as client:
            await client.connect()
            result = await client.check_and_add("https://example.com")
    """

    CRAWLER_VISITED_URLS = "crawler_visited_urls"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        readonly: bool = False
    ) -> None:
        """
        Initialize the Redis client.

        Args:
            host: Redis server hostname.
            port: Redis server port.
            db: Redis database number.
            readonly: If True, only checks URLs without writing to Redis.
                     If not specified, checks the DEBUG environment variable.
        """
        if not readonly:
            import os
            readonly = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')

        self.readonly = readonly
        self._client = None
        self._host = host
        self._port = port
        self._db = db
        self._connected = False
        self._visited_urls: Set[str] | None = None

    @property
    def host(self) -> str:
        """Redis server hostname."""
        return self._host

    @host.setter
    def host(self, value: str) -> None:
        if self._client is not None:
            raise RuntimeError("Cannot change host while connected. Close connection first.")
        self._host = value

    @property
    def port(self) -> int:
        """Redis server port."""
        return self._port

    @port.setter
    def port(self, value: int) -> None:
        if self._client is not None:
            raise RuntimeError("Cannot change port while connected. Close connection first.")
        self._port = value

    @property
    def db(self) -> int:
        """Redis database number."""
        return self._db

    @db.setter
    def db(self, value: int) -> None:
        if self._client is not None:
            raise RuntimeError("Cannot change db while connected. Close connection first.")
        self._db = value

    async def connect(self) -> int:
        """
        Connect to the Redis server.

        Returns:
            0 on success, -1 on failure.
        """
        if self._connected:
            logger.error("Redis client is already opened!")
            return -1

        try:
            import redis.asyncio as redis
            self._client = await redis.Redis(host=self._host, port=self._port, db=self._db, decode_responses=True)
            await self._client.ping()
            self._connected = True
            self._visited_urls = set() if self.readonly else None
            visited_count = await self.get_visited_count()
            logger.info(f"Redis connected to {self._host}:{self._port}, {visited_count} visited URLs")
            return 0
        except Exception as e:
            logger.error(f"Failed to connect Redis, error: {e}")
            self._client = None
            return -1

    async def disconnect(self) -> None:
        """
        Gracefully disconnect and clean up all resources.

        This method closes the connection and clears internal state.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._connected = False
        self._visited_urls = set() if self.readonly else None
        logger.info("Redis connection closed.")

    async def restart(self) -> int:
        """
        Restart the Redis connection by disconnecting and reconnecting.

        Returns:
            0 on success, -1 on failure.
        """
        await self.disconnect()
        return await self.connect()

    async def close(self) -> None:
        """
        Close the Redis connection (alias for disconnect).
        """
        await self.disconnect()

    async def check_and_add(self, url: str) -> int:
        """
        Check if a URL has been visited and add it if not already present.

        In production mode this uses the atomic SADD command. In readonly mode
        it checks internal storage first, then Redis via SISMEMBER, and buffers
        writes locally without touching the server.

        Returns:
            1 if URL is new and was added (or buffered if readonly)
            0 if URL already exists
            -1 on error
        """
        if self._client is None:
            logger.error("Redis client is not yet opened!")
            return -1

        if self.readonly:
            if url in self._visited_urls:
                logger.debug(f"URL already visited (internal storage): {url}")
                return 0
            exists = await self._client.sismember(self.CRAWLER_VISITED_URLS, url)
            if exists:
                logger.debug(f"URL already visited (Redis): {url}")
                return 0
            self._visited_urls.add(url)
            logger.debug(f"Buffered URL in internal storage: {url}")
            return 1
        else:
            try:
                return await self._client.sadd(self.CRAWLER_VISITED_URLS, url)
            except Exception as e:
                logger.error(f"Redis error: {e}")
                return -1

    async def get_visited_count(self) -> int:
        """
        Get the total number of visited URLs.

        Returns:
            Number of visited URLs in Redis, or -1 if not connected.
        """
        if self._client is None:
            logger.error("Redis client is not yet opened!")
            return -1
        try:
            return await self._client.scard(self.CRAWLER_VISITED_URLS)
        except Exception as e:
            logger.error(f"Redis error: {e}")
            return -1

    async def get_visited_urls(self) -> Set[str]:
        """
        Get a copy of all visited URLs from Redis.

        Returns:
            Set of visited URLs.
        """
        if self._client is None:
            logger.error("Redis client is not yet opened!")
            return set()
        try:
            return set(await self._client.smembers(self.CRAWLER_VISITED_URLS))
        except Exception as e:
            logger.error(f"Redis error: {e}")
            return set()

    async def __aenter__(self):
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager, disconnecting Redis on completion."""
        await self.disconnect()
