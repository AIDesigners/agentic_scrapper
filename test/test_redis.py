"""
Unit tests for read-only (debug) mode of Redis client.

This module tests the RedisDBClient in read-only/debug mode to ensure it
correctly buffers writes in internal storage while reading from real Redis.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from redis_driver import RedisDBClient


@pytest.mark.asyncio
async def _cleanup_redis_test_data():
    import redis.asyncio as redis
    c = await redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    await c.flushdb()
    await c.aclose()


class TestRedisDBClientReadonly:
    """Test suite for RedisDBClient in read-only/debug mode."""

    @pytest.mark.asyncio
    async def test_connect_and_close(self):
        client = RedisDBClient(readonly=True)
        result = await client.connect()
        assert result == 0
        assert client._connected

        await client.close()
        assert not client._connected

    @pytest.mark.asyncio
    async def test_check_and_add_new_url(self):
        client = RedisDBClient(readonly=True)
        await client.connect()

        result = await client.check_and_add("https://example.com")
        assert result == 1, "Should return 1 for new URL (buffered)"

        result = await client.check_and_add("https://example.com")
        assert result == 0, "Should return 0 for existing URL"

        await client.close()

    @pytest.mark.asyncio
    async def test_multiple_urls(self):
        await _cleanup_redis_test_data()
        client = RedisDBClient(readonly=True)
        await client.connect()

        urls = [
            "https://example1.com",
            "https://example2.com",
            "https://example3.com",
        ]

        for url in urls:
            result = await client.check_and_add(url)
            assert result == 1, "Should buffer new URL"

        count = len(client._visited_urls)
        assert count == 3

        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with RedisDBClient(readonly=True) as client:
            await client.connect()
            assert client._connected
            result = await client.check_and_add("https://test.com")
            assert result == 1

        assert not client._connected

    @pytest.mark.asyncio
    async def test_get_host_and_port(self):
        client = RedisDBClient(host="testhost", port=6379, readonly=True)
        assert client.host == "testhost"
        assert client.port == 6379

    @pytest.mark.asyncio
    async def test_restart(self):
        client = RedisDBClient(readonly=True)
        result = await client.connect()
        assert result == 0
        assert client._connected

        result = await client.restart()
        assert result == 0
        assert client._connected

        await client.close()

    @pytest.mark.asyncio
    async def test_get_visited_urls(self):
        await _cleanup_redis_test_data()
        client = RedisDBClient(readonly=False)
        await client.connect()

        urls = ["https://a.com", "https://b.com"]
        for url in urls:
            await client.check_and_add(url)

        result = await client.get_visited_urls()
        assert "https://a.com" in result
        assert "https://b.com" in result

        await _cleanup_redis_test_data()
        await client.close()


class TestRedisDBClientProduction:
    """Test suite for RedisDBClient in production mode (atomic writes)."""

    @pytest.mark.asyncio
    async def test_check_and_add_writes_to_redis(self):
        """Verify that production mode uses atomic SADD."""
        await _cleanup_redis_test_data()
        client = RedisDBClient(readonly=False)
        await client.connect()

        test_url = "https://atomic_test_production.com"
        result = await client.check_and_add(test_url)
        assert result == 1

        result = await client.check_and_add(test_url)
        assert result == 0

        count = await client.get_visited_count()
        assert count == 1, "URL must be present in Redis"

        result = await client.get_visited_urls()
        assert test_url in result

        await _cleanup_redis_test_data()
        await client.close()

    @pytest.mark.asyncio
    async def test_readonly_does_not_write_to_redis(self):
        """Verify that readonly mode does NOT write to Redis."""
        await _cleanup_redis_test_data()
        client = RedisDBClient(readonly=True)
        await client.connect()

        test_url = "https://readonly_test_no_write.com"
        result = await client.check_and_add(test_url)
        assert result == 1

        redis_count = await client._client.scard(RedisDBClient.CRAWLER_VISITED_URLS)
        assert redis_count == 0, "Redis must remain unchanged in readonly mode"

        assert test_url in client._visited_urls

        await client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])