"""
Unit tests for mock implementations of Redis client.

This module tests the mock RedisDBClient implementation to ensure it correctly simulates
the behavior of the real client for debugging and unit testing purposes.
"""

import asyncio
import pytest
import sys
import os

# Add the crawler directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'crawler'))

from redis_driver import RedisDBClient as MockRedisDBClient


class TestMockRedisDBClient:
    """Test suite for MockRedisDBClient."""

    @pytest.mark.asyncio
    async def test_connect_and_close(self):
        """Test basic connect and close functionality."""
        client = MockRedisDBClient(use_mock=True)
        result = await client.connect()
        assert result == 0, "Connect should return 0 on success"
        assert client._connected, "Client should be connected"
        
        await client.close()
        assert not client._connected, "Client should be disconnected after close"

    @pytest.mark.asyncio
    async def test_check_and_add_new_url(self):
        """Test adding a new URL."""
        client = MockRedisDBClient(use_mock=True)
        await client.connect()
        
        result = await client.check_and_add("https://example.com")
        assert result == 1, "Should return 1 for new URL (added)"
        
        result = await client.check_and_add("https://example.com")
        assert result == 0, "Should return 0 for existing URL"
        
        await client.close()

    @pytest.mark.asyncio
    async def test_multiple_urls(self):
        """Test adding multiple URLs."""
        client = MockRedisDBClient(use_mock=True)
        await client.connect()
        
        urls = [
            "https://example1.com",
            "https://example2.com",
            "https://example3.com",
        ]
        
        for url in urls:
            result = await client.check_and_add(url)
            assert result == 1, f"Should add new URL {url}"
        
        # Check count
        count = await client.get_visited_count()
        assert count == 3, "Should have 3 visited URLs"
        
        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality."""
        async with MockRedisDBClient(use_mock=True) as client:
            # Note: With the StealthMCPManager-style pattern, connect() must be called explicitly
            await client.connect()
            assert client._connected, "Client should be connected inside context"
            result = await client.check_and_add("https://test.com")
            assert result == 1, "Check and add inside context should succeed"
        
        assert not client._connected, "Client should be disconnected after context exit"

    @pytest.mark.asyncio
    async def test_get_host_and_port(self):
        """Test get_host() and get_port() methods."""
        client = MockRedisDBClient(host="testhost", port=6379, use_mock=True)
        assert client.get_host() == "testhost", "get_host() should return the host"
        assert client.get_port() == 6379, "get_port() should return the port"

    @pytest.mark.asyncio
    async def test_restart(self):
        """Test restart() method disconnects and reconnects."""
        client = MockRedisDBClient(use_mock=True)
        result = await client.connect()
        assert result == 0, "Connect should succeed"
        assert client._connected, "Client should be connected"
        
        result = await client.restart()
        assert result == 0, "Restart should succeed"
        assert client._connected, "Client should be connected after restart"
        
        await client.close()

    @pytest.mark.asyncio
    async def test_visited_url_exists(self):
        """Test checking if URL exists."""
        client = MockRedisDBClient(use_mock=True)
        await client.connect()
        
        url = "https://exists.com"
        await client.check_and_add(url)
        
        exists = await client.visited_url_exists(url)
        assert exists, f"URL {url} should exist"
        
        not_exists = await client.visited_url_exists("https://notexists.com")
        assert not not_exists, "URL should not exist"
        
        await client.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])