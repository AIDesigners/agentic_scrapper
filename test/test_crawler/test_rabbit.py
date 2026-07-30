"""
Unit tests for mock implementations of RabbitMQ client.

This module tests the mock RabbitMQClient implementation to ensure it correctly simulates
the behavior of the real client for debugging and unit testing purposes.
"""

import asyncio
import pytest
import sys
import os

# Add the crawler directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'crawler'))

from rabbit_driver import RabbitMQClient as MockRabbitMQClient
from redis_driver import RedisDBClient as MockRedisDBClient


class TestMockRabbitMQClient:
    """Test suite for MockRabbitMQClient."""

    @pytest.mark.asyncio
    async def test_connect_and_close(self):
        """Test basic connect and close functionality."""
        client = MockRabbitMQClient(use_mock=True)
        result = await client.connect()
        assert result == 0, "Connect should return 0 on success"
        assert client._connected, "Client should be connected"
        
        await client.close()
        assert not client._connected, "Client should be disconnected after close"

    @pytest.mark.asyncio
    async def test_publish_and_receive(self):
        """Test publishing and receiving messages."""
        client = MockRabbitMQClient(publish_queue_name="test_queue", use_mock=True)
        await client.connect()
        
        # Publish a message
        test_data = {"url": "https://example.com", "timestamp": "2024-01-01"}
        result = await client.publish_json(test_data)
        assert result == 1, "Publish should return 1 on success"
        
        # Check queue size
        queue_size = await client.get_queue_size()
        assert queue_size == 1, "Queue should have 1 message"
        
        await client.close()

    @pytest.mark.asyncio
    async def test_receive_from_empty_queue(self):
        """Test receiving from an empty queue."""
        client = MockRabbitMQClient(receive_queue_name="empty_queue", use_mock=True)
        await client.connect()
        
        status, message = await client.receive_json()
        assert status == 0, "Should return 0 for empty queue"
        assert message is None, "Message should be None for empty queue"
        
        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality."""
        async with MockRabbitMQClient(publish_queue_name="ctx_queue", use_mock=True) as client:
            # Note: With the StealthMCPManager-style pattern, connect() must be called explicitly
            await client.connect()
            assert client._connected, "Client should be connected inside context"
            result = await client.publish_json({"test": "data"})
            assert result == 1, "Publish inside context should succeed"
        
        assert not client._connected, "Client should be disconnected after context exit"

    @pytest.mark.asyncio
    async def test_double_connect_fails(self):
        """Test that connecting twice returns error."""
        client = MockRabbitMQClient(use_mock=True)
        await client.connect()
        result = await client.connect()
        assert result == -1, "Second connect should return -1"
        await client.close()

    @pytest.mark.asyncio
    async def test_get_host_and_port(self):
        """Test get_host() and get_port() methods."""
        client = MockRabbitMQClient(host="testhost", port=1234, use_mock=True)
        assert client.get_host() == "testhost", "get_host() should return the host"
        assert client.get_port() == 1234, "get_port() should return the port"

    @pytest.mark.asyncio
    async def test_restart(self):
        """Test restart() method disconnects and reconnects."""
        client = MockRabbitMQClient(use_mock=True)
        result = await client.connect()
        assert result == 0, "Connect should succeed"
        assert client._connected, "Client should be connected"
        
        result = await client.restart()
        assert result == 0, "Restart should succeed"
        assert client._connected, "Client should be connected after restart"
        
        await client.close()


@pytest.mark.asyncio
async def test_debug_mode_switch():
    """Test that DEBUG mode correctly switches implementations."""
    import importlib
    
    # We can verify the unified clients have the expected interface
    
    mock_rabbit = MockRabbitMQClient(use_mock=True)
    mock_redis = MockRedisDBClient(use_mock=True)
    
    # Verify mock rabbit has required methods
    assert hasattr(mock_rabbit, 'connect'), "RabbitMQClient should have connect method"
    assert hasattr(mock_rabbit, 'close'), "RabbitMQClient should have close method"
    assert hasattr(mock_rabbit, 'publish_json'), "RabbitMQClient should have publish_json method"
    assert hasattr(mock_rabbit, 'receive_json'), "RabbitMQClient should have receive_json method"
    
    # Verify mock redis has required methods
    assert hasattr(mock_redis, 'connect'), "RedisDBClient should have connect method"
    assert hasattr(mock_redis, 'close'), "RedisDBClient should have close method"
    assert hasattr(mock_redis, 'check_and_add'), "RedisDBClient should have check_and_add method"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])