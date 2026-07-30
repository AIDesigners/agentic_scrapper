"""
Unit tests for mock implementations of RabbitMQ client.

This module tests the mock RabbitMQClient implementation to ensure it correctly simulates
the behavior of the real client for debugging and unit testing purposes.
"""

import asyncio
import pytest
import sys
import os

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rabbit_driver import RabbitMQClient
from redis_driver import RedisDBClient


class TestRabbitMQClientReadonly:
    """Test suite for RabbitMQClient in readonly/debug mode."""

    @pytest.mark.asyncio
    async def test_connect_and_close(self):
        client = RabbitMQClient(readonly=True)
        result = await client.connect()
        assert result == 0
        assert client._connected

        await client.close()
        assert not client._connected

    @pytest.mark.asyncio
    async def test_publish_and_receive(self):
        client = RabbitMQClient(publish_queue_name="test_queue", readonly=True)
        await client.connect()

        test_data = {"url": "https://example.com", "timestamp": "2024-01-01"}
        result = await client.publish_json(test_data)
        assert result == 1

        queue_size = await client.get_queue_size()
        assert queue_size == 1

        await client.close()

    @pytest.mark.asyncio
    async def test_receive_from_empty_queue(self):
        client = RabbitMQClient(receive_queue_name="empty_queue", readonly=True)
        await client.connect()

        status, message = await client.receive_json()
        assert status == 0
        assert message is None

        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with RabbitMQClient(publish_queue_name="ctx_queue", readonly=True) as client:
            await client.connect()
            assert client._connected
            result = await client.publish_json({"test": "data"})
            assert result == 1

        assert not client._connected

    @pytest.mark.asyncio
    async def test_double_connect_fails(self):
        client = RabbitMQClient(readonly=True)
        await client.connect()
        result = await client.connect()
        assert result == -1
        await client.close()

    @pytest.mark.asyncio
    async def test_get_host_and_port(self):
        client = RabbitMQClient(host="testhost", port=1234, readonly=True)
        assert client.host == "testhost"
        assert client.port == 1234

    @pytest.mark.asyncio
    async def test_restart(self):
        client = RabbitMQClient(readonly=True)
        result = await client.connect()
        assert result == 0
        assert client._connected

        result = await client.restart()
        assert result == 0
        assert client._connected

        await client.close()


@pytest.mark.asyncio
async def test_debug_mode_switch():
    readonly_rabbit = RabbitMQClient(readonly=True)
    readonly_redis = RedisDBClient(readonly=True)

    assert hasattr(readonly_rabbit, 'connect')
    assert hasattr(readonly_rabbit, 'close')
    assert hasattr(readonly_rabbit, 'publish_json')
    assert hasattr(readonly_rabbit, 'receive_json')

    assert hasattr(readonly_redis, 'connect')
    assert hasattr(readonly_redis, 'close')
    assert hasattr(readonly_redis, 'check_and_add')


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])