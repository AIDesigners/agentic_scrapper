"""
Unit tests for mock implementations of RabbitMQ client.

This module tests the mock RabbitMQClient implementation to ensure it correctly simulates
the behavior of the real client for debugging and unit testing purposes.
"""

import base64
import json
import os
import pytest
import re
import sys
import urllib.request

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


@pytest.mark.asyncio
async def test_llm_timeout_tuning():
    # The LLM liveness tuning of the agent is env-overridable at import
    # time with the documented defaults (2h read cap, 600s ping gate)
    import importlib
    scrapper_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'scrapper')
    if scrapper_dir not in sys.path:
        sys.path.insert(0, scrapper_dir)
    os.environ["LLM_READ_TIMEOUT"] = "1234"
    os.environ["LLM_PING_TIMEOUT"] = "4321"
    try:
        import scrapper_agent
        importlib.reload(scrapper_agent)
        assert scrapper_agent.LLM_READ_TIMEOUT_S == 1234.0
        assert scrapper_agent.LLM_PING_TIMEOUT_S == 4321.0
    finally:
        del os.environ["LLM_READ_TIMEOUT"]
        del os.environ["LLM_PING_TIMEOUT"]
    importlib.reload(scrapper_agent)
    assert scrapper_agent.LLM_READ_TIMEOUT_S == 7200.0
    assert scrapper_agent.LLM_PING_TIMEOUT_S == 600.0


def test_broker_config_sanity():
    # Production broker topology and permissions via the management API:
    # read-only probes, no AMQP connection, no objects created.
    # Guards the ACCESS_REFUSED failure class: a user whose `write` regex
    # covers the queue name but not the exchange (`amq.default`) publishes
    # into a silent requeue loop instead of failing loudly.
    auth = {"Authorization": "Basic " + base64.b64encode(b"guest:guest").decode()}

    def get(path):
        req = urllib.request.Request("http://localhost:15672/api" + path, headers=auth)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())

    queues = {q["name"] for q in get("/queues/%2f")}
    assert {"crawler_json_queue", "scrapper_json_queue"} <= queues

    perms = {p["user"]: p for p in get("/permissions/")}
    assert re.search(perms["scrapper"]["write"], "amq.default"), \
        "scrapper must be able to publish (write the default exchange)"
    assert re.search(perms["scrapper"]["read"], "crawler_json_queue"), \
        "scrapper must be able to consume crawler_json_queue"
    assert re.search(perms["crawler"]["write"], "amq.default"), \
        "crawler must be able to publish (write the default exchange)"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])