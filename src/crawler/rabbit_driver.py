"""
RabbitMQ driver module with both real and mock implementations.

This module provides a unified RabbitMQ client that can operate in either
real mode (connecting to an actual RabbitMQ server) or mock mode (using
in-memory storage for testing/debugging).

Usage:
    # Use real RabbitMQ (default)
    rabbit_client = RabbitMQClient(host="localhost", port=5672, vhost="/")
    
    # Use mock RabbitMQ
    rabbit_client = RabbitMQClient(use_mock=True)
    
    # Or use environment variable DEBUG
    import os
    os.environ['DEBUG'] = 'true'  # Will automatically use mock
"""

import logging
import json
from typing import Optional, Tuple, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)
logger.propagate = False
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('(%(funcName)s:%(lineno)d) %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class RabbitMQClient:
    """
    Unified RabbitMQ client that supports both real and mock implementations.
    
    This class acts as a facade that delegates to either the real RabbitMQ
    client implementation or a mock implementation based on the `use_mock`
    parameter or the DEBUG environment variable.
    
    Attributes:
        _connected: Whether the mock client is connected (mock only).
        _messages: Dictionary mapping queue names to message queues (mock only).
        _publish_count: Counter for number of published messages (mock only).
    
    Example:
        # Real RabbitMQ
        async with RabbitMQClient(use_mock=False) as client:
            await client.connect()
            await client.publish_json({"data": "message"})
        
        # Mock RabbitMQ
        async with RabbitMQClient(use_mock=True) as client:
            await client.connect()
            await client.publish_json({"data": "message"})
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        vhost: str = "/",
        publish_queue_name: Optional[str] = None,
        receive_queue_name: Optional[str] = None,
        username: str = "guest",
        password: str = "guest",
        use_mock: Optional[bool] = None
    ) -> None:
        """
        Initialize the RabbitMQ client.
        
        Args:
            host: RabbitMQ server hostname.
            port: RabbitMQ server port.
            vhost: Virtual host to connect to.
            publish_queue_name: Default queue name for publishing.
            receive_queue_name: Default queue name for receiving.
            username: RabbitMQ username.
            password: RabbitMQ password.
            use_mock: If True, uses mock implementation. If False, uses real RabbitMQ.
                      If None, checks the DEBUG environment variable.
        """
        # Determine if we should use mock implementation
        if use_mock is None:
            import os
            use_mock = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')
        
        self._use_mock = use_mock
        self._host = host
        self._port = port
        self._vhost = vhost
        self._publish_queue_name = publish_queue_name
        self._receive_queue_name = receive_queue_name
        self._username = username
        self._password = password
        self._connection = None
        self._channel = None
        
        # Mock-specific attributes
        self._connected = False
        self._messages: Dict[str, deque] = {}
        self._publish_count = 0

    @property
    def host(self) -> str:
        return self._host

    @host.setter
    def host(self, value: str) -> None:
        if self._connection is not None:
            raise RuntimeError("Cannot change host while connected. Close connection first.")
        self._host = value

    @property
    def port(self) -> int:
        return self._port

    @port.setter
    def port(self, value: int) -> None:
        if self._connection is not None:
            raise RuntimeError("Cannot change port while connected. Close connection first.")
        self._port = value

    @property
    def publish_queue_name(self) -> Optional[str]:
        return self._publish_queue_name

    @publish_queue_name.setter
    def publish_queue_name(self, value: Optional[str]) -> None:
        self._publish_queue_name = value

    @property
    def receive_queue_name(self) -> Optional[str]:
        return self._receive_queue_name

    @receive_queue_name.setter
    def receive_queue_name(self, value: Optional[str]) -> None:
        self._receive_queue_name = value

    def get_host(self) -> str:
        """Get the RabbitMQ server host."""
        return self._host

    def get_port(self) -> int:
        """Get the RabbitMQ server port."""
        return self._port

    async def connect(self) -> int:
        """
        Connect to RabbitMQ (real or mock based on configuration).
        
        Returns:
            0 on success, -1 on failure.
        """
        if self._use_mock:
            # Mock implementation
            if self._connected:
                logger.error("RabbitMQ ERROR. Client is already opened!")
                return -1
            try:
                self._connected = True
                self._messages = {}
                logger.info(f"Mock RabbitMQ connected to {self._host}:{self._port}")
                return 0
            except Exception as e:
                logger.error(f"RabbitMQ ERROR. Failed to connect (mock), error: {e}")
                return -1
        else:
            # Real implementation
            if self._connection is not None:
                logger.error("RabbitMQ ERROR. Client is already opened!")
                return -1
            try:
                import aio_pika
                # connect_robust automatically handles reconnects
                self._connection = await aio_pika.connect_robust(
                    host=self._host,
                    port=self._port,
                    virtualhost=self._vhost,
                    login=self._username,
                    password=self._password
                )
                self._channel = await self._connection.channel()
                logger.info(f"RabbitMQ connected to {self._host}:{self._port}")
                return 0
            except Exception as e:
                logger.error(f"RabbitMQ ERROR. Failed to connect, error: {e}")
                self._connection = None
                self._channel = None
                return -1

    async def disconnect(self) -> None:
        """
        Gracefully disconnect and clean up all resources.
        
        This method reverses the connection process by closing the connection
        in the correct order and clearing internal state.
        """
        if self._use_mock:
            self._connected = False
            self._messages = {}
            logger.info("Mock RabbitMQ connection closed.")
        else:
            if self._channel is not None and not self._channel.is_closed:
                await self._channel.close()
                self._channel = None
            if self._connection is not None and not self._connection.is_closed:
                await self._connection.close()
                self._connection = None
            logger.info("RabbitMQ connection closed.")

    async def restart(self) -> int:
        """
        Restart the RabbitMQ connection by disconnecting and reconnecting.
        
        This is useful for recovering from errors or when wanting to refresh
        the connection.
        
        Returns:
            int: Error code from connect() method.
        """
        await self.disconnect()
        return await self.connect()

    async def close(self) -> None:
        """
        Close the RabbitMQ connection (alias for disconnect).
        
        This method is kept for backward compatibility.
        """
        await self.disconnect()

    async def publish_json(self, data: dict, queue_name: Optional[str] = None) -> int:
        """
        Publish a JSON message (real or mock based on configuration).
        
        Args:
            data: The JSON data to publish.
            queue_name: The queue name to publish to.
            
        Returns:
            1 on success, 0 on failure.
        """
        if self._use_mock:
            return await self._publish_json_mock(data, queue_name)
        else:
            return await self._publish_json_real(data, queue_name)

    async def _publish_json_real(self, data: dict, queue_name: Optional[str] = None) -> int:
        """Real RabbitMQ implementation of publish_json."""
        if self._channel is None or self._channel.is_closed:
            logger.error("RabbitMQ ERROR. Client is not connected!")
            return 0
        try:
            import aio_pika
            message_bytes = json.dumps(data).encode("utf-8")
            if queue_name is None:
                queue_name = self._publish_queue_name
            assert queue_name is not None, "RabbitMQ ERROR. Publishing into unspecified queue!"
            await self._channel.default_exchange.publish(
                aio_pika.Message(
                    body=message_bytes,
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    content_type="application/json"
                ),
                routing_key=queue_name
            )
            return 1
        except Exception as e:
            logger.error(f"RabbitMQ ERROR. Failed to publish a message {e}")
            return 0

    async def _publish_json_mock(self, data: dict, queue_name: Optional[str] = None) -> int:
        """Mock RabbitMQ implementation of publish_json."""
        if not self._connected:
            logger.error("RabbitMQ ERROR. Client is not connected!")
            return 0
        try:
            if queue_name is None:
                queue_name = self._publish_queue_name
            assert queue_name is not None, "RabbitMQ ERROR. Publishing into unspecified queue!"
            
            if queue_name not in self._messages:
                self._messages[queue_name] = deque()
            
            self._messages[queue_name].append(data)
            self._publish_count += 1
            logger.debug(f"Mock RabbitMQ published message to queue '{queue_name}': {data}")
            return 1
        except Exception as e:
            logger.error(f"RabbitMQ ERROR. Failed to publish a message {e}")
            return 0

    async def receive_json(self, queue_name: Optional[str] = None) -> Tuple[int, Optional[dict]]:
        """
        Receive a JSON message (real or mock based on configuration).
        
        Args:
            queue_name: The queue name to receive from.
            
        Returns:
            Tuple of (status_code, message):
            - (1, message) if message was received
            - (0, None) if queue is empty
            - (-1, None) on error
        """
        if self._use_mock:
            return await self._receive_json_mock(queue_name)
        else:
            return await self._receive_json_real(queue_name)

    async def _receive_json_real(self, queue_name: Optional[str] = None) -> Tuple[int, Optional[dict]]:
        """Real RabbitMQ implementation of receive_json."""
        if self._channel is None or self._channel.is_closed:
            logger.error("RabbitMQ error. Client is not connected!")
            return (-1, None)
        try:
            from aio_pika.exceptions import QueueEmpty
            if queue_name is None:
                queue_name = self._receive_queue_name
            assert queue_name is not None, "RabbitMQ ERROR. Receiving from unspecified queue!"
            queue = await self._channel.declare_queue(queue_name, passive=True)
            # fail_if_empty=True ensures it returns immediately instead of waiting
            message = await queue.get(fail_if_empty=True)
            # async with message.process() ensures the message is acknowledged
            async with message.process():
                try:
                    payload = json.loads(message.body.decode("utf-8"))
                    return (1, payload)
                except json.JSONDecodeError as e:
                    logger.error(f"RabbitMQ ERROR. Failed to decode JSON payload: {e}")
                    return (-1, None)
        except QueueEmpty:
            return (0, None)
        except Exception as e:
            logger.error(f"RabbitMQ ERROR. Failed to get message from queue '{queue_name}': {e}")
            return (-1, None)

    async def _receive_json_mock(self, queue_name: Optional[str] = None) -> Tuple[int, Optional[dict]]:
        """Mock RabbitMQ implementation of receive_json."""
        if not self._connected:
            logger.error("RabbitMQ error. Client is not connected!")
            return (-1, None)
        try:
            if queue_name is None:
                queue_name = self._receive_queue_name
            assert queue_name is not None, "RabbitMQ ERROR. Receiving from unspecified queue!"
            
            if queue_name not in self._messages or len(self._messages[queue_name]) == 0:
                return (0, None)
            
            message = self._messages[queue_name].popleft()
            logger.debug(f"Mock RabbitMQ received message from queue '{queue_name}': {message}")
            return (1, message)
        except Exception as e:
            logger.error(f"RabbitMQ ERROR. Failed to get message from queue '{queue_name}': {e}")
            return (-1, None)

    # Mock-only methods (for testing purposes)
    async def get_queue_size(self, queue_name: Optional[str] = None) -> int:
        """Get the number of messages in a queue (for testing purposes)."""
        if queue_name is None:
            queue_name = self._publish_queue_name
        if queue_name is None or queue_name not in self._messages:
            return 0
        return len(self._messages[queue_name])

    async def clear_queue(self, queue_name: Optional[str] = None) -> None:
        """Clear all messages from a queue (for testing purposes)."""
        if queue_name is None:
            queue_name = self._publish_queue_name
        if queue_name is not None and queue_name in self._messages:
            self._messages[queue_name].clear()
            logger.debug(f"Mock RabbitMQ cleared queue '{queue_name}'")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
