"""
RabbitMQ driver module with both real and readonly implementations.

This module provides a unified RabbitMQ client that always connects to a
real RabbitMQ server. In readonly mode (DEBUG env var or readonly=True)
reads are non-destructive: a message is taken one at a time and put back
into the queue (the queue is never drained), while writes go to internal
storage so the server remains unchanged.

Usage:
    # Production (default)
    rabbit_client = RabbitMQClient(host="localhost", port=5672, vhost="/")

    # Readonly mode
    rabbit_client = RabbitMQClient(readonly=True)

    # Or use environment variable DEBUG
    import os
    os.environ['DEBUG'] = 'true'  # Will automatically enable readonly mode
"""

import logging
import json
from typing import Optional, List, Tuple, Dict, Any
from collections import deque
import aio_pika
#from aio_pika.exceptions import QueueEmpty, ChannelNotFoundEntity, AMQPChannelError

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
    Unified RabbitMQ client that always connects to a real RabbitMQ server.

    In readonly mode, reads are non-destructive: each received message is
    requeued (the queue is never drained, so runs can be repeated), and
    writes go to an internal buffer so the server remains unchanged.

    Attributes:
        _connected: Whether the client is connected.
        _messages: Dictionary mapping queue names to buffered messages (readonly writes).
        _publish_count: Counter for number of buffered messages (readonly writes).

    Example:
        # Production
        async with RabbitMQClient(readonly=False) as client:
            await client.connect()
            await client.publish_json({"data": "message"})

        # Readonly mode
        async with RabbitMQClient(readonly=True) as client:
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
        readonly: Optional[bool] = None
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
            readonly: If True, enables readonly mode (real connection,
                      non-destructive reads, buffered writes). If False, uses
                      full real RabbitMQ semantics. If None, checks the DEBUG
                      environment variable.
        """
        # Determine if we should use readonly mode
        if readonly is None:
            import os
            readonly = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')

        self.readonly = readonly
        self._host = host
        self._port = port
        self._vhost = vhost
        self._publish_queue_name = publish_queue_name
        self._receive_queue_name = receive_queue_name
        self._username = username
        self._password = password
        self._connection = None
        self._channel = None
        
        # Readonly-mode attributes (write buffer + connection flag)
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

    async def connect(self) -> int:
        """
        Connect to the real RabbitMQ server.

        In readonly mode the same real connection is used, but reads become
        non-destructive (messages are requeued) and writes are buffered.

        Returns:
            0 on success, -1 on failure.
        """
        if self._connection is not None:
            logger.error("RabbitMQ ERROR. Client is already opened!")
            return -1
        try:
            # connect_robust automatically handles reconnects
            self._connection = await aio_pika.connect_robust(
                host=self._host,
                port=self._port,
                virtualhost=self._vhost,
                login=self._username,
                password=self._password
            )
            self._channel = await self._connection.channel()
            self._connected = True
            if self.readonly:
                logger.info(f"RabbitMQ connected to {self._host}:{self._port} (readonly: non-destructive reads, buffered writes)")
            else:
                logger.info(f"RabbitMQ connected to {self._host}:{self._port}")
            return 0
        except Exception as e:
            logger.error(f"RabbitMQ ERROR. Failed to connect, error: {e}")
            self._connection = None
            self._channel = None
            self._connected = False
            return -1

    async def disconnect(self) -> None:
        """
        Gracefully disconnect and clean up all resources.

        This method reverses the connection process by closing the connection
        in the correct order and clearing internal state (including the
        readonly write buffer).
        """
        if self._channel is not None and not self._channel.is_closed:
            await self._channel.close()
            self._channel = None
        if self._connection is not None and not self._connection.is_closed:
            await self._connection.close()
            self._connection = None
        self._connected = False
        if self.readonly:
            self._messages = {}
            logger.info("RabbitMQ connection closed (readonly).")
        else:
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
        Publish a JSON message.

        In readonly mode the message is buffered internally (the real server
        is not modified); in production mode it is published to the broker.

        Args:
            data: The JSON data to publish.
            queue_name: The queue name to publish to.

        Returns:
            1 on success, 0 on failure.
        """
        if self.readonly:
            return await self._publish_json_buffered(data, queue_name)
        else:
            return await self._publish_json_real(data, queue_name)

    async def _publish_json_real(self, data: dict, queue_name: Optional[str] = None) -> int:
        """Real RabbitMQ implementation of publish_json."""
        if self._channel is None or self._channel.is_closed:
            logger.error("RabbitMQ ERROR. Client is not connected!")
            return 0
        try:
            message_bytes = json.dumps(data).encode("utf-8")
            if queue_name is None:
                queue_name = self._publish_queue_name
            assert queue_name is not None, "RabbitMQ ERROR. Publishing into unspecified queue!"
            await self._channel.default_exchange.publish(
                aio_pika.Message(body=message_bytes, delivery_mode=aio_pika.DeliveryMode.PERSISTENT, content_type="application/json"),
                routing_key=queue_name
            )
            return 1
        except Exception as e:
            logger.error(f"RabbitMQ ERROR. Failed to publish a message {e}")
            return 0

    async def _publish_json_buffered(self, data: dict, queue_name: Optional[str] = None) -> int:
        """Readonly implementation of publish_json (buffered internally, the server is not modified)."""
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
            logger.debug(f"Readonly RabbitMQ buffered message to queue '{queue_name}': {data}")
            return 1
        except Exception as e:
            logger.error(f"RabbitMQ ERROR. Failed to publish a message {e}")
            return 0

    async def receive_json(self, queue_name: Optional[str] = None) -> Tuple[int, Optional[dict]]:
        """
        Receive a JSON message, one at a time.

        In readonly mode the message is requeued right after being read, so
        the queue is never drained and the run can be repeated.

        Args:
            queue_name: The queue name to receive from.

        Returns:
            Tuple of (status_code, message):
            - (1, message) if message was received
            - (0, None) if queue is empty (or does not exist, readonly mode)
            - (-1, None) on error
        """
        if self.readonly:
            return await self._receive_json_readonly(queue_name)
        else:
            return await self._receive_json_real(queue_name)

    async def _receive_json_real(self, queue_name: Optional[str] = None) -> Tuple[int, Optional[dict]]:
        """Real RabbitMQ implementation of receive_json."""
        if self._channel is None or self._channel.is_closed :
            logger.error("RabbitMQ error. Client is not connected!")
            return (-1, None)
        if queue_name is None :
            queue_name = self._receive_queue_name
        assert queue_name is not None, "RabbitMQ ERROR. Receiving from unspecified queue!"
        # Create queue handle locally (bypasses broker declare and requires ONLY 'Read' permissions)
        queue = aio_pika.Queue(channel=self._channel, name=queue_name, durable=True,
                               exclusive=False, auto_delete=False, arguments=None)
        try :
            # Fetch single message; fails immediately if empty or missing
            message = await queue.get(fail=True)
        except aio_pika.exceptions.QueueEmpty :
            return (0, None)
        except (aio_pika.exceptions.ChannelNotFoundEntity, aio_pika.exceptions.AMQPChannelError, Exception) as fetch_err :
            logger.debug(f"RabbitMQ: queue '{queue_name}' fetch failed: {fetch_err}")
            # Re-open channel if RabbitMQ closed it on failure
            if self._channel.is_closed :
                self._channel = await self._connection.channel()
            return (0, None)
        # Acknowledge and process message upon successful JSON decode
        async with message.process() :
            try :
                payload = json.loads(message.body.decode("utf-8"))
                return (1, payload)
            except json.JSONDecodeError as e :
                logger.error(f"RabbitMQ ERROR. Failed to decode JSON payload: {e}")
                return (-1, None)

    async def _receive_json_readonly(self, queue_name: Optional[str] = None) -> Tuple[int, Optional[dict]]:
        """
        Readonly implementation of receive_json.

        Reads one message from the real broker WITHOUT removing it: the
        message is retrieved and immediately requeued via reject(requeue=True),
        ensuring the queue is non-destructively read and never drained.
        """
        if self._channel is None or self._channel.is_closed:
            logger.error("RabbitMQ error. Client is not connected!")
            return (-1, None)
        if queue_name is None :
            queue_name = self._receive_queue_name
        assert queue_name is not None, "RabbitMQ ERROR. Receiving from unspecified queue!"
        # Instantiate queue locally in Python without sending AMQP Queue.Declare frame
        queue = aio_pika.Queue(channel=self._channel, name=queue_name, durable=True,
                               exclusive=False, auto_delete=False, arguments=None)
        try :
            message = await queue.get(fail=True)
        except aio_pika.exceptions.QueueEmpty :
            return (0, None)
        except (aio_pika.exceptions.ChannelNotFoundEntity, aio_pika.exceptions.AMQPChannelError) as fetch_err :
            logger.debug(f"Readonly RabbitMQ: queue '{queue_name}' fetch failed: {fetch_err}")
            # Re-open channel if RabbitMQ closed it on failure
            if self._channel.is_closed:
                self._channel = await self._connection.channel()
            return (0, None)
        except Exception as e :
            logger.error(f"RabbitMQ ERROR. Failed to get message from queue '{queue_name}': {e}")
            if self._channel.is_closed:
                self._channel = await self._connection.channel()
            return (-1, None)
        # Non-destructive read: put message back onto the queue immediately
        await message.reject(requeue=True)
        try :
            payload = json.loads(message.body.decode("utf-8"))
            logger.debug(f"Readonly RabbitMQ read message from queue '{queue_name}': {payload}")
            return (1, payload)
        except json.JSONDecodeError as e :
            logger.error(f"RabbitMQ ERROR. Failed to decode JSON payload: {e}")
            return (-1, None)

    # Readonly-buffer helpers (for testing purposes)
    async def get_queue_size(self, queue_name: Optional[str] = None) -> int:
        """Get the number of messages in a queue (for testing purposes)."""
        if queue_name is None:
            queue_name = self._publish_queue_name
        if queue_name is None or queue_name not in self._messages:
            return 0
        return len(self._messages[queue_name])

    async def clear_queue(self, queue_name: Optional[str] = None) -> None:
        """Clear all buffered messages from a queue (for testing purposes)."""
        if queue_name is None:
            queue_name = self._publish_queue_name
        if queue_name is not None and queue_name in self._messages:
            self._messages[queue_name].clear()
            logger.debug(f"Readonly RabbitMQ cleared buffer of queue '{queue_name}'")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
