"""
RabbitMQ driver module with both real and readonly implementations.

This module provides a unified RabbitMQ client that always connects to a
real RabbitMQ server. In readonly mode (DEBUG env var or readonly=True)
reads are non-destructive: a message is taken one at a time and put back
into the queue (the queue is never drained), while writes go to internal
storage so the server remains unchanged.

ALL read operations use ENFORCED MANUAL (deferred) acknowledgment - there
is no automatic-ack option: receive_json() ALWAYS pulls a message WITHOUT
acknowledging it; the message then stays unacknowledged at the broker (held
by this worker - the channel is limited to exactly one in-flight message via
basic.qos prefetch_count=1) until the processing of its payload has
completed and the worker settles it with ack_pending() or requeue_pending().
In production mode ack_pending() consumes the message (removes it from the
queue) and requeue_pending() puts it back for reprocessing. In readonly
mode BOTH settle calls are non-destructive: the message is put back onto the
head of the queue (reject with requeue), so the queue is never drained. If
a worker crashes before settling the message, the broker requeues it for
another worker instead of losing it. Many workers of the same consumer group
may run in parallel: a busy worker holds its single in-flight message while
idle workers pick up the next ones, and the long heartbeat keeps the
connection alive across slow multi-LLM-call processing.

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
    ALL reads use an enforced manual (deferred) acknowledgment: every
    message pulled with receive_json() stays unacknowledged at the broker
    until the worker settles it with ack_pending()/requeue_pending(), so a
    worker crash requeues the in-flight message instead of losing it.
    In production mode ack_pending() consumes the message (removes it from
    the queue) and requeue_pending() puts it back for reprocessing; in
    readonly mode BOTH settle calls are non-destructive - the message is
    put back onto the head of the queue (the queue is never drained, so
    runs can be repeated) - and writes go to an internal buffer so the
    server remains unchanged.
    Attributes:
        PREFETCH_COUNT: basic.qos limit - exactly one unacknowledged
                        message per worker (channel) at a time.
        HEARTBEAT_TIMEOUT: Connection heartbeat timeout (seconds); long
                        enough to survive multi-LLM-call processing of a
                        single message without AMQP traffic.
        _connected: Whether the client is connected.
        _pending_message: The in-flight message pulled with receive_json(),
                          None when no message is in flight.
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
    # Exactly one unacknowledged message per worker at a time (basic.qos):
    # a busy worker holds its single in-flight message until it is settled,
    # so the broker hands the next message to another idle worker of the
    # same consumer group.
    PREFETCH_COUNT = 1
    # Heartbeat timeout (seconds) of the broker connection. Processing one
    # message involves several LLM calls and can take minutes without any
    # AMQP traffic on the connection, so the heartbeat must outlast a full
    # processing cycle; the acknowledgment itself is still manual (see
    # ack_pending()) and happens as soon as the processing completes.
    HEARTBEAT_TIMEOUT = 300
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
        # Enforced deferred-ack (manual acknowledgment) state: the message
        # pulled with receive_json() that is being processed right now; None
        # when no message is in flight.
        self._pending_message = None

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
        non-destructive (the in-flight message is put back onto the head of
        the queue when settled) and writes are buffered.
        The connection is opened with a long heartbeat (HEARTBEAT_TIMEOUT):
        processing a single message involves several LLM calls and can take
        minutes without any AMQP traffic, so the heartbeat must not kill the
        connection during normal slow processing. The channel is limited to
        exactly PREFETCH_COUNT unacknowledged message(s), and messages are
        acknowledged only after their processing completes (see
        receive_json(), ack_pending() and requeue_pending()), so a worker
        crash leaves the message unacknowledged and the broker requeues it
        instead of losing it. The channel is opened as part of the connect
        routine: if the connection is already alive but the channel was
        closed by the broker or a network failure, connect() re-opens the
        channel on the existing connection (re-applying the prefetch limit)
        instead of failing - it is the single place where the channel is
        (re)established.
        Returns:
            0 on success, -1 on failure (or the client is already connected
            with a live channel).
        """
        if self._connection is not None:
            if self._channel is not None and not self._channel.is_closed:
                logger.error("RabbitMQ ERROR. Client is already opened!")
                return -1
            # The connection is alive but the channel was closed by the
            # broker or a network failure: re-open it on the existing
            # connection and re-apply the prefetch limit
            try :
                self._channel = await self._connection.channel()
                await self._channel.set_qos(prefetch_count=self.PREFETCH_COUNT)
            except Exception as e:
                logger.error(f"RabbitMQ ERROR. Failed to re-open the channel, error: {e}")
                return -1
            self._connected = True
            logger.info(f"RabbitMQ channel re-opened on the existing connection to {self._host}:{self._port} "
                        f"(prefetch={self.PREFETCH_COUNT})")
            return 0
        try:
            # connect_robust automatically handles reconnects
            self._connection = await aio_pika.connect_robust(
                host=self._host,
                port=self._port,
                virtualhost=self._vhost,
                login=self._username,
                password=self._password,
                heartbeat=self.HEARTBEAT_TIMEOUT,
            )
            # Open the worker channel and limit it to exactly
            # PREFETCH_COUNT unacknowledged messages (basic.qos). With many
            # workers of the same consumer group this guarantees that each
            # worker holds at most one message in flight; the next message
            # is delivered only to a worker whose previous message has
            # already been acknowledged or rejected.
            self._channel = await self._connection.channel()
            await self._channel.set_qos(prefetch_count=self.PREFETCH_COUNT)
            self._connected = True
            if self.readonly:
                logger.info(f"RabbitMQ connected to {self._host}:{self._port} (readonly: non-destructive reads, buffered writes)")
            else:
                logger.info(f"RabbitMQ connected to {self._host}:{self._port} "
                            f"(manual ack, prefetch={self.PREFETCH_COUNT}, heartbeat={self.HEARTBEAT_TIMEOUT}s)")
            return 0
        except Exception as e:
            logger.error(f"RabbitMQ ERROR. Failed to connect, error: {e}")
            try :
                if self._connection is not None and not self._connection.is_closed :
                    await self._connection.close()
            except Exception :
                pass
            self._connection = None
            self._channel = None
            self._connected = False
            return -1

    async def disconnect(self) -> None:
        """
        Gracefully disconnect and clean up all resources.
        This method reverses the connection process by closing the connection
        in the correct order and clearing internal state (including the
        readonly write buffer and the in-flight message state).
        """
        if self._pending_message is not None :
            logger.info("RabbitMQ: a message is in flight; it is left unacknowledged "
                        "and will be requeued by the broker on channel close.")
            self._pending_message = None
        if self._channel is not None and not self._channel.is_closed :
            await self._channel.close()
            self._channel = None
        if self._connection is not None and not self._connection.is_closed :
            await self._connection.close()
            self._connection = None
        self._connected = False
        if self.readonly :
            self._messages = {}
            logger.info("RabbitMQ connection closed (readonly).")
        else             :
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
                routing_key=queue_name)
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
        The acknowledgment is ALWAYS deferred (enforced - there is no
        automatic-ack option): the message is pulled without being
        acknowledged and stays unacknowledged at the broker (held by this
        worker under the prefetch=1 limit) until the worker settles it with
        ack_pending() or requeue_pending() once the processing of its
        payload is done. If a worker crashes before settling the message,
        the broker requeues it for another worker instead of losing it.
        In production mode ack_pending() consumes the message (removes it
        from the queue) and requeue_pending() puts it back for reprocessing.
        In readonly mode BOTH settle calls are non-destructive: the message
        is put back onto the head of the queue, so the queue is never
        drained and the run can be repeated.
        Args:
            queue_name: The queue name to receive from.
        Returns:
            Tuple of (status_code, message):
            - (1, message) if message was received (it is now in flight and
              must be settled with ack_pending()/requeue_pending())
            - (0, None) if queue is empty
            - (-1, None) on error (or a message is already in flight and not
              settled yet)
        """
        if self.readonly:
            return await self._receive_json_readonly(queue_name)
        else:
            return await self._receive_json_real(queue_name)

    async def ack_pending(self) -> int:
        """
        Manually acknowledge the in-flight message (pulled with receive_json()
        - every read is deferred): this marks the processing of the message
        as fully completed and removes it from the queue (production mode).
        Call this only AFTER the processing of the message payload is done
        (e.g. the story has been published to the processed-stories queue, or
        the outcome was decided to be unprocessable). If a worker crashes
        before this call, the message stays unacknowledged and the broker
        requeues it for another worker instead of losing it.
        In readonly mode this is a non-destructive "consumption": the message
        is put back onto the head of the queue (reject with requeue), so the
        queue is never drained.
        Returns:
            1 on success, 0 on failure (no message in flight, or the
            acknowledgment could not be sent - the broker then requeues the
            message when the dead connection is detected).
        """
        if self._pending_message is None:
            logger.error("RabbitMQ ERROR. No message in flight to acknowledge!")
            return 0
        message, self._pending_message = self._pending_message, None
        try:
            if self.readonly:
                # Non-destructive read: put the message back onto the head
                await message.reject(requeue=True)
            else:
                await message.ack()
            return 1
        except Exception as e:
            logger.error(f"RabbitMQ ERROR. Failed to acknowledge the in-flight message: {e}")
            # The message is unacknowledged at the broker: if the channel
            # died, the broker requeues it there; re-open the channel for
            # the next pull.
            if self._channel is not None and self._channel.is_closed:
                await self.connect()
            return 0

    async def requeue_pending(self) -> int:
        """
        Reject and requeue the in-flight message (pulled with receive_json()
        - every read is deferred): the message is put back to the head of
        the queue and can be processed again (by this or another worker).

        Use it when the processing of the message payload did not complete
        durably (e.g. publishing the result failed), so the work is not
        lost. In readonly mode this behaves the same as ack_pending(): the
        non-destructive put-back of the head message.

        Returns:
            1 on success, 0 on failure (no message in flight, or the reject
            could not be sent - the broker then requeues the message when
            the dead connection is detected).
        """
        if self._pending_message is None:
            logger.error("RabbitMQ ERROR. No message in flight to requeue!")
            return 0
        message, self._pending_message = self._pending_message, None
        try:
            await message.reject(requeue=True)
            return 1
        except Exception as e:
            logger.error(f"RabbitMQ ERROR. Failed to requeue the in-flight message: {e}")
            if self._channel is not None and self._channel.is_closed:
                await self.connect()
            return 0

    async def _receive_json_real(self, queue_name: Optional[str] = None) -> Tuple[int, Optional[dict]]:
        """Real RabbitMQ implementation of receive_json (enforced manual acknowledgment)."""
        if self._channel is None or self._channel.is_closed :
            logger.error("RabbitMQ error. Client is not connected!")
            return (-1, None)
        if self._pending_message is not None :
            # One message in flight at a time (prefetch=1): the previous one
            # must be settled with ack_pending()/requeue_pending() first.
            logger.error("RabbitMQ ERROR. A message is already in flight (pulled without "
                          "acknowledgment) and not settled yet; call ack_pending() or "
                          "requeue_pending() before pulling another one.")
            return (-1, None)
        if queue_name is None :
            queue_name = self._receive_queue_name
        assert queue_name is not None, "RabbitMQ ERROR. Receiving from unspecified queue!"
        # Create queue handle locally (bypasses broker declare and requires ONLY 'Read' permissions)
        queue = aio_pika.Queue(channel=self._channel, name=queue_name, durable=True,
                               exclusive=False, auto_delete=False, arguments=None)
        try :
            # Fetch a single message WITHOUT acknowledging it (no_ack=False):
            # the broker keeps the message (limiting the channel to
            # PREFETCH_COUNT unacked ones) until it is acknowledged or
            # rejected; fails immediately if the queue is empty or missing
            message = await queue.get(fail=True, no_ack=False)
        except aio_pika.exceptions.QueueEmpty :
            return (0, None)
        except (aio_pika.exceptions.ChannelNotFoundEntity, aio_pika.exceptions.AMQPChannelError, Exception) as fetch_err :
            logger.debug(f"RabbitMQ: queue '{queue_name}' fetch failed: {fetch_err}")
            # Re-open channel if RabbitMQ closed it on failure
            if self._channel.is_closed :
                await self.connect()
            return (0, None)
        # The message is in flight (unacknowledged) from here on: the worker
        # must settle it with ack_pending() or requeue_pending() once the
        # processing of its payload completes
        self._pending_message = message
        try :
            payload = json.loads(message.body.decode("utf-8"))
        except json.JSONDecodeError as e :
            logger.error(f"RabbitMQ ERROR. Failed to decode JSON payload: {e}")
            # Drop the poison message: requeueing it would loop forever
            await self.ack_pending()
            return (-1, None)
        return (1, payload)

    async def _receive_json_readonly(self, queue_name: Optional[str] = None) -> Tuple[int, Optional[dict]]:
        """
        Readonly implementation of receive_json (enforced manual acknowledgment).
        Reads one message from the real broker WITHOUT removing it: the
        message is pulled unacknowledged and held in flight; settling it
        (ack_pending()/requeue_pending()) is non-destructive - the message is
        put back onto the head of the queue - so the queue is never drained
        and runs can be repeated.
        """
        if self._channel is None or self._channel.is_closed:
            logger.error("RabbitMQ error. Client is not connected!")
            return (-1, None)
        if self._pending_message is not None :
            # One message in flight at a time (prefetch=1): the previous one
            # must be settled with ack_pending()/requeue_pending() first.
            logger.error("RabbitMQ ERROR. A message is already in flight (pulled without "
                          "acknowledgment) and not settled yet; call ack_pending() or "
                          "requeue_pending() before pulling another one.")
            return (-1, None)
        if queue_name is None :
            queue_name = self._receive_queue_name
        assert queue_name is not None, "RabbitMQ ERROR. Receiving from unspecified queue!"
        # Instantiate queue locally in Python without sending AMQP Queue.Declare frame
        queue = aio_pika.Queue(channel=self._channel, name=queue_name, durable=True,
                               exclusive=False, auto_delete=False, arguments=None)
        try :
            # Fetch a single message WITHOUT acknowledging it (no_ack=False):
            # it is put back onto the head of the queue when settled
            message = await queue.get(fail=True, no_ack=False)
        except aio_pika.exceptions.QueueEmpty :
            return (0, None)
        except (aio_pika.exceptions.ChannelNotFoundEntity, aio_pika.exceptions.AMQPChannelError) as fetch_err :
            logger.debug(f"Readonly RabbitMQ: queue '{queue_name}' fetch failed: {fetch_err}")
            # Re-open channel if RabbitMQ closed it on failure
            if self._channel.is_closed:
                await self.connect()
            return (0, None)
        except Exception as e :
            logger.error(f"RabbitMQ ERROR. Failed to get message from queue '{queue_name}': {e}")
            if self._channel.is_closed:
                await self.connect()
            return (-1, None)
        # Non-destructive deferred read: the message is held in flight (it is
        # put back onto the head of the queue when settled)
        self._pending_message = message
        try :
            payload = json.loads(message.body.decode("utf-8"))
            logger.debug(f"Readonly RabbitMQ read message from queue '{queue_name}': {payload}")
            return (1, payload)
        except json.JSONDecodeError as e :
            logger.error(f"Readonly RabbitMQ ERROR. Failed to decode JSON payload: {e}")
            # Put the poison message back (non-destructive); the read-only
            # consumer treats it as unprocessable
            await self.requeue_pending()
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
