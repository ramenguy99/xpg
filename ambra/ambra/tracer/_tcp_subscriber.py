"""TCP subscriber."""

import asyncio
import logging
import struct
import threading
from typing import Any, Dict, List, Optional

from ._protocol import (
    HEADER_SIZE,
    MessageType,
    decode_disable_pattern,
    decode_enable_pattern,
    decode_header,
    encode_disable_pattern_response,
    encode_enable_pattern_response,
    encode_list_tracepoints_response,
    encode_trace_event,
)
from ._queue import DEFAULT_MAX_QUEUE_SIZE, QueueFullPolicy
from ._subscriber import Subscriber

logger = logging.getLogger(__name__)


class TcpSubscriber(Subscriber):
    """Subscriber that sends trace events over TCP.

    Uses an asyncio event loop on a separate thread for TCP I/O.

    Args:
        host: Server hostname or IP address.
        port: Server port number.
        patterns: List of regex patterns to match trace identifiers.
        connect_timeout: Timeout for connection attempts in seconds.
        reconnect_delay: Delay between reconnection attempts in seconds.
        queue_size: Maximum size of the internal event queue.
        queue_full_policy: Policy when queue is full (WAIT or DROP).

    """

    def __init__(
        self,
        host: str,
        port: int,
        patterns: Optional[List[str]] = None,
        connect_timeout: float = 5.0,
        reconnect_delay: float = 1.0,
        queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        queue_full_policy: QueueFullPolicy = QueueFullPolicy.DROP,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            patterns=patterns,
            verbose=verbose,
        )
        self._host = host
        self._port = port
        self._connect_timeout = connect_timeout
        self._reconnect_delay = reconnect_delay
        self._queue_size = queue_size
        self._queue_full_policy = queue_full_policy

        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._send_queue: asyncio.Queue[Optional[bytes]]  # created in _asyncio_loop
        self._running = False
        self._asyncio_thread: Optional[threading.Thread] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = threading.Event()
        self._main_task: Optional[asyncio.Task[None]] = None

        # Stats
        self._dropped_events = 0
        self._enqueued_events = 0
        self._serialized_bytes = 0

    def check_active(self, identifier: str) -> bool:
        # Skip tracing if not connected
        if not self._connected.is_set():
            return False

        return super().check_active(identifier)

    def start(self) -> None:
        """Start the subscriber, asyncio thread, and base class worker."""
        if self._running:
            return
        self._running = True

        # Crate main stask
        self._main_task = self._loop.create_task(self._async_main())

        # Start asyncio thread
        self._asyncio_thread = threading.Thread(target=self._asyncio_loop, daemon=True)
        self._asyncio_thread.start()

    def stop(self) -> None:
        """Stop the subscriber and asyncio thread."""
        if not self._running:
            return
        self._running = False

        # Signal asyncio to shutdown gracefully
        if self._loop is not None and self._main_task is not None:
            self._loop.call_soon_threadsafe(self._main_task.cancel)

        if self._asyncio_thread is not None:
            self._asyncio_thread.join()
            self._asyncio_thread = None

        self._main_task = None

    def _drain_queue(self) -> None:
        """Drain all pending messages from the send queue."""
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:  # noqa: PERF203
                break

    def _asyncio_loop(self) -> None:
        """Run the asyncio event loop on a dedicated thread."""
        assert self._main_task is not None

        asyncio.set_event_loop(self._loop)
        # Create queue on the loop thread so it binds to self._loop (required for Python 3.8)
        self._send_queue = asyncio.Queue(self._queue_size)

        try:
            self._loop.run_until_complete(self._main_task)
        except asyncio.CancelledError:
            pass
        finally:
            # Cancel any remaining tasks
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

    async def _async_main(self) -> None:
        """Main async coroutine for TCP handling."""
        try:
            while self._running:
                try:
                    # Attempt to connect
                    self._reader, self._writer = await asyncio.wait_for(
                        asyncio.open_connection(self._host, self._port),
                        timeout=self._connect_timeout,
                    )

                    # Drain queue before signaling connected and allowing new
                    # messages to come in.
                    #
                    # The queue is already drained on stop to free up memory,
                    # but there is a window of time after the sender is stopped
                    # and the connection is set to inactive where new messages
                    # could have come in.  Instead of synchronizing there, we
                    # ensure the queue is completely clear here to avoid sending
                    # stale messages to a just connected server.
                    self._drain_queue()

                    # Signal connected
                    self._connected.set()

                    # Run connection
                    receiver_task = asyncio.create_task(self._receive_loop())
                    sender_task = asyncio.create_task(self._send_loop())
                    try:
                        await asyncio.wait(
                            [receiver_task, sender_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        for task in [receiver_task, sender_task]:
                            if not task.done():
                                task.cancel()
                                try:
                                    await task
                                except asyncio.CancelledError:
                                    pass

                        # Drain queue after stopping sender.
                        self._drain_queue()
                except (OSError, asyncio.TimeoutError, ConnectionError) as e:
                    if self._verbose:
                        logger.debug("TCP connection error: %r", e)
                finally:
                    self._connected.clear()
                    await self._disconnect()

                if self._running:
                    await asyncio.sleep(self._reconnect_delay)
        except asyncio.CancelledError:
            self._connected.clear()
            await self._disconnect()
            if self._verbose:
                logger.debug("TCP connection loop cancelled")
            raise

    async def _disconnect(self) -> None:
        """Close TCP connection."""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:  # noqa: S110
                pass
            self._writer = None
        self._reader = None

    async def _receive_loop(self) -> None:
        """Receive and process incoming messages."""
        if self._reader is None:
            return

        while self._running:
            try:
                header = await self._reader.readexactly(HEADER_SIZE)
            except (asyncio.IncompleteReadError, ConnectionError):
                return

            msg_type, value_length = decode_header(header)

            if value_length > 0:
                try:
                    value = await self._reader.readexactly(value_length)
                except (asyncio.IncompleteReadError, ConnectionError):
                    return
            else:
                value = b""

            if msg_type == MessageType.LIST_TRACEPOINTS:
                identifiers = list(self.get_active_tracepoints())
                response = encode_list_tracepoints_response(identifiers)
                await self._enqueue_send(response)

            elif msg_type == MessageType.ENABLE_PATTERN:
                pattern = decode_enable_pattern(value)
                try:
                    self.add_pattern(pattern)
                    response = encode_enable_pattern_response(True)
                except Exception:
                    response = encode_enable_pattern_response(False)
                await self._enqueue_send(response)

            elif msg_type == MessageType.DISABLE_PATTERN:
                pattern = decode_disable_pattern(value)
                try:
                    self.remove_pattern(pattern)
                    response = encode_disable_pattern_response(True)
                except Exception:
                    response = encode_disable_pattern_response(False)
                await self._enqueue_send(response)

    async def _send_loop(self) -> None:
        """Send queued messages to server."""
        assert self._writer is not None

        try:
            self._writer.write(b"AMBR")
            name = "tracer"
            self._writer.write(struct.pack("<I", len(name)))
            self._writer.write(name.encode("utf-8"))
            await self._writer.drain()
        except ConnectionError:
            return

        while self._running:
            data = await self._send_queue.get()

            if data is None:
                return

            try:
                self._writer.write(data)
                await self._writer.drain()
            except ConnectionError:
                return

    async def _enqueue_send(self, data: bytes, wait: bool = True) -> None:
        """Enqueue data for sending (called from asyncio context)."""
        if wait:
            await self._send_queue.put(data)
            # NOTE: currently this counts responses as well
            self._enqueued_events += 1
        else:
            try:
                self._send_queue.put_nowait(data)
                self._enqueued_events += 1
            except asyncio.QueueFull:
                # Silently drop
                self._dropped_events += 1

    def on_trace(self, identifier: str, kwargs: Dict[str, Any]) -> None:
        """Serialize trace data and enqueue for sending.

        Args:
            identifier: The trace identifier.
            kwargs: The trace data.

        """
        serialized = encode_trace_event(identifier, **kwargs)
        self._serialized_bytes += len(serialized)
        asyncio.run_coroutine_threadsafe(
            self._enqueue_send(serialized, self._queue_full_policy == QueueFullPolicy.WAIT), self._loop
        ).result()

    def wait_connected(self, timeout: Optional[float] = None) -> bool:
        """Wait for connection to be established."""
        return self._connected.wait(timeout=timeout)

    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected.is_set()
