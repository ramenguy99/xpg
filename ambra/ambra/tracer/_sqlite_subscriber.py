"""SQLite storage subscriber."""

import queue
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ._queue import DEFAULT_MAX_QUEUE_SIZE, QueueFullPolicy
from ._serializer import create_table, insert_serialized, serialize_kwargs
from ._subscriber import Subscriber


@dataclass
class SqliteConfig:
    journal_mode: Optional[str] = "wal"
    synchronous: Optional[str] = "normal"
    wal_autocheckpoint: Optional[int] = 16384
    page_size: Optional[int] = None
    cache_size: Optional[int] = None


class SqliteSubscriber(Subscriber):
    """Subscriber that stores trace events in SQLite database.

    Uses serialize_kwargs() to serialize data before enqueueing, ensuring
    callers can safely modify their data after trace() returns. Tables are
    created on-demand when the first trace with that identifier arrives.

    Args:
        db_path: Path to the SQLite database file, or ":memory:" for in-memory.
        patterns: List of regex patterns to match trace identifiers.
        auto_commit: Whether to commit after each trace event.
        queue_size: Maximum size of the internal event queue.
        queue_full_policy: Policy when queue is full (WAIT or DROP).

    """

    def __init__(
        self,
        db_path: Union[str, Path],
        patterns: Optional[List[str]] = None,
        queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        queue_full_policy: QueueFullPolicy = QueueFullPolicy.DROP,
        sqlite_config: Optional[SqliteConfig] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(patterns, verbose)

        self._db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._known_tables: Set[str] = set()
        self._db_lock = threading.Lock()

        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        self._queue: queue.Queue[Optional[Tuple[str, Any]]] = queue.Queue(maxsize=queue_size)
        self._queue_full_policy = queue_full_policy

        self._sqlite_config = SqliteConfig() if sqlite_config is None else sqlite_config

        # Stats
        self._dropped_events = 0
        self._enqueued_events = 0

    def start(self) -> None:
        """Start the subscriber and open database connection."""
        if self._running:
            return
        self._running = True

        # Open connection before starting worker thread
        # Connection is opened in the main thread but used in worker thread.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)

        # Configure database
        if self._sqlite_config.journal_mode is not None:
            self._conn.execute(f"PRAGMA journal_mode={self._sqlite_config.journal_mode}")
        if self._sqlite_config.synchronous is not None:
            self._conn.execute(f"PRAGMA synchronous={self._sqlite_config.synchronous}")
        if self._sqlite_config.wal_autocheckpoint is not None:
            self._conn.execute(f"PRAGMA wal_autocheckpoint={self._sqlite_config.wal_autocheckpoint}")
        if self._sqlite_config.page_size is not None:
            self._conn.execute(f"PRAGMA page_size={self._sqlite_config.page_size}")
        if self._sqlite_config.cache_size is not None:
            self._conn.execute(f"PRAGMA cache_size={self._sqlite_config.cache_size}")

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop the subscriber and close database connection."""
        if not self._running:
            return
        self._running = False

        # Send sentinel to wake up worker
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._worker_thread is not None:
            self._worker_thread.join()
            self._worker_thread = None

        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def on_trace(self, identifier: str, kwargs: Dict[str, Any]) -> None:
        """Serialize kwargs to SQLite-compatible format.

        Args:
            identifier: The trace identifier (not used for serialization).
            kwargs: The trace data.

        Returns:
            Dictionary with values converted to SQLite-compatible types.

        """
        serialized = serialize_kwargs(**kwargs)

        # Enqueue for background processing
        if self._queue_full_policy == QueueFullPolicy.WAIT:
            self._queue.put((identifier, serialized))
            self._enqueued_events += 1
        else:
            # DROP policy
            try:
                self._queue.put_nowait((identifier, serialized))
                self._enqueued_events += 1
            except queue.Full:
                # Silently drop
                self._dropped_events += 1

    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        assert self._conn is not None

        while self._running:
            item = self._queue.get()

            if item is None:
                # Sentinel received, exit loop
                break

            identifier, serialized = item

            # Skip messages with no name
            if not identifier:
                continue

            # Escape double quotes by doubling them for SQLite quoted identifiers
            table_name = identifier.replace('"', '""')

            try:
                with self._db_lock:
                    if table_name not in self._known_tables:
                        create_table(self._conn, table_name, serialized)
                        self._known_tables.add(table_name)

                    insert_serialized(self._conn, table_name, serialized)
                    self._conn.commit()
            except Exception:  # noqa: S110
                # Silently ignore handler errors
                pass
