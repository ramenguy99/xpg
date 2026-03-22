"""Core Tracer class for managing subscribers and dispatching traces."""

import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from ._subscriber import Subscriber


class TraceEmitter:
    """Emitter for a specific trace event with pre-matched subscribers.

    Captures subscribers that match a specific identifier, allowing
    efficient emission without re-checking patterns.
    """

    __slots__ = ("_identifier", "_subscribers")

    def __init__(self, identifier: str, subscribers: "List[Subscriber]") -> None:
        self._identifier = identifier
        self._subscribers = subscribers

    def trace(self, **kwargs: Any) -> None:
        """Emit the trace event to the captured subscribers.

        Args:
            **kwargs: Data to include in the trace event.

        """
        for subscriber in self._subscribers:
            subscriber.on_trace(self._identifier, kwargs)


class Tracer:
    """Central tracer that dispatches trace events to registered subscribers.

    Thread-safe: subscriber list is copied under lock, iteration happens outside lock.
    """

    def __init__(self) -> None:
        self._subscribers: List[Subscriber] = []
        self._lock = threading.Lock()

    def register_subscriber(self, subscriber: "Subscriber") -> None:
        """Register a subscriber to receive trace events.

        Args:
            subscriber: The subscriber to register.

        """
        with self._lock:
            if subscriber not in self._subscribers:
                self._subscribers.append(subscriber)
                subscriber.start()

    def unregister_subscriber(self, subscriber: "Subscriber") -> None:
        """Unregister a subscriber.

        Args:
            subscriber: The subscriber to unregister.

        """
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)
                subscriber.stop()

    def trace(self, identifier: str, **kwargs: Any) -> None:
        """Dispatch a trace event to all registered subscribers.

        Args:
            identifier: The trace identifier.
            **kwargs: Data to include in the trace event.

        """
        # Copy subscriber list under lock
        with self._lock:
            for subscriber in self._subscribers:
                if subscriber.check_active(identifier):
                    subscriber.on_trace(identifier, kwargs)

    def get_subscribers(self) -> "List[Subscriber]":
        """Get list of registered subscribers.

        Returns:
            Copy of the subscriber list.

        """
        with self._lock:
            return self._subscribers.copy()

    def trace_if(
        self,
        identifier: str,
        get_kwargs: Callable[[], Dict[str, Any]],
    ) -> None:
        """Dispatch a trace event only if there are matching subscribers.

        The get_kwargs callable is only invoked if there are subscribers
        that match the identifier, enabling lazy evaluation of expensive
        trace data.

        Args:
            identifier: The trace identifier.
            get_kwargs: Callable that returns the trace data dict.

        Example:
            tracer.trace_if("event", lambda: {"data": expensive_computation()})

        """
        with self._lock:
            if not self._subscribers:
                return
            matching = [s for s in self._subscribers if s.check_active(identifier)]
            if not matching:
                return
            kwargs = get_kwargs()
            for subscriber in matching:
                subscriber.on_trace(identifier, kwargs)

    def will_trace(self, identifier: str) -> Optional[TraceEmitter]:
        """Check if tracing should occur and return an emitter if so.

        Returns an emitter object if there are interested subscribers,
        or None otherwise. The emitter captures matching subscribers,
        avoiding repeated iteration. Designed for use with the walrus
        operator for lazy tracing.

        Args:
            identifier: The trace identifier.

        Returns:
            A TraceEmitter with a trace() method, or None.

        Example:
            if t := tracer.will_trace("event"):
                t.trace(data=expensive_computation())

        """
        with self._lock:
            if not self._subscribers:
                return None
            matching = [s for s in self._subscribers if s.check_active(identifier)]
            if not matching:
                return None
            return TraceEmitter(identifier, matching)

    def close(self) -> None:
        """Close the tracer and stop all subscribers.

        Stops all registered subscribers and removes them from the tracer.
        """
        with self._lock:
            for subscriber in self._subscribers:
                subscriber.stop()
            self._subscribers.clear()
