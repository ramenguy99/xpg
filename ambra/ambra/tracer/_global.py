"""Global tracer instance and module-level convenience functions."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from ._tracer import TraceEmitter, Tracer

if TYPE_CHECKING:
    from ._subscriber import Subscriber


_global_tracer: Optional[Tracer] = None


def init_tracer() -> Tracer:
    """Initialize the global tracer instance.

    This function is idempotent - calling it multiple times returns
    the same tracer instance.

    Returns:
        The global Tracer instance.

    """
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def register_subscriber(subscriber: "Subscriber") -> None:
    """Register a subscriber with the global tracer.

    Args:
        subscriber: The subscriber to register.

    """
    if _global_tracer is not None:
        _global_tracer.register_subscriber(subscriber)


def unregister_subscriber(subscriber: "Subscriber") -> None:
    """Unregister a subscriber from the global tracer.

    Args:
        subscriber: The subscriber to unregister.

    """
    if _global_tracer is not None:
        _global_tracer.unregister_subscriber(subscriber)


def close_tracer() -> None:
    """Close the global tracer and stop all subscribers.

    Stops all registered subscribers. The tracer can be reused after
    closing by registering new subscribers.
    """
    if _global_tracer is not None:
        _global_tracer.close()


def trace(identifier: str, **kwargs: Any) -> None:
    """Dispatch a trace event using the global tracer.

    Args:
        identifier: The trace identifier.
        **kwargs: Data to include in the trace event.

    Raises:
        TracerNotInitializedError: If init_tracer() has not been called.

    """
    if _global_tracer is not None:
        _global_tracer.trace(identifier, **kwargs)


def trace_if(
    identifier: str,
    get_kwargs: Callable[[], Dict[str, Any]],
) -> None:
    """Dispatch a trace event only if there are subscribers.

    The get_kwargs callable is only invoked if there are registered
    subscribers, enabling lazy evaluation of expensive trace data.

    Args:
        identifier: The trace identifier.
        get_kwargs: Callable that returns the trace data dict.

    Example:
        trace_if("event", lambda: {"data": expensive_computation()})

    """
    if _global_tracer is not None:
        _global_tracer.trace_if(identifier, get_kwargs)


def will_trace(identifier: str) -> Optional[TraceEmitter]:
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
        if t := will_trace("event"):
            t.trace(data=expensive_computation())

    """
    if _global_tracer is not None:
        return _global_tracer.will_trace(identifier)
    else:
        return None
