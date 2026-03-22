"""Tracing library with pub/sub pattern and configurable subscribers.

This library enables runtime tracing of data through a pub/sub pattern with configurable
subscribers for storage (SQLite) and network transmission (over TCP).

Basic usage:
    from ambra import tracer

    tracer.init_tracer()
    sub = tracer.SqliteSubscriber("storage.db", patterns=["example\\..*"])
    tracer.register_subscriber(sub)

    tracer.trace("example.basic", numbers=[1, 2, 3, 4])

    tracer.close_tracer()
"""

__all__ = [
    "HEADER_SIZE",
    "ConfigError",
    "MessageType",
    "QueueFullPolicy",
    "SqliteConfig",
    "SqliteSubscriber",
    "Subscriber",
    "TcpSubscriber",
    "TraceEmitter",
    "Tracer",
    "close_tracer",
    "create_table",
    "decode_disable_pattern",
    "decode_disable_pattern_response",
    "decode_enable_pattern",
    "decode_enable_pattern_response",
    "decode_header",
    "decode_list_tracepoints_response",
    "decode_trace_event",
    "encode_disable_pattern",
    "encode_disable_pattern_response",
    "encode_enable_pattern",
    "encode_enable_pattern_response",
    "encode_list_tracepoints",
    "encode_list_tracepoints_response",
    "encode_trace_event",
    "from_binary",
    "from_sqlite",
    "from_sqlite_all",
    "from_sqlite_value",
    "init_from_config",
    "init_tracer",
    "insert_serialized",
    "load_config",
    "load_config_from_string",
    "register_subscriber",
    "register_subscriber_type",
    "serialize_kwargs",
    "to_binary",
    "to_binary_into",
    "to_sqlite",
    "trace",
    "trace_if",
    "unregister_subscriber",
    "will_trace",
]

# Core classes
# Configuration
from ._config import (
    ConfigError,
    init_from_config,
    load_config,
    load_config_from_string,
    register_subscriber_type,
)

# Global tracer functions
from ._global import (
    close_tracer,
    init_tracer,
    register_subscriber,
    trace,
    trace_if,
    unregister_subscriber,
    will_trace,
)

# Protocol constants and functions
from ._protocol import (
    HEADER_SIZE,
    MessageType,
    decode_disable_pattern,
    decode_disable_pattern_response,
    decode_enable_pattern,
    decode_enable_pattern_response,
    decode_header,
    decode_list_tracepoints_response,
    decode_trace_event,
    encode_disable_pattern,
    encode_disable_pattern_response,
    encode_enable_pattern,
    encode_enable_pattern_response,
    encode_list_tracepoints,
    encode_list_tracepoints_response,
    encode_trace_event,
)

# Queue policy
from ._queue import QueueFullPolicy

# Serializer functions
from ._serializer import (
    create_table,
    from_binary,
    from_sqlite,
    from_sqlite_all,
    from_sqlite_value,
    insert_serialized,
    serialize_kwargs,
    to_binary,
    to_binary_into,
    to_sqlite,
)
from ._sqlite_subscriber import SqliteConfig, SqliteSubscriber
from ._subscriber import Subscriber
from ._tcp_subscriber import TcpSubscriber
from ._tracer import TraceEmitter, Tracer
