"""TCP protocol message definitions.

All messages use the unified TLV (Type-Length-Value) format:
    [4B type][4B format][8B length (uint64)][value bytes]
All integers are little-endian.

Message Types:
    0x01 TRACE_EVENT: client -> server
         Value: [8B identifier_len][identifier UTF-8][to_binary(**kwargs)]
    0x02 LIST_TRACEPOINTS: server -> client
         Value: (empty, length=0)
    0x03 LIST_TRACEPOINTS_RESPONSE: client -> server
         Value: [8B count][for each: [8B len][identifier UTF-8]]
    0x04 ENABLE_PATTERN: server -> client
         Value: [pattern UTF-8]
    0x05 ENABLE_PATTERN_RESPONSE: client -> server
         Value: [1B status: 0=ok, 1=error]
    0x06 DISABLE_PATTERN: server -> client
         Value: [pattern UTF-8]
    0x07 DISABLE_PATTERN_RESPONSE: client -> server
         Value: [1B status: 0=ok, 1=error]
"""

import io
import struct
from enum import IntEnum
from typing import Any, List, Tuple, Union

from ._serializer import to_binary_into


class MessageType(IntEnum):
    """Protocol message types."""

    TRACE_EVENT = (1 << 21) + 0x01
    LIST_TRACEPOINTS = (1 << 21) + 0x02
    LIST_TRACEPOINTS_RESPONSE = (1 << 21) + 0x03
    ENABLE_PATTERN = (1 << 21) + 0x04
    ENABLE_PATTERN_RESPONSE = (1 << 21) + 0x05
    DISABLE_PATTERN = (1 << 21) + 0x06
    DISABLE_PATTERN_RESPONSE = (1 << 21) + 0x07


# Header size: 4 byte type + 4 byte format + 8 bytes length
HEADER_SIZE = 16


def _write_header(buf: io.BytesIO, msg_type: MessageType) -> int:
    """Write message header with placeholder length, return position of length field."""
    buf.write(struct.pack("<II", msg_type, 0))  # Message type and format
    length_pos = buf.tell()
    buf.write(struct.pack("<Q", 0))  # Placeholder
    return length_pos


def _finish_message(buf: io.BytesIO, length_pos: int) -> bytes:
    """Fix up the length field and return the complete message."""
    end_pos = buf.tell()
    value_length = end_pos - length_pos - 8  # Subtract 8 for the length field itself
    buf.seek(length_pos)
    buf.write(struct.pack("<Q", value_length))
    return buf.getvalue()


def encode_trace_event(identifier: str, **kwargs: Any) -> bytes:
    """Encode a trace event message.

    Args:
        identifier: The trace identifier.
        **kwargs: Data to serialize using to_binary().

    Returns:
        Encoded message bytes.

    """
    buf = io.BytesIO()
    length_pos = _write_header(buf, MessageType.TRACE_EVENT)

    identifier_bytes = identifier.encode("utf-8")
    buf.write(struct.pack("<Q", len(identifier_bytes)))
    buf.write(identifier_bytes)
    to_binary_into(buf, **kwargs)

    return _finish_message(buf, length_pos)


def encode_list_tracepoints() -> bytes:
    """Encode a list tracepoints request message.

    Returns:
        Encoded message bytes.

    """
    buf = io.BytesIO()
    buf.write(struct.pack("<IIQ", MessageType.LIST_TRACEPOINTS, 0, 0))
    return buf.getvalue()


def encode_list_tracepoints_response(identifiers: List[str]) -> bytes:
    """Encode a list tracepoints response message.

    Args:
        identifiers: List of tracepoint identifiers.

    Returns:
        Encoded message bytes.

    """
    buf = io.BytesIO()
    length_pos = _write_header(buf, MessageType.LIST_TRACEPOINTS_RESPONSE)

    buf.write(struct.pack("<Q", len(identifiers)))
    for ident in identifiers:
        ident_bytes = ident.encode("utf-8")
        buf.write(struct.pack("<Q", len(ident_bytes)))
        buf.write(ident_bytes)

    return _finish_message(buf, length_pos)


def encode_enable_pattern(pattern: str) -> bytes:
    """Encode an enable pattern message.

    Args:
        pattern: The pattern to enable.

    Returns:
        Encoded message bytes.

    """
    buf = io.BytesIO()
    length_pos = _write_header(buf, MessageType.ENABLE_PATTERN)
    buf.write(pattern.encode("utf-8"))
    return _finish_message(buf, length_pos)


def encode_enable_pattern_response(success: bool) -> bytes:
    """Encode an enable pattern response message.

    Args:
        success: Whether the operation succeeded.

    Returns:
        Encoded message bytes.

    """
    buf = io.BytesIO()
    length_pos = _write_header(buf, MessageType.ENABLE_PATTERN_RESPONSE)
    buf.write(struct.pack("B", 0 if success else 1))
    return _finish_message(buf, length_pos)


def encode_disable_pattern(pattern: str) -> bytes:
    """Encode a disable pattern message.

    Args:
        pattern: The pattern to disable.

    Returns:
        Encoded message bytes.

    """
    buf = io.BytesIO()
    length_pos = _write_header(buf, MessageType.DISABLE_PATTERN)
    buf.write(pattern.encode("utf-8"))
    return _finish_message(buf, length_pos)


def encode_disable_pattern_response(success: bool) -> bytes:
    """Encode a disable pattern response message.

    Args:
        success: Whether the operation succeeded.

    Returns:
        Encoded message bytes.

    """
    buf = io.BytesIO()
    length_pos = _write_header(buf, MessageType.DISABLE_PATTERN_RESPONSE)
    buf.write(struct.pack("B", 0 if success else 1))
    return _finish_message(buf, length_pos)


def decode_header(data: Union[bytes, memoryview]) -> Tuple[MessageType, int]:
    """Decode a message header.

    Args:
        data: Header bytes (must be HEADER_SIZE bytes).

    Returns:
        Tuple of (message_type, value_length).

    Raises:
        ValueError: If the message type is unknown.

    """
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Header too short: {len(data)} bytes, expected {HEADER_SIZE}")

    view = memoryview(data)
    msg_type, format, value_length = struct.unpack("<IIQ", view[:HEADER_SIZE])
    if format != 0:
        raise ValueError(f"Unknown format type: 0x{format:08x}. Must be 0.")

    try:
        msg_type = MessageType(msg_type)
    except ValueError:
        raise ValueError(f"Unknown message type: 0x{msg_type:08x}")  # noqa: B904

    return msg_type, value_length


def decode_trace_event(value: Union[bytes, memoryview]) -> Tuple[str, memoryview]:
    """Decode a trace event message value.

    Args:
        value: The message value bytes.

    Returns:
        Tuple of (identifier, data).

    """
    view = memoryview(value)
    identifier_len = struct.unpack("<Q", view[0:8])[0]
    identifier = view[8 : 8 + identifier_len].tobytes().decode("utf-8")
    data = view[8 + identifier_len :]
    return identifier, data


def decode_list_tracepoints_response(value: Union[bytes, memoryview]) -> List[str]:
    """Decode a list tracepoints response message value.

    Args:
        value: The message value bytes.

    Returns:
        List of tracepoint identifiers.

    """
    view = memoryview(value)
    count = struct.unpack("<Q", view[0:8])[0]
    identifiers: List[str] = []
    offset = 8
    for _ in range(count):
        ident_len = struct.unpack("<Q", view[offset : offset + 8])[0]
        offset += 8
        identifiers.append(view[offset : offset + ident_len].tobytes().decode("utf-8"))
        offset += ident_len
    return identifiers


def decode_enable_pattern(value: Union[bytes, memoryview]) -> str:
    """Decode an enable pattern message value.

    Args:
        value: The message value bytes.

    Returns:
        The pattern string.

    """
    if isinstance(value, memoryview):
        return value.tobytes().decode("utf-8")
    return value.decode("utf-8")


def decode_enable_pattern_response(value: Union[bytes, memoryview]) -> bool:
    """Decode an enable pattern response message value.

    Args:
        value: The message value bytes.

    Returns:
        True if successful, False otherwise.

    """
    return value[0] == 0


def decode_disable_pattern(value: Union[bytes, memoryview]) -> str:
    """Decode a disable pattern message value.

    Args:
        value: The message value bytes.

    Returns:
        The pattern string.

    """
    if isinstance(value, memoryview):
        return value.tobytes().decode("utf-8")
    return value.decode("utf-8")


def decode_disable_pattern_response(value: Union[bytes, memoryview]) -> bool:
    """Decode a disable pattern response message value.

    Args:
        value: The message value bytes.

    Returns:
        True if successful, False otherwise.

    """
    return value[0] == 0
