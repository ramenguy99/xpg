"""Serialization/Deserialization library with SQLite and binary format support.

API
---
    to_binary(**kwargs) -> bytes
    from_binary(data, keys=None) -> dict
    serialize_kwargs(**kwargs) -> dict
    create_table(conn, table_name, columns) -> None
    insert_serialized(conn, table_name, serialized) -> None
    to_sqlite(conn, table_name='data', **kwargs) -> None
    from_sqlite(conn, table_name='data', keys=None) -> dict
    from_sqlite_all(conn, table_name='data', keys=None) -> dict[str, list]

Supported types: None, bool, int, float, str, bytes, list, tuple, dict, numpy.ndarray.
Bools are converted to int (True->1, False->0) and lose type info on round-trip.
Unsupported types raise TypeError.

Binary Format
-------------
All integers are little-endian. Variable-size data is padded to 8-byte alignment.

    [8B num_entries]
    For each entry:
        [8B key_length][UTF-8 key][padding to 8B][value]

Value encoding by type code (8 bytes, little-endian uint64):
    0 NONE:  (no data)
    1 INT:   [8B signed int]
    2 FLOAT: [8B double]
    3 STR:   [8B length][UTF-8 data][padding to 8B]
    4 LIST:  [8B count][recursive values...]
    5 TUPLE: [8B count][recursive values...]
    6 DICT:  [8B count][recursive key-value pairs...]
    7 BYTES: [8B length][raw bytes][padding to 8B]
    8 NUMPY: [8B length][np.save() output][padding to 8B]

SQLite Format
-------------
Each kwarg becomes a column. Native type mapping:
    int/bool -> INTEGER, float -> REAL, str -> TEXT, None -> NULL

Complex types stored as BLOB:
    - list/tuple/dict/bytes: serialized with type code prefix (same as binary)
    - numpy.ndarray: raw np.save() output (detected by 0x93 magic header)

Use create_table() to create a table with columns matching the kwargs.
Call conn.commit() after create_table()/to_sqlite() if persistence is needed.
"""

import io
import sqlite3
import struct
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class TypeCode(IntEnum):
    """Type codes for binary serialization."""

    NONE = 0
    INT = 1
    FLOAT = 2
    STR = 3
    LIST = 4
    TUPLE = 5
    DICT = 6
    BYTES = 7
    NUMPY = 8


# NumPy magic bytes (first byte of np.save output)
_NUMPY_MAGIC = 0x93

# Padding bytes for 8-byte alignment
_PADDING = b"\x00" * 8


def _write_padding(buf: io.BytesIO, length: int) -> None:
    """Write padding bytes to align to 8-byte boundary."""
    padding = (-length) & 7
    if padding:
        buf.write(_PADDING[:padding])


def _serialize_value(buf: io.BytesIO, value: object) -> None:
    """Serialize a single value to a BytesIO buffer."""
    if value is None:
        buf.write(struct.pack("<Q", TypeCode.NONE))
        return

    if isinstance(value, bool):
        # Convert bool to int (True -> 1, False -> 0)
        buf.write(struct.pack("<Q", TypeCode.INT))
        buf.write(struct.pack("<q", int(value)))
        return

    if isinstance(value, int):
        buf.write(struct.pack("<Q", TypeCode.INT))
        buf.write(struct.pack("<q", value))
        return

    if isinstance(value, float):
        buf.write(struct.pack("<Q", TypeCode.FLOAT))
        buf.write(struct.pack("<d", value))
        return

    if isinstance(value, str):
        encoded = value.encode("utf-8")
        buf.write(struct.pack("<Q", TypeCode.STR))
        buf.write(struct.pack("<Q", len(encoded)))
        buf.write(encoded)
        _write_padding(buf, len(encoded))
        return

    if isinstance(value, bytes):
        buf.write(struct.pack("<Q", TypeCode.BYTES))
        buf.write(struct.pack("<Q", len(value)))
        buf.write(value)
        _write_padding(buf, len(value))
        return

    if isinstance(value, list):
        buf.write(struct.pack("<Q", TypeCode.LIST))
        buf.write(struct.pack("<Q", len(value)))
        for item in value:
            _serialize_value(buf, item)
        return

    if isinstance(value, tuple):
        buf.write(struct.pack("<Q", TypeCode.TUPLE))
        buf.write(struct.pack("<Q", len(value)))
        for item in value:
            _serialize_value(buf, item)
        return

    if isinstance(value, dict):
        buf.write(struct.pack("<Q", TypeCode.DICT))
        buf.write(struct.pack("<Q", len(value)))
        for k, v in value.items():
            _serialize_value(buf, k)
            _serialize_value(buf, v)
        return

    if isinstance(value, np.ndarray):
        buf.write(struct.pack("<Q", TypeCode.NUMPY))
        # Write placeholder for length, save position
        length_pos = buf.tell()
        buf.write(struct.pack("<Q", 0))
        # Write numpy data directly to buffer
        start_pos = buf.tell()
        np.save(buf, value)
        end_pos = buf.tell()
        npy_length = end_pos - start_pos
        # Go back and write actual length
        buf.seek(length_pos)
        buf.write(struct.pack("<Q", npy_length))
        buf.seek(end_pos)
        _write_padding(buf, npy_length)
        return

    raise TypeError(f"Unsupported type: {type(value).__name__}")


def _skip_value(view: memoryview, offset: int) -> int:
    """Skip over a serialized value in a memoryview without deserializing it."""
    type_code = struct.unpack("<Q", view[offset : offset + 8])[0]
    offset += 8

    if type_code == TypeCode.NONE:
        return offset

    if type_code == TypeCode.INT or type_code == TypeCode.FLOAT:
        return offset + 8

    if type_code == TypeCode.STR or type_code == TypeCode.BYTES or type_code == TypeCode.NUMPY:
        length: int = struct.unpack("<Q", view[offset : offset + 8])[0]
        offset += 8
        padding = (-length) & 7
        return offset + length + padding

    if type_code == TypeCode.LIST or type_code == TypeCode.TUPLE:
        count = struct.unpack("<Q", view[offset : offset + 8])[0]
        offset += 8
        for _ in range(count):
            offset = _skip_value(view, offset)
        return offset

    if type_code == TypeCode.DICT:
        count = struct.unpack("<Q", view[offset : offset + 8])[0]
        offset += 8
        for _ in range(count):
            offset = _skip_value(view, offset)  # Skip key
            offset = _skip_value(view, offset)  # Skip value
        return offset

    raise ValueError(f"Unknown type code: {type_code}")


def _deserialize_value(view: memoryview, offset: int) -> Tuple[object, int]:
    """Deserialize a single value from a memoryview at the given offset."""
    type_code = struct.unpack("<Q", view[offset : offset + 8])[0]
    offset += 8

    if type_code == TypeCode.NONE:
        return None, offset

    if type_code == TypeCode.INT:
        value = struct.unpack("<q", view[offset : offset + 8])[0]
        return value, offset + 8

    if type_code == TypeCode.FLOAT:
        value = struct.unpack("<d", view[offset : offset + 8])[0]
        return value, offset + 8

    if type_code == TypeCode.STR:
        length = struct.unpack("<Q", view[offset : offset + 8])[0]
        offset += 8
        str_value = view[offset : offset + length].tobytes().decode("utf-8")
        offset += length
        offset += (-length) & 7  # Skip padding
        return str_value, offset

    if type_code == TypeCode.BYTES:
        length = struct.unpack("<Q", view[offset : offset + 8])[0]
        offset += 8
        bytes_value = view[offset : offset + length].tobytes()
        offset += length
        offset += (-length) & 7  # Skip padding
        return bytes_value, offset

    if type_code == TypeCode.LIST:
        count = struct.unpack("<Q", view[offset : offset + 8])[0]
        offset += 8
        items = []
        for _ in range(count):
            item, offset = _deserialize_value(view, offset)
            items.append(item)
        return items, offset

    if type_code == TypeCode.TUPLE:
        count = struct.unpack("<Q", view[offset : offset + 8])[0]
        offset += 8
        items = []
        for _ in range(count):
            item, offset = _deserialize_value(view, offset)
            items.append(item)
        return tuple(items), offset

    if type_code == TypeCode.DICT:
        count = struct.unpack("<Q", view[offset : offset + 8])[0]
        offset += 8
        result: Dict[object, object] = {}
        for _ in range(count):
            dict_key, offset = _deserialize_value(view, offset)
            dict_value, offset = _deserialize_value(view, offset)
            result[dict_key] = dict_value
        return result, offset

    if type_code == TypeCode.NUMPY:
        length = struct.unpack("<Q", view[offset : offset + 8])[0]
        offset += 8
        npy_data = view[offset : offset + length]
        offset += length
        offset += (-length) & 7  # Skip padding
        return np.load(io.BytesIO(npy_data)), offset

    raise ValueError(f"Unknown type code: {type_code}")


def to_binary_into(buf: io.BytesIO, **kwargs: object) -> None:
    """Serialize kwargs to binary format into an existing buffer.

    Args:
        buf: BytesIO buffer to write into
        **kwargs: Key-value pairs to serialize

    Raises:
        TypeError: If a value has an unsupported type

    """
    buf.write(struct.pack("<Q", len(kwargs)))

    for key, value in kwargs.items():
        key_bytes = key.encode("utf-8")
        buf.write(struct.pack("<Q", len(key_bytes)))
        buf.write(key_bytes)
        _write_padding(buf, len(key_bytes))
        _serialize_value(buf, value)


def to_binary(**kwargs: object) -> bytes:
    """Serialize kwargs to binary format.

    Args:
        **kwargs: Key-value pairs to serialize

    Returns:
        Binary representation of the data

    Raises:
        TypeError: If a value has an unsupported type

    """
    buf = io.BytesIO()
    to_binary_into(buf, **kwargs)
    return buf.getvalue()


def from_binary(data: Union[bytes, memoryview], keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Deserialize from binary format.

    Args:
        data: Binary data to deserialize
        keys: Optional list of keys to deserialize. If None, all keys are returned.

    Returns:
        Dictionary of deserialized key-value pairs

    """
    view = memoryview(data)
    offset = 0

    num_entries = struct.unpack("<Q", view[offset : offset + 8])[0]
    offset += 8

    keys_set = set(keys) if keys is not None else None
    result = {}

    for _ in range(num_entries):
        key_length = struct.unpack("<Q", view[offset : offset + 8])[0]
        offset += 8
        key = view[offset : offset + key_length].tobytes().decode("utf-8")
        offset += key_length
        offset += (-key_length) & 7  # Skip padding

        if keys_set is None or key in keys_set:
            value, offset = _deserialize_value(view, offset)
            result[key] = value
        else:
            offset = _skip_value(view, offset)

    return result


def _get_sqlite_type(value: object) -> str:
    """Get SQLite type for a Python value."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "INTEGER"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    if isinstance(value, str):
        return "TEXT"
    if isinstance(value, (bytes, list, tuple, dict, np.ndarray)):
        return "BLOB"
    raise TypeError(f"Unsupported type: {type(value).__name__}")


def _to_sqlite_value(value: object) -> object:
    """Convert a Python value to SQLite-compatible value."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, (bytes, list, tuple, dict)):
        # Serialize with type code prefix
        buf = io.BytesIO()
        _serialize_value(buf, value)
        return buf.getvalue()
    if isinstance(value, np.ndarray):
        # Store raw np.save output (no type code needed, has magic header)
        buf = io.BytesIO()
        np.save(buf, value)
        return buf.getvalue()
    raise TypeError(f"Unsupported type: {type(value).__name__}")


def from_sqlite_value(value: object) -> object:
    """Convert a SQLite value back to Python type."""
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, bytes):
        # Check for numpy magic header
        if len(value) > 0 and value[0] == _NUMPY_MAGIC:
            return np.load(io.BytesIO(value))
        # Otherwise, it's a serialized list/tuple/dict/bytes
        result, _ = _deserialize_value(memoryview(value), 0)
        return result
    return value


def serialize_kwargs(**kwargs: object) -> Dict[str, Any]:
    """Serialize kwargs values to SQLite-compatible format.

    This is a lower-level function that only serializes the values without
    performing any database operations. Use this when you need to serialize
    data before the actual database write.

    Args:
        **kwargs: Key-value pairs to serialize

    Returns:
        Dictionary with same keys but values converted to SQLite-compatible types

    Raises:
        TypeError: If a value has an unsupported type

    """
    return {k: _to_sqlite_value(v) for k, v in kwargs.items()}


def insert_serialized(
    conn: sqlite3.Connection,
    table_name: str,
    serialized: Dict[str, Any],
) -> None:
    """Insert pre-serialized data into a SQLite table.

    This is a lower-level function that inserts data that has already been
    serialized with serialize_kwargs().

    Args:
        conn: Open SQLite connection
        table_name: Name of the table to insert into
        serialized: Dictionary of pre-serialized key-value pairs

    Note:
        The caller is responsible for calling conn.commit() if persistence is desired.

    """
    if not serialized:
        return

    placeholders = ", ".join(["?" for _ in serialized])
    column_names = ", ".join([f'"{k}"' for k in serialized])
    insert_sql = f'INSERT INTO "{table_name}" ({column_names}) VALUES ({placeholders})'  # noqa: S608

    conn.execute(insert_sql, list(serialized.values()))


def create_table(
    conn: sqlite3.Connection,
    table_name: str,
    columns: Dict[str, Any],
) -> None:
    """Create a table with columns matching the given dictionary.

    Args:
        conn: Open SQLite connection
        table_name: Name of the table to create
        columns: Dictionary where keys are column names and values determine types

    Raises:
        TypeError: If a value has an unsupported type

    Note:
        The caller is responsible for calling conn.commit() if persistence is desired.

    """
    if not columns:
        return

    col_defs = []
    for key, value in columns.items():
        sql_type = _get_sqlite_type(value)
        col_defs.append(f'"{key}" {sql_type}')

    create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(col_defs)})'
    conn.execute(create_sql)


def to_sqlite(conn: sqlite3.Connection, table_name: str = "data", **kwargs: object) -> None:
    """Serialize kwargs and insert as a row in a SQLite table.

    Args:
        conn: Open SQLite connection
        table_name: Name of the table to insert into
        **kwargs: Key-value pairs to serialize

    Raises:
        TypeError: If a value has an unsupported type

    Note:
        The caller is responsible for calling conn.commit() if persistence is desired.

    """
    if not kwargs:
        return

    insert_serialized(conn, table_name, serialize_kwargs(**kwargs))


def _build_select(
    table_name: str,
    keys: Optional[List[str]],
    where: str,
) -> Tuple[str, Optional[List[str]]]:
    """Build a SELECT query and return (sql, known_keys).

    known_keys is the list of column names when keys is provided,
    or None when using SELECT * (requiring cursor.description).
    """
    if keys is not None:
        column_names = ", ".join([f'"{k}"' for k in keys])
        known_keys = keys
    else:
        column_names = "*"
        known_keys = None

    select_sql = f'SELECT {column_names} FROM "{table_name}" WHERE {where}'  # noqa: S608
    return select_sql, known_keys


def from_sqlite(
    conn: sqlite3.Connection,
    table_name: str = "data",
    keys: Optional[List[str]] = None,
    where: str = "TRUE",
) -> Dict[str, Any]:
    """Deserialize a single row from SQLite.

    Args:
        conn: Open SQLite connection
        table_name: Name of the table to read from
        keys: Optional list of keys to deserialize. If None, all columns are returned.
        where: SQL WHERE clause (default: "TRUE" for all rows).

    Returns:
        Dictionary of deserialized key-value pairs, or empty dict if no rows.

    """
    select_sql, known_keys = _build_select(table_name, keys, where)
    cursor = conn.execute(select_sql)

    row = cursor.fetchone()
    if row is None:
        return {}

    col_names = known_keys if known_keys is not None else [desc[0] for desc in cursor.description]

    return {name: from_sqlite_value(value) for name, value in zip(col_names, row)}


def from_sqlite_all(
    conn: sqlite3.Connection,
    table_name: str = "data",
    keys: Optional[List[str]] = None,
    where: str = "TRUE",
) -> Dict[str, List[Any]]:
    """Deserialize all rows from SQLite into column-oriented lists.

    Args:
        conn: Open SQLite connection
        table_name: Name of the table to read from
        keys: Optional list of keys to deserialize. If None, all columns are returned.
        where: SQL WHERE clause (default: "TRUE" for all rows).

    Returns:
        Dictionary mapping column names to lists of deserialized values.
        Returns empty dict if no rows.

    """
    select_sql, known_keys = _build_select(table_name, keys, where)
    cursor = conn.execute(select_sql)

    rows = cursor.fetchall()
    if not rows:
        return {}

    col_names = known_keys if known_keys is not None else [desc[0] for desc in cursor.description]

    result: Dict[str, List[Any]] = {name: [] for name in col_names}
    for row in rows:
        for name, value in zip(col_names, row):
            result[name].append(from_sqlite_value(value))

    return result
