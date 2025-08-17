import asyncio
import atexit
import json
import pickle
import struct
from dataclasses import dataclass
from enum import Enum
from threading import Thread
from typing import Callable, Dict, Optional, Tuple, Union

from .config import ServerConfig


@dataclass
class Client:
    address: str
    port: int
    name: str


class MessageId(Enum):
    NONE = 0  # Reserved for end of connection

    # Default message ids
    INPUT = 1
    OBJECT = 2

    # Starting id available for users
    USER = 1 << 20


class Format(Enum):
    # Default formats
    BINARY = 0
    PICKLE = 2
    JSON = 1

    # Starting id available for users
    USER = 1 << 20


@dataclass
class RawMessage:
    id: int
    format: int
    data: bytes


class Server:
    def __init__(
        self,
        on_raw_message_async: Callable[[Client, RawMessage], None],
        config: ServerConfig,
    ):
        self.connections: Dict[Tuple[str, int], asyncio.StreamWriter] = {}

        def entry() -> None:
            async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                try:
                    client_address = writer.get_extra_info("peername")
                    self.connections[client_address] = writer

                    print(f"Server: client {client_address[0]}:{client_address[1]} opened connection")
                    magic = await reader.readexactly(4)
                    if magic != b"AMBR":
                        return

                    # Header
                    client_name_length = struct.unpack("<I", await reader.readexactly(4))[0]
                    if client_name_length > 0:
                        pass
                    client_name = (await reader.readexactly(client_name_length)).decode(
                        encoding="utf-8", errors="ignore"
                    )
                    client = Client(client_address[0], client_address[1], client_name)

                    print(f'Server: client {client_address[0]}:{client_address[1]} registered as "{client_name}"')

                    # Messages
                    while True:
                        id = struct.unpack("<I", await reader.readexactly(4))[0]
                        if id == 0:
                            print(f"Server: client {client_address[0]}:{client_address[1]} closed connection")
                            break
                        format, length = struct.unpack("<IQ", await reader.readexactly(12))
                        data = await reader.readexactly(length)

                        on_raw_message_async(client, RawMessage(id, format, data))
                except asyncio.exceptions.IncompleteReadError:
                    print(
                        f"Server: client {client_address[0]}:{client_address[1]} unexpected EOF before closing connection"
                    )
                except Exception as e:
                    print(f"Server: exception while handling client connection {e}")

                del self.connections[client_address]
                writer.close()
                await writer.wait_closed()

            async def main() -> None:
                server = await asyncio.start_server(handle, config.address, config.port)

                async with server:
                    await server.start_serving()
                    await self.stop

                    # Stop accepting new connections
                    server.close()

                    # Close all existing open connections
                    connections = self.connections.values()
                    for c in connections:
                        c.close()

            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(main())

        self.enabled = config.enabled

        if self.enabled:
            self.loop = asyncio.new_event_loop()
            self.stop = self.loop.create_future()

            self.thread = Thread(None, entry, "Server", daemon=True)
            self.thread.start()
            atexit.register(self.shutdown)

    def shutdown(self) -> None:
        if self.enabled and not self.stop.done():
            self.loop.call_soon_threadsafe(self.stop.set_result, None)
            self.thread.join()


@dataclass
class InputMessage:
    num: int

    @classmethod
    def from_raw(cls, data: bytes) -> "InputMessage":
        return InputMessage(0)

    @classmethod
    def from_json(cls, obj: object) -> "InputMessage":
        return InputMessage(0)


@dataclass
class ObjectMessage:
    @classmethod
    def from_raw(cls, data: bytes) -> "ObjectMessage":
        return ObjectMessage()

    @classmethod
    def from_json(cls, obj: object) -> "ObjectMessage":
        return ObjectMessage()


Message = Union[
    InputMessage,
    ObjectMessage,
]

_binary_dispatch: Dict[int, Callable[[bytes], Message]] = {
    MessageId.INPUT.value: InputMessage.from_raw,
    MessageId.OBJECT.value: ObjectMessage.from_raw,
}

_json_dispatch: Dict[int, Callable[[object], Message]] = {
    MessageId.INPUT.value: InputMessage.from_json,
    MessageId.OBJECT.value: ObjectMessage.from_json,
}


def _parse_builtin_messages_binary(raw: RawMessage) -> Optional[Message]:
    fn = _binary_dispatch.get(raw.id)
    if not fn:
        return None
    return fn(raw.data)


def _parse_builtin_messages_json(raw: RawMessage) -> Optional[Message]:
    fn = _json_dispatch.get(raw.id)
    if not fn:
        return None
    obj = json.loads(raw.data)
    return fn(obj)


def _parse_builtin_messages_pickle(raw: RawMessage) -> Optional[Message]:
    return pickle.loads(raw.data)  # type: ignore


_dispatch = {
    Format.BINARY.value: _parse_builtin_messages_binary,
    Format.JSON.value: _parse_builtin_messages_json,
    Format.PICKLE.value: _parse_builtin_messages_pickle,
}


def parse_builtin_messages(raw: RawMessage) -> Optional[Message]:
    format_fn = _dispatch.get(raw.format)
    if not format_fn:
        return None

    return format_fn(raw)
