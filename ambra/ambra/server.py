from socketserver import TCPServer, StreamRequestHandler
from dataclasses import dataclass
import struct
from typing import Callable, Union, Optional, TypeAlias
from threading import Thread
from enum import Enum

from .utils.io import read_exact
import json
import pickle

@dataclass
class Client:
    address: str
    port: int
    name: str

class MessageId(Enum):
    NONE = 0 # Reserved for end of connection

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
    data: bytearray

class Server:
    def __init__(self, address: str, port: int, on_raw_message_async: Callable[[Client, RawMessage], None]):
        class Handler(StreamRequestHandler):
            def handle(self):
                try:
                    print(f"Server: client {self.client_address[0]}:{self.client_address[1]} opened connection")
                    magic = read_exact(self.rfile, 4)
                    if magic != b"AMBR":
                        return

                    # Header
                    client_name_length = struct.unpack("<I", read_exact(self.rfile, 4))[0]
                    if client_name_length > 0:
                        pass
                    client_name = read_exact(self.rfile, client_name_length).decode(encoding="utf-8", errors="ignore")
                    client = Client(self.client_address[0], self.client_address[1], client_name)

                    print(f"Server: client {self.client_address[0]}:{self.client_address[1]} registered as \"{client_name}\"")

                    # Messages
                    while True:
                        id = struct.unpack("<I", read_exact(self.rfile, 4))[0]
                        if id == 0:
                            print(f"Server: client {self.client_address[0]}:{self.client_address[1]} closed connection")
                            break
                        format, length = struct.unpack("<IQ", read_exact(self.rfile, 12))
                        data = read_exact(self.rfile, length)

                        on_raw_message_async(client, RawMessage(id, format, data))
                except EOFError:
                    print(f"Server: client {self.client_address[0]}:{self.client_address[1]} unexpected EOF before closing connection")

        self.server = TCPServer((address, port), Handler)
        self.thread = Thread(None, self._server_entry, "Server", daemon=True)
        self.thread.start()
        # atexit.register(self.shutdown)
    
    def shutdown(self):
        self.server.shutdown()
        self.thread.join()
    
    def _server_entry(self):
        with self.server:
            self.server.serve_forever()


@dataclass
class InputMessage:
    num: int

    @classmethod
    def from_raw(cls, data: bytearray):
        return InputMessage(0)

    @classmethod
    def from_json(cls, obj: object):
        return InputMessage(0)

@dataclass
class ObjectMessage:
    name: str

    @classmethod
    def from_raw(cls, data: bytearray):
        return ObjectMessage()

    @classmethod
    def from_json(cls, obj: object):
        return ObjectMessage()

Message: TypeAlias = Union[
    InputMessage,
    ObjectMessage,
]

_binary_dispatch = {
    MessageId.INPUT.value: InputMessage.from_raw,
    MessageId.OBJECT.value: ObjectMessage.from_raw,
}

_json_dispatch = {
    MessageId.INPUT.value: InputMessage.from_json,
    MessageId.OBJECT.value: ObjectMessage.from_json,
}

def _parse_builtin_messages_binary(raw: RawMessage) -> Optional[Message]:
    fn = _binary_dispatch.get(raw.id)
    if not fn:
        return None
    return fn(raw)

def _parse_builtin_messages_json(raw: RawMessage) -> Optional[Message]:
    fn = _json_dispatch.get(raw.id)
    if not fn:
        return None
    obj = json.loads(raw.data)
    return fn(obj)

def _parse_builtin_messages_pickle(raw: RawMessage) -> Optional[Message]:
    return pickle.loads(raw.data)

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
