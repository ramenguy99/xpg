import socket
import struct

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("localhost", 9168))

name = "CLIENT".encode(encoding="utf-8")
name_length = struct.pack("<I", len(name))
s.send(b"AMBR" + name_length + name)

for i in range(10):
    data = f"Data is {i}".encode("utf-8") + bytearray(1024 * 1024)
    msg_header = struct.pack("<IIQ", 1 << 20, 0, len(data))
    s.send(msg_header + data)
s.send(struct.pack("<I", 0))
