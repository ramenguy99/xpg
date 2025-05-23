from time import perf_counter_ns
from contextlib import contextmanager
import io

@contextmanager
def profile(name: str):
    begin = perf_counter_ns()
    yield
    end = perf_counter_ns()
    delta = end - begin
    print(f"{name}: {delta * 1e-6:.3f}ms")

def read_exact_into(file: io.FileIO, view: memoryview):
    bread = 0
    while bread < len(view):
        n = file.readinto(view[bread:])
        if n == 0:
            raise EOFError()
        else:
            bread += n

def read_exact(file: io.FileIO, size: int):
    out = bytearray(size)
    view = memoryview(out)
    read_exact_into(file, view)
    return out

def read_exact_at_offset_into(file: io.FileIO, offset: int, view: memoryview):
    file.seek(offset, io.SEEK_SET)
    return read_exact_into(file, view)

def read_exact_at_offset(file: io.FileIO, offset: int, size: int):
    file.seek(offset, io.SEEK_SET)
    return read_exact(file, size)
