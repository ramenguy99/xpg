from time import perf_counter
from contextlib import contextmanager

@contextmanager
def profile(name: str):
    begin = perf_counter()
    yield
    end = perf_counter()
    delta = end - begin
    print(f"{name}: {delta * 1000:.3f}ms")