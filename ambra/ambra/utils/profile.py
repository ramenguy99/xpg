from time import perf_counter_ns
from contextlib import contextmanager


@contextmanager
def profile(name: str):
    print(f"{name}: ", end="")
    begin = perf_counter_ns()
    yield
    end = perf_counter_ns()
    delta = end - begin
    print(f"{delta * 1e-6:.3f}ms")
