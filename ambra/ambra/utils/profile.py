# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from contextlib import contextmanager
from time import perf_counter_ns


@contextmanager
def profile(name: str):  # type: ignore
    begin = perf_counter_ns()
    yield
    end = perf_counter_ns()
    delta = end - begin
    print(f"{name}: {delta * 1e-6:.3f}ms")  # noqa: T201
